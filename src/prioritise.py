"""Rank TYK2 compounds as candidates for an ankylosing spondylitis programme.

Hard filters (a compound must pass all of them):
  - potency       pChEMBL(TYK2) >= 8 (<= 10 nM)
  - selectivity   >= 100-fold (2 log units) over every JAK it was tested against,
                  and tested against at least one of JAK1/2/3 (oral JAK1/2
                  inhibition is linked to the class safety warnings)
  - drug-likeness <= 1 Lipinski violation, TPSA <= 140, no PAINS alerts

Survivors are scored on a 0-1 scale from potency, selectivity, ligand efficiency
and QED, then capped at two compounds per scaffold so the list covers distinct
chemical series. Clinical TYK2 compounds are located in the table for comparison.

Writes results/candidate_ranking.csv (all passing compounds) and
results/top_candidates.csv (diverse shortlist).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import requests
from rdkit import Chem, RDLogger
from rdkit.Chem import QED
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem.Scaffolds import MurckoScaffold

from process_activities import standardise_smiles

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
OT_DIR = ROOT / "data" / "raw" / "opentargets"

MIN_POTENCY = 8.0
MIN_SELECTIVITY = 2.0
MAX_TPSA = 140
MAX_PER_SCAFFOLD = 2
N_SHORTLIST = 25

# Clinical TYK2-directed compounds to benchmark against (ChEMBL IDs)
REFERENCE_DRUGS = {
    "deucravacitinib": "CHEMBL4435170",
    "zasocitinib": "CHEMBL5314423",
    "ropsacitinib": "CHEMBL4459585",
    "brepocitinib": "CHEMBL4297477",
}

_pains = FilterCatalogParams()
_pains.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
PAINS = FilterCatalog(_pains)


def fetch_reference_smiles() -> pd.DataFrame:
    """Standardised structures of reference drugs (from ChEMBL and the Open Targets drug list)."""
    ids = dict(REFERENCE_DRUGS)
    drugs_file = OT_DIR / "tyk2_drugs.csv"
    if drugs_file.exists():
        for r in pd.read_csv(drugs_file).itertuples():
            ids.setdefault(r.drug.lower(), r.chembl_id)
    rows = []
    for name, chembl_id in ids.items():
        try:
            r = requests.get(f"https://www.ebi.ac.uk/chembl/api/data/molecule/{chembl_id}",
                             params={"format": "json"}, timeout=60)
            r.raise_for_status()
            smi = ((r.json().get("molecule_structures") or {}).get("canonical_smiles"))
        except requests.RequestException as e:
            print(f"  could not fetch {name} ({chembl_id}): {e}")
            continue
        if smi:
            rows.append({"reference_drug": name, "reference_chembl_id": chembl_id,
                         "smiles_std": standardise_smiles(smi)})
    return pd.DataFrame(rows).drop_duplicates("smiles_std")


def scale(s: pd.Series, lo: float, hi: float) -> pd.Series:
    return ((s - lo) / (hi - lo)).clip(0, 1)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    compounds = pd.read_csv(PROCESSED_DIR / "tyk2_compounds.csv")
    sel = pd.read_csv(PROCESSED_DIR / "tyk2_selectivity.csv")
    df = compounds.merge(
        sel[["smiles_std", "domain", "p_jak1", "p_jak2", "p_jak3",
             "sel_jak1", "sel_jak2", "sel_jak3", "n_jak_tested", "min_selectivity"]],
        on="smiles_std", how="left",
    )

    mols = df["smiles_std"].map(Chem.MolFromSmiles)
    df["qed"] = mols.map(QED.qed)
    df["pains"] = mols.map(PAINS.HasMatch)
    df["scaffold"] = df["smiles_std"].map(lambda s: MurckoScaffold.MurckoScaffoldSmiles(smiles=s))

    refs = fetch_reference_smiles()
    df = df.merge(refs, on="smiles_std", how="left")

    df["pass_potency"] = df["pchembl_median"] >= MIN_POTENCY
    df["pass_selectivity"] = (df["n_jak_tested"] > 0) & (df["min_selectivity"] >= MIN_SELECTIVITY)
    df["pass_druglike"] = (df["lipinski_violations"] <= 1) & (df["tpsa"] <= MAX_TPSA) & ~df["pains"]
    df["passes_all"] = df["pass_potency"] & df["pass_selectivity"] & df["pass_druglike"]

    # Selectivity measured against one JAK is weaker evidence than a full panel,
    # so completeness of the off-target profile is part of the score.
    df["score"] = (
        0.30 * scale(df["pchembl_median"], MIN_POTENCY, 10)
        + 0.25 * scale(df["min_selectivity"], MIN_SELECTIVITY, 4)
        + 0.20 * (df["n_jak_tested"] / 3)
        + 0.15 * scale(df["ligand_efficiency"], 0.25, 0.45)
        + 0.10 * df["qed"]
    ).round(3)

    cols = ["molecule_chembl_id", "reference_drug", "smiles_std", "score", "pchembl_median",
            "binned_only", "domain", "sel_jak1", "sel_jak2", "sel_jak3", "min_selectivity",
            "n_jak_tested", "ligand_efficiency", "qed", "mw", "logp", "tpsa",
            "lipinski_violations", "scaffold"]
    ranked = df[df["passes_all"]].sort_values("score", ascending=False)
    ranked[cols].to_csv(RESULTS_DIR / "candidate_ranking.csv", index=False)

    shortlist = (ranked.groupby("scaffold", sort=False).head(MAX_PER_SCAFFOLD)
                 .head(N_SHORTLIST))
    shortlist[cols].to_csv(RESULTS_DIR / "top_candidates.csv", index=False)

    print("Filter funnel:")
    print(f"  all compounds             {len(df)}")
    print(f"  potency <= 10 nM          {df['pass_potency'].sum()}")
    print(f"  + JAK selectivity >=100x  {(df['pass_potency'] & df['pass_selectivity']).sum()}")
    print(f"  + drug-like               {df['passes_all'].sum()}")
    print(f"  shortlist (diverse)       {len(shortlist)} from "
          f"{shortlist['scaffold'].nunique()} scaffolds")

    found = df[df["reference_drug"].notna()]
    print("\nReference drugs in the TYK2 dataset:")
    if found.empty:
        print("  none")
    for r in found.itertuples():
        rank = (ranked.index.get_loc(r.Index) + 1) if r.passes_all else None
        print(f"  {r.reference_drug:<22} pChEMBL {r.pchembl_median:.2f}  "
              f"min sel {r.min_selectivity if pd.notna(r.min_selectivity) else np.nan:.2f}  "
              f"passes={r.passes_all}  rank={rank}")


if __name__ == "__main__":
    main()
