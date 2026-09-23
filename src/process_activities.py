"""Clean, standardise and aggregate raw ChEMBL activity data.

Reads data/raw/<target>_activities_raw.csv (from get_chembl_data.py) and writes
data/processed/<target>_compounds.csv with one row per standardised compound.

    python src/process_activities.py            # TYK2 only
    python src/process_activities.py --all      # TYK2 + JAK1/2/3
"""

import argparse
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

from get_chembl_data import TARGETS

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

ACTIVE_THRESHOLD = 7.0  # pChEMBL >= 7 (<= 100 nM) counts as active

# A document reporting only a handful of distinct values for many compounds is
# reporting potency ranges (e.g. "1-10 nM" stored as 5.5 nM), typically a patent.
# Caught either by a hard cap on distinct values, or by most records piling onto
# a few values (which also catches patents that mix in a handful of exact numbers).
BINNED_MIN_RECORDS = 20
BINNED_MAX_DISTINCT = 5
BINNED_TOP_VALUES = 5
BINNED_TOP_SHARE = 0.9

_largest_fragment = rdMolStandardize.LargestFragmentChooser()
_uncharger = rdMolStandardize.Uncharger()


def standardise_smiles(smiles: str) -> str | None:
    """Strip salts/solvents, neutralise charges and return canonical SMILES."""
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return None
    mol = rdMolStandardize.Cleanup(mol)
    mol = _largest_fragment.choose(mol)
    mol = _uncharger.uncharge(mol)
    return Chem.MolToSmiles(mol)


def annotate_domain(description: str) -> str:
    """Guess which kinase domain an assay targets from its description.

    For TYK2, JH2 is the pseudokinase (allosteric, e.g. deucravacitinib) domain;
    JH1 is the catalytic kinase (ATP-site) domain.
    """
    d = str(description).lower()
    if "jh2" in d or "pseudokinase" in d:
        return "JH2"
    if "jh1" in d or "kinase domain" in d or "catalytic domain" in d:
        return "JH1"
    return "unspecified"


def flag_binned(df: pd.DataFrame) -> pd.Series:
    """True for records from documents that report binned (range) values."""
    grouped = df.groupby("document_chembl_id")["standard_value"]
    stats = pd.DataFrame({
        "n": grouped.size(),
        "n_distinct": grouped.nunique(),
        "top_share": grouped.apply(
            lambda s: s.value_counts().head(BINNED_TOP_VALUES).sum() / len(s)),
    })
    big_enough = stats["n"] >= BINNED_MIN_RECORDS
    concentrated = (stats["n_distinct"] <= BINNED_MAX_DISTINCT) | (stats["top_share"] >= BINNED_TOP_SHARE)
    return df["document_chembl_id"].isin(stats[big_enough & concentrated].index)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)
    df = df[df["target_organism"] == "Homo sapiens"]
    df = df[df["data_validity_comment"].isna()]
    df = df[df["potential_duplicate"].fillna(0).astype(int) == 0]
    df = df[df["standard_units"] == "nM"]
    df = df.dropna(subset=["canonical_smiles", "pchembl_value"])
    print(f"Filtering: {n0} -> {len(df)} activity records")

    df = df.copy()
    df["binned"] = flag_binned(df)
    print(f"Binned (range-reported) records: {df['binned'].sum()} ({df['binned'].mean():.0%})")

    df["smiles_std"] = df["canonical_smiles"].map(standardise_smiles)
    n_bad = df["smiles_std"].isna().sum()
    if n_bad:
        print(f"Dropping {n_bad} records with unparseable SMILES")
    df = df.dropna(subset=["smiles_std"])
    df["domain"] = df["assay_description"].map(annotate_domain)
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse repeated measurements to one row per standardised structure.

    pchembl_median uses every record; pchembl_measured uses only non-binned
    records (NaN when a compound only has binned values).
    """
    agg = (
        df.groupby("smiles_std")
        .agg(
            molecule_chembl_id=("molecule_chembl_id", "first"),
            pchembl_median=("pchembl_value", "median"),
            pchembl_std=("pchembl_value", "std"),
            n_measurements=("pchembl_value", "size"),
            standard_types=("standard_type", lambda s: ",".join(sorted(set(s)))),
            domains=("domain", lambda s: ",".join(sorted(set(s)))),
            binned_only=("binned", "all"),
        )
        .reset_index()
    )
    measured = (
        df[~df["binned"]].groupby("smiles_std")["pchembl_value"]
        .agg(pchembl_measured="median", n_measured="size")
    )
    agg = agg.merge(measured, on="smiles_std", how="left")
    agg["n_measured"] = agg["n_measured"].fillna(0).astype(int)
    agg["active"] = (agg["pchembl_median"] >= ACTIVE_THRESHOLD).astype(int)
    return agg


def add_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    mols = df["smiles_std"].map(Chem.MolFromSmiles)
    df = df.copy()
    df["mw"] = mols.map(Descriptors.MolWt)
    df["logp"] = mols.map(Crippen.MolLogP)
    df["hbd"] = mols.map(Lipinski.NumHDonors)
    df["hba"] = mols.map(Lipinski.NumHAcceptors)
    df["tpsa"] = mols.map(rdMolDescriptors.CalcTPSA)
    df["rot_bonds"] = mols.map(Lipinski.NumRotatableBonds)
    df["heavy_atoms"] = mols.map(lambda m: m.GetNumHeavyAtoms())
    df["lipinski_violations"] = (
        (df["mw"] > 500).astype(int)
        + (df["logp"] > 5).astype(int)
        + (df["hbd"] > 5).astype(int)
        + (df["hba"] > 10).astype(int)
    )
    # Ligand efficiency (kcal/mol per heavy atom), 1.37 ~ 2.303*RT at 298 K
    df["ligand_efficiency"] = 1.37 * df["pchembl_median"] / df["heavy_atoms"]
    return df


def process_target(name: str) -> pd.DataFrame:
    print(f"\n=== {name.upper()} ===")
    raw = pd.read_csv(RAW_DIR / f"{name}_activities_raw.csv")
    df = clean(raw)
    compounds = add_descriptors(aggregate(df))

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / f"{name}_compounds.csv"
    compounds.to_csv(out, index=False)

    print(f"Saved {len(compounds)} unique compounds to {out}")
    print(f"Active (pChEMBL >= {ACTIVE_THRESHOLD}): "
          f"{compounds['active'].sum()} ({compounds['active'].mean():.1%})")
    print(f"Compounds with measured (non-binned) values: {compounds['pchembl_measured'].notna().sum()}")
    print("Assay domain annotation (records):")
    print(df["domain"].value_counts().to_string())
    return compounds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--all", action="store_true", help="process every target in TARGETS")
    args = parser.parse_args()
    for name in (TARGETS if args.all else ["tyk2"]):
        process_target(name)


if __name__ == "__main__":
    main()
