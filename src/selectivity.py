"""TYK2 selectivity over JAK1, JAK2 and JAK3.

Joins the per-compound tables from process_activities.py --all on standardised
SMILES and computes selectivity = pChEMBL(TYK2) - pChEMBL(JAKx), in log units
(1 = 10-fold, 2 = 100-fold selective for TYK2).

Two caveats drive the design of this module:

1. Binned values cannot be differenced. A patent that reports both TYK2 and
   JAK2 as ">1 uM" yields a selectivity of exactly 0 that means nothing. So
   selectivity is computed from measured (non-binned) values only; a compound
   whose value on either side is binned gets NaN.

2. Assay mismatch. TYK2 JH2 (pseudokinase) binding constants are not comparable
   with JAK1/2/3 catalytic-site IC50s: they measure different sites by different
   methods. Selectivity for JH2 compounds is reported but flagged as
   cross-assay, and should be read as indicative only.

Writes data/processed/tyk2_selectivity.csv.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
OFF_TARGETS = ["jak1", "jak2", "jak3"]
SEL_COLS = [f"sel_{n}" for n in OFF_TARGETS]


def primary_domain(domains: str) -> str:
    s = set(str(domains).split(","))
    if s == {"JH2"}:
        return "JH2"
    if s == {"JH1"}:
        return "JH1"
    return "Other"


def build_selectivity_table() -> pd.DataFrame:
    tyk2 = pd.read_csv(PROCESSED_DIR / "tyk2_compounds.csv")
    tyk2["domain"] = tyk2["domains"].map(primary_domain)
    table = tyk2[["smiles_std", "molecule_chembl_id", "pchembl_median", "pchembl_measured",
                  "binned_only", "domain"]]
    table = table.rename(columns={"pchembl_median": "p_tyk2_all",
                                  "pchembl_measured": "p_tyk2",
                                  "binned_only": "tyk2_binned_only"})

    for name in OFF_TARGETS:
        off = pd.read_csv(PROCESSED_DIR / f"{name}_compounds.csv")
        off = off[["smiles_std", "pchembl_median", "pchembl_measured"]].rename(columns={
            "pchembl_median": f"p_{name}_all", "pchembl_measured": f"p_{name}"})
        table = table.merge(off, on="smiles_std", how="left")
        # Measured on both sides only: differencing binned values is meaningless
        table[f"sel_{name}"] = table["p_tyk2"] - table[f"p_{name}"]

    table["n_jak_tested"] = table[SEL_COLS].notna().sum(axis=1)
    # Worst-case window over the JAKs the compound was tested against
    table["min_selectivity"] = table[SEL_COLS].min(axis=1)
    # JH2 binding vs JAK catalytic IC50 is not a like-for-like comparison
    table["cross_assay"] = table["domain"] == "JH2"
    return table


def main() -> None:
    table = build_selectivity_table()
    out = PROCESSED_DIR / "tyk2_selectivity.csv"
    table.to_csv(out, index=False)

    tested = table[table["n_jak_tested"] > 0]
    print(f"{len(tested)} of {len(table)} TYK2 compounds have measured values for "
          f"TYK2 and at least one other JAK")
    print(f"{(table['n_jak_tested'] == 3).sum()} have measured values for all three")

    print("\nMedian selectivity (log units) by TYK2 assay domain:")
    print(tested.groupby("domain")[SEL_COLS + ["min_selectivity"]].median().round(2).to_string())
    print("\nCompounds per domain contributing to those medians:")
    print(tested.groupby("domain")[SEL_COLS].count().to_string())

    strict = tested[(tested["min_selectivity"] >= 2) & ~tested["cross_assay"]]
    print(f"\n>=100-fold selective over every JAK tested (same-assay only): {len(strict)}")
    print(f"  ... including cross-assay JH2 compounds: "
          f"{(tested['min_selectivity'] >= 2).sum()}")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
