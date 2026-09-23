"""Download bioactivity data for TYK2 and the other JAK family kinases from ChEMBL.

Saves raw activity records to data/raw/<target>_activities_raw.csv.

    python src/get_chembl_data.py            # TYK2 only
    python src/get_chembl_data.py --all      # TYK2 + JAK1/2/3 (for selectivity)
"""

import argparse
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"

# Human JAK family single-protein targets
TARGETS = {
    "tyk2": "CHEMBL3553",
    "jak1": "CHEMBL2835",
    "jak2": "CHEMBL2971",
    "jak3": "CHEMBL2148",
}
TYK2_TARGET_ID = TARGETS["tyk2"]

COLUMNS = [
    "activity_id",
    "molecule_chembl_id",
    "canonical_smiles",
    "standard_type",
    "standard_relation",
    "standard_value",
    "standard_units",
    "pchembl_value",
    "assay_chembl_id",
    "assay_type",
    "assay_description",
    "target_organism",
    "data_validity_comment",
    "potential_duplicate",
    "document_chembl_id",
    "document_year",
]


def fetch_activities(target_id: str = TYK2_TARGET_ID) -> pd.DataFrame:
    """Fetch all IC50/Ki/Kd activities with a pChEMBL value for a target."""
    params = {
        "target_chembl_id": target_id,
        "standard_type__in": "IC50,Ki,Kd",
        "standard_relation": "=",
        "pchembl_value__isnull": "false",
        "limit": 1000,
        "format": "json",
    }

    rows, url = [], f"{BASE_URL}/activity"
    with requests.Session() as session:
        while url:
            r = session.get(url, params=params, timeout=120)
            r.raise_for_status()
            data = r.json()
            rows.extend(data["activities"])
            meta = data["page_meta"]
            print(f"  fetched {len(rows)}/{meta['total_count']}")
            nxt = meta["next"]
            # The "next" link already carries the query string
            url, params = (f"https://www.ebi.ac.uk{nxt}", None) if nxt else (None, None)

    return pd.DataFrame(rows).reindex(columns=COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--all", action="store_true", help="fetch every target in TARGETS")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name in (TARGETS if args.all else ["tyk2"]):
        target_id = TARGETS[name]
        print(f"Fetching activities for {name.upper()} ({target_id})...")
        df = fetch_activities(target_id)
        out = RAW_DIR / f"{name}_activities_raw.csv"
        df.to_csv(out, index=False)
        print(f"Saved {len(df)} records "
              f"({df['molecule_chembl_id'].nunique()} unique molecules) to {out}")


if __name__ == "__main__":
    main()
