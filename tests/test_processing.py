import pandas as pd
import pytest

from process_activities import aggregate, annotate_domain, flag_binned, standardise_smiles
from selectivity import primary_domain


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("CC(=O)Oc1ccccc1C(=O)O", "CC(=O)Oc1ccccc1C(=O)O"),         # unchanged
        ("CC(=O)Oc1ccccc1C(=O)O.[Na+].[Cl-]", "CC(=O)Oc1ccccc1C(=O)O"),  # salt stripped
        ("CC(=O)Oc1ccccc1C(=O)[O-].[Na+]", "CC(=O)Oc1ccccc1C(=O)O"),     # neutralised
        ("C[NH3+].[Cl-]", "CN"),
    ],
)
def test_standardise_smiles(raw, expected):
    assert standardise_smiles(raw) == expected


@pytest.mark.parametrize("bad", [None, float("nan"), "not_a_smiles"])
def test_standardise_smiles_invalid(bad):
    assert standardise_smiles(bad) is None


@pytest.mark.parametrize(
    "description, expected",
    [
        ("TYK2 JH2 Domain Binding Assay", "JH2"),
        ("Binding to TYK2 pseudokinase domain", "JH2"),
        ("Inhibition of TYK2 JH1 kinase", "JH1"),
        ("Inhibition of recombinant TYK2 kinase domain", "JH1"),
        ("Inhibition of TYK2 in human whole blood", "unspecified"),
        (None, "unspecified"),
    ],
)
def test_annotate_domain(description, expected):
    assert annotate_domain(description) == expected


def test_primary_domain():
    assert primary_domain("JH2") == "JH2"
    assert primary_domain("JH1") == "JH1"
    assert primary_domain("JH1,JH2") == "Other"
    assert primary_domain("unspecified") == "Other"


def _records(doc, values):
    return pd.DataFrame({"document_chembl_id": doc, "standard_value": values})


def test_flag_binned():
    binned = _records("PATENT", [0.6, 5.5] * 15)             # 30 records, 2 distinct values
    # Mostly binned, with a few exact values mixed in: caught by the top-share rule
    mostly = _records("PATENT2", [0.6] * 20 + [5.5] * 8 + [1.1, 2.3])
    measured = _records("PAPER", [float(v) for v in range(1, 31)])
    small = _records("SMALL", [5.5] * 5)                     # too few records to judge
    df = pd.concat([binned, mostly, measured, small], ignore_index=True)
    flags = flag_binned(df)
    assert flags[df["document_chembl_id"] == "PATENT"].all()
    assert flags[df["document_chembl_id"] == "PATENT2"].all()
    assert not flags[df["document_chembl_id"] == "PAPER"].any()
    assert not flags[df["document_chembl_id"] == "SMALL"].any()


def test_aggregate_separates_measured_and_binned():
    df = pd.DataFrame({
        "smiles_std": ["CCO", "CCO", "CCN"],
        "molecule_chembl_id": ["A", "A", "B"],
        "pchembl_value": [8.0, 9.0, 6.0],
        "standard_type": ["IC50", "Kd", "IC50"],
        "domain": ["JH2", "JH2", "unspecified"],
        "binned": [False, True, True],
    })
    out = aggregate(df).set_index("smiles_std")
    assert out.loc["CCO", "pchembl_median"] == 8.5
    assert out.loc["CCO", "pchembl_measured"] == 8.0   # binned record excluded
    assert out.loc["CCO", "n_measured"] == 1
    assert not out.loc["CCO", "binned_only"]
    assert out.loc["CCN", "binned_only"]
    assert pd.isna(out.loc["CCN", "pchembl_measured"])
    assert out.loc["CCO", "active"] == 1 and out.loc["CCN", "active"] == 0
