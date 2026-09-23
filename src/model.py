"""QSAR models for TYK2 potency.

Two tasks, each evaluated with 5-fold cross-validation under a random split and a
scaffold-grouped split (whole Bemis-Murcko scaffolds held out together):

  regression      pChEMBL from measured (non-binned) values only
  classification  active = pChEMBL >= 7 (100 nM), using all compounds; binned
                  values can be used here because 100 nM is a bin edge

Outputs:
  results/model_metrics.csv       mean +/- sd of each metric per task/split/model
  results/cv_predictions.csv      out-of-fold predictions with nearest-train similarity
  results/models/*.joblib         final random forests fitted on all data
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import spearmanr
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import GroupKFold, KFold

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "tyk2_compounds.csv"
RESULTS_DIR = ROOT / "results"
MODEL_DIR = RESULTS_DIR / "models"

SEED = 42
N_FOLDS = 5
FP_RADIUS, FP_SIZE = 2, 2048

_fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_SIZE)


def featurise(smiles: pd.Series) -> tuple[np.ndarray, list]:
    """Morgan count fingerprints (model features) and bit fingerprints (similarity)."""
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    X = np.array([_fpgen.GetCountFingerprintAsNumPy(m) for m in mols], dtype=np.float32)
    fps = [_fpgen.GetFingerprint(m) for m in mols]
    return X, fps


def scaffold(smiles: str) -> str:
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles) or smiles
    except Exception:
        return smiles


def rf_regressor():
    return RandomForestRegressor(n_estimators=500, max_features=0.3, min_samples_leaf=1,
                                 n_jobs=-1, random_state=SEED)


def rf_classifier():
    return RandomForestClassifier(n_estimators=500, max_features="sqrt", class_weight="balanced",
                                  n_jobs=-1, random_state=SEED)


TASKS = {
    "regression": {
        "models": {"random_forest": rf_regressor,
                   "mean_baseline": lambda: DummyRegressor(strategy="mean")},
    },
    "classification": {
        "models": {"random_forest": rf_classifier,
                   "prior_baseline": lambda: DummyClassifier(strategy="prior")},
    },
}


def regression_metrics(y, pred) -> dict:
    return {
        "rmse": root_mean_squared_error(y, pred),
        "mae": mean_absolute_error(y, pred),
        "r2": r2_score(y, pred),
        "spearman": spearmanr(y, pred).statistic if np.ptp(pred) > 1e-6 else 0.0,
    }


def classification_metrics(y, prob) -> dict:
    pred = (prob >= 0.5).astype(int)
    return {
        "roc_auc": roc_auc_score(y, prob),
        "pr_auc": average_precision_score(y, prob),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "mcc": matthews_corrcoef(y, pred),
    }


def splits(split: str, n: int, groups: np.ndarray):
    if split == "random":
        return KFold(N_FOLDS, shuffle=True, random_state=SEED).split(np.zeros(n))
    return GroupKFold(N_FOLDS, shuffle=True, random_state=SEED).split(np.zeros(n), groups=groups)


def max_train_similarity(fps, train_idx, test_idx) -> np.ndarray:
    train_fps = [fps[i] for i in train_idx]
    return np.array([max(DataStructs.BulkTanimotoSimilarity(fps[i], train_fps)) for i in test_idx])


def cross_validate(task: str, df: pd.DataFrame, X: np.ndarray, fps: list, y: np.ndarray):
    metric_fn = regression_metrics if task == "regression" else classification_metrics
    groups = df["scaffold"].to_numpy()
    fold_scores, oof = [], []

    for split in ("random", "scaffold"):
        for fold, (tr, te) in enumerate(splits(split, len(df), groups)):
            sim = max_train_similarity(fps, tr, te)
            for model_name, make in TASKS[task]["models"].items():
                model = make().fit(X[tr], y[tr])
                pred = (model.predict(X[te]) if task == "regression"
                        else model.predict_proba(X[te])[:, 1])
                fold_scores.append({"task": task, "split": split, "model": model_name,
                                    "fold": fold, **metric_fn(y[te], pred)})
                if model_name == "random_forest":
                    oof.append(pd.DataFrame({
                        "task": task, "split": split, "fold": fold,
                        "molecule_chembl_id": df["molecule_chembl_id"].to_numpy()[te],
                        "y_true": y[te], "y_pred": pred, "max_train_sim": sim,
                    }))
            print(f"  {task:<14} {split:<8} fold {fold} done")
    return pd.DataFrame(fold_scores), pd.concat(oof, ignore_index=True)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    df_all = pd.read_csv(DATA_FILE)
    df_all["scaffold"] = df_all["smiles_std"].map(scaffold)
    datasets = {
        "regression": (df_all[df_all["pchembl_measured"].notna()].reset_index(drop=True),
                       "pchembl_measured"),
        "classification": (df_all, "active"),
    }

    all_scores, all_oof = [], []
    for task, (df, target) in datasets.items():
        print(f"{task}: {len(df)} compounds, {df['scaffold'].nunique()} scaffolds")
        X, fps = featurise(df["smiles_std"])
        y = df[target].to_numpy()
        scores, oof = cross_validate(task, df, X, fps, y)
        all_scores.append(scores)
        all_oof.append(oof)

        final = TASKS[task]["models"]["random_forest"]().fit(X, y)
        joblib.dump(final, MODEL_DIR / f"tyk2_rf_{task}.joblib", compress=3)

    scores = pd.concat(all_scores)
    metric_cols = [c for c in scores.columns if c not in ("task", "split", "model", "fold")]
    summary = (
        scores.groupby(["task", "split", "model"])[metric_cols]
        .agg(["mean", "std"]).round(3)
    )
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.dropna(axis=1, how="all").reset_index()
    summary.to_csv(RESULTS_DIR / "model_metrics.csv", index=False)
    pd.concat(all_oof).to_csv(RESULTS_DIR / "cv_predictions.csv", index=False)

    print("\nCross-validation summary (mean over folds):")
    show = [c for c in summary.columns if c.endswith("_mean")]
    print(summary[["task", "split", "model", *show]].to_string(index=False))


if __name__ == "__main__":
    main()
