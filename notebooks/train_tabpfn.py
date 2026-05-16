"""Train TabPFN on the same split as train_simple.py for direct comparison.

TabPFN-3 (package `tabpfn`, v8+) handles up to 1M rows x 200 features, so our
~160k x ~17 train set fits natively. Requires `pip install tabpfn` and a CUDA
GPU for reasonable runtimes.

Run from the repo root:
    python notebooks/train_tabpfn.py
"""
from __future__ import annotations

import tarfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
MODEL_DIR = DATA_DIR / "model"
ARCHIVE = DATA_DIR / "data.tar.xz"
CSV_PATH = MODEL_DIR / "data_for_model_final.csv"

NUMERICAL_FEATURES = [
    "sin_time",
    "cos_time",
    "sin_day",
    "cos_day",
    "mean_delay",
    "temp_max_combined",
    "temp_min_combined",
    "prcp_max_combined",
    "snow_max_combined",
    "wspd_max_combined",
    "wpgt_max_combined",
]
CATEGORICAL_FEATURES = [
    "trip",
    "weekday",
    "public_holiday",
    "covid_lockdown",
    "coco_max_combined",
]
TARGET = "target_good_bad"
RANDOM_STATE = 42


def ensure_csv() -> Path:
    if CSV_PATH.exists():
        return CSV_PATH
    if not ARCHIVE.exists():
        raise FileNotFoundError(f"Neither {CSV_PATH} nor {ARCHIVE} exists")
    print(f"Extracting {CSV_PATH.name} from {ARCHIVE.name}...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(ARCHIVE, "r:xz") as tar:
        member = tar.getmember(f"data/{CSV_PATH.name}")
        member.name = CSV_PATH.name
        tar.extract(member, path=MODEL_DIR)
    return CSV_PATH


def encode_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    """Return X with cat columns as int codes, plus their column indices."""
    feature_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_cols].copy()
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype("category").cat.codes.astype(np.int64)
    cat_indices = [feature_cols.index(c) for c in CATEGORICAL_FEATURES]
    return X, cat_indices


def main() -> None:
    csv_path = ensure_csv()
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"  rows: {len(df):,}  cols: {df.shape[1]}")

    X, cat_indices = encode_features(df)
    y = df[TARGET].astype(int).to_numpy()
    print(f"  class balance: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"  categorical column indices: {cat_indices}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  train: {len(X_train):,}   test: {len(X_test):,}")

    clf = TabPFNClassifier(
        device="auto",
        categorical_features_indices=cat_indices,
        random_state=RANDOM_STATE,
        show_progress_bar=True,
    )

    t0 = time.time()
    print("\nFitting TabPFN...")
    clf.fit(X_train, y_train)
    fit_elapsed = time.time() - t0

    t0 = time.time()
    print("Predicting on test set...")
    y_pred = clf.predict(X_test)
    pred_elapsed = time.time() - t0

    print("\n=== TabPFN test metrics ===")
    print(f"Fit wall time:        {fit_elapsed:.1f}s")
    print(f"Predict wall time:    {pred_elapsed:.1f}s")
    print(f"Accuracy:             {accuracy_score(y_test, y_pred):.4f}")
    print(f"Balanced accuracy:    {balanced_accuracy_score(y_test, y_pred):.4f}")
    print("\nConfusion matrix [0 bad, 1 good]:")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["bad (0)", "good (1)"]))


if __name__ == "__main__":
    main()
