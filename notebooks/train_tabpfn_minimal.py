"""Minimal-feature TabPFN: how much can 5 raw categoricals do?

Strips the model down to the most basic, un-engineered signals:
    bhf, time_of_day, weekday, month, trip

No weather, no lag features, no holiday/lockdown flags, no cyclical encoding.
TabPFN's built-in categorical support handles all five directly.

Same 15-min target and 80/20 split as train_v5.py so the metrics can be
compared head-to-head against the full-feature pipeline.

Run from the repo root:
    python notebooks/train_tabpfn_minimal.py
"""
from __future__ import annotations

import tarfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
ARCHIVE = DATA_DIR / "data.tar.xz"
CSV_PATH = DATA_DIR / "data_for_model_final.csv"

DELAY_THRESHOLD_MIN = 15
RANDOM_STATE = 42

MINIMAL_FEATURES = ["bhf", "time_of_day", "weekday", "month", "trip"]


def ensure_csv() -> Path:
    if CSV_PATH.exists():
        return CSV_PATH
    if not ARCHIVE.exists():
        raise FileNotFoundError(f"Neither {CSV_PATH} nor {ARCHIVE} exists")
    print(f"Extracting {CSV_PATH.name} from {ARCHIVE.name}...")
    with tarfile.open(ARCHIVE, "r:xz") as tar:
        member = tar.getmember(f"data/{CSV_PATH.name}")
        member.name = CSV_PATH.name
        tar.extract(member, path=DATA_DIR)
    return CSV_PATH


def encode_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    X = df[MINIMAL_FEATURES].copy()
    for col in MINIMAL_FEATURES:
        X[col] = X[col].astype("category").cat.codes.astype(np.int64)
    cat_indices = list(range(len(MINIMAL_FEATURES)))
    return X, cat_indices


def main() -> None:
    csv_path = ensure_csv()
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    df = df[df["adelay"] != 9999].copy()
    df["bad"] = (df["adelay"] > DELAY_THRESHOLD_MIN).astype(int)
    print(f"  rows: {len(df):,}  "
          f"class balance: {df['bad'].value_counts().to_dict()}")
    print(f"  features ({len(MINIMAL_FEATURES)}): {MINIMAL_FEATURES}")
    for col in MINIMAL_FEATURES:
        print(f"    {col}: {df[col].nunique()} unique values")

    X, cat_indices = encode_features(df)
    y = df["bad"].to_numpy()

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
    print("\nFitting TabPFN (minimal features)...")
    clf.fit(X_train, y_train)
    fit_elapsed = time.time() - t0

    t0 = time.time()
    print("Predicting on test set...")
    y_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    pred_elapsed = time.time() - t0

    print("\n=== TabPFN (minimal features) test metrics ===")
    print(f"Fit wall time:        {fit_elapsed:.1f}s")
    print(f"Predict wall time:    {pred_elapsed:.1f}s")
    print(f"Accuracy:             {accuracy_score(y_test, y_pred):.4f}")
    print(f"Balanced accuracy:    {balanced_accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC:              {roc_auc_score(y_test, y_proba):.4f}")
    print(f"PR-AUC (bad class):   {average_precision_score(y_test, y_proba):.4f}")
    print(f"bad recall:           {recall_score(y_test, y_pred, pos_label=1):.4f}")
    print(f"bad F1:               {f1_score(y_test, y_pred, pos_label=1):.4f}")
    print("\nConfusion matrix [0 good, 1 bad]:")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["good (0)", "bad (1)"]))

    print("\nReference (full feature set, same target/split):")
    print("  v5 XGBoost (balanced):  ROC-AUC 0.8138, bad_recall 0.65, bad_F1 0.476")
    print("  v3 LightGBM (balanced): ROC-AUC 0.8095, bad_recall 0.65, bad_F1 0.477")


if __name__ == "__main__":
    main()
