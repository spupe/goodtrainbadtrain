"""Simplified retraining of the goodtrainbadtrain LightGBM classifier.

Reproduces the original pipeline (same features, same target) but with a
plain 80/20 stratified split and no grid search / feature selection, so we
can quickly get a baseline test-set metric.

Run from the repo root:
    python notebooks/train_simple.py
"""
from __future__ import annotations

import tarfile
from pathlib import Path

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
ARCHIVE = DATA_DIR / "data.tar.xz"
CSV_PATH = DATA_DIR / "data_for_model_final.csv"

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
NOMINAL_FEATURES = ["trip", "weekday", "public_holiday", "covid_lockdown"]
ORDINAL_FEATURES = ["coco_max_combined"]
TARGET = "target_good_bad"


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


def build_pipeline() -> Pipeline:
    # The pre-formatted CSV has no NaNs, so no imputers are needed.
    preprocessor = ColumnTransformer(
        [
            ("num", MinMaxScaler(), NUMERICAL_FEATURES),
            ("nom", OneHotEncoder(handle_unknown="ignore"), NOMINAL_FEATURES),
            ("ord", OrdinalEncoder(), ORDINAL_FEATURES),
        ]
    )
    classifier = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=64,
        random_state=42,
    )
    return Pipeline([("preprocess", preprocessor), ("model", classifier)])


def main() -> None:
    csv_path = ensure_csv()
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"  rows: {len(df):,}  cols: {df.shape[1]}")

    feature_cols = NUMERICAL_FEATURES + NOMINAL_FEATURES + ORDINAL_FEATURES
    X = df[feature_cols]
    y = df[TARGET].astype(int)
    print(f"  class balance: {y.value_counts().to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  train: {len(X_train):,}   test: {len(X_test):,}")

    pipe = build_pipeline()
    print("Fitting LightGBM...")
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    print("\n=== Test set metrics ===")
    print(f"Accuracy:           {accuracy_score(y_test, y_pred):.4f}")
    print(f"Balanced accuracy:  {balanced_accuracy_score(y_test, y_pred):.4f}")
    print("\nConfusion matrix (rows=true, cols=pred), labels=[0 bad, 1 good]:")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["bad (0)", "good (1)"]))


if __name__ == "__main__":
    main()
