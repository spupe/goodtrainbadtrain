"""v2 training: 15-min target, leak-free lag features, class-weighted models.

Changes vs. train_simple.py:
  - Target: bad = (adelay > 15). The positive class (1) is now "delayed".
  - Lag features per train number (prev arrival, rolling 7/30 means) replace
    the leaky `mean_delay` column (which was computed across the full
    dataset, including each row's own adelay).
  - LightGBM with class_weight='balanced' and XGBoost with scale_pos_weight
    so the minority "bad" class isn't neglected.
  - Reports ROC-AUC and PR-AUC alongside accuracy for threshold-independent
    comparison across runs.

Run from the repo root:
    python notebooks/train_v2.py
"""
from __future__ import annotations

import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
ARCHIVE = DATA_DIR / "data.tar.xz"
CSV_PATH = DATA_DIR / "data_for_model_final.csv"

DELAY_THRESHOLD_MIN = 15
ADELAY_CAP_MIN = 120  # 9999 (no arrival) is filtered; remaining outliers capped
RANDOM_STATE = 42

NUMERICAL_FEATURES = [
    "sin_time",
    "cos_time",
    "sin_day",
    "cos_day",
    "prev_delay",
    "roll7_delay",
    "roll30_delay",
    "temp_max_combined",
    "temp_min_combined",
    "prcp_max_combined",
    "snow_max_combined",
    "wspd_max_combined",
    "wpgt_max_combined",
]
NOMINAL_FEATURES = ["trip", "weekday", "public_holiday", "covid_lockdown"]
ORDINAL_FEATURES = ["coco_max_combined"]


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


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["zugnr", "date"]).reset_index(drop=True)
    # Cap pathological values; 9999 rows are already filtered upstream.
    capped = df["adelay"].clip(upper=ADELAY_CAP_MIN)
    by_train = capped.groupby(df["zugnr"])
    df["prev_delay"] = by_train.shift(1)
    df["roll7_delay"] = by_train.transform(
        lambda s: s.shift(1).rolling(7, min_periods=1).mean()
    )
    df["roll30_delay"] = by_train.transform(
        lambda s: s.shift(1).rolling(30, min_periods=1).mean()
    )
    # Fill the first-ever arrival of each train (no history) with the global mean.
    fill = capped.mean()
    for col in ("prev_delay", "roll7_delay", "roll30_delay"):
        df[col] = df[col].fillna(fill)
    return df


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", MinMaxScaler(), NUMERICAL_FEATURES),
            ("nom", OneHotEncoder(handle_unknown="ignore"), NOMINAL_FEATURES),
            ("ord", OrdinalEncoder(), ORDINAL_FEATURES),
        ]
    )


def report(name: str, y_test, y_pred, y_proba) -> None:
    print(f"\n=== {name} ===")
    print(f"Accuracy:             {accuracy_score(y_test, y_pred):.4f}")
    print(f"Balanced accuracy:    {balanced_accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC:              {roc_auc_score(y_test, y_proba):.4f}")
    print(f"PR-AUC (bad class):   {average_precision_score(y_test, y_proba):.4f}")
    print("Confusion matrix [0 good, 1 bad]:")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))
    print(classification_report(y_test, y_pred, target_names=["good (0)", "bad (1)"]))


def main() -> None:
    csv_path = ensure_csv()
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"  raw rows: {len(df):,}")

    df = df[df["adelay"] != 9999].copy()
    print(f"  after dropping adelay==9999: {len(df):,}")

    df = add_lag_features(df)

    df["bad"] = (df["adelay"] > DELAY_THRESHOLD_MIN).astype(int)
    print(f"  target class balance (1=bad >{DELAY_THRESHOLD_MIN}min): "
          f"{df['bad'].value_counts().to_dict()}")

    feature_cols = NUMERICAL_FEATURES + NOMINAL_FEATURES + ORDINAL_FEATURES
    X = df[feature_cols]
    y = df["bad"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  train: {len(X_train):,}   test: {len(X_test):,}")

    n_good = int((y_train == 0).sum())
    n_bad = int((y_train == 1).sum())
    scale_pos_weight = n_good / n_bad
    print(f"  scale_pos_weight (n_good / n_bad in train): {scale_pos_weight:.3f}")

    pre = make_preprocessor()

    lgbm = Pipeline(
        [
            ("preprocess", pre),
            (
                "model",
                LGBMClassifier(
                    n_estimators=1000,
                    learning_rate=0.05,
                    max_depth=6,
                    num_leaves=64,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    verbose=-1,
                ),
            ),
        ]
    )
    xgb = Pipeline(
        [
            ("preprocess", make_preprocessor()),
            (
                "model",
                XGBClassifier(
                    n_estimators=1000,
                    learning_rate=0.05,
                    max_depth=6,
                    scale_pos_weight=scale_pos_weight,
                    random_state=RANDOM_STATE,
                    eval_metric="logloss",
                    tree_method="hist",
                    verbosity=0,
                ),
            ),
        ]
    )

    print("\nFitting LightGBM (class_weight=balanced)...")
    lgbm.fit(X_train, y_train)
    print("Fitting XGBoost (scale_pos_weight)...")
    xgb.fit(X_train, y_train)

    for name, model in [("LightGBM (balanced)", lgbm), ("XGBoost (scale_pos_weight)", xgb)]:
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        report(name, y_test, y_pred, y_proba)


if __name__ == "__main__":
    main()
