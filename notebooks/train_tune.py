"""Hyperparameter tuning + LightGBM vs XGBoost comparison.

Runs a RandomizedSearchCV (20 iterations, 3-fold) on each model over the
same train split, then prints best CV score, best params, and held-out
test-set metrics side by side.

Run from the repo root:
    python notebooks/train_tune.py
"""
from __future__ import annotations

import tarfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import loguniform, uniform
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from xgboost import XGBClassifier

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
NOMINAL_FEATURES = ["trip", "weekday", "public_holiday", "covid_lockdown"]
ORDINAL_FEATURES = ["coco_max_combined"]
TARGET = "target_good_bad"

N_ITER = 20
CV_FOLDS = 3
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


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", MinMaxScaler(), NUMERICAL_FEATURES),
            ("nom", OneHotEncoder(handle_unknown="ignore"), NOMINAL_FEATURES),
            ("ord", OrdinalEncoder(), ORDINAL_FEATURES),
        ]
    )


def lgbm_search(X_train, y_train) -> RandomizedSearchCV:
    pipe = Pipeline(
        [
            ("preprocess", make_preprocessor()),
            (
                "model",
                LGBMClassifier(
                    n_estimators=500,
                    random_state=RANDOM_STATE,
                    verbose=-1,
                    n_jobs=1,
                ),
            ),
        ]
    )
    param_dist = {
        "model__num_leaves": [15, 31, 63, 127],
        "model__max_depth": [-1, 6, 8, 10],
        "model__learning_rate": loguniform(0.01, 0.2),
        "model__min_child_samples": [10, 20, 50, 100],
        "model__reg_alpha": loguniform(1e-3, 1.0),
        "model__reg_lambda": loguniform(1e-3, 1.0),
        "model__subsample": uniform(0.6, 0.4),
        "model__colsample_bytree": uniform(0.6, 0.4),
    }
    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=N_ITER,
        cv=CV_FOLDS,
        scoring="accuracy",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
    )
    search.fit(X_train, y_train)
    return search


def xgb_search(X_train, y_train) -> RandomizedSearchCV:
    pipe = Pipeline(
        [
            ("preprocess", make_preprocessor()),
            (
                "model",
                XGBClassifier(
                    n_estimators=500,
                    random_state=RANDOM_STATE,
                    eval_metric="logloss",
                    verbosity=0,
                    n_jobs=1,
                    tree_method="hist",
                ),
            ),
        ]
    )
    param_dist = {
        "model__max_depth": [4, 6, 8, 10],
        "model__learning_rate": loguniform(0.01, 0.2),
        "model__min_child_weight": [1, 3, 5, 10],
        "model__subsample": uniform(0.6, 0.4),
        "model__colsample_bytree": uniform(0.6, 0.4),
        "model__reg_alpha": loguniform(1e-3, 1.0),
        "model__reg_lambda": loguniform(1e-3, 1.0),
        "model__gamma": loguniform(1e-3, 1.0),
    }
    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=N_ITER,
        cv=CV_FOLDS,
        scoring="accuracy",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
    )
    search.fit(X_train, y_train)
    return search


def report(name: str, search: RandomizedSearchCV, X_test, y_test, elapsed: float) -> dict:
    y_pred = search.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    bacc = balanced_accuracy_score(y_test, y_pred)
    print(f"\n=== {name} ===")
    print(f"Wall time:            {elapsed:.1f}s")
    print(f"Best CV accuracy:     {search.best_score_:.4f}")
    print(f"Test accuracy:        {acc:.4f}")
    print(f"Test balanced acc.:   {bacc:.4f}")
    print("Best params:")
    for k, v in sorted(search.best_params_.items()):
        key = k.replace("model__", "")
        if isinstance(v, float):
            print(f"  {key:22s} {v:.4g}")
        else:
            print(f"  {key:22s} {v}")
    print("Confusion matrix [0 bad, 1 good]:")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))
    print(classification_report(y_test, y_pred, target_names=["bad (0)", "good (1)"]))
    return {"name": name, "cv": search.best_score_, "test_acc": acc, "test_bacc": bacc}


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
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  train: {len(X_train):,}   test: {len(X_test):,}")
    print(f"\nRandomizedSearchCV: n_iter={N_ITER}, cv={CV_FOLDS}, scoring=accuracy\n")

    np.random.seed(RANDOM_STATE)

    t0 = time.time()
    print(">>> Tuning LightGBM...")
    lgbm = lgbm_search(X_train, y_train)
    lgbm_elapsed = time.time() - t0

    t0 = time.time()
    print("\n>>> Tuning XGBoost...")
    xgb = xgb_search(X_train, y_train)
    xgb_elapsed = time.time() - t0

    results = [
        report("LightGBM", lgbm, X_test, y_test, lgbm_elapsed),
        report("XGBoost", xgb, X_test, y_test, xgb_elapsed),
    ]

    print("\n=== Comparison ===")
    print(f"{'model':10s}  {'CV acc':>8s}  {'test acc':>9s}  {'test bal.acc':>13s}")
    for r in results:
        print(f"{r['name']:10s}  {r['cv']:8.4f}  {r['test_acc']:9.4f}  {r['test_bacc']:13.4f}")


if __name__ == "__main__":
    main()
