"""v4 training: regression head with 15-min thresholding.

Predicts adelay (minutes) directly with LightGBM and XGBoost regressors,
then thresholds the prediction at 15 min for the binary metrics. The intuition:
classification at a fixed cutoff throws away the fact that a 4-minute and a
14-minute delay are very different events even though they share a label.
Regression keeps that gradient.

Same preprocessing, split, and hyperparameters as train_v3.py so the
classification metrics are directly comparable.

Run from the repo root:
    python notebooks/train_v4.py
"""
from __future__ import annotations

import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from xgboost import XGBRegressor

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
ARCHIVE = DATA_DIR / "data.tar.xz"
CSV_PATH = DATA_DIR / "data_for_model_final.csv"

DELAY_THRESHOLD_MIN = 15
ADELAY_CAP_MIN = 120
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
    capped = df["adelay"].clip(upper=ADELAY_CAP_MIN)
    by_train = capped.groupby(df["zugnr"])
    df["prev_delay"] = by_train.shift(1)
    df["roll7_delay"] = by_train.transform(
        lambda s: s.shift(1).rolling(7, min_periods=1).mean()
    )
    df["roll30_delay"] = by_train.transform(
        lambda s: s.shift(1).rolling(30, min_periods=1).mean()
    )
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


def build_models() -> dict[str, Pipeline]:
    return {
        "LightGBM regressor": Pipeline(
            [
                ("pre", make_preprocessor()),
                (
                    "model",
                    LGBMRegressor(
                        n_estimators=1000,
                        learning_rate=0.05,
                        max_depth=6,
                        num_leaves=64,
                        random_state=RANDOM_STATE,
                        verbose=-1,
                    ),
                ),
            ]
        ),
        "XGBoost regressor": Pipeline(
            [
                ("pre", make_preprocessor()),
                (
                    "model",
                    XGBRegressor(
                        n_estimators=1000,
                        learning_rate=0.05,
                        max_depth=6,
                        random_state=RANDOM_STATE,
                        tree_method="hist",
                        verbosity=0,
                    ),
                ),
            ]
        ),
    }


def evaluate(name: str, model: Pipeline, X_test, y_test_cont, y_test_bin) -> dict:
    y_pred_cont = model.predict(X_test)
    y_pred_bin = (y_pred_cont > DELAY_THRESHOLD_MIN).astype(int)

    mae = mean_absolute_error(y_test_cont, y_pred_cont)
    rmse = float(np.sqrt(mean_squared_error(y_test_cont, y_pred_cont)))
    metrics = {
        "name": name,
        "mae": mae,
        "rmse": rmse,
        "accuracy": accuracy_score(y_test_bin, y_pred_bin),
        "bal_acc": balanced_accuracy_score(y_test_bin, y_pred_bin),
        "roc_auc": roc_auc_score(y_test_bin, y_pred_cont),
        "pr_auc": average_precision_score(y_test_bin, y_pred_cont),
        "bad_recall": recall_score(y_test_bin, y_pred_bin, pos_label=1),
        "bad_f1": f1_score(y_test_bin, y_pred_bin, pos_label=1),
    }
    print(f"\n=== {name} ===")
    print(f"Regression MAE:       {mae:.3f} min")
    print(f"Regression RMSE:      {rmse:.3f} min")
    print(f"Accuracy @ >15 min:   {metrics['accuracy']:.4f}")
    print(f"Balanced accuracy:    {metrics['bal_acc']:.4f}")
    print(f"ROC-AUC (raw score):  {metrics['roc_auc']:.4f}")
    print(f"PR-AUC (raw score):   {metrics['pr_auc']:.4f}")
    print("Confusion matrix @ threshold=15 min [0 good, 1 bad]:")
    print(confusion_matrix(y_test_bin, y_pred_bin, labels=[0, 1]))
    print(classification_report(y_test_bin, y_pred_bin, target_names=["good (0)", "bad (1)"]))
    return metrics


def main() -> None:
    csv_path = ensure_csv()
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    df = df[df["adelay"] != 9999].copy()
    df = add_lag_features(df)
    df["adelay_cap"] = df["adelay"].clip(upper=ADELAY_CAP_MIN)
    df["bad"] = (df["adelay_cap"] > DELAY_THRESHOLD_MIN).astype(int)
    print(f"  rows: {len(df):,}  "
          f"class balance: {df['bad'].value_counts().to_dict()}")
    print(f"  adelay stats (capped at {ADELAY_CAP_MIN}): "
          f"mean={df['adelay_cap'].mean():.2f}  median={df['adelay_cap'].median():.1f}  "
          f"std={df['adelay_cap'].std():.2f}  max={df['adelay_cap'].max():.0f}")

    feature_cols = NUMERICAL_FEATURES + NOMINAL_FEATURES + ORDINAL_FEATURES
    X = df[feature_cols]
    y_cont = df["adelay_cap"].to_numpy()
    y_bin = df["bad"].to_numpy()

    X_train, X_test, y_train_cont, y_test_cont, y_train_bin, y_test_bin = train_test_split(
        X, y_cont, y_bin, test_size=0.2, random_state=RANDOM_STATE, stratify=y_bin
    )
    print(f"  train: {len(X_train):,}   test: {len(X_test):,}")

    baseline_acc = (y_test_bin == 0).mean()
    print(f"  trivial baseline accuracy (always good): {baseline_acc:.4f}")

    models = build_models()
    results = []
    for name, model in models.items():
        print(f"\nFitting {name}...")
        model.fit(X_train, y_train_cont)
        results.append(evaluate(name, model, X_test, y_test_cont, y_test_bin))

    print("\n" + "=" * 100)
    print("Summary (regression metrics on adelay; classification metrics @ threshold=15 min)")
    print("=" * 100)
    header = (
        f"{'model':22s} {'MAE':>7s} {'RMSE':>7s} "
        f"{'acc':>7s} {'bal_acc':>9s} {'ROC-AUC':>9s} {'PR-AUC':>8s} "
        f"{'bad_recall':>11s} {'bad_F1':>8s}"
    )
    print(header)
    print("-" * 100)
    for r in sorted(results, key=lambda d: -d["roc_auc"]):
        print(
            f"{r['name']:22s} {r['mae']:7.3f} {r['rmse']:7.3f} "
            f"{r['accuracy']:7.4f} {r['bal_acc']:9.4f} {r['roc_auc']:9.4f} "
            f"{r['pr_auc']:8.4f} {r['bad_recall']:11.4f} {r['bad_f1']:8.4f}"
        )
    print("\nFor reference, v3 (classification) tied at ROC-AUC ~0.810, PR-AUC ~0.56.")
    print("Anything materially above those means the regression is using delay magnitude information")
    print("that the binary classifier was throwing away.")


if __name__ == "__main__":
    main()
