"""v3 training: 4-variant ablation to isolate class-weighting effect.

Identical preprocessing to train_v2.py (15-min target, leak-free lag features),
but trains four models so the effect of class weighting alone is visible:

    LightGBM   x   {unweighted, balanced}
    XGBoost    x   {unweighted, scale_pos_weight}

A summary table at the end compares them against the trivial "predict always
good" baseline so it is obvious whether the headline accuracy is above or
below the no-model floor.

Run from the repo root:
    python notebooks/train_v3.py
"""
from __future__ import annotations

import tarfile
from pathlib import Path

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from xgboost import XGBClassifier

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


def build_models(scale_pos_weight: float) -> dict[str, Pipeline]:
    common_lgbm = dict(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=64,
        random_state=RANDOM_STATE,
        verbose=-1,
    )
    common_xgb = dict(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        tree_method="hist",
        verbosity=0,
    )
    return {
        "LightGBM (unweighted)": Pipeline(
            [("pre", make_preprocessor()), ("model", LGBMClassifier(**common_lgbm))]
        ),
        "LightGBM (balanced)": Pipeline(
            [
                ("pre", make_preprocessor()),
                ("model", LGBMClassifier(class_weight="balanced", **common_lgbm)),
            ]
        ),
        "XGBoost (unweighted)": Pipeline(
            [("pre", make_preprocessor()), ("model", XGBClassifier(**common_xgb))]
        ),
        "XGBoost (scale_pos_weight)": Pipeline(
            [
                ("pre", make_preprocessor()),
                ("model", XGBClassifier(scale_pos_weight=scale_pos_weight, **common_xgb)),
            ]
        ),
    }


def evaluate(name: str, model: Pipeline, X_test, y_test) -> dict:
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    metrics = {
        "name": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "bal_acc": balanced_accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
        "bad_recall": recall_score(y_test, y_pred, pos_label=1),
        "bad_f1": f1_score(y_test, y_pred, pos_label=1),
    }
    print(f"\n=== {name} ===")
    print(f"Accuracy:             {metrics['accuracy']:.4f}")
    print(f"Balanced accuracy:    {metrics['bal_acc']:.4f}")
    print(f"ROC-AUC:              {metrics['roc_auc']:.4f}")
    print(f"PR-AUC (bad class):   {metrics['pr_auc']:.4f}")
    print("Confusion matrix [0 good, 1 bad]:")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))
    print(classification_report(y_test, y_pred, target_names=["good (0)", "bad (1)"]))
    return metrics


def main() -> None:
    csv_path = ensure_csv()
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    df = df[df["adelay"] != 9999].copy()
    df = add_lag_features(df)
    df["bad"] = (df["adelay"] > DELAY_THRESHOLD_MIN).astype(int)
    print(f"  rows: {len(df):,}  "
          f"class balance: {df['bad'].value_counts().to_dict()}")

    feature_cols = NUMERICAL_FEATURES + NOMINAL_FEATURES + ORDINAL_FEATURES
    X = df[feature_cols]
    y = df["bad"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    n_good_tr = int((y_train == 0).sum())
    n_bad_tr = int((y_train == 1).sum())
    scale_pos_weight = n_good_tr / n_bad_tr
    print(f"  train: {len(X_train):,}   test: {len(X_test):,}")
    print(f"  scale_pos_weight: {scale_pos_weight:.3f}")

    baseline_acc = (y_test == 0).mean()
    print(f"  trivial baseline (always good) accuracy on test: {baseline_acc:.4f}")

    models = build_models(scale_pos_weight)
    results = []
    for name, model in models.items():
        print(f"\nFitting {name}...")
        model.fit(X_train, y_train)
        results.append(evaluate(name, model, X_test, y_test))

    print("\n" + "=" * 92)
    print("Summary (sorted by accuracy)")
    print("=" * 92)
    header = f"{'model':32s} {'acc':>7s} {'bal_acc':>9s} {'ROC-AUC':>9s} {'PR-AUC':>8s} {'bad_recall':>11s} {'bad_F1':>8s}"
    print(header)
    print("-" * 92)
    print(f"{'trivial (always good)':32s} {baseline_acc:7.4f} {0.5:9.4f} {0.5:9.4f} "
          f"{(y_test==1).mean():8.4f} {0.0:11.4f} {0.0:8.4f}")
    for r in sorted(results, key=lambda d: -d["accuracy"]):
        print(f"{r['name']:32s} {r['accuracy']:7.4f} {r['bal_acc']:9.4f} "
              f"{r['roc_auc']:9.4f} {r['pr_auc']:8.4f} {r['bad_recall']:11.4f} "
              f"{r['bad_f1']:8.4f}")


if __name__ == "__main__":
    main()
