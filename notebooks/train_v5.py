"""v5 training: station + network-state features.

Adds three new signals on top of v3's setup:

  bhf                   nominal       arrival station, one-hot encoded
  net_delay_60min       numeric       mean adelay of ALL trains arriving
                                      in the 60 min before this row
  station_delay_60min   numeric       mean adelay at THIS station in
                                      the 60 min before this row

Both rolling windows use closed='left' so the current row's adelay is never
included in its own features (no leakage). Classification head with
class_weight='balanced' / scale_pos_weight, matching v3's best operating
point so the comparison is clean.

Run from the repo root:
    python notebooks/train_v5.py
"""
from __future__ import annotations

import tarfile
import time
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
NETWORK_WINDOW = "60min"
RANDOM_STATE = 42

NUMERICAL_FEATURES = [
    "sin_time",
    "cos_time",
    "sin_day",
    "cos_day",
    "prev_delay",
    "roll7_delay",
    "roll30_delay",
    "net_delay_60min",
    "station_delay_60min",
    "temp_max_combined",
    "temp_min_combined",
    "prcp_max_combined",
    "snow_max_combined",
    "wspd_max_combined",
    "wpgt_max_combined",
]
NOMINAL_FEATURES = ["trip", "weekday", "public_holiday", "covid_lockdown", "bhf"]
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


def add_network_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-windowed network-state features (no future leakage).

    `net_delay_60min`: rolling mean across all rows in the last 60 min.
    `station_delay_60min`: rolling mean per `bhf` over the last 60 min.

    Both use closed='left' so the current row is excluded from its own
    window.
    """
    work = df.copy()
    work["adelay_cap"] = work["adelay"].clip(upper=ADELAY_CAP_MIN)
    work["date_dt"] = pd.to_datetime(work["date"])

    # System-wide network state
    sys = work.sort_values("date_dt").set_index("date_dt")
    sys["net_delay_60min"] = (
        sys["adelay_cap"].rolling(NETWORK_WINDOW, closed="left").mean()
    )

    # Per-station network state
    station_parts = []
    for bhf, g in sys.groupby("bhf", sort=False):
        s = g["adelay_cap"].rolling(NETWORK_WINDOW, closed="left").mean()
        station_parts.append(s.rename("station_delay_60min"))
    station = pd.concat(station_parts).sort_index()
    sys["station_delay_60min"] = station

    out = sys.reset_index(drop=True)
    fill = work["adelay_cap"].mean()
    out["net_delay_60min"] = out["net_delay_60min"].fillna(fill)
    out["station_delay_60min"] = out["station_delay_60min"].fillna(fill)
    # Drop helper, keep the new feature columns aligned to row index
    return out.drop(columns=["adelay_cap"])


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", MinMaxScaler(), NUMERICAL_FEATURES),
            ("nom", OneHotEncoder(handle_unknown="ignore"), NOMINAL_FEATURES),
            ("ord", OrdinalEncoder(), ORDINAL_FEATURES),
        ]
    )


def build_models(scale_pos_weight: float) -> dict[str, Pipeline]:
    return {
        "LightGBM (balanced)": Pipeline(
            [
                ("pre", make_preprocessor()),
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
        ),
        "XGBoost (scale_pos_weight)": Pipeline(
            [
                ("pre", make_preprocessor()),
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

    t0 = time.time()
    df = add_lag_features(df)
    df = add_network_features(df)
    print(f"  feature construction: {time.time() - t0:.1f}s")

    df["bad"] = (df["adelay"] > DELAY_THRESHOLD_MIN).astype(int)
    print(f"  rows: {len(df):,}  "
          f"class balance: {df['bad'].value_counts().to_dict()}")
    print(f"  unique bhf (stations): {df['bhf'].nunique()}")
    print("  new feature stats:")
    print(df[["net_delay_60min", "station_delay_60min"]].describe().round(2))

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
    print(f"  trivial baseline accuracy: {baseline_acc:.4f}")

    models = build_models(scale_pos_weight)
    results = []
    for name, model in models.items():
        print(f"\nFitting {name}...")
        model.fit(X_train, y_train)
        results.append(evaluate(name, model, X_test, y_test))

    print("\n" + "=" * 92)
    print("Summary (v5: + bhf, + 60-min network state, + 60-min station state)")
    print("=" * 92)
    header = (
        f"{'model':32s} {'acc':>7s} {'bal_acc':>9s} {'ROC-AUC':>9s} "
        f"{'PR-AUC':>8s} {'bad_recall':>11s} {'bad_F1':>8s}"
    )
    print(header)
    print("-" * 92)
    for r in sorted(results, key=lambda d: -d["roc_auc"]):
        print(
            f"{r['name']:32s} {r['accuracy']:7.4f} {r['bal_acc']:9.4f} "
            f"{r['roc_auc']:9.4f} {r['pr_auc']:8.4f} {r['bad_recall']:11.4f} "
            f"{r['bad_f1']:8.4f}"
        )
    print("\nv3 reference (no station / no network state):")
    print(f"{'LightGBM (balanced)':32s} {0.7760:7.4f} {0.7239:9.4f} "
          f"{0.8095:9.4f} {0.5569:8.4f} {0.6477:11.4f} {0.4769:8.4f}")
    print(f"{'XGBoost (scale_pos_weight)':32s} {0.7755:7.4f} {0.7224:9.4f} "
          f"{0.8078:9.4f} {0.5539:8.4f} {0.6450:11.4f} {0.4753:8.4f}")


if __name__ == "__main__":
    main()
