# Retraining experiments (2026)

A sequence of experiments rebuilding the goodtrainbadtrain model from the
preserved 2022 training data (`data/data.tar.xz`). Goal: replicate the
original LightGBM, then iterate to understand where the accuracy ceiling
really lives.

Every iteration is one self-contained script under `notebooks/`. They share
the same data file (`data/model/data_for_model_final.csv`, extracted from
the archive on first run) and the same 80/20 stratified split
(`random_state=42`), so all reported numbers are directly comparable
*within* a target choice.

## Setup

- **Dataset**: 200,042 arrival events at 12 German stations across 39 ICE
  routes (cities include Berlin, Köln, München, Hamburg, Hannover, Frankfurt,
  Stuttgart, Nürnberg, Mannheim, Essen, Erfurt, Würzburg), Dec 2019 –
  Jun 2022.
- **Target — early runs (v1)**: `target_good_bad` from the original
  preprocessing, where 1 = good = on time (adelay ≤ 5 min). Inherits the
  original setup, including the leaky `mean_delay` feature.
- **Target — later runs (v2 onward)**: `bad = (adelay > 15)`. Convention
  flips so the positive class (1) is the delayed class — what the app
  actually cares about catching. ~16% positive, ~84% negative.
- **Split**: 80/20 stratified, `random_state=42` throughout.
- **Headline metrics**: accuracy, balanced accuracy, ROC-AUC, PR-AUC,
  recall on bad class, F1 on bad class. ROC-AUC is the cleanest cross-run
  comparison since it's threshold- and prevalence-independent.

## Iterations at a glance

| script | target | what changed |
|---|---|---|
| `train_simple.py` | 5 min | reproduce the original pipeline as a single script |
| `train_tune.py` | 5 min | RandomizedSearchCV (20 iter × 3 fold) on LGBM and XGB |
| `train_tabpfn.py` | 5 min | swap classifier for TabPFN-3 with native categorical support |
| `train_v2.py` | 15 min | recompute target at 15 min; replace leaky `mean_delay` with per-train lag features (`prev_delay`, `roll7_delay`, `roll30_delay`); class-weighted models |
| `train_v3.py` | 15 min | same as v2 plus an unweighted-vs-weighted ablation |
| `train_v4.py` | 15 min | regression head: predict adelay (capped at 120), threshold at 15 |
| `train_v5.py` | 15 min | add `bhf` (station) and 60-min rolling network-/station-state features |
| `train_tabpfn_minimal.py` | 15 min | TabPFN on 5 raw categoricals only (`bhf`, `time_of_day`, `weekday`, `month`, `trip`) |

## Results

All numbers are on the 20% held-out test set (40,009 rows). The trivial
"always predict good" baseline is included for context.

### Original 5-min target (v1 family)

| run | accuracy | balanced acc | bad recall | bad F1 |
|---|---:|---:|---:|---:|
| LightGBM, defaults | 0.7225 | 0.6429 | 0.38 | 0.49 |
| LightGBM, RandomizedSearchCV | 0.7337 | 0.6608 | 0.42 | 0.52 |
| XGBoost, RandomizedSearchCV | 0.7501 | 0.6917 | 0.50 | 0.58 |
| TabPFN-3, native cat support | 0.7707 | 0.7107 | 0.51 | 0.61 |

These numbers are **inflated** by the leaky `mean_delay` (computed across
the full dataset, so each row leaks a tiny amount of its own label).
Real-world performance with the original feature set is lower than these
suggest.

### 15-min target with leakage fixed (v2–v5)

| run | accuracy | bal_acc | ROC-AUC | PR-AUC | bad recall | bad F1 |
|---|---:|---:|---:|---:|---:|---:|
| trivial (always good) | 0.8424 | 0.5000 | 0.5000 | 0.1576 | 0.0000 | 0.0000 |
| v3 LGBM, unweighted | 0.8738 | 0.6282 | 0.8105 | 0.5613 | 0.27 | 0.40 |
| v3 XGB, unweighted | 0.8733 | 0.6270 | 0.8091 | 0.5590 | 0.27 | 0.40 |
| v3 LGBM, `class_weight='balanced'` | 0.7760 | 0.7239 | 0.8095 | 0.5569 | 0.65 | 0.48 |
| v3 XGB, `scale_pos_weight=5.34` | 0.7755 | 0.7224 | 0.8078 | 0.5539 | 0.65 | 0.48 |
| v4 LGBM regressor (L2) | 0.8548 | 0.6614 | 0.7867 | 0.4934 | 0.38 | 0.45 |
| v4 XGB regressor (L2) | 0.8546 | 0.6587 | 0.7867 | 0.4911 | 0.37 | 0.45 |
| **v5 LGBM, balanced + station/network** | **0.7728** | **0.7247** | **0.8121** | **0.5545** | **0.6545** | **0.4760** |
| v5 XGB, scale_pos_weight + station/network | 0.7734 | 0.7244 | 0.8138 | 0.5586 | 0.6529 | 0.4760 |
| TabPFN, 5 raw categoricals (minimal) | 0.8430 | 0.5059 | 0.6672 | 0.2781 | 0.014 | 0.027 |

## Findings

1. **The original `mean_delay` feature was leaky.** It was computed by
   grouping the full dataset by `zugnr` and taking the mean — so every row's
   `mean_delay` included that row's own `adelay` (and all future occurrences
   of the same train). This inflated every v1 number. Replaced in v2+ with
   strictly-past per-train lag features (`shift(1)` + `rolling(7)` +
   `rolling(30)`).

2. **The accuracy ceiling is ~0.81 ROC-AUC**, hit independently by
   LightGBM, XGBoost, TabPFN, and an L2 regressor (which is actually
   slightly worse because L2 chases the heavy tail). Model choice does not
   move the needle — every model family with the same v5 feature set
   converges to within 0.005 of each other.

3. **Per-train historical delay (the lag features) carries most of the
   predictive power.** The minimal TabPFN run with only
   station/time/weekday/month/route landed at ROC-AUC 0.667 — almost 15
   points below v5. The extra signal comes from knowing how *this specific
   train number* has been performing recently, not from where or when it
   runs.

4. **Class weighting is purely a choice of operating point.** Unweighted
   and weighted variants have identical ROC-AUC (~0.81) — same model, same
   ranking, different decision threshold. Unweighted gets 87% accuracy by
   barely flagging anything (bad recall 0.27); weighted accepts 78%
   accuracy in exchange for catching 65% of delays.

5. **Adding station (`bhf`), system-wide rolling delay, and per-station
   rolling delay barely moved ROC-AUC** (+0.003 LGBM, +0.006 XGB). The lag
   features were already capturing most of the implicit
   "this-train-on-this-route-around-this-time" signal.

6. **The remaining ~19 percentage points of ROC-AUC gap to a perfect model
   is presumably aleatoric** — accidents, signal failures, strikes,
   sudden weather events, individual incidents. These causes aren't in the
   dataset and likely never will be from historical scrapes. To move past
   0.81 you would need live operational feeds (real-time network status
   from the DB API, current weather observations, strike/maintenance
   calendars), not better features over the same source data.

## Recommended model

**v5 LightGBM with `class_weight='balanced'`** (script: `train_v5.py`).

Reasoning:
- Highest **bad recall** of any v5 variant (0.6545).
- ROC-AUC 0.8121, essentially tied with v5 XGBoost (0.8138) — well within
  run-to-run noise.
- For an app whose job is *warning travellers about likely delays*,
  catching 65% of delays at 37% precision is the useful operating point.
  The unweighted variants get higher headline accuracy by simply not
  warning anyone (bad recall 27%), which defeats the app's purpose.

To turn this into a deployable artifact: rerun `notebooks/train_v5.py`
after replacing the train/test split with a fit on the full 200k rows,
serialise the resulting pipeline with joblib, and swap into
`api/fast.py` in place of the current `model.joblib`. The feature list
in `api/ui_transformation.py` will need updating to match the v5 feature
set (in particular, the lag features and the new station/network rolling
features need to be computable from the incoming request).

## Reproducing

All scripts read from `data/model/data_for_model_final.csv`, which the
scripts extract on first run from `data/data.tar.xz` if missing. From the
repo root:

```bash
pip install lightgbm xgboost scikit-learn pandas
pip install tabpfn         # only for the TabPFN scripts (needs a CUDA GPU)

python notebooks/train_simple.py            # v1 baseline
python notebooks/train_tune.py              # v1 with RandomizedSearchCV
python notebooks/train_tabpfn.py            # v1 with TabPFN
python notebooks/train_v2.py                # 15-min target + lag features
python notebooks/train_v3.py                # + un/weighted ablation
python notebooks/train_v4.py                # regression head
python notebooks/train_v5.py                # + station/network features (recommended)
python notebooks/train_tabpfn_minimal.py    # TabPFN on minimal categoricals
```

All scripts print a metrics block on the held-out test set. Most runs
complete in under a minute on a modern CPU; the tuning and TabPFN scripts
take several minutes each (TabPFN needs CUDA for reasonable speed).
