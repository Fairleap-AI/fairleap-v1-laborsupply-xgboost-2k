<p align="center">
  <img src="assets/logo.png"/>
  <h1 align="center">fairleap-v1-laborsupply-xgboost-2k</h1>
</p>

An XGBoost regressor that forecasts a Gojek/GOTO driver's **daily labour supply** — hours worked —
from forecast earnings, calendar context, a wellness score, and the driver's own earnings history.

> ## 🚨 This model does not work
>
> **R² = −0.383 on its own test split — worse than predicting the mean.** 750 boosting rounds
> produced trees averaging 2.1 nodes; most are a single leaf and contributed nothing. It is
> published for provenance and reproducibility, not for use. See
> [Limitations](#%EF%B8%8F-limitations--bias) before doing anything with it.

It is the labour-supply half of the Fairleap forecasting pair. Its first input feature is the output
of [`fairleap-v1-earnings-xgboost-11k`](https://github.com/Fairleap-AI/fairleap-v1-earnings-xgboost-11k),
so it **cannot forecast standalone** — the caller supplies the earnings series.

## 📊 Model Details

| | |
|---|---|
| **Architecture** | `xgboost.sklearn.XGBRegressor`, `booster=gbtree` |
| **Objective** | `reg:squarederror` |
| **Boosting rounds** | 750 |
| **Max depth** | 3 |
| **Learning rate** | 0.3 |
| **Random state** | 42 |
| **Input features** | 21 (ordered — see below) |
| **Output** | 1 continuous value: predicted hours worked |
| **Total tree nodes** | 1,580 (415 splits + 1,165 leaves) — the "2k" in the name |
| **Mean nodes per tree** | 2.1 of a possible 15 — most trees are a bare leaf |
| **Artifact** | `app/hours_model.pkl`, 540 KB, `joblib` |
| **Version** | v1 |
| **License** | MIT |

The node count is itself a finding: a healthy depth-3 model would approach 15 nodes per tree, as its
earnings counterpart does at 14.1. At 2.1, gradient boosting found almost nothing to split on.

## 🎯 Intended Use

**Research and provenance only.** This artifact documents what the Fairleap v1 pipeline actually
shipped and provides a baseline for a replacement. It should not inform any decision.

### Out-of-Scope Use

- **Everything operational.** A model with negative R² carries less information than the training
  mean. Substituting a constant would be more accurate and more honest.
- **Scheduling, shift allocation, or workload guidance.** Do not tell a driver how long to work based
  on this output.
- **Pay, eligibility, credit, insurance, or employment decisions.** Never.
- **Fatigue, safety, or wellness inference.** `wellness_score` is an input, not something this model
  is validated to reason about.

## 🔢 Feature Schema

**Feature order is load-bearing.** This is a plain `XGBRegressor` with no column-name validation at
predict time — passing the right columns in the wrong order produces plausible numbers, not an error.

| # | Feature | Type | Description |
|---|---|---|---|
| 0 | `earnings` | float | **Forecast daily earnings, IDR.** Supplied by the caller |
| 1 | `day_of_week` | int 0–6 | Monday = 0 |
| 2 | `is_weekend` | int 0/1 | 1 when `day_of_week >= 5` |
| 3 | `wellness_score` | int | Self-reported wellness, constant across the forecast window |
| 4 | `rolling_mean_7` | float | Mean of the last 7 historical daily earnings |
| 5 | `rolling_std_7` | float | Population std of the last 7 historical daily earnings |
| 6 | `rolling_mean_14` | float | Mean of the last 14 historical daily earnings |
| 7–20 | `lag_1` … `lag_14` | float | Daily earnings 1–14 days before the target day |

Features 1–20 are identical to the earnings model's, in the same order; this model prepends
`earnings`. Rolling statistics are computed once from the tail of the supplied history and are
**constant across every day in a forecast window**. Lags fall back to `NaN` before the start of the
supplied history; XGBoost handles `NaN` natively.

## 🚀 How to Use

### Directly

```python
import joblib
import pandas as pd

model = joblib.load("app/hours_model.pkl")

FEATURES = ["earnings", "day_of_week", "is_weekend", "wellness_score",
            "rolling_mean_7", "rolling_std_7", "rolling_mean_14"] + \
           [f"lag_{i}" for i in range(1, 15)]

X = pd.DataFrame([{...}], columns=FEATURES)   # order matters, earnings first
hours = abs(model.predict(X)[0])
```

### As a service

```sh
pip install -r requirements.txt
python wsgi.py                                    # dev, port 5000
gunicorn --bind 0.0.0.0:5000 wsgi:app             # production
docker compose up                                 # container
```

`GET /` returns a healthcheck and the route table.

`POST /predict/hours` — feature construction from raw daily logs is handled for you by
`app/regressor_utils.py`. The `earnings` field takes the earnings service's `predictions` array
verbatim, so the two services pipe together directly:

```jsonc
{
  "start": "2025-05-13",
  "end": "2025-05-20",
  "wellness_score": 20,
  "daily_logs": [
    { "day": "2025-03-25", "total_earnings": 155000, "total_distance": 100.0,
      "total_fare": 150000, "total_tip": 5000, "total_trips": 8 }
  ],
  "earnings": [
    { "date": "2025-05-13", "earnings": 171613.640625 }
  ]
}
```

Every day in `[start, end]` must have an entry in `earnings`; a gap returns `400` rather than
reaching the model as `NaN`. Supply at least 14 days of `daily_logs` for the lag features to be
populated. Response:

```jsonc
{
  "status": "success",
  "unit": "hours",
  "predictions": [
    { "date": "2025-05-13", "predicted_hours_worked": 0.01619946025311947 }
  ]
}
```

Those magnitudes are real output, not a formatting error — see Limitations.

### Configuration

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_PATH` | `./app/hours_model.pkl` | Path to the joblib artifact |
| `PORT` | `5000` | Listen port |

The model is loaded at import time; a failure raises `RuntimeError` and the process will not start.

## 📚 Training Data

[**fairleap-ai/fairleap-driver-earnings-regression-500**](https://huggingface.co/datasets/fairleap-ai/fairleap-driver-earnings-regression-500) — 500 rows, MIT.

Fully synthetic ride-event records generated by `data_gen.py`, one row per completed ride:

| Column | Description |
|---|---|
| `driver_id` | Synthetic driver identifier |
| `timestamp` | Ride timestamp, unique and strictly increasing per driver |
| `day_of_week` | 0–6, Monday = 0 |
| `hour_of_day` | 0–23 |
| `location_cluster` | Indonesian city label |
| `hours_worked` | Hours attributed to the ride — **the target** |
| `rides_completed` | Rides in the record |
| `earnings` | Earnings in IDR — feature 0 |
| `wellness_score` | Self-reported driver wellness |
| `preferred_location` | Driver's stated preferred city |
| `avg_ride_duration_minutes` | Mean ride duration |

The target is per-ride hours, which is why it sits near 0.0067 rather than near a working day. This
is the root of the magnitude problem described below.

## 🔬 Training Procedure

Lags 1–14 and 7/14-day rolling statistics are derived from the `earnings` column, rows with
resulting `NaN`s are dropped, and the frame is split 80/20 with `train_test_split`.

```python
XGBRegressor(n_estimators=750, learning_rate=0.3, max_depth=3, random_state=42)
```

Identical hyperparameters to the earnings model. Nothing was tuned for this target.

## 📈 Evaluation

Held-out 20% split of the dataset above.

| Metric | Value | Reading |
|---|---|---|
| MAE | 0.0067 hours (~24 seconds) | Looks excellent; it is not — the target barely varies |
| R² | **−0.3833** | Worse than predicting the training mean |

MAE and R² disagree here because the target's variance is tiny. A low absolute error on a
near-constant target is not skill. **R² is the metric to read.**

## ⚠️ Limitations & Bias

- **Negative R².** The model is worse than a constant. This is the headline fact about it.
- **Degenerate trees.** 2.1 nodes per tree against a possible 15. Boosting found almost no usable
  splits, consistent with a target that carries little signal from these features.
- **Target/serving unit mismatch.** Trained on *per-ride* hours (~0.0067 h) but served as a *daily*
  forecast. The output is not interpretable as hours worked in a day, and no rescaling is applied.
- **Train/serve skew.** Lags and rolling windows are built at training time over per-ride event rows
  — the dataset carries `hour_of_day` and multiple rows per driver per day. At serving time
  `regressor_utils.py` builds one row per day with lags over daily totals. `lag_1` means "the
  previous ride" during training and "yesterday" during inference.
- **Compounding error.** Feature 0 comes from a model that itself scores R² = 0.397, so this model's
  main input is already substantially wrong before it starts.
- **Synthetic data only.** Never validated against real Gojek/GOTO driver data.
- **Static rolling features.** Rolling statistics do not advance across the forecast window.
- **No geographic or seasonal signal.** `location_cluster` and `preferred_location` exist in the
  dataset but not in the feature set. Holidays, weather, promotions and surge are absent entirely.
- **`abs()` on output.** Predictions are passed through `abs()`, masking rather than fixing negative
  predictions.
- **No uncertainty estimate.** A bare point prediction from a model with negative R² reads as far
  more confident than it is.

### If you are replacing this model

Aggregate the dataset to daily totals before building lags so training and serving agree; predict
daily hours rather than per-ride hours; and check that a constant baseline is beaten before tuning
anything else.

## 🛠️ Tech Stacks
- **xgboost**: An optimized gradient boosting library designed to be highly efficient, flexible, and portable for supervised learning problems.
- **scikit-learn**: A robust machine learning library that provides simple and efficient tools for data mining and data analysis.
- **pandas**: A powerful data manipulation and analysis library offering labeled data structures and operations for manipulating numerical tables and time series.
- **numpy**: A foundational library for numerical computing in Python, supporting large, multi-dimensional arrays and matrices.
- **joblib**: A library for lightweight pipelining and efficient serialization of Python objects, often used for persisting machine learning models.
- **flask**: A lightweight and flexible WSGI web application framework designed to get applications up and running quickly.
- **gunicorn**: A Python WSGI HTTP server for UNIX that's commonly used to serve Flask or Django web applications in production.

## ⚙️ Installation

```sh
git clone https://github.com/Fairleap-AI/fairleap-v1-laborsupply-xgboost-2k
cd fairleap-v1-laborsupply-xgboost-2k
docker compose up
```

## 📝 License
This project is licensed under the MIT License.
