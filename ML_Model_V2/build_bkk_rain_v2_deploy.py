# %% [markdown]
# # BKK_Rain_V2_Deploy — the fit that ships
#
# `BKK_Rain_V2` established that IMERG labels beat Open-Meteo labels at all six targets, but it
# was a *measurement* notebook: every model it fitted was scored and thrown away. Nothing on disk
# can produce a forecast. This notebook produces the deployable artifact — one final model per
# target fitted on all 2.45 M rows, each paired with an isotonic calibrator and an operating
# threshold, saved as a `.joblib` bundle.
#
# **The calibration problem.** A calibrator must never be fitted on rows its model was trained on,
# or it learns to correct an over-confidence that only exists in-sample. `BKK_Rain_V2` solved this
# inside each fold by holding back 12 % of the fold's training rows. A single final model cannot
# use that trick without either (a) throwing away 12 % of its training data, or (b) calibrating on
# a chronological tail that falls entirely inside post-2025-10-01 provisional Late Run IMERG, whose
# detection runs ~12 % low — that would bake the provisional bias into the calibration map.
#
# So this notebook **cross-fits** the calibrator, the scheme `sklearn`'s
# `CalibratedClassifierCV(ensemble=False)` uses:
#
# 1. Run leave-one-year-out CV to produce an out-of-fold probability for every row.
# 2. Fit **one** isotonic regression per target on all of those OOF probabilities.
# 3. Fit the final model on **100 %** of the rows.
#
# The calibrator never sees any model's training rows, and the shipped model keeps every row.
# The residual approximation — the fold models see ~80 % of the data and are therefore slightly
# less sharp than the final model the calibrator is applied to — is the accepted cost of the
# scheme, and is the reason the reported metrics below are, if anything, mildly pessimistic.
#
# The operating threshold is chosen the same way: F1-optimal on the calibrated OOF probabilities,
# which are out-of-sample for every row.

# %%
import gc
import json
import os
import platform
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psycopg2
import sklearn
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 240)
pd.set_option("display.float_format", "{:.4f}".format)

# Fitted on a numpy array, LightGBM still stamps placeholder feature names (Column_0..N) onto
# `feature_names_in_`, then warns on every numpy `predict_proba` that the names are missing.
# Matching is positional either way, so the warning carries no information here - it just fires
# once per fold per target and buries the log.
warnings.filterwarnings(
    "ignore", message="X does not have valid feature names", category=UserWarning)

# %% [markdown]
# ## 1. Configuration
#
# Hyperparameters, features and the model plan are **byte-identical to `BKK_Rain_V2`**, which in
# turn took them unchanged from `BKK_Rain_V1`. Nothing is re-tuned here; re-tuning on the same rows
# that produced the reported scores would invalidate them. The only new knobs are the ones that
# govern saving.

# %%
DB_CONFIG = {
    "host": os.getenv("PGHOST", "localhost"),
    "port": int(os.getenv("PGPORT", "5432")),
    "dbname": os.getenv("PGDATABASE", "postgres"),
    "user": os.getenv("PGUSER", "postgres"),
    "password": os.getenv("PGPASSWORD", "Pass1234"),
}

PRECOMPUTE_TABLE_NAME = '"OM_BKK_DATA_PRECOMPUTE"'
IMERG_TABLE_NAME = '"IMERG_BKK_DATA"'
PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name == "ML_Model_V2":            # allow running from either directory
    PROJECT_ROOT = PROJECT_ROOT.parent
V2_DIR = PROJECT_ROOT / "ML_Model_V2" / "trained_models" / "om_bkk_rain_v2_imerg"

HORIZONS = [1, 3, 6]
RAIN_THRESHOLDS_MM = [0.1, 1.0]
MAX_LEAD = max(HORIZONS)
MODEL_PLAN = {1: "lightgbm", 3: "hist_gradient_boosting", 6: "hist_gradient_boosting"}
RANDOM_STATE = 42
PURGE_HOURS = 24

# Smoke mode: keep every Nth grid cell but the full time span, so the fold structure, the
# calibration and the saving path all execute against real data in a couple of minutes.
SMOKE_GRID_MODULO = int(os.getenv("V2_DEPLOY_SMOKE", "0"))
SMOKE = SMOKE_GRID_MODULO > 0

# Folds smaller than this are dropped. Scaled down in smoke mode so the same six years survive
# subsampling - otherwise every year falls under the bar and there is no CV left to test.
MIN_FOLD_ROWS = 50_000 // SMOKE_GRID_MODULO if SMOKE else 50_000

# Smoke artifacts go somewhere else entirely. The operating threshold is part of each filename,
# so a smoke model would not overwrite its real counterpart - it would sit next to it, loadable,
# trained on a twentieth of the data. Separate directories make that mistake impossible.
OUTPUT_DIR = PROJECT_ROOT / "ML_Model_V2" / "trained_models" / (
    "om_bkk_rain_v2_deploy_smoke" if SMOKE else "om_bkk_rain_v2_deploy")

LOG_PATH = OUTPUT_DIR / "run.log"


def log(message):
    """Print, and mirror to a file so a long background run can be followed live."""
    stamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


log(f"mode={f'SMOKE (every {SMOKE_GRID_MODULO}th grid cell)' if SMOKE else 'FULL'}")
log(f"output -> {OUTPUT_DIR}")

# %% [markdown]
# ## 2. Features — unchanged from V1/V2
#
# `precipitation` and its lags stay in. They are ECMWF's forecast of rain, not an observation of
# it, and they remain the strongest features. That is the standing caveat on this whole model: the
# fix in V2 was to the *label*, so V2 still inherits whatever bias the ECMWF forecast carries — it
# has just stopped being graded on reproducing it.

# %%
BASELINE_FEATURE_COLUMNS = [
    "temperature_2m", "relative_humidity_2m", "pressure_msl", "surface_pressure",
    "dew_point_2m", "precipitation", "cloud_cover", "wind_speed_10m",
    "wind_direction_10m", "temperature_dew_point_spread", "pressure_msl_change_3h",
    "pressure_msl_change_6h", "precipitation_lag_1h", "precipitation_lag_2h",
    "precipitation_lag_3h", "precipitation_lag_6h", "precipitation_sum_past_3h",
    "precipitation_sum_past_6h", "precipitation_sum_past_12h", "precipitation_sum_past_24h",
    "cloud_cover_lag_1h", "cloud_cover_lag_3h", "cloud_cover_lag_6h",
    "humidity_lag_1h", "humidity_lag_3h", "humidity_lag_6h",
    "wind_speed_lag_1h", "wind_speed_lag_3h", "hour_sin", "hour_cos",
    "month_sin", "month_cos", "grid_row", "grid_column", "latitude", "longitude",
]

NEIGHBOR_FEATURE_COLUMNS = [
    "neighbor_count", "neighbor_precipitation_mean", "neighbor_precipitation_max",
    "neighbor_precipitation_sum", "neighbor_rain_count", "neighbor_rain_rate",
    "neighbor_cloud_cover_mean", "neighbor_cloud_cover_max", "neighbor_relative_humidity_mean",
    "neighbor_relative_humidity_max", "neighbor_pressure_msl_mean", "neighbor_pressure_msl_min",
    "neighbor_pressure_msl_max", "neighbor_temperature_2m_mean", "neighbor_dew_point_2m_mean",
    "neighbor_temperature_dew_point_spread_mean", "neighbor_wind_speed_10m_mean",
    "neighbor_wind_speed_10m_max", "row_minus_precipitation_mean", "row_plus_precipitation_mean",
    "column_minus_precipitation_mean", "column_plus_precipitation_mean", "row_minus_cloud_cover_mean",
    "row_plus_cloud_cover_mean", "column_minus_cloud_cover_mean", "column_plus_cloud_cover_mean",
    "neighbor_precipitation_mean_minus_center", "neighbor_cloud_cover_mean_minus_center",
    "neighbor_relative_humidity_mean_minus_center", "center_pressure_msl_minus_neighbor_mean",
]

FEATURE_COLUMNS = BASELINE_FEATURE_COLUMNS + NEIGHBOR_FEATURE_COLUMNS
log(f"features: {len(FEATURE_COLUMNS)} (identical to V1 and V2)")

# %% [markdown]
# ## 3. Load features and future IMERG rainfall
#
# Identical query to `BKK_Rain_V2`. `LEAD(precipitation_mm, k)` walks rows, not clock hours, so
# each lead is guarded: it counts only when its timestamp is exactly *k* hours after the anchor.
# Without that guard a gap in the IMERG series would silently pull a later hour into the label.

# %%
ROW_FILTER_SQL = '''
      om.pressure_msl_change_6h IS NOT NULL
      AND om.precipitation_lag_6h IS NOT NULL
      AND om.precipitation_sum_past_24h IS NOT NULL
      AND om.cloud_cover_lag_6h IS NOT NULL
      AND om.humidity_lag_6h IS NOT NULL
      AND om.wind_speed_lag_3h IS NOT NULL
      AND om.neighbor_count > 0
'''
SMOKE_FILTER_SQL = f"\n      AND om.grid_number % {SMOKE_GRID_MODULO} = 0\n" if SMOKE else ""


def connect():
    return psycopg2.connect(**DB_CONFIG)


def build_query():
    feature_sql = ",\n        ".join(
        f"COALESCE(om.{c}::real, 'NaN'::real) AS {c}" for c in FEATURE_COLUMNS)
    lead_sql = ",\n            ".join(
        f"LEAD(precipitation_mm, {k}) OVER w AS p{k}, "
        f"LEAD(local_observation_time, {k}) OVER w AS t{k}"
        for k in range(1, MAX_LEAD + 1))
    guarded_sql = ",\n        ".join(
        f"CASE WHEN im.t{k} = im.t_local + interval '{k} hour' "
        f"THEN im.p{k}::real ELSE 'NaN'::real END AS p{k}"
        for k in range(1, MAX_LEAD + 1))
    return f'''
    WITH im AS (
        SELECT grid_number, local_observation_time AS t_local,
            {lead_sql}
        FROM {IMERG_TABLE_NAME}
        WHERE is_complete_hour
        WINDOW w AS (PARTITION BY grid_number ORDER BY local_observation_time)
    )
    SELECT om.grid_number, om.local_forecast_time,
        {feature_sql},
        {guarded_sql}
    FROM {PRECOMPUTE_TABLE_NAME} om
    JOIN im ON im.grid_number = om.grid_number AND im.t_local = om.local_forecast_time
    WHERE {ROW_FILTER_SQL}{SMOKE_FILTER_SQL}
    ORDER BY om.local_forecast_time, om.grid_number
    '''


def read_training_data(batch_size=200_000):
    query = build_query()
    n_feat = len(FEATURE_COLUMNS)
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(f'''
                SELECT count(*) FROM {PRECOMPUTE_TABLE_NAME} om
                JOIN (SELECT grid_number, local_observation_time AS t_local
                      FROM {IMERG_TABLE_NAME} WHERE is_complete_hour) im
                  ON im.grid_number = om.grid_number AND im.t_local = om.local_forecast_time
                WHERE {ROW_FILTER_SQL}{SMOKE_FILTER_SQL}''')
            n_rows = cur.fetchone()[0]
        log(f"allocating {n_rows:,} x {n_feat} float32 "
            f"({n_rows * n_feat * 4 / 1e9:.2f} GB) + {MAX_LEAD} lead columns")

        x = np.empty((n_rows, n_feat), dtype="float32")
        future = np.empty((n_rows, MAX_LEAD), dtype="float32")
        t = np.empty(n_rows, dtype="datetime64[s]")
        unit = np.empty(n_rows, dtype="int32")
        filled = 0
        with conn.cursor(name="bkk_v2_deploy") as cur:
            cur.itersize = batch_size
            cur.execute(query)
            while filled < n_rows:
                batch = cur.fetchmany(batch_size)
                if not batch:
                    break
                block = np.asarray(batch, dtype="object")
                take = len(batch)
                unit[filled:filled + take] = block[:, 0].astype("int32")
                t[filled:filled + take] = block[:, 1].astype("datetime64[s]")
                x[filled:filled + take] = block[:, 2:2 + n_feat].astype("float32")
                future[filled:filled + take] = block[:, 2 + n_feat:].astype("float32")
                filled += take
                del block
        gc.collect()
    return x[:filled], future[:filled], t[:filled], unit[:filled]


started = time.time()
x_all, future_rain, forecast_time, unit_id = read_training_data()
log(f"loaded {x_all.shape[0]:,} rows in {(time.time() - started) / 60:.1f} min")
log(f"span {forecast_time.min()} .. {forecast_time.max()}   cells: {np.unique(unit_id).size}")

# %% [markdown]
# ## 4. Build the six targets
#
# `hN_Xmm` is 1 when IMERG recorded at least *X* mm in any of the next *N* hours. A row is kept
# only when all six future hours survived the gap guard, so every target is built on exactly the
# same rows and the six models are directly comparable.

# %%
complete = ~np.isnan(future_rain[:, :MAX_LEAD]).any(axis=1)
log(f"rows with a complete {MAX_LEAD}h future window: {complete.sum():,} ({complete.mean():.1%})")

x_all = np.ascontiguousarray(x_all[complete])
future_rain = future_rain[complete]
forecast_time = forecast_time[complete]
unit_id = unit_id[complete]
del complete
gc.collect()

TARGETS = [(h, thr) for thr in RAIN_THRESHOLDS_MM for h in HORIZONS]
TARGET_NAMES = [f"h{h}_{thr}mm" for h, thr in TARGETS]
y_all = np.empty((x_all.shape[0], len(TARGETS)), dtype="int8")
for i, (h, thr) in enumerate(TARGETS):
    y_all[:, i] = (future_rain[:, :h].max(axis=1) >= thr).astype("int8")

del future_rain
gc.collect()

balance = pd.DataFrame({
    "target": TARGET_NAMES,
    "horizon_h": [h for h, _ in TARGETS],
    "threshold_mm": [thr for _, thr in TARGETS],
    "base_rate": y_all.mean(axis=0),
    "positives": y_all.sum(axis=0),
})
print(balance.to_string(index=False))

# %%
stamps = pd.to_datetime(forecast_time)
year_key = stamps.year.to_numpy()
PURGE = np.timedelta64(PURGE_HOURS, "h")


def purged_train_mask(test_mask, times=forecast_time, purge=PURGE):
    """Training rows: not held out, and not within `purge` of any held-out timestamp."""
    test_times = np.unique(times[test_mask])
    position = np.searchsorted(test_times, times)
    left = test_times[np.clip(position - 1, 0, len(test_times) - 1)]
    right = test_times[np.clip(position, 0, len(test_times) - 1)]
    distance = np.minimum(np.abs(times - left), np.abs(times - right))
    return ~test_mask & (distance > purge)


YEAR_FOLDS = [int(y) for y in np.unique(year_key) if (year_key == y).sum() >= MIN_FOLD_ROWS]
log(f"year folds: {YEAR_FOLDS}")

# %% [markdown]
# ## 5. Stage 1 — cross-fitted out-of-fold probabilities
#
# One model per (fold, target), fitted on the purged training rows and predicting the held-out
# year. Unlike `BKK_Rain_V2` these fold models train on **all** their training rows — no 12 % is
# held back, because the calibrator is no longer fitted per fold. That makes each fold model as
# close as possible to the final model whose probabilities the calibrator will eventually map.

# %%
def build_model(model_name, y_fit):
    if model_name == "lightgbm":
        scale_pos_weight = float((y_fit == 0).sum() / max((y_fit == 1).sum(), 1))
        return LGBMClassifier(
            objective="binary", n_estimators=500, learning_rate=0.04, num_leaves=63,
            min_child_samples=80, subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE,
            n_jobs=-1, verbose=-1,
        )
    return HistGradientBoostingClassifier(
        learning_rate=0.06, max_iter=250, max_leaf_nodes=31,
        l2_regularization=0.05, random_state=RANDOM_STATE, early_stopping=False,
    )


oof_raw = np.full((x_all.shape[0], len(TARGETS)), np.nan, dtype="float32")
fit_log = []
cv_started = time.time()

for year in YEAR_FOLDS:
    test_mask = year_key == year
    train_mask = purged_train_mask(test_mask)
    x_train, x_test = x_all[train_mask], x_all[test_mask]
    log(f"fold {year}: test {test_mask.sum():,}  train {train_mask.sum():,}")
    for i, (horizon, threshold) in enumerate(TARGETS):
        model_name = MODEL_PLAN[horizon]
        started = time.time()
        y_train = y_all[train_mask, i]
        model = build_model(model_name, y_train)
        model.fit(x_train, y_train)
        oof_raw[test_mask, i] = model.predict_proba(x_test)[:, 1].astype("float32")
        elapsed = time.time() - started
        fit_log.append({"stage": "oof", "fold": str(year), "target": TARGET_NAMES[i],
                        "horizon_h": horizon, "threshold_mm": threshold, "model": model_name,
                        "train_rows": int(train_mask.sum()), "fit_seconds": elapsed})
        log(f"    {TARGET_NAMES[i]:12} {model_name:22} {elapsed / 60:.1f} min")
        del model, y_train
        gc.collect()
    del x_train, x_test
    gc.collect()

coverage = np.isfinite(oof_raw).all(axis=1).mean()
log(f"OOF stage finished in {(time.time() - cv_started) / 60:.1f} min; coverage {coverage:.4f}")

# %% [markdown]
# ## 6. Stage 2 — one isotonic calibrator per target, and the operating threshold
#
# The calibrator is fitted on every row's out-of-fold probability. The threshold sweep is exact
# rather than a 0.05 grid: sorting the probabilities once and taking cumulative true positives
# gives precision and recall at *every* possible cut, so the reported F1 is the true maximum, not
# the best of nineteen guesses. It also replaces ~1,400 full-array `f1_score` passes with one sort.

# %%
def sweep(probabilities, y_true):
    """Exact F1-optimal cut for the rule `probability >= threshold`.

    Sorting once and taking cumulative true positives gives precision and recall at every
    possible cut, so this returns the true maximum rather than the best point on a coarse grid.
    """
    order = np.argsort(probabilities, kind="stable")[::-1]
    p_sorted = probabilities[order]
    tp = np.cumsum(y_true[order].astype("int64"))
    predicted_positive = np.arange(1, len(p_sorted) + 1, dtype="int64")
    total_positive = int(tp[-1])
    if total_positive == 0:
        return 0.5, 0.0, 0.0, 0.0
    precision = tp / predicted_positive
    recall = tp / total_positive
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(precision), where=(precision + recall) > 0)
    # a cut is only realisable at the end of a run of tied probabilities
    realisable = np.empty(len(p_sorted), dtype=bool)
    realisable[:-1] = p_sorted[:-1] > p_sorted[1:]
    realisable[-1] = True
    best = int(np.where(realisable, f1, -1.0).argmax())
    result = (float(p_sorted[best]), float(precision[best]),
              float(recall[best]), float(f1[best]))
    del order, p_sorted, tp, predicted_positive, precision, recall, f1, realisable
    gc.collect()
    return result


calibrators, thresholds, oof_metrics = {}, {}, []
for i, name in enumerate(TARGET_NAMES):
    y_true = y_all[:, i]
    raw = oof_raw[:, i]
    valid = np.isfinite(raw)

    isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    isotonic.fit(raw[valid], y_true[valid])
    calibrated = isotonic.predict(raw).astype("float32")
    calibrators[name] = isotonic

    threshold, precision, recall, f1 = sweep(calibrated[valid], y_true[valid])
    thresholds[name] = threshold

    base = float(y_true[valid].mean())
    climatology = base * (1 - base)
    row = {
        "target": name, "horizon_h": TARGETS[i][0], "threshold_mm": TARGETS[i][1],
        "model": MODEL_PLAN[TARGETS[i][0]], "rows": int(valid.sum()), "base_rate": base,
        "roc_auc": float(roc_auc_score(y_true[valid], calibrated[valid])),
        "pr_auc": float(average_precision_score(y_true[valid], calibrated[valid])),
        "brier_raw": float(brier_score_loss(y_true[valid], raw[valid])),
        "brier_calibrated": float(brier_score_loss(y_true[valid], calibrated[valid])),
        "brier_climatology": climatology,
        "probability_threshold": threshold,
        "precision": precision, "recall": recall, "f1": f1,
        "isotonic_knots": int(len(isotonic.X_thresholds_)),
    }
    row["pr_auc_lift"] = row["pr_auc"] / base
    row["brier_skill"] = 1 - row["brier_calibrated"] / climatology
    row["brier_skill_raw"] = 1 - row["brier_raw"] / climatology
    oof_metrics.append(row)
    log(f"  {name:12} thr={threshold:.4f}  F1={f1:.4f}  ROC={row['roc_auc']:.4f}  "
        f"BSS {row['brier_skill_raw']:+.3f} -> {row['brier_skill']:+.3f}  "
        f"knots={row['isotonic_knots']:,}")
    del calibrated, valid, raw
    gc.collect()

oof_metrics = pd.DataFrame(oof_metrics)
print(oof_metrics[["target", "base_rate", "roc_auc", "pr_auc_lift", "probability_threshold",
                   "precision", "recall", "f1", "brier_skill"]].to_string(index=False))

del oof_raw
gc.collect()

# %% [markdown]
# ## 7. Stage 3 — the final fit, on every row
#
# Each target gets one model trained on all rows. This is the estimator that ships; the calibrator
# from stage 2 and the threshold from stage 2 travel with it in the same bundle.

# %%
final_models = {}
for i, name in enumerate(TARGET_NAMES):
    horizon, rain_threshold = TARGETS[i]
    model_name = MODEL_PLAN[horizon]
    started = time.time()
    y_full = y_all[:, i]
    model = build_model(model_name, y_full)
    model.fit(x_all, y_full)
    elapsed = time.time() - started
    final_models[name] = model
    fit_log.append({"stage": "final", "fold": "all", "target": name, "horizon_h": horizon,
                    "threshold_mm": rain_threshold, "model": model_name,
                    "train_rows": int(x_all.shape[0]), "fit_seconds": elapsed})
    log(f"  final {name:12} {model_name:22} {elapsed / 60:.1f} min on {x_all.shape[0]:,} rows")
    gc.collect()

# %% [markdown]
# ## 8. Save
#
# One `.joblib` per target holding the model, its calibrator, its threshold, the feature order and
# the metrics that describe it. The filename carries the horizon, the model family, the operating
# threshold and the rain threshold, following the `BKK_Rain_V1` convention.
#
# Loading and predicting:
#
# ```python
# bundle = joblib.load(path)
# x = frame[bundle["feature_columns"]].to_numpy("float32")     # order matters
# probability = bundle["calibrator"].predict(bundle["model"].predict_proba(x)[:, 1])
# will_rain = probability >= bundle["probability_threshold"]
# ```

# %%
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
run_stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
saved = []

for i, name in enumerate(TARGET_NAMES):
    horizon, rain_threshold = TARGETS[i]
    model_name = MODEL_PLAN[horizon]
    threshold = thresholds[name]
    metrics = oof_metrics[oof_metrics["target"] == name].iloc[0].to_dict()
    filename = (f"om_bkk_rain_v2_deploy_next_{horizon}h_{model_name}"
                f"_prob_threshold_{threshold:.4f}_rain_threshold_{rain_threshold}mm.joblib")
    bundle = {
        "target": name,
        "horizon_h": horizon,
        "rain_threshold_mm": rain_threshold,
        "model": final_models[name],
        "model_family": model_name,
        "calibrator": calibrators[name],
        "probability_threshold": threshold,
        "feature_columns": FEATURE_COLUMNS,
        "label_source": "IMERG satellite observation (is_complete_hour, gap-guarded leads)",
        "label_definition": f"IMERG >= {rain_threshold} mm in any of the next {horizon} h",
        "trained_on_rows": int(x_all.shape[0]),
        "trained_span": [str(forecast_time.min()), str(forecast_time.max())],
        "calibration": "isotonic, cross-fitted on leave-one-year-out OOF probabilities",
        "threshold_selection": "exact F1-optimal sweep over calibrated OOF probabilities",
        "metrics_are_out_of_fold": True,
        "metrics": metrics,
        "random_state": RANDOM_STATE,
        "trained_at_utc": run_stamp,
        "versions": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "sklearn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "caveats": [
            "IMERG measures area rain over an ~11 km cell; a gauge measures a point. This model "
            "predicts 'rain somewhere in the cell', not 'rain at a given address'.",
            "IMERG cannot carry intensity - for gauge-hours above 10 mm it reports at least half "
            "the gauge amount only 2.9% of the time. This can never be a heavy-rain warning.",
            "Features are ECMWF forecast output. V2 fixed the label, not the input, so the model "
            "still inherits ECMWF's bias - it has just stopped being graded on reproducing it.",
            "IMERG after 2025-10-01 is provisional Late Run, whose detection runs ~12% low. "
            "Rows from that period are in the training set and in the OOF metrics.",
        ],
    }
    joblib.dump(bundle, OUTPUT_DIR / filename, compress=3)
    size_mb = (OUTPUT_DIR / filename).stat().st_size / 1e6
    saved.append({"target": name, "file": filename, "size_mb": size_mb,
                  "probability_threshold": threshold})
    log(f"  saved {filename}  ({size_mb:.1f} MB)")

saved = pd.DataFrame(saved)
fit_log = pd.DataFrame(fit_log)
oof_metrics.to_csv(OUTPUT_DIR / "deploy_oof_metrics.csv", index=False)
fit_log.to_csv(OUTPUT_DIR / "deploy_fit_log.csv", index=False)
saved.to_csv(OUTPUT_DIR / "deploy_artifacts.csv", index=False)

# %% [markdown]
# ## 9. Does the shipped model match what `BKK_Rain_V2` reported?
#
# The cross-fitted numbers should land close to `BKK_Rain_V2`'s calibrated column. They will not be
# identical — the fold models here train on 12 % more rows, the calibrator is fitted on ~8x more
# points, and the threshold sweep is exact rather than gridded — but a large gap would mean
# something broke rather than something improved.

# %%
comparison = pd.DataFrame()
v2_results_path = V2_DIR / "v2_oof_results.csv"
if v2_results_path.exists():
    v2 = pd.read_csv(v2_results_path)
    v2 = v2[v2["probabilities"] == "calibrated"].set_index("target")
    rows = []
    for _, r in oof_metrics.iterrows():
        if r["target"] not in v2.index:
            continue
        prior = v2.loc[r["target"]]
        rows.append({
            "target": r["target"],
            "roc_v2": prior["roc_auc"], "roc_deploy": r["roc_auc"],
            "roc_delta": r["roc_auc"] - prior["roc_auc"],
            "f1_v2": prior["f1"], "f1_deploy": r["f1"], "f1_delta": r["f1"] - prior["f1"],
            "bss_v2": prior["brier_skill"], "bss_deploy": r["brier_skill"],
            "bss_delta": r["brier_skill"] - prior["brier_skill"],
        })
    comparison = pd.DataFrame(rows)
    print(comparison.to_string(index=False))
    comparison.to_csv(OUTPUT_DIR / "deploy_vs_v2_cv.csv", index=False)
else:
    log(f"no V2 CV results at {v2_results_path}; skipping comparison")

# %%
metadata = {
    "purpose": "Deployable BKK_Rain_V2: one final model per target fitted on all rows, with a "
               "cross-fitted isotonic calibrator and a baked-in operating threshold.",
    "built_by": "ML_Model_V2/build_bkk_rain_v2_deploy.py",
    "trained_at_utc": run_stamp,
    "smoke_mode": SMOKE,
    "smoke_grid_modulo": SMOKE_GRID_MODULO if SMOKE else None,
    "precompute_table": PRECOMPUTE_TABLE_NAME,
    "label_table": IMERG_TABLE_NAME,
    "rows": int(x_all.shape[0]),
    "span": [str(forecast_time.min()), str(forecast_time.max())],
    "grid_cells": int(np.unique(unit_id).size),
    "horizons": HORIZONS,
    "rain_thresholds_mm": RAIN_THRESHOLDS_MM,
    "model_plan": MODEL_PLAN,
    "feature_columns": FEATURE_COLUMNS,
    "random_state": RANDOM_STATE,
    "purge_hours": PURGE_HOURS,
    "year_folds": [str(y) for y in YEAR_FOLDS],
    "oof_coverage": float(coverage),
    "calibration": "isotonic, cross-fitted on leave-one-year-out OOF probabilities "
                   "(sklearn CalibratedClassifierCV(ensemble=False) scheme); the calibrator "
                   "never sees rows its model was trained on, and the final model keeps all rows",
    "threshold_selection": "exact F1-optimal sweep over calibrated OOF probabilities",
    "metrics": oof_metrics.to_dict(orient="records"),
    "artifacts": saved.to_dict(orient="records"),
    "comparison_to_v2_cv": comparison.to_dict(orient="records") if len(comparison) else [],
    "versions": {
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "sklearn": sklearn.__version__, "joblib": joblib.__version__,
    },
}
(OUTPUT_DIR / "deploy_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
log(f"wrote metadata and {len(saved)} model bundles to {OUTPUT_DIR}")

# %% [markdown]
# ## 10. Load-back check
#
# The artifact is only real if it survives a round trip. Each bundle is re-loaded from disk and
# asked to predict on a slice of rows it was trained on; the check is that the pipeline runs and
# returns probabilities in range, not that the predictions are good.

# %%
check_rows = min(5_000, x_all.shape[0])
checks = []
for record in saved.itertuples():
    bundle = joblib.load(OUTPUT_DIR / record.file)
    x_check = x_all[:check_rows]
    probability = bundle["calibrator"].predict(
        bundle["model"].predict_proba(x_check)[:, 1])
    flag = probability >= bundle["probability_threshold"]
    checks.append({
        "target": bundle["target"],
        "features_match": bundle["feature_columns"] == FEATURE_COLUMNS,
        "prob_min": float(probability.min()), "prob_max": float(probability.max()),
        "in_range": bool((probability >= 0).all() and (probability <= 1).all()),
        "predicted_rain_share": float(flag.mean()),
        "actual_rain_share": float(y_all[:check_rows, TARGET_NAMES.index(bundle["target"])].mean()),
    })
    del bundle
    gc.collect()

checks = pd.DataFrame(checks)
print(checks.to_string(index=False))
assert checks["features_match"].all(), "feature order did not round-trip"
assert checks["in_range"].all(), "calibrated probabilities left [0, 1]"
log("load-back check passed for all bundles")
