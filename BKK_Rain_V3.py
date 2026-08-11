# %% [markdown]
# # BKK_Rain_V3 — does observed-rain history help, and how much survives latency?
#
# `BKK_Rain_V2` established that IMERG labels beat Open-Meteo labels, but it kept V1's
# feature set: **66 Open-Meteo columns and not one observed-rain feature**. The model is
# asked "did the satellite see rain next hour" while being told only what ECMWF forecast.
# Rain is strongly autocorrelated, so the single most informative input for `rain_any_next`
# is "was it raining recently" — which V2 has only as an ECMWF proxy that correlates with
# the truth at 0.18.
#
# V3 keeps V2's protocol verbatim (leave-one-year-out CV, 24 h purge, isotonic calibration,
# IMERG labels, 1/3/6 h windows, 0.1 and 1.0 mm thresholds) and changes exactly one thing:
# it adds **observed IMERG precipitation history** (lags and rolling sums). It trains three
# feature sets on identical rows so the lever is measured cleanly:
#
# | config | features | question |
# |---|---|---|
# | `om_only` | 66 Open-Meteo (V2 set) | control — reproduces V2 |
# | `imerg_deploy` | 66 + observed history shifted 6 h older | the deployable gain (IMERG Early Run latency) |
# | `imerg_research` | 66 + observed history at 0 h lag | the information ceiling |
#
# Plus two things V2 lacks and a reviewer asks for first: a **persistence baseline**
# ("it rained recently → predict rain") and a **run_type breakout** (permanent vs
# provisional IMERG), both nearly free.
#
# The gap between `imerg_deploy` and `imerg_research` is the cost of IMERG's latency; the
# gap between `om_only` and the persistence baseline tells you whether the model beats the
# trivial nowcast at all.
#
# Run as three memory-bounded phases (only ~3.6 GB RAM free on this box):
#     python BKK_Rain_V3.py load_panel
#     python BKK_Rain_V3.py load_history
#     python BKK_Rain_V3.py cv

# %%
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

# %% [markdown]
# ## 1. Configuration — copied from V2, plus the IMERG-history switches

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
PROJECT_ROOT = Path(__file__).resolve().parent
V1_DIR = PROJECT_ROOT / "ML_Model_V2" / "trained_models" / "om_bkk_rain_any_v1"
V2_DIR = PROJECT_ROOT / "ML_Model_V2" / "trained_models" / "om_bkk_rain_v2_imerg"
OUTPUT_DIR = PROJECT_ROOT / "ML_Model_V2" / "trained_models" / "om_bkk_rain_v3_imerg_history"
CACHE = Path(os.getenv("V3_CACHE", str(PROJECT_ROOT / "ML_Model_V2" / "_v3_cache")))
CACHE.mkdir(parents=True, exist_ok=True)

HORIZONS = [1, 3, 6]
RAIN_THRESHOLDS_MM = [0.1, 1.0]
MAX_LEAD = max(HORIZONS)
MODEL_PLAN = {1: "lightgbm", 3: "hist_gradient_boosting", 6: "hist_gradient_boosting"}
PROBABILITY_THRESHOLDS_TO_TEST = np.arange(0.05, 0.96, 0.05)
RANDOM_STATE = 42
PURGE_HOURS = 24
MIN_FOLD_ROWS = 50_000
CALIBRATION_FRACTION = 0.12

# IMERG Final Run lags ~3.5 months; even the Early Run is ~4-6 h behind real time. So every
# observed-history feature must be shifted back by the latency before it is a fair predictor.
# The deployable config emulates the Early Run; the research config (0 h) is the upper bound.
IMERG_DEPLOY_LAG_HOURS = 6
MAX_HISTORY_LAG = 30  # deepest lag pulled: deployable 24 h rolling sum ends at t-6, spans to t-30

TARGETS = [(h, thr) for thr in RAIN_THRESHOLDS_MM for h in HORIZONS]
TARGET_NAMES = [f"h{h}_{thr}mm" for h, thr in TARGETS]

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
FEATURE_COLUMNS = BASELINE_FEATURE_COLUMNS + NEIGHBOR_FEATURE_COLUMNS  # 66, identical to V2

ROW_FILTER_SQL = """
      om.pressure_msl_change_6h IS NOT NULL
      AND om.precipitation_lag_6h IS NOT NULL
      AND om.precipitation_sum_past_24h IS NOT NULL
      AND om.cloud_cover_lag_6h IS NOT NULL
      AND om.humidity_lag_6h IS NOT NULL
      AND om.wind_speed_lag_3h IS NOT NULL
      AND om.neighbor_count > 0
"""


def connect():
    return psycopg2.connect(**DB_CONFIG)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# %% [markdown]
# ## 2. Derived IMERG-history feature names
#
# From a matrix of guarded precip lags `L` (column k-1 = observed mm at t-k, NaN across a
# gap) both blocks are built the same way; the deployable block just uses lags offset by the
# latency. Every feature the trees see is therefore strictly in the past relative to when the
# forecast would be issued.

# %%
def history_feature_names():
    return [
        "ih_precip_lag1", "ih_precip_lag2", "ih_precip_lag3", "ih_precip_lag6",
        "ih_precip_sum3", "ih_precip_sum6", "ih_precip_sum12", "ih_precip_sum24",
        "ih_rain_lag1", "ih_rain_count3", "ih_rain_count6", "ih_hours_since_rain6",
    ]


def build_history_block(lags, offset):
    """lags: (N, MAX_HISTORY_LAG) guarded precip; offset: latency hours (0 or deploy lag).

    Column c of `lags` is precip at t-(c+1). With an offset, the freshest hour the model may
    see is t-(offset+1), so every relative lag shifts by `offset`.
    """
    def col(k):  # observed precip at t-k, k>=1
        return lags[:, k - 1]

    o = offset
    lag1, lag2, lag3, lag6 = col(o + 1), col(o + 2), col(o + 3), col(o + 6)
    # rolling sums over the window ending at the freshest available hour
    sum3 = lags[:, o:o + 3].sum(axis=1)
    sum6 = lags[:, o:o + 6].sum(axis=1)
    sum12 = lags[:, o:o + 12].sum(axis=1)
    sum24 = lags[:, o:o + 24].sum(axis=1)
    rainy = (lags >= 0.1)
    rain_lag1 = rainy[:, o].astype("float32")
    rain_count3 = rainy[:, o:o + 3].sum(axis=1).astype("float32")
    rain_count6 = rainy[:, o:o + 6].sum(axis=1).astype("float32")
    # hours since last observed rain, searched over the freshest 6 available hours (capped at 6)
    window = rainy[:, o:o + 6]
    first_rain = np.where(window.any(axis=1), window.argmax(axis=1).astype("float32"), 6.0)
    block = np.column_stack([
        lag1, lag2, lag3, lag6, sum3, sum6, sum12, sum24,
        rain_lag1, rain_count3, rain_count6, first_rain,
    ]).astype("float32")
    return block


# %% [markdown]
# ## 3. Phase `load_panel` — Open-Meteo features + IMERG-derived targets (V2 loader verbatim)

# %%
def build_panel_query():
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
    return f"""
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
    WHERE {ROW_FILTER_SQL}
    ORDER BY om.local_forecast_time, om.grid_number
    """


def phase_load_panel(batch_size=200_000):
    query = build_panel_query()
    n_feat = len(FEATURE_COLUMNS)
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT count(*) FROM {PRECOMPUTE_TABLE_NAME} om
                JOIN (SELECT grid_number, local_observation_time AS t_local
                      FROM {IMERG_TABLE_NAME} WHERE is_complete_hour) im
                  ON im.grid_number = om.grid_number AND im.t_local = om.local_forecast_time
                WHERE {ROW_FILTER_SQL}""")
            n_rows = cur.fetchone()[0]
        log(f"panel rows {n_rows:,} x {n_feat} feat "
            f"({n_rows * n_feat * 4 / 1e9:.2f} GB) + {MAX_LEAD} future cols")

        x = np.empty((n_rows, n_feat), dtype="float32")
        future = np.empty((n_rows, MAX_LEAD), dtype="float32")
        t = np.empty(n_rows, dtype="datetime64[s]")
        unit = np.empty(n_rows, dtype="int32")
        filled = 0
        with conn.cursor(name="bkk_v3_panel") as cur:
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
                print(f"  {filled:,} / {n_rows:,}", end="\r", flush=True)
        print()
    x, future, t, unit = x[:filled], future[:filled], t[:filled], unit[:filled]

    # drop rows without a complete MAX_LEAD future window, then build the 6 targets
    complete = ~np.isnan(future).any(axis=1)
    x = np.ascontiguousarray(x[complete])
    future = future[complete]
    t = t[complete]
    unit = unit[complete]
    log(f"complete future window: {complete.sum():,} ({complete.mean():.1%})")

    y = np.empty((x.shape[0], len(TARGETS)), dtype="int8")
    for i, (h, thr) in enumerate(TARGETS):
        y[:, i] = (future[:, :h].max(axis=1) >= thr).astype("int8")

    np.save(CACHE / "base_X.npy", x)
    np.save(CACHE / "y.npy", y)
    np.save(CACHE / "forecast_time.npy", t)
    np.save(CACHE / "unit_id.npy", unit)
    log(f"saved base_X {x.shape}, y {y.shape}")
    balance = pd.DataFrame({
        "target": TARGET_NAMES, "horizon_h": [h for h, _ in TARGETS],
        "threshold_mm": [thr for _, thr in TARGETS],
        "base_rate": y.mean(axis=0), "positives": y.sum(axis=0),
    })
    print(balance.to_string(index=False))


# %% [markdown]
# ## 4. Phase `load_history` — observed IMERG precip lags 1..30, aligned to the panel
#
# Same gap guard as the labels: `LAG(precip, k)` only counts if its timestamp is exactly
# k hours before the anchor, so a lag never reaches across a missing hour. run_type is read
# from the anchor row for the provisional-vs-permanent breakout.

# %%
def build_history_query():
    lag_sql = ",\n            ".join(
        f"LAG(precipitation_mm, {k}) OVER w AS p{k}, "
        f"LAG(local_observation_time, {k}) OVER w AS q{k}"
        for k in range(1, MAX_HISTORY_LAG + 1))
    guarded_sql = ",\n        ".join(
        f"CASE WHEN h.q{k} = h.t_local - interval '{k} hour' "
        f"THEN h.p{k}::real ELSE 'NaN'::real END AS l{k}"
        for k in range(1, MAX_HISTORY_LAG + 1))
    return f"""
    WITH h AS (
        SELECT grid_number, local_observation_time AS t_local, run_type,
            {lag_sql}
        FROM {IMERG_TABLE_NAME}
        WHERE is_complete_hour
        WINDOW w AS (PARTITION BY grid_number ORDER BY local_observation_time)
    )
    SELECT h.grid_number, h.t_local, h.run_type,
        {guarded_sql}
    FROM h
    ORDER BY h.t_local, h.grid_number
    """


def phase_load_history(batch_size=200_000):
    unit = np.load(CACHE / "unit_id.npy")
    tf = np.load(CACHE / "forecast_time.npy")
    panel = pd.DataFrame({"grid_number": unit, "t_local": tf})
    panel["row_order"] = np.arange(len(panel))
    log(f"panel keys: {len(panel):,}")

    query = build_history_query()
    n_lag = MAX_HISTORY_LAG
    # stream history rows, keep only those matching a panel key via a hash set of (grid,time)
    key_index = {}
    for go, gr, ro in zip(panel["grid_number"].to_numpy(),
                          panel["t_local"].to_numpy(), panel["row_order"].to_numpy()):
        key_index[(int(go), np.datetime64(gr))] = int(ro)

    lags = np.full((len(panel), n_lag), np.nan, dtype="float32")
    run_type = np.array([None] * len(panel), dtype=object)
    matched = 0
    with connect() as conn:
        with conn.cursor(name="bkk_v3_history") as cur:
            cur.itersize = batch_size
            cur.execute(query)
            seen = 0
            while True:
                batch = cur.fetchmany(batch_size)
                if not batch:
                    break
                block = np.asarray(batch, dtype="object")
                grids = block[:, 0].astype("int64")
                times = block[:, 1].astype("datetime64[s]")
                rts = block[:, 2]
                vals = block[:, 3:3 + n_lag].astype("float32")
                for j in range(len(block)):
                    ro = key_index.get((int(grids[j]), np.datetime64(times[j])))
                    if ro is not None:
                        lags[ro] = vals[j]
                        run_type[ro] = rts[j]
                        matched += 1
                seen += len(block)
                print(f"  scanned {seen:,}  matched {matched:,}", end="\r", flush=True)
        print()
    log(f"matched {matched:,} / {len(panel):,} panel rows to IMERG history")

    research = build_history_block(lags, offset=0)
    deploy = build_history_block(lags, offset=IMERG_DEPLOY_LAG_HOURS)
    # persistence anchors: observed precip at the freshest available hour for each config
    persist_research = lags[:, 0].astype("float32")                       # t-1
    persist_deploy = lags[:, IMERG_DEPLOY_LAG_HOURS].astype("float32")    # t-(lag+1)

    np.save(CACHE / "imerg_research.npy", research)
    np.save(CACHE / "imerg_deploy.npy", deploy)
    np.save(CACHE / "persist_research.npy", persist_research)
    np.save(CACHE / "persist_deploy.npy", persist_deploy)
    np.save(CACHE / "run_type.npy", run_type.astype("U16"))
    log(f"saved history blocks research {research.shape} deploy {deploy.shape}")
    print("history NaN share (research):",
          pd.Series(np.isnan(research).mean(axis=0), index=history_feature_names()).round(4).to_string())


# %% [markdown]
# ## 5. Phase `cv` — leave-one-year-out CV for each config, then score
#
# `fit_predict_calibrated`, the purge and the honest leave-one-year-out threshold are all
# lifted from V2 so the three configs differ only in their columns.

# %%
def fit_predict_calibrated(x_train, y_train, x_test, model_name, rng):
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.isotonic import IsotonicRegression

    n = len(y_train)
    calib_size = int(n * CALIBRATION_FRACTION)
    shuffled = rng.permutation(n)
    calib_index, fit_index = shuffled[:calib_size], shuffled[calib_size:]

    if model_name == "lightgbm":
        scale_pos_weight = float((y_train[fit_index] == 0).sum()
                                 / max((y_train[fit_index] == 1).sum(), 1))
        model = LGBMClassifier(
            objective="binary", n_estimators=500, learning_rate=0.04, num_leaves=63,
            min_child_samples=80, subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE,
            n_jobs=-1, verbose=-1,
        )
    else:
        model = HistGradientBoostingClassifier(
            learning_rate=0.06, max_iter=250, max_leaf_nodes=31,
            l2_regularization=0.05, random_state=RANDOM_STATE, early_stopping=False,
        )
    model.fit(x_train[fit_index], y_train[fit_index])
    raw_calib = model.predict_proba(x_train[calib_index])[:, 1]
    isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    isotonic.fit(raw_calib, y_train[calib_index])
    raw_test = model.predict_proba(x_test)[:, 1].astype("float32")
    cal_test = isotonic.predict(raw_test).astype("float32")
    del model, isotonic
    gc.collect()
    return raw_test, cal_test


def assemble(config):
    base = np.load(CACHE / "base_X.npy")
    if config == "om_only":
        return base, FEATURE_COLUMNS
    block_file = "imerg_research.npy" if config == "imerg_research" else "imerg_deploy.npy"
    block = np.load(CACHE / block_file)
    x = np.ascontiguousarray(np.concatenate([base, block], axis=1))
    del base, block
    gc.collect()
    return x, FEATURE_COLUMNS + history_feature_names()


def purged_train_mask(test_mask, times, purge):
    test_times = np.unique(times[test_mask])
    position = np.searchsorted(test_times, times)
    left = test_times[np.clip(position - 1, 0, len(test_times) - 1)]
    right = test_times[np.clip(position, 0, len(test_times) - 1)]
    distance = np.minimum(np.abs(times - left), np.abs(times - right))
    return ~test_mask & (distance > purge)


def pick_threshold(probabilities, y_true):
    from sklearn.metrics import f1_score
    best_f1, best_threshold = -1.0, 0.5
    for candidate in PROBABILITY_THRESHOLDS_TO_TEST:
        score = f1_score(y_true, (probabilities >= candidate).astype("int8"), zero_division=0)
        if score > best_f1:
            best_f1, best_threshold = score, float(candidate)
    return best_threshold


def score_probabilities(probabilities, y, year_key, year_folds, label, config, extra=None):
    from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                                 precision_score, recall_score, roc_auc_score)
    rows = []
    for index in range(len(TARGETS)):
        y_true = y[:, index]
        p = probabilities[:, index]
        valid = np.isfinite(p)
        predictions = np.zeros(len(y_true), dtype="int8")
        chosen = {}
        for year in year_folds:
            this_year = (year_key == year) & valid
            others = valid & ~(year_key == year)
            thr = pick_threshold(p[others], y_true[others])
            chosen[year] = thr
            predictions[this_year] = (p[this_year] >= thr).astype("int8")
        base = float(y_true[valid].mean())
        pr = float(average_precision_score(y_true[valid], p[valid]))
        # Brier needs probabilities in [0, 1]; the persistence baseline scores with raw
        # precip amounts (fine for the rank-based ROC/PR), so skip Brier when out of range.
        clim = base * (1 - base)
        pmin, pmax = float(np.nanmin(p[valid])), float(np.nanmax(p[valid]))
        if 0.0 <= pmin and pmax <= 1.0:
            brier = float(brier_score_loss(y_true[valid], p[valid]))
            brier_skill = 1 - brier / clim if clim else float("nan")
        else:
            brier, brier_skill = float("nan"), float("nan")
        row = {
            "config": config, "target": TARGET_NAMES[index], "horizon_h": TARGETS[index][0],
            "threshold_mm": TARGETS[index][1], "probabilities": label,
            "rows": int(valid.sum()), "base_rate": base,
            "roc_auc": float(roc_auc_score(y_true[valid], p[valid])),
            "pr_auc": pr, "pr_auc_lift": pr / base,
            "precision": float(precision_score(y_true[valid], predictions[valid], zero_division=0)),
            "recall": float(recall_score(y_true[valid], predictions[valid], zero_division=0)),
            "f1": float(f1_score(y_true[valid], predictions[valid], zero_division=0)),
            "brier_score": brier, "brier_skill": brier_skill,
        }
        if extra:
            row.update(extra)
        rows.append(row)
    return rows


def run_config_cv(config, y, forecast_time, year_key, year_folds):
    x, columns = assemble(config)
    log(f"config {config}: X {x.shape}")
    purge = np.timedelta64(PURGE_HOURS, "h")
    rng = np.random.default_rng(RANDOM_STATE)
    oof_raw = np.full((x.shape[0], len(TARGETS)), np.nan, dtype="float32")
    oof_cal = np.full((x.shape[0], len(TARGETS)), np.nan, dtype="float32")
    fit_log = []
    for year in year_folds:
        test_mask = year_key == year
        train_mask = purged_train_mask(test_mask, forecast_time, purge)
        x_train, x_test = x[train_mask], x[test_mask]
        log(f"  fold {year}: test {test_mask.sum():,} train {train_mask.sum():,}")
        for i, (horizon, threshold) in enumerate(TARGETS):
            model_name = MODEL_PLAN[horizon]
            t0 = time.time()
            raw, cal = fit_predict_calibrated(x_train, y[train_mask, i], x_test, model_name, rng)
            oof_raw[test_mask, i] = raw
            oof_cal[test_mask, i] = cal
            fit_log.append({"config": config, "fold": str(year), "target": TARGET_NAMES[i],
                            "model": model_name, "train_rows": int(train_mask.sum()),
                            "fit_seconds": time.time() - t0})
            log(f"    {TARGET_NAMES[i]:12} {model_name:22} {(time.time()-t0)/60:.1f} min")
        del x_train, x_test
        gc.collect()
    del x
    gc.collect()
    np.save(CACHE / f"oof_cal_{config}.npy", oof_cal)
    np.save(CACHE / f"oof_raw_{config}.npy", oof_raw)
    results = score_probabilities(oof_cal, y, year_key, year_folds, "calibrated", config)
    results += score_probabilities(oof_raw, y, year_key, year_folds, "raw", config)
    return results, fit_log


def phase_cv():
    from sklearn.metrics import average_precision_score, roc_auc_score
    y = np.load(CACHE / "y.npy")
    forecast_time = np.load(CACHE / "forecast_time.npy")
    run_type = np.load(CACHE / "run_type.npy")
    stamps = pd.to_datetime(forecast_time)
    year_key = stamps.year.to_numpy()
    year_folds = [int(v) for v in np.unique(year_key) if (year_key == v).sum() >= MIN_FOLD_ROWS]
    log(f"folds: {year_folds}   rows {len(y):,}")

    all_results, all_fitlog = [], []
    for config in ["om_only", "imerg_deploy", "imerg_research"]:
        results, fit_log = run_config_cv(config, y, forecast_time, year_key, year_folds)
        all_results += results
        all_fitlog += fit_log
        pd.DataFrame(all_results).to_csv(OUTPUT_DIR / "v3_oof_results.csv", index=False)  # checkpoint
        pd.DataFrame(all_fitlog).to_csv(OUTPUT_DIR / "v3_fit_log.csv", index=False)
    results_df = pd.DataFrame(all_results)

    # --- persistence baselines (no training): observed rain at the freshest available hour
    persist_rows = []
    for tag, anchor_file in [("persist_research", "persist_research.npy"),
                             ("persist_deploy", "persist_deploy.npy")]:
        anchor = np.load(CACHE / anchor_file)
        probs = np.repeat(anchor[:, None], len(TARGETS), axis=1)  # amount as score for ROC/PR
        persist_rows += score_probabilities(probs, y, year_key, year_folds, "calibrated", tag)
    persist_df = pd.DataFrame(persist_rows)

    # --- run_type breakout for the deployable config (the one that would ship)
    oof_deploy = np.load(CACHE / "oof_cal_imerg_deploy.npy")
    runtype_rows = []
    for rt in ["permanent", "provisional"]:
        mask = run_type == rt
        if mask.sum() < MIN_FOLD_ROWS:
            continue
        sub = score_probabilities(oof_deploy[mask], y[mask], year_key[mask], year_folds,
                                  "calibrated", "imerg_deploy", extra={"run_type": rt})
        runtype_rows += sub
    runtype_df = pd.DataFrame(runtype_rows)

    # --- V3 (deployable, calibrated) vs V2 (calibrated) on identical rows
    v2_results = pd.read_csv(V2_DIR / "v2_oof_results.csv")
    v2_cal = v2_results[v2_results["probabilities"] == "calibrated"].set_index("target")
    v3_dep = results_df[(results_df["config"] == "imerg_deploy")
                        & (results_df["probabilities"] == "calibrated")].set_index("target")
    v3_res = results_df[(results_df["config"] == "imerg_research")
                        & (results_df["probabilities"] == "calibrated")].set_index("target")
    v3_om = results_df[(results_df["config"] == "om_only")
                       & (results_df["probabilities"] == "calibrated")].set_index("target")
    comp = pd.DataFrame({
        "base_rate": v3_om["base_rate"],
        "roc_v2": v2_cal["roc_auc"], "roc_om_only": v3_om["roc_auc"],
        "roc_deploy": v3_dep["roc_auc"], "roc_research": v3_res["roc_auc"],
        "f1_v2": v2_cal["f1"], "f1_om_only": v3_om["f1"],
        "f1_deploy": v3_dep["f1"], "f1_research": v3_res["f1"],
    })
    comp["roc_gain_deploy"] = comp["roc_deploy"] - comp["roc_om_only"]
    comp["roc_gain_research"] = comp["roc_research"] - comp["roc_om_only"]
    comp["f1_gain_deploy"] = comp["f1_deploy"] - comp["f1_om_only"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_DIR / "v3_oof_results.csv", index=False)
    persist_df.to_csv(OUTPUT_DIR / "v3_persistence.csv", index=False)
    runtype_df.to_csv(OUTPUT_DIR / "v3_runtype_breakout.csv", index=False)
    comp.to_csv(OUTPUT_DIR / "v3_vs_v2_comparison.csv")
    pd.DataFrame(all_fitlog).to_csv(OUTPUT_DIR / "v3_fit_log.csv", index=False)

    metadata = {
        "purpose": "V2 protocol + observed IMERG precipitation history; measures the gain from "
                   "observed-rain features and how much survives IMERG Early Run latency",
        "configs": {
            "om_only": "66 Open-Meteo features (V2 set); control",
            "imerg_deploy": f"66 + observed IMERG history shifted {IMERG_DEPLOY_LAG_HOURS} h "
                            f"(emulates IMERG Early Run latency); the deployable model",
            "imerg_research": "66 + observed IMERG history at 0 h lag; information ceiling",
        },
        "history_features": history_feature_names(),
        "imerg_deploy_lag_hours": IMERG_DEPLOY_LAG_HOURS,
        "protocol": "leave-one-year-out CV, 24 h purge, isotonic calibration, IMERG labels, "
                    "F1-optimal leave-one-year-out thresholds (copied from V2)",
        "horizons": HORIZONS, "rain_thresholds_mm": RAIN_THRESHOLDS_MM,
        "rows": int(len(y)), "year_folds": [str(v) for v in year_folds],
        "results": results_df.to_dict(orient="records"),
        "persistence": persist_df.to_dict(orient="records"),
        "runtype_breakout": runtype_df.to_dict(orient="records"),
        "v3_vs_v2": comp.reset_index().to_dict(orient="records"),
        "caveats": [
            "imerg_research features are not available in real time (IMERG Final Run lags "
            "~3.5 months); only imerg_deploy is operationally honest.",
            "The gap between imerg_research and imerg_deploy is the cost of latency, not a "
            "model improvement.",
            "Everything after 2025-10-01 is provisional Late Run IMERG; the run_type breakout "
            "isolates its effect.",
        ],
    }
    (OUTPUT_DIR / "v3_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    log("=== V3 vs V2 (calibrated, identical protocol) ===")
    print(comp.round(4).to_string())
    log("=== persistence baselines (calibrated F1) ===")
    print(persist_df[persist_df.threshold_mm == 0.1][
        ["config", "horizon_h", "roc_auc", "f1"]].round(4).to_string(index=False))
    log(f"saved V3 outputs to {OUTPUT_DIR}")


# %%
if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    phase = sys.argv[1] if len(sys.argv) > 1 else "cv"
    log(f"phase: {phase}")
    t0 = time.time()
    if phase == "load_panel":
        phase_load_panel()
    elif phase == "load_history":
        phase_load_history()
    elif phase == "cv":
        phase_cv()
    else:
        raise SystemExit(f"unknown phase {phase!r}; use load_panel | load_history | cv")
    log(f"phase {phase} done in {(time.time()-t0)/60:.1f} min")
