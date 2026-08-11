# %% [markdown]
# # Thailand_Rain_V8 — the reanalysis→forecast gap, measured
#
# Every model in this project was trained and scored on Open-Meteo *archive* features, which
# are ERA5(-Land) reanalysis at ~5-day latency — unusable at deployment time. A deployed
# system would call the forecast API instead. This experiment measures what that swap costs,
# on one month (June 2026) where three things exist at once:
#
#   - ERA5 features        ("OM_THAILAND_DATA_PRECOMPUTE", the training diet)
#   - forecast features    ("OM_THAILAND_FORECAST_PRECOMPUTE", archived ecmwf_ifs runs from
#                           the historical-forecast API — verified bit-identical to what the
#                           live API served in May 2026, r=1.000 on 6,384 sampled hours)
#   - IMERG labels         (all 833 grids, provisional Late Run — same grader for both)
#
# The IR features (hw_block_offset0) are satellite observations and identical in both
# configs; only the 66 OM columns change. Models are trained exactly as in the V3/V7 CV
# (same params, purge, calibration) on all panel rows outside June 2026, then each fitted
# model predicts BOTH test matrices, so the gap is measured model-for-model on identical
# rows and the only moving part is ERA5-vs-forecast input.
#
# Phases:
#     python Thailand_Rain_V8_forecast_gap.py compare   # feature distributions + rain agreement
#     python Thailand_Rain_V8_forecast_gap.py score     # the gate: ROC/PR-AUC gap per target
#
# (`compare` needs only the two precompute tables; `score` also needs the V3/V7 caches.)

# %%
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import psycopg2

from BKK_Rain_V3 import (
    FEATURE_COLUMNS, MODEL_PLAN, PROJECT_ROOT, PURGE_HOURS, RANDOM_STATE,
    TARGETS, TARGET_NAMES, fit_predict_calibrated, log, purged_train_mask,
)
from Thailand_Rain_V3 import CACHE, DB_CONFIG, IMERG_TABLE_NAME
from Thailand_Rain_V7_himawari_ir import himawari_feature_names

ERA5_TABLE = '"OM_THAILAND_DATA_PRECOMPUTE"'
FCST_TABLE = '"OM_THAILAND_FORECAST_PRECOMPUTE"'
OUTPUT_DIR = PROJECT_ROOT / "ML_Model_V2" / "trained_models" / "om_thailand_rain_v8_forecast_gap"

TEST_START = "2026-06-01 00:00:00"
TEST_END = "2026-06-30 23:00:00"   # inclusive, local
RAIN_MM = 0.1


def connect():
    return psycopg2.connect(**DB_CONFIG)


# %% [markdown]
# ## Phase `compare` — are the inputs even the same weather?
#
# Joins the two precompute tables on identical (grid, hour) keys for June and reports, per
# feature column: bias, MAE, and Pearson r. Then the part that decides deployability for a
# rain model: how well each precipitation column agrees with the other, and with observed
# IMERG rain on the same rows. BKK's ERA5-vs-forecast precipitation correlated at only 0.18
# with 5 h of timing offset — if that repeats nationwide, the swap will hurt.

# %%
def phase_compare():
    num_cols = [c for c in FEATURE_COLUMNS
                if c not in ("grid_row", "grid_column", "latitude", "longitude",
                             "hour_sin", "hour_cos", "month_sin", "month_cos")]
    col_sql = ",\n        ".join(
        f"e.{c} AS era5_{c}, f.{c} AS fcst_{c}" for c in num_cols)
    query = f"""
    SELECT e.grid_number, e.local_forecast_time,
        {col_sql},
        im.precipitation_max_mm AS imerg_cell_max
    FROM {ERA5_TABLE} e
    JOIN {FCST_TABLE} f
      ON f.grid_number = e.grid_number
     AND f.local_forecast_time = e.local_forecast_time
    LEFT JOIN {IMERG_TABLE_NAME} im
      ON im.grid_number = e.grid_number
     AND im.local_observation_time = e.local_forecast_time
     AND im.is_complete_hour
    WHERE e.local_forecast_time BETWEEN TIMESTAMP '{TEST_START}' AND TIMESTAMP '{TEST_END}'
    """
    conn = connect()
    try:
        df = pd.read_sql(query, conn)
    finally:
        conn.close()
    log(f"joined June rows: {len(df):,} "
        f"({df['grid_number'].nunique()} grids, IMERG present {df['imerg_cell_max'].notna().mean():.1%})")

    rows = []
    for c in num_cols:
        a, b = df[f"era5_{c}"], df[f"fcst_{c}"]
        m = a.notna() & b.notna()
        a, b = a[m], b[m]
        rows.append({"feature": c, "rows": int(m.sum()),
                     "era5_mean": a.mean(), "fcst_mean": b.mean(),
                     "bias_fcst_minus_era5": (b - a).mean(),
                     "mae": (b - a).abs().mean(),
                     "pearson_r": a.corr(b)})
    out = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_DIR / "v8_feature_agreement.csv", index=False)
    log("=== ERA5 vs forecast, identical June rows (worst-agreeing 15 features) ===")
    print(out.sort_values("pearson_r").head(15).round(3).to_string(index=False))

    # Rain-hour agreement, three ways
    m = df["era5_precipitation"].notna() & df["fcst_precipitation"].notna()
    d = df[m]
    era5_wet = d["era5_precipitation"] >= RAIN_MM
    fcst_wet = d["fcst_precipitation"] >= RAIN_MM
    agree = {
        "rows": int(len(d)),
        "era5_wet_rate": float(era5_wet.mean()),
        "fcst_wet_rate": float(fcst_wet.mean()),
        "precip_r_era5_vs_fcst": float(d["era5_precipitation"].corr(d["fcst_precipitation"])),
        "wet_hour_jaccard": float((era5_wet & fcst_wet).sum() / max((era5_wet | fcst_wet).sum(), 1)),
    }
    di = d[d["imerg_cell_max"].notna()]
    imerg_wet = di["imerg_cell_max"] >= RAIN_MM
    agree.update({
        "imerg_rows": int(len(di)),
        "imerg_wet_rate": float(imerg_wet.mean()),
        "precip_r_era5_vs_imerg": float(di["era5_precipitation"].corr(di["imerg_cell_max"])),
        "precip_r_fcst_vs_imerg": float(di["fcst_precipitation"].corr(di["imerg_cell_max"])),
        "wet_jaccard_era5_imerg": float(((di["era5_precipitation"] >= RAIN_MM) & imerg_wet).sum()
                                        / max(((di["era5_precipitation"] >= RAIN_MM) | imerg_wet).sum(), 1)),
        "wet_jaccard_fcst_imerg": float(((di["fcst_precipitation"] >= RAIN_MM) & imerg_wet).sum()
                                        / max(((di["fcst_precipitation"] >= RAIN_MM) | imerg_wet).sum(), 1)),
    })
    json.dump(agree, open(OUTPUT_DIR / "v8_rain_agreement.json", "w"), indent=2)
    log("=== rain-hour agreement (June, all hourly rows) ===")
    for k, v in agree.items():
        print(f"  {k:28s} {v:.4f}" if isinstance(v, float) else f"  {k:28s} {v:,}")


# %% [markdown]
# ## Phase `score` — the gate
#
# Test rows are the cached panel's June 2026 rows (stride hours, IMERG-labeled). For each
# config x target: fit on all purged non-June rows using ERA5 features exactly as the CV did,
# then predict a stacked test matrix [ERA5 test rows; forecast test rows] so raw and
# calibrated probabilities for both variants come from the *same* fitted model and isotonic.
#
# Reported per target: ROC-AUC and PR-AUC for era5-fed vs forecast-fed, plus the delta.
# A partial-results CSV is rewritten after every fit, V3-checkpoint style.

# %%
def load_forecast_test_matrix(unit, tf, test_index):
    """FEATURE_COLUMNS from the forecast precompute, aligned to the cached panel rows."""
    feature_sql = ",\n        ".join(
        f"COALESCE({c}::real, 'NaN'::real) AS {c}" for c in FEATURE_COLUMNS)
    query = f"""
    SELECT grid_number, local_forecast_time,
        {feature_sql}
    FROM {FCST_TABLE}
    WHERE local_forecast_time BETWEEN TIMESTAMP '{TEST_START}' AND TIMESTAMP '{TEST_END}'
    """
    conn = connect()
    try:
        cur = conn.cursor()
        cur.execute(query)
        rows = cur.fetchall()
    finally:
        conn.close()
    lookup = {(int(g), np.datetime64(t, "s")): np.asarray(v, dtype="float32")
              for g, t, *v in rows}
    x = np.full((len(test_index), len(FEATURE_COLUMNS)), np.nan, dtype="float32")
    missing = 0
    for j, row in enumerate(test_index):
        v = lookup.get((int(unit[row]), np.datetime64(tf[row], "s")))
        if v is None:
            missing += 1
        else:
            x[j] = v
    log(f"forecast test matrix {x.shape}; panel rows missing from forecast table: {missing}")
    if missing:
        raise SystemExit("forecast precompute does not cover every June panel row — "
                         "check the fetch warm-up window")
    return x


def phase_score():
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = np.load(CACHE / "y.npy")
    tf = np.load(CACHE / "forecast_time.npy")
    unit = np.load(CACHE / "unit_id.npy")
    base_x = np.load(CACHE / "base_X.npy")
    hw = np.load(CACHE / "hw_block_offset0.npy")

    test_mask = (tf >= np.datetime64("2026-06-01")) & (tf < np.datetime64("2026-07-01"))
    test_index = np.flatnonzero(test_mask)
    train_mask = purged_train_mask(test_mask, tf, np.timedelta64(PURGE_HOURS, "h"))
    log(f"June test rows {test_mask.sum():,}   train rows {train_mask.sum():,} "
        f"(purge {PURGE_HOURS} h)")

    fcst_x_test = load_forecast_test_matrix(unit, tf, test_index)
    era5_x_test = base_x[test_mask]

    # sanity: static columns must agree exactly between the two sources
    static_ix = [FEATURE_COLUMNS.index(c) for c in ("grid_row", "grid_column", "latitude", "longitude")]
    if not np.allclose(era5_x_test[:, static_ix], fcst_x_test[:, static_ix], equal_nan=True):
        raise SystemExit("static geometry columns disagree between ERA5 and forecast matrices")

    configs = {
        "om_only": (base_x, None),
        "hw_offset0h": (np.ascontiguousarray(np.concatenate([base_x, hw], axis=1)), hw),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / "v8_gap_results.csv"
    results = pd.read_csv(results_path).to_dict("records") if results_path.exists() else []
    done = {(r["config"], r["target"]) for r in results}

    n_test = len(test_index)
    for ci, (config, (x_full, hw_block)) in enumerate(configs.items()):
        x_train_all = x_full[train_mask]
        if hw_block is None:
            stacked_test = np.vstack([era5_x_test, fcst_x_test])
        else:
            hw_test = hw_block[test_mask]
            stacked_test = np.vstack([
                np.concatenate([era5_x_test, hw_test], axis=1),
                np.concatenate([fcst_x_test, hw_test], axis=1),
            ])
        for i, (horizon, threshold) in enumerate(TARGETS):
            if (config, TARGET_NAMES[i]) in done:
                log(f"  {config}/{TARGET_NAMES[i]}: already scored, skipping")
                continue
            t0 = time.time()
            rng = np.random.default_rng([RANDOM_STATE, 80, ci, i])
            raw, cal = fit_predict_calibrated(
                x_train_all, y[train_mask, i], stacked_test, MODEL_PLAN[horizon], rng)
            y_test = y[test_mask, i]
            row = {"config": config, "target": TARGET_NAMES[i],
                   "test_rows": n_test, "base_rate": float(y_test.mean())}
            for variant, sl in (("era5", slice(0, n_test)), ("fcst", slice(n_test, None))):
                row[f"roc_{variant}"] = float(roc_auc_score(y_test, cal[sl]))
                row[f"pr_{variant}"] = float(average_precision_score(y_test, cal[sl]))
            row["roc_gap"] = row["roc_fcst"] - row["roc_era5"]
            row["pr_gap"] = row["pr_fcst"] - row["pr_era5"]
            row["fit_minutes"] = (time.time() - t0) / 60
            results.append(row)
            pd.DataFrame(results).to_csv(results_path, index=False)
            log(f"  {config}/{TARGET_NAMES[i]:10s} roc {row['roc_era5']:.4f} -> "
                f"{row['roc_fcst']:.4f} (gap {row['roc_gap']:+.4f})   "
                f"pr {row['pr_era5']:.4f} -> {row['pr_fcst']:.4f}   "
                f"{row['fit_minutes']:.1f} min")
        del x_train_all, stacked_test

    out = pd.DataFrame(results)
    log("=== reanalysis→forecast gap, June 2026, identical rows & models ===")
    print(out.round(4).to_string(index=False))
    log(f"saved to {results_path}")


# %%
if __name__ == "__main__":
    phase = sys.argv[1] if len(sys.argv) > 1 else "compare"
    log(f"phase: {phase}")
    t0 = time.time()
    if phase == "compare":
        phase_compare()
    elif phase == "score":
        phase_score()
    else:
        raise SystemExit(f"unknown phase {phase!r}; use compare | score")
    log(f"phase {phase} done in {(time.time() - t0) / 60:.1f} min")
