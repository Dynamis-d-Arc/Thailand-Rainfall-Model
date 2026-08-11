# %% [markdown]
# # Thailand_Rain_V9 — why does the Himawari gain halve in 2026?
#
# V7's IR increment is stable at +0.041..0.052 ROC (h1_1.0mm) across five 2024/2025 blocks,
# then drops to +0.023 (May-Jun 2026) and +0.015 (Jul 2026). The calendar twins kill the
# seasonal explanation outright — May-Jun 2025 scored +0.044 and Jul 2024/2025 scored
# +0.042/+0.046 — and `om_only` holds its usual level in 2026, so the base model didn't get
# better; the IR features stopped adding as much. Provisional-IMERG-as-such is also out:
# block 7 (Sep-Oct 2025) is provisional and shows the *best* gain (+0.052).
#
# Remaining suspects, each with a targeted no-training slice below:
#
#   A. IR feed degradation   — failed slots / hw_valid / NaN share by month; IR brightness-
#                              temperature distributions by month (calibration drift).
#   B. Raw signal decay      — ROC of the untrained IR signal (-tb_min_env_lag1) against the
#                              same labels, by month. If THIS decays, the cold-cloud→rain
#                              relationship itself broke (feed or grader side); if it holds
#                              while the model gain decays, it's a model/feature interaction.
#   C. Grader change         — IMERG product_version / wet-rate / intensity trends by month.
#   D. Spatial signature     — gain by region (grid_row terciles) x era: a feed problem is
#                              geographically uniform, a regime/grader shift often is not.
#
# Everything runs off the V3/V7 caches + light SQL; no model is trained.
#
# VERDICT (2026-08-11, from the slices below): the decay is **meteorological, not a defect**.
# The raw untrained IR signal itself drops (ROC 0.72-0.76 in calendar twins → 0.68-0.69 in
# 2026) while feed health (hw_valid ~99%, stable Tb distributions) and the grader (version 07
# throughout; provisional Oct 2025 scores raw 0.82) are unchanged. Slice E pins the mechanism:
# the share of labeled rain events with any cold-top (≤235 K) signature falls from 85% in
# 2024/25 to 72% in 2026 (Jul: 70%), deep tops (<220 K) from ~65% to 44-60%, median cloud-top
# over rain warms ~211 K → 219-223 K. The weak 2026 early wet season produced more warm-topped
# shallow rain, which IR physically cannot see — so the IR increment shrinks in exact
# proportion to the convective share. Deployment implication: the V7 gain is regime-dependent;
# track p(cold-top | rain) as a live health metric, and expect the gain to recover with deep
# convection. A warm-rain-capable feed (radar, microwave) is the only way to close this gap.
#
# Phase:
#     python Thailand_Rain_V9_2026_decay.py diagnose

# %%
import json
import sys
import time

import numpy as np
import pandas as pd
import psycopg2

from BKK_Rain_V3 import PROJECT_ROOT, TARGET_NAMES, log
from Thailand_Rain_V3 import CACHE, DB_CONFIG, IMERG_TABLE_NAME, block_keys
from Thailand_Rain_V7_himawari_ir import himawari_feature_names

OUTPUT_DIR = PROJECT_ROOT / "ML_Model_V2" / "trained_models" / "om_thailand_rain_v9_2026_decay"
TARGET = "h1_1.0mm"          # the headline target; 0.1mm variant reported alongside
COVERED_FOLDS = [0, 1, 5, 6, 7, 11, 12]


def connect():
    return psycopg2.connect(**DB_CONFIG)


def month_key(tf):
    stamps = pd.to_datetime(tf)
    return stamps.year.to_numpy() * 100 + stamps.month.to_numpy()


# %%
def phase_diagnose():
    from sklearn.metrics import roc_auc_score

    y = np.load(CACHE / "y.npy")
    tf = np.load(CACHE / "forecast_time.npy")
    unit = np.load(CACHE / "unit_id.npy")
    hw_block = np.load(CACHE / "hw_block_offset0.npy")
    oof_hw = np.load(CACHE / "oof_cal_hw_offset0h.npy")
    oof_om = np.load(CACHE / "oof_cal_om_only.npy")
    fold_key = block_keys(tf)
    hw_names = himawari_feature_names()

    ti = {n: i for i, n in enumerate(TARGET_NAMES)}
    months = month_key(tf)
    in_scope = np.isin(fold_key, COVERED_FOLDS)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- A + B: monthly trajectory of model gain, raw-signal ROC, and IR quality ----
    env1 = hw_block[:, hw_names.index("hw_tb_min_env_lag1")]
    cold1 = hw_block[:, hw_names.index("hw_cold235_env_lag1")]
    valid = hw_block[:, hw_names.index("hw_valid")] > 0

    rows = []
    for mo in np.unique(months[in_scope]):
        base_m = in_scope & (months == mo)
        for tname in (TARGET, "h1_0.1mm"):
            i = ti[tname]
            m = base_m & np.isfinite(oof_hw[:, i]) & np.isfinite(oof_om[:, i])
            if m.sum() < 5000 or len(np.unique(y[m, i])) < 2:
                continue
            mv = m & valid & np.isfinite(env1)
            rows.append({
                "month": int(mo), "target": tname, "rows": int(m.sum()),
                "base_rate": float(y[m, i].mean()),
                "om_only": float(roc_auc_score(y[m, i], oof_om[m, i])),
                "hw_offset0h": float(roc_auc_score(y[m, i], oof_hw[m, i])),
                "raw_ir_roc": float(roc_auc_score(y[mv, i], -env1[mv]))
                              if mv.sum() > 1000 and len(np.unique(y[mv, i])) == 2 else np.nan,
                "hw_valid_share": float(valid[base_m].mean()),
                "env1_nan_share": float(np.isnan(env1[base_m]).mean()),
                "tb_min_env_p10": float(np.nanpercentile(env1[base_m & valid], 10))
                                  if (base_m & valid).any() else np.nan,
                "tb_min_env_median": float(np.nanmedian(env1[base_m & valid]))
                                     if (base_m & valid).any() else np.nan,
                "cold235_wet_mean": float(np.nanmean(cold1[base_m & valid])),
            })
    monthly = pd.DataFrame(rows)
    monthly["roc_gain"] = monthly["hw_offset0h"] - monthly["om_only"]
    monthly.to_csv(OUTPUT_DIR / "v9_monthly_trajectory.csv", index=False)
    hm = monthly[monthly.target == TARGET]
    log(f"=== monthly trajectory ({TARGET}) ===")
    print(hm[["month", "rows", "base_rate", "om_only", "hw_offset0h", "roc_gain",
              "raw_ir_roc", "hw_valid_share", "tb_min_env_p10", "tb_min_env_median",
              "cold235_wet_mean"]].round(4).to_string(index=False))

    # ---- C: what the grader was doing, straight from the DB ----
    conn = connect()
    try:
        grader = pd.read_sql(f"""
            SELECT date_trunc('month', local_observation_time) AS month,
                   run_type,
                   MIN(product_version) AS min_version, MAX(product_version) AS max_version,
                   COUNT(*) AS rows,
                   AVG((precipitation_max_mm >= 1.0)::int) AS wet1_rate,
                   AVG((precipitation_max_mm >= 0.1)::int) AS wet01_rate,
                   AVG(precipitation_max_mm) FILTER (WHERE precipitation_max_mm >= 1.0)
                       AS wet_intensity,
                   AVG(pixel_count) AS pixel_count_mean
            FROM {IMERG_TABLE_NAME}
            WHERE is_complete_hour
            GROUP BY 1, 2 ORDER BY 1
        """, conn)
    finally:
        conn.close()
    grader.to_csv(OUTPUT_DIR / "v9_grader_by_month.csv", index=False)
    g = grader[grader["month"] >= "2025-03-01"]
    log("=== IMERG grader by month (2025-03 on) ===")
    print(g.round(4).to_string(index=False))

    # ---- D: gain by region x era ----
    conn = connect()
    try:
        gr = pd.read_sql('SELECT grid_number, grid_row, latitude FROM "Thailand_Grid_25km"', conn)
    finally:
        conn.close()
    terciles = np.quantile(gr["grid_row"], [1 / 3, 2 / 3])
    band = np.where(gr["grid_row"] <= terciles[0], "south",
                    np.where(gr["grid_row"] <= terciles[1], "central", "north"))
    band_of = dict(zip(gr["grid_number"].astype(int), band))
    row_band = np.array([band_of[int(g_)] for g_ in unit])
    year = (months // 100)

    from sklearn.metrics import roc_auc_score as roc
    i = ti[TARGET]
    reg_rows = []
    for era, era_mask in [("2024_25", in_scope & (year <= 2025)),
                          ("2026", in_scope & (year == 2026))]:
        for b in ("north", "central", "south"):
            m = era_mask & (row_band == b) & np.isfinite(oof_hw[:, i]) & np.isfinite(oof_om[:, i])
            if m.sum() < 5000 or len(np.unique(y[m, i])) < 2:
                continue
            mv = m & valid & np.isfinite(env1)
            reg_rows.append({
                "era": era, "band": b, "rows": int(m.sum()),
                "base_rate": float(y[m, i].mean()),
                "om_only": float(roc(y[m, i], oof_om[m, i])),
                "hw_offset0h": float(roc(y[m, i], oof_hw[m, i])),
                "raw_ir_roc": float(roc(y[mv, i], -env1[mv])) if mv.sum() > 1000 else np.nan,
            })
    regional = pd.DataFrame(reg_rows)
    regional["roc_gain"] = regional["hw_offset0h"] - regional["om_only"]
    regional.to_csv(OUTPUT_DIR / "v9_regional_gain.csv", index=False)
    log(f"=== gain by region x era ({TARGET}) ===")
    print(regional.round(4).to_string(index=False))

    # ---- E: rain type — what share of labeled rain has a cold-top signature at all? ----
    # This is the slice that closed the case: IR can only see rain from cold (deep
    # convective) cloud tops. If the rain regime shifts warm/shallow, the IR increment
    # must shrink no matter how healthy the feed is.
    i = ti[TARGET]
    rt_rows = []
    for mo in np.unique(months[in_scope]):
        m = in_scope & (months == mo) & valid & np.isfinite(env1)
        wet = m & (y[:, i] == 1)
        dry = m & (y[:, i] == 0)
        if wet.sum() < 1000:
            continue
        rt_rows.append({
            "month": int(mo), "wet_rows": int(wet.sum()),
            "p_cold235_given_rain": float((cold1[wet] > 0).mean()),
            "p_tbmin_lt220_given_rain": float((env1[wet] < 220).mean()),
            "tbmin_median_wet": float(np.median(env1[wet])),
            "p_cold235_given_dry": float((cold1[dry] > 0).mean()),
        })
    raintype = pd.DataFrame(rt_rows)
    raintype.to_csv(OUTPUT_DIR / "v9_rain_type_by_month.csv", index=False)
    log(f"=== cold-top share of rain events by month ({TARGET}) ===")
    print(raintype.round(3).to_string(index=False))

    log(f"saved v9 diagnostics to {OUTPUT_DIR}")


# %%
if __name__ == "__main__":
    phase = sys.argv[1] if len(sys.argv) > 1 else "diagnose"
    log(f"phase: {phase}")
    t0 = time.time()
    if phase == "diagnose":
        phase_diagnose()
    else:
        raise SystemExit(f"unknown phase {phase!r}; use diagnose")
    log(f"phase {phase} done in {(time.time() - t0) / 60:.1f} min")
