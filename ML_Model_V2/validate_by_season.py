"""Score a deployed V2 bundle inside specific seasonal windows.

The headline OOF metrics average six whole years, so they are dominated by dry-season hours
where "no rain" is nearly always right. This script scores the shipped model inside chosen
windows instead, which separates two effects the annual average hides:

  * **season** - wet vs dry, by comparing monsoon and dry windows;
  * **label vintage** - final-run vs provisional Late Run IMERG, by comparing the same season
    across years either side of the 2025-10-01 cutover.

Note the rows here are *in-sample*: the deployed model trained on every row in the database, so
these figures are optimistic. They are useful as an upper bound and for comparing windows to each
other, not as a replacement for the out-of-fold numbers in `deploy_oof_metrics.csv`.

Usage:
    python ML_Model_V2/validate_by_season.py
    python ML_Model_V2/validate_by_season.py --target h1_1.0mm
"""

import argparse
import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psycopg2
from sklearn.metrics import precision_score, recall_score, roc_auc_score

warnings.filterwarnings("ignore", message="X does not have valid feature names")

DB_CONFIG = {
    "host": os.getenv("PGHOST", "localhost"),
    "port": int(os.getenv("PGPORT", "5432")),
    "dbname": os.getenv("PGDATABASE", "postgres"),
    "user": os.getenv("PGUSER", "postgres"),
    "password": os.getenv("PGPASSWORD", "Pass1234"),
}
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "ML_Model_V2" / "trained_models" / "om_bkk_rain_v2_deploy"
OUTPUT_DIR = MODEL_DIR

# IMERG switches to provisional Late Run here, with roughly 12% lower detection.
PROVISIONAL_FROM = pd.Timestamp("2025-10-01")

WINDOWS = [
    ("wet 2024", "2024-06-01", "2024-07-20"),
    ("wet 2025", "2025-06-01", "2025-07-20"),
    ("dry 2026", "2026-01-01", "2026-02-20"),
    ("wet 2026", "2026-06-01", "2026-07-20"),
]


def find_bundle(target):
    horizon = target.split("_")[0][1:]
    rain_mm = target.split("_")[1]
    matches = [p for p in MODEL_DIR.glob(f"*_next_{horizon}h_*")
               if p.name.endswith(f"rain_threshold_{rain_mm}.joblib")]
    if not matches:
        raise SystemExit(f"no bundle for {target} in {MODEL_DIR}")
    return sorted(matches)[0]


def score_window(bundle, start, end):
    feature_columns = bundle["feature_columns"]
    horizon = bundle["horizon_h"]
    rain_mm = bundle["rain_threshold_mm"]
    feature_sql = ",\n".join(f"COALESCE(om.{c}::real,'NaN'::real)" for c in feature_columns)
    lead_sql = ",\n".join(
        f"LEAD(precipitation_mm,{k}) OVER w AS p{k}, "
        f"LEAD(local_observation_time,{k}) OVER w AS t{k}" for k in range(1, horizon + 1))
    guard_sql = ",\n".join(
        f"CASE WHEN im.t{k}=im.t_local+interval '{k} hour' THEN im.p{k}::real ELSE 'NaN'::real END"
        for k in range(1, horizon + 1))
    query = f'''
    WITH im AS (
        SELECT grid_number, local_observation_time AS t_local, {lead_sql}
        FROM "IMERG_BKK_DATA" WHERE is_complete_hour
        WINDOW w AS (PARTITION BY grid_number ORDER BY local_observation_time))
    SELECT om.local_forecast_time, {feature_sql}, {guard_sql}
    FROM "OM_BKK_DATA_PRECOMPUTE" om
    JOIN im ON im.grid_number = om.grid_number AND im.t_local = om.local_forecast_time
    WHERE om.local_forecast_time >= timestamp '{start}'
      AND om.local_forecast_time <  timestamp '{end}'
      AND om.pressure_msl_change_6h IS NOT NULL AND om.precipitation_lag_6h IS NOT NULL
      AND om.precipitation_sum_past_24h IS NOT NULL AND om.cloud_cover_lag_6h IS NOT NULL
      AND om.humidity_lag_6h IS NOT NULL AND om.wind_speed_lag_3h IS NOT NULL
      AND om.neighbor_count > 0'''
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()

    block = np.asarray(rows, dtype="object")
    n = len(feature_columns)
    x = block[:, 1:1 + n].astype("float32")
    future = block[:, 1 + n:].astype("float32")
    complete = ~np.isnan(future).any(axis=1)
    x, future = x[complete], future[complete]
    y = (future.max(axis=1) >= rain_mm).astype("int8")

    probability = bundle["calibrator"].predict(bundle["model"].predict_proba(x)[:, 1])
    # the window sits inside one calendar month band, so its own month's cut applies; bundles
    # retuned by fit_seasonal_thresholds.py carry one per month, older ones only the global
    month = pd.Timestamp(start).month
    seasonal = bundle.get("seasonal_thresholds")
    threshold = float(seasonal[month]) if seasonal else bundle["probability_threshold"]
    fired = (probability >= threshold).astype("int8")
    return {
        "rows": int(len(y)),
        "threshold": threshold,
        "base_rate": float(y.mean()),
        "mean_probability": float(probability.mean()),
        "roc_auc": float(roc_auc_score(y, probability)),
        "precision": float(precision_score(y, fired, zero_division=0)),
        "recall": float(recall_score(y, fired, zero_division=0)),
        "fired_share": float(fired.mean()),
    }


ALL_TARGETS = ["h1_0.1mm", "h3_0.1mm", "h6_0.1mm", "h1_1.0mm", "h3_1.0mm", "h6_1.0mm"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default=None, help="one target; omit to run all six")
    args = parser.parse_args()
    targets = [args.target] if args.target else ALL_TARGETS

    results = []
    for target in targets:
        path = find_bundle(target)
        bundle = joblib.load(path)
        mode = "monthly" if bundle.get("seasonal_thresholds") else "global"
        print(f"\n{target}  (OOF ROC {bundle['metrics']['roc_auc']:.4f}, {mode} threshold)")
        for label, start, end in WINDOWS:
            row = score_window(bundle, start, end)
            row["target"] = target
            row["window"] = label
            row["label_vintage"] = ("provisional" if pd.Timestamp(start) >= PROVISIONAL_FROM
                                    else "final run")
            # F1 of the trivial "always rain" rule, the bar any operating point must clear
            row["f1_always_yes"] = 2 * row["base_rate"] / (row["base_rate"] + 1)
            row["f1"] = (2 * row["precision"] * row["recall"] / (row["precision"] + row["recall"])
                         if row["precision"] + row["recall"] > 0 else 0.0)
            results.append(row)
            print(f"  {label:10} {row['label_vintage']:12} base {row['base_rate']:.3f}  "
                  f"thr {row['threshold']:.3f}  ROC {row['roc_auc']:.4f}  "
                  f"precision {row['precision']:.3f}  recall {row['recall']:.3f}  "
                  f"F1 {row['f1']:.3f} (always-yes {row['f1_always_yes']:.3f})  "
                  f"fires {row['fired_share']:.1%}", flush=True)
        del bundle

    frame = pd.DataFrame(results)[
        ["target", "window", "label_vintage", "rows", "base_rate", "mean_probability",
         "threshold", "roc_auc", "precision", "recall", "f1", "f1_always_yes", "fired_share"]]
    suffix = args.target or "all_targets"
    out = OUTPUT_DIR / f"seasonal_validation_{suffix}.csv"
    frame.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
