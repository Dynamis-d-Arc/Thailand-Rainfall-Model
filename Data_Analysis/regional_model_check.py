"""Regional check of the Thailand rain models: does skill vary by part of the country?

Scores the already-trained V1 models on rows they never saw (V1's validation + test period,
2026-03-15 17:00 onward). No retraining. Each city is evaluated over its own grid cell plus
the 8 surrounding cells, so estimates come from ~22k rows rather than ~2.5k.

The window spans two regimes, reported separately:
  dry  2026-03-15 .. 2026-04-30   (late dry season)
  wet  2026-05-01 .. 2026-06-25   (monsoon onset)

Raw PR-AUC is not comparable between them - a random model scores the base rate - so
normalised AP (AP - base) / (1 - base) is reported alongside.
"""

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psycopg2
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

DB_CONFIG = {
    "host": os.getenv("PGHOST", "localhost"),
    "port": int(os.getenv("PGPORT", "5432")),
    "dbname": os.getenv("PGDATABASE", "postgres"),
    "user": os.getenv("PGUSER", "postgres"),
    "password": os.getenv("PGPASSWORD", "Pass1234"),
}

PRECOMPUTE_TABLE = '"OM_THAILAND_DATA_PRECOMPUTE"'
PAIR_TABLE = '"OM_THAILAND_GRID_PAIRS"'
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "ML_Model_V2" / "trained_models" / "om_thailand_rain_any_final_4_neighbor_models"
OUTPUT_CSV = MODEL_DIR / "regional_check_by_city.csv"

# Everything from V1's train_end onward is out-of-sample for these models.
OOS_START = "2026-03-15 17:00:00"
WET_START = "2026-05-01 00:00:00"

HORIZONS = [1, 2, 3, 6]
MODEL_PLAN = {1: "lightgbm", 2: "lightgbm", 3: "hist_gradient_boosting", 6: "hist_gradient_boosting"}

CITIES = [
    ("Chiang Mai", "North", 18.7883, 98.9853),
    ("Ubon Ratchathani", "East", 15.2448, 104.8473),
    ("Kanchanaburi", "West", 14.0227, 99.5328),
    ("Hat Yai", "South", 7.0086, 100.4747),
    ("Bangkok", "Central", 13.7563, 100.5018),
]

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
TARGET_COLUMNS = [f"rain_any_next_{h}h" for h in HORIZONS]


def connect():
    return psycopg2.connect(**DB_CONFIG)


def load_models():
    models, thresholds = {}, {}
    meta = json.loads((MODEL_DIR / "final_4_metadata.json").read_text(encoding="utf-8"))
    thresholds = {int(k): float(v) for k, v in meta["selected_probability_thresholds"].items()}
    for path in MODEL_DIR.glob("om_thailand_rain_any_next_*h_*.joblib"):
        horizon = int(path.name.split("next_")[1].split("h_")[0])
        models[horizon] = joblib.load(path)
    missing = set(HORIZONS) - set(models)
    if missing:
        raise FileNotFoundError(f"Missing trained models for horizons {sorted(missing)}")
    return models, thresholds


def city_blocks(conn):
    """Nearest grid cell to each city, plus its 8 neighbours."""
    with conn.cursor() as cur:
        cur.execute(f'SELECT DISTINCT grid_number, latitude, longitude FROM "OM_Thailand_Data"')
        grid = np.array(cur.fetchall(), dtype="float64")

    blocks = {}
    for name, direction, lat, lon in CITIES:
        km = 111.0 * np.sqrt(
            (grid[:, 1] - lat) ** 2 + ((grid[:, 2] - lon) * np.cos(np.radians(lat))) ** 2
        )
        centre = int(grid[int(km.argmin()), 0])
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT neighbor_grid_number FROM {PAIR_TABLE} WHERE center_grid_number = %s",
                (centre,),
            )
            cells = sorted({centre} | {int(r[0]) for r in cur.fetchall()})
        blocks[name] = {"direction": direction, "centre": centre, "cells": cells,
                        "centre_km": float(km.min())}
        print(f"  {name:18s} {direction:8s} centre grid {centre:4d} "
              f"({km.min():.1f} km away), {len(cells)} cells in block")
    return blocks


def load_block(conn, cells):
    feature_sql = ",\n        ".join(
        f"COALESCE({c}::real, 'NaN'::real) AS {c}" for c in FEATURE_COLUMNS
    )
    target_sql = ",\n        ".join(f"{t}::smallint" for t in TARGET_COLUMNS)
    query = f"""
    SELECT local_forecast_time,
        {feature_sql},
        {target_sql}
    FROM {PRECOMPUTE_TABLE}
    WHERE grid_number = ANY(%s)
      AND local_forecast_time >= TIMESTAMP '{OOS_START}'
      AND rain_any_next_6h IS NOT NULL
      AND neighbor_count > 0
      AND pressure_msl_change_6h IS NOT NULL
      AND precipitation_lag_6h IS NOT NULL
      AND cloud_cover_lag_6h IS NOT NULL
      AND humidity_lag_6h IS NOT NULL
      AND wind_speed_lag_3h IS NOT NULL
    ORDER BY local_forecast_time
    """
    with conn.cursor() as cur:
        cur.execute(query, (cells,))
        rows = cur.fetchall()
    block = np.asarray(rows, dtype="object")
    n_feat = len(FEATURE_COLUMNS)
    times = block[:, 0].astype("datetime64[s]")
    x = block[:, 1:1 + n_feat].astype("float32")
    y = block[:, 1 + n_feat:].astype("int8")
    return times, x, y


def score(y_true, probabilities, threshold):
    base = float(y_true.mean())
    out = {"rows": int(len(y_true)), "rain_rate": base}
    if y_true.min() == y_true.max():
        out.update({"roc_auc": np.nan, "pr_auc": np.nan, "norm_ap": np.nan, "brier": np.nan})
    else:
        ap = float(average_precision_score(y_true, probabilities))
        out.update({
            "roc_auc": float(roc_auc_score(y_true, probabilities)),
            "pr_auc": ap,
            "norm_ap": (ap - base) / (1 - base),
            "brier": float(brier_score_loss(y_true, probabilities)),
        })
    y_pred = (probabilities >= threshold).astype("int8")
    out.update({
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    })
    return out


def main():
    models, thresholds = load_models()
    print(f"Loaded models for horizons {sorted(models)}; thresholds {thresholds}\n")

    conn = connect()
    try:
        print("Locating city blocks:")
        blocks = city_blocks(conn)
        print()

        wet_start = np.datetime64(WET_START.replace(" ", "T"))
        records = []
        for name, info in blocks.items():
            times, x, y = load_block(conn, info["cells"])
            print(f"{name}: {len(times):,} rows "
                  f"({str(times.min())[:10]} .. {str(times.max())[:10]})")
            regimes = {
                "dry (Mar-Apr)": times < wet_start,
                "wet (May-Jun)": times >= wet_start,
                "all": np.ones(len(times), dtype=bool),
            }
            for horizon_index, horizon in enumerate(HORIZONS):
                probabilities = models[horizon].predict_proba(x)[:, 1]
                for regime, mask in regimes.items():
                    if mask.sum() == 0:
                        continue
                    row = {"city": name, "direction": info["direction"],
                           "centre_grid": info["centre"], "cells": len(info["cells"]),
                           "regime": regime, "horizon_h": horizon,
                           "model": MODEL_PLAN[horizon], "threshold": thresholds[horizon]}
                    row.update(score(y[mask, horizon_index], probabilities[mask], thresholds[horizon]))
                    records.append(row)
    finally:
        conn.close()

    results = pd.DataFrame(records)
    results.to_csv(OUTPUT_CSV, index=False)

    pd.set_option("display.width", 250)
    fmt = "{:.3f}".format
    for horizon in HORIZONS:
        for regime in ["dry (Mar-Apr)", "wet (May-Jun)"]:
            sub = results[(results.horizon_h == horizon) & (results.regime == regime)]
            print(f"\n===== {horizon}h | {regime} =====")
            print(sub[["direction", "city", "rows", "rain_rate", "roc_auc", "pr_auc",
                       "norm_ap", "precision", "recall", "f1"]]
                  .to_string(index=False, float_format=fmt))

    print(f"\nSaved {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
