import argparse
import os
import time

import psycopg2


DB_CONFIG = {
    "host": os.getenv("PGHOST", "localhost"),
    "port": int(os.getenv("PGPORT", "5432")),
    "dbname": os.getenv("PGDATABASE", "postgres"),
    "user": os.getenv("PGUSER", "postgres"),
    "password": os.getenv("PGPASSWORD", "Pass1234"),
}

SOURCE_TABLE = '"OM_BKK_DATA"'
PRECOMPUTE_TABLE = '"OM_BKK_DATA_PRECOMPUTE"'
RAIN_THRESHOLD_MM = 0.1
HORIZONS = [1, 2, 3, 6]


def connect():
    return psycopg2.connect(**DB_CONFIG)


def log(message):
    elapsed = time.strftime("%H:%M:%S")
    print(f"[{elapsed}] {message}", flush=True)


def future_precip_sql():
    return ",\n            ".join(
        f"LEAD(precipitation, {horizon}) OVER w AS precipitation_next_{horizon}h"
        for horizon in range(1, max(HORIZONS) + 1)
    )


def rain_any_target_sql(horizon):
    future_checks = " OR ".join(
        f"precipitation_next_{future_hour}h >= {RAIN_THRESHOLD_MM}"
        for future_hour in range(1, horizon + 1)
    )
    return (
        f"CASE "
        f"WHEN {future_checks} THEN 1 "
        f"WHEN precipitation_next_{horizon}h IS NULL THEN NULL "
        f"ELSE 0 END AS rain_any_next_{horizon}h"
    )


def create_precompute_sql(unlogged=False):
    table_type = "UNLOGGED TABLE" if unlogged else "TABLE"
    target_sql = ",\n            ".join(rain_any_target_sql(horizon) for horizon in HORIZONS)
    return f"""
CREATE {table_type} {PRECOMPUTE_TABLE} AS
WITH base_features AS (
    SELECT
        grid_number,
        grid_row,
        grid_column,
        longitude,
        latitude,
        forecast_time,
        local_forecast_time,
        timezone,
        temperature_2m,
        cape,
        dew_point_2m,
        relative_humidity_2m,
        precipitation,
        cloud_cover,
        surface_pressure,
        wind_direction_10m,
        wind_speed_10m,
        convective_inhibition,
        pressure_msl,
        model,
        source_api,
        fetched_at,
        temperature_2m - dew_point_2m AS temperature_dew_point_spread,
        pressure_msl - LAG(pressure_msl, 3) OVER w AS pressure_msl_change_3h,
        pressure_msl - LAG(pressure_msl, 6) OVER w AS pressure_msl_change_6h,
        LAG(precipitation, 1) OVER w AS precipitation_lag_1h,
        LAG(precipitation, 2) OVER w AS precipitation_lag_2h,
        LAG(precipitation, 3) OVER w AS precipitation_lag_3h,
        LAG(precipitation, 6) OVER w AS precipitation_lag_6h,
        SUM(precipitation) OVER (
            PARTITION BY grid_number ORDER BY local_forecast_time
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ) AS precipitation_sum_past_3h,
        SUM(precipitation) OVER (
            PARTITION BY grid_number ORDER BY local_forecast_time
            ROWS BETWEEN 5 PRECEDING AND CURRENT ROW
        ) AS precipitation_sum_past_6h,
        SUM(precipitation) OVER (
            PARTITION BY grid_number ORDER BY local_forecast_time
            ROWS BETWEEN 11 PRECEDING AND CURRENT ROW
        ) AS precipitation_sum_past_12h,
        SUM(precipitation) OVER (
            PARTITION BY grid_number ORDER BY local_forecast_time
            ROWS BETWEEN 23 PRECEDING AND CURRENT ROW
        ) AS precipitation_sum_past_24h,
        LAG(cloud_cover, 1) OVER w AS cloud_cover_lag_1h,
        LAG(cloud_cover, 3) OVER w AS cloud_cover_lag_3h,
        LAG(cloud_cover, 6) OVER w AS cloud_cover_lag_6h,
        LAG(relative_humidity_2m, 1) OVER w AS humidity_lag_1h,
        LAG(relative_humidity_2m, 3) OVER w AS humidity_lag_3h,
        LAG(relative_humidity_2m, 6) OVER w AS humidity_lag_6h,
        LAG(wind_speed_10m, 1) OVER w AS wind_speed_lag_1h,
        LAG(wind_speed_10m, 3) OVER w AS wind_speed_lag_3h,
        {future_precip_sql()},
        SIN(2 * pi() * EXTRACT(HOUR FROM local_forecast_time) / 24.0) AS hour_sin,
        COS(2 * pi() * EXTRACT(HOUR FROM local_forecast_time) / 24.0) AS hour_cos,
        SIN(2 * pi() * EXTRACT(MONTH FROM local_forecast_time) / 12.0) AS month_sin,
        COS(2 * pi() * EXTRACT(MONTH FROM local_forecast_time) / 12.0) AS month_cos
    FROM {SOURCE_TABLE}
    WINDOW w AS (PARTITION BY grid_number ORDER BY local_forecast_time)
), neighbor_features AS (
    SELECT
        c.grid_number,
        c.forecast_time,
        COUNT(n.grid_number)::double precision AS neighbor_count,
        AVG(n.precipitation) AS neighbor_precipitation_mean,
        MAX(n.precipitation) AS neighbor_precipitation_max,
        SUM(n.precipitation) AS neighbor_precipitation_sum,
        SUM(CASE WHEN n.precipitation >= {RAIN_THRESHOLD_MM} THEN 1 ELSE 0 END)::double precision AS neighbor_rain_count,
        AVG(CASE WHEN n.precipitation >= {RAIN_THRESHOLD_MM} THEN 1.0 ELSE 0.0 END) AS neighbor_rain_rate,
        AVG(n.cloud_cover) AS neighbor_cloud_cover_mean,
        MAX(n.cloud_cover) AS neighbor_cloud_cover_max,
        AVG(n.relative_humidity_2m) AS neighbor_relative_humidity_mean,
        MAX(n.relative_humidity_2m) AS neighbor_relative_humidity_max,
        AVG(n.pressure_msl) AS neighbor_pressure_msl_mean,
        MIN(n.pressure_msl) AS neighbor_pressure_msl_min,
        MAX(n.pressure_msl) AS neighbor_pressure_msl_max,
        AVG(n.temperature_2m) AS neighbor_temperature_2m_mean,
        AVG(n.dew_point_2m) AS neighbor_dew_point_2m_mean,
        AVG(n.temperature_dew_point_spread) AS neighbor_temperature_dew_point_spread_mean,
        AVG(n.wind_speed_10m) AS neighbor_wind_speed_10m_mean,
        MAX(n.wind_speed_10m) AS neighbor_wind_speed_10m_max,
        AVG(n.precipitation) FILTER (WHERE n.grid_row < c.grid_row) AS row_minus_precipitation_mean,
        AVG(n.precipitation) FILTER (WHERE n.grid_row > c.grid_row) AS row_plus_precipitation_mean,
        AVG(n.precipitation) FILTER (WHERE n.grid_column < c.grid_column) AS column_minus_precipitation_mean,
        AVG(n.precipitation) FILTER (WHERE n.grid_column > c.grid_column) AS column_plus_precipitation_mean,
        AVG(n.cloud_cover) FILTER (WHERE n.grid_row < c.grid_row) AS row_minus_cloud_cover_mean,
        AVG(n.cloud_cover) FILTER (WHERE n.grid_row > c.grid_row) AS row_plus_cloud_cover_mean,
        AVG(n.cloud_cover) FILTER (WHERE n.grid_column < c.grid_column) AS column_minus_cloud_cover_mean,
        AVG(n.cloud_cover) FILTER (WHERE n.grid_column > c.grid_column) AS column_plus_cloud_cover_mean
    FROM base_features c
    LEFT JOIN base_features n
      ON n.forecast_time = c.forecast_time
     AND n.grid_row BETWEEN c.grid_row - 1 AND c.grid_row + 1
     AND n.grid_column BETWEEN c.grid_column - 1 AND c.grid_column + 1
     AND n.grid_number <> c.grid_number
    GROUP BY c.grid_number, c.forecast_time
)
SELECT
    b.*,
    nf.neighbor_count,
    nf.neighbor_precipitation_mean,
    nf.neighbor_precipitation_max,
    nf.neighbor_precipitation_sum,
    nf.neighbor_rain_count,
    nf.neighbor_rain_rate,
    nf.neighbor_cloud_cover_mean,
    nf.neighbor_cloud_cover_max,
    nf.neighbor_relative_humidity_mean,
    nf.neighbor_relative_humidity_max,
    nf.neighbor_pressure_msl_mean,
    nf.neighbor_pressure_msl_min,
    nf.neighbor_pressure_msl_max,
    nf.neighbor_temperature_2m_mean,
    nf.neighbor_dew_point_2m_mean,
    nf.neighbor_temperature_dew_point_spread_mean,
    nf.neighbor_wind_speed_10m_mean,
    nf.neighbor_wind_speed_10m_max,
    COALESCE(nf.row_minus_precipitation_mean, 0.0) AS row_minus_precipitation_mean,
    COALESCE(nf.row_plus_precipitation_mean, 0.0) AS row_plus_precipitation_mean,
    COALESCE(nf.column_minus_precipitation_mean, 0.0) AS column_minus_precipitation_mean,
    COALESCE(nf.column_plus_precipitation_mean, 0.0) AS column_plus_precipitation_mean,
    COALESCE(nf.row_minus_cloud_cover_mean, b.cloud_cover) AS row_minus_cloud_cover_mean,
    COALESCE(nf.row_plus_cloud_cover_mean, b.cloud_cover) AS row_plus_cloud_cover_mean,
    COALESCE(nf.column_minus_cloud_cover_mean, b.cloud_cover) AS column_minus_cloud_cover_mean,
    COALESCE(nf.column_plus_cloud_cover_mean, b.cloud_cover) AS column_plus_cloud_cover_mean,
    nf.neighbor_precipitation_mean - b.precipitation AS neighbor_precipitation_mean_minus_center,
    nf.neighbor_cloud_cover_mean - b.cloud_cover AS neighbor_cloud_cover_mean_minus_center,
    nf.neighbor_relative_humidity_mean - b.relative_humidity_2m AS neighbor_relative_humidity_mean_minus_center,
    b.pressure_msl - nf.neighbor_pressure_msl_mean AS center_pressure_msl_minus_neighbor_mean,
    {target_sql},
    now() AS precomputed_at
FROM base_features b
LEFT JOIN neighbor_features nf
  ON nf.grid_number = b.grid_number
 AND nf.forecast_time = b.forecast_time;
"""


CREATE_INDEX_SQL = f"""
ALTER TABLE {PRECOMPUTE_TABLE}
    ADD CONSTRAINT om_bkk_data_precompute_pk PRIMARY KEY (grid_number, forecast_time);

CREATE INDEX om_bkk_data_precompute_local_time_idx
    ON {PRECOMPUTE_TABLE} (local_forecast_time);

CREATE INDEX om_bkk_data_precompute_grid_local_time_idx
    ON {PRECOMPUTE_TABLE} (grid_number, local_forecast_time);

CREATE INDEX om_bkk_data_precompute_split_ready_idx
    ON {PRECOMPUTE_TABLE} (local_forecast_time, grid_number)
    WHERE precipitation_lag_6h IS NOT NULL
      AND precipitation_sum_past_24h IS NOT NULL
      AND cloud_cover_lag_6h IS NOT NULL
      AND humidity_lag_6h IS NOT NULL
      AND wind_speed_lag_3h IS NOT NULL;
"""


SUMMARY_SQL = f"""
SELECT
    COUNT(*) AS row_count,
    COUNT(DISTINCT grid_number) AS grid_count,
    MIN(local_forecast_time) AS min_local_time,
    MAX(local_forecast_time) AS max_local_time,
    SUM(CASE WHEN rain_any_next_1h IS NOT NULL THEN 1 ELSE 0 END) AS labeled_1h_rows,
    SUM(CASE WHEN rain_any_next_2h IS NOT NULL THEN 1 ELSE 0 END) AS labeled_2h_rows,
    SUM(CASE WHEN rain_any_next_3h IS NOT NULL THEN 1 ELSE 0 END) AS labeled_3h_rows,
    SUM(CASE WHEN rain_any_next_6h IS NOT NULL THEN 1 ELSE 0 END) AS labeled_6h_rows
FROM {PRECOMPUTE_TABLE};
"""


def table_exists(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s);", ("public.OM_BKK_DATA_PRECOMPUTE",))
        return cur.fetchone()[0] is not None


def rebuild_table(conn, replace=False, unlogged=False):
    conn.autocommit = True
    with conn.cursor() as cur:
        if replace:
            log(f"Dropping existing {PRECOMPUTE_TABLE} if present")
            cur.execute(f"DROP TABLE IF EXISTS {PRECOMPUTE_TABLE};")
        elif table_exists(conn):
            raise RuntimeError(
                f"{PRECOMPUTE_TABLE} already exists. Re-run with --replace to rebuild it."
            )

        log(f"Creating {PRECOMPUTE_TABLE} with precomputed features")
        cur.execute(create_precompute_sql(unlogged=unlogged))

        log("Adding primary key and indexes")
        cur.execute(CREATE_INDEX_SQL)

        log("Analyzing precompute table")
        cur.execute(f"ANALYZE {PRECOMPUTE_TABLE};")

        cur.execute(SUMMARY_SQL)
        return cur.fetchone()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Build PostgreSQL table "OM_BKK_DATA_PRECOMPUTE" from "OM_BKK_DATA" '
            "with baseline lag features, neighbor-grid features, and rain-any labels."
        )
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help='Drop and rebuild "OM_BKK_DATA_PRECOMPUTE" if it already exists.',
    )
    parser.add_argument(
        "--unlogged",
        action="store_true",
        help="Create as an UNLOGGED table for faster rebuilds. Data is not crash-safe until rebuilt.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    started = time.time()
    with connect() as conn:
        summary = rebuild_table(conn, replace=args.replace, unlogged=args.unlogged)

    log(
        "Summary: "
        f"rows={summary[0]:,}, grids={summary[1]:,}, "
        f"time={summary[2]} to {summary[3]}, "
        f"labeled_1h={summary[4]:,}, labeled_2h={summary[5]:,}, "
        f"labeled_3h={summary[6]:,}, labeled_6h={summary[7]:,}"
    )
    log(f"Finished in {(time.time() - started) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
