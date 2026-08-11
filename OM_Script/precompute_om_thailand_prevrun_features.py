"""Build "OM_THAILAND_PREVRUN_PRECOMPUTE" from "OM_Thailand_PrevRun_Data".

Previous-run (+24h lead) twin of precompute_om_thailand_features.py (June 2026 gap experiment).

Thailand counterpart of precompute_om_bkk_features.py. Two differences from the
Bangkok build:

1. "OM_Thailand_Data" has no cape / convective_inhibition / model columns, so
   those pass-through fields are dropped. No modelling feature used them.
2. Neighbour aggregation goes through a precomputed grid-adjacency pair table
   instead of a per-timestamp spatial self-join. Thailand has 833 grids against
   Bangkok's 96, and the self-join form degenerates to 833 x 833 pairs per hour
   (~5.7 billion row comparisons) because a CTE offers the planner no index.

The source panel was unbalanced when this was written (all 833 grids to
2026-06-25, only 100 beyond it), so everything after PANEL_END_LOCAL was cut to
keep the chronological split geographically constant. The 2026-08-10 backfill
closed that gap and extended history back to 2024-07-01; all 833 grids now run
the full 2024-07-01 -> 2026-07-19 span, verified day by day at 833 x 24. The cut
is kept as a guard so a future partial fetch cannot silently unbalance the panel.

The rain_any_next_* label columns built here are ERA5-derived and exist only for
continuity with the earlier Thailand notebooks. Modelling that follows the V3
protocol ignores them: BKK_Rain_V3.build_panel_query takes features from the
precompute table and builds its targets from observed IMERG instead. IMERG
cell-mean correlates with ERA5 precipitation at r=0.12 on identical
grid/hour rows, so the two label sets are not interchangeable.
"""

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

SOURCE_TABLE = '"OM_Thailand_PrevRun_Data"'
PRECOMPUTE_TABLE = '"OM_THAILAND_PREVRUN_PRECOMPUTE"'
PAIR_TABLE = '"OM_THAILAND_PREVRUN_GRID_PAIRS"'
RAIN_THRESHOLD_MM = 0.1
HORIZONS = [1, 2, 3, 6]

# Last local hour at which all 833 grids still report.
PANEL_END_LOCAL = "2026-06-30 23:00:00"

SESSION_TUNING_SQL = """
SET work_mem = '256MB';
SET maintenance_work_mem = '256MB';
SET synchronous_commit = off;
"""


def connect():
    return psycopg2.connect(**DB_CONFIG)


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# --------------------------------------------------------------------------
# Grid adjacency
# --------------------------------------------------------------------------

CREATE_PAIRS_SQL = f"""
DROP TABLE IF EXISTS {PAIR_TABLE};
CREATE TABLE {PAIR_TABLE} AS
WITH grids AS (
    SELECT DISTINCT grid_number, grid_row, grid_column FROM {SOURCE_TABLE}
)
SELECT
    c.grid_number AS center_grid_number,
    n.grid_number AS neighbor_grid_number,
    (n.grid_row < c.grid_row)       AS is_row_minus,
    (n.grid_row > c.grid_row)       AS is_row_plus,
    (n.grid_column < c.grid_column) AS is_column_minus,
    (n.grid_column > c.grid_column) AS is_column_plus
FROM grids c
JOIN grids n
  ON n.grid_row BETWEEN c.grid_row - 1 AND c.grid_row + 1
 AND n.grid_column BETWEEN c.grid_column - 1 AND c.grid_column + 1
 AND n.grid_number <> c.grid_number;

CREATE INDEX om_thailand_prevrun_grid_pairs_neighbor_idx
    ON {PAIR_TABLE} (neighbor_grid_number);
ANALYZE {PAIR_TABLE};
"""


# --------------------------------------------------------------------------
# Neighbour aggregates
# --------------------------------------------------------------------------

CREATE_NEIGHBOR_SQL = f"""
DROP TABLE IF EXISTS om_thailand_prevrun_neighbor_tmp;
CREATE UNLOGGED TABLE om_thailand_prevrun_neighbor_tmp AS
SELECT
    p.center_grid_number AS grid_number,
    s.local_forecast_time,
    COUNT(*)::double precision AS neighbor_count,
    AVG(s.precipitation) AS neighbor_precipitation_mean,
    MAX(s.precipitation) AS neighbor_precipitation_max,
    SUM(s.precipitation) AS neighbor_precipitation_sum,
    SUM(CASE WHEN s.precipitation >= {RAIN_THRESHOLD_MM} THEN 1 ELSE 0 END)::double precision
        AS neighbor_rain_count,
    AVG(CASE WHEN s.precipitation >= {RAIN_THRESHOLD_MM} THEN 1.0 ELSE 0.0 END)
        AS neighbor_rain_rate,
    AVG(s.cloud_cover) AS neighbor_cloud_cover_mean,
    MAX(s.cloud_cover) AS neighbor_cloud_cover_max,
    AVG(s.relative_humidity_2m) AS neighbor_relative_humidity_mean,
    MAX(s.relative_humidity_2m) AS neighbor_relative_humidity_max,
    AVG(s.pressure_msl) AS neighbor_pressure_msl_mean,
    MIN(s.pressure_msl) AS neighbor_pressure_msl_min,
    MAX(s.pressure_msl) AS neighbor_pressure_msl_max,
    AVG(s.temperature_2m) AS neighbor_temperature_2m_mean,
    AVG(s.dew_point_2m) AS neighbor_dew_point_2m_mean,
    AVG(s.temperature_2m - s.dew_point_2m) AS neighbor_temperature_dew_point_spread_mean,
    AVG(s.wind_speed_10m) AS neighbor_wind_speed_10m_mean,
    MAX(s.wind_speed_10m) AS neighbor_wind_speed_10m_max,
    AVG(s.precipitation) FILTER (WHERE p.is_row_minus)    AS row_minus_precipitation_mean,
    AVG(s.precipitation) FILTER (WHERE p.is_row_plus)     AS row_plus_precipitation_mean,
    AVG(s.precipitation) FILTER (WHERE p.is_column_minus) AS column_minus_precipitation_mean,
    AVG(s.precipitation) FILTER (WHERE p.is_column_plus)  AS column_plus_precipitation_mean,
    AVG(s.cloud_cover) FILTER (WHERE p.is_row_minus)    AS row_minus_cloud_cover_mean,
    AVG(s.cloud_cover) FILTER (WHERE p.is_row_plus)     AS row_plus_cloud_cover_mean,
    AVG(s.cloud_cover) FILTER (WHERE p.is_column_minus) AS column_minus_cloud_cover_mean,
    AVG(s.cloud_cover) FILTER (WHERE p.is_column_plus)  AS column_plus_cloud_cover_mean
FROM {SOURCE_TABLE} s
JOIN {PAIR_TABLE} p ON p.neighbor_grid_number = s.grid_number
WHERE s.local_forecast_time <= TIMESTAMP '{PANEL_END_LOCAL}'
GROUP BY p.center_grid_number, s.local_forecast_time;

ALTER TABLE om_thailand_prevrun_neighbor_tmp
    ADD CONSTRAINT om_thailand_prevrun_neighbor_tmp_pk PRIMARY KEY (grid_number, local_forecast_time);
ANALYZE om_thailand_prevrun_neighbor_tmp;
"""


# --------------------------------------------------------------------------
# Base lag features + labels, joined to neighbour aggregates
# --------------------------------------------------------------------------

def future_precip_sql():
    return ",\n        ".join(
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
    target_sql = ",\n    ".join(rain_any_target_sql(horizon) for horizon in HORIZONS)
    return f"""
DROP TABLE IF EXISTS {PRECOMPUTE_TABLE};
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
        dew_point_2m,
        relative_humidity_2m,
        precipitation,
        cloud_cover,
        surface_pressure,
        wind_direction_10m,
        wind_speed_10m,
        pressure_msl,
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
    WHERE local_forecast_time <= TIMESTAMP '{PANEL_END_LOCAL}'
    WINDOW w AS (PARTITION BY grid_number ORDER BY local_forecast_time)
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
LEFT JOIN om_thailand_prevrun_neighbor_tmp nf
  ON nf.grid_number = b.grid_number
 AND nf.local_forecast_time = b.local_forecast_time;
"""


CREATE_INDEX_SQL = f"""
ALTER TABLE {PRECOMPUTE_TABLE}
    ADD CONSTRAINT om_thailand_prevrun_precompute_pk PRIMARY KEY (grid_number, local_forecast_time);

CREATE INDEX om_thailand_prevrun_precompute_local_time_idx
    ON {PRECOMPUTE_TABLE} (local_forecast_time);

CREATE INDEX om_thailand_prevrun_precompute_split_ready_idx
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
    SUM(CASE WHEN rain_any_next_6h IS NOT NULL THEN 1 ELSE 0 END) AS labeled_6h_rows,
    SUM(CASE WHEN neighbor_count > 0 THEN 1 ELSE 0 END) AS rows_with_neighbors
FROM {PRECOMPUTE_TABLE};
"""


def rebuild(conn, unlogged=False, keep_temp=False):
    with conn.cursor() as cur:
        cur.execute(SESSION_TUNING_SQL)

        log("Building grid adjacency pair table")
        cur.execute(CREATE_PAIRS_SQL)
        cur.execute(f"SELECT count(*) FROM {PAIR_TABLE};")
        log(f"  {cur.fetchone()[0]:,} adjacency pairs")

        log("Aggregating neighbour features (slowest step)")
        started = time.time()
        cur.execute(CREATE_NEIGHBOR_SQL)
        log(f"  done in {(time.time() - started) / 60:.1f} min")

        log(f"Creating {PRECOMPUTE_TABLE} with lag features and rain-any labels")
        started = time.time()
        cur.execute(create_precompute_sql(unlogged=unlogged))
        log(f"  done in {(time.time() - started) / 60:.1f} min")

        log("Adding primary key and indexes")
        cur.execute(CREATE_INDEX_SQL)

        log("Analyzing precompute table")
        cur.execute(f"ANALYZE {PRECOMPUTE_TABLE};")

        if not keep_temp:
            cur.execute("DROP TABLE IF EXISTS om_thailand_prevrun_neighbor_tmp;")

        cur.execute(SUMMARY_SQL)
        return cur.fetchone()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Build PostgreSQL table "OM_THAILAND_DATA_PRECOMPUTE" from "OM_Thailand_Data" '
            "with baseline lag features, neighbor-grid features, and rain-any labels."
        )
    )
    parser.add_argument(
        "--unlogged",
        action="store_true",
        help="Create as an UNLOGGED table for faster rebuilds. Not crash-safe until rebuilt.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep the intermediate neighbour aggregate table for inspection.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    started = time.time()

    # NOTE: do not use `with connect() as conn`. In psycopg2 the connection context
    # manager wraps a *transaction*, not the connection lifetime: entering it opens a
    # transaction block, so a later `conn.autocommit = True` is reported as True by the
    # driver but ignored by the server. Every CREATE then lands in one multi-hour
    # transaction that stays invisible to other sessions and rolls back entirely on any
    # late failure. Set autocommit on the bare connection instead.
    conn = connect()
    conn.autocommit = True
    try:
        summary = rebuild(conn, unlogged=args.unlogged, keep_temp=args.keep_temp)
    finally:
        conn.close()

    log(
        "Summary: "
        f"rows={summary[0]:,}, grids={summary[1]:,}, "
        f"time={summary[2]} to {summary[3]}, "
        f"labeled_1h={summary[4]:,}, labeled_2h={summary[5]:,}, "
        f"labeled_3h={summary[6]:,}, labeled_6h={summary[7]:,}, "
        f"with_neighbors={summary[8]:,}"
    )
    log(f"Finished in {(time.time() - started) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
