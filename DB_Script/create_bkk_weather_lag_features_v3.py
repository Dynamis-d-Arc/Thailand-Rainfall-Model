import argparse
import os
from datetime import datetime

import psycopg2


DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "postgres",
    "user": "postgres",
    "password": os.getenv("PGPASSWORD", "Pass1234"),
}

SOURCE_TABLE = '"New_BKK_Data"'
OUTPUT_TABLE = "bkk_weather_lag_features_v3"
LAG_FEATURE_COLUMNS = [
    "cloud_cover",
    "relative_humidity_2m",
    "dew_point_2m",
]
LAG_HOURS = [1, 2, 3, 4, 5, 6, 24, 48]
RAIN_TARGET_HOURS = list(range(1, 49))
RAIN_THRESHOLD_MM = 0.1
BASE_COLUMNS = [
    "grid_number",
    "grid_row",
    "grid_column",
    "longitude",
    "latitude",
    "forecast_time",
    "temperature_2m",
    "dew_point_2m",
    "relative_humidity_2m",
    "precipitation",
    "cloud_cover",
    "surface_pressure",
    "wind_direction_10m",
    "wind_speed_10m",
    "pressure_msl",
    "model",
    "fetched_at",
]


def log(message):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def connect():
    return psycopg2.connect(**DB_CONFIG)


def lag_select_sql():
    expressions = []
    for column in LAG_FEATURE_COLUMNS:
        for lag_hour in LAG_HOURS:
            expressions.append(
                f"    LAG({column}, {lag_hour}) OVER w AS {column}_lag_{lag_hour}h"
            )
    return ",\n".join(expressions)


def rain_target_select_sql():
    expressions = []
    for target_hour in RAIN_TARGET_HOURS:
        next_precipitation = f"LEAD(precipitation, {target_hour}) OVER w"
        expressions.append(
            "    CASE\n"
            f"        WHEN {next_precipitation} IS NULL THEN NULL\n"
            f"        WHEN {next_precipitation} >= {RAIN_THRESHOLD_MM} THEN 1\n"
            "        ELSE 0\n"
            f"    END AS rain_next_{target_hour}h"
        )
    return ",\n".join(expressions)


def not_null_filter_sql():
    required_columns = [
        "temperature_2m",
        "dew_point_2m",
        "relative_humidity_2m",
        "precipitation",
        "cloud_cover",
        "surface_pressure",
        "wind_direction_10m",
        "wind_speed_10m",
        "pressure_msl",
    ]
    return "\n      AND ".join(f"{column} IS NOT NULL" for column in required_columns)


def create_table_sql(output_table):
    base_select = ",\n    ".join(BASE_COLUMNS)
    return f"""
DROP TABLE IF EXISTS {output_table};

CREATE TABLE {output_table} AS
SELECT
    {base_select},
{lag_select_sql()},
{rain_target_select_sql()}
FROM {SOURCE_TABLE}
WHERE {not_null_filter_sql()}
WINDOW w AS (
    PARTITION BY grid_number
    ORDER BY forecast_time
);

ALTER TABLE {output_table}
    ADD PRIMARY KEY (grid_number, forecast_time);

CREATE INDEX {output_table}_forecast_time_idx
    ON {output_table} (forecast_time);

CREATE INDEX {output_table}_grid_time_idx
    ON {output_table} (grid_number, forecast_time);

ANALYZE {output_table};
"""


def summary_sql(output_table):
    lag_null_checks = ",\n    ".join(
        f"COUNT(*) FILTER (WHERE {column}_lag_{lag_hour}h IS NULL) "
        f"AS missing_{column}_lag_{lag_hour}h"
        for column in LAG_FEATURE_COLUMNS
        for lag_hour in LAG_HOURS
    )
    rain_target_null_checks = ",\n    ".join(
        f"COUNT(*) FILTER (WHERE rain_next_{target_hour}h IS NULL) "
        f"AS missing_rain_next_{target_hour}h"
        for target_hour in [1, 2, 3, 4, 5, 6, 24, 48]
    )
    return f"""
SELECT
    COUNT(*) AS row_count,
    COUNT(DISTINCT grid_number) AS grid_count,
    MIN(forecast_time) AS min_forecast_time,
    MAX(forecast_time) AS max_forecast_time,
    {lag_null_checks},
    {rain_target_null_checks}
FROM {output_table};
"""


def create_lag_feature_table(conn, output_table):
    sql = create_table_sql(output_table)
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()


def fetch_summary(conn, output_table):
    with conn.cursor() as cur:
        cur.execute(summary_sql(output_table))
        columns = [description[0] for description in cur.description]
        values = cur.fetchone()
    return dict(zip(columns, values))


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Create a PostgreSQL lag-feature table from "New_BKK_Data" without '
            "CAPE or convective inhibition."
        )
    )
    parser.add_argument(
        "--output-table",
        default=OUTPUT_TABLE,
        help=f"Output table name. Defaults to {OUTPUT_TABLE}.",
    )
    parser.add_argument(
        "--print-sql",
        action="store_true",
        help="Print the generated SQL and exit without creating the table.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.print_sql:
        print(create_table_sql(args.output_table))
        return

    log(f"Creating lag feature table: {args.output_table}")
    with connect() as conn:
        create_lag_feature_table(conn, args.output_table)
        summary = fetch_summary(conn, args.output_table)

    log("Created lag feature table.")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
