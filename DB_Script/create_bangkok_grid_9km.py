import argparse
import math
import os
import shutil
import subprocess

try:
    import psycopg2
    from psycopg2.extras import execute_values
except ModuleNotFoundError:
    psycopg2 = None
    execute_values = None


DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "postgres",
    "user": "postgres",
    "password": "Pass1234"
}

# Approximate Bangkok administrative bounding box.
BANGKOK_BBOX = {
    "min_lon": 100.327,
    "max_lon": 100.938,
    "min_lat": 13.494,
    "max_lat": 13.955,
}
SPACING_KM = 9.0
TABLE_NAME = '"Bangkok_Grid_9km"'
DEFAULT_PSQL = r"C:\Program Files\PostgreSQL\18\bin\psql.exe"


CREATE_TABLE_SQL = f"""
CREATE EXTENSION IF NOT EXISTS postgis;

DROP VIEW IF EXISTS bangkok_grid_9km_geojson;
DROP VIEW IF EXISTS bangkok_grid_9km;
DROP FUNCTION IF EXISTS nearest_bangkok_grid_9km(double precision, double precision);
DROP TABLE IF EXISTS {TABLE_NAME};

CREATE TABLE {TABLE_NAME} (
    grid_number integer PRIMARY KEY,
    grid_row integer NOT NULL,
    grid_column integer NOT NULL,
    longitude double precision NOT NULL,
    latitude double precision NOT NULL,
    approximate_resolution_km double precision NOT NULL DEFAULT 9.0,
    geom geometry(Point, 4326)
      GENERATED ALWAYS AS (ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)) STORED,
    geog geography(Point, 4326)
      GENERATED ALWAYS AS (ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)::geography) STORED,
    source text NOT NULL DEFAULT 'Bangkok bounding-box coordinate grid with approximately 9 km spacing, aligned to ECMWF HRES nominal resolution',
    created_at timestamp with time zone NOT NULL DEFAULT now(),
    UNIQUE (grid_row, grid_column),
    UNIQUE (longitude, latitude)
);

CREATE INDEX bangkok_grid_9km_geom_gix
  ON {TABLE_NAME} USING gist (geom);

CREATE INDEX bangkok_grid_9km_geog_gix
  ON {TABLE_NAME} USING gist (geog);

CREATE VIEW bangkok_grid_9km AS
SELECT
    grid_number,
    grid_row,
    grid_column,
    longitude,
    latitude,
    approximate_resolution_km,
    geom,
    geog,
    source,
    created_at
FROM {TABLE_NAME};

CREATE VIEW bangkok_grid_9km_geojson AS
SELECT
    grid_number,
    grid_row,
    grid_column,
    longitude,
    latitude,
    approximate_resolution_km,
    ST_AsGeoJSON(geom)::jsonb AS geometry
FROM {TABLE_NAME}
ORDER BY grid_number;

CREATE OR REPLACE FUNCTION nearest_bangkok_grid_9km(
    input_longitude double precision,
    input_latitude double precision
)
RETURNS TABLE (
    grid_number integer,
    longitude double precision,
    latitude double precision,
    distance_m double precision
)
LANGUAGE sql
STABLE
AS $$
    SELECT
        g.grid_number,
        g.longitude,
        g.latitude,
        ST_Distance(
            g.geog,
            ST_SetSRID(ST_MakePoint(input_longitude, input_latitude), 4326)::geography
        ) AS distance_m
    FROM {TABLE_NAME} g
    ORDER BY g.geog <-> ST_SetSRID(ST_MakePoint(input_longitude, input_latitude), 4326)::geography
    LIMIT 1;
$$;
"""


INSERT_SQL = f"""
INSERT INTO {TABLE_NAME} (
    grid_number,
    grid_row,
    grid_column,
    longitude,
    latitude
)
VALUES %s;
"""


SUMMARY_SQL = f"""
WITH nearest AS (
    SELECT
        g1.grid_number,
        MIN(ST_Distance(g1.geog, g2.geog)) AS nearest_m
    FROM {TABLE_NAME} g1
    JOIN {TABLE_NAME} g2
      ON g1.grid_number <> g2.grid_number
     AND ST_DWithin(g1.geog, g2.geog, 15000)
    GROUP BY g1.grid_number
)
SELECT
    COUNT(*) AS grid_count,
    ROUND(MIN(latitude)::numeric, 6) AS min_latitude,
    ROUND(MAX(latitude)::numeric, 6) AS max_latitude,
    ROUND(MIN(longitude)::numeric, 6) AS min_longitude,
    ROUND(MAX(longitude)::numeric, 6) AS max_longitude,
    ROUND(AVG(nearest_m)::numeric, 2) AS avg_nearest_spacing_m,
    ROUND(MIN(nearest_m)::numeric, 2) AS min_nearest_spacing_m,
    ROUND(MAX(nearest_m)::numeric, 2) AS max_nearest_spacing_m
FROM {TABLE_NAME}
LEFT JOIN nearest USING (grid_number);
"""


def connect():
    if psycopg2 is None:
        raise RuntimeError("psycopg2 is not installed. Run with --method psql instead.")
    
    # Use the password from DB_CONFIG if available, otherwise check env
    password = DB_CONFIG.get("password") or os.getenv("PGPASSWORD")
    if not password:
        raise RuntimeError("Set PGPASSWORD before running this script or update DB_CONFIG.")
        
    return psycopg2.connect(**DB_CONFIG)

def build_grid_rows(bbox, spacing_km):
    mid_lat = (bbox["min_lat"] + bbox["max_lat"]) / 2.0
    lat_step = spacing_km / 110.574
    lon_step = spacing_km / (111.320 * math.cos(math.radians(mid_lat)))

    latitudes = coordinate_axis(bbox["min_lat"], bbox["max_lat"], lat_step)
    longitudes = coordinate_axis(bbox["min_lon"], bbox["max_lon"], lon_step)

    rows = []
    grid_number = 1
    for row_index, latitude in enumerate(latitudes, start=1):
        for column_index, longitude in enumerate(longitudes, start=1):
            rows.append(
                (
                    grid_number,
                    row_index,
                    column_index,
                    round(longitude, 6),
                    round(latitude, 6),
                )
            )
            grid_number += 1
    return rows, lat_step, lon_step


def coordinate_axis(min_value, max_value, step):
    values = []
    current = min_value
    while current <= max_value + 1e-12:
        values.append(current)
        current += step
    if not math.isclose(values[-1], max_value) and max_value - values[-1] > step * 0.35:
        values.append(max_value)
    return values


def create_grid(conn, rows):
    with conn.cursor() as cur:
        cur.execute(CREATE_TABLE_SQL)
        execute_values(cur, INSERT_SQL, rows, page_size=1000)
        cur.execute(f"ANALYZE {TABLE_NAME}")
    conn.commit()


def rows_values_sql(rows):
    return ",\n".join(
        (
            f"    ({grid_number}, {grid_row}, {grid_column}, "
            f"{longitude:.6f}, {latitude:.6f})"
        )
        for grid_number, grid_row, grid_column, longitude, latitude in rows
    )


def full_grid_sql(rows):
    insert_sql = f"""
INSERT INTO {TABLE_NAME} (
    grid_number,
    grid_row,
    grid_column,
    longitude,
    latitude
)
VALUES
{rows_values_sql(rows)};

ANALYZE {TABLE_NAME};
"""
    return CREATE_TABLE_SQL + insert_sql


def resolve_psql_path(psql_path):
    candidates = [
        psql_path,
        os.getenv("PSQL"),
        DEFAULT_PSQL,
        shutil.which("psql"),
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    raise RuntimeError(
        "Could not find psql.exe. Pass --psql-path or set the PSQL environment variable."
    )


def run_psql(sql, psql_path, extra_args=None):
    if not os.getenv("PGPASSWORD"):
        raise RuntimeError("Set PGPASSWORD before running this script.")

    command = [
        resolve_psql_path(psql_path),
        "-v",
        "ON_ERROR_STOP=1",
        "-h",
        DB_CONFIG["host"],
        "-p",
        str(DB_CONFIG["port"]),
        "-U",
        DB_CONFIG["user"],
        "-d",
        DB_CONFIG["dbname"],
    ]
    if extra_args:
        command.extend(extra_args)

    return subprocess.run(
        command,
        input=sql,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=True,
    ).stdout


def create_grid_with_psql(rows, psql_path):
    run_psql(full_grid_sql(rows), psql_path)


def print_summary_with_psql(row_count, lat_step, lon_step, psql_path):
    output = run_psql(SUMMARY_SQL, psql_path, ["-At", "-F", ","]).strip()
    fields = output.split(",")
    print_summary_row(fields, row_count, lat_step, lon_step)


def print_summary(conn, row_count, lat_step, lon_step):
    with conn.cursor() as cur:
        cur.execute(SUMMARY_SQL)
        summary = cur.fetchone()

    print_summary_row(summary, row_count, lat_step, lon_step)


def print_summary_row(summary, row_count, lat_step, lon_step):
    grid_count = int(summary[0])
    print(
        'Created PostgreSQL table "Bangkok_Grid_9km": '
        f"points={grid_count:,}, "
        f"lat={summary[1]}..{summary[2]}, lon={summary[3]}..{summary[4]}, "
        f"nearest_spacing_m avg/min/max={summary[5]}/{summary[6]}/{summary[7]}, "
        f"degree_steps lat={lat_step:.6f}, lon={lon_step:.6f}, "
        f"generated_rows={row_count:,}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description='Create PostgreSQL table "Bangkok_Grid_9km" with approximately 9 km coordinate spacing.'
    )
    parser.add_argument("--min-lon", type=float, default=BANGKOK_BBOX["min_lon"])
    parser.add_argument("--max-lon", type=float, default=BANGKOK_BBOX["max_lon"])
    parser.add_argument("--min-lat", type=float, default=BANGKOK_BBOX["min_lat"])
    parser.add_argument("--max-lat", type=float, default=BANGKOK_BBOX["max_lat"])
    parser.add_argument("--spacing-km", type=float, default=SPACING_KM)
    parser.add_argument(
        "--method",
        choices=["auto", "psycopg2", "psql"],
        default="auto",
        help="Database execution method. Defaults to psycopg2 when available, otherwise psql.",
    )
    parser.add_argument(
        "--psql-path",
        default=None,
        help="Path to psql.exe. Defaults to PSQL env var, PostgreSQL 18 default path, or PATH lookup.",
    )
    args = parser.parse_args()

    bbox = {
        "min_lon": args.min_lon,
        "max_lon": args.max_lon,
        "min_lat": args.min_lat,
        "max_lat": args.max_lat,
    }
    rows, lat_step, lon_step = build_grid_rows(bbox, args.spacing_km)

    method = args.method
    if method == "auto":
        method = "psycopg2" if psycopg2 is not None else "psql"

    if method == "psql":
        create_grid_with_psql(rows, args.psql_path)
        print_summary_with_psql(len(rows), lat_step, lon_step, args.psql_path)
        return

    with connect() as conn:
        create_grid(conn, rows)
        print_summary(conn, len(rows), lat_step, lon_step)


if __name__ == "__main__":
    main()
