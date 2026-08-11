import argparse
import json
import math
import os
from pathlib import Path

try:
    import psycopg2
    from psycopg2.extras import execute_values
except ModuleNotFoundError:
    psycopg2 = None
    execute_values = None


DB_CONFIG = {
    "host": os.getenv("PGHOST", "localhost"),
    "port": int(os.getenv("PGPORT", "5432")),
    "dbname": os.getenv("PGDATABASE", "postgres"),
    "user": os.getenv("PGUSER", "postgres"),
    "password": os.getenv("PGPASSWORD", "Pass1234"),
}

BOUNDARY_PATH = Path(__file__).with_name("thailand_boundary.geojson")
SPACING_KM = 25.0
TABLE_NAME = '"Thailand_Grid_25km"'


CREATE_TABLE_SQL = f"""
CREATE EXTENSION IF NOT EXISTS postgis;

DROP VIEW IF EXISTS thailand_grid_25km_geojson;
DROP VIEW IF EXISTS thailand_grid_25km;
DROP FUNCTION IF EXISTS nearest_thailand_grid_25km(double precision, double precision);
DROP TABLE IF EXISTS {TABLE_NAME};

CREATE TABLE {TABLE_NAME} (
    grid_number integer PRIMARY KEY,
    grid_row integer NOT NULL,
    grid_column integer NOT NULL,
    longitude double precision NOT NULL,
    latitude double precision NOT NULL,
    approximate_resolution_km double precision NOT NULL DEFAULT 25.0,
    geom geometry(Point, 4326)
      GENERATED ALWAYS AS (ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)) STORED,
    geog geography(Point, 4326)
      GENERATED ALWAYS AS (ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)::geography) STORED,
    source text NOT NULL DEFAULT 'Thailand land-only coordinate grid with approximately 25 km spacing, clipped to Thailand boundary',
    created_at timestamp with time zone NOT NULL DEFAULT now(),
    UNIQUE (grid_row, grid_column),
    UNIQUE (longitude, latitude)
);

CREATE INDEX thailand_grid_25km_geom_gix
  ON {TABLE_NAME} USING gist (geom);

CREATE INDEX thailand_grid_25km_geog_gix
  ON {TABLE_NAME} USING gist (geog);

CREATE VIEW thailand_grid_25km AS
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

CREATE VIEW thailand_grid_25km_geojson AS
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

CREATE OR REPLACE FUNCTION nearest_thailand_grid_25km(
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
     AND ST_DWithin(g1.geog, g2.geog, 40000)
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
        raise RuntimeError("This script requires psycopg2. Install psycopg2-binary in the active Python environment.")
    return psycopg2.connect(**DB_CONFIG)


def coordinate_axis(min_value, max_value, step):
    values = []
    current = min_value
    while current <= max_value + 1e-12:
        values.append(current)
        current += step
    if values and not math.isclose(values[-1], max_value) and max_value - values[-1] > step * 0.35:
        values.append(max_value)
    return values


def load_boundary_features(path):
    with path.open("r", encoding="utf-8") as boundary_file:
        geojson = json.load(boundary_file)

    features = geojson.get("features", [])
    geometries = [json.dumps(feature["geometry"]) for feature in features if feature.get("geometry")]
    if not geometries:
        raise RuntimeError(f"No geometries found in {path}.")
    return geometries


def load_boundary(conn, geometry_json_values):
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS thailand_land_boundary;")
        cur.execute(
            """
            CREATE TEMP TABLE thailand_boundary_features (
                geometry_json text NOT NULL
            ) ON COMMIT DROP;
            """
        )
        execute_values(
            cur,
            "INSERT INTO thailand_boundary_features (geometry_json) VALUES %s;",
            [(geometry_json,) for geometry_json in geometry_json_values],
            page_size=100,
        )
        cur.execute(
            """
            CREATE TABLE thailand_land_boundary AS
            SELECT
                1 AS id,
                ST_Multi(
                    ST_UnaryUnion(
                        ST_Collect(
                            ST_MakeValid(ST_SetSRID(ST_GeomFromGeoJSON(geometry_json), 4326))
                        )
                    )
                )::geometry(MultiPolygon, 4326) AS geom
            FROM thailand_boundary_features;
            """
        )
        cur.execute(
            """
            CREATE INDEX thailand_land_boundary_geom_gix
              ON thailand_land_boundary USING gist (geom);
            """
        )
        cur.execute("ANALYZE thailand_land_boundary;")
    conn.commit()


def fetch_boundary_bbox(conn):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                ST_XMin(ST_Extent(geom)),
                ST_XMax(ST_Extent(geom)),
                ST_YMin(ST_Extent(geom)),
                ST_YMax(ST_Extent(geom))
            FROM thailand_land_boundary;
            """
        )
        min_lon, max_lon, min_lat, max_lat = cur.fetchone()
    return {
        "min_lon": float(min_lon),
        "max_lon": float(max_lon),
        "min_lat": float(min_lat),
        "max_lat": float(max_lat),
    }


def build_candidate_rows(bbox, spacing_km):
    mid_lat = (bbox["min_lat"] + bbox["max_lat"]) / 2.0
    lat_step = spacing_km / 110.574
    lon_step = spacing_km / (111.320 * math.cos(math.radians(mid_lat)))

    latitudes = coordinate_axis(bbox["min_lat"], bbox["max_lat"], lat_step)
    longitudes = coordinate_axis(bbox["min_lon"], bbox["max_lon"], lon_step)

    rows = []
    for row_index, latitude in enumerate(latitudes, start=1):
        for column_index, longitude in enumerate(longitudes, start=1):
            rows.append((row_index, column_index, round(longitude, 6), round(latitude, 6)))
    return rows, lat_step, lon_step


def filter_land_rows(conn, candidate_rows):
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TEMP TABLE thailand_grid_25km_candidates (
                grid_row integer NOT NULL,
                grid_column integer NOT NULL,
                longitude double precision NOT NULL,
                latitude double precision NOT NULL
            ) ON COMMIT DROP;
            """
        )
        execute_values(
            cur,
            """
            INSERT INTO thailand_grid_25km_candidates (
                grid_row,
                grid_column,
                longitude,
                latitude
            )
            VALUES %s;
            """,
            candidate_rows,
            page_size=1000,
        )
        cur.execute(
            """
            SELECT
                row_number() OVER (ORDER BY c.grid_row, c.grid_column)::integer AS grid_number,
                c.grid_row,
                c.grid_column,
                c.longitude,
                c.latitude
            FROM thailand_grid_25km_candidates c
            JOIN thailand_land_boundary b
              ON ST_Covers(b.geom, ST_SetSRID(ST_MakePoint(c.longitude, c.latitude), 4326))
            ORDER BY c.grid_row, c.grid_column;
            """
        )
        return cur.fetchall()


def create_grid(conn, rows):
    with conn.cursor() as cur:
        cur.execute(CREATE_TABLE_SQL)
        execute_values(cur, INSERT_SQL, rows, page_size=1000)
        cur.execute(f"ANALYZE {TABLE_NAME};")
    conn.commit()


def print_summary(conn, row_count, candidate_count, lat_step, lon_step):
    with conn.cursor() as cur:
        cur.execute(SUMMARY_SQL)
        summary = cur.fetchone()

    print(
        'Created PostgreSQL table "Thailand_Grid_25km": '
        f"points={int(summary[0]):,}, "
        f"lat={summary[1]}..{summary[2]}, lon={summary[3]}..{summary[4]}, "
        f"nearest_spacing_m avg/min/max={summary[5]}/{summary[6]}/{summary[7]}, "
        f"degree_steps lat={lat_step:.6f}, lon={lon_step:.6f}, "
        f"candidate_rows={candidate_count:,}, land_rows={row_count:,}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description='Create PostgreSQL table "Thailand_Grid_25km" with approximately 25 km land-only coordinate spacing.'
    )
    parser.add_argument(
        "--boundary-path",
        type=Path,
        default=BOUNDARY_PATH,
        help="GeoJSON file containing Thailand land boundary polygons.",
    )
    parser.add_argument("--spacing-km", type=float, default=SPACING_KM)
    args = parser.parse_args()

    geometry_json_values = load_boundary_features(args.boundary_path)
    with connect() as conn:
        load_boundary(conn, geometry_json_values)
        bbox = fetch_boundary_bbox(conn)
        candidate_rows, lat_step, lon_step = build_candidate_rows(bbox, args.spacing_km)
        land_rows = filter_land_rows(conn, candidate_rows)
        create_grid(conn, land_rows)
        print_summary(conn, len(land_rows), len(candidate_rows), lat_step, lon_step)


if __name__ == "__main__":
    main()
