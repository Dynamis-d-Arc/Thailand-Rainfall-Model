"""Thailand counterpart of create_imerg_bkk_data_table.py.

Two schema differences from the Bangkok table, both forced by cell size. Bangkok's
9 km cells are smaller than one IMERG V07 pixel (0.1 deg, ~11 km), so a point sample
at the cell centre *is* the cell. A 25 km Thailand cell spans ~5-6 IMERG pixels, so a
point sample would reintroduce the point-vs-cell mismatch that capped the V5 gauge
gain. Every hour therefore carries both a cell-mean and a cell-max reduction:

    precipitation_mm      cell-average accumulation  -> "how much fell over the cell"
    precipitation_max_mm  wettest-pixel accumulation -> "did it rain anywhere in the cell"

The modelling step picks which one defines the label; storing both means that choice
does not require a refetch. pixel_count records how many unmasked IMERG pixels the
reduction actually saw, so partial-coverage cells (coast, borders) are identifiable.
"""

import os

import psycopg2


DB_CONFIG = {
    "host": os.getenv("PGHOST", "localhost"),
    "port": int(os.getenv("PGPORT", "5432")),
    "dbname": os.getenv("PGDATABASE", "postgres"),
    "user": os.getenv("PGUSER", "postgres"),
    "password": os.getenv("PGPASSWORD", "Pass1234"),
}

TABLE_NAME = '"IMERG_THAILAND_DATA"'
GRID_TABLE = '"Thailand_Grid_25km"'


CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    grid_number integer NOT NULL,
    grid_row integer NOT NULL,
    grid_column integer NOT NULL,
    longitude double precision NOT NULL,
    latitude double precision NOT NULL,
    observation_time timestamp with time zone NOT NULL,
    local_observation_time timestamp without time zone NOT NULL,
    timezone text NOT NULL DEFAULT 'Asia/Bangkok',
    precipitation_mm double precision,
    precipitation_max_mm double precision,
    precipitation_rate_mean_mm_h double precision,
    precipitation_rate_max_mm_h double precision,
    precipitation_rate_first_half_hour_mm_h double precision,
    precipitation_rate_second_half_hour_mm_h double precision,
    pixel_count double precision,
    half_hour_count smallint NOT NULL,
    is_complete_hour boolean GENERATED ALWAYS AS (half_hour_count = 2) STORED,
    product_short_name text NOT NULL DEFAULT 'GPM_3IMERGHH',
    product_version text NOT NULL DEFAULT '07',
    run_type text NOT NULL DEFAULT 'unknown',
    source_service text NOT NULL DEFAULT 'Google Earth Engine / NASA GPM IMERG V07',
    fetched_at timestamp with time zone NOT NULL DEFAULT now(),
    PRIMARY KEY (grid_number, observation_time),
    CONSTRAINT imerg_thailand_grid_fk
        FOREIGN KEY (grid_number) REFERENCES {GRID_TABLE} (grid_number),
    CONSTRAINT imerg_thailand_half_hour_count_check
        CHECK (half_hour_count BETWEEN 0 AND 2),
    CONSTRAINT imerg_thailand_precipitation_check
        CHECK (precipitation_mm IS NULL OR precipitation_mm >= 0),
    CONSTRAINT imerg_thailand_precipitation_max_check
        CHECK (precipitation_max_mm IS NULL OR precipitation_max_mm >= 0),
    CONSTRAINT imerg_thailand_mean_rate_check
        CHECK (precipitation_rate_mean_mm_h IS NULL OR precipitation_rate_mean_mm_h >= 0),
    CONSTRAINT imerg_thailand_max_rate_check
        CHECK (precipitation_rate_max_mm_h IS NULL OR precipitation_rate_max_mm_h >= 0)
);

CREATE INDEX IF NOT EXISTS imerg_thailand_data_observation_time_idx
    ON {TABLE_NAME} (observation_time);

CREATE INDEX IF NOT EXISTS imerg_thailand_data_local_observation_time_idx
    ON {TABLE_NAME} (local_observation_time);

CREATE INDEX IF NOT EXISTS imerg_thailand_data_complete_time_idx
    ON {TABLE_NAME} (observation_time, grid_number)
    WHERE is_complete_hour;
"""


SUMMARY_SQL = f"""
SELECT
    COUNT(*) AS row_count,
    COUNT(DISTINCT grid_number) AS grid_count,
    MIN(observation_time) AS min_time,
    MAX(observation_time) AS max_time,
    COUNT(*) FILTER (WHERE is_complete_hour) AS complete_hour_rows
FROM {TABLE_NAME};
"""


def connect():
    return psycopg2.connect(**DB_CONFIG)


def ensure_table(conn):
    with conn.cursor() as cur:
        cur.execute(CREATE_TABLE_SQL)
    conn.commit()


def fetch_summary(conn):
    with conn.cursor() as cur:
        cur.execute(SUMMARY_SQL)
        return cur.fetchone()


def main():
    conn = connect()
    try:
        ensure_table(conn)
        summary = fetch_summary(conn)
    finally:
        conn.close()

    print(
        f"Created/verified {TABLE_NAME}: rows={summary[0]:,}, "
        f"grids={summary[1]:,}, time={summary[2]}..{summary[3]}, "
        f"complete_hour_rows={summary[4]:,}",
        flush=True,
    )


if __name__ == "__main__":
    main()
