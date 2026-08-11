import argparse
import os
import time
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import requests

try:
    import psycopg2
    from psycopg2.extras import execute_values
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "This script requires psycopg2. Install it with: pip install psycopg2-binary"
    ) from exc


DB_CONFIG = {
    "host": os.getenv("PGHOST", "localhost"),
    "port": int(os.getenv("PGPORT", "5432")),
    "dbname": os.getenv("PGDATABASE", "postgres"),
    "user": os.getenv("PGUSER", "postgres"),
    "password": os.getenv("PGPASSWORD", "Pass1234"),
}

GRID_TABLE = '"Thailand_Grid_25km"'
OUTPUT_TABLE = '"OM_Thailand_PrevRun_Data"'
TIMEZONE_NAME = "Asia/Bangkok"
LOCAL_TIMEZONE = ZoneInfo(TIMEZONE_NAME)
HISTORICAL_WEATHER_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
PREV_SUFFIX = "_previous_day1"
FORECAST_MODEL = "ecmwf_ifs"

HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "pressure_msl",
    "surface_pressure",
    "dew_point_2m",
    "precipitation",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
]


CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {OUTPUT_TABLE} (
    grid_number integer NOT NULL,
    grid_row integer NOT NULL,
    grid_column integer NOT NULL,
    longitude double precision NOT NULL,
    latitude double precision NOT NULL,
    forecast_time timestamp with time zone NOT NULL,
    local_forecast_time timestamp without time zone NOT NULL,
    timezone text NOT NULL DEFAULT '{TIMEZONE_NAME}',
    temperature_2m double precision,
    relative_humidity_2m double precision,
    pressure_msl double precision,
    surface_pressure double precision,
    dew_point_2m double precision,
    precipitation double precision,
    cloud_cover double precision,
    wind_speed_10m double precision,
    wind_direction_10m double precision,
    source_api text NOT NULL,
    fetched_at timestamp with time zone NOT NULL DEFAULT now(),
    PRIMARY KEY (grid_number, forecast_time)
);

CREATE INDEX IF NOT EXISTS om_thailand_prevrun_data_forecast_time_idx
  ON {OUTPUT_TABLE} (forecast_time);

CREATE INDEX IF NOT EXISTS om_thailand_prevrun_data_grid_time_idx
  ON {OUTPUT_TABLE} (grid_number, forecast_time);
"""


INSERT_SQL = f"""
INSERT INTO {OUTPUT_TABLE} (
    grid_number,
    grid_row,
    grid_column,
    longitude,
    latitude,
    forecast_time,
    local_forecast_time,
    timezone,
    temperature_2m,
    relative_humidity_2m,
    pressure_msl,
    surface_pressure,
    dew_point_2m,
    precipitation,
    cloud_cover,
    wind_speed_10m,
    wind_direction_10m,
    source_api
)
VALUES %s
ON CONFLICT (grid_number, forecast_time) DO UPDATE SET
    grid_row = EXCLUDED.grid_row,
    grid_column = EXCLUDED.grid_column,
    longitude = EXCLUDED.longitude,
    latitude = EXCLUDED.latitude,
    local_forecast_time = EXCLUDED.local_forecast_time,
    timezone = EXCLUDED.timezone,
    temperature_2m = EXCLUDED.temperature_2m,
    relative_humidity_2m = EXCLUDED.relative_humidity_2m,
    pressure_msl = EXCLUDED.pressure_msl,
    surface_pressure = EXCLUDED.surface_pressure,
    dew_point_2m = EXCLUDED.dew_point_2m,
    precipitation = EXCLUDED.precipitation,
    cloud_cover = EXCLUDED.cloud_cover,
    wind_speed_10m = EXCLUDED.wind_speed_10m,
    wind_direction_10m = EXCLUDED.wind_direction_10m,
    source_api = EXCLUDED.source_api,
    fetched_at = now();
"""


SUMMARY_SQL = f"""
SELECT
    COUNT(*) AS row_count,
    COUNT(DISTINCT grid_number) AS grid_count,
    MIN(local_forecast_time) AS min_local_time,
    MAX(local_forecast_time) AS max_local_time
FROM {OUTPUT_TABLE};
"""


def log(message):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def parse_date(value):
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid date: {value}. Use YYYY-MM-DD.") from exc


def default_end_date():
    return datetime.now(LOCAL_TIMEZONE).date() - timedelta(days=1)


def default_start_date():
    return default_end_date() - timedelta(days=364)


def connect():
    return psycopg2.connect(**DB_CONFIG)


def ensure_output_table(conn):
    with conn.cursor() as cur:
        cur.execute(CREATE_TABLE_SQL)
    conn.commit()


def replace_output_table(conn):
    with conn.cursor() as cur:
        cur.execute(f"TRUNCATE TABLE {OUTPUT_TABLE};")
    conn.commit()


def fetch_grid_points(conn):
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT grid_number, grid_row, grid_column, longitude, latitude
            FROM {GRID_TABLE}
            ORDER BY grid_number;
            """
        )
        return [
            {
                "grid_number": row[0],
                "grid_row": row[1],
                "grid_column": row[2],
                "longitude": float(row[3]),
                "latitude": float(row[4]),
            }
            for row in cur.fetchall()
        ]


def chunked(items, size):
    for index in range(0, len(items), size):
        yield items[index : index + size]


def date_chunks(start_date, end_date, days_per_chunk):
    current = start_date
    while current <= end_date:
        chunk_end = min(current + timedelta(days=days_per_chunk - 1), end_date)
        yield current, chunk_end
        current = chunk_end + timedelta(days=1)


def request_params(points, start_date, end_date):
    return {
        "latitude": ",".join(str(point["latitude"]) for point in points),
        "longitude": ",".join(str(point["longitude"]) for point in points),
        "hourly": ",".join(v + PREV_SUFFIX for v in HOURLY_VARIABLES),
        "timezone": TIMEZONE_NAME,
        "models": FORECAST_MODEL,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
    }


def fetch_weather_batch(points, start_date, end_date, timeout, retries, retry_sleep):
    params = request_params(points, start_date, end_date)
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(HISTORICAL_WEATHER_URL, params=params, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict):
                payload = [payload]
            if len(payload) != len(points):
                raise RuntimeError(
                    f"Open-Meteo returned {len(payload)} payloads for {len(points)} points."
                )
            return payload, "open_meteo_previous_run_day1_" + FORECAST_MODEL
        except (requests.RequestException, RuntimeError) as exc:
            last_error = exc
            if attempt == retries:
                break
            log(
                f"Request failed for {start_date}..{end_date}; "
                f"retrying in {retry_sleep}s ({attempt}/{retries})"
            )
            time.sleep(retry_sleep)

    raise RuntimeError(f"Open-Meteo request failed after {retries} attempts: {last_error}")


def parse_open_meteo_local_time(value):
    local_dt = datetime.fromisoformat(value)
    aware_local_dt = local_dt.replace(tzinfo=LOCAL_TIMEZONE)
    return aware_local_dt.astimezone(timezone.utc), local_dt


def value_at(hourly, variable, index):
    values = hourly.get(variable + PREV_SUFFIX)
    if values is None or index >= len(values):
        return None
    return values[index]


def weather_rows(points, payloads, source_api):
    rows = []
    for point, payload in zip(points, payloads):
        hourly = payload.get("hourly") or {}
        times = hourly.get("time") or []
        for index, time_value in enumerate(times):
            forecast_time, local_forecast_time = parse_open_meteo_local_time(time_value)
            values = [value_at(hourly, variable, index) for variable in HOURLY_VARIABLES]
            rows.append(
                (
                    point["grid_number"],
                    point["grid_row"],
                    point["grid_column"],
                    point["longitude"],
                    point["latitude"],
                    forecast_time,
                    local_forecast_time,
                    TIMEZONE_NAME,
                    *values,
                    source_api,
                )
            )
    return rows


def insert_rows(conn, rows):
    if not rows:
        return 0
    with conn.cursor() as cur:
        execute_values(cur, INSERT_SQL, rows, page_size=5000)
    conn.commit()
    return len(rows)


def existing_row_count(conn, points, start_date, end_date):
    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date + timedelta(days=1), datetime.min.time())
    grid_numbers = [point["grid_number"] for point in points]
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT COUNT(*)
            FROM {OUTPUT_TABLE}
            WHERE grid_number = ANY(%s)
              AND local_forecast_time >= %s
              AND local_forecast_time < %s;
            """,
            (grid_numbers, start_dt, end_dt),
        )
        return cur.fetchone()[0]


def expected_row_count(points, start_date, end_date):
    days = (end_date - start_date).days + 1
    return len(points) * days * 24


def fetch_summary(conn):
    with conn.cursor() as cur:
        cur.execute(SUMMARY_SQL)
        return cur.fetchone()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Fetch Open-Meteo PREVIOUS-RUNS values (ecmwf_ifs, previous_day1 = +24h lead) for "Thailand_Grid_25km" '
            'and upsert it into PostgreSQL table "OM_Thailand_PrevRun_Data".'
        )
    )
    parser.add_argument("--start-date", type=parse_date, default=default_start_date())
    parser.add_argument("--end-date", type=parse_date, default=default_end_date())
    parser.add_argument("--days-per-request", type=int, default=31)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument("--sleep-seconds", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument(
        "--skip-complete-batches",
        action="store_true",
        help="Skip point/date batches that already have all expected hourly rows.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.end_date < args.start_date:
        raise RuntimeError("--end-date must be on or after --start-date.")
    if args.days_per_request < 1:
        raise RuntimeError("--days-per-request must be at least 1.")
    if args.batch_size < 1:
        raise RuntimeError("--batch-size must be at least 1.")

    with connect() as conn:
        ensure_output_table(conn)
        points = fetch_grid_points(conn)
        if not points:
            raise RuntimeError(f"No coordinates found in {GRID_TABLE}.")

        point_batches = list(chunked(points, args.batch_size))
        date_ranges = list(date_chunks(args.start_date, args.end_date, args.days_per_request))

        log(
            f"Grid points={len(points):,}; point batches={len(point_batches):,}; "
            f"date chunks={len(date_ranges):,}; range={args.start_date}..{args.end_date}"
        )

        if args.dry_run:
            estimated_rows = len(points) * len(date_ranges) * args.days_per_request * 24
            log(f"Estimated max rows before final partial chunk adjustment: {estimated_rows:,}")
            return

        if args.replace:
            log(f"Clearing existing rows from {OUTPUT_TABLE}")
            replace_output_table(conn)

        total_written = 0
        total_requests = 0
        for start_date, end_date in date_ranges:
            log(f"Fetching {start_date}..{end_date}")
            for batch_index, point_batch in enumerate(point_batches, start=1):
                if args.skip_complete_batches:
                    existing_rows = existing_row_count(conn, point_batch, start_date, end_date)
                    expected_rows = expected_row_count(point_batch, start_date, end_date)
                    if existing_rows >= expected_rows:
                        log(
                            f"  batch {batch_index}/{len(point_batches)}: "
                            f"skipped {existing_rows:,}/{expected_rows:,} existing rows"
                        )
                        continue

                payloads, source_api = fetch_weather_batch(
                    point_batch,
                    start_date,
                    end_date,
                    args.timeout,
                    args.retries,
                    args.retry_sleep,
                )
                rows = weather_rows(point_batch, payloads, source_api)
                written = insert_rows(conn, rows)
                total_written += written
                total_requests += 1
                log(
                    f"  batch {batch_index}/{len(point_batches)}: "
                    f"upserted {written:,} rows"
                )
                time.sleep(args.sleep_seconds)

        summary = fetch_summary(conn)

    log(
        f"Finished. API requests={total_requests:,}; rows inserted/updated={total_written:,}."
    )
    print(
        f'{OUTPUT_TABLE} summary: rows={summary[0]:,}, grids={summary[1]:,}, '
        f"local_time={summary[2]}..{summary[3]}"
    )


if __name__ == "__main__":
    main()
