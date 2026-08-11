import argparse
import os
import sys
import tempfile
from collections import defaultdict
from contextlib import nullcontext
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import psycopg2
from psycopg2.extras import execute_values


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_SCRIPT_DIR = PROJECT_ROOT / "DB_Script"
if str(DB_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(DB_SCRIPT_DIR))

from create_imerg_bkk_data_table import CREATE_TABLE_SQL  # noqa: E402


DB_CONFIG = {
    "host": os.getenv("PGHOST", "localhost"),
    "port": int(os.getenv("PGPORT", "5432")),
    "dbname": os.getenv("PGDATABASE", "postgres"),
    "user": os.getenv("PGUSER", "postgres"),
    "password": os.getenv("PGPASSWORD", "Pass1234"),
}

GRID_TABLE = '"Bangkok_Grid_9km"'
OM_TABLE = '"OM_BKK_DATA"'
OUTPUT_TABLE = '"IMERG_BKK_DATA"'
PRODUCT_SHORT_NAME = "GPM_3IMERGHH"
PRODUCT_VERSION = "07"
RUN_TYPE = "Final"
SOURCE_SERVICE = "NASA Earthdata Harmony / GES DISC"
TIMEZONE_NAME = "Asia/Bangkok"
LOCAL_TIMEZONE = ZoneInfo(TIMEZONE_NAME)
FINAL_RUN_SAFETY_DELAY_DAYS = 120
DEFAULT_VARIABLES = [
    "Grid/precipitation",
    "Grid/randomError",
    "Grid/probabilityLiquidPrecipitation",
    "Grid/precipitationQualityIndex",
]


INSERT_SQL = f"""
INSERT INTO {OUTPUT_TABLE} (
    grid_number,
    grid_row,
    grid_column,
    longitude,
    latitude,
    imerg_longitude,
    imerg_latitude,
    observation_time,
    local_observation_time,
    timezone,
    precipitation_mm,
    precipitation_rate_mean_mm_h,
    precipitation_rate_max_mm_h,
    precipitation_rate_first_half_hour_mm_h,
    precipitation_rate_second_half_hour_mm_h,
    random_error_mean_mm_h,
    probability_liquid_precipitation_mean_pct,
    precipitation_quality_index_mean,
    half_hour_count,
    product_short_name,
    product_version,
    run_type,
    source_service,
    source_granules
)
VALUES %s
ON CONFLICT (grid_number, observation_time) DO UPDATE SET
    grid_row = EXCLUDED.grid_row,
    grid_column = EXCLUDED.grid_column,
    longitude = EXCLUDED.longitude,
    latitude = EXCLUDED.latitude,
    imerg_longitude = EXCLUDED.imerg_longitude,
    imerg_latitude = EXCLUDED.imerg_latitude,
    local_observation_time = EXCLUDED.local_observation_time,
    timezone = EXCLUDED.timezone,
    precipitation_mm = EXCLUDED.precipitation_mm,
    precipitation_rate_mean_mm_h = EXCLUDED.precipitation_rate_mean_mm_h,
    precipitation_rate_max_mm_h = EXCLUDED.precipitation_rate_max_mm_h,
    precipitation_rate_first_half_hour_mm_h = EXCLUDED.precipitation_rate_first_half_hour_mm_h,
    precipitation_rate_second_half_hour_mm_h = EXCLUDED.precipitation_rate_second_half_hour_mm_h,
    random_error_mean_mm_h = EXCLUDED.random_error_mean_mm_h,
    probability_liquid_precipitation_mean_pct = EXCLUDED.probability_liquid_precipitation_mean_pct,
    precipitation_quality_index_mean = EXCLUDED.precipitation_quality_index_mean,
    half_hour_count = EXCLUDED.half_hour_count,
    product_short_name = EXCLUDED.product_short_name,
    product_version = EXCLUDED.product_version,
    run_type = EXCLUDED.run_type,
    source_service = EXCLUDED.source_service,
    source_granules = EXCLUDED.source_granules,
    fetched_at = now();
"""


SUMMARY_SQL = f"""
SELECT
    COUNT(*) AS row_count,
    COUNT(DISTINCT grid_number) AS grid_count,
    MIN(observation_time) AS min_time,
    MAX(observation_time) AS max_time,
    COUNT(*) FILTER (WHERE is_complete_hour) AS complete_rows,
    COUNT(*) FILTER (
        WHERE is_complete_hour AND precipitation_rate_max_mm_h >= 2.5
    ) AS moderate_or_heavier_rows,
    COUNT(*) FILTER (
        WHERE is_complete_hour AND precipitation_rate_max_mm_h >= 7.5
    ) AS heavy_candidate_rows
FROM {OUTPUT_TABLE};
"""


def log(message):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def require_nasa_dependencies():
    missing = []
    modules = {}
    for import_name, package_name in [
        ("earthaccess", "earthaccess"),
        ("harmony", "harmony-py"),
        ("netCDF4", "netCDF4"),
    ]:
        try:
            modules[import_name] = __import__(import_name)
        except ModuleNotFoundError:
            missing.append(package_name)
    if missing:
        packages = " ".join(missing)
        raise RuntimeError(
            "Missing NASA ingestion dependencies. Install them with:\n"
            f"  python -m pip install {packages}\n"
            "or:\n"
            "  python -m pip install -r IMERG_Script/requirements.txt"
        )
    return modules["earthaccess"], modules["harmony"], modules["netCDF4"]


def connect():
    return psycopg2.connect(**DB_CONFIG)


def parse_date(value):
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid date {value!r}; use YYYY-MM-DD.") from exc


def ensure_table(conn):
    with conn.cursor() as cur:
        cur.execute(CREATE_TABLE_SQL)
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
                "grid_number": int(row[0]),
                "grid_row": int(row[1]),
                "grid_column": int(row[2]),
                "longitude": float(row[3]),
                "latitude": float(row[4]),
            }
            for row in cur.fetchall()
        ]


def existing_om_date_range(conn):
    with conn.cursor() as cur:
        cur.execute(f"SELECT MIN(forecast_time), MAX(forecast_time) FROM {OM_TABLE};")
        start, end = cur.fetchone()
    if start is None or end is None:
        raise RuntimeError(f"{OM_TABLE} is empty; pass --start-date and --end-date explicitly.")
    return start.date(), end.date()


def resolve_date_range(conn, requested_start, requested_end):
    om_start, om_end = existing_om_date_range(conn)
    safe_final_end = date.today() - timedelta(days=FINAL_RUN_SAFETY_DELAY_DAYS)
    start_date = requested_start or om_start
    end_date = requested_end or min(om_end, safe_final_end)
    if end_date < start_date:
        raise RuntimeError(
            f"Resolved end date {end_date} is before start date {start_date}. "
            "Final Run data has an approximate 3.5-month latency; pass explicit dates if needed."
        )
    return start_date, end_date, om_start, om_end, safe_final_end


def date_chunks(start_date, end_date, days_per_chunk):
    current = start_date
    while current <= end_date:
        chunk_end = min(current + timedelta(days=days_per_chunk - 1), end_date)
        yield current, chunk_end
        current = chunk_end + timedelta(days=1)


def utc_chunk_bounds(start_date, end_date):
    start = datetime.combine(start_date, time.min, tzinfo=timezone.utc)
    stop_exclusive = datetime.combine(end_date + timedelta(days=1), time.min, tzinfo=timezone.utc)
    return start, stop_exclusive


def fetch_complete_row_count(conn, start_date, end_date):
    start, stop = utc_chunk_bounds(start_date, end_date)
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT COUNT(*)
            FROM {OUTPUT_TABLE}
            WHERE observation_time >= %s
              AND observation_time < %s
              AND is_complete_hour;
            """,
            (start, stop),
        )
        return int(cur.fetchone()[0])


def delete_date_range(conn, start_date, end_date):
    start, stop = utc_chunk_bounds(start_date, end_date)
    with conn.cursor() as cur:
        cur.execute(
            f"DELETE FROM {OUTPUT_TABLE} WHERE observation_time >= %s AND observation_time < %s;",
            (start, stop),
        )
        deleted = cur.rowcount
    conn.commit()
    return deleted


def bangkok_bbox(points, padding_degrees):
    return (
        min(point["longitude"] for point in points) - padding_degrees,
        min(point["latitude"] for point in points) - padding_degrees,
        max(point["longitude"] for point in points) + padding_degrees,
        max(point["latitude"] for point in points) + padding_degrees,
    )


def concept_id_from_result(result):
    value = getattr(result, "concept_id", None)
    if callable(value):
        value = value()
    if value:
        return str(value)
    summary = result.summary() if hasattr(result, "summary") else {}
    value = summary.get("concept-id") or summary.get("concept_id")
    if not value:
        raise RuntimeError(f"Could not extract a CMR concept ID from collection result: {result}")
    return str(value)


def resolve_collection_id(earthaccess, override):
    if override:
        return override
    results = earthaccess.search_datasets(
        short_name=PRODUCT_SHORT_NAME,
        version=PRODUCT_VERSION,
        count=10,
    )
    if not results:
        raise RuntimeError(
            f"NASA CMR returned no collection for {PRODUCT_SHORT_NAME} version {PRODUCT_VERSION}."
        )
    ges_disc = [result for result in results if "GES_DISC" in concept_id_from_result(result)]
    selected = ges_disc[0] if ges_disc else results[0]
    return concept_id_from_result(selected)


def earthdata_login(earthaccess, interactive, persist):
    if interactive:
        auth = earthaccess.login(strategy="interactive", persist=persist)
    elif os.getenv("EARTHDATA_TOKEN") or (
        os.getenv("EARTHDATA_USERNAME") and os.getenv("EARTHDATA_PASSWORD")
    ):
        auth = earthaccess.login(strategy="environment")
    else:
        auth = earthaccess.login(strategy="netrc")
    token_payload = earthaccess.get_edl_token()
    token = (
        token_payload.get("access_token")
        if isinstance(token_payload, dict)
        else token_payload
    )
    if not token:
        raise RuntimeError("Earthdata authentication succeeded but no EDL token was returned.")
    return auth, token


def submit_harmony_subset(
    harmony_module,
    client,
    collection_id,
    bbox,
    start_date,
    end_date,
    variables,
    max_results,
):
    start, stop_exclusive = utc_chunk_bounds(start_date, end_date)
    request = harmony_module.Request(
        collection=harmony_module.Collection(id=collection_id),
        spatial=harmony_module.BBox(*bbox),
        temporal={"start": start, "stop": stop_exclusive - timedelta(microseconds=1)},
        variables=variables,
        format="application/x-netcdf4",
        # IMERG GPM_3IMERGHH supports spatial/variable subsetting but not Harmony
        # concatenation. Each returned granule is tiny because only Bangkok and the
        # selected variables are included; files are removed after the chunk is loaded.
        concatenate=False,
        max_results=max_results,
        skip_preview=True,
        labels=["imerg-bkk-ingestion"],
    )
    if not request.is_valid():
        raise RuntimeError(f"Invalid Harmony request: {request.error_messages()}")
    return client.submit(request)


def download_harmony_results(client, job_id, directory):
    directory.mkdir(parents=True, exist_ok=True)
    futures = client.download_all(job_id, directory=str(directory), overwrite=True)
    paths = [Path(future.result()) for future in futures]
    data_paths = [path for path in paths if path.suffix.lower() in {".nc", ".nc4", ".h5", ".hdf5"}]
    if not data_paths:
        raise RuntimeError(f"Harmony returned no readable NetCDF/HDF files. Download results: {paths}")
    return data_paths


def iter_groups(group):
    yield group
    for subgroup in group.groups.values():
        yield from iter_groups(subgroup)


def normalize_variable_name(name):
    return name.rsplit("/", 1)[-1].lower()


def find_variable(dataset, desired_names, required=True):
    desired = {normalize_variable_name(name) for name in desired_names}
    for group in iter_groups(dataset):
        for name, variable in group.variables.items():
            if normalize_variable_name(name) in desired:
                return variable
    if required:
        available = sorted(
            f"{group.path}/{name}" for group in iter_groups(dataset) for name in group.variables
        )
        raise KeyError(f"Missing variable {sorted(desired)}. Available variables: {available}")
    return None


def numeric_array(variable):
    values = variable[:]
    if np.ma.isMaskedArray(values):
        values = values.filled(np.nan)
    values = np.asarray(values, dtype="float64")
    fill_value = getattr(variable, "_FillValue", None)
    if fill_value is not None:
        values[np.isclose(values, float(fill_value), equal_nan=False)] = np.nan
    missing_value = getattr(variable, "missing_value", None)
    if missing_value is not None and np.isscalar(missing_value):
        values[np.isclose(values, float(missing_value), equal_nan=False)] = np.nan
    return values


def dimension_kind(name):
    lowered = name.lower()
    if "time" in lowered:
        return "time"
    if lowered in {"lon", "longitude", "x"} or "lon" in lowered:
        return "lon"
    if lowered in {"lat", "latitude", "y"} or "lat" in lowered:
        return "lat"
    return None


def variable_cube(variable):
    values = numeric_array(variable)
    dimensions = list(variable.dimensions)
    selections = []
    kept_dimensions = []
    for axis, dimension in enumerate(dimensions):
        kind = dimension_kind(dimension)
        if kind:
            selections.append(slice(None))
            kept_dimensions.append(kind)
        elif values.shape[axis] == 1:
            selections.append(0)
        else:
            raise ValueError(
                f"Variable {variable.name} has unsupported non-spatial dimension "
                f"{dimension} with size {values.shape[axis]}."
            )
    values = values[tuple(selections)]
    if "lat" not in kept_dimensions or "lon" not in kept_dimensions:
        raise ValueError(f"Variable {variable.name} lacks latitude/longitude dimensions: {dimensions}")
    if "time" not in kept_dimensions:
        values = np.expand_dims(values, axis=0)
        kept_dimensions.insert(0, "time")
    order = [kept_dimensions.index(kind) for kind in ("time", "lat", "lon")]
    return np.transpose(values, axes=order)


def python_utc_datetime(value):
    result = datetime(
        value.year,
        value.month,
        value.day,
        value.hour,
        value.minute,
        int(value.second),
        tzinfo=timezone.utc,
    )
    return result


def times_from_dataset(netCDF4, dataset, expected_count, path):
    time_variable = find_variable(dataset, ["time"], required=False)
    if time_variable is not None and hasattr(time_variable, "units"):
        decoded = netCDF4.num2date(
            time_variable[:],
            units=time_variable.units,
            calendar=getattr(time_variable, "calendar", "standard"),
            only_use_cftime_datetimes=False,
        )
        decoded = np.atleast_1d(decoded).tolist()
        times = [python_utc_datetime(value) for value in decoded]
        if len(times) == expected_count:
            return times

    import re

    match = re.search(r"(\d{8})[-_.]S?(\d{6})", path.name)
    if match and expected_count == 1:
        return [
            datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        ]
    raise ValueError(
        f"Could not decode {expected_count} time values from {path.name}. "
        "Keep the downloaded file and inspect its time coordinate."
    )


def nearest_axis_indices(axis_values, requested_values):
    axis_values = np.asarray(axis_values, dtype="float64").squeeze()
    if axis_values.ndim != 1:
        raise ValueError(f"Expected a one-dimensional IMERG coordinate axis, got {axis_values.shape}")
    requested_values = np.asarray(requested_values, dtype="float64")
    return np.abs(axis_values[:, None] - requested_values[None, :]).argmin(axis=0)


def optional_cube(dataset, names):
    variable = find_variable(dataset, names, required=False)
    return variable_cube(variable) if variable is not None else None


def finite_or_none(value):
    value = float(value)
    return value if np.isfinite(value) else None


def read_half_hour_samples(netCDF4, paths, points):
    samples = defaultdict(dict)
    requested_longitudes = [point["longitude"] for point in points]
    requested_latitudes = [point["latitude"] for point in points]

    for path in paths:
        log(f"Reading subset {path.name}")
        with netCDF4.Dataset(path) as dataset:
            longitude_variable = find_variable(dataset, ["lon", "longitude"])
            latitude_variable = find_variable(dataset, ["lat", "latitude"])
            longitudes = numeric_array(longitude_variable).squeeze()
            latitudes = numeric_array(latitude_variable).squeeze()
            longitude_indices = nearest_axis_indices(longitudes, requested_longitudes)
            latitude_indices = nearest_axis_indices(latitudes, requested_latitudes)

            precipitation = variable_cube(find_variable(dataset, ["precipitation"]))
            random_error = optional_cube(dataset, ["randomError"])
            liquid_probability = optional_cube(dataset, ["probabilityLiquidPrecipitation"])
            quality_index = optional_cube(
                dataset,
                ["PrecipitationQualityIndex", "precipitationQualityIndex"],
            )
            times = times_from_dataset(netCDF4, dataset, precipitation.shape[0], path)

            for time_index, sample_time in enumerate(times):
                hour = sample_time.replace(minute=0, second=0, microsecond=0)
                for point_index, point in enumerate(points):
                    lat_index = int(latitude_indices[point_index])
                    lon_index = int(longitude_indices[point_index])
                    precipitation_rate = finite_or_none(
                        precipitation[time_index, lat_index, lon_index]
                    )
                    if precipitation_rate is not None and precipitation_rate < 0:
                        precipitation_rate = None
                    sample = {
                        "time": sample_time,
                        "point": point,
                        "imerg_longitude": float(longitudes[lon_index]),
                        "imerg_latitude": float(latitudes[lat_index]),
                        "precipitation_rate": precipitation_rate,
                        "random_error": finite_or_none(random_error[time_index, lat_index, lon_index])
                        if random_error is not None
                        else None,
                        "liquid_probability": finite_or_none(
                            liquid_probability[time_index, lat_index, lon_index]
                        )
                        if liquid_probability is not None
                        else None,
                        "quality_index": finite_or_none(quality_index[time_index, lat_index, lon_index])
                        if quality_index is not None
                        else None,
                        "source_granule": path.name,
                    }
                    samples[(point["grid_number"], hour)][sample_time] = sample
    return samples


def mean_non_null(values):
    available = [value for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(available)) if available else None


def aggregate_hourly_rows(samples, store_source_granules=False):
    rows = []
    for (_, hour), samples_by_time in sorted(samples.items()):
        hourly_samples = sorted(samples_by_time.values(), key=lambda sample: sample["time"])
        valid_precipitation = [
            sample for sample in hourly_samples if sample["precipitation_rate"] is not None
        ]
        half_hour_count = min(len(valid_precipitation), 2)
        point = hourly_samples[0]["point"]
        rates = [sample["precipitation_rate"] for sample in valid_precipitation[:2]]
        precipitation_mm = float(sum(rate * 0.5 for rate in rates)) if half_hour_count == 2 else None
        mean_rate = float(np.mean(rates)) if rates else None
        max_rate = float(np.max(rates)) if rates else None
        first_rate = next(
            (
                sample["precipitation_rate"]
                for sample in valid_precipitation
                if sample["time"].minute < 30
            ),
            None,
        )
        second_rate = next(
            (
                sample["precipitation_rate"]
                for sample in valid_precipitation
                if sample["time"].minute >= 30
            ),
            None,
        )
        local_hour = hour.astimezone(LOCAL_TIMEZONE).replace(tzinfo=None)
        rows.append(
            (
                point["grid_number"],
                point["grid_row"],
                point["grid_column"],
                point["longitude"],
                point["latitude"],
                hourly_samples[0]["imerg_longitude"],
                hourly_samples[0]["imerg_latitude"],
                hour,
                local_hour,
                TIMEZONE_NAME,
                precipitation_mm,
                mean_rate,
                max_rate,
                first_rate,
                second_rate,
                mean_non_null(sample["random_error"] for sample in hourly_samples),
                mean_non_null(sample["liquid_probability"] for sample in hourly_samples),
                mean_non_null(sample["quality_index"] for sample in hourly_samples),
                half_hour_count,
                PRODUCT_SHORT_NAME,
                PRODUCT_VERSION,
                RUN_TYPE,
                SOURCE_SERVICE,
                sorted({sample["source_granule"] for sample in hourly_samples})
                if store_source_granules
                else [],
            )
        )
    return rows


def upsert_rows(conn, rows):
    if not rows:
        return 0
    with conn.cursor() as cur:
        execute_values(cur, INSERT_SQL, rows, page_size=5000)
    conn.commit()
    return len(rows)


def fetch_summary(conn):
    with conn.cursor() as cur:
        cur.execute(SUMMARY_SQL)
        return cur.fetchone()


def parse_variables(value):
    variables = [item.strip() for item in value.split(",") if item.strip()]
    if "precipitation" not in {name.rsplit("/", 1)[-1] for name in variables}:
        raise argparse.ArgumentTypeError("--variables must include precipitation.")
    return variables


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Download spatially subsetted NASA IMERG Final V07 half-hourly data, aggregate it '
            'to hourly Bangkok-grid rows, and upsert PostgreSQL table "IMERG_BKK_DATA".'
        )
    )
    parser.add_argument("--start-date", type=parse_date, default=None)
    parser.add_argument("--end-date", type=parse_date, default=None)
    parser.add_argument(
        "--chunk-days",
        type=int,
        default=7,
        help="Days per Harmony request. Seven days corresponds to at most 336 half-hour granules.",
    )
    parser.add_argument(
        "--bbox-padding-degrees",
        type=float,
        default=0.15,
        help="Padding around the Bangkok grid so nearest IMERG cells are always included.",
    )
    parser.add_argument(
        "--variables",
        type=parse_variables,
        default=DEFAULT_VARIABLES,
        help="Comma-separated IMERG V07 variables to request.",
    )
    parser.add_argument(
        "--collection-id",
        default=None,
        help="Optional CMR collection concept ID; otherwise resolved from GPM_3IMERGHH V07.",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=1000,
        help="Maximum input granules per Harmony request; must exceed chunk_days * 48.",
    )
    parser.add_argument(
        "--interactive-login",
        action="store_true",
        help="Prompt for NASA Earthdata credentials instead of using environment variables or _netrc.",
    )
    parser.add_argument(
        "--persist-login",
        action="store_true",
        help="With --interactive-login, persist credentials using earthaccess.",
    )
    parser.add_argument(
        "--keep-downloads",
        action="store_true",
        help="Keep Harmony subset files under --download-dir after successful database writes.",
    )
    parser.add_argument(
        "--store-source-granules",
        action="store_true",
        help=(
            "Store subset filenames in every hourly row. Disabled by default to avoid "
            "repeating long filenames across millions of grid rows."
        ),
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=PROJECT_ROOT / "IMERG_Script" / "downloads",
    )
    parser.add_argument(
        "--replace-range",
        action="store_true",
        help="Delete only the requested date range before loading it again.",
    )
    parser.add_argument(
        "--no-skip-complete-chunks",
        action="store_true",
        help="Reprocess chunks even when all expected complete hourly rows already exist.",
    )
    parser.add_argument(
        "--schema-only",
        action="store_true",
        help="Create/verify IMERG_BKK_DATA and exit without NASA authentication.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the resolved range, bounding box, and chunks without contacting NASA.",
    )
    return parser.parse_args()


def validate_args(args):
    if args.chunk_days < 1:
        raise RuntimeError("--chunk-days must be at least 1.")
    if args.max_results < args.chunk_days * 48:
        raise RuntimeError("--max-results must be at least chunk_days * 48.")
    if args.bbox_padding_degrees < 0:
        raise RuntimeError("--bbox-padding-degrees cannot be negative.")
    if args.persist_login and not args.interactive_login:
        raise RuntimeError("--persist-login requires --interactive-login.")


def chunk_directory_context(args, start_date, end_date):
    label = f"{start_date:%Y%m%d}_{end_date:%Y%m%d}"
    if args.keep_downloads:
        directory = args.download_dir.resolve() / label
        directory.mkdir(parents=True, exist_ok=True)
        return nullcontext(directory)
    return tempfile.TemporaryDirectory(prefix=f"imerg_bkk_{label}_")


def main():
    args = parse_args()
    validate_args(args)

    with connect() as conn:
        ensure_table(conn)
        if args.schema_only:
            log(f"Created/verified {OUTPUT_TABLE}; schema-only run complete.")
            return

        points = fetch_grid_points(conn)
        if not points:
            raise RuntimeError(f"No Bangkok grid points found in {GRID_TABLE}.")
        start_date, end_date, om_start, om_end, safe_final_end = resolve_date_range(
            conn, args.start_date, args.end_date
        )
        bbox = bangkok_bbox(points, args.bbox_padding_degrees)
        chunks = list(date_chunks(start_date, end_date, args.chunk_days))
        expected_total_rows = len(points) * ((end_date - start_date).days + 1) * 24

        log(
            f"OM history={om_start}..{om_end}; safe Final cutoff={safe_final_end}; "
            f"IMERG request={start_date}..{end_date}"
        )
        log(
            f"Grid points={len(points)}; bbox={bbox}; chunks={len(chunks)}; "
            f"expected hourly rows={expected_total_rows:,}"
        )
        if args.dry_run:
            for chunk_start, chunk_end in chunks:
                print(f"{chunk_start}..{chunk_end}")
            return

        if args.replace_range:
            deleted = delete_date_range(conn, start_date, end_date)
            log(f"Deleted {deleted:,} existing rows inside the requested date range.")

        earthaccess, harmony_module, netCDF4 = require_nasa_dependencies()
        _, token = earthdata_login(
            earthaccess,
            interactive=args.interactive_login,
            persist=args.persist_login,
        )
        collection_id = resolve_collection_id(earthaccess, args.collection_id)
        client = harmony_module.Client(token=token)
        log(f"Using NASA collection {collection_id} ({PRODUCT_SHORT_NAME} V{PRODUCT_VERSION}).")

        total_upserted = 0
        for chunk_number, (chunk_start, chunk_end) in enumerate(chunks, start=1):
            expected_chunk_rows = len(points) * ((chunk_end - chunk_start).days + 1) * 24
            existing_complete = fetch_complete_row_count(conn, chunk_start, chunk_end)
            if not args.no_skip_complete_chunks and existing_complete >= expected_chunk_rows:
                log(
                    f"Chunk {chunk_number}/{len(chunks)} {chunk_start}..{chunk_end}: "
                    f"skipped ({existing_complete:,}/{expected_chunk_rows:,} complete rows)."
                )
                continue

            log(
                f"Chunk {chunk_number}/{len(chunks)} {chunk_start}..{chunk_end}: "
                "submitting Harmony subset."
            )
            job_id = submit_harmony_subset(
                harmony_module,
                client,
                collection_id,
                bbox,
                chunk_start,
                chunk_end,
                args.variables,
                args.max_results,
            )
            log(f"Harmony job: {job_id}")

            context = chunk_directory_context(args, chunk_start, chunk_end)
            with context as directory_value:
                directory = Path(directory_value)
                paths = download_harmony_results(client, job_id, directory)
                downloaded_bytes = sum(path.stat().st_size for path in paths)
                samples = read_half_hour_samples(netCDF4, paths, points)
                rows = aggregate_hourly_rows(
                    samples,
                    store_source_granules=args.store_source_granules,
                )
                written = upsert_rows(conn, rows)
                total_upserted += written
                complete_in_result = sum(row[18] == 2 for row in rows)
                log(
                    f"Chunk {chunk_number}/{len(chunks)}: upserted {written:,} rows; "
                    f"complete={complete_in_result:,}; files={len(paths)}; "
                    f"subset_bytes={downloaded_bytes:,}."
                )

        summary = fetch_summary(conn)

    log(f"Finished; rows inserted/updated this run={total_upserted:,}.")
    print(
        f"{OUTPUT_TABLE}: rows={summary[0]:,}, grids={summary[1]:,}, "
        f"time={summary[2]}..{summary[3]}, complete={summary[4]:,}, "
        f"max_rate>=2.5={summary[5]:,}, max_rate>=7.5={summary[6]:,}",
        flush=True,
    )


if __name__ == "__main__":
    main()
