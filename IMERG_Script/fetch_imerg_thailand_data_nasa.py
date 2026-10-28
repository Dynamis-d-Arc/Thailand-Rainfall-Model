"""Direct-NASA fallback for "IMERG_THAILAND_DATA".

fetch_imerg_thailand_data_gee.py samples IMERG from Earth Engine, whose copy of the Late Run
can lag NASA's own archive by days (26-28 Oct 2026: 2.5 days). This script fills the UTC hours
the table is still missing straight from the GES DISC Late Run half-hourly files, as OPeNDAP
subsets over the Thailand window, and writes the same columns with the same cell footprint.
That footprint is Earth Engine's, not the geometric one: reduceRegions at scale 11132 m works
on a grid of 11132 m pixels anchored at the image origin, which drifts 0.0013 deg from the native
0.1 deg grid at Thailand's longitude, each grid pixel resampled nearest-neighbour from the native
pixel under its centre, with area weights kept in 1/256 steps (smaller ones vanish). Reproducing
that grid matches all 833 stored pixel counts; the geometric rectangle matched 780. Hours already
complete in the table are never touched, so Earth Engine and this script can both run and the
table decides who fills what.

    python IMERG_Script/fetch_imerg_thailand_data_nasa.py --start-date 2026-10-26 --end-date 2026-10-28
    python IMERG_Script/fetch_imerg_thailand_data_nasa.py --start-date 2026-10-25 --end-date 2026-10-25 --compare

--compare recomputes hours Earth Engine already delivered and reports the differences instead
of writing (the parity check). Needs an Earthdata login, as fetch_imerg_bkk_data.py does:
~/_netrc (python -c "import earthaccess; earthaccess.login(strategy='interactive', persist=True)")
or EARTHDATA_USERNAME / EARTHDATA_PASSWORD.
"""
import argparse
import math
import re
import sys
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning, module="earthaccess")

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import fetch_imerg_thailand_data_gee as gee  # noqa: E402  table, grid, upsert, cell size
from fetch_imerg_bkk_data import (  # noqa: E402  NetCDF helpers + dependency check
    find_variable, log, require_nasa_dependencies, variable_cube,
)

SOURCE_SERVICE = "NASA GES DISC OPeNDAP / GPM IMERG Late V07"
RUN_TYPE = "provisional"          # what Earth Engine calls the Late Run in this table
PIXEL_DEG = 0.1                   # native IMERG pixel
EE_PIXEL_DEG = gee.GEE_PIXEL_SCALE_METERS / 111319.4907932736   # EE's 11132 m in EPSG:4326 degrees
WEIGHT_STEP = 1 / 256             # EE keeps a pixel's area weight in 1/256 steps; below that it is gone
SHORT_NAME, VERSION = "GPM_3IMERGHHL", "07"   # Late Run, half-hourly, as catalogued in CMR
STAMP = re.compile(r"\.(\d{8})-S(\d{6})-E\d{6}\.\d{4}\.V\d{2}[A-Z]?\.HDF5")
COLUMNS = [
    "grid_number", "grid_row", "grid_column", "longitude", "latitude", "observation_time",
    "local_observation_time", "timezone", "precipitation_mm", "precipitation_max_mm",
    "precipitation_pixelmax_mm", "precipitation_rate_mean_mm_h", "precipitation_rate_max_mm_h",
    "precipitation_rate_first_half_hour_mm_h", "precipitation_rate_second_half_hour_mm_h",
    "pixel_count", "half_hour_count", "product_short_name", "product_version", "run_type",
    "source_service",
]


def pixel_window(lo, hi, origin, sign, native_origin):
    """Earth Engine's reduce grid along one axis over [lo, hi]: pixels of EE_PIXEL_DEG counted
    from `origin` in direction `sign` (lon: -180 eastward; lat: 90 southward). Returns the
    native 0.1 deg pixel index under each grid pixel's centre and its exact overlap fraction."""
    a, b = sorted(((lo - origin) * sign, (hi - origin) * sign))
    k = np.arange(math.floor(a / EE_PIXEL_DEG), math.ceil(b / EE_PIXEL_DEG))
    overlap = (np.minimum(b, (k + 1) * EE_PIXEL_DEG) - np.maximum(a, k * EE_PIXEL_DEG)) / EE_PIXEL_DEG
    centre = origin + sign * (k + 0.5) * EE_PIXEL_DEG
    return np.floor((centre - native_origin) / PIXEL_DEG).astype(int), overlap


def cell_footprint(cell):
    """(native lat indices, native lon indices, 2-D weights) of Earth Engine's grid pixels over
    the cell rectangle; a weight of 0 marks a pixel Earth Engine drops (area under 1/256)."""
    h = gee.CELL_HALF_DEG
    lon_idx, o_lon = pixel_window(cell["longitude"] - h, cell["longitude"] + h, -180.0, 1, -180.0)
    lat_idx, o_lat = pixel_window(cell["latitude"] - h, cell["latitude"] + h, 90.0, -1, -90.0)
    return lat_idx, lon_idx, np.floor(np.outer(o_lat, o_lon) / WEIGHT_STEP) * WEIGHT_STEP


def valid_mask(block, weights):
    """Pixels with a real rate (IMERG fill is negative) that Earth Engine would count at all."""
    return np.isfinite(block) & (np.nan_to_num(block, nan=-1.0) >= 0) & (weights > 0)


def half_hour_stats(block, weights):
    """(area-weighted mean rate, max rate, valid pixel count) or None when nothing is valid."""
    valid = valid_mask(block, weights)
    if not valid.any():
        return None
    w = weights[valid]
    return float((block[valid] * w).sum() / w.sum()), float(block[valid].max()), int(valid.sum())


def hour_row(cell, hour, first, second, weights):
    """One table row from the two half-hour rate blocks (mm/h; None when unpublished), in the
    tuple layout of fetch_imerg_thailand_data_gee.rows_from_features."""
    stats = [half_hour_stats(b, weights) if b is not None else None for b in (first, second)]
    present = [s for s in stats if s is not None]
    means, maxes, counts = ([s[k] for s in present] for k in range(3))
    complete = len(present) == 2
    pixelmax = None
    if complete:
        both = valid_mask(first, weights) & valid_mask(second, weights)
        if both.any():
            pixelmax = float((0.5 * first + 0.5 * second)[both].max())
    return (
        cell["grid_number"], cell["grid_row"], cell["grid_column"],
        cell["longitude"], cell["latitude"],
        hour, hour.astimezone(gee.LOCAL_TIMEZONE).replace(tzinfo=None), gee.TIMEZONE_NAME,
        0.5 * sum(means) if complete else None,
        0.5 * sum(maxes) if complete else None,
        pixelmax,
        float(np.mean(means)) if means else None,
        float(np.max(maxes)) if maxes else None,
        stats[0][0] if stats[0] else None,
        stats[1][0] if stats[1] else None,
        float(np.mean(counts)) if counts else None,
        len(present), gee.PRODUCT_SHORT_NAME, gee.PRODUCT_VERSION, RUN_TYPE, SOURCE_SERVICE,
    )


def login(earthaccess):
    """~/_netrc first (earthaccess mints a fresh token from it every run), else the EARTHDATA_*
    environment variables. A stale EARTHDATA_TOKEN in the environment must not win over _netrc."""
    from earthaccess.auth import netrc_path
    auth = earthaccess.login(strategy="netrc" if netrc_path().exists() else "environment")
    if not getattr(auth, "token", None):
        raise RuntimeError(
            "Earthdata login failed; create ~/_netrc once with: python -c \"import earthaccess; "
            "earthaccess.login(strategy='interactive', persist=True)\"")
    return earthaccess.get_requests_https_session()


def table_hours(conn, start, stop):
    """{UTC hour: complete rows} for the table's hours in [start, stop)."""
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT observation_time, COUNT(*) FILTER (WHERE is_complete_hour) "
            f"FROM {gee.OUTPUT_TABLE} WHERE observation_time >= %s AND observation_time < %s "
            f"GROUP BY 1",
            (start, stop),
        )
        return {t.astimezone(timezone.utc): int(n) for t, n in cur.fetchall()}


def granule_start(name):
    m = STAMP.search(name)
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)


def find_granules(earthaccess, hours):
    """{half-hour start: OPeNDAP URL} for every Late Run granule NASA's catalogue (CMR) lists
    inside the wanted hours. The catalogue always names the live hosts; the on-prem
    gpm1.gesdisc host stopped answering in the October 2026 archive migration."""
    lo, hi = min(hours), max(hours) + timedelta(hours=1)
    found = earthaccess.search_data(short_name=SHORT_NAME, version=VERSION, count=2000,
                                    temporal=(lo.isoformat(), hi.isoformat()))
    wanted, out = set(hours), {}
    for g in found:
        urls = [u["URL"] for u in g["umm"].get("RelatedUrls", [])
                if u.get("Subtype") == "OPENDAP DATA"]
        start = granule_start(g["umm"]["GranuleUR"])
        if urls and start.replace(minute=0) in wanted:
            out[start] = urls[0]
    return out


def subset_block(session, netCDF4, opendap_url, lat_win, lon_win):
    """The Thailand window of one granule as a (lat, lon) rate array via an OPeNDAP subset."""
    r = session.get(f"{opendap_url}.nc4?precipitation[0:0][{lon_win[0]}:{lon_win[1]}]"
                    f"[{lat_win[0]}:{lat_win[1]}]", timeout=300)
    r.raise_for_status()
    with netCDF4.Dataset("subset.nc4", memory=r.content) as ds:
        return variable_cube(find_variable(ds, ["precipitation"]))[0]   # (lat, lon)


def compare(conn, rows):
    """Parity of recomputed rows against what Earth Engine already wrote."""
    import pandas as pd
    new = pd.DataFrame(rows, columns=COLUMNS)
    new["observation_time"] = pd.to_datetime(new.observation_time, utc=True)
    old = pd.read_sql(
        f"SELECT grid_number, observation_time, precipitation_mm, precipitation_max_mm, "
        f"precipitation_pixelmax_mm, pixel_count FROM {gee.OUTPUT_TABLE} "
        f"WHERE observation_time = ANY(%s)",
        conn, params=(list(new.observation_time.drop_duplicates().dt.to_pydatetime()),))
    old["observation_time"] = pd.to_datetime(old.observation_time, utc=True)
    m = new.merge(old, on=["grid_number", "observation_time"], suffixes=("_nasa", "_gee"))
    log(f"compare: {len(m):,} rows in both sources")
    for col in ("precipitation_pixelmax_mm", "precipitation_max_mm", "precipitation_mm"):
        d = (m[f"{col}_nasa"] - m[f"{col}_gee"]).abs()
        log(f"  {col}: max |diff| {d.max():.4f} mm, rows > 0.005 mm: {int((d > 0.005).sum())}")
    log(f"  pixel_count equal: {(m.pixel_count_nasa == m.pixel_count_gee).mean():.3%}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--start-date", type=gee.parse_date, required=True, help="UTC date")
    ap.add_argument("--end-date", type=gee.parse_date, required=True, help="UTC date, inclusive")
    ap.add_argument("--compare", action="store_true",
                    help="recompute hours already in the table and report differences; no writes")
    ap.add_argument("--max-hours", type=int, default=None)
    args = ap.parse_args()

    earthaccess, _harmony, netCDF4 = require_nasa_dependencies()
    conn = gee.connect()
    try:
        gee.ensure_table(conn)
        cells = gee.fetch_grid_cells(conn)
        start, stop = gee.utc_chunk_bounds(args.start_date, args.end_date)
        have = table_hours(conn, start, stop)
        hours = [start + timedelta(hours=k)
                 for k in range(int((stop - start).total_seconds() // 3600))]
        if args.compare:
            hours = [h for h in hours if have.get(h, 0) >= len(cells)]
        else:
            hours = [h for h in hours if have.get(h, 0) < len(cells)]
        hours = hours[: args.max_hours] if args.max_hours else hours
        log(f"{args.start_date}..{args.end_date}: {len(hours)} hour(s) to "
            f"{'compare' if args.compare else 'fill'}")
        written = unpublished = 0
        if not hours:
            return

        session = login(earthaccess)
        footprints = {c["grid_number"]: cell_footprint(c) for c in cells}
        lat_win = (min(int(f[0].min()) for f in footprints.values()),
                   max(int(f[0].max()) for f in footprints.values()))
        lon_win = (min(int(f[1].min()) for f in footprints.values()),
                   max(int(f[1].max()) for f in footprints.values()))

        granules = find_granules(earthaccess, hours)
        rows = []
        for hour in hours:
            halves = {}
            for minute in (0, 30):
                url = granules.get(hour + timedelta(minutes=minute))
                if url:
                    halves[minute] = subset_block(session, netCDF4, url, lat_win, lon_win)
            if not halves:
                unpublished += 1
                continue
            for cell in cells:
                lat_idx, lon_idx, w = footprints[cell["grid_number"]]
                sel = np.ix_(lat_idx - lat_win[0], lon_idx - lon_win[0])
                rows.append(hour_row(cell, hour,
                                     halves[0][sel] if 0 in halves else None,
                                     halves[30][sel] if 30 in halves else None, w))
            log(f"  {hour:%Y-%m-%d %H}Z: {len(halves)} half-hour(s)")
        if unpublished:
            log(f"  {unpublished} hour(s) not published by NASA yet")

        if args.compare:
            compare(conn, rows)
        else:
            written = gee.upsert_rows(conn, rows)
            log(f"upserted {written:,} rows ({sum(r[16] == 2 for r in rows):,} complete)")
    finally:
        conn.close()
        print(f"NASA_FETCH_SUMMARY hours={len(hours) - unpublished} rows={written}", flush=True)


if __name__ == "__main__":
    main()
