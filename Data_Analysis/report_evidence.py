"""Stored evidence for figures quoted in the Form E1 progress report that had no output file.

Each section recomputes one figure from the database (or, for `himawari` and `ifs_age`, measures it
live) and writes one CSV to Data_Analysis/report_evidence/. NFR4 says every reported number must be
reproducible from a stored output; these files are that output.

    section          report figure
    catalogue        TMD lists 227 stations across six regions
    merge_regions    the 85 stations collected so far cover only the central and northern regions
    om_rain_share    Open-Meteo rain in 530,436 of 2,420,544 hours (21.9%)
    tmd_completeness 26 central stations, 37-97% complete depending on the year
    tmd_scrape_rate  about one second per station-day
    imerg_om_corr    IMERG vs ECMWF precipitation r = 0.21 (0.07-0.38 by month)
    late_run_pod     Late Run detection roughly 12% lower against the three in-grid gauges
    late_run_start   everything after 1 October 2025 is Late Run
    may_pilot        archive matches the May 2026 live-scraped pilot
    himawari         each Himawari scan appears about 12 minutes after it starts
    ifs_age          the newest IFS forecast is 7-13 hours old at issue time

Usage:
    python Data_Analysis/report_evidence.py                 # every section
    python Data_Analysis/report_evidence.py late_run_pod    # one or more sections
"""
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

from compare_v0_v1_on_imerg import haversine_km
from om_vs_tmd_verification import DB_CONFIG, PAIRS, UTC_TO_LOCAL_HOURS

OUTPUT_DIR = Path(__file__).resolve().parent / "report_evidence"
CATALOGUES = ["TMD_WEATHER_STATION", "EASTERN_TMD_WEATHER-STATION", "NORTHERN_TMD_WEATHER-STATION",
              "NORTHERN_EASTERN_TMD_WEATHER-STATION", "SOUTHERN_EASTERN_TMD_WEATHER-STATION",
              "SOUTHERN_WESTERN_TMD_WEATHER-STATION"]
RAIN_MM = 0.1
PILOT_START, PILOT_END = "2026-05-11 00:00", "2026-05-24 23:00"
PILOT_COLOCATED_KM = 0.5
PILOT_VARS = ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation",
              "pressure_msl", "cloud_cover", "wind_speed_10m", "wind_direction_10m"]
HIMAWARI_BUCKET, HIMAWARI_SEGMENTS, HIMAWARI_DAYS = "noaa-himawari9", (4, 5), 3
IFS_META_URL = "https://api.open-meteo.com/data/ecmwf_ifs/static/meta.json"


def query(sql, params=None):
    with psycopg2.connect(**DB_CONFIG) as conn:
        return pd.read_sql(sql, conn, params=params)


def save(df, name):
    OUTPUT_DIR.mkdir(exist_ok=True)
    df.to_csv(OUTPUT_DIR / name, index=False)
    print(f"{name}:\n{df.to_string(index=False)}\n")


def catalogue():
    union = " UNION ALL ".join(f"SELECT '{t}' AS source_table, region_name, station FROM \"{t}\""
                               for t in CATALOGUES)
    df = query(f"SELECT source_table, region_name, COUNT(DISTINCT station) AS stations "
               f"FROM ({union}) s GROUP BY 1, 2 ORDER BY 1")
    total = pd.DataFrame([{"source_table": "all", "region_name": f"{df.region_name.nunique()} regions",
                           "stations": df.stations.sum()}])
    save(pd.concat([df, total]), "tmd_station_catalogue.csv")


def merge_regions():
    union = " UNION ALL ".join(f"SELECT DISTINCT region, region_name FROM \"{t}\"" for t in CATALOGUES)
    save(query(f"SELECT m.region, r.region_name, COUNT(DISTINCT m.station) AS stations "
               f"FROM \"TMD_DATA_MERGE\" m LEFT JOIN ({union}) r USING (region) GROUP BY 1, 2 ORDER BY 1"),
         "tmd_merge_station_regions.csv")


def om_rain_share():
    # the five-year Open-Meteo window that overlaps the TMD gauge record (gauges end 24 June 2026)
    df = query(f"""SELECT MIN(local_forecast_time) AS window_start, MAX(local_forecast_time) AS window_end,
                          COUNT(DISTINCT grid_number) AS cells, COUNT(*) AS hours,
                          SUM((precipitation >= {RAIN_MM})::int) AS rain_hours
                   FROM "OM_BKK_DATA" WHERE local_forecast_time < '2026-06-25'""")
    df["rain_share"] = df.rain_hours / df.hours
    save(df, "om_bkk_rain_hour_share.csv")


def tmd_panel_stations():
    """The 26 central-catalogue stations that make up the TMD modelling panel."""
    return query('SELECT DISTINCT station FROM "BKK_TMD_WEATHER_DATA" '
                 'WHERE station IN (SELECT station FROM "TMD_WEATHER_STATION")').station.tolist()


def tmd_completeness():
    stations = tmd_panel_stations()
    df = query("""SELECT EXTRACT(YEAR FROM utc_time)::int AS year, COUNT(DISTINCT (station, utc_time)) AS station_hours,
                         MIN(utc_time) AS first_hour, MAX(utc_time) AS last_hour
                  FROM "BKK_TMD_WEATHER_DATA" WHERE station = ANY(%(s)s) AND precipitation_mm IS NOT NULL
                  GROUP BY 1 ORDER BY 1""", {"s": stations})
    end = df.last_hour.max()
    year_start = pd.to_datetime(df.year.astype(str) + "-01-01")
    year_end = (year_start + pd.offsets.YearBegin()).where(df.year < end.year, end + pd.Timedelta(hours=1))
    df["possible_station_hours"] = len(stations) * (year_end - year_start) / pd.Timedelta(hours=1)
    df["completeness"] = df.station_hours / df.possible_station_hours
    df.insert(1, "stations", len(stations))
    save(df.drop(columns=["first_hour"]), "tmd_central_completeness_by_year.csv")


def tmd_scrape_rate():
    """Seconds between consecutive station-days, from the loaded_at stamp each scraped day carries."""
    loads = query('SELECT station, utc_time::date AS day, MIN(loaded_at) AS loaded_at '
                  'FROM "BKK_TMD_WEATHER_DATA" GROUP BY 1, 2 ORDER BY 3')
    gaps = loads.loaded_at.diff().dt.total_seconds()
    within_run = gaps[(gaps > 0) & (gaps <= 60)]          # longer gaps are pauses between runs
    median = within_run.median()
    save(pd.DataFrame([{
        "station_days": len(loads), "gaps_used": len(within_run),
        "median_s": median, "p10_s": within_run.quantile(0.1), "p90_s": within_run.quantile(0.9),
        "hours_26_stations_5_years": 26 * 5 * 365 * median / 3600,
        "days_227_stations_5_years": 227 * 5 * 365 * median / 86400,
    }]), "tmd_scrape_seconds_per_station_day.csv")


def imerg_om_corr():
    df = query("""SELECT TO_CHAR(om.local_forecast_time, 'YYYY-MM') AS month, COUNT(*) AS rows,
                         CORR(im.precipitation_mm, om.precipitation) AS r_cell_mean,
                         CORR(im.precipitation_max_mm, om.precipitation) AS r_cell_max
                  FROM "OM_Thailand_Data" om
                  JOIN "IMERG_THAILAND_DATA" im
                    ON im.grid_number = om.grid_number AND im.local_observation_time = om.local_forecast_time
                  GROUP BY ROLLUP (1) ORDER BY 1 NULLS FIRST""")
    save(df.fillna({"month": "all"}), "imerg_vs_om_precip_correlation.csv")


def late_run_pod():
    """POD of IMERG against the three in-grid gauges, Final vs Late Run, on the same calendar months."""
    cells = {station: cell for station, (cell, _) in PAIRS.items()}
    gauge = query(f"""SELECT station, utc_time + INTERVAL '{UTC_TO_LOCAL_HOURS} hour' AS t_local,
                             SUM(precipitation_mm) AS gauge_mm
                      FROM "BKK_TMD_WEATHER_DATA" WHERE station = ANY(%(s)s) AND precipitation_mm IS NOT NULL
                      GROUP BY 1, 2""", {"s": list(cells)})
    gauge["grid_number"] = gauge.station.map(cells)
    imerg = query("""SELECT grid_number, local_observation_time AS t_local, precipitation_mm AS imerg_mm, run_type
                     FROM "IMERG_BKK_DATA" WHERE is_complete_hour AND grid_number = ANY(%(c)s)""",
                  {"c": list(cells.values())})
    df = gauge.merge(imerg, on=["grid_number", "t_local"])
    df = df[df.gauge_mm >= RAIN_MM]
    df["era"] = np.where(df.run_type == "provisional", "late_run", "final_run")
    df["month"] = df.t_local.dt.month
    df["detected"] = df.imerg_mm >= RAIN_MM
    by_month = (df.groupby(["month", "era"]).detected.agg(gauge_rain_hours="size", detected="sum")
                .reset_index())
    by_month["pod"] = by_month.detected / by_month.gauge_rain_hours
    shared = set(df[df.era == "late_run"].month)
    same = by_month[by_month.month.isin(shared)]
    summary = (same.groupby("era")[["gauge_rain_hours", "detected"]].sum().reset_index()
               .assign(month="months in both eras", pod=lambda d: d.detected / d.gauge_rain_hours))
    final, late = summary.set_index("era").pod[["final_run", "late_run"]]
    summary["late_vs_final"] = late / final - 1
    save(pd.concat([by_month, summary]), "imerg_late_run_pod_vs_gauges.csv")


def late_run_start():
    """Where each IMERG table switches from Final Run to provisional Late Run."""
    save(pd.concat([query(f"""SELECT '{t}' AS source_table, run_type, MIN(local_observation_time) AS first_hour,
                                     MAX(local_observation_time) AS last_hour, COUNT(*) AS rows
                              FROM "{t}" GROUP BY run_type ORDER BY 3""")
                    for t in ("IMERG_BKK_DATA", "IMERG_THAILAND_DATA")]), "imerg_run_type_periods.csv")


def may_pilot():
    """Each 25 km cell against its nearest point in the May 2026 live-scraped pilot table."""
    cells = query("""SELECT DISTINCT grid_number, longitude, latitude FROM "OM_Thailand_Data"
                     WHERE local_forecast_time = %(t)s""", {"t": PILOT_START})
    points = query("SELECT DISTINCT ecmwf_hres_point_id, longitude, latitude FROM openmeteo_thailand_core_weather")
    dist = haversine_km(cells.latitude.to_numpy()[:, None], cells.longitude.to_numpy()[:, None],
                        points.latitude.to_numpy()[None, :], points.longitude.to_numpy()[None, :])
    nearest = dist.argmin(axis=1)
    cells["ecmwf_hres_point_id"] = points.ecmwf_hres_point_id.to_numpy()[nearest]
    cells["distance_km"] = dist[np.arange(len(cells)), nearest]
    cols = ", ".join(PILOT_VARS)
    archive = query(f"""SELECT grid_number, local_forecast_time AS t, {cols} FROM "OM_Thailand_Data"
                        WHERE local_forecast_time BETWEEN %(a)s AND %(b)s""", {"a": PILOT_START, "b": PILOT_END})
    pilot = query(f"""SELECT ecmwf_hres_point_id, weather_time AS t, {cols} FROM openmeteo_thailand_core_weather
                      WHERE ecmwf_hres_point_id = ANY(%(p)s)""",
                  {"p": cells.ecmwf_hres_point_id.unique().tolist()})
    df = (archive.merge(cells[["grid_number", "ecmwf_hres_point_id", "distance_km"]], on="grid_number")
          .merge(pilot, on=["ecmwf_hres_point_id", "t"], suffixes=("_archive", "_pilot")))
    for v in PILOT_VARS:
        df[f"same_{v}"] = (df[f"{v}_archive"] - df[f"{v}_pilot"]).abs() <= 1e-6
    per_cell = df.groupby("grid_number").agg(
        distance_km=("distance_km", "first"), hours=("t", "size"),
        **{f"identical_{v}": (f"same_{v}", "mean") for v in PILOT_VARS}).reset_index()
    per_cell.to_csv(OUTPUT_DIR / "may_2026_pilot_vs_archive_per_cell.csv", index=False)
    # Temperature and dew point are elevation-adjusted for the exact requested coordinate, so a pilot
    # point a few hundred metres away differs by rounding; the other six fields are raw model output.
    raw_vars = [v for v in PILOT_VARS if v not in ("temperature_2m", "dew_point_2m")]
    near = per_cell[per_cell.distance_km <= PILOT_COLOCATED_KM]
    same_point = near[(near[[f"identical_{v}" for v in raw_vars]] >= 0.99).all(axis=1)].grid_number
    part = df[df.grid_number.isin(same_point)]
    rows = [{"variable": v, "cells_within_km": len(near), "same_model_point_cells": len(same_point),
             "cell_hours": len(part), "share_identical": part[f"same_{v}"].mean(),
             "max_abs_diff": (part[f"{v}_archive"] - part[f"{v}_pilot"]).abs().max(),
             "r": part[f"{v}_archive"].corr(part[f"{v}_pilot"])} for v in PILOT_VARS]
    save(pd.DataFrame(rows), "may_2026_pilot_vs_archive.csv")


def himawari():
    """Minutes from scan start to the S3 upload of the B13 segments V10 reads, for the last few days."""
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    now = pd.Timestamp.now(tz="UTC").floor("h")
    rows = []
    for slot in pd.date_range(now - pd.Timedelta(days=HIMAWARI_DAYS), now - pd.Timedelta(hours=1), freq="h"):
        prefix = f"AHI-L1b-FLDK/{slot:%Y/%m/%d/%H}00/"
        objects = s3.list_objects_v2(Bucket=HIMAWARI_BUCKET, Prefix=prefix).get("Contents", [])
        uploads = [o["LastModified"] for o in objects
                   if any(f"_B13_FLDK_R20_S{s:02d}10" in o["Key"] for s in HIMAWARI_SEGMENTS)]
        if len(uploads) == len(HIMAWARI_SEGMENTS):
            rows.append({"scan_start_utc": slot, "minutes_to_upload": (max(uploads) - slot).total_seconds() / 60})
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "himawari_b13_upload_latency_per_scan.csv", index=False)
    m = df.minutes_to_upload
    save(pd.DataFrame([{"scans": len(df), "first_scan": df.scan_start_utc.min(), "last_scan": df.scan_start_utc.max(),
                        "median_min": m.median(), "p10_min": m.quantile(0.1), "p90_min": m.quantile(0.9),
                        "max_min": m.max()}]), "himawari_b13_upload_latency.csv")


def ifs_age():
    """Open-Meteo's own run metadata: how old the newest ECMWF IFS run is between updates."""
    with urllib.request.urlopen(IFS_META_URL, timeout=30) as r:
        meta = json.load(r)
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "ecmwf_ifs_meta_snapshot.json").write_text(json.dumps(meta, indent=2))
    delay_h = (meta["last_run_availability_time"] - meta["last_run_initialisation_time"]) / 3600
    interval_h = meta["update_interval_seconds"] / 3600
    save(pd.DataFrame([{"fetched_utc": pd.Timestamp.now(tz="UTC").floor("s"),
                        "run_initialisation_utc": pd.Timestamp(meta["last_run_initialisation_time"], unit="s"),
                        "run_available_utc": pd.Timestamp(meta["last_run_availability_time"], unit="s"),
                        "availability_delay_h": delay_h, "update_interval_h": interval_h,
                        "age_at_issue_min_h": delay_h, "age_at_issue_max_h": delay_h + interval_h}]),
         "ecmwf_ifs_forecast_age.csv")


SECTIONS = {f.__name__: f for f in (catalogue, merge_regions, om_rain_share, tmd_completeness, tmd_scrape_rate,
                                    imerg_om_corr, late_run_pod, late_run_start, may_pilot, himawari, ifs_age)}


def main():
    for name in sys.argv[1:] or SECTIONS:
        SECTIONS[name]()


if __name__ == "__main__":
    main()
