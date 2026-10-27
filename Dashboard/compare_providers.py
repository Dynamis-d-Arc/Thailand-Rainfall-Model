"""Compare the live V10 model against public weather services over Bangkok.

Every hour this records, for the nine 25 km grid cells covering the Bangkok metro
area, the rain probability each forecaster gives for the next 1, 3 and 6 hours:

  model         the live V10/V13 prediction CSV for that issue hour (both thresholds)
  climatology   the cell's usual wet frequency for this month and hour (no-skill floor)
  open_meteo    Open-Meteo hourly precipitation_probability (keyless)
  google        Google Weather API hourly precipitation.probability   (needs key)
  accuweather   AccuWeather 12-hour hourly PrecipitationProbability   (needs key;
                central cell only, the free tier allows 50 calls/day)
  weatherapi    WeatherAPI.com hourly chance_of_rain                   (needs key)
  tomorrow      Tomorrow.io hourly precipitationProbability            (needs key)
  openweathermap OpenWeatherMap 3-hourly pop                           (needs key)

Provider hourly probabilities refer to the PRECEDING hour (Open-Meteo: "probability of
precipitation ... of the preceding hour"), so the value stamped issue+k covers
[issue+k-1, issue+k) and hours k=1..h cover the label window [issue, issue+h). The log keeps
1 - prod(1 - p_k) in p_0.1mm plus the per-hour list; `score` grades the 1 h rows as the
primary comparison and reports every multi-hour provider row twice, as bounds on P(any rain
in the window): "[max]" = max_k p_k (lower bound) and "[1-prod]" = 1 - prod(1 - p_k), the
independence product, an upper bound when hours are positively correlated. Neither is the
true window probability, which the hourly marginals do not determine.
Provider probabilities are for "measurable rain at the point", so they are graded
at the 0.1 mm threshold only; the model is graded at both thresholds.

`score` grades every logged forecast whose IMERG labels have arrived, against two
label definitions on the same cells: the project's cell-max label (rain anywhere
in the 25 km cell) and the cell-mean label (area-average, closer to a point
forecast). It also adds a persistence row (observed IMERG in the hour(s) before the
issue hour carried forward) and writes a paired comparison - each provider against the model on
exactly the same issue hours and cells - to provider_scores.csv and a Markdown
summary to provider_comparison.md, ready for the report.

API keys: environment variables GOOGLE_WEATHER_API_KEY, ACCUWEATHER_API_KEY,
WEATHERAPI_KEY, TOMORROW_API_KEY, OPENWEATHERMAP_API_KEY, or the same names as
fields in Dashboard/provider_keys.json (git-ignored). Providers without a key are
skipped and logged.

Usage:
    python Dashboard/compare_providers.py fetch [--stamp YYYYMMDD_HHMM] [--providers a,b]
    python Dashboard/compare_providers.py score
    python Dashboard/compare_providers.py report
    python Dashboard/compare_providers.py run          # fetch + score (the scheduled task)
"""

import argparse
import json
import math
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROJ = ROOT.parent
PRED_DIR = PROJ / "ML_Model_V2" / "trained_models" / "om_thailand_rain_v10_deploy"
OUT_DIR = Path(os.getenv("PROVIDER_COMPARE_DIR", str(PRED_DIR)))
FORECAST_LOG = OUT_DIR / "provider_forecasts.csv"
LABEL_VERSION = "labels_v2"             # window [issue, issue+h) on the wettest pixel's hourly total
SCORES_CSV = OUT_DIR / f"provider_scores_{LABEL_VERSION}.csv"
SUMMARY_MD = OUT_DIR / f"provider_comparison_{LABEL_VERSION}.md"
CLIM_PATH = PRED_DIR / f"v10_climatology_{LABEL_VERSION}.csv.gz"
KEYS_FILE = ROOT / "provider_keys.json"
ACCU_CACHE = ROOT / "data" / "accuweather_locations.json"

# the nine nationwide cells whose centres fall over the Bangkok metro area
BKK_GRIDS = [183, 182, 169, 199, 184, 168, 198, 170, 200]
CENTRAL_GRID = 183
HORIZONS = [1, 3, 6]
THRESHOLDS = [0.1, 1.0]
TZ_OFFSET = timedelta(hours=7)          # Asia/Bangkok, no DST
HTTP_TIMEOUT = 25
UA = "thailand-rainwatch-capstone (brandon.ng.kh@gmail.com)"

IMERG_TABLE = '"IMERG_THAILAND_DATA"'
IMERG_LABEL_COLUMN = "precipitation_pixelmax_mm"
DB_CONFIG = {
    "host": os.getenv("PGHOST", "localhost"),
    "port": int(os.getenv("PGPORT", "5432")),
    "dbname": os.getenv("PGDATABASE", "postgres"),
    "user": os.getenv("PGUSER", "postgres"),
    "password": os.getenv("PGPASSWORD", "Pass1234"),
}


def log(msg):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


# ----------------------------------------------------------------------------- keys / http

def api_keys():
    names = {"google": "GOOGLE_WEATHER_API_KEY", "accuweather": "ACCUWEATHER_API_KEY",
             "weatherapi": "WEATHERAPI_KEY", "tomorrow": "TOMORROW_API_KEY",
             "openweathermap": "OPENWEATHERMAP_API_KEY"}
    file_keys = {}
    if KEYS_FILE.exists():
        try:
            file_keys = json.loads(KEYS_FILE.read_text(encoding="utf-8"))
        except Exception as exc:
            log(f"provider_keys.json unreadable: {exc}")
    out = {}
    for prov, env in names.items():
        val = os.getenv(env) or file_keys.get(env) or file_keys.get(prov)
        if val:
            out[prov] = val.strip()
    return out


def get_json(url, headers=None):
    req = urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})})
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as r:
        return json.loads(r.read().decode("utf-8"))


def combine(pops):
    """P(any rain over the window) from per-hour probabilities, assuming independence."""
    p = 1.0
    for x in pops:
        p *= (1.0 - float(x))
    return 1.0 - p


def window_rows(provider, grid, issue, hourly):
    """rows for HORIZONS from {lead_hour(1..6): p} ; None if any lead missing."""
    rows = []
    for h in HORIZONS:
        pops = [hourly.get(k) for k in range(1, h + 1)]
        if any(p is None for p in pops):
            continue
        rows.append({"provider": provider, "grid_number": grid, "horizon_h": h,
                     "p_0.1mm": round(combine(pops), 4), "p_1.0mm": np.nan,
                     "hourly_pop": json.dumps([round(float(p), 3) for p in pops])})
    return rows


# ----------------------------------------------------------------------------- cells

def bkk_cells(stamp):
    """DataFrame grid_number, latitude, longitude (+ model p_ columns if the CSV exists)."""
    path = PRED_DIR / f"v10_predictions_{stamp}.csv"
    if path.exists():
        d = pd.read_csv(path)
        return d[d.grid_number.isin(BKK_GRIDS)].copy(), True
    # no prediction for this hour (machine asleep / run failed): coordinates from any CSV
    anyp = sorted(PRED_DIR.glob("v10_predictions_*.csv"))
    if not anyp:
        raise RuntimeError("no prediction CSV on disk to take grid coordinates from")
    d = pd.read_csv(anyp[-1], usecols=["grid_number", "latitude", "longitude"])
    return d[d.grid_number.isin(BKK_GRIDS)].copy(), False


# ----------------------------------------------------------------------------- providers

def fetch_open_meteo(cells, issue):
    lats = ",".join(f"{v:.4f}" for v in cells.latitude)
    lons = ",".join(f"{v:.4f}" for v in cells.longitude)
    url = ("https://api.open-meteo.com/v1/forecast?latitude=" + lats + "&longitude=" + lons +
           "&hourly=precipitation_probability,precipitation&forecast_days=2&timezone=Asia%2FBangkok")
    data = get_json(url)
    data = data if isinstance(data, list) else [data]
    rows = []
    for grid, blk in zip(cells.grid_number, data):
        times = blk["hourly"]["time"]
        pops = blk["hourly"]["precipitation_probability"]
        hourly = {}
        for k in range(1, 7):
            t = (issue + timedelta(hours=k)).strftime("%Y-%m-%dT%H:%M")
            if t in times:
                v = pops[times.index(t)]
                if v is not None:
                    hourly[k] = v / 100.0
        rows += window_rows("open_meteo", int(grid), issue, hourly)
    return rows


def fetch_google(cells, issue, key):
    rows = []
    for grid, lat, lon in zip(cells.grid_number, cells.latitude, cells.longitude):
        url = ("https://weather.googleapis.com/v1/forecast/hours:lookup?key=" + key +
               f"&location.latitude={lat:.4f}&location.longitude={lon:.4f}&hours=8")
        d = get_json(url)
        hourly = {}
        for fh in d.get("forecastHours", []):
            st = fh.get("interval", {}).get("startTime", "")
            try:
                t_utc = datetime.strptime(st[:19], "%Y-%m-%dT%H:%M:%S")
            except ValueError:
                continue
            k = int(round((t_utc + TZ_OFFSET - issue).total_seconds() / 3600))
            pr = fh.get("precipitation", {}).get("probability", {}).get("percent")
            if 1 <= k <= 6 and pr is not None:
                hourly[k] = pr / 100.0
        rows += window_rows("google", int(grid), issue, hourly)
    return rows


def fetch_accuweather(cells, issue, key):
    cache = {}
    if ACCU_CACHE.exists():
        cache = json.loads(ACCU_CACHE.read_text(encoding="utf-8"))
    rows = []
    sub = cells[cells.grid_number == CENTRAL_GRID]
    for grid, lat, lon in zip(sub.grid_number, sub.latitude, sub.longitude):
        gkey = str(int(grid))
        if gkey not in cache:
            d = get_json("https://dataservice.accuweather.com/locations/v1/cities/geoposition/search"
                         f"?apikey={key}&q={lat:.4f}%2C{lon:.4f}")
            cache[gkey] = d["Key"]
            ACCU_CACHE.parent.mkdir(exist_ok=True)
            ACCU_CACHE.write_text(json.dumps(cache), encoding="utf-8")
        d = get_json(f"https://dataservice.accuweather.com/forecasts/v1/hourly/12hour/{cache[gkey]}"
                     f"?apikey={key}&metric=true")
        hourly = {}
        for fh in d:
            t_local = datetime.strptime(fh["DateTime"][:19], "%Y-%m-%dT%H:%M:%S")
            k = int(round((t_local - issue).total_seconds() / 3600))
            if 1 <= k <= 6 and fh.get("PrecipitationProbability") is not None:
                hourly[k] = fh["PrecipitationProbability"] / 100.0
        rows += window_rows("accuweather", int(grid), issue, hourly)
    return rows


def fetch_weatherapi(cells, issue, key):
    rows = []
    for grid, lat, lon in zip(cells.grid_number, cells.latitude, cells.longitude):
        d = get_json("https://api.weatherapi.com/v1/forecast.json?key=" + key +
                     f"&q={lat:.4f},{lon:.4f}&days=2&aqi=no&alerts=no")
        hourly = {}
        for day in d.get("forecast", {}).get("forecastday", []):
            for fh in day.get("hour", []):
                t_local = datetime.strptime(fh["time"], "%Y-%m-%d %H:%M")
                k = int(round((t_local - issue).total_seconds() / 3600))
                if 1 <= k <= 6 and fh.get("chance_of_rain") is not None:
                    hourly[k] = float(fh["chance_of_rain"]) / 100.0
        rows += window_rows("weatherapi", int(grid), issue, hourly)
    return rows


def fetch_tomorrow(cells, issue, key):
    rows = []
    for grid, lat, lon in zip(cells.grid_number, cells.latitude, cells.longitude):
        d = get_json("https://api.tomorrow.io/v4/weather/forecast?location=" +
                     f"{lat:.4f},{lon:.4f}&timesteps=1h&apikey={key}")
        hourly = {}
        for fh in d.get("timelines", {}).get("hourly", []):
            t_utc = datetime.strptime(fh["time"][:19], "%Y-%m-%dT%H:%M:%S")
            k = int(round((t_utc + TZ_OFFSET - issue).total_seconds() / 3600))
            pr = fh.get("values", {}).get("precipitationProbability")
            if 1 <= k <= 6 and pr is not None:
                hourly[k] = float(pr) / 100.0
        rows += window_rows("tomorrow", int(grid), issue, hourly)
        time.sleep(0.5)     # free tier: 25 requests/hour
    return rows


def fetch_openweathermap(cells, issue, key):
    rows = []
    for grid, lat, lon in zip(cells.grid_number, cells.latitude, cells.longitude):
        d = get_json("https://api.openweathermap.org/data/2.5/forecast?" +
                     f"lat={lat:.4f}&lon={lon:.4f}&appid={key}")
        blocks = [(datetime.utcfromtimestamp(b["dt"]) + TZ_OFFSET, float(b.get("pop", 0.0)))
                  for b in d.get("list", [])]
        hourly = {}
        for k in range(1, 7):
            t = issue + timedelta(hours=k)
            for start, pop in blocks:
                if start <= t < start + timedelta(hours=3):
                    hourly[k] = pop
                    break
        rows += window_rows("openweathermap", int(grid), issue, hourly)
    return rows


PROVIDERS = {
    "open_meteo": (fetch_open_meteo, None),
    "google": (fetch_google, "google"),
    "accuweather": (fetch_accuweather, "accuweather"),
    "weatherapi": (fetch_weatherapi, "weatherapi"),
    "tomorrow": (fetch_tomorrow, "tomorrow"),
    "openweathermap": (fetch_openweathermap, "openweathermap"),
}


# ----------------------------------------------------------------------------- fetch

def model_rows(cells, issue):
    rows = []
    for _, r in cells.iterrows():
        for h in HORIZONS:
            rows.append({"provider": "model", "grid_number": int(r.grid_number), "horizon_h": h,
                         "p_0.1mm": float(r[f"p_h{h}_0.1mm"]), "p_1.0mm": float(r[f"p_h{h}_1.0mm"]),
                         "hourly_pop": ""})
    return rows


def climatology_rows(cells, issue):
    if not CLIM_PATH.exists():
        return []
    clim = pd.read_csv(CLIM_PATH)
    sel = clim[(clim.month == issue.month) & (clim.hour == issue.hour) &
               clim.grid_number.isin(cells.grid_number)].set_index("grid_number")
    rows = []
    for grid in cells.grid_number:
        if int(grid) not in sel.index:
            continue
        for h in HORIZONS:
            rows.append({"provider": "climatology", "grid_number": int(grid), "horizon_h": h,
                         "p_0.1mm": float(sel.loc[int(grid), f"p_h{h}_0.1mm"]),
                         "p_1.0mm": float(sel.loc[int(grid), f"p_h{h}_1.0mm"]), "hourly_pop": ""})
    return rows


def append_rows(path, rows, key_cols):
    if not rows:
        return 0
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        merged = pd.concat([old, new], ignore_index=True).drop_duplicates(subset=key_cols, keep="last")
    else:
        merged = new
    merged = merged.sort_values(key_cols)
    tmp = path.with_suffix(".tmp")
    merged.to_csv(tmp, index=False)
    os.replace(tmp, path)
    return len(new)


def cmd_fetch(args):
    if args.stamp:
        stamp = args.stamp
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H00")
    issue = datetime.strptime(stamp, "%Y%m%d_%H%M")
    cells, have_model = bkk_cells(stamp)
    log(f"fetch {stamp}: {len(cells)} Bangkok cells, model CSV {'present' if have_model else 'MISSING'}")
    keys = api_keys()
    wanted = [p.strip() for p in args.providers.split(",")] if args.providers else list(PROVIDERS)
    rows = []
    if have_model:
        rows += model_rows(cells, issue)
    rows += climatology_rows(cells, issue)
    for name in wanted:
        if name in ("none", ""):
            continue
        if name not in PROVIDERS:
            log(f"  unknown provider {name}")
            continue
        fn, keyname = PROVIDERS[name]
        if keyname and keyname not in keys:
            log(f"  {name}: skipped (no API key)")
            continue
        try:
            got = fn(cells, issue, keys[keyname]) if keyname else fn(cells, issue)
            log(f"  {name}: {len(got)} rows")
            rows += got
        except Exception as exc:
            log(f"  {name}: FAILED {type(exc).__name__}: {str(exc)[:160]}")
    fetched = datetime.now().strftime("%Y-%m-%d %H:%M")
    for r in rows:
        r.update({"stamp": stamp, "issue_local": issue.strftime("%Y-%m-%d %H:%M:%S"),
                  "fetched_at": fetched})
    n = append_rows(FORECAST_LOG, rows, ["stamp", "provider", "grid_number", "horizon_h"])
    log(f"  logged {n} rows -> {FORECAST_LOG.name}")


# ----------------------------------------------------------------------------- score

def metrics(y, p, flag_thr=0.5):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    f = (p >= flag_thr).astype(int)
    tp = int(((f == 1) & (y == 1)).sum()); fp = int(((f == 1) & (y == 0)).sum())
    fn = int(((f == 0) & (y == 1)).sum())
    roc = np.nan
    if 0 < y.mean() < 1:
        from sklearn.metrics import roc_auc_score
        roc = float(roc_auc_score(y, p))
    return {"n": int(len(y)), "base_rate": round(float(y.mean()), 4),
            "brier": round(float(np.mean((np.clip(p, 0, 1) - y) ** 2)), 4),
            "roc_auc": round(roc, 4) if np.isfinite(roc) else np.nan,
            "pod": round(tp / (tp + fn), 3) if tp + fn else np.nan,
            "far": round(fp / (tp + fp), 3) if tp + fp else np.nan,
            "csi": round(tp / (tp + fp + fn), 3) if tp + fp + fn else np.nan}


def label_hours(issue, h):
    """IMERG rows (start-stamped) covering the window [issue, issue+h)."""
    return [issue + timedelta(hours=k) for k in range(h)]


def persistence_hours(issue, h):
    """The h observed hours before the issue hour, oldest first: [issue-h, issue)."""
    return [issue - timedelta(hours=k) for k in range(h, 0, -1)]


def window_bounds(pops):
    """(lower, upper) bounds on P(any rain in the window) from the hourly marginals p_k."""
    return max(float(p) for p in pops), combine(pops)


def load_imerg(lo, hi):
    import psycopg2
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        cur = conn.cursor()
        cur.execute(f"SELECT grid_number, local_observation_time, {IMERG_LABEL_COLUMN}, precipitation_mm "
                    f"FROM {IMERG_TABLE} WHERE is_complete_hour AND grid_number = ANY(%s) "
                    f"AND local_observation_time BETWEEN %s AND %s", (BKK_GRIDS, lo, hi))
        df = pd.DataFrame(cur.fetchall(), columns=["grid_number", "t", "pmax", "pmean"])
    finally:
        conn.close()
    if df.empty:
        return None, None
    df["t"] = pd.to_datetime(df["t"])
    return (df.pivot(index="grid_number", columns="t", values="pmax"),
            df.pivot(index="grid_number", columns="t", values="pmean"))


def cmd_score(args):
    if not FORECAST_LOG.exists():
        log("nothing to score yet")
        return
    fc = pd.read_csv(FORECAST_LOG, parse_dates=["issue_local"])
    issues = sorted(fc.issue_local.unique())
    lo = pd.Timestamp(min(issues)) - timedelta(hours=5)
    hi = pd.Timestamp(max(issues)) + timedelta(hours=6)
    pmax, pmean = load_imerg(lo.to_pydatetime(), hi.to_pydatetime())
    if pmax is None:
        log("no IMERG labels in range yet")
        return

    def label(kind, grid, issue, h, thr):
        src = pmax if kind == "cellmax" else pmean
        hours = label_hours(issue, h)
        if grid not in src.index or any(t not in src.columns for t in hours):
            return None
        vals = src.loc[grid, hours].to_numpy(dtype=float)
        if np.isnan(vals).any():
            return None
        return int((vals >= thr).any())

    def persistence(grid, issue, h, thr):
        hours = persistence_hours(issue, h)
        if grid not in pmax.index or any(t not in pmax.columns for t in hours):
            return None
        vals = pmax.loc[grid, hours].to_numpy(dtype=float)
        if np.isnan(vals).any():
            return None
        return float(np.clip(vals.max() / thr, 0, 1))

    # long table: one row per (issue, grid, h, provider) with labels attached
    recs = []
    pcol = {thr: list(fc.columns).index(f"p_{thr}mm") for thr in THRESHOLDS}
    for r in fc.itertuples(index=False):
        issue = r.issue_local.to_pydatetime()
        pops = (json.loads(r.hourly_pop)
                if isinstance(r.hourly_pop, str) and r.hourly_pop.startswith("[") else None)
        for kind in ("cellmax", "cellmean"):
            for thr in THRESHOLDS:
                p = r[pcol[thr]]      # positional: pandas mangles dotted column names
                if p is None or (isinstance(p, float) and np.isnan(p)):
                    continue
                y = label(kind, int(r.grid_number), issue, int(r.horizon_h), thr)
                if y is None:
                    continue
                # multi-hour provider rows are bounds, not probabilities (module docstring)
                variants = [(r.provider, float(p))]
                if pops and int(r.horizon_h) > 1:
                    lower, upper = window_bounds(pops)
                    variants = [(f"{r.provider}[max]", lower), (f"{r.provider}[1-prod]", upper)]
                for prov, pv in variants:
                    recs.append({"stamp": r.stamp, "issue_local": issue,
                                 "grid_number": int(r.grid_number), "horizon_h": int(r.horizon_h),
                                 "provider": prov, "label": kind, "threshold_mm": thr,
                                 "p": pv, "y": y})
        # persistence, once per (issue, grid, h)
        if r.provider == "model":
            for kind in ("cellmax", "cellmean"):
                for thr in THRESHOLDS:
                    pp = persistence(int(r.grid_number), issue, int(r.horizon_h), thr)
                    y = label(kind, int(r.grid_number), issue, int(r.horizon_h), thr)
                    if pp is None or y is None:
                        continue
                    recs.append({"stamp": r.stamp, "issue_local": issue, "grid_number": int(r.grid_number),
                                 "horizon_h": int(r.horizon_h), "provider": "persistence",
                                 "label": kind, "threshold_mm": thr, "p": pp, "y": y})
    if not recs:
        log("forecasts logged but none have complete labels yet")
        return
    L = pd.DataFrame(recs)

    # paired comparison: each forecaster vs the model on identical (stamp, grid, h) keys
    out = []
    model = L[L.provider == "model"]
    for (prov, kind, thr, h), g in L.groupby(["provider", "label", "threshold_mm", "horizon_h"]):
        keys = g[["stamp", "grid_number"]].drop_duplicates()
        m = model[(model.label == kind) & (model.threshold_mm == thr) & (model.horizon_h == h)]
        m = m.merge(keys, on=["stamp", "grid_number"])
        g2 = g.merge(m[["stamp", "grid_number"]], on=["stamp", "grid_number"])
        if len(g2) < 20 or len(m) < 20:
            continue
        row = {"provider": prov, "label": kind, "threshold_mm": thr, "horizon_h": h,
               "issue_hours": int(g2.stamp.nunique()), "first_issue": str(g2.issue_local.min())[:16],
               "last_issue": str(g2.issue_local.max())[:16]}
        row.update({f"{k}": v for k, v in metrics(g2.y, g2.p).items()})
        mm = metrics(m.y, m.p)
        row.update({f"model_{k}": v for k, v in mm.items() if k in ("brier", "roc_auc", "pod", "far", "csi")})
        out.append(row)
    S = pd.DataFrame(out).sort_values(["label", "threshold_mm", "horizon_h", "provider"])
    S.to_csv(SCORES_CSV, index=False)
    log(f"scored {len(L):,} forecast-label pairs over {L.stamp.nunique()} issue hours -> {SCORES_CSV.name}")
    write_summary(S)


def write_summary(S):
    lines = ["# Bangkok: live model vs public weather services (labels_v2)", "",
             f"Generated {datetime.now():%Y-%m-%d %H:%M}. Nine 25 km cells over the Bangkok metro area, "
             "graded against IMERG. Each provider row is paired with the model on exactly the same "
             "issue hours and cells; 'model' columns are the model on that provider's sample. "
             "Flags use p >= 0.5 for every forecaster. Providers are graded at 0.1 mm only "
             "(their probability means 'measurable rain at the point'). Labels: wettest IMERG "
             "pixel's hourly total over the window [issue, issue+h); provider hour k refers to the "
             "preceding hour, so hours 1..h cover that same window. The 1 h rows are the primary "
             "comparison. Multi-hour provider rows are BOUNDS on P(any rain in the window), not "
             "probabilities: [max] = max of the hourly values (lower bound), [1-prod] = the "
             "independence product (upper bound under positive hour-to-hour correlation).", ""]
    for kind, title in (("cellmax", "Label: rain anywhere in the cell (project label, IMERG cell-max)"),
                        ("cellmean", "Label: area-average rain (IMERG cell-mean, closer to a point forecast)")):
        lines += [f"## {title}", ""]
        for thr in THRESHOLDS:
            sub = S[(S.label == kind) & (S.threshold_mm == thr)]
            if sub.empty:
                continue
            lines += [f"### Threshold {thr} mm", "",
                      "| Horizon | Forecaster | Issue hours | n | Wet rate | ROC-AUC | Brier | POD | FAR | CSI | Model ROC-AUC (same rows) | Model Brier |",
                      "|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for r in sub.itertuples(index=False):
                f = lambda v, d=2: "—" if (v is None or (isinstance(v, float) and np.isnan(v))) else f"{v:.{d}f}"
                lines.append(f"| {r.horizon_h} h | {r.provider} | {r.issue_hours} | {r.n} | {f(r.base_rate)} | "
                             f"{f(r.roc_auc)} | {f(r.brier, 3)} | {f(r.pod)} | {f(r.far)} | {f(r.csi)} | "
                             f"{f(r.model_roc_auc)} | {f(r.model_brier, 3)} |")
            lines.append("")
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")
    log(f"summary -> {SUMMARY_MD.name}")


def cmd_report(args):
    if SUMMARY_MD.exists():
        print(SUMMARY_MD.read_text(encoding="utf-8"))
    else:
        print("no summary yet - run `score` once labels have arrived")


# ----------------------------------------------------------------------------- excel export

XLSX_PATH = OUT_DIR / f"provider_comparison_{LABEL_VERSION}.xlsx"
PROVIDER_ORDER = ["model", "open_meteo", "google", "accuweather", "weatherapi", "tomorrow",
                  "openweathermap", "climatology", "persistence"]


def cmd_export(args):
    """provider_comparison.xlsx: paired scores, central-cell forecasts side by side, raw logs."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    FONT = "Arial"
    base = Font(name=FONT, size=10)
    bold = Font(name=FONT, size=10, bold=True)
    title = Font(name=FONT, size=13, bold=True)
    note = Font(name=FONT, size=9, italic=True, color="555555")
    head_fill = PatternFill("solid", fgColor="DCE6F1")
    model_fill = PatternFill("solid", fgColor="EAF3E4")
    thin = Side(style="thin", color="BFBFBF")
    box = Border(top=thin, bottom=thin, left=thin, right=thin)

    def write_table(ws, r0, headers, rows, fmts=None, widths=None):
        for j, h in enumerate(headers, 1):
            c = ws.cell(row=r0, column=j, value=h)
            c.font, c.fill, c.border = bold, head_fill, box
            c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        for i, row in enumerate(rows, 1):
            for j, v in enumerate(row, 1):
                if isinstance(v, float) and np.isnan(v):
                    v = None
                c = ws.cell(row=r0 + i, column=j, value=v)
                c.font, c.border = base, box
                if fmts and fmts[j - 1]:
                    c.number_format = fmts[j - 1]
        if widths:
            for j, w in enumerate(widths, 1):
                ws.column_dimensions[get_column_letter(j)].width = w
        return r0 + len(rows)

    wb = Workbook()

    # ---- Sheet 1: Summary (paired scores)
    ws = wb.active
    ws.title = "Summary"
    ws["A1"] = "Bangkok: live model vs public weather services"
    ws["A1"].font = title
    ws["A2"] = (f"Refreshed {datetime.now():%Y-%m-%d %H:%M}. Nine 25 km cells over the Bangkok metro "
                "area, graded against IMERG. Each provider row is paired with the model on exactly the "
                "same issue hours and cells; the two right-hand columns are the model on that provider's "
                "sample. Flags use p >= 0.5 for every forecaster. Providers are graded at 0.1 mm only.")
    ws["A2"].font = note
    ws["A2"].alignment = Alignment(wrap_text=True, vertical="top")
    ws.merge_cells("A2:M2")
    ws.row_dimensions[2].height = 42
    r = 4
    S = pd.read_csv(SCORES_CSV) if SCORES_CSV.exists() else pd.DataFrame()
    if S.empty:
        ws.cell(row=r, column=1, value="No scored rows yet. IMERG labels arrive about 34 hours after "
                "each issue hour; this sheet fills in automatically on the next hourly run once they do.").font = note
    else:
        S["_o"] = S.provider.map({p: i for i, p in enumerate(PROVIDER_ORDER)}).fillna(99)
        for kind, heading in (("cellmax", "Label: rain anywhere in the cell (project label, IMERG cell-max)"),
                              ("cellmean", "Label: area-average rain (IMERG cell-mean, closer to a point forecast)")):
            for thr in THRESHOLDS:
                sub = S[(S.label == kind) & (S.threshold_mm == thr)].sort_values(["horizon_h", "_o"])
                if sub.empty:
                    continue
                ws.cell(row=r, column=1, value=f"{heading} - threshold {thr} mm").font = bold
                r += 1
                headers = ["Horizon (h)", "Forecaster", "Issue hours", "n", "Wet rate", "ROC-AUC",
                           "Brier", "POD", "FAR", "CSI", "Model ROC-AUC (same rows)", "Model Brier (same rows)",
                           "ROC lift (model - forecaster)"]
                rows, first = [], r + 1
                for k, x in enumerate(sub.itertuples(index=False)):
                    rr = first + k
                    rows.append([x.horizon_h, x.provider, x.issue_hours, x.n, x.base_rate, x.roc_auc,
                                 x.brier, x.pod, x.far, x.csi, x.model_roc_auc, x.model_brier,
                                 f'=IF(OR(F{rr}="",K{rr}=""),"",K{rr}-F{rr})'])
                fm = [None, None, None, None, "0.00", "0.000", "0.000", "0.00", "0.00", "0.00", "0.000", "0.000", "+0.000;-0.000;0.000"]
                r = write_table(ws, r, headers, rows, fm,
                                [11, 15, 11, 8, 9, 9, 9, 8, 8, 8, 13, 12, 14])
                for k, x in enumerate(sub.itertuples(index=False)):
                    if x.provider == "model":
                        for j in range(1, len(headers) + 1):
                            ws.cell(row=first + k, column=j).fill = model_fill
                r += 2
    ws.freeze_panes = "A4"

    # ---- Sheet 2: Central cell forecasts side by side (+ observed when known)
    ws2 = wb.create_sheet("Central cell")
    ws2["A1"] = f"Grid {CENTRAL_GRID} (central Bangkok): probability of rain in the next h hours, by forecaster"
    ws2["A1"].font = title
    ws2["A2"] = ("Model and climatology at 0.1 mm; providers are their own precipitation-probability "
                 "combined over the window (1 - prod(1 - p)). Observed = IMERG cell-max >= 0.1 mm in the "
                 "window (blank until labels arrive). Brier rows recompute from this sheet.")
    ws2["A2"].font = note
    ws2["A2"].alignment = Alignment(wrap_text=True, vertical="top")
    ws2.merge_cells("A2:L2")
    ws2.row_dimensions[2].height = 30
    fc = pd.read_csv(FORECAST_LOG, parse_dates=["issue_local"]) if FORECAST_LOG.exists() else pd.DataFrame()
    obs = {}
    if not fc.empty:
        try:
            lo = fc.issue_local.min() - timedelta(hours=1)
            hi = fc.issue_local.max() + timedelta(hours=6)
            pmax, _ = load_imerg(lo.to_pydatetime(), hi.to_pydatetime())
            if pmax is not None and CENTRAL_GRID in pmax.index:
                for issue in fc.issue_local.unique():
                    issue = pd.Timestamp(issue).to_pydatetime()
                    for h in HORIZONS:
                        hrs = [issue + timedelta(hours=k) for k in range(1, h + 1)]
                        if all(t in pmax.columns for t in hrs):
                            v = pmax.loc[CENTRAL_GRID, hrs].to_numpy(dtype=float)
                            if not np.isnan(v).any():
                                obs[(issue, h)] = int((v >= 0.1).any())
        except Exception as exc:
            log(f"export: observed column skipped ({exc})")
    r = 4
    if fc.empty:
        ws2.cell(row=r, column=1, value="No forecasts logged yet.").font = note
    else:
        cc = fc[fc.grid_number == CENTRAL_GRID]
        provs = [p for p in PROVIDER_ORDER if p in set(cc.provider)] + \
                sorted(set(cc.provider) - set(PROVIDER_ORDER))
        for h in HORIZONS:
            ws2.cell(row=r, column=1, value=f"Next {h} hour{'s' if h > 1 else ''}").font = bold
            r += 1
            piv = cc[cc.horizon_h == h].pivot_table(index="issue_local", columns="provider",
                                                    values="p_0.1mm", aggfunc="last")
            headers = ["Issue (local)"] + provs + ["Observed (0/1)"]
            rows, first = [], r + 1
            for issue, rowv in piv.iterrows():
                rows.append([issue.to_pydatetime()] + [rowv.get(p, np.nan) for p in provs] +
                            [obs.get((issue.to_pydatetime(), h), None)])
            last = first + len(rows) - 1
            fm = ["yyyy-mm-dd hh:mm"] + ["0.00"] * len(provs) + ["0"]
            r = write_table(ws2, r, headers, rows, fm, [17] + [13] * len(provs) + [14])
            obs_col = get_column_letter(len(provs) + 2)
            # Brier per forecaster over the rows that have an observation (live formulas)
            ws2.cell(row=r + 1, column=1, value="Mean probability").font = bold
            ws2.cell(row=r + 2, column=1, value="Brier vs observed").font = bold
            ws2.cell(row=r + 3, column=1, value="Hours with observation").font = bold
            for j, p in enumerate(provs, 2):
                col = get_column_letter(j)
                c = ws2.cell(row=r + 1, column=j, value=f'=IF(COUNT({col}{first}:{col}{last})=0,"",AVERAGE({col}{first}:{col}{last}))')
                c.number_format, c.font = "0.00", base
                c = ws2.cell(row=r + 2, column=j,
                             value=f'=IF(COUNT({obs_col}{first}:{obs_col}{last})=0,"",'
                                   f'SUMPRODUCT(({obs_col}{first}:{obs_col}{last}<>"")*({col}{first}:{col}{last}<>"")*'
                                   f'({col}{first}:{col}{last}-{obs_col}{first}:{obs_col}{last})^2)'
                                   f'/SUMPRODUCT(({obs_col}{first}:{obs_col}{last}<>"")*({col}{first}:{col}{last}<>"")))')
                c.number_format, c.font = "0.000", base
                c = ws2.cell(row=r + 3, column=j,
                             value=f'=SUMPRODUCT(({obs_col}{first}:{obs_col}{last}<>"")*({col}{first}:{col}{last}<>""))')
                c.font = base
            r += 5
    ws2.freeze_panes = "B4"

    # ---- Sheet 3/4: raw logs
    def raw_sheet(name, df, fmts=None):
        w = wb.create_sheet(name)
        if df is None or df.empty:
            w["A1"] = "empty"
            w["A1"].font = note
            return
        widths = []
        for c in df.columns:
            longest = df[c].dropna().astype(str).str.len().max()
            longest = 0 if pd.isna(longest) else int(longest)
            widths.append(max(10, min(22, max(longest, len(str(c))) + 2)))
        write_table(w, 1, list(df.columns), df.values.tolist(), fmts, widths)
        w.freeze_panes = "A2"
    if not fc.empty:
        fc2 = fc.copy()
        fc2["issue_local"] = fc2["issue_local"].dt.strftime("%Y-%m-%d %H:%M")
        raw_sheet("Forecast log", fc2.sort_values(["stamp", "provider", "grid_number", "horizon_h"]))
    else:
        raw_sheet("Forecast log", None)
    raw_sheet("Scores (full)", S.drop(columns=["_o"], errors="ignore") if not S.empty else None)

    # ---- Sheet 5: notes
    wn = wb.create_sheet("Notes")
    notes = [
        ("What this compares", "The live V13 nationwide model against public weather services for the nine 25 km grid "
         "cells whose centres fall over the Bangkok metro area (grid 183 is central Bangkok)."),
        ("Issue hour", "Local time (Asia/Bangkok) the forecast was made. The model runs at about 10 minutes past "
         "the hour; providers are queried at 20 past. Hours the laptop was asleep have no rows."),
        ("Windows", "Next 1, 3 and 6 hours after the issue hour."),
        ("Provider probability", "Each service's hourly precipitation probability, combined over the window as "
         "1 - prod(1 - p_k) assuming independent hours. It means 'measurable rain at the point', so providers "
         "are graded at the 0.1 mm threshold only. OpenWeatherMap gives 3-hourly pop, applied to each hour in "
         "the block. AccuWeather is fetched for the central cell only (free tier: 50 calls/day)."),
        ("Labels", "IMERG satellite rain, complete hours only. 'cellmax' = rain anywhere in the 25 km cell "
         "(the label the model was trained on). 'cellmean' = area-average rain, fairer to point forecasts."),
        ("Baselines", "climatology = the cell's observed wet frequency for this month and hour (2024-07..2026-07, "
         "+-1 h pooled). persistence = observed IMERG at issue time carried forward; it uses data the live "
         "model does not have (IMERG latency > 4 h), so it is a ceiling, not a competitor."),
        ("Metrics", "ROC-AUC pooled over cells and issue hours; Brier = mean squared probability error; "
         "POD / FAR / CSI at a common p >= 0.5 flag. Differences in ROC-AUC under about 0.05 are not "
         "meaningful at this sample size."),
        ("Pairing", "On the Summary sheet every forecaster is scored on exactly the rows it has in common "
         "with the model, and the model's own score on those rows sits beside it."),
        ("Files", "provider_forecasts.csv (raw log), provider_scores.csv (scored), provider_comparison.md "
         "(Markdown table), all next to the V10 prediction CSVs. Refreshed hourly by the 'Rainwatch Provider "
         "Compare' scheduled task."),
    ]
    wn.column_dimensions["A"].width = 22
    wn.column_dimensions["B"].width = 110
    for i, (k, v) in enumerate(notes, 1):
        wn.cell(row=i, column=1, value=k).font = bold
        c = wn.cell(row=i, column=2, value=v)
        c.font = base
        c.alignment = Alignment(wrap_text=True, vertical="top")
        wn.row_dimensions[i].height = 15 * (1 + len(v) // 105)

    for w in wb.worksheets:
        for row in w.iter_rows():
            for c in row:
                if c.font is None or c.font.name != FONT:
                    c.font = Font(name=FONT, size=c.font.size if c.font else 10,
                                  bold=c.font.bold if c.font else False,
                                  italic=c.font.italic if c.font else False,
                                  color=c.font.color if c.font else None)
    tmp = XLSX_PATH.with_suffix(".tmp.xlsx")
    wb.save(tmp)
    os.replace(tmp, XLSX_PATH)
    log(f"workbook -> {XLSX_PATH.name}")


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "score", "report", "export", "run"])
    ap.add_argument("--stamp", default=None, help="YYYYMMDD_HHMM issue hour (default: current hour)")
    ap.add_argument("--providers", default=None, help="comma list, or 'none' (default: all)")
    ap.add_argument("--log-file", default=None)
    args = ap.parse_args()
    if args.log_file:
        lf = Path(args.log_file)
        lf.parent.mkdir(parents=True, exist_ok=True)
        if lf.exists() and lf.stat().st_size > 2_000_000:
            lf.replace(lf.with_name(lf.name + ".1"))
        sys.stdout = sys.stderr = open(lf, "a", encoding="utf-8", buffering=1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.cmd in ("fetch", "run"):
        cmd_fetch(args)
    if args.cmd in ("score", "run"):
        try:
            cmd_score(args)
        except Exception as exc:
            log(f"score FAILED {type(exc).__name__}: {exc}")
    if args.cmd in ("export", "run"):
        try:
            cmd_export(args)
        except Exception as exc:
            log(f"export FAILED {type(exc).__name__}: {exc}")
    if args.cmd == "report":
        cmd_report(args)


if __name__ == "__main__":
    main()
