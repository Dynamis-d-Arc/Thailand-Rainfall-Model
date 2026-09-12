"""Build the IMERG climatology table used as a verification baseline.

For every (grid cell, calendar month, local hour of day) and every deployed target
(h in 1/3/6 hours x threshold 0.1/1.0 mm) this stores the observed frequency of the
label "IMERG cell-max >= threshold in ANY of the next h hours" - exactly the label
definition verify_imerg.py scores the live model against.

The climatology is the "no-skill" reference: a forecaster who knows only where,
when in the year and when in the day it usually rains. Hours are pooled +-1 h
(circular) so each (cell, month, hour) rests on ~180 samples instead of ~60.

History window: 2024-07-01 .. 2026-07-31 (two full wet seasons); everything from
2026-08 onward is left out so the baseline never sees the hours it is scored on.

Usage:
    python Dashboard/build_climatology.py            # writes v10_climatology.csv.gz
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

ROOT = Path(__file__).resolve().parent
PROJ = ROOT.parent
PRED_DIR = PROJ / "ML_Model_V2" / "trained_models" / "om_thailand_rain_v10_deploy"
OUT = PRED_DIR / "v10_climatology.csv.gz"
META = PRED_DIR / "v10_climatology_meta.json"

IMERG_TABLE = '"IMERG_THAILAND_DATA"'
HIST_LO = "2024-07-01 00:00:00"
HIST_HI = "2026-08-01 00:00:00"      # exclusive
TARGETS = [(1, 0.1), (3, 0.1), (6, 0.1), (1, 1.0), (3, 1.0), (6, 1.0)]
POOL_HOURS = 1                       # +-1 h circular pooling of the diurnal bin

DB_CONFIG = {
    "host": os.getenv("PGHOST", "localhost"),
    "port": int(os.getenv("PGPORT", "5432")),
    "dbname": os.getenv("PGDATABASE", "postgres"),
    "user": os.getenv("PGUSER", "postgres"),
    "password": os.getenv("PGPASSWORD", "Pass1234"),
}


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def load_pmax():
    """grid x hourly-local-time matrix of precipitation_max_mm (NaN = missing hour)."""
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        cur = conn.cursor(name="clim_cur")
        cur.itersize = 500_000
        cur.execute(
            f"SELECT grid_number, local_observation_time, precipitation_max_mm "
            f"FROM {IMERG_TABLE} WHERE is_complete_hour "
            f"AND local_observation_time >= %s AND local_observation_time < %s",
            (HIST_LO, HIST_HI))
        chunks = []
        while True:
            rows = cur.fetchmany(cur.itersize)
            if not rows:
                break
            chunks.append(pd.DataFrame(rows, columns=["grid", "t", "pmax"]))
            log(f"  fetched {sum(len(c) for c in chunks):,} rows")
        cur.close()
    finally:
        conn.close()
    df = pd.concat(chunks, ignore_index=True)
    df["t"] = pd.to_datetime(df["t"])
    mat = df.pivot(index="grid", columns="t", values="pmax")
    full = pd.date_range(HIST_LO, HIST_HI, freq="h", inclusive="left")
    mat = mat.reindex(columns=full)
    log(f"matrix {mat.shape[0]} cells x {mat.shape[1]} hours, "
        f"{mat.isna().to_numpy().mean():.1%} missing")
    return mat


def forward_window_max(values, h):
    """max over columns t+1..t+h for each t; NaN where any of those hours is missing."""
    n_cells, n_t = values.shape
    out = np.full((n_cells, n_t), np.nan)
    ok = np.ones((n_cells, n_t), dtype=bool)
    stack = np.full((h, n_cells, n_t), np.nan)
    for k in range(1, h + 1):
        stack[k - 1, :, :n_t - k] = values[:, k:]
    ok = ~np.isnan(stack).any(axis=0)
    with np.errstate(all="ignore"):
        mx = np.nanmax(np.where(np.isnan(stack), -np.inf, stack), axis=0)
    out[ok] = mx[ok]
    return out


def main():
    log("loading IMERG history")
    mat = load_pmax()
    values = mat.to_numpy(dtype=float)
    times = mat.columns
    month = times.month.to_numpy()
    hour = times.hour.to_numpy()
    grids = mat.index.to_numpy()

    # per (month, hour) bin: sum of positives and count of valid samples, per cell
    sums = {}
    counts = {}
    for h, thr in TARGETS:
        name = f"h{h}_{thr}mm"
        wmax = forward_window_max(values, h)
        valid = ~np.isnan(wmax)
        pos = valid & (wmax >= thr)
        s = np.zeros((len(grids), 12, 24))
        c = np.zeros((len(grids), 12, 24))
        for m in range(1, 13):
            for hh in range(24):
                sel = (month == m) & (hour == hh)
                s[:, m - 1, hh] = pos[:, sel].sum(axis=1)
                c[:, m - 1, hh] = valid[:, sel].sum(axis=1)
        sums[name], counts[name] = s, c
        log(f"  {name}: base rate {pos.sum() / max(valid.sum(), 1):.4f}")

    # circular +-POOL_HOURS pooling of the diurnal bins
    def pool(a):
        return sum(np.roll(a, k, axis=2) for k in range(-POOL_HOURS, POOL_HOURS + 1))

    rows = []
    g_idx, m_idx, h_idx = np.meshgrid(np.arange(len(grids)), np.arange(12), np.arange(24),
                                      indexing="ij")
    out = pd.DataFrame({
        "grid_number": grids[g_idx.ravel()],
        "month": (m_idx.ravel() + 1).astype(np.int16),
        "hour": h_idx.ravel().astype(np.int16),
    })
    for name in sums:
        ps, pc = pool(sums[name]), pool(counts[name])
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"p_{name}"] = (ps / pc).ravel().astype(np.float32)
        out[f"n_{name}"] = pc.ravel().astype(np.int32)
    out.to_csv(OUT, index=False, float_format="%.4f", compression="gzip")
    meta = {
        "built": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "history": [HIST_LO[:10], "2026-07-31"],
        "pool_hours": POOL_HOURS,
        "rows": int(len(out)),
        "label": "IMERG precipitation_max_mm >= thr in ANY of next h hours, complete hours only",
        "samples_per_bin_median": int(np.median(out["n_h1_0.1mm"])),
    }
    META.write_text(json.dumps(meta, indent=1))
    log(f"wrote {OUT.name} ({len(out):,} rows, median {meta['samples_per_bin_median']} "
        f"samples per bin)")


if __name__ == "__main__":
    main()
