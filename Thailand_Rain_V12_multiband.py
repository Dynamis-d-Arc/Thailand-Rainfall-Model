# %% [markdown]
# # Thailand_Rain_V12 — multi-band Himawari vs the warm-rain ceiling
#
# V9 closed the 2026-decay case with a physical mechanism: 10.4 um IR (B13) cannot see
# warm/shallow rain, and p(cold-top | rain) fell 0.85 -> 0.70. This experiment asks whether
# other AHI 2-km bands recover part of what B13 misses:
#
# | band | wavelength | why it might see warm rain |
# |---|---|---|
# | B07 | 3.9 um  | low-cloud/drizzle microphysics (night); solar-contaminated by day |
# | B11 | 8.6 um  | B11-B13 BTD discriminates cloud phase — water tops vs ice tops |
# | B15 | 12.4 um | B13-B15 split window tracks optical depth / low warm cloud |
#
# Pilot-first, exactly like V7: fetch the three bands ONLY for the warm-rain regime months
# (May-Jul 2026), then a no-training screen on the decisive slice — labeled rows with
# **no cold top at all** (hw_cold235_env_lag1 == 0), the rain B13 cannot rank by
# construction. Pre-registered go rule: any BTD feature (or the rank combo) reaches
# ROC >= 0.60 on that warm slice for h1_1.0mm. If go, fetch the full panel span and run
# the V7 CV protocol with an extended feature block (phases `build` / `cv`, written after
# the pilot decides).
#
# Usage:
#     python Thailand_Rain_V12_multiband.py fetch B11 2026-05-01 2026-08-01
#     python Thailand_Rain_V12_multiband.py screen

# %%
import bz2
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from BKK_Rain_V3 import TARGET_NAMES, log
from Thailand_Rain_V3 import CACHE
from Thailand_Rain_V7_himawari_ir import (
    GRID_CSV, HW_DIR, HW_TMP, UTC_OFFSET_HOURS,
    s3_client, slot_candidates,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "ML_Model_V2" / "trained_models" / \
    "om_thailand_rain_v12_multiband"
CROP_LL_BBOX = (97.0, 5.3, 106.0, 20.9)
MB_STAT_NAMES = ["tb_mean", "tb_min", "tb_mean_env", "tb_min_env"]
BANDS = ["B07", "B11", "B15"]
PILOT_LOCAL = ("2026-05-01", "2026-08-01")
GO_RULE = "warm-slice ROC >= 0.60 for h1_1.0mm on any BTD feature or the rank combo"


def band_dir(band):
    return HW_DIR / f"mb_{band}"


# %% [markdown]
# ## Fetch — V7's day-sweep, parameterized by band
#
# Same buckets, segments, masks and crop as the B13 fetch (every AHI IR band is 2 km /
# R20, so the cached pixel masks apply unchanged). Stats per slot are leaner than V7's:
# mean/min over the cell and the 0.5 deg environment — BTDs are built from these later.

# %%
def segment_key_band(band, satcode, t_utc, segment):
    ts = pd.Timestamp(t_utc)
    return (f"AHI-L1b-FLDK/{ts.year}/{ts.month:02d}/{ts.day:02d}/{ts.hour:02d}00/"
            f"HS_{satcode}_{ts.year}{ts.month:02d}{ts.day:02d}_{ts.hour:02d}00_"
            f"{band}_FLDK_R20_S{segment:02d}10.DAT")


def download_segments_band(client, band, t_utc, segments):
    HW_TMP.mkdir(parents=True, exist_ok=True)
    for bucket, satcode in slot_candidates(t_utc):
        paths = []
        try:
            for seg in segments:
                key = segment_key_band(band, satcode, t_utc, seg)
                raw = HW_TMP / (Path(key).name + ".bz2")
                client.download_file(bucket, key + ".bz2", str(raw))
                out = HW_TMP / Path(key).name
                out.write_bytes(bz2.decompress(raw.read_bytes()))
                raw.unlink()
                paths.append(out)
            return paths
        except Exception:
            for p in paths:
                p.unlink(missing_ok=True)
            continue
    return None


def load_tb_band(band, paths):
    from satpy import Scene
    scn = Scene(reader="ahi_hsd", filenames=[str(p) for p in paths])
    scn.load([band])
    scn = scn.crop(ll_bbox=CROP_LL_BBOX)
    tb = scn[band].values.astype("float32")
    lons, lats = scn[band].attrs["area"].get_lonlats()
    return tb, lons.astype("float32"), lats.astype("float32")


def mb_slot_stats(tb, masks):
    flat = tb.ravel()
    out = np.full((len(masks["cell"]), len(MB_STAT_NAMES)), np.nan, dtype="float32")
    for c, (cell_ix, env_ix) in enumerate(zip(masks["cell"], masks["env"])):
        cell = flat[cell_ix]
        env = flat[env_ix]
        cell = cell[np.isfinite(cell)]
        env = env[np.isfinite(env)]
        if len(cell):
            out[c, 0] = cell.mean()
            out[c, 1] = cell.min()
        if len(env):
            out[c, 2] = env.mean()
            out[c, 3] = env.min()
    return out


_worker = {}


def _mb_worker_init(band, segments, mask_path):
    mask_file = np.load(mask_path)
    n_cells = sum(1 for k in mask_file.files if k.startswith("cell_"))
    _worker["masks"] = {"shape": tuple(mask_file["shape"]),
                        "cell": [mask_file[f"cell_{i}"] for i in range(n_cells)],
                        "env": [mask_file[f"env_{i}"] for i in range(n_cells)]}
    _worker["band"] = band
    _worker["segments"] = segments
    _worker["client"] = s3_client()


def _mb_fetch_one_slot(t_utc):
    paths = download_segments_band(_worker["client"], _worker["band"], t_utc,
                                   _worker["segments"])
    if paths is None:
        return None
    try:
        tb, lons, lats = load_tb_band(_worker["band"], paths)
        if lons.shape != _worker["masks"]["shape"]:
            return None   # unexpected geometry; count as failed rather than misalign
        return mb_slot_stats(tb, _worker["masks"])
    except Exception:
        return None
    finally:
        for p in paths:
            p.unlink(missing_ok=True)


def phase_fetch(band, local_start, local_end):
    from concurrent.futures import ProcessPoolExecutor
    assert band in BANDS, f"band must be one of {BANDS}"
    cfg = json.loads((HW_DIR / "fetch_config.json").read_text(encoding="utf-8"))
    segments = cfg["segments"]
    out_dir = band_dir(band)
    out_dir.mkdir(parents=True, exist_ok=True)
    workers = int(os.getenv("HW_FETCH_WORKERS", "4"))

    # dense hourly UTC sweep; lag features need HW_LOOKBACK hours before the window
    start_utc = pd.Timestamp(local_start) - pd.Timedelta(hours=UTC_OFFSET_HOURS + 4)
    end_utc = pd.Timestamp(local_end) - pd.Timedelta(hours=UTC_OFFSET_HOURS)
    hours = pd.date_range(start_utc, end_utc, freq="h", inclusive="left")
    days = pd.DatetimeIndex(hours.normalize()).unique().sort_values()
    log(f"{band}: {len(hours):,} UTC hours across {len(days):,} days, "
        f"segments {segments}, workers {workers}")

    done = failed = 0
    t0 = time.time()
    with ProcessPoolExecutor(
            max_workers=workers, initializer=_mb_worker_init,
            initargs=(band, segments, str(HW_DIR / "pixel_masks.npz"))) as pool:
        for day in days:
            day_file = out_dir / f"day_{pd.Timestamp(day):%Y%m%d}.npz"
            if day_file.exists():
                continue
            day_hours = hours[(hours >= day) & (hours < day + pd.Timedelta(days=1))]
            stats = np.full((len(day_hours), 833, len(MB_STAT_NAMES)), np.nan,
                            dtype="float32")
            for j, result in enumerate(pool.map(_mb_fetch_one_slot, day_hours)):
                if result is None:
                    failed += 1
                else:
                    stats[j] = result
            np.savez_compressed(day_file,
                                hours=day_hours.to_numpy(dtype="datetime64[s]"),
                                stats=stats)
            done += 1
            rate = done / max((time.time() - t0) / 3600, 1e-9)
            print(f"  {band} {pd.Timestamp(day):%Y-%m-%d}  {done}/{len(days)} days  "
                  f"failed slots {failed:,}  {rate:.1f} days/h  "
                  f"eta {(len(days) - done) / max(rate, 1e-9):.1f} h", flush=True)
    log(f"{band} fetch complete: {done:,} new days, {failed:,} failed slots")


# %% [markdown]
# ## Screen — the pre-registered no-training go/no-go
#
# Rows: panel rows in the pilot window with valid B13 (hw_block) and valid new-band
# stats at lag 1. The decisive subset is `warm`: rows with hw_cold235_env_lag1 == 0 —
# zero cold-top fraction anywhere in the 0.5 deg environment, so B13's cold-top features
# carry no signal. If a BTD ranks rain there, it sees what B13 cannot.

# %%
def load_band_lookup(band):
    """{utc datetime64[s] -> (833, 4) stats} from the band's day files."""
    lookup = {}
    for f in sorted(band_dir(band).glob("day_*.npz")):
        z = np.load(f)
        for h, s in zip(z["hours"], z["stats"]):
            lookup[np.datetime64(h, "s")] = s
    return lookup


def rows_lag1_stats(lookup, t_local, unit_ix):
    """Per-row lag-1 band stats aligned to panel rows (NaN when the slot is missing)."""
    utc = (t_local - np.timedelta64(UTC_OFFSET_HOURS + 1, "h")).astype("datetime64[s]")
    out = np.full((len(t_local), len(MB_STAT_NAMES)), np.nan, dtype="float32")
    uniq, inverse = np.unique(utc, return_inverse=True)
    for k, u in enumerate(uniq):
        s = lookup.get(u)
        if s is None:
            continue
        rows = np.flatnonzero(inverse == k)
        out[rows] = s[unit_ix[rows]]
    return out


def auc(y, x):
    from sklearn.metrics import roc_auc_score
    m = np.isfinite(x)
    if m.sum() < 200 or len(np.unique(y[m])) < 2:
        return np.nan, int(m.sum())
    return float(roc_auc_score(y[m], x[m])), int(m.sum())


def phase_screen():
    from Thailand_Rain_V7_himawari_ir import himawari_feature_names
    hw_names = himawari_feature_names()
    tf = np.load(CACHE / "forecast_time.npy")
    y = np.load(CACHE / "y.npy")
    unit = np.load(CACHE / "unit_id.npy")
    hw = np.load(CACHE / "hw_block_offset0.npy")
    grid = pd.read_csv(GRID_CSV).sort_values("grid_number")
    unit_ix = np.searchsorted(grid["grid_number"].to_numpy(), unit)

    lo, hi = (np.datetime64(PILOT_LOCAL[0]), np.datetime64(PILOT_LOCAL[1]))
    in_pilot = (tf >= lo) & (tf < hi)
    b13_mean = hw[:, hw_names.index("hw_tb_mean_lag1")]
    b13_envmin = hw[:, hw_names.index("hw_tb_min_env_lag1")]
    cold235 = hw[:, hw_names.index("hw_cold235_env_lag1")]
    base = in_pilot & np.isfinite(b13_mean) & np.isfinite(cold235)
    log(f"pilot rows {in_pilot.sum():,}, with valid B13 {base.sum():,}")

    stats = {}
    for band in BANDS:
        lookup = load_band_lookup(band)
        log(f"{band}: {len(lookup):,} slots cached")
        stats[band] = rows_lag1_stats(lookup, tf, unit_ix)

    MEAN, ENVMIN = MB_STAT_NAMES.index("tb_mean"), MB_STAT_NAMES.index("tb_min_env")
    feats = {
        "btd_11_13_mean": stats["B11"][:, MEAN] - b13_mean,
        "btd_13_15_mean": b13_mean - stats["B15"][:, MEAN],
        "btd_07_13_mean": stats["B07"][:, MEAN] - b13_mean,
        "btd_11_13_envmin": stats["B11"][:, ENVMIN] - b13_envmin,
        "b11_tb_mean": stats["B11"][:, MEAN],
        "b15_tb_mean": stats["B15"][:, MEAN],
        "b07_tb_mean": stats["B07"][:, MEAN],
        "b13_tb_mean_control": b13_mean,       # sanity: known-good signal on `all`
        "cold235_control": cold235,            # sanity: ~0.5 on the warm slice
    }

    ti = {n: i for i, n in enumerate(TARGET_NAMES)}
    results = []
    for tgt in ["h1_1.0mm", "h1_0.1mm", "h3_1.0mm"]:
        yy = y[:, ti[tgt]]
        slices = {
            "all": base,
            "warm": base & (cold235 == 0),
            "cold": base & (cold235 > 0),
        }
        for sname, m in slices.items():
            for fname, x in feats.items():
                a, n = auc(yy[m], x[m])
                results.append({"target": tgt, "slice": sname, "feature": fname,
                                "n": n, "base_rate": round(float(yy[m].mean()), 4),
                                "roc": round(a, 4) if np.isfinite(a) else np.nan,
                                "roc_oriented": round(max(a, 1 - a), 4)
                                                if np.isfinite(a) else np.nan})
        # rank combo of the three mean-BTDs, each oriented on the same slice (screen
        # optimism is acceptable: this decides a download, not a headline)
        for sname, m in slices.items():
            combo = np.zeros(int(m.sum()), dtype="float64")
            ok = np.ones(int(m.sum()), dtype=bool)
            yy_m = yy[m]
            for fname in ["btd_11_13_mean", "btd_13_15_mean", "btd_07_13_mean"]:
                x = feats[fname][m]
                ok &= np.isfinite(x)
                a, _ = auc(yy_m, x)
                sign = 1.0 if (np.isfinite(a) and a >= 0.5) else -1.0
                r = pd.Series(np.where(np.isfinite(x), x, np.nan)).rank(pct=True).to_numpy()
                combo += sign * np.where(np.isfinite(r), r, 0.5)
            a, n = auc(yy_m[ok], combo[ok])
            results.append({"target": tgt, "slice": sname, "feature": "btd_rank_combo",
                            "n": n, "base_rate": round(float(yy_m[ok].mean()), 4)
                                                 if ok.sum() else np.nan,
                            "roc": round(a, 4) if np.isfinite(a) else np.nan,
                            "roc_oriented": round(max(a, 1 - a), 4)
                                            if np.isfinite(a) else np.nan})

    out = pd.DataFrame(results)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_DIR / "v12_screen.csv", index=False)
    log(f"saved {OUTPUT_DIR / 'v12_screen.csv'}")
    show = out[(out["target"] == "h1_1.0mm")]
    for sname in ["warm", "cold", "all"]:
        s = show[show["slice"] == sname].sort_values("roc_oriented", ascending=False)
        log(f"=== h1_1.0mm {sname} (n={s['n'].max():,}, "
            f"base {s['base_rate'].iloc[0]}) ===")
        print(s[["feature", "roc", "roc_oriented", "n"]].to_string(index=False))
    log(f"GO RULE: {GO_RULE}")


# %%
if __name__ == "__main__":
    phase = sys.argv[1] if len(sys.argv) > 1 else "screen"
    log(f"phase: {phase}")
    t0 = time.time()
    if phase == "fetch":
        band = sys.argv[2]
        start = sys.argv[3] if len(sys.argv) > 3 else PILOT_LOCAL[0]
        end = sys.argv[4] if len(sys.argv) > 4 else PILOT_LOCAL[1]
        phase_fetch(band, start, end)
    elif phase == "screen":
        phase_screen()
    else:
        raise SystemExit(f"unknown phase {phase!r}")
    log(f"phase {phase} done in {(time.time() - t0) / 60:.1f} min")
