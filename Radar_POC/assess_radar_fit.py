"""Phase 4 POC: how the archived TMD radar frames line up with the model's 833-cell grid.

Reads only the radar archive (TMD_RADAR_ARCHIVE) and the static deploy grid CSV, writes to
Radar_POC/outputs/:
  radar_cell_coverage.csv   per cell: radar pixels inside it, share ever seen with echo, wet frames
  radar_png_vs_nc.csv       PNG frame decoded back to mm/h vs the NetCDF frame of the same stamp
  radar_coverage_map.png    the per-cell echo share on a map
Coverage is a LOWER BOUND: the composite stores values only where there is echo, so an empty
pixel means "no echo" or "no radar" alike; only the union over many frames approaches the footprint.

Usage:
    python Radar_POC/assess_radar_fit.py
"""

import csv
import gzip
import io
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.io import netcdf_file

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
ARCHIVE = Path(os.getenv("TMD_RADAR_ARCHIVE", r"C:\Users\Brandon\Documents\TMD_Radar_Archive"))
GRID_CSV = ROOT.parent / "ML_Model_V2" / "trained_models" / "om_thailand_rain_v10_deploy" / "deploy" / "th_grid_25km.csv"
LAT0, LON0, RES, NY, NX = 4.0, 95.0, 0.01, 1851, 1301
LAT_N, LON_E = 22.5, 108.0
# TMD viewer's colour bar (index_composite.html): colour -> lower edge of its mm/h bin. Four colours
# are reused for two adjacent bins, so a PNG pixel of those colours is known only to the wider bin.
BAR = [("54A431", 0.21), ("5EB738", 0.24), ("66C43C", 0.50), ("69CB5A", 1.03), ("73DE45", 1.47),
       ("7BEC4B", 2.10), ("F3F453", 3.00), ("DEDF4B", 4.29), ("D6D648", 6.13), ("F1C946", 8.76),
       ("D79C37", 12.5), ("E3622A", 25.6), ("D43320", 36.5), ("CA325D", 52.2), ("B72D54", 74.6),
       ("E345F5", 152.5), ("EA8CF8", 217.9), ("F3CBFA", 311.4), ("EFE6F1", 445.0)]
EDGES = [lo for _, lo in BAR] + [math.inf]


def cell_lookup():
    """(NY, NX) int array: grid_number of the 25 km cell holding each 0.01 deg pixel, 0 = none."""
    g = pd.read_csv(GRID_CSV)
    dlat = np.median(np.diff(np.sort(g.latitude.unique())))
    dlon = np.median(np.diff(np.sort(g.longitude.unique())))
    r0 = g.iloc[0]
    lat = LAT0 + RES * np.arange(NY)
    lon = LON0 + RES * np.arange(NX)
    rows = np.rint((lat - r0.latitude) / dlat).astype(int) + int(r0.grid_row)
    cols = np.rint((lon - r0.longitude) / dlon).astype(int) + int(r0.grid_column)
    key = {(int(a), int(b)): int(n) for a, b, n in zip(g.grid_row, g.grid_column, g.grid_number)}
    lut = np.zeros((NY, NX), dtype=np.int16)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            lut[i, j] = key.get((r, c), 0)
    return g, lut


def load_nc(path):
    """Archived .nc.gz -> (NY, NX) float32 rain rate, south row first, NaN where no echo."""
    f = netcdf_file(io.BytesIO(gzip.decompress(path.read_bytes())), "r", mmap=False)
    a = np.asarray(f.variables["var0_1_203_surface"][0], dtype=np.float32)
    f.close()
    a[a > 1e20] = np.nan
    return a


def png_pixel_index(shape, mercator):
    """NetCDF (i, j) index under each PNG pixel centre, for a PNG spanning the composite box."""
    h, w = shape
    lon = LON0 + (np.arange(w) + 0.5) / w * (LON_E - LON0)
    merc = lambda d: np.log(np.tan(np.pi / 4 + np.radians(d) / 2))
    f = (np.arange(h) + 0.5) / h
    if mercator:
        y = merc(LAT_N) - f * (merc(LAT_N) - merc(LAT0))
        lat = np.degrees(np.arctan(np.sinh(y)))
    else:
        lat = LAT_N - f * (LAT_N - LAT0)
    i = np.clip(np.rint((lat - LAT0) / RES).astype(int), 0, NY - 1)
    j = np.clip(np.rint((lon - LON0) / RES).astype(int), 0, NX - 1)
    return i[:, None], j[None, :]


def decode_png(path):
    """PNG -> (bin index or -1 for transparent / off-palette, n off-palette opaque pixels)."""
    im = np.asarray(Image.open(path).convert("RGBA"))
    rgb = (im[..., 0].astype(np.int32) << 16) | (im[..., 1].astype(np.int32) << 8) | im[..., 2]
    out = np.full(rgb.shape, -1, dtype=np.int16)
    for k, (hexc, _) in enumerate(BAR):
        out[(rgb == int(hexc, 16)) & (im[..., 3] > 0)] = k
    return out, int(((im[..., 3] > 0) & (out < 0)).sum())


def compare(png_bins, nc, idx):
    v = nc[idx]
    opaque, echo = png_bins >= 0, np.isfinite(v)
    both = opaque & (v > 0)
    lo = np.array(EDGES)[np.where(both, png_bins, 0)]
    hi = np.array(EDGES)[np.where(both, png_bins, 0) + 1]
    in_bin = both & (v >= lo) & (v < hi)
    return {"png_px": int(opaque.sum()), "nc_echo_px": int(echo.sum()), "nc_rain_px": int((v > 0).sum()),
            "png_px_on_nc_echo": round(float((opaque & echo).sum() / max(opaque.sum(), 1)), 4),
            "nc_rain_px_drawn": round(float(both.sum() / max((v > 0).sum(), 1)), 4),
            "nc_zero_px_drawn": round(float((opaque & (v == 0)).sum() / max((v == 0).sum(), 1)), 4),
            "value_in_png_bin": round(float(in_bin.sum() / max(both.sum(), 1)), 4)}


def main():
    OUT.mkdir(exist_ok=True)
    man = pd.read_csv(ARCHIVE / "manifest.csv")
    g, lut = cell_lookup()
    n_cells = len(g)
    px_per_cell = np.bincount(lut.ravel(), minlength=n_cells + 1)
    print(f"grid: {n_cells} cells, {int((lut > 0).sum())} radar pixels inside them, "
          f"{px_per_cell[1:].min()}-{px_per_cell[1:].max()} per cell "
          f"(cells outside the composite box: {int((px_per_cell[1:] == 0).sum())})")

    nc_rows = man[man["product"] == "nc"].sort_values("stamp_utc")
    png_rows = man[man["product"] == "png"].sort_values("stamp_utc")
    echo_any = np.zeros((NY, NX), dtype=bool)
    wet_frames = np.zeros(n_cells + 1, dtype=int)     # stamps with any rain > 0 inside the cell
    nc = {}
    for r in nc_rows.itertuples():
        a = load_nc(ARCHIVE / r.path)
        nc[r.stamp_utc] = a
        echo_any |= np.isfinite(a)
        wet_frames += np.bincount(lut[a > 0], minlength=n_cells + 1) > 0

    # georeference check: which PNG projection puts PNG colour where the NetCDF has echo?
    cmp_rows, merc_idx, lin_idx = [], None, None
    for r in png_rows.itertuples():
        if r.stamp_utc not in nc:
            continue
        bins, off = decode_png(ARCHIVE / r.path)
        merc_idx = merc_idx or png_pixel_index(bins.shape, True)
        lin_idx = lin_idx or png_pixel_index(bins.shape, False)
        for label, idx in (("mercator", merc_idx), ("plate_carree", lin_idx)):
            cmp_rows.append({"png_stamp": r.stamp_utc, "nc_stamp": r.stamp_utc, "projection": label,
                             "off_palette_px": off, **compare(bins, nc[r.stamp_utc], idx)})
        prev = [s for s in nc if s < r.stamp_utc]
        if prev:   # same PNG against the frame 15 min earlier: agreement must drop if stamps are right
            cmp_rows.append({"png_stamp": r.stamp_utc, "nc_stamp": prev[-1], "projection": "mercator",
                             "off_palette_px": off, **compare(bins, nc[prev[-1]], merc_idx)})
    if cmp_rows:
        pd.DataFrame(cmp_rows).to_csv(OUT / "radar_png_vs_nc.csv", index=False)
        print(pd.DataFrame(cmp_rows).to_string(index=False))

    # PNG frames extend the echo union over the 6 h the NetCDF archive does not reach back to;
    # a stamp counts once in wet_frames, from its NetCDF when there is one
    for r in png_rows.itertuples():
        bins, _ = decode_png(ARCHIVE / r.path)
        mi, mj = png_pixel_index(bins.shape, True)
        hit = bins >= 0
        ii, jj = np.broadcast_to(mi, hit.shape)[hit], np.broadcast_to(mj, hit.shape)[hit]
        echo_any[ii, jj] = True
        if r.stamp_utc not in nc:
            cells = np.unique(lut[ii, jj])
            wet_frames[cells[cells > 0]] += 1
    union = echo_any

    seen = np.bincount(lut[union], minlength=n_cells + 1)
    cov = pd.DataFrame({"grid_number": np.arange(1, n_cells + 1), "radar_px": px_per_cell[1:],
                        "echo_ever_share": np.round(seen[1:] / np.maximum(px_per_cell[1:], 1), 4),
                        "wet_frames": wet_frames[1:]})
    cov = g[["grid_number", "latitude", "longitude"]].merge(cov, on="grid_number")
    cov.to_csv(OUT / "radar_cell_coverage.csv", index=False)
    n_frames = man.stamp_utc.nunique()
    print(f"\n{n_frames} stamps: {len(nc_rows)} NetCDF + {len(png_rows)} PNG frames "
          f"({min(man.stamp_utc)} to {max(man.stamp_utc)})")
    for thr in (0.0, 0.05, 0.25, 0.5):
        print(f"cells with echo share > {thr:.2f}: {(cov.echo_ever_share > thr).sum()} of {n_cells}")
    print(f"cells wet (rain > 0) in at least one of {n_frames} frames: {(cov.wet_frames > 0).sum()}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 8))
    sc = ax.scatter(cov.longitude, cov.latitude, c=cov.echo_ever_share, s=14, marker="s",
                    cmap="viridis", vmin=0, vmax=1)
    fig.colorbar(sc, ax=ax, shrink=0.6, label="share of cell's 0.01 deg pixels with echo in any frame")
    ax.set_title(f"TMD radar echo union per 25 km cell\n{min(man.stamp_utc)} to {max(man.stamp_utc)} "
                 f"({n_frames} frames, lower bound)", fontsize=9)
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(OUT / "radar_coverage_map.png", dpi=130)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
