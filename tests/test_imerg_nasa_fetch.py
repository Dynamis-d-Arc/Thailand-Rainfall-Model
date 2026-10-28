"""The direct-NASA IMERG fallback must reproduce the Earth Engine cell footprint and columns.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_imerg_nasa_fetch.py -q
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "IMERG_Script"))

import fetch_imerg_thailand_data_nasa as nasa  # noqa: E402

CELL = {"grid_number": 1, "grid_row": 1, "grid_column": 1, "longitude": 100.0, "latitude": 15.0}
HOUR = datetime(2026, 10, 27, 12, 0, tzinfo=timezone.utc)


def test_footprint_is_earth_engines_grid_not_the_geometric_rectangle():
    # 100.0 +- 0.1126 touches four native columns: 99.8, 99.9, 100.0, 100.1
    idx, overlap = nasa.pixel_window(100.0 - 0.1126, 100.0 + 0.1126, -180.0, 1, -180.0)
    assert idx.tolist() == [2798, 2799, 2800, 2801]
    assert np.allclose(overlap[1:3], 1.0) and 0 < overlap[0] < 0.2 and 0 < overlap[3] < 0.2
    # the lat axis runs north to south from 90 but yields native indices counted up from -90
    idx, _ = nasa.pixel_window(15.0 - 0.1126, 15.0 + 0.1126, 90.0, -1, -90.0)
    assert idx.tolist() == [1051, 1050, 1049, 1048]
    # two real cells whose stored Earth Engine pixel counts (2026-10-25 12:00Z) the geometric
    # rectangle got wrong: grid 10 drops a corner below 1/256, grid 122 drags in a sliver column
    # (exact coordinates matter: a 2e-5 deg shift moves a sliver across the 1/256 boundary)
    grid = {}
    for line in (ROOT / "ML_Model_V2" / "trained_models" / "om_thailand_rain_v10_deploy" / "deploy"
                 / "th_grid_25km.csv").read_text().splitlines()[1:]:
        n, _, _, lon, lat = line.split(",")[:5]
        grid[int(n)] = {"longitude": float(lon), "latitude": float(lat)}
    lat_idx, lon_idx, w = nasa.cell_footprint(grid[10])
    assert w.shape == (4, 4) and int((w > 0).sum()) == 15
    lat_idx, lon_idx, w = nasa.cell_footprint(grid[122])
    assert int((w > 0).sum()) == 10 and lon_idx.tolist() == [2822, 2823, 2824, 2825]


def test_hour_row_matches_the_label_definitions():
    w = np.ones((2, 2))
    first = np.array([[1.0, 2.0], [3.0, 4.0]])
    second = np.array([[4.0, 3.0], [2.0, 1.0]])
    row = dict(zip(nasa.COLUMNS, nasa.hour_row(CELL, HOUR, first, second, w)))
    assert row["precipitation_pixelmax_mm"] == 2.5          # every pixel totals 0.5*(a+b) = 2.5
    assert row["precipitation_max_mm"] == 4.0               # 0.5*max(first) + 0.5*max(second)
    assert row["precipitation_mm"] == 2.5
    assert row["pixel_count"] == 4 and row["half_hour_count"] == 2
    assert row["local_observation_time"] == datetime(2026, 10, 27, 19, 0)

    # a fill value drops that pixel from its half-hour and from the per-pixel total
    first[0, 0] = -9999.9
    row = dict(zip(nasa.COLUMNS, nasa.hour_row(CELL, HOUR, first, second, w)))
    assert row["precipitation_pixelmax_mm"] == 2.5 and row["precipitation_rate_first_half_hour_mm_h"] == 3.0
    assert row["pixel_count"] == 3.5

    # a zero-weight pixel (dropped by Earth Engine) never counts, even when it is the wettest
    w[1, 1] = 0.0
    first = np.array([[1.0, 2.0], [3.0, 40.0]])
    row = dict(zip(nasa.COLUMNS, nasa.hour_row(CELL, HOUR, first, second, w)))
    assert row["precipitation_rate_max_mm_h"] == 4.0 and row["pixel_count"] == 3

    # one published half-hour: rates recorded, hourly totals stay NULL
    row = dict(zip(nasa.COLUMNS, nasa.hour_row(CELL, HOUR, first, None, w)))
    assert row["half_hour_count"] == 1
    assert row["precipitation_mm"] is None and row["precipitation_pixelmax_mm"] is None
