"""Focused checks for the 2026-10-27 label fix: window [t, t+h), per-pixel cell max,
missing hours and provider alignment.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_label_windows.py -q

(the env var skips an installed pytest plugin whose zarr/numcodecs DLL this machine's
Application Control policy blocks; nothing here needs it)
"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "Dashboard", ROOT / "IMERG_Script"):
    sys.path.insert(0, str(p))

import BKK_Rain_V3 as protocol               # noqa: E402
import Thailand_Rain_V3 as th                # noqa: E402
import build_climatology as clim             # noqa: E402
import compare_providers as prov             # noqa: E402
import fetch_imerg_thailand_data_gee as fetch  # noqa: E402
import verify_imerg as verify                # noqa: E402

ISSUE = datetime(2026, 10, 20, 14, 0)


def test_training_window_starts_at_issue_hour():
    assert protocol.LEAD_HOURS == [0, 1, 2, 3, 4, 5]
    sql = th.build_panel_query()
    assert f"LEAD({th.IMERG_LABEL_COLUMN}, 0) OVER w AS p0" in sql
    assert f"LEAD({th.IMERG_LABEL_COLUMN}, 6)" not in sql
    assert th.IMERG_LABEL_COLUMN == "precipitation_pixelmax_mm"


def test_verify_window_and_persistence_hours():
    leads = verify.lead_hours(ISSUE)
    assert leads[0] == ISSUE and leads[-1] == ISSUE + timedelta(hours=5)
    hist = verify.hist_hours(ISSUE)
    assert hist == [ISSUE - timedelta(hours=k) for k in range(6, 0, -1)]
    # persistence for horizon h = the h observed hours ending at the issue hour (exclusive)
    assert hist[-1:] == [ISSUE - timedelta(hours=1)]
    assert hist[-3:] == [ISSUE - timedelta(hours=k) for k in (3, 2, 1)]
    assert not set(leads) & set(hist)


def test_persistence_raw_changes_only_the_roc(tmp_path):
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    cells = np.arange(1, verify.N_CELLS + 1)
    hours = verify.hist_hours(ISSUE)[-1:] + verify.lead_hours(ISSUE)[:1]
    pmax = pd.DataFrame(rng.gamma(0.5, 2.0, (len(cells), 2)), index=cells, columns=hours)
    pred = pd.DataFrame({"grid_number": cells, "p_h1_1.0mm": 0.5, "flag_h1_1.0mm": 1})
    path = tmp_path / "pred.csv"
    pred.to_csv(path, index=False)
    _, base, _ = verify.score_issue("x", ISSUE, path, pmax, None, {("x", "h1_1.0mm")}, set(), None, {})
    rows = {r["baseline"]: r for r in base}
    old, raw = rows["persistence"], rows["persistence_raw"]
    assert {k: v for k, v in old.items() if k not in ("baseline", "roc_auc")} == \
           {k: v for k, v in raw.items() if k not in ("baseline", "roc_auc")}
    y = (pmax[hours[1]] >= 1.0).astype(int)
    assert raw["roc_auc"] == round(roc_auc_score(y, pmax[hours[0]]), 4)   # ranks the raw score
    assert raw["roc_auc"] != old["roc_auc"]                               # the clip ties wet cells


def test_lagged_persistence_reads_hours_six_before_issue_and_om_only_reads_its_columns(tmp_path):
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(1)
    cells = np.arange(1, verify.N_CELLS + 1)
    lag = verify.hist_hours(ISSUE, verify.AVAILABLE_LAG_H)
    assert lag[-1] == ISSUE - timedelta(hours=7)                 # h1 reads [issue-7, issue-6)
    assert not set(lag) & set(verify.hist_hours(ISSUE))
    hours = lag[-1:] + verify.lead_hours(ISSUE)[:1]
    pmax = pd.DataFrame(rng.gamma(0.5, 2.0, (len(cells), 2)), index=cells, columns=hours)
    om_p = rng.random(len(cells))
    pred = pd.DataFrame({"grid_number": cells, "p_h1_1.0mm": 0.5, "flag_h1_1.0mm": 1,
                         "om_p_h1_1.0mm": om_p, "om_flag_h1_1.0mm": (om_p > 0.5).astype(int)})
    path = tmp_path / "pred.csv"
    pred.to_csv(path, index=False)
    _, base, _ = verify.score_issue("x", ISSUE, path, pmax, None, {("x", "h1_1.0mm")}, set(), None, {})
    rows = {r["baseline"]: r for r in base}
    y = (pmax[hours[1]] >= 1.0).astype(int)
    assert rows["persistence_lag6h"]["roc_auc"] == round(roc_auc_score(y, pmax[hours[0]]), 4)
    assert rows["om_only"]["roc_auc"] == round(roc_auc_score(y, om_p), 4)
    assert "persistence" not in rows          # its [issue-1, issue) hour is absent here

    pred.drop(columns=["om_p_h1_1.0mm", "om_flag_h1_1.0mm"]).to_csv(path, index=False)
    _, base, _ = verify.score_issue("x", ISSUE, path, pmax, None, {("x", "h1_1.0mm")}, set(), None, {})
    assert "om_only" not in {r["baseline"] for r in base}   # files before the shadow: skipped


def test_climatology_window_and_missing_hours():
    values = np.array([[0.0, 1.0, 0.0, np.nan, 2.0, 0.5]])
    h1 = clim.forward_window_max(values, 1)[0]
    np.testing.assert_array_equal(np.isnan(h1), [False, False, False, True, False, False])
    assert h1[1] == 1.0 and h1[4] == 2.0
    h2 = clim.forward_window_max(values, 2)[0]
    assert h2[0] == 1.0 and h2[1] == 1.0          # [t, t+2) = max(v_t, v_t+1)
    assert np.isnan(h2[2]) and np.isnan(h2[3])    # a missing hour inside the window -> NaN
    assert h2[4] == 2.0 and np.isnan(h2[5])       # last column: the window runs off the end


def test_provider_alignment_and_bounds():
    assert prov.label_hours(ISSUE, 3) == [ISSUE + timedelta(hours=k) for k in (0, 1, 2)]
    assert prov.persistence_hours(ISSUE, 3) == [ISSUE - timedelta(hours=k) for k in (3, 2, 1)]
    # Open-Meteo's value stamped issue+k covers [issue+k-1, issue+k): hours 1..h == label window
    covered = [ISSUE + timedelta(hours=k - 1) for k in range(1, 4)]
    assert covered == prov.label_hours(ISSUE, 3)
    lower, upper = prov.window_bounds([0.2, 0.5, 0.3])
    assert lower == 0.5 and upper == pytest.approx(1 - 0.8 * 0.5 * 0.7)
    assert lower <= upper


def test_moving_spatial_maxima():
    first = np.array([[1.08, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, np.nan]])
    second = np.array([[0.0, 0.0, 1.06], [0.0, 0.2, 0.0], [0.0, 0.0, np.nan]])
    stored_style = 0.5 * np.nanmax(first) + 0.5 * np.nanmax(second)
    assert stored_style == pytest.approx(1.07)
    assert fetch.hourly_pixel_max(first, second) == pytest.approx(0.54)   # wettest pixel (0, 0)
    # same pixel in both halves -> the two definitions agree
    assert fetch.hourly_pixel_max(first, first) == pytest.approx(np.nanmax(first))


def test_ingestion_timestamps_and_missing_half_hours():
    utc = timezone.utc
    t0 = datetime(2026, 10, 20, 7, 0, tzinfo=utc)
    band_map = {
        "a_precipitation": (t0, "provisional"),
        "b_precipitation": (t0 + timedelta(minutes=30), "provisional"),
        "c_precipitation": (t0 + timedelta(hours=1), "provisional"),   # its 08:30 half is missing
    }
    props = {"grid_number": 2,
             "a_precipitation_mean": 0.37, "a_precipitation_max": 1.08, "a_precipitation_count": 9,
             "b_precipitation_mean": 0.20, "b_precipitation_max": 1.06, "b_precipitation_count": 9,
             "c_precipitation_mean": 0.10, "c_precipitation_max": 0.50, "c_precipitation_count": 9,
             "hour_2026102007_max": 0.87, "hour_2026102007_mean": 0.285}
    cells = {2: {"grid_row": 2, "grid_column": 18, "longitude": 101.26, "latitude": 5.84}}
    rows = fetch.rows_from_features([{"properties": props}], cells, band_map)
    by_hour = {r[5]: r for r in rows}
    full = by_hour[t0]
    assert full[6] == datetime(2026, 10, 20, 14, 0)     # local stamp = START of the hour covered
    assert full[16] == 2                                # half_hour_count
    assert full[9] == pytest.approx(1.07) and full[10] == pytest.approx(0.87)
    partial = by_hour[t0 + timedelta(hours=1)]
    assert partial[16] == 1
    assert partial[8] is None and partial[9] is None and partial[10] is None
