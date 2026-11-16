"""The cloud job must give up at once on a first-batch 429 (a fresh runner is the fix),
while a local run keeps the long back-off.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_om_fail_fast.py -q
"""
import sys
from pathlib import Path

import pandas as pd
import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import Thailand_Rain_V10_deploy as deploy  # noqa: E402

GRID = pd.DataFrame({"grid_number": [1], "latitude": [13.75], "longitude": [100.5]})


def refuse(monkeypatch):
    """Every Open-Meteo call answers 429; returns the lists of calls and sleeps."""
    calls, sleeps = [], []

    def get(*a, **k):
        calls.append(1)
        resp = requests.Response()
        resp.status_code = 429
        return resp

    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr(deploy.time, "sleep", sleeps.append)
    return calls, sleeps


def test_cloud_gives_up_on_first_429(monkeypatch):
    calls, sleeps = refuse(monkeypatch)
    monkeypatch.setenv("OM_429_FAIL_FAST", "1")
    with pytest.raises(requests.exceptions.HTTPError):
        deploy.fetch_live_om(GRID)
    assert len(calls) == 1 and sleeps == []


@pytest.mark.parametrize("value", [None, ""])
def test_local_keeps_the_back_off(monkeypatch, value):
    calls, sleeps = refuse(monkeypatch)
    if value is None:
        monkeypatch.delenv("OM_429_FAIL_FAST", raising=False)
    else:
        monkeypatch.setenv("OM_429_FAIL_FAST", value)   # what the workflow sets on attempt 3
    with pytest.raises(requests.exceptions.HTTPError):
        deploy.fetch_live_om(GRID)
    assert len(calls) == 6 and sleeps == [60, 120, 180, 240, 300]
