"""Self-check for rainwatch_server.predict_tick and _failure_record.

Run:  python Dashboard/test_predict_tick.py   (plain asserts, no pytest needed)
"""
import importlib.util
import sys
import tempfile
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("rs", HERE / "rainwatch_server.py")
rs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rs)


def make_env(tmp, clock, outcome):
    """outcome: 'ok' writes the wanted CSV, 'stale' writes an older hour, 'fail' sets last_error."""
    rs.PRED_DIR = Path(tmp)
    rs.DATA = Path(tmp)
    rs.bkk_now = lambda: clock[0]
    launches, logs = [], []
    rs.log = logs.append

    def fake_run_predict():
        now = clock[0]
        launches.append(now)
        rs.STATUS["last_error"] = None
        if outcome[0] == "ok":
            (Path(tmp) / f"v10_predictions_{now:%Y%m%d_%H00}.csv").write_text("x")
        elif outcome[0] == "stale":
            (Path(tmp) / f"v10_predictions_{now:%Y%m%d}_{now.hour - 1:02d}00.csv").write_text("x")
        else:
            rs.STATUS["last_error"] = "boom"

    rs.run_predict = fake_run_predict
    return launches, logs


def test_existing_bangkok_stamp_is_recognised():
    with tempfile.TemporaryDirectory() as tmp:
        clock = [datetime(2026, 10, 26, 11, 15)]
        launches, _ = make_env(tmp, clock, ["ok"])
        (Path(tmp) / "v10_predictions_20261026_1100.csv").write_text("x")
        state = {"fail_streak": 0, "launched": {}}
        for _ in range(5):
            assert rs.predict_tick(state) == 60
        assert launches == []


def test_waits_for_run_minute_then_runs_once():
    with tempfile.TemporaryDirectory() as tmp:
        clock = [datetime(2026, 10, 26, 11, 5)]
        launches, _ = make_env(tmp, clock, ["ok"])
        state = {"fail_streak": 0, "launched": {}}
        assert rs.predict_tick(state) == 60 and launches == []
        clock[0] = datetime(2026, 10, 26, 11, 10)
        assert rs.predict_tick(state) == 60 and len(launches) == 1
        for _ in range(3):
            rs.predict_tick(state)
        assert len(launches) == 1


def test_failed_run_retries_once_then_stops_until_next_hour():
    with tempfile.TemporaryDirectory() as tmp:
        clock = [datetime(2026, 10, 26, 11, 10)]
        launches, logs = make_env(tmp, clock, ["fail"])
        state = {"fail_streak": 0, "launched": {}}
        assert rs.predict_tick(state) == 600 and len(launches) == 1
        clock[0] = datetime(2026, 10, 26, 11, 21)
        assert rs.predict_tick(state) == 1200 and len(launches) == 2
        assert "no further launches for 20261026_1100" in logs[-1]
        clock[0] = datetime(2026, 10, 26, 11, 42)
        for _ in range(10):
            assert rs.predict_tick(state) == 60
        assert len(launches) == 2
        clock[0] = datetime(2026, 10, 26, 12, 10)       # new hour: guard resets, cool-down keeps escalating
        assert rs.predict_tick(state) == 1800 and len(launches) == 3


def test_run_that_writes_an_older_hour_is_capped_too():
    with tempfile.TemporaryDirectory() as tmp:
        clock = [datetime(2026, 10, 26, 11, 10)]
        launches, _ = make_env(tmp, clock, ["stale"])
        state = {"fail_streak": 0, "launched": {}}
        for _ in range(6):
            rs.predict_tick(state)
        assert len(launches) == rs.MAX_LAUNCHES_PER_HOUR


def test_failure_record_keeps_exit_code_and_full_output():
    with tempfile.TemporaryDirectory() as tmp:
        clock = [datetime(2026, 10, 26, 11, 47, 52)]
        make_env(tmp, clock, ["fail"])
        msg = rs._failure_record(1, "line one\n" * 50, b"Traceback\nValueError: boom\n")
        assert msg.startswith("predict exited 1: ValueError: boom [full output: data/predict_fail_20261026_114752.log]")
        saved = (Path(tmp) / "predict_fail_20261026_114752.log").read_text()
        assert "exit code: 1" in saved and saved.count("line one") == 50 and "ValueError: boom" in saved


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok ", t.__name__)
    print(f"{len(tests)} checks passed")
