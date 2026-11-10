"""Archive TMD's nationwide weather-radar composite (Phase 4 POC: data collection only).

Source: https://weather.tmd.go.th/composite/ (Thai Meteorological Department, public, no key).
  data/composite.nc       LATEST FRAME ONLY. Rain rate (mm/h) on a 0.01 deg grid, 1851 x 1301,
                          lat 4-22.5 N, lon 95-108 E; PCAPPI 2 km national mosaic, Z = 200 R^1.6.
                          NetCDF-3 written by wgrib2, variable var0_1_203_surface, fill 9.999e20.
  radar.sharing.latest    'YYYY-MM-DD HH:MM' UTC stamp of that frame.
  images_composite.list   the last 25 frames (6 h) as images/zr/NN.png overlays, oldest first.
TMD keeps no archive: a NetCDF frame not fetched within its 15 minutes as "latest" is gone.
The PNGs are colour-binned renders of the same field that stay up for 6 h, so they backfill
gaps (machine asleep) at reduced precision.

Each run (Task Scheduler, every 5 min):
  1. latest NetCDF: skip if stored; else download, parse it fully (a truncated file fails),
     take the stamp from the file's own reference time, store the original bytes gzip'd.
  2. PNG frames: download every listed frame not yet stored, then re-read the list and drop
     the batch if it changed underneath us.
Duplicates: a (product, stamp) already in the manifest is never fetched again; a file whose
bytes match an earlier frame is still stored but flagged in `dup_of`.

Everything is UTC (TMD's stamps are UTC; Bangkok = UTC+7, added only in the report).
Layout under ARCHIVE (env TMD_RADAR_ARCHIVE):
  nc/YYYY/MM/DD/tmd_composite_YYYYMMDD_HHMMZ.nc.gz   original NetCDF bytes, gzip (lossless)
  png/YYYY/MM/DD/tmd_zr_YYYYMMDD_HHMMZ.png           original PNG bytes
  manifest.csv         one row per stored file (sha256 is of the original, un-gzipped bytes)
  collection_log.csv   one 'run' row per run plus one row per problem
  slot_inventory.csv, collection_report.md           written by `report`

Usage:
    python Radar_POC/tmd_radar_collect.py run [--log-file PATH]
    python Radar_POC/tmd_radar_collect.py report
    python Radar_POC/tmd_radar_collect.py health     # exit 1 if the newest NetCDF is > 60 min old
    python Radar_POC/tmd_radar_collect.py selftest
"""

import argparse
import csv
import gzip
import hashlib
import io
import os
import re
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import netcdf_file

BASE = "https://weather.tmd.go.th/composite/"
ARCHIVE = Path(os.getenv("TMD_RADAR_ARCHIVE", r"C:\Users\Brandon\Documents\TMD_Radar_Archive"))
MANIFEST = ARCHIVE / "manifest.csv"
RUNLOG = ARCHIVE / "collection_log.csv"
SLOT = timedelta(minutes=15)
BKK = timedelta(hours=7)
HTTP_TIMEOUT = 60
RETRY_WAITS = (10, 30)          # between the 3 attempts; the 5-min schedule is the outer retry
DEADLINE_S = 240                # leave the rest of a PNG backlog to the next run
STALE_MIN = 60                  # `health` fails when the newest NetCDF is older: 3 frames missed
UA = "thailand-rainwatch-capstone radar-poc"
NC_VAR = "var0_1_203_surface"
NC_SHAPE = (1, 1851, 1301)
PNG_SIZE = (1800, 2644)         # Web-Mercator render of the same lon 95-108, lat 4-22.5 box
MANIFEST_COLS = ["stamp_utc", "product", "path", "bytes", "sha256", "fetched_utc", "lag_min",
                 "valid_px", "rain_px", "max_mmh", "layout_ok", "dup_of", "source_url"]
LOG_COLS = ["run_utc", "event", "product", "stamp_utc", "detail", "list_first_utc", "list_last_utc"]


def log(msg):
    print(f"[{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}Z] {msg}", flush=True)


def iso(t):
    return t.strftime("%Y-%m-%dT%H:%MZ")


def iso_s(t):
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(text):
    return datetime.strptime(text, "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)


def parse_stamp(text):
    """TMD's '2026-10-29 08:00' (UTC, optionally with :SS) -> aware datetime."""
    return datetime.strptime(text.strip()[:16], "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)


def parse_list(text):
    """images_composite.list -> sorted [(stamp, 'zr/NN.png')], parsed the way TMD's viewer does."""
    out = []
    for line in text.splitlines():
        m = re.match(r'^(\S+)\s+"([^"]+)"\s+overlay=(.+)$', line.strip())
        if not m:
            continue
        overlays = [s.strip() for s in m.group(3).split(",")]
        rf = next((f for f in overlays if re.search(r"(zr|\d{2}\.png)", f, re.I)), None)
        if rf:
            out.append((parse_stamp(m.group(2)), rf if "/" in rf else "zr/" + rf))
    return sorted(out)


def missing_slots(stamps):
    """15-min slots absent between the first and last of a sorted list of stamps."""
    have, out = set(stamps), []
    t = stamps[0] if stamps else None
    while stamps and t <= stamps[-1]:
        if t not in have:
            out.append(t)
        t += SLOT
    return out


def http_get(path):
    url = BASE + path
    for attempt, wait in enumerate((*RETRY_WAITS, None), 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Cache-Control": "no-cache"})
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as r:
                body, want = r.read(), r.headers.get("Content-Length")
            if want is not None and int(want) != len(body):
                raise IOError(f"short read {len(body)}/{want} bytes")
            return body
        except Exception as exc:
            if wait is None:
                raise IOError(f"{path}: {type(exc).__name__}: {exc} (after {attempt} attempts)") from exc
            log(f"{path} attempt {attempt} failed ({type(exc).__name__}: {exc}); retry in {wait}s")
            time.sleep(wait)


def nc_summary(body):
    """Parse the whole NetCDF -> (stamp, valid_px, rain_px, max_mmh, layout_ok). Raises if unreadable."""
    f = netcdf_file(io.BytesIO(body), "r", mmap=False)
    try:
        stamp = datetime.fromtimestamp(float(f.variables["time"][0]), timezone.utc)
        var = f.variables.get(NC_VAR)
        if var is None:
            return stamp, "", "", "", False
        a = np.asarray(var[:], dtype=np.float32)
        v = a[a < 1e20]
        return stamp, v.size, int((v > 0).sum()), round(float(v.max()), 2) if v.size else 0.0, a.shape == NC_SHAPE
    finally:
        f.close()


def png_summary(body):
    """Decode the PNG fully -> (rain_px, layout_ok); rain_px counts coloured (opaque) pixels."""
    im = Image.open(io.BytesIO(body))
    im.load()
    rain = int((np.asarray(im)[..., 3] > 0).sum()) if im.mode == "RGBA" else ""
    return rain, im.size == PNG_SIZE


def read_csv(path):
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def append_csv(path, cols, row):
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        if new:
            w.writeheader()
        w.writerow(row)


def store(rel, data):
    path = ARCHIVE / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


class Run:
    def __init__(self):
        self.t0 = time.monotonic()
        self.utc = datetime.now(timezone.utc)
        rows = read_csv(MANIFEST)
        self.have = {(r["product"], r["stamp_utc"]) for r in rows}
        self.sha_stamp = {r["sha256"]: r["stamp_utc"] for r in rows}
        self.known_gaps = {r["stamp_utc"] for r in read_csv(RUNLOG) if r["event"] == "source_gap"}
        self.events = []
        self.list_window = ("", "")

    def event(self, event, product, stamp="", detail=""):
        self.events.append({"event": event, "product": product, "stamp_utc": stamp, "detail": detail})
        log(f"{event} {product} {stamp} {detail}".strip())

    def keep(self, product, stamp, rel, body, url, valid_px, rain_px, max_mmh, layout_ok):
        """Store the file first, then its manifest row: a crash in between only causes a re-fetch."""
        store(rel, gzip.compress(body, compresslevel=9) if rel.endswith(".gz") else body)
        sha = hashlib.sha256(body).hexdigest()
        dup = self.sha_stamp.get(sha, "")
        if dup:
            self.event("duplicate_content", product, iso(stamp), f"same bytes as {dup}")
        if not layout_ok:
            self.event("layout_changed", product, iso(stamp), "stored anyway; check the format")
        fetched = datetime.now(timezone.utc)
        append_csv(MANIFEST, MANIFEST_COLS, {
            "stamp_utc": iso(stamp), "product": product, "path": rel, "bytes": len(body), "sha256": sha,
            "fetched_utc": iso_s(fetched), "lag_min": round((fetched - stamp).total_seconds() / 60, 1),
            "valid_px": valid_px, "rain_px": rain_px, "max_mmh": max_mmh, "layout_ok": int(layout_ok),
            "dup_of": dup, "source_url": url})
        self.have.add((product, iso(stamp)))
        self.sha_stamp.setdefault(sha, iso(stamp))

    def collect_nc(self):
        shared = parse_stamp(http_get("radar.sharing.latest").decode("utf-8", "replace"))
        if ("nc", iso(shared)) in self.have:
            return f"nc {iso(shared)} already stored"
        body = http_get("data/composite.nc")
        try:
            stamp, valid, rain, mx, ok = nc_summary(body)
        except Exception as exc:
            self.event("verify_failed", "nc", iso(shared), f"{type(exc).__name__}: {exc}")
            return "nc unreadable"
        if stamp != shared:     # the file's own time is the authority
            self.event("stamp_mismatch", "nc", iso(stamp), f"radar.sharing.latest said {iso(shared)}")
            if ("nc", iso(stamp)) in self.have:
                return f"nc {iso(stamp)} already stored"
        rel = f"nc/{stamp:%Y/%m/%d}/tmd_composite_{stamp:%Y%m%d_%H%M}Z.nc.gz"
        self.keep("nc", stamp, rel, body, BASE + "data/composite.nc", valid, rain, mx, ok)
        return f"nc {iso(stamp)} stored ({rain} rain px, max {mx} mm/h)"

    def collect_png(self):
        text = http_get("images_composite.list").decode("utf-8", "replace")
        frames = parse_list(text)
        if not frames:
            raise IOError("images_composite.list lists no frames")
        stamps = [s for s, _ in frames]
        self.list_window = (iso(stamps[0]), iso(stamps[-1]))
        todo = [(s, f) for s, f in frames if ("png", iso(s)) not in self.have]
        got = []
        for s, f in todo:
            if time.monotonic() - self.t0 > DEADLINE_S:
                self.event("deadline", "png", iso(s), f"{len(todo) - len(got)} frames left for the next run")
                break
            try:
                got.append((s, f, http_get("images/" + f)))
            except Exception as exc:
                self.event("fetch_failed", "png", iso(s), str(exc))
        if got and http_get("images_composite.list").decode("utf-8", "replace") != text:
            self.event("list_changed", "png", "", f"dropped {len(got)} frames fetched mid-update")
            return f"png list changed mid-run, {len(got)} dropped"
        stored = 0
        for s, f, body in got:
            try:
                rain, ok = png_summary(body)
            except Exception as exc:
                self.event("verify_failed", "png", iso(s), f"{type(exc).__name__}: {exc}")
                continue
            self.keep("png", s, f"png/{s:%Y/%m/%d}/tmd_zr_{s:%Y%m%d_%H%M}Z.png", body, BASE + "images/" + f,
                      "", rain, "", ok)
            stored += 1
        gaps = missing_slots(stamps)
        for g in gaps:
            if iso(g) not in self.known_gaps:
                self.known_gaps.add(iso(g))
                self.event("source_gap", "png", iso(g), "slot missing from TMD's frame list")
        return f"png stored {stored} of {len(todo)} new, list {len(frames)} frames, {len(gaps)} source gaps"


def cmd_run():
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    run = Run()
    summary = []
    for product, step in (("nc", run.collect_nc), ("png", run.collect_png)):
        try:
            summary.append(step())
        except Exception as exc:
            run.event("fetch_failed", product, "", f"{type(exc).__name__}: {exc}")
            summary.append(f"{product} FAILED")
    detail = "; ".join(summary) + f"; {time.monotonic() - run.t0:.0f}s"
    for e in run.events:
        append_csv(RUNLOG, LOG_COLS, {"run_utc": iso_s(run.utc), **e})
    append_csv(RUNLOG, LOG_COLS, {"run_utc": iso_s(run.utc), "event": "run", "detail": detail,
                                  "list_first_utc": run.list_window[0], "list_last_utc": run.list_window[1]})
    log(detail)


def cmd_health():
    """One status line; exit 1 when the newest NetCDF is over STALE_MIN old (the alert hook)."""
    now = datetime.now(timezone.utc)
    stamps = [parse_iso(r["stamp_utc"]) for r in read_csv(MANIFEST) if r["product"] == "nc"]
    recent = [r for r in read_csv(RUNLOG) if r["event"] != "run"
              and now - datetime.strptime(r["run_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
              <= timedelta(minutes=STALE_MIN)]
    problems = f"; problems in the last {STALE_MIN} min: " + ", ".join(
        sorted({f"{r['event']} {r['product']}" for r in recent})) if recent else ""
    if not stamps:
        print("no NetCDF frame archived yet" + problems)
        sys.exit(1)
    age = (now - max(stamps)).total_seconds() / 60
    print(f"newest NetCDF {iso(max(stamps))}, {age:.0f} min old" + problems)
    sys.exit(0 if age <= STALE_MIN else 1)


# ----------------------------------------------------------------------------- report

def slot_status(has_nc, has_png, source_gap):
    if has_nc and has_png:
        return "nc+png"
    if has_nc or has_png:
        return "nc_only" if has_nc else "png_only"
    # source_gap: a run saw TMD's list skip this slot, so it was never published
    return "source_gap" if source_gap else "not_collected"


def inventory(manifest, runlog, end):
    nc = {r["stamp_utc"] for r in manifest if r["product"] == "nc"}
    png = {r["stamp_utc"] for r in manifest if r["product"] == "png"}
    gaps = {r["stamp_utc"] for r in runlog if r["event"] == "source_gap"}
    t, rows = min(parse_iso(s) for s in nc | png), []
    while t <= end:
        k = iso(t)
        rows.append({"slot_utc": k, "slot_bkk": (t + BKK).strftime("%Y-%m-%d %H:%M"),
                     "nc": int(k in nc), "png": int(k in png),
                     "status": slot_status(k in nc, k in png, k in gaps)})
        t += SLOT
    return rows


def gap_runs(rows, bad):
    """Contiguous runs of slots whose status is in `bad` -> [(first, last, n)]."""
    out = []
    for r in rows:
        if r["status"] in bad:
            if out and parse_iso(out[-1][1]) + SLOT == parse_iso(r["slot_utc"]):
                out[-1] = (out[-1][0], r["slot_utc"], out[-1][2] + 1)
            else:
                out.append((r["slot_utc"], r["slot_utc"], 1))
    return out


def cmd_report():
    manifest, runlog = read_csv(MANIFEST), read_csv(RUNLOG)
    if not manifest:
        sys.exit("manifest is empty - nothing collected yet")
    # end at the newest slot TMD has listed, so frames not yet published don't count as missing
    end = max(parse_iso(r["stamp_utc"]) for r in manifest)
    rows = inventory(manifest, runlog, end)
    with open(ARCHIVE / "slot_inventory.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    counts = {s: sum(r["status"] == s for r in rows) for s in
              ("nc+png", "nc_only", "png_only", "source_gap", "not_collected")}
    n = len(rows)
    runs = [datetime.strptime(r["run_utc"], "%Y-%m-%dT%H:%M:%SZ") for r in runlog if r["event"] == "run"]
    idle = [(a, b) for a, b in zip(runs, runs[1:]) if b - a > timedelta(minutes=20)]
    lags = sorted(float(r["lag_min"]) for r in manifest if r["product"] == "nc")
    problems = {}
    for r in runlog:
        if r["event"] != "run":
            problems[(r["event"], r["product"])] = problems.get((r["event"], r["product"]), 0) + 1
    days = {}
    for r in rows:
        d = days.setdefault(r["slot_utc"][:10], {"slots": 0, "nc": 0, "png": 0})
        d["slots"] += 1
        d["nc"] += r["nc"]
        d["png"] += r["png"]
    pct = lambda k: f"{100 * k / n:.1f}%"
    md = [f"# TMD radar collection report",
          f"",
          f"Generated {iso_s(datetime.now(timezone.utc))} from `manifest.csv` and `collection_log.csv`. "
          f"All times UTC unless marked BKK (UTC+7).",
          f"",
          f"Slots {rows[0]['slot_utc']} to {rows[-1]['slot_utc']}: {n} x 15 min.",
          f"",
          f"| status | slots | share |", f"|---|---:|---:|"]
    md += [f"| {k} | {v} | {pct(v)} |" for k, v in counts.items()]
    md += ["", f"NetCDF (full precision) in {pct(counts['nc+png'] + counts['nc_only'])} of slots; "
               f"any product in {pct(n - counts['source_gap'] - counts['not_collected'])}.", ""]
    if lags:
        q = lambda p: lags[min(len(lags) - 1, int(p * len(lags)))]
        md += [f"NetCDF first-seen lag after its stamp (min): min {lags[0]}, median {q(0.5)}, p90 {q(0.9)} "
               f"(n={len(lags)}; 5-min polling, so true latency is up to 5 min less).", ""]
    md += ["## Per day (UTC)", "", "| day | slots | nc | png |", "|---|---:|---:|---:|"]
    md += [f"| {d} | {v['slots']} | {v['nc']} | {v['png']} |" for d, v in sorted(days.items())]
    md += ["", "## Missing slots (no product at all)", ""]
    gaps = gap_runs(rows, {"source_gap", "not_collected"})
    md += [f"- {a} to {b} ({k} slots, {next(r['status'] for r in rows if r['slot_utc'] == a)})"
           for a, b, k in gaps] or ["- none"]
    md += ["", "## NetCDF missing but PNG backfilled", ""]
    md += [f"- {a} to {b} ({k} slots)" for a, b, k in gap_runs(rows, {"png_only"})] or ["- none"]
    md += ["", "## Collector idle > 20 min (machine asleep or task not running)", ""]
    md += [f"- {iso_s(a)} to {iso_s(b)} ({(b - a).total_seconds() / 3600:.1f} h)" for a, b in idle] or ["- none"]
    md += ["", "## Problems logged", ""]
    md += [f"- {e} / {p}: {k}" for (e, p), k in sorted(problems.items())] or ["- none"]
    md += ["", f"Runs logged: {len(runs)}."]
    (ARCHIVE / "collection_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


# ----------------------------------------------------------------------------- selftest

def cmd_selftest():
    t = lambda s: parse_stamp(s)
    lst = ('background_THA.png "2026-10-29 07:45" overlay=zr/23.png\n'
           'background_THA.png "2026-10-29 07:15" overlay=21.png\n'
           '# comment\n')
    frames = parse_list(lst)
    assert frames == [(t("2026-10-29 07:15"), "zr/21.png"), (t("2026-10-29 07:45"), "zr/23.png")], frames
    assert missing_slots([s for s, _ in frames]) == [t("2026-10-29 07:30")]
    assert parse_stamp("2026-10-29 08:00:00") == datetime(2026, 10, 29, 8, 0, tzinfo=timezone.utc)

    buf = io.BytesIO()
    f = netcdf_file(buf, "w")
    f.createDimension("time", 1)
    f.createDimension("latitude", 3)
    f.createDimension("longitude", 2)
    f.createVariable("time", "d", ("time",))[:] = [1793260800.0]          # 2026-10-29 08:00 UTC
    v = f.createVariable(NC_VAR, "f", ("time", "latitude", "longitude"))
    v[:] = np.array([[[9.999e20, 0.0], [1.5, 9.999e20], [167.5, 9.999e20]]], dtype=np.float32)
    f.flush()
    body = buf.getvalue()
    f.close()
    stamp, valid, rain, mx, ok = nc_summary(body)
    assert (stamp, valid, rain, mx, ok) == (t("2026-10-29 08:00"), 3, 2, 167.5, False), (stamp, valid, rain, mx, ok)
    try:
        nc_summary(body[:-8])
        raise AssertionError("truncated NetCDF parsed")
    except AssertionError:
        raise
    except Exception:
        pass

    assert slot_status(True, False, True) == "nc_only"
    assert slot_status(False, False, True) == "source_gap"
    assert slot_status(False, False, False) == "not_collected"
    man = [{"product": "nc", "stamp_utc": "2026-10-29T07:00Z"}, {"product": "png", "stamp_utc": "2026-10-29T07:00Z"},
           {"product": "png", "stamp_utc": "2026-10-29T07:15Z"}]
    rl = [{"event": "run", "stamp_utc": ""}, {"event": "source_gap", "stamp_utc": "2026-10-29T07:30Z"}]
    rows = inventory(man, rl, t("2026-10-29 07:45"))
    assert [r["status"] for r in rows] == ["nc+png", "png_only", "source_gap", "not_collected"], rows
    assert rows[0]["slot_bkk"] == "2026-10-29 14:00"
    assert gap_runs(rows, {"source_gap", "not_collected"}) == [("2026-10-29T07:30Z", "2026-10-29T07:45Z", 2)]
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report", "health", "selftest"])
    ap.add_argument("--log-file", default=None)
    args = ap.parse_args()
    if args.log_file:
        lf = Path(args.log_file)
        lf.parent.mkdir(parents=True, exist_ok=True)
        if lf.exists() and lf.stat().st_size > 2_000_000:
            lf.replace(lf.with_name(lf.name + ".1"))
        sys.stdout = sys.stderr = open(lf, "a", encoding="utf-8", buffering=1)
    {"run": cmd_run, "report": cmd_report, "health": cmd_health, "selftest": cmd_selftest}[args.cmd]()


if __name__ == "__main__":
    main()
