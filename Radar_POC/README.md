# TMD radar archive (Phase 4 POC)

Data-collection proof of concept, started 2026-10-29. It archives the Thai Meteorological
Department's live nationwide radar composite so a radar feature can be tested later. Nothing
here touches the live model, predict, verify or the dashboard, and none of it is in the report.

- Code: this folder, branch `radar-poc` (worktree `C:\Users\Brandon\Documents\Phase1_radar_poc`).
  The branch is not pushed.
- Data: `C:\Users\Brandon\Documents\TMD_Radar_Archive` (outside every git checkout, so a worktree
  removal can't delete it).
- Scheduled task: `TMD Radar Collector`, every 5 min.

## Source

All of it is on `https://weather.tmd.go.th/composite/`. It is public, needs no key or
registration, and was checked on 2026-10-29.

| URL (under `/composite/`) | What it is | Kept? |
|---|---|---|
| `data/composite.nc` | **Latest frame only.** NetCDF-3 (wgrib2 output), variable `var0_1_203_surface`, 1851 × 1301 at 0.01° (lat 4.00–22.50 N, lon 95.00–108.00 E), fill 9.999e20, 9.66 MB | yes, gzip'd original bytes |
| `data/composite.ascii` | The same grid as ESRI ASCII: all values on one line, then a `YYYY-MM-DD HH:MM:SS` line. Matches the NetCDF to 1e-7 | one sample in `meta/` |
| `radar.sharing.latest` | `YYYY-MM-DD HH:MM` (UTC) stamp of the latest frame | used to skip known frames |
| `images_composite.list` | The last 25 frames (6 h) as `images/zr/00.png` (oldest) to `24.png` (newest) | used |
| `images/zr/NN.png` | 1800 × 2644 RGBA matplotlib render of the same field, colour-binned | yes, original bytes |

**Product.** A national mosaic of PCAPPI at 2 km, converted to rain rate with Marshall–Palmer
Z = 200 R^1.6 (from the viewer's product note; the file itself has no units attribute). The
values are mm/h:
- they are quantised: 0, then 1.04, 1.15, … rising in widening steps (167.5 was the largest seen);
- **no value is between 0 and ~1.04**, so rates below about 1 mm/h are floored to 0;
- a value of 0 marks weak echo, and on 29 Oct 0-valued pixels were 2–3 times as many as rain pixels;
- the fill value marks both "no echo" and "no radar". **There is no coverage mask.**

**Cadence and latency.** Frames are stamped in UTC at :00, :15, :30 and :45. On 29 Oct:
- the 08:15 frame first appeared between 08:26:35 and 08:31:32 UTC, 11–17 min after its stamp;
- the 08:30 frame appeared between 08:41:55 and 08:42:16, 12 min after its stamp;
- the 9 Oct check found 10–15 min.

At the 08:30 switch, a probe polled every 20 s. The frame list, `radar.sharing.latest`, the
NetCDF and the PNGs all changed within the same 20-s interval, and the old newest PNG shifted to
slot 23 in that step. No half-updated state was seen. `collection_report.md` keeps the measured
lag up to date.

**The PNG frames** are in Web Mercator over the same box. 99% of PNG colour falls on NetCDF echo
of the same stamp, against 61% if you assume a plain lat/lon grid. The palette is exact (0
off-palette pixels). There are 19 colours for 23 labelled bins, and four colours are shared by two
bins. Decoded, a pixel's NetCDF value is usually inside its colour's bin (median on the bin edge),
with about one bin of scatter from resampling. The three lowest colours (0.21–0.72 mm/h) are drawn
over NetCDF zeros. So a PNG gives rain/no-rain almost exactly and intensity to about one bin, which
is good enough to backfill a missed NetCDF slot.

**Terms** (`disclaimer.html`, in Thai, copy in `meta/`):
- TMD publishes the data as a public service, and users carry the risk of using it;
- the data is near-real-time and has not been quality-checked;
- echoes may be non-rain: aircraft, birds, insects, ground clutter, anomalous propagation, sea clutter;
- light rain and rain close to a radar can be missed;
- times are UTC (Thailand = UTC+7);
- the Bangkok and Royal Rainmaking radars belong to BMA and the Department of Royal Rainmaking.

No licence is named and nothing forbids collecting or reusing the data. Credit TMD whenever the
data is used.

**Dead or other routes (not used).**
- `THA_QPE.php` and `THA_Z.php` point at `compositeQPE_VTBB_latest.png`, `.asc.zip` and
  `compositeZC_VTBB_latest.png`, which all return 404 today.
- From the 2026-10-09 check, not re-verified:
  - per-station GIF loops (`<code>.php`, 27 stations, 1–2 h);
  - BMA's two Bangkok radars ("commercial use absolutely prohibited");
  - RainViewer (TMD-sourced, 2 h, personal use);
  - `www2.tmd.go.th/radarfile/` (manifests stop July 2026).
- TMD's open-data portal has no radar, and no public radar archive exists anywhere.

## How the collector works

`tmd_radar_collect.py run` runs every 5 min (Task Scheduler: IgnoreNew, runs on battery, 10-min
time limit, no wake timer).

1. **NetCDF.** It reads `radar.sharing.latest` and stops if that stamp is already stored.
   Otherwise it downloads `composite.nc` and parses the whole file, so a truncated download
   fails. The file's own reference time is the stamp; a mismatch is logged. The original bytes
   are stored gzip'd at level 9 (lossless, ~140 KB per frame).
2. **PNG.** It downloads every listed frame not yet stored, then re-reads the list. If the list
   changed mid-run, the batch is dropped and the next run fetches it again. Each PNG is decoded to
   check it.
3. **Retries.** Every HTTP GET gets 3 attempts, 10 s and 30 s apart, and Content-Length is checked.
   The 5-min schedule is the outer retry. A run stops pulling a PNG backlog after 240 s.
4. **Duplicates.** A (product, stamp) pair in the manifest is never fetched again. A file whose
   bytes equal an earlier frame's is still stored, but flagged in `dup_of`.

Everything is UTC. File names end in `Z`, and CSV columns end in `_utc`. The report adds Bangkok
time (UTC+7) for lining up with prediction stamps. This PC's clock runs at UTC+8, not Bangkok
time, so the collector never uses local time.

### Archive layout

```
TMD_Radar_Archive/
  nc/YYYY/MM/DD/tmd_composite_YYYYMMDD_HHMMZ.nc.gz   original NetCDF bytes (gzip)
  png/YYYY/MM/DD/tmd_zr_YYYYMMDD_HHMMZ.png           original PNG bytes
  manifest.csv          one row per stored file: stamp, sha256 of the original bytes, size,
                        fetch time, lag, echo / rain pixel counts, max mm/h, layout_ok, dup_of
  collection_log.csv    one 'run' row per run (with the list window it saw) + one row per problem
  slot_inventory.csv    per 15-min slot: nc / png / status (written by `report`)
  collection_report.md  coverage, missing slots, idle spells, lag, problem counts (by `report`)
  collector.log         text log (rotates at 2 MB)
  meta/                 viewer page, disclaimer, home page, basemap, one ASCII sample (29 Oct)
```

To get a frame back: `gzip.decompress(path.read_bytes())` returns the exact bytes TMD served,
and its sha256 matches the manifest.

**Slot statuses** in the inventory:
- `nc+png`, `nc_only` or `png_only`: the slot was collected;
- `source_gap`: a run saw TMD's frame list skip the slot (logged once per slot), so TMD never
  published it;
- `not_collected`: no run looked while the slot was available.

Storage is about 32 MB/day at full capture, or roughly 1 GB a month.

### Operating it

```powershell
# status and missing-scan summary (rewrites slot_inventory.csv and collection_report.md)
& "C:\Users\Brandon\Documents\Phase 1\.venv\Scripts\python.exe" Radar_POC\tmd_radar_collect.py report
# radar persistence vs IMERG labels vs the live model (needs Postgres; labels lag ~1.5 days)
& "C:\Users\Brandon\Documents\Phase 1\.venv\Scripts\python.exe" Radar_POC\score_radar_persistence.py
# pause / resume / remove the task
Disable-ScheduledTask -TaskName 'TMD Radar Collector'
Enable-ScheduledTask  -TaskName 'TMD Radar Collector'
Unregister-ScheduledTask -TaskName 'TMD Radar Collector' -Confirm:$false
```

Run both from the worktree root. The task uses the main checkout's `.venv` and this worktree's
script, so **don't remove the worktree while the task is registered**.

## Limitations

- **Sleep is permanent loss for the NetCDF.** TMD serves only the latest file, so every 15 min the
  laptop is asleep costs a full-precision frame. PNG backfill covers sleeps up to 6 h; anything
  longer is gone. The hourly predict history shows the laptop is awake only ~8–14 h/day
  (`rainwatch-sleep-gaps`). The fixes would be a wake timer, keeping the PC awake, or running the
  collector off the laptop. A GitHub Actions copy would need a workflow on `main` and would
  commit ~200 files a day, so it was left out of this POC. The off-laptop option is set out in
  [`gcp/GCP_SETUP.md`](gcp/GCP_SETUP.md): a Google Cloud e2-micro at about $3.65 a month.
- No coverage mask: you can't tell "no echo" from "no radar" in one frame. The union over many
  frames, or the radar sites and their ranges, has to stand in for one.
- Rates come from one fixed Z–R. Clutter, anomalous propagation, the cone of silence near each
  radar, and beam overshoot at long range are not corrected. In the 29 Oct sample one radar showed
  a radial interference spoke.
- Sub-1 mm/h rain reads as 0, which matters for the 0.1 mm targets.
- The source is a single "latest" file with no versioning or Last-Modified header. A frame TMD
  re-issues under the same stamp would not be fetched again.
