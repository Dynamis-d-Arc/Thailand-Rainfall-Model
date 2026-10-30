# %% [markdown]
# # Thailand_Rain_V15 — rain onset, confidence intervals, reliability figures
#
# No model is fitted. Three questions about the predictions V13 (blocked CV) and V14
# (forward-in-time) already made:
#
#  - `obs`:       observed IMERG rain in the hour just before each panel row's label window.
#                 IMERG stamps an hour by its start and labels_v2 covers stamps t..t+h-1
#                 (Docs/label_fix_2026-10-27.md), so "raining at issue time" is stamp t-1.
#                 V7's `imerg_persistence` reads stamp t, which is right only for the old
#                 [t+1, t+1+h) window; under labels_v2 it is the first label hour.
#  - `onset`:     condition on the state at issue time. Among rows dry at t-1, ROC ranks
#                 onset against stays-dry; among rows wet at t-1, continuing against stopping.
#                 Persistence is constant inside each state, so these ROCs are pure
#                 change-of-state skill.
#  - `bootstrap`: day-block bootstrap (all rows of a local calendar day move together) for
#                 ROC per target and state, the IR gain, the V14 forward-minus-CV ROC gap and
#                 the calibration ECE improvement. Weighted AUC from one sort per score array.
#  - `figures`:   reliability diagrams from V14's v14_reliability.csv.
#
# Usage:
#     python Thailand_Rain_V15_onset_ci.py obs
#     python Thailand_Rain_V15_onset_ci.py onset
#     python Thailand_Rain_V15_onset_ci.py bootstrap [n_reps]
#     python Thailand_Rain_V15_onset_ci.py figures
#
# ## Result (2026-10-30, labels_v2; v15_onset_scores.csv, v15_bootstrap_ci.csv,
# ## v15_reliability_2026.png/.svg; 95% intervals from 1,000 day-block replicates)
#
# - Onset is predicted, not just ongoing rain. Among rows dry at t-1, the IR model ranks
#   onset against stays-dry at ROC 0.866-0.886 (V13 OOF), and IR adds +0.013..+0.040 there,
#   every interval clear of zero. That is 36-82% of the gain on rows already raining
#   (+0.029..+0.049), the share falling with horizon. These ROCs still contain place-and-season skill (constant
#   within a state for persistence, not for climatology).
# - Stopping is the weak spot. Among rows wet at t-1, continuing vs stopping scores ROC
#   0.704-0.779 in CV and 0.621-0.740 on the 2026 forward test.
# - At the deployed F1 thresholds the IR model flags fewer onsets than om_only on all six
#   targets (CV POD 0.41-0.80 vs 0.47-0.82) with fewer false alarms (onset-row FAR
#   0.37-0.74 vs 0.40-0.82), and catches 75-96% of continuing rain.
# - Intervals: CV ROC about +-0.005 (e.g. h1 1.0 mm 0.914 [0.910, 0.919]); IR gain
#   +0.019..+0.047 with every interval at least +0.017; the V14 forward-minus-CV ROC gap is
#   below zero for five targets but never beyond -0.004 (h6 0.1 mm interval includes 0); the
#   time-held-out calibrator lowers ECE by +0.004..+0.013, all intervals clear of zero.
# - Day blocks treat consecutive days as independent; multi-day systems would widen the
#   intervals somewhat. Weekly blocks were not run.

# %%
import sys
import time

import numpy as np
import pandas as pd

from BKK_Rain_V3 import TARGETS, TARGET_NAMES, log, pick_threshold
from Thailand_Rain_V3 import CACHE, IMERG_LABEL_COLUMN, IMERG_TABLE_NAME, connect
import Thailand_Rain_V3 as v3
from Thailand_Rain_V10_deploy import PROJECT_ROOT
import Thailand_Rain_V14_forward as v14

OUTPUT_DIR = (PROJECT_ROOT / "ML_Model_V2" / "trained_models"
              / f"om_thailand_rain_v15_onset_ci_{v3.LABEL_VERSION}")
OBS_PATH = CACHE / "imerg_prev_hour_pixelmax.npy"
CONFIGS = ["om_only", "hw_allseason"]
STATES = ["all", "dry_now", "wet_now"]


def load_panel():
    return (np.load(CACHE / "y.npy"), np.load(CACHE / "forecast_time.npy"),
            np.load(CACHE / "unit_id.npy"))


# %%
def phase_obs():
    y, tf, unit = load_panel()
    prev = tf - np.timedelta64(1, "h")
    lo, hi = pd.Timestamp(prev.min()).to_pydatetime(), pd.Timestamp(prev.max()).to_pydatetime()
    hours = sorted({int(h) for h in pd.DatetimeIndex(np.unique(prev)).hour})
    log(f"pulling {IMERG_LABEL_COLUMN} at stamps t-1 (local hours {hours}), {lo} .. {hi}")
    conn = connect()
    try:
        cur = conn.cursor()
        cur.execute(f"""
            SELECT grid_number, local_observation_time, {IMERG_LABEL_COLUMN}
            FROM {IMERG_TABLE_NAME}
            WHERE is_complete_hour AND {IMERG_LABEL_COLUMN} IS NOT NULL
              AND local_observation_time BETWEEN %s AND %s
              AND EXTRACT(HOUR FROM local_observation_time)::int = ANY(%s)
        """, (lo, hi, hours))
        rows = cur.fetchall()
    finally:
        conn.close()
    obs = pd.DataFrame(rows, columns=["grid_number", "t", "mm"])
    obs["t"] = obs["t"].astype("datetime64[s]")
    panel = pd.DataFrame({"grid_number": unit.astype("int64"), "t": prev.astype("datetime64[s]")})
    merged = panel.merge(obs, on=["grid_number", "t"], how="left", validate="many_to_one")
    out = merged["mm"].to_numpy(dtype="float32")
    assert len(out) == len(y)

    # the t-1 hour must be a different hour from the h1 label's own: agree often, never always
    for i, (h, thr) in enumerate(TARGETS):
        if h != 1:
            continue
        m = np.isfinite(out)
        agree = float(((out[m] >= thr) == (y[m, i] == 1)).mean())
        log(f"  h1 {thr} mm: wet-now {float((out[m] >= thr).mean()):.3f}, "
            f"label base {float(y[m, i].mean()):.3f}, agreement {agree:.3f}")
        assert agree < 0.99, "t-1 rain matches the h1 label almost exactly: window overlap?"
    log(f"matched {np.isfinite(out).sum():,} / {len(out):,} rows")
    np.save(OBS_PATH, out)


# %%
def state_masks(obs, thr):
    known = np.isfinite(obs)
    wet = known & (obs >= thr)
    return {"all": np.ones(len(obs), bool), "dry_now": known & ~wet, "wet_now": wet}


def phase_onset():
    from sklearn.metrics import roc_auc_score
    y, tf, _ = load_panel()
    obs = np.load(OBS_PATH)
    oof = {c: np.load(CACHE / f"oof_cal_{c}.npy") for c in CONFIGS}
    train, calib, test = v14.split_masks(tf)
    later = calib | test
    fwd = {}
    for c in CONFIGS:
        from sklearn.isotonic import IsotonicRegression
        cols = []
        for i in range(len(TARGETS)):
            d = np.load(v14.pred_path(c, i))
            raw = d["raw"]
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            cols.append(iso.fit(raw[calib[later]], y[calib, i]).predict(raw[test[later]]))
        fwd[c] = np.column_stack(cols).astype("float32")

    rows = []
    for i, (h, thr) in enumerate(TARGETS):
        name = TARGET_NAMES[i]
        masks = state_masks(obs, thr)
        thresholds = {c: pick_threshold(oof[c][:, i], y[:, i]) for c in CONFIGS}
        for state, m in masks.items():
            for source, period in (("cv_oof", np.ones(len(y), bool)), ("forward_cal_time", test)):
                sel = m & period
                yy = y[sel, i]
                row = {"target": name, "state": state, "source": source, "rows": int(sel.sum()),
                       "base_rate": float(yy.mean())}
                for c in CONFIGS:
                    p = oof[c][sel, i] if source == "cv_oof" else fwd[c][m[test], i]
                    row[f"{c}_roc"] = float(roc_auc_score(yy, p))
                    flags = p >= thresholds[c]
                    row[f"{c}_pod"] = float(flags[yy == 1].mean())
                    row[f"{c}_far"] = float((yy[flags] == 0).mean()) if flags.any() else np.nan
                row["ir_gain"] = row["hw_allseason_roc"] - row["om_only_roc"]
                rows.append(row)
        log(f"  scored {name}")

    out = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_DIR / "v15_onset_scores.csv", index=False)
    pd.set_option("display.width", 200)
    show = out[out["source"] == "cv_oof"][["target", "state", "rows", "base_rate", "om_only_roc",
                                            "hw_allseason_roc", "ir_gain", "hw_allseason_pod"]]
    log("=== V13 OOF by state at issue time (onset = dry_now positives) ===")
    print(show.round(3).to_string(index=False))
    log(f"saved {OUTPUT_DIR / 'v15_onset_scores.csv'}")


# %% [markdown]
# ## Day-block bootstrap
#
# A replicate draws n_days days with replacement; every row inherits its day's count as a
# weight. `TiedAuc` sorts a score array once and groups equal scores (isotonic outputs are
# heavily tied), so each replicate's weighted AUC is two bincounts and a cumsum. Rows outside
# a subset get weight 0, which is exactly the subset AUC.

# %%
class TiedAuc:
    def __init__(self, score, y):
        _, group = np.unique(score, return_inverse=True)
        self.n_groups = int(group.max()) + 1
        # positives and negatives get their own index, the other class a dump bin at the end,
        # so a replicate is two plain bincounts with no per-call masking
        yb = y.astype(bool)
        self.pos_idx = np.where(yb, group, self.n_groups)
        self.neg_idx = np.where(yb, self.n_groups, group)

    def __call__(self, w):
        pos = np.bincount(self.pos_idx, weights=w, minlength=self.n_groups + 1)[:-1]
        neg = np.bincount(self.neg_idx, weights=w, minlength=self.n_groups + 1)[:-1]
        below = np.cumsum(neg) - neg
        total = pos.sum() * neg.sum()
        return float(((below + 0.5 * neg) * pos).sum() / total) if total else np.nan


def weighted_ece(p, y, w, bins=v14.RELIABILITY_BINS):
    b = np.minimum((p * bins).astype(int), bins - 1)
    gap = np.abs(np.bincount(b, w * p, bins) - np.bincount(b, w * y, bins))
    return float(gap.sum() / w.sum())


def self_check_auc():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    s = np.round(rng.random(50_000), 2)                 # many ties, like isotonic output
    y = (rng.random(50_000) < s).astype("int8")
    w = rng.integers(0, 4, 50_000).astype(float)
    a = TiedAuc(s, y)
    assert abs(a(np.ones(len(s))) - roc_auc_score(y, s)) < 1e-12
    assert abs(a(w) - roc_auc_score(y, s, sample_weight=w)) < 1e-12
    log("TiedAuc matches sklearn (unweighted and weighted, with ties)")


def phase_bootstrap(n_reps=1000):
    self_check_auc()
    y, tf, _ = load_panel()
    obs = np.load(OBS_PATH)
    day = pd.DatetimeIndex(tf).normalize()
    _, day_idx = np.unique(day, return_inverse=True)
    n_days = int(day_idx.max()) + 1
    train, calib, test = v14.split_masks(tf)
    later = calib | test
    log(f"{len(y):,} rows in {n_days} days; test period {int(test.sum()):,} rows")

    from sklearn.isotonic import IsotonicRegression
    masks = {thr: {s: m.astype(float) for s, m in state_masks(obs, thr).items()}
             for thr in sorted({thr for _, thr in TARGETS})}
    prepared = []
    for i, (h, thr) in enumerate(TARGETS):
        aucs = {c: TiedAuc(np.load(CACHE / f"oof_cal_{c}.npy")[:, i], y[:, i]) for c in CONFIGS}
        # V14 on the test rows: forward (time-held-out isotonic) vs the V13 OOF, and the
        # random-row vs time-held-out calibration error, paired on the same replicate
        d = np.load(v14.pred_path("hw_allseason", i))
        raw, cal_random = d["raw"], d["cal_random"]
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        cal_time = iso.fit(raw[calib[later]], y[calib, i]).predict(raw[test[later]])
        prepared.append({
            "name": TARGET_NAMES[i], "thr": thr, "aucs": aucs,
            "fwd": TiedAuc(cal_time, y[test, i]),
            "cv_test": TiedAuc(np.load(CACHE / "oof_cal_hw_allseason.npy")[test, i], y[test, i]),
            "cal_random": cal_random[test[later]], "cal_time": cal_time,
            "yt": y[test, i].astype(float)})
        log(f"  prepared {TARGET_NAMES[i]}")

    def evaluate(w, wt):
        out = {}
        weighted = {thr: {s: w * mf for s, mf in states.items()} for thr, states in masks.items()}
        for p in prepared:
            name = p["name"]
            for state, ws in weighted[p["thr"]].items():
                roc = {c: p["aucs"][c](ws) for c in CONFIGS}
                for c in CONFIGS:
                    out[f"cv_roc|{c}|{name}|{state}"] = roc[c]
                out[f"ir_gain|{name}|{state}"] = roc["hw_allseason"] - roc["om_only"]
            out[f"forward_minus_cv|hw_allseason|{name}|all"] = p["fwd"](wt) - p["cv_test"](wt)
            out[f"ece_random_minus_time|hw_allseason|{name}|all"] = (
                weighted_ece(p["cal_random"], p["yt"], wt) - weighted_ece(p["cal_time"], p["yt"], wt))
        return out

    ones = np.ones(len(y))
    point = evaluate(ones, ones[test])
    keys = list(point)
    rng = np.random.default_rng([v3.RANDOM_STATE, 150])
    draws = np.empty((n_reps, len(keys)))
    t0 = time.time()
    for r in range(n_reps):
        w = np.bincount(rng.integers(0, n_days, n_days), minlength=n_days)[day_idx].astype(float)
        res = evaluate(w, w[test])
        draws[r] = [res[k] for k in keys]
        if r in (0, 9) or (r + 1) % 100 == 0:
            log(f"  replicate {r + 1}/{n_reps}  ({(time.time() - t0) / (r + 1):.1f} s each)")

    rows = []
    for j, key in enumerate(keys):
        stat, *rest = key.split("|")
        config, target, state = rest if len(rest) == 3 else ("hw_allseason - om_only", *rest)
        lo, hi = np.nanpercentile(draws[:, j], [2.5, 97.5])
        rows.append({"statistic": stat, "config": config, "target": target, "state": state,
                     "point": point[key], "ci_low": lo, "ci_high": hi,
                     "se": float(np.nanstd(draws[:, j], ddof=1)), "n_reps": n_reps,
                     "block": "local calendar day"})
    out = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_DIR / "v15_bootstrap_ci.csv", index=False)
    pd.set_option("display.width", 200)
    log("=== day-block bootstrap, 95% percentile intervals ===")
    print(out[out["state"] == "all"].drop(columns=["n_reps", "block"]).round(4).to_string(index=False))
    log(f"saved {OUTPUT_DIR / 'v15_bootstrap_ci.csv'}")


# %% [markdown]
# ## Reliability figures (report, light surface)
#
# Static small multiples: thresholds in rows, horizons in columns, a count strip under each
# reliability panel on its own axis (never a second y-scale). Colors are the first three
# categorical slots of the dataviz reference palette, which validate all-pairs for small
# multiples; aqua sits below 3:1 on the surface, so every series also has its own marker and
# the table view is v14_reliability.csv.

# %%
FIG_SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
FIG_SERIES = [("cv_oof", "Blocked CV (V13), same 2026 rows", "#2a78d6", "o"),
              ("forward_cal_random", "Forward, random-row calibration (deployed recipe)", "#eb6834", "s"),
              ("forward_cal_time", "Forward, time-held-out calibration", "#1baf7a", "^")]


def phase_figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rel = pd.read_csv(v14.OUTPUT_DIR / "v14_reliability.csv")
    rel = rel[(rel["config"] == "hw_allseason") & (rel["n"] > 0)]
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.edgecolor": AXIS,
                         "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": MUTED,
                         "text.color": INK, "axes.titlecolor": INK, "svg.fonttype": "none"})
    fig = plt.figure(figsize=(10, 8.2), facecolor=FIG_SURFACE)
    grid = fig.add_gridspec(5, 3, height_ratios=[3, 1, 0.5, 3, 1], hspace=0.12, wspace=0.18,
                            left=0.07, right=0.98, top=0.86, bottom=0.07)
    for r, thr in enumerate((0.1, 1.0)):
        for c, h in enumerate((1, 3, 6)):
            name = f"h{h}_{thr}mm"
            ax = fig.add_subplot(grid[3 * r, c], facecolor=FIG_SURFACE)
            strip = fig.add_subplot(grid[3 * r + 1, c], sharex=ax, facecolor=FIG_SURFACE)
            ax.plot([0, 1], [0, 1], color=AXIS, lw=1, ls=(0, (4, 3)), zorder=1)
            for variant, label, color, marker in FIG_SERIES:
                d = rel[(rel["target"] == name) & (rel["variant"] == variant)]
                ax.plot(d["mean_forecast"], d["observed_rate"], color=color, lw=1.5, marker=marker,
                        ms=5, mec=FIG_SURFACE, mew=1, label=label, zorder=3)
            d = rel[(rel["target"] == name) & (rel["variant"] == "forward_cal_time")]
            centers = d["bin"].str.split("-").str[0].astype(float) + 0.5 / v14.RELIABILITY_BINS
            strip.bar(centers, d["n"], width=0.9 / v14.RELIABILITY_BINS, color=MUTED,
                      edgecolor=FIG_SURFACE, lw=1)
            strip.set_yscale("log")
            strip.set_ylim(1e3, 2e6)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title(f"+{h} h, ≥{thr} mm", fontsize=10, loc="left", pad=4)
            for a in (ax, strip):
                a.grid(True, color=GRID, lw=0.6)
                a.set_axisbelow(True)
                for side in ("top", "right"):
                    a.spines[side].set_visible(False)
            ax.tick_params(labelbottom=False)
            if c == 0:
                ax.set_ylabel("Observed rain frequency")
                strip.set_ylabel("Forecasts")
            else:
                ax.tick_params(labelleft=False)
                strip.tick_params(labelleft=False)
            if r == 1:
                strip.set_xlabel("Forecast probability")
            else:
                strip.tick_params(labelbottom=False)
    fig.suptitle("Reliability of the IR model on Jan–Jul 2026 (997,934 cell-hours)",
                 x=0.07, ha="left", y=0.975, fontsize=12, color=INK)
    fig.text(0.07, 0.935, "Points on the dashed line are perfectly calibrated. Count strips: "
             "forecasts per 0.1 bin, log scale. Source: v14_reliability.csv",
             fontsize=8.5, color=INK_2)
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.065, 0.925), ncol=3,
               frameon=False, fontsize=8.5, handlelength=2.2, columnspacing=1.6)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(OUTPUT_DIR / f"v15_reliability_2026.{ext}", dpi=200, facecolor=FIG_SURFACE)
    plt.close(fig)
    log(f"saved {OUTPUT_DIR / 'v15_reliability_2026.png'} (+ .svg)")


# %%
if __name__ == "__main__":
    phase = sys.argv[1] if len(sys.argv) > 1 else "onset"
    log(f"phase: {phase}")
    t0 = time.time()
    if phase == "obs":
        phase_obs()
    elif phase == "onset":
        phase_onset()
    elif phase == "bootstrap":
        phase_bootstrap(int(sys.argv[2]) if len(sys.argv) > 2 else 1000)
    elif phase == "figures":
        phase_figures()
    else:
        raise SystemExit(f"unknown phase {phase!r}; use obs | onset | bootstrap | figures")
    log(f"phase {phase} done in {(time.time() - t0) / 60:.1f} min")
