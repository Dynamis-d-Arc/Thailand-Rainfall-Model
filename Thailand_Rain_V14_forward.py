# %% [markdown]
# # Thailand_Rain_V14 — forward-in-time check of the deployed recipe
#
# Every earlier Thailand score is blocked CV: a held-out 2-month block is predicted by a model
# trained on data both before and after it, and the isotonic calibrator is fitted on a random
# 12% of the training rows (`fit_full` / `BKK_Rain_V3.fit_predict_calibrated`), so the same
# storm can sit in both the fit rows and the calibration rows. This experiment replays
# deployment instead:
#
#   train      2024-07-01 .. 2025-06-29   booster with the exact V13 recipe (`fit_full`: 88% fit,
#                                         isotonic on a random 12% -> `cal_random`)
#   calibrate  2025-07-01 .. 2025-12-30   a second isotonic fitted on this period only
#                                         (`cal_time`); every F1 threshold is picked here
#   test       2026-01-01 .. 2026-07-19   untouched, scored once
#
# A 24 h gap before each boundary matches the CV purge. One booster feeds both calibrators,
# so `cal_random` vs `cal_time` isolates the calibration split. The V13 OOF on the same test
# rows (`cv_oof`) is the blocked-CV reference. Training here is 12 months against ~23 in CV,
# so a forward-vs-CV gap mixes "less data" with "no future data".
#
# Usage:
#     python Thailand_Rain_V14_forward.py fit      # 2 configs x 6 targets, resumable
#     python Thailand_Rain_V14_forward.py report
#
# ## Result (2026-10-30, labels_v2; v14_forward_scores.csv, v14_reliability.csv)
#
# - Blocked CV is not optimistic on ranking. On the same 997,934 test rows, forward ROC sits
#   0.000-0.003 below the V13 OOF for hw_allseason (wet May-Jul 0.002-0.004, dry -0.001-0.003),
#   with half the training data and nothing from the future.
# - The IR gain survives: hw - om ROC is +0.016..+0.037 forward vs +0.015..+0.036 in CV, and
#   within 0.002 per regime.
# - The random-row calibrator costs a little. Time-held-out isotonic: mean ECE 0.017 vs 0.026
#   (hw_allseason), Brier skill +0.002..+0.004. CV OOF Brier skill stays 0.000..0.006 ahead.
# - Both calibrators are overconfident at the top in 2026: h1_1.0mm forecasts near 0.85
#   verify at about 0.72, a year-shift effect that neither split removes.
# - One split, no confidence intervals yet.

# %%
import gc
import json
import sys
import time

import numpy as np
import pandas as pd

from BKK_Rain_V3 import MODEL_PLAN, PURGE_HOURS, RANDOM_STATE, TARGETS, TARGET_NAMES, log, pick_threshold
from Thailand_Rain_V3 import CACHE
import Thailand_Rain_V3 as v3
from Thailand_Rain_V10_deploy import fit_full, PROJECT_ROOT

OUTPUT_DIR = (PROJECT_ROOT / "ML_Model_V2" / "trained_models"
              / f"om_thailand_rain_v14_forward_{v3.LABEL_VERSION}")
# hw_block_offset0.npy is the block V13 CV and the live fit used (Docs/label_fix_2026-10-27.md)
CONFIGS = {"om_only": [], "hw_allseason": ["hw_block_offset0.npy"]}
CALIB_START = np.datetime64("2025-07-01")
TEST_START = np.datetime64("2026-01-01")
GAP = np.timedelta64(PURGE_HOURS, "h")
RELIABILITY_BINS = 10


def split_masks(tf):
    train = tf < CALIB_START - GAP
    calib = (tf >= CALIB_START) & (tf < TEST_START - GAP)
    test = tf >= TEST_START
    assert tf[calib].min() - tf[train].max() > GAP and tf[test].min() - tf[calib].max() > GAP
    return train, calib, test


def pred_path(config, i):
    return CACHE / f"v14_forward_{config}_{i}.npz"


# %%
def phase_fit():
    y = np.load(CACHE / "y.npy")
    tf = np.load(CACHE / "forecast_time.npy")
    train, calib, test = split_masks(tf)
    later = calib | test
    log(f"train {train.sum():,}  calib {calib.sum():,}  test {test.sum():,} rows")

    for config, blocks in CONFIGS.items():
        todo = [i for i in range(len(TARGETS)) if not pred_path(config, i).exists()]
        if not todo:
            log(f"{config}: all targets done, skipping")
            continue
        parts = [np.load(CACHE / "base_X.npy")] + [np.load(CACHE / b) for b in blocks]
        x = np.ascontiguousarray(np.concatenate(parts, axis=1)) if len(parts) > 1 else parts[0]
        del parts
        x_train, x_later = x[train], x[later]
        del x
        gc.collect()
        log(f"{config}: X_train {x_train.shape}")
        for i in todo:
            t0 = time.time()
            rng = np.random.default_rng([RANDOM_STATE, 140, i])
            model, iso_random = fit_full(x_train, y[train, i], MODEL_PLAN[TARGETS[i][0]], rng)
            raw = model.predict_proba(x_later)[:, 1].astype("float32")
            np.savez(pred_path(config, i), raw=raw, n_later=int(later.sum()),
                     cal_random=iso_random.predict(raw).astype("float32"))
            del model, iso_random
            gc.collect()
            log(f"  {config} {TARGET_NAMES[i]:10} {(time.time() - t0) / 60:.1f} min")
        del x_train, x_later
        gc.collect()


# %%
def score(yy, p, thr):
    from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                                 precision_score, recall_score, roc_auc_score)
    base = float(yy.mean())
    brier = float(brier_score_loss(yy, p))
    bins = np.minimum((p * RELIABILITY_BINS).astype(int), RELIABILITY_BINS - 1)
    n = np.bincount(bins, minlength=RELIABILITY_BINS)
    gap = np.abs(np.bincount(bins, p, RELIABILITY_BINS) - np.bincount(bins, yy, RELIABILITY_BINS))
    flags = (p >= thr).astype("int8")
    return {"rows": int(len(yy)), "base_rate": base,
            "roc_auc": float(roc_auc_score(yy, p)), "pr_auc": float(average_precision_score(yy, p)),
            "brier_score": brier, "brier_skill": 1 - brier / (base * (1 - base)),
            "ece": float(gap.sum() / n.sum()), "threshold": thr,
            "precision": float(precision_score(yy, flags, zero_division=0)),
            "recall": float(recall_score(yy, flags, zero_division=0)),
            "f1": float(f1_score(yy, flags, zero_division=0)), "flag_rate": float(flags.mean())}


def phase_report():
    from sklearn.isotonic import IsotonicRegression
    y = np.load(CACHE / "y.npy")
    tf = np.load(CACHE / "forecast_time.npy")
    train, calib, test = split_masks(tf)
    later = calib | test
    in_calib, in_test = calib[later], test[later]
    months = pd.DatetimeIndex(tf[test]).month.to_numpy()
    regimes = {"all": np.ones(len(months), bool), "dry Jan-Apr": months <= 4,
               "wet May-Jul": months >= 5}

    rows, rel = [], []
    for config in CONFIGS:
        oof = np.load(CACHE / f"oof_cal_{config}.npy")
        assert np.isfinite(oof[test]).all(), f"{config} OOF does not cover the test period"
        for i, name in enumerate(TARGET_NAMES):
            d = np.load(pred_path(config, i))
            assert int(d["n_later"]) == int(later.sum()), "split changed since fit; rerun fit"
            raw, cal_random = d["raw"], d["cal_random"]
            yc = y[calib, i]
            iso_time = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            cal_time = iso_time.fit(raw[in_calib], yc).predict(raw).astype("float32")
            ref = np.isfinite(oof[:, i]) & ~test      # CV threshold from OOF rows outside the test period
            variants = {
                "cv_oof": (oof[test, i], pick_threshold(oof[ref, i], y[ref, i])),
                "forward_raw": (raw[in_test], pick_threshold(raw[in_calib], yc)),
                "forward_cal_random": (cal_random[in_test], pick_threshold(cal_random[in_calib], yc)),
                "forward_cal_time": (cal_time[in_test], pick_threshold(cal_time[in_calib], yc)),
            }
            for vname, (p, thr) in variants.items():
                yt = y[test, i]
                for rname, rm in regimes.items():
                    rows.append({"config": config, "target": name, "variant": vname,
                                 "regime": rname, **score(yt[rm], p[rm], thr)})
                bins = np.minimum((p * RELIABILITY_BINS).astype(int), RELIABILITY_BINS - 1)
                for b in range(RELIABILITY_BINS):
                    m = bins == b
                    rel.append({"config": config, "target": name, "variant": vname,
                                "bin": f"{b / RELIABILITY_BINS:.1f}-{(b + 1) / RELIABILITY_BINS:.1f}",
                                "n": int(m.sum()),
                                "mean_forecast": float(p[m].mean()) if m.any() else np.nan,
                                "observed_rate": float(yt[m].mean()) if m.any() else np.nan})
            log(f"  scored {config} {name}")

    out = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_DIR / "v14_forward_scores.csv", index=False)
    pd.DataFrame(rel).to_csv(OUTPUT_DIR / "v14_reliability.csv", index=False)
    span = lambda m: f"{pd.Timestamp(tf[m].min()):%Y-%m-%d %H:%M} .. {pd.Timestamp(tf[m].max()):%Y-%m-%d %H:%M}"
    json.dump({"label_version": v3.LABEL_VERSION, "gap_hours": PURGE_HOURS,
               "train": span(train), "calibrate": span(calib), "test": span(test),
               "rows": {"train": int(train.sum()), "calibrate": int(calib.sum()), "test": int(test.sum())},
               "configs": CONFIGS, "booster_recipe": "Thailand_Rain_V10_deploy.fit_full (V13 params)"},
              open(OUTPUT_DIR / "v14_split.json", "w"), indent=2)

    a = out[out["regime"] == "all"].pivot_table(index=["config", "target"], columns="variant",
                                                values=["roc_auc", "brier_skill", "ece", "f1"])
    pd.set_option("display.width", 200)
    log("=== test period 2026-01..07, all rows ===")
    print(a.round(3).to_string())
    log(f"saved {OUTPUT_DIR}")


# %%
if __name__ == "__main__":
    phase = sys.argv[1] if len(sys.argv) > 1 else "report"
    log(f"phase: {phase}")
    t0 = time.time()
    if phase == "fit":
        phase_fit()
    elif phase == "report":
        phase_report()
    else:
        raise SystemExit(f"unknown phase {phase!r}; use fit | report")
    log(f"phase {phase} done in {(time.time() - t0) / 60:.1f} min")
