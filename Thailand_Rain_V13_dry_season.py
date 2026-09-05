# %% [markdown]
# # Thailand_Rain_V13 — closing the dry-season training gap
#
# The deployed V10 models were trained only on the 7 IR-covered wet-season blocks; the
# dry months (Nov-Apr) had no Himawari history, so a dry-season serve would feed the
# models IR values from months the training never saw. After the dry-season B13 fetch
# (362 missing days) and a rebuilt hw_block, this experiment:
#
#  - `cv`:     re-runs the V7 protocol over ALL 13 blocks under a fresh config name
#              (`hw_allseason`). The old hw_offset0h checkpoints are NOT reused: their
#              training rows contained all-NaN IR for dry blocks, which the rebuilt
#              block changes. om_only needs no refit — its OOF covers all 13 folds.
#  - `report`: IR gain vs om_only on identical rows, per block and pooled dry vs wet.
#  - `deploy`: V10-style full-panel fit on all 13 blocks with thresholds from the
#              hw_allseason OOF, written to this experiment's own deploy/ directory.
#              Swapping into the live deploy dir is a deliberate manual step after
#              the report is reviewed.
#
# Usage:
#     python Thailand_Rain_V13_dry_season.py cv
#     python Thailand_Rain_V13_dry_season.py report
#     python Thailand_Rain_V13_dry_season.py deploy

# %%
import gc
import json
import sys
import time

import joblib
import numpy as np
import pandas as pd

from BKK_Rain_V3 import MODEL_PLAN, RANDOM_STATE, TARGETS, TARGET_NAMES, log, pick_threshold
from Thailand_Rain_V3 import CACHE
import Thailand_Rain_V3 as v3
from Thailand_Rain_V7_himawari_ir import himawari_feature_names
from Thailand_Rain_V10_deploy import ALL_FEATURES, fit_full, PROJECT_ROOT

OUTPUT_DIR = PROJECT_ROOT / "ML_Model_V2" / "trained_models" / "om_thailand_rain_v13_allseason"
DEPLOY_DIR = OUTPUT_DIR / "deploy"
CONFIG = "hw_allseason"
DRY_MONTHS = {11, 12, 1, 2, 3, 4}
MIN_COVERAGE = 0.90


def load_panel():
    y = np.load(CACHE / "y.npy")
    tf = np.load(CACHE / "forecast_time.npy")
    fold_key = v3.block_keys(tf)
    return y, tf, fold_key


def fold_is_dry(fold, fold_key, tf):
    months = pd.DatetimeIndex(tf[fold_key == fold]).month
    return float(np.isin(months, list(DRY_MONTHS)).mean()) >= 0.5


# %%
def phase_cv():
    v3.CONFIG_BLOCKS[CONFIG] = [("hw_block_offset0.npy", himawari_feature_names)]
    y, tf, fold_key = load_panel()
    folds = [int(v) for v in np.unique(fold_key)
             if (fold_key == v).sum() >= v3.MIN_FOLD_ROWS]

    block = np.load(CACHE / "hw_block_offset0.npy", mmap_mode="r")
    for fold in folds:
        cov = float(np.asarray(block[fold_key == fold, -1] > 0).mean())
        tag = "dry" if fold_is_dry(fold, fold_key, tf) else "wet"
        log(f"  block {fold} ({v3.block_label(fold, tf)}, {tag}): IR coverage {cov:.1%}")
        assert cov >= MIN_COVERAGE, (
            f"block {fold} coverage {cov:.1%} < {MIN_COVERAGE:.0%} — "
            f"run the dry-season fetch and V7 build_features first")
    del block
    gc.collect()

    log(f"all-season CV: {len(folds)} folds, config {CONFIG}")
    results, fit_log = v3.run_cv(CONFIG, y, tf, fold_key, folds)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(OUTPUT_DIR / "v13_cv_results.csv", index=False)
    pd.DataFrame(fit_log).to_csv(OUTPUT_DIR / "v13_fit_log.csv", index=False)
    log("cv complete")


# %%
def phase_report():
    from sklearn.metrics import average_precision_score, roc_auc_score
    y, tf, fold_key = load_panel()
    hw = np.load(CACHE / f"oof_cal_{CONFIG}.npy")
    om = np.load(CACHE / "oof_cal_om_only.npy")

    folds = sorted(int(v) for v in np.unique(fold_key)
                   if np.isfinite(hw[fold_key == v]).any())
    rows = []

    def score(mask, span, regime):
        for i, name in enumerate(TARGET_NAMES):
            m = mask & np.isfinite(hw[:, i]) & np.isfinite(om[:, i])
            if m.sum() < 5000 or len(np.unique(y[m, i])) < 2:
                continue
            yy = y[m, i]
            rows.append({
                "span": span, "regime": regime, "target": name, "rows": int(m.sum()),
                "base_rate": round(float(yy.mean()), 4),
                "om_roc": round(float(roc_auc_score(yy, om[m, i])), 4),
                "hw_roc": round(float(roc_auc_score(yy, hw[m, i])), 4),
                "om_pr": round(float(average_precision_score(yy, om[m, i])), 4),
                "hw_pr": round(float(average_precision_score(yy, hw[m, i])), 4),
            })

    dry_mask = np.zeros(len(y), dtype=bool)
    wet_mask = np.zeros(len(y), dtype=bool)
    for fold in folds:
        fm = fold_key == fold
        dry = fold_is_dry(fold, fold_key, tf)
        (dry_mask if dry else wet_mask)[fm] = True
        score(fm, v3.block_label(fold, tf), "dry" if dry else "wet")
    score(dry_mask, "ALL DRY", "dry")
    score(wet_mask, "ALL WET", "wet")
    score(dry_mask | wet_mask, "ALL", "all")

    out = pd.DataFrame(rows)
    out["roc_gain"] = (out["hw_roc"] - out["om_roc"]).round(4)
    out["pr_gain"] = (out["hw_pr"] - out["om_pr"]).round(4)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_DIR / "v13_dry_wet_report.csv", index=False)
    show = out[out["span"].isin(["ALL DRY", "ALL WET", "ALL"])]
    log("=== pooled IR gain vs om_only (identical rows) ===")
    print(show[["span", "target", "rows", "base_rate", "om_roc", "hw_roc",
                "roc_gain", "pr_gain"]].to_string(index=False))
    log(f"saved {OUTPUT_DIR / 'v13_dry_wet_report.csv'}")


# %%
def phase_deploy():
    import lightgbm
    import sklearn
    y, tf, fold_key = load_panel()
    base_x = np.load(CACHE / "base_X.npy")
    hw = np.load(CACHE / "hw_block_offset0.npy")
    x = np.ascontiguousarray(np.concatenate([base_x, hw], axis=1))
    del base_x, hw
    gc.collect()

    oof = np.load(CACHE / f"oof_cal_{CONFIG}.npy")
    log(f"all-season deploy fit: {len(y):,} rows x {x.shape[1]} features")

    DEPLOY_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    for i, (horizon, threshold_mm) in enumerate(TARGETS):
        tname = TARGET_NAMES[i]
        out_path = DEPLOY_DIR / f"th_rain_{tname}.joblib"
        if out_path.exists():
            log(f"  {tname}: artifact exists, skipping (delete to refit)")
            continue
        t0 = time.time()
        m_oof = np.isfinite(oof[:, i])
        op_threshold = pick_threshold(oof[m_oof, i], y[m_oof, i])
        rng = np.random.default_rng([RANDOM_STATE, 130, i])
        model, calibrator = fit_full(x, y[:, i], MODEL_PLAN[horizon], rng)
        joblib.dump({
            "model": model, "calibrator": calibrator, "threshold": op_threshold,
            "feature_names": ALL_FEATURES, "target": tname, "horizon_h": horizon,
            "rain_threshold_mm": threshold_mm, "trained_rows": int(len(y)),
            "notes": "Thailand V13 all-season fit (dry-season IR included); "
                     "see v13_manifest.json",
        }, out_path, compress=3)
        mins = (time.time() - t0) / 60
        summary.append({"target": tname, "model": MODEL_PLAN[horizon],
                        "op_threshold": op_threshold, "trained_rows": int(len(y)),
                        "fit_minutes": round(mins, 2)})
        pd.DataFrame(summary).to_csv(OUTPUT_DIR / "v13_fit_summary.csv", index=False)
        log(f"  {tname}: threshold {op_threshold:.2f}, fit {mins:.1f} min")

    manifest = {
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": "hw_allseason: V10 feature set, trained on ALL 13 blocks after the "
                  "dry-season Himawari backfill (previously 7 wet-season blocks only)",
        "supersedes": "om_thailand_rain_v10_deploy/deploy (wet-season-only training)",
        "thresholds_from": f"out-of-fold {CONFIG} probabilities, all folds",
        "library_versions": {"sklearn": sklearn.__version__,
                             "lightgbm": lightgbm.__version__},
    }
    json.dump(manifest, open(OUTPUT_DIR / "v13_manifest.json", "w"), indent=2)
    log("deploy artifacts written — review v13_dry_wet_report.csv, then copy "
        "deploy/th_rain_*.joblib over om_thailand_rain_v10_deploy/deploy/ to go live")


# %%
if __name__ == "__main__":
    phase = sys.argv[1] if len(sys.argv) > 1 else "report"
    log(f"phase: {phase}")
    t0 = time.time()
    if phase == "cv":
        phase_cv()
    elif phase == "report":
        phase_report()
    elif phase == "deploy":
        phase_deploy()
    else:
        raise SystemExit(f"unknown phase {phase!r}; use cv | report | deploy")
    log(f"phase {phase} done in {(time.time() - t0) / 60:.1f} min")
