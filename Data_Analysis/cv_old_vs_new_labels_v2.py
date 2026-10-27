"""Matched-sample comparison of the pre-fix and labels_v2 cross-validation results.

Joins the old fold cache (ML_Model_V2/_th_cache: window [t+1, t+1+h) on precipitation_max_mm)
with the new one (_th_cache_labels_v2: window [t, t+h) on precipitation_pixelmax_mm) on
(grid cell, issue hour) and scores, on exactly the same rows:

  old_model_old_labels   what the report's Tables 21/24 were built from
  old_model_new_labels   the old out-of-fold probabilities graded against the corrected labels
                         (isolates the label change from the retraining)
  new_model_new_labels   the retrained out-of-fold probabilities against the corrected labels

for the om_only control and the hw_allseason (IR) config, ROC-AUC, PR-AUC and Brier per
target, pooled and per 2-month block. Writes
ML_Model_V2/trained_models/om_thailand_rain_v13_allseason_labels_v2/cv_old_vs_new_labels_v2.csv.

    python Data_Analysis/cv_old_vs_new_labels_v2.py [--old-cache PATH]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
NEW_CACHE = ROOT / "ML_Model_V2" / "_th_cache_labels_v2"
OUT = ROOT / "ML_Model_V2" / "trained_models" / "om_thailand_rain_v13_allseason_labels_v2"
TARGETS = ["h1_0.1mm", "h3_0.1mm", "h6_0.1mm", "h1_1.0mm", "h3_1.0mm", "h6_1.0mm"]
CONFIGS = {"om_only": "oof_cal_om_only.npy", "hw_allseason": "oof_cal_hw_allseason.npy"}
BLOCK_MONTHS = 2


def load(cache):
    d = {"y": np.load(cache / "y.npy"), "t": np.load(cache / "forecast_time.npy"),
         "unit": np.load(cache / "unit_id.npy")}
    for cfg, fname in CONFIGS.items():
        d[cfg] = np.load(cache / fname) if (cache / fname).exists() else None
    return d


def block_of(t):
    s = pd.to_datetime(t)
    months = s.year.to_numpy() * 12 + (s.month.to_numpy() - 1)
    return (months - months.min()) // BLOCK_MONTHS, s


def metrics(y, p):
    if len(np.unique(y)) < 2:
        return {"roc_auc": np.nan, "pr_auc": np.nan, "brier": np.nan}
    return {"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "brier": brier_score_loss(y, np.clip(p, 0, 1))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-cache", default=str(ROOT.parents[2] / "ML_Model_V2" / "_th_cache")
                    if ROOT.name == "codex-label-fix" else str(ROOT / "ML_Model_V2" / "_th_cache"))
    args = ap.parse_args()
    old, new = load(Path(args.old_cache)), load(NEW_CACHE)

    key_old = pd.MultiIndex.from_arrays([old["unit"], old["t"]])
    key_new = pd.MultiIndex.from_arrays([new["unit"], new["t"]])
    common = key_old.intersection(key_new)
    io = key_old.get_indexer(common)
    inew = key_new.get_indexer(common)
    print(f"old rows {len(key_old):,}, new rows {len(key_new):,}, matched {len(common):,}")
    blocks, stamps = block_of(new["t"][inew])
    spans = {b: f"{stamps[blocks == b].min():%Y-%m}..{stamps[blocks == b].max():%Y-%m}"
             for b in np.unique(blocks)}

    rows = []
    for cfg in CONFIGS:
        if old[cfg] is None or new[cfg] is None:
            print(f"{cfg}: missing OOF array, skipped")
            continue
        po, pn = old[cfg][io], new[cfg][inew]
        for i, name in enumerate(TARGETS):
            yo, yn = old["y"][io, i], new["y"][inew, i]
            ok = np.isfinite(po[:, i]) & np.isfinite(pn[:, i])
            for span, m in [("ALL", ok)] + [(spans[b], ok & (blocks == b)) for b in spans]:
                if m.sum() < 5000:
                    continue
                base = {"config": cfg, "target": name, "span": span, "rows": int(m.sum()),
                        "old_base_rate": float(yo[m].mean()), "new_base_rate": float(yn[m].mean()),
                        "labels_changed_pct": float(100 * (yo[m] != yn[m]).mean())}
                for tag, y, p in [("old_model_old_labels", yo[m], po[m, i]),
                                  ("old_model_new_labels", yn[m], po[m, i]),
                                  ("new_model_new_labels", yn[m], pn[m, i])]:
                    rows.append({**base, "scoring": tag, **metrics(y, p)})
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "cv_old_vs_new_labels_v2.csv", index=False, float_format="%.4f")
    pooled = out[out.span == "ALL"].pivot_table(index=["config", "target"], columns="scoring",
                                                 values="roc_auc").round(4)
    print(pooled.to_string())
    print(f"saved {OUT / 'cv_old_vs_new_labels_v2.csv'}")


if __name__ == "__main__":
    main()
