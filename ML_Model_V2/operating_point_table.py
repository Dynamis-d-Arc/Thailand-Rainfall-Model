"""What each fixed probability threshold actually buys, per target and per season.

Three fitted schemes were tried before this (`fit_seasonal_thresholds.py`) and each degenerated
somewhere: F1-optimal cuts collapse toward always-yes where rain is common, TSS-optimal cuts
over-fire where rain is rare, and a precision floor stops binding once the base rate exceeds it.
All three were optimising a *statistic*, which is the wrong object.

The probabilities are already isotonically calibrated, and `validate_by_season.py` shows they
track the seasonal base rate (wet 2024 mean 0.576 against 0.592 observed; dry 2026 mean 0.067
against 0.048). When that holds, the season is carried by the probability itself - a calibrated
0.30 means 30% in January and in July alike - so a *fixed* threshold is already
season-appropriate, and the only real question is where on the precision/recall curve to sit.

For a calibrated forecast that choice is a cost ratio, not a fit: firing at p >= t is optimal
when a miss costs t/(1-t) times a false alarm. t = 0.5 treats them equally; t = 0.25 says a miss
hurts three times as much.

This script prints that curve so the operating point can be chosen deliberately.

Usage:
    python ML_Model_V2/operating_point_table.py
    python ML_Model_V2/operating_point_table.py --target h6_0.1mm
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "ML_Model_V2" / "trained_models" / "om_bkk_rain_v2_deploy"
OOF_PATH = MODEL_DIR / "oof_probabilities.npz"

TARGETS = [(h, thr) for thr in [0.1, 1.0] for h in [1, 3, 6]]
TARGET_NAMES = [f"h{h}_{thr}mm" for h, thr in TARGETS]
WET_MONTHS = frozenset(range(5, 11))
GRID = np.round(np.arange(0.05, 0.91, 0.05), 2)


def find_bundle(horizon, rain_mm):
    return next(p for p in MODEL_DIR.glob(f"*_next_{horizon}h_*")
                if p.name.endswith(f"rain_threshold_{rain_mm}mm.joblib"))


def curve(probabilities, y_true):
    """Precision, recall, F1 and fire rate at every threshold on the grid."""
    out = []
    positives = int(y_true.sum())
    for t in GRID:
        fired = probabilities >= t
        tp = int(np.count_nonzero(fired & (y_true == 1)))
        n_fired = int(np.count_nonzero(fired))
        precision = tp / n_fired if n_fired else np.nan
        recall = tp / positives if positives else np.nan
        f1 = (2 * precision * recall / (precision + recall)
              if n_fired and positives and (precision + recall) > 0 else 0.0)
        out.append({"threshold": float(t), "precision": precision, "recall": recall,
                    "f1": f1, "fired_share": n_fired / len(y_true)})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default=None)
    parser.add_argument("--set", type=float, default=None, dest="set_threshold",
                        help="write this fixed threshold into the bundle(s) and drop any "
                             "per-month cuts")
    args = parser.parse_args()

    store = np.load(OOF_PATH, allow_pickle=False)
    oof_raw, y_all = store["oof_raw"], store["y_all"]
    stamps = pd.to_datetime(store["forecast_time"].astype("datetime64[s]"))
    month_key = stamps.month.to_numpy()
    wet = np.isin(month_key, list(WET_MONTHS))

    names = [args.target] if args.target else TARGET_NAMES
    rows = []
    for name in names:
        i = TARGET_NAMES.index(name)
        horizon, rain_mm = TARGETS[i]
        path = find_bundle(horizon, rain_mm)
        bundle = joblib.load(path)
        calibrated = bundle["calibrator"].predict(oof_raw[:, i]).astype("float32")
        y_true = y_all[:, i]

        if args.set_threshold is not None:
            t = float(args.set_threshold)
            achieved = {}
            for season, mask in (("all", np.ones(len(y_true), bool)),
                                 ("wet", wet), ("dry", ~wet)):
                fired = calibrated[mask] >= t
                truth = y_true[mask] == 1
                tp = int(np.count_nonzero(fired & truth))
                n_fired = int(np.count_nonzero(fired))
                positives = int(np.count_nonzero(truth))
                achieved[season] = {
                    "precision": tp / n_fired if n_fired else None,
                    "recall": tp / positives if positives else None,
                    "fired_share": float(fired.mean()),
                }
            bundle.pop("seasonal_thresholds", None)
            bundle["probability_threshold"] = t
            bundle["threshold_mode"] = "fixed"
            bundle["threshold_selection"] = (
                f"fixed at {t}, chosen as a cost ratio rather than fitted: for a calibrated "
                f"forecast, firing at p >= {t} is optimal when a miss costs {t / (1 - t):.2f}x a "
                "false alarm. Calibration already carries the seasonal base rate, so a fixed cut "
                "holds precision across seasons where an F1-fitted one did not.")
            bundle["metrics_by_season_at_threshold"] = achieved
            for key in ("precision", "recall"):
                if achieved["all"][key] is not None:
                    bundle["metrics"][key] = achieved["all"][key]
            joblib.dump(bundle, path, compress=3)
            print(f"{name:12} set to {t}: "
                  f"wet precision {achieved['wet']['precision']:.3f} "
                  f"recall {achieved['wet']['recall']:.3f} | "
                  f"dry precision {achieved['dry']['precision']:.3f} "
                  f"recall {achieved['dry']['recall']:.3f}")
        del bundle

        for season, mask in (("all", np.ones(len(y_true), bool)), ("wet", wet), ("dry", ~wet)):
            base = float(y_true[mask].mean())
            for entry in curve(calibrated[mask], y_true[mask]):
                entry.update(target=name, season=season, base_rate=base)
                rows.append(entry)

        print(f"\n=== {name}  (wet base {y_true[wet].mean():.3f}, "
              f"dry base {y_true[~wet].mean():.3f}) ===")
        print(f"{'thr':>5} | {'wet prec':>8} {'wet rec':>7} {'wet fire':>8} | "
              f"{'dry prec':>8} {'dry rec':>7} {'dry fire':>8}")
        wet_curve = {e["threshold"]: e for e in curve(calibrated[wet], y_true[wet])}
        dry_curve = {e["threshold"]: e for e in curve(calibrated[~wet], y_true[~wet])}
        for t in GRID:
            w, d = wet_curve[float(t)], dry_curve[float(t)]
            print(f"{t:5.2f} | {w['precision']:8.3f} {w['recall']:7.3f} {w['fired_share']:8.1%} | "
                  f"{d['precision']:8.3f} {d['recall']:7.3f} {d['fired_share']:8.1%}")
        del calibrated

    frame = pd.DataFrame(rows)[["target", "season", "base_rate", "threshold",
                                "precision", "recall", "f1", "fired_share"]]
    out = MODEL_DIR / "operating_point_curves.csv"
    frame.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
