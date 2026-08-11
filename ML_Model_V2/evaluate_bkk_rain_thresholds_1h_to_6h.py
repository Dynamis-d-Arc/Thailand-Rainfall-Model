import argparse
import importlib
import json
import os
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psycopg2
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "postgres",
    "user": "postgres",
    "password": os.getenv("PGPASSWORD", "Pass1234"),
}

FEATURE_TABLE = "bkk_weather_lag_features_v3"
MODEL_DIR = Path(__file__).with_name("trained_models")
OUTPUT_CSV = MODEL_DIR / "rain_threshold_results_1h_to_6h.csv"
HORIZONS = [1, 2, 3, 4, 5, 6]
TARGET_COLUMNS = [f"rain_next_{horizon}h" for horizon in HORIZONS]
DEFAULT_THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


def install_sklearn_pickle_compatibility():
    """Allow scikit-learn 1.8 HistGradientBoosting pickles to load on 1.9."""
    try:
        loss_module = importlib.import_module("sklearn._loss")
    except ImportError:
        return

    if not hasattr(loss_module, "CyHalfBinomialLoss") and hasattr(
        loss_module,
        "HalfBinomialLoss",
    ):
        setattr(loss_module, "CyHalfBinomialLoss", loss_module.HalfBinomialLoss)

    import sys

    sys.modules.setdefault("_loss", loss_module)


def log(message):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def connect():
    return psycopg2.connect(**DB_CONFIG)


def metadata_path(model_dir, horizon):
    return model_dir / f"bkk_rain_next_{horizon}h_v3_hist_gradient_boosting_metadata.json"


def load_metadata(model_dir, horizon):
    path = metadata_path(model_dir, horizon)
    if not path.exists():
        raise FileNotFoundError(f"Missing metadata file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_model(metadata):
    path = Path(metadata["model_path"])
    if not path.exists():
        raise FileNotFoundError(f"Missing model file: {path}")
    install_sklearn_pickle_compatibility()
    return joblib.load(path)


def feature_not_null_sql(feature_columns):
    return " AND ".join(f"{column} IS NOT NULL" for column in feature_columns)


def fetch_horizon_data(conn, table, feature_columns, target_column, sample_rows):
    columns = ["grid_number", "forecast_time", *feature_columns, target_column]
    query = f"""
    SELECT {", ".join(columns)}
    FROM {table}
    WHERE {target_column} IS NOT NULL
      AND {feature_not_null_sql(feature_columns)}
    ORDER BY forecast_time, grid_number
    """
    if sample_rows:
        query = f"""
        SELECT *
        FROM ({query}) rows_for_evaluation
        ORDER BY random()
        LIMIT {int(sample_rows)}
        """
    return pd.read_sql_query(query, conn, parse_dates=["forecast_time"])


def fetch_all_horizon_data(conn, table, feature_columns, target_columns, sample_rows):
    columns = ["grid_number", "forecast_time", *feature_columns, *target_columns]
    query = f"""
    SELECT {", ".join(columns)}
    FROM {table}
    WHERE {feature_not_null_sql(feature_columns)}
    ORDER BY forecast_time, grid_number
    """
    if sample_rows:
        query = f"""
        SELECT *
        FROM ({query}) rows_for_evaluation
        ORDER BY random()
        LIMIT {int(sample_rows)}
        """
    return pd.read_sql_query(query, conn, parse_dates=["forecast_time"])


def add_time_split(df, train_fraction, validation_fraction):
    unique_times = np.array(sorted(df["forecast_time"].unique()))
    if len(unique_times) < 3:
        raise RuntimeError("Not enough forecast times to split train/validation/test.")

    train_end = unique_times[max(1, int(len(unique_times) * train_fraction))]
    validation_end = unique_times[
        max(2, int(len(unique_times) * (train_fraction + validation_fraction)))
    ]

    df = df.copy()
    df["split"] = "test"
    df.loc[df["forecast_time"] < train_end, "split"] = "train"
    df.loc[
        (df["forecast_time"] >= train_end) & (df["forecast_time"] < validation_end),
        "split",
    ] = "validation"
    return df


def add_time_split_from_boundaries(df, train_end, validation_end):
    df = df.copy()
    df["split"] = "test"
    df.loc[df["forecast_time"] < train_end, "split"] = "train"
    df.loc[
        (df["forecast_time"] >= train_end) & (df["forecast_time"] < validation_end),
        "split",
    ] = "validation"
    return df


def safe_rate(numerator, denominator):
    if denominator == 0:
        return np.nan
    return numerator / denominator


def threshold_metrics(y_true, probabilities, threshold):
    y_pred = (probabilities >= threshold).astype("int8")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "predicted_rain_rate": float(y_pred.mean()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float(safe_rate(tn, tn + fp)),
        "false_positive_rate": float(safe_rate(fp, fp + tn)),
        "false_negative_rate": float(safe_rate(fn, fn + tp)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def evaluate_horizon(conn, args, horizon, thresholds):
    log(f"Loading metadata/model for {horizon}h...")
    metadata = load_metadata(args.model_dir, horizon)
    model = load_model(metadata)
    feature_columns = metadata["feature_columns"]
    target_column = metadata["target_column"]

    log(f"Fetching evaluation rows for {target_column}...")
    df = fetch_horizon_data(
        conn,
        args.feature_table,
        feature_columns,
        target_column,
        args.sample_rows,
    )
    if df.empty:
        raise RuntimeError(f"No rows found for {target_column}.")

    log(f"Rebuilding chronological split for {target_column}...")
    df = add_time_split(df, args.train_fraction, args.validation_fraction)
    test_df = df[df["split"] == "test"]
    x_test = test_df[feature_columns].astype("float32")
    y_test = test_df[target_column].astype("int8").to_numpy()

    log(f"Predicting probabilities for {target_column} on {len(test_df):,} test rows...")
    probabilities = model.predict_proba(x_test)[:, 1]

    base_metrics = {
        "horizon_h": horizon,
        "target_column": target_column,
        "rows": int(len(y_test)),
        "rain_rate": float(y_test.mean()),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
        "brier_score": float(brier_score_loss(y_test, probabilities)),
    }

    rows = []
    for threshold in thresholds:
        rows.append({**base_metrics, **threshold_metrics(y_test, probabilities, threshold)})
    return rows


def evaluate_horizon_from_df(df, args, horizon, thresholds, split_boundaries):
    log(f"Loading metadata/model for {horizon}h...")
    metadata = load_metadata(args.model_dir, horizon)
    model = load_model(metadata)
    feature_columns = metadata["feature_columns"]
    target_column = metadata["target_column"]
    train_end, validation_end = split_boundaries

    log(f"Preparing test rows for {target_column}...")
    horizon_df = df[df[target_column].notna()].copy()
    horizon_df = add_time_split_from_boundaries(horizon_df, train_end, validation_end)
    test_df = horizon_df[horizon_df["split"] == "test"]
    x_test = test_df[feature_columns].astype("float32")
    y_test = test_df[target_column].astype("int8").to_numpy()

    log(f"Predicting probabilities for {target_column} on {len(test_df):,} test rows...")
    probabilities = model.predict_proba(x_test)[:, 1]

    base_metrics = {
        "horizon_h": horizon,
        "target_column": target_column,
        "rows": int(len(y_test)),
        "rain_rate": float(y_test.mean()),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
        "brier_score": float(brier_score_loss(y_test, probabilities)),
    }

    rows = []
    for threshold in thresholds:
        rows.append({**base_metrics, **threshold_metrics(y_test, probabilities, threshold)})
    return rows


def parse_thresholds(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate probability thresholds for Bangkok rain models, 1h to 6h."
    )
    parser.add_argument("--feature-table", default=FEATURE_TABLE)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV)
    parser.add_argument(
        "--thresholds",
        type=parse_thresholds,
        default=DEFAULT_THRESHOLDS,
        help="Comma-separated thresholds, e.g. 0.1,0.2,0.3,0.4,0.5",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help="Optional random sample size per horizon for quick checks. Defaults to full data.",
    )
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    return parser.parse_args()


def main():
    args = parse_args()
    all_rows = []

    log("Starting threshold evaluation for 1h-6h models...")
    with connect() as conn:
        first_metadata = load_metadata(args.model_dir, HORIZONS[0])
        feature_columns = first_metadata["feature_columns"]
        log("Fetching shared evaluation rows for all 1h-6h targets...")
        df = fetch_all_horizon_data(
            conn,
            args.feature_table,
            feature_columns,
            TARGET_COLUMNS,
            args.sample_rows,
        )
        if df.empty:
            raise RuntimeError(f"No complete feature rows found in {args.feature_table}.")

        unique_times = np.array(sorted(df["forecast_time"].unique()))
        train_end = unique_times[max(1, int(len(unique_times) * args.train_fraction))]
        validation_end = unique_times[
            max(2, int(len(unique_times) * (args.train_fraction + args.validation_fraction)))
        ]
        split_boundaries = (train_end, validation_end)
        log(f"Loaded {len(df):,} rows for shared evaluation.")

        for horizon in HORIZONS:
            all_rows.extend(
                evaluate_horizon_from_df(df, args, horizon, args.thresholds, split_boundaries)
            )

    results = pd.DataFrame(all_rows).sort_values(["horizon_h", "threshold"])
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_csv, index=False)

    log(f"Saved threshold results to {args.output_csv}")
    print(
        results[
            [
                "horizon_h",
                "threshold",
                "rain_rate",
                "predicted_rain_rate",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "pr_auc",
                "brier_score",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
