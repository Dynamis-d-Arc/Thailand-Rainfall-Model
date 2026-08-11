import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psycopg2
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "postgres",
    "user": "postgres",
    "password": os.getenv("PGPASSWORD", "Pass1234"),
}

FEATURE_TABLE = "bkk_weather_lag_features_v3"
TARGET_COLUMN = "rain_next_2h"
MODEL_DIR = Path(__file__).with_name("trained_models")
MODEL_NAME = "bkk_rain_next_2h_v3_hist_gradient_boosting"

CURRENT_FEATURE_COLUMNS = [
    "temperature_2m",
    "dew_point_2m",
    "relative_humidity_2m",
    "precipitation",
    "cloud_cover",
    "surface_pressure",
    "wind_direction_10m",
    "wind_speed_10m",
    "pressure_msl",
]
LAG_SOURCE_COLUMNS = [
    "cloud_cover",
    "relative_humidity_2m",
    "dew_point_2m",
]
LAG_HOURS = [1, 2, 3, 4, 5, 6, 24, 48]
LAG_FEATURE_COLUMNS = [
    f"{column}_lag_{hour}h"
    for column in LAG_SOURCE_COLUMNS
    for hour in LAG_HOURS
]
FEATURE_COLUMNS = CURRENT_FEATURE_COLUMNS + LAG_FEATURE_COLUMNS


def log(message):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def connect():
    return psycopg2.connect(**DB_CONFIG)


def feature_not_null_sql():
    return " AND ".join(f"{column} IS NOT NULL" for column in FEATURE_COLUMNS)


def fetch_training_data(conn, table, sample_rows):
    columns = ["grid_number", "forecast_time", *FEATURE_COLUMNS, TARGET_COLUMN]
    query = f"""
    SELECT {", ".join(columns)}
    FROM {table}
    WHERE {TARGET_COLUMN} IS NOT NULL
      AND {feature_not_null_sql()}
    ORDER BY forecast_time, grid_number
    """
    if sample_rows:
        query = f"""
        SELECT *
        FROM ({query}) rows_for_training
        ORDER BY random()
        LIMIT {int(sample_rows)}
        """
    return pd.read_sql_query(query, conn, parse_dates=["forecast_time"])


def add_time_split(df, train_fraction, validation_fraction):
    log("Building chronological train/validation/test split...")
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
    return df, train_end, validation_end


def build_model():
    return HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=250,
        max_leaf_nodes=31,
        l2_regularization=0.05,
        random_state=42,
    )


def evaluate(model, x, y, split_name):
    log(f"Evaluating {split_name} split on {len(y):,} rows...")
    probabilities = model.predict_proba(x)[:, 1]
    predictions = (probabilities >= 0.5).astype("int8")
    metrics = {
        "split": split_name,
        "rows": int(len(y)),
        "rain_rate": float(y.mean()),
        "predicted_rain_rate_at_0_5": float(predictions.mean()),
        "brier_score": float(brier_score_loss(y, probabilities)),
    }
    if len(set(y)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y, probabilities))
        metrics["pr_auc"] = float(average_precision_score(y, probabilities))
    return metrics


def model_path(model_dir):
    return model_dir / f"{MODEL_NAME}.joblib"


def metadata_path(model_dir):
    return model_dir / f"{MODEL_NAME}_metadata.json"


def train(args):
    log(f"Connecting to PostgreSQL and loading rows from {args.feature_table}...")
    log(f"Target column: {TARGET_COLUMN}")
    with connect() as conn:
        df = fetch_training_data(conn, args.feature_table, args.sample_rows)

    if df.empty:
        raise RuntimeError(f"No training rows found in {args.feature_table}.")

    log(f"Loaded {len(df):,} complete rows.")
    df, train_end, validation_end = add_time_split(
        df,
        args.train_fraction,
        args.validation_fraction,
    )

    train_df = df[df["split"] == "train"]
    validation_df = df[df["split"] == "validation"]
    test_df = df[df["split"] == "test"]
    log(
        "Split sizes: "
        f"train={len(train_df):,}, "
        f"validation={len(validation_df):,}, "
        f"test={len(test_df):,}"
    )

    log("Preparing float32 feature matrix and int8 target...")
    x_train = train_df[FEATURE_COLUMNS].astype("float32")
    y_train = train_df[TARGET_COLUMN].astype("int8")

    model = build_model()
    log(f"Training {MODEL_NAME} on {len(x_train):,} rows...")
    model.fit(x_train, y_train)
    log("Training complete. Starting evaluation...")

    metrics = {
        "train": evaluate(
            model,
            train_df[FEATURE_COLUMNS].astype("float32"),
            train_df[TARGET_COLUMN].astype("int8"),
            "train",
        ),
        "validation": evaluate(
            model,
            validation_df[FEATURE_COLUMNS].astype("float32"),
            validation_df[TARGET_COLUMN].astype("int8"),
            "validation",
        ),
        "test": evaluate(
            model,
            test_df[FEATURE_COLUMNS].astype("float32"),
            test_df[TARGET_COLUMN].astype("int8"),
            "test",
        ),
    }

    args.model_dir.mkdir(parents=True, exist_ok=True)
    output_model_path = model_path(args.model_dir)
    output_metadata_path = metadata_path(args.model_dir)

    log(f"Saving model to {output_model_path}...")
    joblib.dump(model, output_model_path)

    metadata = {
        "model_name": MODEL_NAME,
        "model_path": str(output_model_path),
        "feature_table": args.feature_table,
        "target_column": TARGET_COLUMN,
        "rain_threshold_mm": 0.1,
        "feature_columns": FEATURE_COLUMNS,
        "current_feature_columns": CURRENT_FEATURE_COLUMNS,
        "lag_source_columns": LAG_SOURCE_COLUMNS,
        "lag_hours": LAG_HOURS,
        "train_end_exclusive": str(train_end),
        "validation_end_exclusive": str(validation_end),
        "metrics": metrics,
    }

    log(f"Saving metadata to {output_metadata_path}...")
    output_metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    log("Done.")
    return metadata


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a V3 Bangkok rain/no-rain model for the next 2 hours."
    )
    parser.add_argument("--feature-table", default=FEATURE_TABLE)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help="Optional random sample size for quick experiments. Defaults to full table.",
    )
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    return parser.parse_args()


def main():
    args = parse_args()
    metadata = train(args)
    print(json.dumps(metadata["metrics"], indent=2))


if __name__ == "__main__":
    main()
