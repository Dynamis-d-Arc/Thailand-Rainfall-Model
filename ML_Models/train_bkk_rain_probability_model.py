import argparse
import json
import os
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import psycopg2
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "postgres",
    "user": "postgres",
    "password": os.getenv("PGPASSWORD", "Pass1234"),
}

TRAINING_TABLE = "bkk_rain_ml_training"
TARGET_COLUMN = "rain_next_1h"
MODEL_DIR = Path(__file__).with_name("trained_models")

FEATURE_COLUMNS = [
    "temperature_2m",
    "dew_point_2m",
    "temperature_dew_point_spread",
    "relative_humidity_2m",
    "precipitation",
    "precipitation_lag_1h",
    "precipitation_lag_3h",
    "precipitation_sum_past_3h",
    "precipitation_sum_past_6h",
    "cloud_cover",
    "cloud_cover_avg_past_3h",
    "pressure_msl",
    "pressure_msl_change_3h",
    "surface_pressure",
    "wind_speed_10m",
    "wind_speed_10m_avg_past_3h",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
]


def log(message):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def connect():
    return psycopg2.connect(**DB_CONFIG)


def fetch_training_data(conn, table, target, sample_rows=None):
    columns = ["grid_number", "forecast_time", *FEATURE_COLUMNS, target]
    column_sql = ", ".join(columns)

    if sample_rows:
        log(f"Loading a random sample of {sample_rows:,} rows from {table}...")
        query = f"""
        SELECT {column_sql}
        FROM (
            SELECT {column_sql}
            FROM {table}
            WHERE {target} IS NOT NULL
            ORDER BY random()
            LIMIT {int(sample_rows)}
        ) sampled
        ORDER BY forecast_time, grid_number;
        """
    else:
        log(f"Loading all rows from {table}. This can take a while...")
        query = f"""
        SELECT {column_sql}
        FROM {table}
        WHERE {target} IS NOT NULL
        ORDER BY forecast_time, grid_number;
        """

    return pd.read_sql_query(query, conn, parse_dates=["forecast_time"])


def add_split_column(df, train_fraction, validation_fraction):
    unique_times = np.array(sorted(df["forecast_time"].unique()))
    train_end = unique_times[int(len(unique_times) * train_fraction)]
    validation_end = unique_times[int(len(unique_times) * (train_fraction + validation_fraction))]

    df = df.copy()
    df["split"] = "test"
    df.loc[df["forecast_time"] < train_end, "split"] = "train"
    df.loc[
        (df["forecast_time"] >= train_end) & (df["forecast_time"] < validation_end),
        "split",
    ] = "validation"
    return df, train_end, validation_end


def build_model(model_type):
    if model_type == "logistic":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    return HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=250,
        max_leaf_nodes=31,
        l2_regularization=0.05,
        random_state=42,
    )


def evaluate(model, x, y, split_name):
    probabilities = model.predict_proba(x)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "split": split_name,
        "rows": int(len(y)),
        "rain_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "pr_auc": float(average_precision_score(y, probabilities)),
        "brier_score": float(brier_score_loss(y, probabilities)),
        "log_loss": float(log_loss(y, probabilities)),
        "confusion_matrix": confusion_matrix(y, predictions).tolist(),
        "classification_report": classification_report(y, predictions, output_dict=True),
    }
    return metrics


def print_metrics(metrics):
    print(f"\n{metrics['split'].upper()} METRICS")
    print(f"Rows:        {metrics['rows']:,}")
    print(f"Rain rate:   {metrics['rain_rate']:.4f}")
    print(f"ROC-AUC:     {metrics['roc_auc']:.4f}")
    print(f"PR-AUC:      {metrics['pr_auc']:.4f}")
    print(f"Brier score: {metrics['brier_score']:.4f}")
    print(f"Log loss:    {metrics['log_loss']:.4f}")
    print(f"Confusion matrix [[TN, FP], [FN, TP]]: {metrics['confusion_matrix']}")


def save_artifacts(model, args, metrics, train_end, validation_end):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"bkk_{args.target}_probability_{args.model_type}.joblib"
    metadata_path = MODEL_DIR / f"bkk_{args.target}_probability_{args.model_type}_metadata.json"

    joblib.dump(model, model_path)
    metadata = {
        "model_path": str(model_path),
        "training_table": args.table,
        "target": args.target,
        "model_type": args.model_type,
        "feature_columns": FEATURE_COLUMNS,
        "train_end_exclusive": str(train_end),
        "validation_end_exclusive": str(validation_end),
        "metrics": metrics,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return model_path, metadata_path


def main():
    parser = argparse.ArgumentParser(
        description="Train a Bangkok grid rainfall probability model from PostgreSQL."
    )
    parser.add_argument("--table", default=TRAINING_TABLE)
    parser.add_argument("--target", default=TARGET_COLUMN)
    parser.add_argument(
        "--model-type",
        choices=["hist_gradient_boosting", "logistic"],
        default="hist_gradient_boosting",
    )
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help="Optional row sample for quick experiments. Leave empty to train on all rows.",
    )
    args = parser.parse_args()

    log("Connecting to PostgreSQL...")
    with connect() as conn:
        df = fetch_training_data(conn, args.table, args.target, args.sample_rows)

    if df.empty:
        raise RuntimeError(f"No training rows found in {args.table}.")

    log(f"Finished loading {len(df):,} rows.")
    log("Creating time-based train/validation/test split...")
    df, train_end, validation_end = add_split_column(
        df,
        args.train_fraction,
        args.validation_fraction,
    )

    log("Loaded data")
    print(f"Rows: {len(df):,}", flush=True)
    print(f"Date range: {df['forecast_time'].min()} to {df['forecast_time'].max()}", flush=True)
    print(f"Train before: {train_end}", flush=True)
    print(f"Validation before: {validation_end}", flush=True)
    print(df["split"].value_counts().sort_index(), flush=True)

    train_df = df[df["split"] == "train"]
    validation_df = df[df["split"] == "validation"]
    test_df = df[df["split"] == "test"]

    log("Preparing feature matrices...")
    x_train = train_df[FEATURE_COLUMNS].astype("float32")
    y_train = train_df[args.target].astype("int8")
    x_validation = validation_df[FEATURE_COLUMNS].astype("float32")
    y_validation = validation_df[args.target].astype("int8")
    x_test = test_df[FEATURE_COLUMNS].astype("float32")
    y_test = test_df[args.target].astype("int8")

    model = build_model(args.model_type)
    log(f"Training {args.model_type} model on {len(x_train):,} rows...")
    model.fit(x_train, y_train)

    log("Evaluating validation split...")
    validation_metrics = evaluate(model, x_validation, y_validation, "validation")
    log("Evaluating test split...")
    test_metrics = evaluate(model, x_test, y_test, "test")

    print_metrics(validation_metrics)
    print_metrics(test_metrics)

    log("Saving model artifacts...")
    model_path, metadata_path = save_artifacts(
        model,
        args,
        metrics={"validation": validation_metrics, "test": test_metrics},
        train_end=train_end,
        validation_end=validation_end,
    )

    print(f"\nSaved model: {model_path}", flush=True)
    print(f"Saved metadata: {metadata_path}", flush=True)
    log("Done.")


if __name__ == "__main__":
    main()
