from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def calculate_metrics(actual, predicted):
    """Calculate regression metrics."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual = actual[valid]
    predicted = predicted[valid]

    if len(actual) == 0:
        return {
            "observations": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "r2": np.nan,
        }

    errors = actual - predicted
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))

    ss_total = np.sum((actual - np.mean(actual)) ** 2)
    ss_residual = np.sum(errors ** 2)

    r2 = (
        1 - ss_residual / ss_total
        if ss_total > 0
        else np.nan
    )

    return {
        "observations": len(actual),
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2) if np.isfinite(r2) else np.nan,
    }


def evaluate_holdout(
    input_csv,
    output_dir,
    actual_col="actual_flow",
    prediction_col="xgboost_prediction",
    baseline_col="seasonal_naive_prediction",
):
    """
    Evaluate XGBoost holdout predictions against a previous-hour baseline.

    The input CSV must contain:
      - sensor_id
      - actual_col
      - prediction_col
      - baseline_col (for baseline comparison)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)

    required = {
        "sensor_id",
        actual_col,
        prediction_col,
        baseline_col,
    }
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns in holdout CSV: {sorted(missing)}"
        )

    for column in [actual_col, prediction_col, baseline_col]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(
        subset=["sensor_id", actual_col, prediction_col]
    ).copy()

    if df.empty:
        raise ValueError("No valid holdout observations found.")

    # Evaluate each sensor independently.
    metric_rows = []

    for sensor_id, group in df.groupby("sensor_id"):
        actual = group[actual_col]
        predicted = group[prediction_col]

        xgb_metrics = calculate_metrics(actual, predicted)

        row = {
            "sensor_id": sensor_id,
            "model": "xgboost",
            **xgb_metrics,
        }

        # Baseline comparison uses only rows where lag_1 is available.
        baseline_valid = group[baseline_col].notna()

        if baseline_valid.any():
            baseline_metrics = calculate_metrics(
                group.loc[baseline_valid, actual_col],
                group.loc[baseline_valid, baseline_col],
            )
            row["baseline_mae"] = baseline_metrics["mae"]
            row["baseline_rmse"] = baseline_metrics["rmse"]
            row["baseline_r2"] = baseline_metrics["r2"]
        else:
            row["baseline_mae"] = np.nan
            row["baseline_rmse"] = np.nan
            row["baseline_r2"] = np.nan

        metric_rows.append(row)

    metrics = pd.DataFrame(metric_rows)

    # Overall XGBoost performance.
    overall_xgb = calculate_metrics(
        df[actual_col],
        df[prediction_col],
    )

    baseline_valid = df[baseline_col].notna()

    if baseline_valid.any():
        overall_baseline = calculate_metrics(
            df.loc[baseline_valid, actual_col],
            df.loc[baseline_valid, baseline_col],
        )
    else:
        overall_baseline = None

    metrics.to_csv(
        output_dir / "evaluation_metrics.csv",
        index=False,
    )

    summary = {
        "total_observations": len(df),
        "total_sensors": int(df["sensor_id"].nunique()),
        "xgboost": overall_xgb,
        "baseline": overall_baseline,
    }

    with open(
        output_dir / "evaluation_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2, allow_nan=False, default=str)

    return df, metrics, summary


def plot_actual_vs_predicted(
    df,
    output_dir,
    actual_col="actual_flow",
    prediction_col="xgboost_prediction",
    timestamp_col="timestamp",
    max_sensors=6,
):
    """Plot actual versus predicted traffic for selected sensors."""
    output_dir = Path(output_dir)

    data = df.copy()

    if timestamp_col in data.columns:
        data[timestamp_col] = pd.to_datetime(
            data[timestamp_col],
            errors="coerce",
        )
        data = data.sort_values(timestamp_col)

    # Select randomly sensors among the most holdout observations.
    sensor_ids = (
        data.groupby("sensor_id")
        .size()
        .sort_values(ascending=False)
        .head(20 * max_sensors)
        .sample(n=max_sensors, random_state=42)
        .index
    )

    for sensor_id in sensor_ids:
        group = data[data["sensor_id"] == sensor_id]

        x = (
            group[timestamp_col]
            if timestamp_col in group.columns
            else np.arange(len(group))
        )

        plt.figure(figsize=(12, 5))
        plt.plot(
            x,
            group[actual_col],
            label="Actual traffic",
            linewidth=2,
        )
        plt.plot(
            x,
            group[prediction_col],
            label="XGBoost prediction",
            linewidth=2,
        )

        plt.title(f"Traffic forecast - Sensor {sensor_id}")
        plt.xlabel("Time")
        plt.ylabel("Vehicles per hour")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(
            output_dir / f"sensor_{sensor_id}_forecast.png",
            dpi=150,
        )
        plt.close()


def plot_residuals(df, output_dir, actual_col="actual_flow",
                   prediction_col="xgboost_prediction"):
    """Plot prediction errors."""
    residuals = (
        df[actual_col].to_numpy()
        - df[prediction_col].to_numpy()
    )

    plt.figure(figsize=(9, 5))
    plt.hist(residuals, bins=80)
    plt.axvline(0, linestyle="--")
    plt.title("XGBoost prediction error distribution")
    plt.xlabel("Actual - predicted (vehicles/hour)")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "residual_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.scatter(df[prediction_col], residuals, alpha=0.5, s=3)
    plt.axhline(0, linestyle="--")
    plt.title("Residuals versus predicted traffic")
    plt.xlabel("Predicted traffic (vehicles/hour)")
    plt.ylabel("Residual (vehicles/hour)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "residuals_vs_prediction.png", dpi=150)
    plt.close()


def plot_sensor_metrics(metrics, output_dir):
    """Compare XGBoost and baseline MAE across sensors."""
    plot_data = metrics.dropna(
        subset=["mae", "baseline_mae"]
    ).sort_values("mae")

    if plot_data.empty:
        return

    plot_data = plot_data.head(20)

    x = np.arange(len(plot_data))
    width = 0.38

    plt.figure(figsize=(12, 6))
    plt.bar(
        x - width / 2,
        plot_data["mae"],
        width,
        label="XGBoost MAE",
    )
    plt.bar(
        x + width / 2,
        plot_data["baseline_mae"],
        width,
        label="Previous-hour baseline MAE",
    )

    plt.xticks(
        x,
        plot_data["sensor_id"].astype(str),
        rotation=45,
        ha="right",
    )
    plt.ylabel("MAE (vehicles/hour)")
    plt.xlabel("Sensor")
    plt.title("XGBoost versus baseline performance")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "sensor_mae_comparison.png", dpi=150)
    plt.close()


def run_evaluation(
    input_csv,
    output_dir,
    actual_col="actual_flow",
    prediction_col="xgboost_prediction",
    baseline_col="seasonal_naive_prediction",
    timestamp_col="timestamp",
):
    """Run evaluation and generate all visualizations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df, metrics, summary = evaluate_holdout(
        input_csv=input_csv,
        output_dir=output_dir,
        actual_col=actual_col,
        prediction_col=prediction_col,
        baseline_col=baseline_col,
    )

    plot_actual_vs_predicted(
        df,
        output_dir,
        actual_col,
        prediction_col,
        timestamp_col,
    )

    plot_residuals(
        df,
        output_dir,
        actual_col,
        prediction_col,
    )

    plot_sensor_metrics(metrics, output_dir)

    return summary
