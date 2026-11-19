"""CPU-friendly traffic forecasting and anomaly detection for OpenMobility.

The model is trained locally on the normalized Paris traffic Parquet dataset.
It is not a pretrained traffic model.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm
from xgboost import XGBRegressor

REQUIRED_COLUMNS = {"timestamp", "sensor_id", "flow_vehicles_per_hour"}


def _load_sensor_frame(
    parquet_path: Path,
    min_rows: int = 240,
) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    raw = pl.read_parquet(parquet_path)

    missing = REQUIRED_COLUMNS - set(raw.columns)
    if missing:
        raise ValueError(
            f"Normalized dataset is missing columns: {sorted(missing)}"
        )

    columns = [
        "timestamp",
        "sensor_id",
        "flow_vehicles_per_hour",
    ]
    if "occupancy_pct" in raw.columns:
        columns.append("occupancy_pct")

    pdf = raw.select(columns).to_pandas()

    pdf["timestamp"] = pd.to_datetime(
        pdf["timestamp"], utc=True, errors="coerce"
    ).dt.tz_convert("Europe/Paris")

    pdf["flow_vehicles_per_hour"] = pd.to_numeric(
        pdf["flow_vehicles_per_hour"], errors="coerce"
    )

    if "occupancy_pct" in pdf:
        pdf["occupancy_pct"] = pd.to_numeric(
            pdf["occupancy_pct"], errors="coerce"
        )

    result = {}
    skipped = []

    for sensor, group in pdf.groupby("sensor_id"):
        group = group.dropna(
            subset=["timestamp", "flow_vehicles_per_hour"]
        )

        hourly = (
            group.groupby("timestamp")["flow_vehicles_per_hour"]
            .mean()
            .sort_index()
            .resample("1h")
            .mean()
        )

        valid_hours = int(hourly.notna().sum())

        # Only sensors passing the minimum-history condition are used.
        if valid_hours < min_rows:
            skipped.append({
                "sensor_id": str(sensor),
                "valid_hours": valid_hours,
                "reason": "insufficient_history",
            })
            continue

        observed = hourly.notna()
        flow_features = hourly.ffill(limit=3)

        frame = pd.DataFrame(index=hourly.index)
        frame["target"] = hourly
        frame["target_observed"] = observed
        frame["lag_1"] = flow_features.shift(1)
        frame["lag_24"] = flow_features.shift(24)
        frame["rolling_mean_24"] = (
            flow_features.shift(1).rolling(24, min_periods=6).mean()
        )

        frame["hour"] = frame.index.hour
        frame["day_of_week"] = frame.index.dayofweek
        frame["is_weekend"] = (
            frame["day_of_week"] >= 5
        ).astype(int)

        frame["hour_sin"] = np.sin(
            2 * np.pi * frame["hour"] / 24
        )
        frame["hour_cos"] = np.cos(
            2 * np.pi * frame["hour"] / 24
        )
        frame["dow_sin"] = np.sin(
            2 * np.pi * frame["day_of_week"] / 7
        )
        frame["dow_cos"] = np.cos(
            2 * np.pi * frame["day_of_week"] / 7
        )

        if "occupancy_pct" in group.columns:
            occupancy = (
                group.groupby("timestamp")["occupancy_pct"]
                .mean()
                .sort_index()
                .resample("1h")
                .mean()
                .reindex(frame.index)
            )
            frame["occupancy_lag_1"] = occupancy.ffill(limit=3).shift(1)
        else:
            frame["occupancy_lag_1"] = np.nan

        # Keep observed targets. Feature warm-up rows are filtered later.
        frame = frame.loc[frame["target_observed"]]

        result[str(sensor)] = frame

    return result, skipped


FEATURES = [
    "lag_1", "lag_24", "rolling_mean_24",
    "hour", "day_of_week", "is_weekend",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "occupancy_lag_1",
]


def _model() -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=250,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        tree_method="hist",
        n_jobs=4,
        random_state=42,
    )


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    errors = predicted - actual
    nonzero = np.abs(actual) > 1e-8
    mape = float(
        np.mean(np.abs(errors[nonzero] / actual[nonzero])) * 100) if nonzero.any() else float("nan")
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mape_pct": mape,
        "observations": int(len(actual)),
    }


def _seasonal_prediction(frame: pd.DataFrame) -> np.ndarray:
    # At the same weekday/hour last week; use yesterday at the same hour if
    # a weekly lag is unavailable (normally removed by feature filtering).
    return frame["lag_24"].to_numpy(dtype=float)


def _make_future_features(history: pd.Series, timestamp: pd.Timestamp, occupancy: float) -> dict:
    values = history.to_numpy(dtype=float)
    def lag(n: int) -> float:
        return float(values[-n]) if len(values) >= n else float(values[-1])
    tail24 = values[-24:] if len(values) else np.array([0.0])
    dow = timestamp.dayofweek
    hour = timestamp.hour
    return {
        "lag_1": lag(1), "lag_24": lag(24),
        "rolling_mean_24": float(np.mean(tail24)),
        "hour": hour, "day_of_week": dow, "is_weekend": int(dow >= 5),
        "hour_sin": float(np.sin(2 * np.pi * hour / 24)),
        "hour_cos": float(np.cos(2 * np.pi * hour / 24)),
        "dow_sin": float(np.sin(2 * np.pi * dow / 7)),
        "dow_cos": float(np.cos(2 * np.pi * dow / 7)),
        "occupancy_lag_1": occupancy,
    }


def run_predictive_analytics(
    parquet_path: Path,
    output_dir: Path = Path("reports/forecasting"),
    min_sensor_rows: int = 240,
    horizon: int = 24,
) -> dict:
    """Train/evaluate one XGBoost model per sensor and forecast the next hours."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sensors, skipped_sensors = _load_sensor_frame(
        parquet_path,
        min_rows=min_sensor_rows,
    )

    print(f"Sensors eligible for modelling: {len(sensors)}")
    print(f"Sensors skipped: {len(skipped_sensors)}")

    if not sensors:
        summary = {
            "status": "no_eligible_sensors",
            "minimum_required_hours": min_sensor_rows,
            "eligible_sensors": 0,
            "skipped_sensors": skipped_sensors,
        }

        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

        return summary

    metrics_rows = []
    prediction_frames = []
    future_frames = []
    anomaly_frames = []
    model_dir = output_dir / "models"
    model_dir.mkdir(exist_ok=True)

    for sensor_id, frame in tqdm(
        sensors.items(),
        total=len(sensors),
        desc="Training and predicting",
        unit="sensor",
    ):
        model_frame = frame.dropna(
            subset=["target", "lag_1", "lag_24", "rolling_mean_24"]
        )
        split = int(len(model_frame) * 0.8)
        train, test = model_frame.iloc[:split], model_frame.iloc[split:]
        if len(train) < 80 or len(test) < 12:
            skipped_sensors.append({
                "sensor_id": sensor_id,
                "usable_rows": len(model_frame),
                "reason": "insufficient_rows_for_train_test_split",
            })
            continue

        model = _model()
        model.fit(train[FEATURES], train["target"])
        xgb_pred = np.maximum(0, model.predict(test[FEATURES]))
        baseline_pred = np.maximum(0, _seasonal_prediction(test))
        actual = test["target"].to_numpy(dtype=float)

        metrics_rows.extend([
            {"sensor_id": sensor_id, "model": "seasonal_naive_24h", 
             **_metrics(actual, baseline_pred)},
            {"sensor_id": sensor_id, "model": "xgboost", **_metrics(actual, xgb_pred)},
        ])

        test_out = pd.DataFrame({
            "timestamp": test.index,
            "sensor_id": sensor_id,
            "actual_flow": actual,
            "seasonal_naive_prediction": baseline_pred,
            "xgboost_prediction": xgb_pred,
        })
        prediction_frames.append(test_out)

        # Robust residual anomaly score on the chronological holdout.
        residual = actual - xgb_pred
        rolling_center = pd.Series(residual, index=test.index).rolling(24, min_periods=8).median()
        abs_dev = (pd.Series(residual, index=test.index) - rolling_center).abs()
        rolling_mad = abs_dev.rolling(24, min_periods=8).median()
        scale = (1.4826 * rolling_mad).replace(0, np.nan)
        score = (pd.Series(residual, index=test.index) - rolling_center) / scale
        anomaly = test_out.copy()
        anomaly["residual"] = residual
        anomaly["robust_z_score"] = score.to_numpy()
        anomaly["is_anomaly"] = anomaly["robust_z_score"].abs() >= 3.5
        anomaly["anomaly_type"] = np.select(
            [anomaly["residual"] > 0, anomaly["residual"] < 0],
            ["traffic_spike", "traffic_drop"],
            default="none",
        )
        anomaly_frames.append(anomaly.loc[anomaly["is_anomaly"]])

        # Refit on all available rows before generating the next 24-hour outlook.
        final_model = _model()
        final_model.fit(model_frame[FEATURES], model_frame["target"])
        final_model.save_model(str(model_dir / f"sensor_{sensor_id}.json"))

        # Use the full observed hourly series, not just rows surviving the lag
        # warm-up, so the 24-hour seasonal lag is available when possible.
        history = frame["target"].dropna().copy()
        last_timestamp = history.index[-1]
        occ = float(
            frame["occupancy_lag_1"].dropna()
            .iloc[-1]) if frame["occupancy_lag_1"].notna().any() else 0.0
        rows = []
        for step in range(1, horizon + 1):
            ts = last_timestamp + pd.Timedelta(hours=step)
            features = _make_future_features(history, ts, occ)
            x = pd.DataFrame([features], columns=FEATURES)
            predicted = max(0.0, float(final_model.predict(x)[0]))
            rows.append({
                "timestamp": ts,
                "sensor_id": sensor_id,
                "forecast_horizon_hours": step,
                "predicted_flow": predicted,
            })
            history.loc[ts] = predicted
        future_frames.append(pd.DataFrame(rows))

    if not metrics_rows:
        summary = {
            "status": "no_modelable_sensors",
            "minimum_required_hours": min_sensor_rows,
            "eligible_sensors": len(sensors),
            "skipped_sensors": skipped_sensors,
        }

        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

        return summary

    metrics_df = pd.DataFrame(metrics_rows)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    forecasts_df = pd.concat(
        future_frames, ignore_index=True) if future_frames else pd.DataFrame()
    anomalies_df = pd.concat(
        anomaly_frames, ignore_index=True) if anomaly_frames else pd.DataFrame()

    metrics_df.to_csv(output_dir / "model_metrics.csv", index=False)
    predictions_df.to_csv(output_dir / "holdout_predictions.csv", index=False)
    forecasts_df.to_csv(output_dir / "next_24h_forecasts.csv", index=False)
    anomalies_df.to_csv(output_dir / "traffic_anomalies.csv", index=False)

    metrics_df.to_parquet(output_dir / "model_metrics.parquet", index=False)
    predictions_df.to_parquet(output_dir / "holdout_predictions.parquet", index=False)
    forecasts_df.to_parquet(output_dir / "next_24h_forecasts.parquet", index=False)
    anomalies_df.to_parquet(output_dir / "traffic_anomalies.parquet", index=False)



    summary = {
        "sensors_modelled": int(metrics_df["sensor_id"].nunique()),
        "holdout_rows": int(len(predictions_df)),
        "forecast_rows": int(len(forecasts_df)),
        "anomalies_detected": int(len(anomalies_df)),
        "metrics_file": str(output_dir / "model_metrics.csv"),
        "predictions_file": str(output_dir / "holdout_predictions.csv"),
        "forecast_file": str(output_dir / "next_24h_forecasts.csv"),
        "anomalies_file": str(output_dir / "traffic_anomalies.csv"),
        "models_directory": str(model_dir),
        "mean_metrics_by_model": metrics_df.groupby("model")[["mae", "rmse", "mape_pct"]]
        .mean().round(3).to_dict(),
        "status": "success",
        "eligible_sensors": len(sensors),
        "skipped_sensors": skipped_sensors,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
