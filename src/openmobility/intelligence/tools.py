
from __future__ import annotations

from pathlib import Path

import json

import pandas as pd

from openmobility.processing.evaluation import run_evaluation
from openmobility.processing.exploratory import build_exploratory_outputs
from openmobility.processing.geography import build_geographic_dataset
from openmobility.processing.predictive import run_predictive_analytics
from openmobility.processing.quality import build_quality_report
from openmobility.processing.network import build_network_topology



NORMALIZED = Path("data/processed/paris_traffic_normalized.parquet")
GEO_RAW = Path("data/raw/paris_sensor_geography.geojson")
REPORTS = Path("reports/intelligence")

def _require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{description} not found: {path}. "
            "Run the relevant ingestion/preprocessing command first."
        )

def get_quality() -> dict:
    if not NORMALIZED.exists():
        raise FileNotFoundError("Run preprocessing first.")
    return build_quality_report(NORMALIZED)


def get_traffic_patterns() -> dict:
    if not NORMALIZED.exists():
        raise FileNotFoundError("Run preprocessing first.")
    return build_exploratory_outputs(
        NORMALIZED,
        REPORTS / "exploratory",
    )


def _forecast_artifacts_are_current(output_dir: Path) -> bool:
    summary_path = output_dir / "summary.json"
    required = [
        summary_path,
        output_dir / "model_metrics.csv",
        output_dir / "next_24h_forecasts.csv",
        output_dir / "traffic_anomalies.csv",
    ]

    if not all(path.exists() for path in required):
        return False

    source_mtime = NORMALIZED.stat().st_mtime
    return all(
        path.stat().st_mtime >= source_mtime
        for path in required
    )


def get_forecast_bundle(
    *,
    horizon: int = 24,
    refresh: bool = False,
) -> dict:
    _require_file(NORMALIZED, "Normalized traffic dataset")

    if not 1 <= horizon <= 168:
        raise ValueError("Forecast horizon must be between 1 and 168.")

    output_dir = REPORTS / "forecasting"

    if refresh or not _forecast_artifacts_are_current(output_dir):
        summary = run_predictive_analytics(
            NORMALIZED,
            output_dir=output_dir,
            horizon=horizon,
        )
    else:
        summary = json.loads(
            (output_dir / "summary.json").read_text(
                encoding="utf-8"
            )
        )

    bundle = {"summary": summary}

    if summary.get("status") != "success":
        return bundle

    forecasts = pd.read_csv(
        output_dir / "next_24h_forecasts.csv"
    )
    metrics = pd.read_csv(
        output_dir / "model_metrics.csv"
    )
    anomalies = pd.read_csv(
        output_dir / "traffic_anomalies.csv"
    )

    if not forecasts.empty:
        bundle["forecast_summary"] = {
            "rows": int(len(forecasts)),
            "sensors": int(forecasts["sensor_id"].nunique()),
            "mean_predicted_flow": float(
                forecasts["predicted_flow"].mean()
            ),
            "maximum_predicted_flow": float(
                forecasts["predicted_flow"].max()
            ),
            "minimum_predicted_flow": float(
                forecasts["predicted_flow"].min()
            ),
            "forecast_start": str(forecasts["timestamp"].min()),
            "forecast_end": str(forecasts["timestamp"].max()),
        }

    if not metrics.empty:
        bundle["model_metrics"] = (
            metrics.groupby("model")[
                ["mae", "rmse", "mape_pct"]
            ]
            .mean()
            .round(3)
            .to_dict()
        )

    bundle["anomaly_count"] = int(len(anomalies))
    bundle["anomaly_examples"] = (
        anomalies.head(20).to_dict(orient="records")
    )

    return bundle


def get_geographic_analysis() -> dict:
    _require_file(NORMALIZED, "Normalized traffic dataset")
    _require_file(GEO_RAW, "Geographic reference")

    output_path = REPORTS / "geography" / "sensor_road_join.parquet"

    summary = build_geographic_dataset(
        GEO_RAW,
        NORMALIZED,
        output_path,
    )

    joined = pd.read_parquet(output_path)

    matched = joined.loc[joined["mean_flow"].notna()]

    summary["mean_flow_matched_segments"] = (
        float(matched["mean_flow"].mean())
        if not matched.empty
        else None
    )
    summary["mean_occupancy_matched_segments"] = (
        float(matched["mean_occupancy"].mean())
        if not matched.empty
        else None
    )
    summary["high_flow_segments"] = (
        matched.nlargest(20, "mean_flow")[
            ["iu_ac", "mean_flow", "mean_occupancy"]
        ].to_dict(orient="records")
        if not matched.empty
        else []
    )

    return summary


def get_network_analysis(
    *,
    snap_tolerance_m: float = 5.0,
) -> dict:
    _require_file(NORMALIZED, "Normalized traffic dataset")
    _require_file(GEO_RAW, "Geographic reference")

    return build_network_topology(
        GEO_RAW,
        NORMALIZED,
        REPORTS / "network" / "network_summary.json",
        snap_tolerance_m=snap_tolerance_m,
    )

def get_model_evaluation() -> dict:
    predictions = Path(
        "reports/intelligence/forecasting/holdout_predictions.csv"
    )
    if not predictions.exists():
        raise FileNotFoundError(
            "Run predictive analytics before model evaluation."
        )
    return run_evaluation(
        predictions,
        Path("reports/intelligence/evaluation"),
    )
