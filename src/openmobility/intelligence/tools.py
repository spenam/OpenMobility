
from __future__ import annotations

from pathlib import Path

from openmobility.processing.evaluation import run_evaluation
from openmobility.processing.exploratory import build_exploratory_outputs
from openmobility.processing.geography import build_geographic_dataset
from openmobility.processing.predictive import run_predictive_analytics
from openmobility.processing.quality import build_quality_report


NORMALIZED = Path("data/processed/paris_traffic_normalized.parquet")
GEO_RAW = Path("data/raw/paris_sensor_geography.geojson")


def get_quality() -> dict:
    if not NORMALIZED.exists():
        raise FileNotFoundError("Run preprocessing first.")
    return build_quality_report(NORMALIZED)


def get_traffic_patterns() -> dict:
    if not NORMALIZED.exists():
        raise FileNotFoundError("Run preprocessing first.")
    return build_exploratory_outputs(
        NORMALIZED,
        Path("reports/intelligence/exploratory"),
    )


def get_forecasts(horizon: int = 24) -> dict:
    if not NORMALIZED.exists():
        raise FileNotFoundError("Run preprocessing first.")
    if not 1 <= horizon <= 168:
        raise ValueError("Forecast horizon must be between 1 and 168.")
    return run_predictive_analytics(
        NORMALIZED,
        output_dir=Path("reports/intelligence/forecasting"),
        horizon=horizon,
    )


def get_geographic_analysis() -> dict:
    if not NORMALIZED.exists() or not GEO_RAW.exists():
        raise FileNotFoundError(
            "Normalized traffic and geographic reference data are required."
        )
    return build_geographic_dataset(
        GEO_RAW,
        NORMALIZED,
        Path("reports/intelligence/geography.parquet"),
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
