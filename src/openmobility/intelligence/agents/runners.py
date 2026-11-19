from __future__ import annotations

from collections.abc import Mapping

from openmobility.intelligence import tools
from openmobility.intelligence.graph import (
    AgentRunner,
    ToolBudget,
)
from openmobility.intelligence.schemas import (
    AgentName,
    Evidence,
    SpecialistFinding,
)
from openmobility.intelligence.state import InvestigationState


def _scope_value(
    state: InvestigationState,
    key: str,
    default,
):
    return state.get("investigation_scope", {}).get(key, default)


def _traffic_runner(
    state: InvestigationState,
    budget: ToolBudget,
) -> SpecialistFinding:
    quality = budget.call(
        "get_traffic_quality",
        tools.get_traffic_quality,
        result_summary=lambda result: (
            f"Quality report for {result.get('total_rows', 0)} rows."
        ),
    )

    patterns = budget.call(
        "get_traffic_patterns",
        tools.get_traffic_patterns,
        result_summary=lambda result: (
            f"Traffic summaries generated for "
            f"{result.get('sensors', 0)} sensors."
        ),
    )

    observations = [
        f"Dataset contains {quality['total_rows']} rows "
        f"across {quality['unique_sensors']} sensors.",
        (
            f"{quality['rows_with_both_measurements']} rows contain "
            "both flow and occupancy measurements."
        ),
        (
            f"{quality['duplicate_sensor_timestamp_pairs']} "
            "duplicate sensor/timestamp pairs were found."
        ),
        (
            f"Exploratory analysis processed {patterns['rows']} rows "
            f"and generated {patterns['hour_groups']} hourly groups."
        ),
    ]

    statistics = quality.get("measurement_statistics", {})
    if statistics:
        observations.append(
            "Measurement statistics: "
            + ", ".join(
                f"{key}={value}"
                for key, value in statistics.items()
            )
        )

    return SpecialistFinding(
        agent="traffic",
        summary=(
            "Reviewed traffic data quality and generated "
            "hourly, weekday, and sensor-level traffic summaries."
        ),
        observations=observations,
        conclusions=[
            "These outputs describe the available dataset; "
            "they do not establish live road conditions."
        ],
        evidence=[
            Evidence(
                source="processing.quality.build_quality_report",
                description="Normalized traffic data quality metrics.",
                dataset=str(tools.NORMALIZED),
                metrics={
                    "rows": quality["total_rows"],
                    "sensors": quality["unique_sensors"],
                    "duplicate_pairs": quality[
                        "duplicate_sensor_timestamp_pairs"
                    ],
                },
            ),
            Evidence(
                source="processing.exploratory.build_exploratory_outputs",
                description="Hourly, weekday, and sensor coverage summaries.",
                dataset=str(tools.NORMALIZED),
                artifact_path=patterns["output_directory"],
                metrics={
                    "hour_groups": patterns["hour_groups"],
                    "weekday_groups": patterns["weekday_groups"],
                },
            ),
        ],
        limitations=[
            "The dataset's timestamp range must be checked before "
            "describing the results as current.",
            "Aggregate hourly patterns can conceal local variation.",
        ],
        confidence=0.95,
    )


def _predictive_runner(
    state: InvestigationState,
    budget: ToolBudget,
) -> SpecialistFinding:
    horizon = int(
        _scope_value(state, "forecast_horizon_hours", 24)
    )
    refresh = bool(_scope_value(state, "refresh_analytics", False))
    min_sensor_rows = int(
        _scope_value(state, "min_sensor_rows", 40)
    )

    bundle = budget.call(
        "get_forecast_bundle",
        tools.get_forecast_bundle,
        horizon=horizon,
        min_sensor_rows=min_sensor_rows,
        refresh=refresh,
        arguments={
            "horizon": horizon,
            "refresh": refresh,
            "min_sensor_rows": min_sensor_rows,
        },
        result_summary=lambda result: (
            f"Forecast analytics status: "
            f"{result.get('summary', {}).get('status', 'unknown')}."
        ),
    )

    summary = bundle.get("summary", {})

    if summary.get("status") != "success":
        return SpecialistFinding(
            agent="predictive",
            summary="Forecast analytics could not produce model outputs.",
            observations=[
                f"Forecast status: {summary.get('status', 'unknown')}."
            ],
            limitations=[
                "No valid forecasting results were available.",
                str(summary.get("skipped_sensors", []))[:1000],
            ],
            confidence=0.0,
            status="failed",
        )

    observations = [
        f"Models were trained for {summary['sensors_modelled']} sensors.",
        f"{summary['holdout_rows']} holdout predictions were evaluated.",
        f"{summary['forecast_rows']} future forecast rows were generated.",
        f"{summary['anomalies_detected']} holdout anomalies were detected.",
    ]

    forecast_summary = bundle.get("forecast_summary")
    if forecast_summary:
        observations.append(
            "Forecast summary: "
            + ", ".join(
                f"{key}={value}"
                for key, value in forecast_summary.items()
            )
        )

    limitations = [
        "Forecasts are model outputs, not guaranteed future conditions.",
        "The existing predictive pipeline forecasts each sensor "
        "independently; it does not model network-wide propagation.",
    ]

    if refresh:
        limitations.append(
            "Forecast models were explicitly refreshed for this run."
        )
    else:
        limitations.append(
            "Existing forecast artifacts may have been reused."
        )

    return SpecialistFinding(
        agent="predictive",
        summary=(
            "Reviewed sensor-level XGBoost forecasts, holdout metrics, "
            "and residual anomaly outputs."
        ),
        observations=observations,
        conclusions=[
            "Forecast reliability should be interpreted alongside "
            "holdout metrics and sensor data coverage."
        ],
        evidence=[
            Evidence(
                source="processing.predictive.run_predictive_analytics",
                description="Forecast and holdout analysis summary.",
                dataset=str(tools.NORMALIZED),
                artifact_path=str(tools.REPORTS / "forecasting"),
                metrics={
                    "sensors_modelled": summary["sensors_modelled"],
                    "holdout_rows": summary["holdout_rows"],
                    "forecast_rows": summary["forecast_rows"],
                    "anomalies_detected": summary["anomalies_detected"],
                },
            )
        ],
        limitations=limitations,
        confidence=0.85,
    )


def _geospatial_runner(
    state: InvestigationState,
    budget: ToolBudget,
) -> SpecialistFinding:
    result = budget.call(
        "get_geographic_analysis",
        tools.get_geographic_analysis,
        result_summary=lambda value: (
            f"Geographic join matched "
            f"{value.get('matched_segments', 0)} road segments."
        ),
    )

    observations = [
        f"Geographic reference contains "
        f"{result['geographic_segments']} segments.",
        f"{result['matched_segments']} segments matched traffic data.",
        f"{result['unmatched_geographic_segments']} segments "
        "did not match a traffic sensor.",
        f"Match percentage: {result['matched_percentage']}%.",
    ]

    if result.get("mean_flow_matched_segments") is not None:
        observations.append(
            "Mean flow across matched segments: "
            f"{result['mean_flow_matched_segments']:.2f} vehicles/hour."
        )

    return SpecialistFinding(
        agent="geospatial",
        summary=(
            "Joined the geographic road reference with aggregated "
            "sensor measurements and reviewed spatial coverage."
        ),
        observations=observations,
        conclusions=[
            "The join identifies geographic segments with matching "
            "sensor identifiers; it does not prove that every sensor "
            "measurement represents the full road segment."
        ],
        evidence=[
            Evidence(
                source="processing.geography.build_geographic_dataset",
                description="Road geometry joined to aggregated traffic data.",
                dataset=str(tools.GEO_RAW),
                artifact_path=str(
                    tools.REPORTS / "geography" / "sensor_road_join.parquet"
                ),
                metrics={
                    "geographic_segments": result["geographic_segments"],
                    "matched_segments": result["matched_segments"],
                    "unmatched_segments": result[
                        "unmatched_geographic_segments"
                    ],
                    "matched_percentage": result["matched_percentage"],
                },
            )
        ],
        limitations=[
            "Spatial coverage is constrained by the iu_ac/sensor_id join.",
            "The existing geographic join is not a full network analysis.",
        ],
        confidence=0.9,
    )


def _network_runner(
    state: InvestigationState,
    budget: ToolBudget,
) -> SpecialistFinding:
    tolerance = float(
        _scope_value(state, "network_snap_tolerance_m", 5.0)
    )

    result = budget.call(
        "get_network_analysis",
        tools.get_network_analysis,
        snap_tolerance_m=tolerance,
        arguments={"snap_tolerance_m": tolerance},
        result_summary=lambda value: (
            f"Built a graph with {value.get('graph_nodes', 0)} nodes "
            f"and {value.get('graph_edges', 0)} edges."
        ),
    )

    observations = [
        f"Road graph contains {result['graph_nodes']} nodes "
        f"and {result['graph_edges']} edges.",
        f"Detected {result['connected_components']} connected components.",
        f"Detected {result['intersection_nodes']} nodes with degree 3 or more.",
        f"Detected {result['dead_end_nodes']} degree-1 nodes.",
        f"Matched {result['matched_graph_edges']} graph edges "
        f"to traffic identifiers "
        f"({result['matched_edge_percentage']}%).",
        f"Identified {result['high_flow_segment_count']} "
        "segments in the configured high-flow percentile."
    ]

    return SpecialistFinding(
        agent="network",
        summary=(
            "Derived a geometric road graph and assessed connectivity, "
            "intersection structure, and traffic-data coverage."
        ),
        observations=observations,
        conclusions=[
            "High-flow segments are candidates for further investigation, "
            "not confirmed bottlenecks.",
            "Connected components reflect geometric connectivity under "
            "the configured snapping tolerance."
        ],
        evidence=[
            Evidence(
                source="processing.network.build_network_topology",
                description="Road topology and traffic coverage summary.",
                dataset=str(tools.GEO_RAW),
                artifact_path=result["output_path"],
                metrics={
                    "nodes": result["graph_nodes"],
                    "edges": result["graph_edges"],
                    "components": result["connected_components"],
                    "intersection_nodes": result["intersection_nodes"],
                    "matched_edge_percentage": result[
                        "matched_edge_percentage"
                    ],
                    "high_flow_segment_count": result[
                        "high_flow_segment_count"
                    ],
                },
            )
        ],
        limitations=result["limitations"],
        confidence=0.75,
    )


def build_agent_runners() -> Mapping[AgentName, AgentRunner]:
    """Return the four registered specialist runners."""
    return {
        "traffic": _traffic_runner,
        "predictive": _predictive_runner,
        "geospatial": _geospatial_runner,
        "network": _network_runner,
    }