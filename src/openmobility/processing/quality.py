from pathlib import Path

import polars as pl


def build_quality_report(path: Path) -> dict:
    """Compute basic quality metrics for normalized traffic data."""
    df = pl.read_parquet(path)

    if df.is_empty():
        raise ValueError("The normalized dataset is empty.")

    total = df.height

    missing = {}
    for column in (
        "timestamp",
        "sensor_id",
        "flow_vehicles_per_hour",
        "occupancy_pct",
        "traffic_state",
    ):
        if column in df.columns:
            count = df[column].null_count()
            missing[column] = {
                "count": count,
                "percentage": round(100 * count / total, 2),
            }

    duplicate_count = (
        total - df.unique(subset=["sensor_id", "timestamp"]).height
    )

    report = {
        "total_rows": total,
        "unique_sensors": df["sensor_id"].n_unique(),
        "timestamp_min_utc": str(df["timestamp"].min()),
        "timestamp_max_utc": str(df["timestamp"].max()),
        "duplicate_sensor_timestamp_pairs": duplicate_count,
        "missing_values": missing,
    }

    # A measurement is usable for initial traffic analysis only when
    # both numeric measurements are present.
    usable = df.filter(
        pl.col("flow_vehicles_per_hour").is_not_null()
        & pl.col("occupancy_pct").is_not_null()
    )

    report["rows_with_both_measurements"] = usable.height
    report["rows_missing_at_least_one_measurement"] = total - usable.height

    if usable.height:
        report["measurement_statistics"] = (
            usable.select(
                pl.col("flow_vehicles_per_hour").min().alias("flow_min"),
                pl.col("flow_vehicles_per_hour").median().alias("flow_median"),
                pl.col("flow_vehicles_per_hour").max().alias("flow_max"),
                pl.col("occupancy_pct").min().alias("occupancy_min"),
                pl.col("occupancy_pct").median().alias("occupancy_median"),
                pl.col("occupancy_pct").max().alias("occupancy_max"),
            )
            .to_dicts()[0]
        )

    report["traffic_state_distribution"] = (
        df.group_by("traffic_state")
        .len()
        .sort("len", descending=True)
        .to_dicts()
    )

    return report
