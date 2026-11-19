from pathlib import Path

import polars as pl


def build_exploratory_outputs(
    parquet_path: Path,
    output_dir: Path,
) -> dict:
    """Build exploratory traffic summaries from the normalized dataset."""
    df = pl.read_parquet(parquet_path)

    required = {
        "timestamp",
        "sensor_id",
        "flow_vehicles_per_hour",
        "occupancy_pct",
    }
    missing_columns = required - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Normalized dataset is missing columns: {sorted(missing_columns)}"
        )
    if df.is_empty():
        raise ValueError("The normalized dataset is empty.")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert correctly stored UTC instants to Paris local time.
    df = df.with_columns(
        pl.col("timestamp")
        .dt.convert_time_zone("Europe/Paris")
        .alias("timestamp_paris")
    )

    # Traffic flow by hour of day (Paris local time).
    by_hour = (
        df.group_by(
            pl.col("timestamp_paris").dt.hour().alias("hour")
        )
        .agg(
            pl.col("flow_vehicles_per_hour").mean().alias("flow_mean"),
            pl.col("flow_vehicles_per_hour").median().alias("flow_median"),
            pl.col("flow_vehicles_per_hour").count().alias("observations"),
        )
        .sort("hour")
    )
    by_hour.write_csv(output_dir / "traffic_flow_by_hour.csv")

    # Traffic flow by day of week (ISO weekday: Monday=1, Sunday=7).
    by_weekday = (
        df.group_by(
            pl.col("timestamp_paris").dt.weekday().alias("weekday")
        )
        .agg(
            pl.col("flow_vehicles_per_hour").mean().alias("flow_mean"),
            pl.col("flow_vehicles_per_hour").median().alias("flow_median"),
            pl.col("flow_vehicles_per_hour").count().alias("observations"),
        )
        .sort("weekday")
    )
    by_weekday.write_csv(output_dir / "traffic_flow_by_weekday.csv")

    # Missing measurement rates for each sensor.
    by_sensor = (
        df.group_by("sensor_id")
        .agg(
            pl.len().alias("total_rows"),
            pl.col("flow_vehicles_per_hour")
            .is_null()
            .sum()
            .alias("missing_flow_rows"),
            pl.col("occupancy_pct")
            .is_null()
            .sum()
            .alias("missing_occupancy_rows"),
            (
                pl.col("flow_vehicles_per_hour").is_null()
                | pl.col("occupancy_pct").is_null()
            )
            .sum()
            .alias("rows_missing_at_least_one"),
        )
        .with_columns(
            (
                100 * pl.col("missing_flow_rows") / pl.col("total_rows")
            ).round(2).alias("missing_flow_pct"),
            (
                100 * pl.col("missing_occupancy_rows") / pl.col("total_rows")
            ).round(2).alias("missing_occupancy_pct"),
            (
                100 * pl.col("rows_missing_at_least_one")
                / pl.col("total_rows")
            ).round(2).alias("missing_any_pct"),
        )
        .sort("missing_any_pct", descending=True)
    )
    by_sensor.write_csv(output_dir / "missing_values_by_sensor.csv")

    # Summary of the generated data.
    return {
        "rows": df.height,
        "sensors": df["sensor_id"].n_unique(),
        "hour_groups": by_hour.height,
        "weekday_groups": by_weekday.height,
        "sensor_rows": by_sensor.height,
        "output_directory": str(output_dir),
    }
