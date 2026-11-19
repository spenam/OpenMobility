
from pathlib import Path

import geopandas as gpd
import polars as pl


def build_geographic_dataset(
    geojson_path: Path,
    traffic_path: Path,
    output_path: Path,
) -> dict:
    """Join road geometry with aggregated traffic measurements."""

    # Read geographic reference.
    roads = gpd.read_file(geojson_path)

    if "iu_ac" not in roads.columns:
        raise ValueError(
            "Geographic reference does not contain 'iu_ac'. "
            f"Available columns: {list(roads.columns)}"
        )

    if roads.geometry.is_empty.all():
        raise ValueError("Geographic reference contains no geometry.")

    roads["iu_ac"] = roads["iu_ac"].astype("string")

    # GeoJSON coordinates are expected in WGS84.
    if roads.crs is None:
        roads = roads.set_crs("EPSG:4326")

    roads = roads.to_crs("EPSG:4326")

    # Read normalized traffic data.
    traffic = pl.read_parquet(traffic_path)

    required = {
        "sensor_id",
        "flow_vehicles_per_hour",
        "occupancy_pct",
    }
    missing = required - set(traffic.columns)

    if missing:
        raise ValueError(
            f"Traffic data is missing columns: {sorted(missing)}"
        )

    # Aggregate measurements by sensor.
    summary = (
        traffic.group_by("sensor_id")
        .agg(
            pl.len().alias("total_rows"),
            pl.col("flow_vehicles_per_hour")
            .mean()
            .alias("mean_flow"),
            pl.col("occupancy_pct")
            .mean()
            .alias("mean_occupancy"),
            pl.col("flow_vehicles_per_hour")
            .is_null()
            .mean()
            .alias("missing_flow_rate"),
            pl.col("occupancy_pct")
            .is_null()
            .mean()
            .alias("missing_occupancy_rate"),
        )
    )

    summary_pd = summary.to_pandas()
    summary_pd["sensor_id"] = summary_pd["sensor_id"].astype("string")

    # Join traffic to geographic road segments.
    result = roads.merge(
        summary_pd,
        left_on="iu_ac",
        right_on="sensor_id",
        how="left",
        indicator=True,
    )

    matched = int((result["_merge"] == "both").sum())
    unmatched = int((result["_merge"] == "left_only").sum())

    result = result.drop(columns=["_merge"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)

    return {
        "geographic_segments": len(roads),
        "matched_segments": matched,
        "unmatched_geographic_segments": unmatched,
        "traffic_sensors": summary.height,
        "matched_percentage": round(
            100 * matched / len(roads), 2
        ) if len(roads) else 0,
        "output_path": str(output_path),
    }
