
import polars as pl

from openmobility.processing.exploratory import (
    build_exploratory_outputs,
)


def test_exploratory_outputs_preserve_missing_measurements(tmp_path):
    source = tmp_path / "sample.parquet"
    output = tmp_path / "eda"

    df = pl.DataFrame(
        {
            "timestamp": [
                "2025-10-06T08:00:00+00:00",
                "2025-10-06T09:00:00+00:00",
                "2025-10-07T08:00:00+00:00",
            ],
            "sensor_id": ["A", "A", "B"],
            "flow_vehicles_per_hour": [100.0, None, 200.0],
            "occupancy_pct": [5.0, 6.0, None],
        }
    ).with_columns(
        pl.col("timestamp").str.to_datetime(time_zone="UTC")
    )

    df.write_parquet(source)

    result = build_exploratory_outputs(source, output)

    assert result["rows"] == 3
    assert result["sensors"] == 2

    sensor_stats = pl.read_csv(output / "missing_values_by_sensor.csv")
    sensor_a = sensor_stats.filter(pl.col("sensor_id") == "A")
    assert sensor_a["missing_flow_rows"][0] == 1
    assert sensor_a["total_rows"][0] == 2

    # The source dataset is not filtered or overwritten.
    assert pl.read_parquet(source).height == 3
