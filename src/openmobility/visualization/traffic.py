from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from datetime import datetime


def plot_traffic_summaries(
    output_dir: Path,
    chart_dir: Path,
) -> None:
    chart_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # 1. Hourly traffic flow
    # --------------------------------------------------
    hourly = pl.read_csv(
        output_dir / "traffic_flow_by_hour.csv"
    )

    hourly_rows = {
        int(row["hour"]): row
        for row in hourly.to_dicts()
    }

    hours = list(range(24))

    mean_values = [
        hourly_rows[h]["flow_mean"]
        if h in hourly_rows else float("nan")
        for h in hours
    ]

    median_values = [
        hourly_rows[h]["flow_median"]
        if h in hourly_rows else float("nan")
        for h in hours
    ]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        hours,
        mean_values,
        marker="o",
        label="Mean flow",
    )
    ax.plot(
        hours,
        median_values,
        marker="o",
        label="Median flow",
    )

    ax.set(
        title="Traffic flow by hour (Paris local time)",
        xlabel="Hour of day",
        ylabel="Vehicles per hour",
        xticks=hours,
    )

    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        chart_dir / "traffic_flow_by_hour.png",
        dpi=150,
    )
    plt.close(fig)

    # --------------------------------------------------
    # 2. Weekday traffic flow
    # --------------------------------------------------
    weekdays = pl.read_csv(
        output_dir / "traffic_flow_by_weekday.csv"
    )

    weekday_names = [
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday",
    ]

    weekday_numbers = weekdays["weekday"].to_list()

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        weekday_numbers,
        weekdays["flow_mean"].to_list(),
        marker="o",
        label="Mean flow",
    )
    ax.plot(
        weekday_numbers,
        weekdays["flow_median"].to_list(),
        marker="o",
        label="Median flow",
    )

    ax.set(
        title="Traffic flow by weekday",
        xlabel="Day of week",
        ylabel="Vehicles per hour",
        xticks=list(range(1, 8)),
        xticklabels=weekday_names,
    )

    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        chart_dir / "traffic_flow_by_weekday.png",
        dpi=150,
    )
    plt.close(fig)


def plot_sensor_availability(
    output_dir: Path,
    chart_dir: Path,
    max_sensors: int = 100,
) -> None:
    """Plot measurement completeness for the most observed sensors."""
    chart_dir.mkdir(parents=True, exist_ok=True)

    data = pl.read_csv(
        output_dir / "sensor_hour_availability.csv"
    )

    if data.is_empty():
        raise ValueError("No sensor availability data found.")

    # Rank sensors by number of recorded observations.
    sensor_counts = (
        data.group_by("sensor_id")
        .agg(pl.col("record_count").sum().alias("total"))
        .sort("total", descending=True)
        .head(max_sensors)
    )

    selected_sensors = sensor_counts["sensor_id"].to_list()

    # Sort the observed hours chronologically.
    selected = data.filter(
        pl.col("sensor_id").is_in(selected_sensors)
    ).sort("hour")

    #hour_values = sorted(selected["hour"].unique().to_list())
    hour_values = sorted(selected["hour"].unique().to_list())
    sensor_values = selected_sensors

    sensor_index = {
        sensor: index
        for index, sensor in enumerate(sensor_values)
    }
    hour_index = {
        hour: index
        for index, hour in enumerate(hour_values)
    }

    # NaN = no record for that sensor-hour.
    # 0 = record exists, but at least one measurement is missing.
    # 1 = both flow and occupancy are present.
    matrix = np.full(
        (len(sensor_values), len(hour_values)),
        np.nan,
    )

    for row in selected.to_dicts():
        i = sensor_index[row["sensor_id"]]
        j = hour_index[row["hour"]]

        matrix[i, j] = row["complete_measurement_rate"]

    fig, ax = plt.subplots(
        figsize=(max(10, len(hour_values) * 0.45), 8)
    )

    image = ax.imshow(
        matrix,
        aspect="auto",
        interpolation="none",
        vmin=0,
        vmax=1,
    )

    ax.set_title(
        f"Sensor measurement availability "
        f"(top {len(sensor_values)} sensors)"
    )
    ax.set_xlabel("Hour")
    ax.set_ylabel("Sensor")

    ax.set_xticks(range(len(hour_values)))
    ax.set_xticklabels(
        [
            datetime.strptime(
                hour,
                "%Y-%m-%dT%H:%M:%S.%f%z"
            ).strftime("%m-%d %H:%M")
            for hour in hour_values
        ],
        rotation=45,
        ha="right",
    )

    # Avoid cluttering the y-axis with hundreds of sensor IDs.
    ax.set_yticks([])

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(
        "Complete measurement rate (0 to 1)"
    )

    fig.tight_layout()

    fig.savefig(
        chart_dir / "sensor_availability.png",
        dpi=150,
    )
    plt.close(fig)
