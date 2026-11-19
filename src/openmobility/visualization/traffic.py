
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl


def plot_traffic_summaries(
    output_dir: Path,
    chart_dir: Path,
) -> None:
    chart_dir.mkdir(parents=True, exist_ok=True)

    hourly = pl.read_csv(output_dir / "traffic_flow_by_hour.csv")
    weekdays = pl.read_csv(output_dir / "traffic_flow_by_weekday.csv")

    # Hourly traffic flow.
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        hourly["hour"].to_list(),
        hourly["flow_mean"].to_list(),
        marker="o",
        label="Mean flow",
    )
    ax.plot(
        hourly["hour"].to_list(),
        hourly["flow_median"].to_list(),
        marker="o",
        label="Median flow",
    )
    ax.set(
        title="Traffic flow by hour (Paris local time)",
        xlabel="Hour of day",
        ylabel="Vehicles per hour",
        xticks=list(range(24)),
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(chart_dir / "traffic_flow_by_hour.png", dpi=150)
    plt.close(fig)

    # Weekday traffic flow.
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
    fig.savefig(chart_dir / "traffic_flow_by_weekday.png", dpi=150)
    plt.close(fig)
