from __future__ import annotations

import json
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich import print
from rich.console import Console
from rich.table import Table

from openmobility.ingestion.paris_geography import (
    fetch_geographic_reference,
)
from openmobility.ingestion.paris_traffic import fetch_records
from openmobility.intelligence.agents.runners import (
    build_agent_runners,
)
from openmobility.intelligence.graph import (
    build_intelligence_graph,
)
from openmobility.intelligence.state import (
    create_initial_state,
)
from openmobility.processing.evaluation import run_evaluation
from openmobility.processing.exploratory import build_exploratory_outputs
from openmobility.processing.geography import (
    build_geographic_dataset,
)
from openmobility.processing.normalize import inspect_schema, normalize_records
from openmobility.processing.predictive import run_predictive_analytics
from openmobility.processing.quality import build_quality_report
from openmobility.visualization.maps import create_traffic_map
from openmobility.visualization.traffic import (
    plot_sensor_availability,
    plot_traffic_summaries,
)

load_dotenv()
console = Console()

app = typer.Typer(no_args_is_help=True, help="OpenMobility Paris data tools")
RAW = Path("data/raw/paris_traffic.jsonl")
NORMALIZED = Path("data/processed/paris_traffic_normalized.parquet")
GEO_RAW = Path("data/raw/paris_sensor_geography.geojson")
GEO_PROCESSED = Path("data/processed/paris_sensor_geography.parquet")
TRAFFIC_MAP = Path("reports/eda/sensor_traffic_map.html")
HOLDOUT_PREDICTIONS = Path("reports/forecasting/holdout_predictions.csv")


@app.command()
def fetch(
    max_records: int = typer.Option(5000, min=1, help="Maximum number of records to download."),
    page_size: int = typer.Option(100, min=1, max=100, help="API page size (max 100)."),
) -> None:
    """Download a bounded sample from the Paris permanent traffic counters API."""
    count = fetch_records(RAW, max_records=max_records, page_size=page_size)
    print(f"[green]Saved {count:,} records to {RAW}[/green]")


@app.command()
def inspect() -> None:
    """Print source columns, inferred types and a few raw records."""
    if not RAW.exists():
        raise typer.BadParameter(f"{RAW} not found. Run `openmobility fetch` first.")
    report = inspect_schema(RAW)
    print(f"[bold]Rows:[/bold] {report['rows']:,}")
    table = Table("Column", "Inferred type")
    for column, dtype in report["dtypes"].items():
        table.add_row(column, dtype)
    print(table)
    print("[bold]Sample records:[/bold]")
    print(json.dumps(report["sample"], ensure_ascii=False, indent=2, default=str))


@app.command()
def preprocess() -> None:
    """Normalize source records to a compact Parquet table."""
    if not RAW.exists():
        raise typer.BadParameter(f"{RAW} not found. Run `openmobility fetch` first.")
    frame = normalize_records(RAW, NORMALIZED)
    print(f"[green]Wrote {frame.height:,} normalized rows to {NORMALIZED}[/green]")
    print(frame.head(5))

@app.command()
def quality() -> None:
    """Generate a data-quality report."""
    if not NORMALIZED.exists():
        raise typer.BadParameter(
            "Normalized dataset not found. Run `openmobility preprocess` first."
        )

    report = build_quality_report(NORMALIZED)
    print(json.dumps(report, indent=2, default=str))

@app.command()
def explore(
    output_dir: Path = Path("reports/eda"),
) -> None:
    """Generate exploratory traffic summaries and charts."""
    if not NORMALIZED.exists():
        raise typer.BadParameter(
            f"Normalized dataset not found: {NORMALIZED}. Run preprocess first."
        )

    summary = build_exploratory_outputs(NORMALIZED, output_dir)

    chart_dir = output_dir / "charts"
    plot_traffic_summaries(output_dir, chart_dir)
    plot_sensor_availability(output_dir, chart_dir)

    print("[bold green]Exploratory analysis completed.[/bold green]")
    print(json.dumps(summary, indent=2))
    print(f"Charts saved to: {chart_dir}")

@app.command()
def fetch_geography() -> None:
    """Download the Paris road-network geographic reference."""
    path = fetch_geographic_reference(GEO_RAW)
    print(f"[green]Geographic reference downloaded:[/green] {path}")


@app.command()
def map_traffic() -> None:
    """Join traffic measurements to road geometry and create a map."""
    if not GEO_RAW.exists():
        raise typer.BadParameter(
            "Geographic reference not found. Run fetch-geography first."
        )

    if not NORMALIZED.exists():
        raise typer.BadParameter(
            "Normalized traffic data not found. Run preprocess first."
        )

    summary = build_geographic_dataset(
        GEO_RAW,
        NORMALIZED,
        GEO_PROCESSED,
    )

    print("[bold]Geographic join summary[/bold]")
    print(json.dumps(summary, indent=2))

    if summary["matched_segments"] == 0:
        raise typer.BadParameter(
            "No geographic segments matched traffic sensors. "
            "Check the sensor ID formats."
        )

    output = create_traffic_map(
        GEO_PROCESSED,
        TRAFFIC_MAP,
    )

    print(f"[green]Traffic map created:[/green] {output}")

@app.command()
def predict(
    output_dir: Path = Path("reports/forecasting"),
    horizon: int = typer.Option(24, min=1, max=168),
    min_sensor_rows: int = typer.Option(40, min=20),
    verbose: bool = typer.Option(False, help="Print detailed progress messages."),
) -> None:
    """Train/evaluate XGBoost traffic models and detect anomalies."""
    if not NORMALIZED.exists():
        raise typer.BadParameter(
            f"Normalized dataset not found: {NORMALIZED}. Run preprocess first."
        )
    summary = run_predictive_analytics(
        NORMALIZED,
        output_dir=output_dir,
        min_sensor_rows=min_sensor_rows,
        horizon=horizon,
    )
    if verbose:
        print("[bold]Predictive analytics summary[/bold]")
        print(json.dumps(summary, indent=2))

@app.command()
def evaluate(
    output_dir: Path = Path("reports/evaluation"),
    min_sensor_rows: int = typer.Option(40, min=20),
    verbose: bool = typer.Option(False, help="Print detailed progress messages."),
) -> None:
    """Evaluate XGBoost traffic models and compare to baseline."""
    if not HOLDOUT_PREDICTIONS.exists():
        raise typer.BadParameter(
            f"Predictions dataset not found: {HOLDOUT_PREDICTIONS}. Run preprocess first."
        )
    summary = run_evaluation(
        HOLDOUT_PREDICTIONS,
        output_dir=output_dir,
    )
    if verbose:
        print("[bold]Evaluation summary[/bold]")
        print(json.dumps(summary, indent=2))

@app.command()
def intelligence(
    question: str = typer.Argument(
        ...,
        help="Question to investigate using OpenMobility data.",
    ),
    max_iterations: int = typer.Option(4, min=1, max=4),
    max_tool_calls: int = typer.Option(12, min=1, max=20),
    horizon: int = typer.Option(24, min=1, max=168),
    refresh: bool = typer.Option(
        False,
        help="Force predictive analytics to retrain.",
    ),
) -> None:
    """Run an autonomous multi-agent mobility investigation."""
    state = create_initial_state(
        question,
        investigation_scope={
            "city": "Paris",
            "forecast_horizon_hours": horizon,
            "refresh_analytics": refresh,
        },
        max_iterations=max_iterations,
        max_tool_calls=max_tool_calls,
    )

    graph = build_intelligence_graph(
        build_agent_runners(),
    )

    result = graph.invoke(
        state,
        config={"recursion_limit": 20},
    )

    console.print("\n[bold yellow]Specialist diagnostics[/bold yellow]")

    for finding in result.get("specialist_findings", []):
        console.print_json(
            data=finding.model_dump()
            if hasattr(finding, "model_dump")
            else finding
        )

    if result.get("errors"):
        console.print("\n[bold red]Investigation errors[/bold red]")
        for error in result["errors"]:
            console.print(str(error))

    if result.get("final_report"):
        print("[bold green]Intelligence report[/bold green]")
        print(
            json.dumps(
                result["final_report"],
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )
    else:
        print("[bold red]Investigation did not complete[/bold red]")
        print("Status:", result.get("status"))
        print("Errors:", result.get("errors", []))



if __name__ == "__main__":
    app()
