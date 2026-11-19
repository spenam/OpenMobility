from __future__ import annotations

import json
from pathlib import Path

import typer
from rich import print
from rich.table import Table

from openmobility.ingestion.paris_traffic import fetch_records
from openmobility.processing.normalize import inspect_schema, normalize_records

app = typer.Typer(no_args_is_help=True, help="OpenMobility Paris data tools")
RAW = Path("data/raw/paris_traffic.jsonl")
NORMALIZED = Path("data/processed/paris_traffic_normalized.parquet")


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


if __name__ == "__main__":
    app()
