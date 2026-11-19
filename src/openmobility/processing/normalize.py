"""Schema inspection and conservative normalization of raw Paris traffic records."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import polars as pl

DATE_CANDIDATES = (
    "t_1h", "date_heure", "dateheure", "date_et_heure_de_comptage",
    "horodate", "timestamp", "datetime", "date",
)
SENSOR_CANDIDATES = (
    "iu_ac", "identifiant_capteur", "id_capteur", "identifiant_station",
    "id_station", "station_id", "sensor_id", "id",
)
FLOW_CANDIDATES = (
    "debit_horaire", "debit", "comptage", "nombre_vehicules",
    "nb_vehicules", "flow", "traffic_flow" "volume", "q"
)
OCCUPANCY_CANDIDATES = (
    "taux_occupation", "occupation", "occupancy", "occupancy_rate", "taux_occup",
)
STATE_CANDIDATES = ("etat_trafic", "etat", "traffic_state", "etat_de_la_circulation")


def _key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _find_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    normalized = {_key(column): column for column in columns}
    for candidate in candidates:
        if _key(candidate) in normalized:
            return normalized[_key(candidate)]
    # A conservative substring fallback for French source field naming variations.
    for candidate in candidates:
        needle = _key(candidate)
        for norm, original in normalized.items():
            if len(needle) >= 5 and (needle in norm or norm in needle):
                return original
    return None


def load_jsonl(path: Path) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_number}") from exc
    if not rows:
        raise ValueError(f"No records found in {path}")
    return pl.from_dicts(rows, infer_schema_length=None, strict=False)


def inspect_schema(path: Path) -> dict[str, Any]:
    frame = load_jsonl(path)
    return {
        "rows": frame.height,
        "columns": frame.columns,
        "dtypes": {name: str(dtype) for name, dtype in frame.schema.items()},
        "sample": frame.head(3).to_dicts(),
    }


def normalize_records(input_path: Path, output_path: Path) -> pl.DataFrame:
    frame = load_jsonl(input_path)
    columns = frame.columns

    date_col = _find_column(columns, DATE_CANDIDATES)
    sensor_col = _find_column(columns, SENSOR_CANDIDATES)
    flow_col = _find_column(columns, FLOW_CANDIDATES)
    occupancy_col = _find_column(columns, OCCUPANCY_CANDIDATES)
    state_col = _find_column(columns, STATE_CANDIDATES)

    if date_col is None or sensor_col is None:
        raise ValueError(
            "Could not confidently identify timestamp and sensor ID columns.\n"
            f"Available columns: {columns}\n"
            "Run `openmobility inspect`, then add the exact source field names to "
            "DATE_CANDIDATES / SENSOR_CANDIDATES in processing/normalize.py."
        )
    if flow_col is None and occupancy_col is None:
        raise ValueError(
            "Could not identify a flow or occupancy column. "
            f"Available columns: {columns}. Inspect the source schema and update candidates."
        )

    expressions = [
        pl.col(date_col).cast(pl.String).str.to_datetime(strict=False, time_zone="UTC").alias("timestamp"),
        pl.col(sensor_col).cast(pl.String).alias("sensor_id"),
    ]
    if flow_col:
        expressions.append(pl.col(flow_col).cast(pl.Float64, strict=False).alias("flow_vehicles_per_hour"))
    if occupancy_col:
        expressions.append(pl.col(occupancy_col).cast(pl.Float64, strict=False).alias("occupancy_pct"))
    if state_col:
        expressions.append(pl.col(state_col).cast(pl.String).alias("traffic_state"))

    normalized = frame.select(expressions).filter(
        pl.col("timestamp").is_not_null() & pl.col("sensor_id").is_not_null()
    )
    if normalized.is_empty():
        raise ValueError("Normalization produced zero valid rows; inspect timestamp formats.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.write_parquet(output_path, compression="zstd")
    return normalized
