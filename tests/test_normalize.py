import json
from pathlib import Path

import pytest

from openmobility.processing.normalize import normalize_records


def test_normalize_minimal_source(tmp_path: Path) -> None:
    source = tmp_path / "sample.jsonl"
    output = tmp_path / "normalized.parquet"
    rows = [
        {"t_1h": "2025-01-01T08:00:00", "iu_ac": "sensor-1",
         "debit_horaire": 120, "taux_occupation": 32.5, "etat_trafic": "Fluide"},
        {"t_1h": "2025-01-01T09:00:00", "iu_ac": "sensor-1",
         "debit_horaire": 160, "taux_occupation": 48.0, "etat_trafic": "Pré-saturé"},
    ]
    source.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    result = normalize_records(source, output)
    assert result.height == 2
    assert {
        "timestamp", 
        "sensor_id", 
        "flow_vehicles_per_hour", 
        "occupancy_pct"
    } <= set(result.columns)
    assert output.exists()


def test_normalize_fails_loudly_on_unknown_schema(tmp_path: Path) -> None:
    source = tmp_path / "unknown.jsonl"
    source.write_text(json.dumps({"foo": "bar", "baz": 1}) + "\n")
    with pytest.raises(ValueError, match="Could not confidently identify"):
        normalize_records(source, tmp_path / "out.parquet")
