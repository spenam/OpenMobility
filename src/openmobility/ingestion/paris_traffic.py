"""Conservative paginated downloader for the Paris Open Data traffic API."""
from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

API_BASE = "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets"
DATASET_ID = "comptages-routiers-permanents"
PORTAL_URL = "https://opendata.paris.fr/explore/dataset/comptages-routiers-permanents/"


def fetch_records(
    output: Path,
    max_records: int = 5000,
    page_size: int = 100,
    pause_seconds: float = 0.15,
    timeout: int = 30,
) -> int:
    """Fetch at most ``max_records`` records into JSONL, preserving source fields."""
    if max_records < 1:
        raise ValueError("max_records must be >= 1")
    if not 1 <= page_size <= 100:
        raise ValueError("page_size must be between 1 and 100")

    output.parent.mkdir(parents=True, exist_ok=True)
    endpoint = f"{API_BASE}/{DATASET_ID}/records"
    session = requests.Session()
    session.headers.update({"User-Agent": "OpenMobility/0.1 (open research project)"})

    written = 0
    offset = 0
    with output.open("w", encoding="utf-8") as stream:
        while written < max_records:
            limit = min(page_size, max_records - written)
            response = session.get(
                endpoint,
                params={"limit": limit, "offset": offset},
                timeout=timeout,
            )
            response.raise_for_status()
            payload: dict[str, Any] = response.json()
            records = payload.get("results", [])
            if not isinstance(records, list):
                raise RuntimeError("Unexpected API response: 'results' is not a list")
            if not records:
                break

            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += len(records)
            offset += len(records)

            total = payload.get("total_count")
            print(f"Downloaded {written:,}" + (f" / {total:,}" if isinstance(total, int) else ""))
            if len(records) < limit:
                break
            time.sleep(pause_seconds)

    metadata = {
        "dataset_id": DATASET_ID,
        "portal_url": PORTAL_URL,
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "records_downloaded": written,
        "max_records_requested": max_records,
        "page_size": page_size,
    }
    output.with_suffix(output.suffix + ".metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return written
