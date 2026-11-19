"""Conservative paginated downloader for the Paris Open Data traffic API."""
from __future__ import annotations

import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import requests

API_BASE = "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets"
DATASET_ID = "comptages-routiers-permanents"
PORTAL_URL = "https://opendata.paris.fr/explore/dataset/comptages-routiers-permanents/"


def fetch_records(
    output: Path,
    max_records: int = 50_000,
    page_size: int = 100,
    pause_seconds: float = 0.05,
    timeout: int = 30,
) -> int:
    """Fetch the newest traffic records, moving backwards by hour."""

    if max_records < 1:
        raise ValueError("max_records must be >= 1")

    if not 1 <= page_size <= 100:
        raise ValueError("page_size must be between 1 and 100")

    output.parent.mkdir(parents=True, exist_ok=True)

    endpoint = f"{API_BASE}/{DATASET_ID}/records"

    session = requests.Session()
    session.headers.update({
        "User-Agent": "OpenMobility/0.1 (open research project)"
    })

    # Find the newest available measurement.
    response = session.get(
        endpoint,
        params={
            "limit": 1,
            "order_by": "t_1h DESC",
        },
        timeout=timeout,
    )
    response.raise_for_status()

    results = response.json().get("results", [])

    if not results:
        raise RuntimeError("No records found in the API")

    latest = datetime.fromisoformat(results[0]["t_1h"])

    # Start at the beginning of the newest measurement hour.
    current_hour = latest.replace(
        minute=0,
        second=0,
        microsecond=0,
    )

    written = 0
    max_offset = 10_000

    with output.open("w", encoding="utf-8") as stream:

        while written < max_records:

            next_hour = current_hour + timedelta(hours=1)

            where = (
                f"t_1h >= '{current_hour.isoformat()}' "
                f"AND t_1h < '{next_hour.isoformat()}'"
            )

            # Count records in this hour.
            response = session.get(
                endpoint,
                params={
                    "limit": 1,
                    "where": where,
                },
                timeout=timeout,
            )
            response.raise_for_status()

            total = response.json().get("total_count", 0)

            if total > max_offset:
                raise RuntimeError(
                    f"Hour {current_hour} contains {total} records. "
                    "Further partitioning is required."
                )

            offset = 0

            while offset < total and written < max_records:

                limit = min(
                    page_size,
                    total - offset,
                    max_records - written,
                    max_offset - offset,
                )

                response = session.get(
                    endpoint,
                    params={
                        "limit": limit,
                        "offset": offset,
                        "where": where,
                        "order_by": "t_1h DESC, iu_ac ASC",
                    },
                    timeout=timeout,
                )
                response.raise_for_status()

                records = response.json().get("results", [])

                if not records:
                    break

                for record in records:
                    stream.write(
                        json.dumps(record, ensure_ascii=False) + "\n"
                    )

                written += len(records)
                offset += len(records)

                print(
                    f"Downloaded {written:,} / {max_records:,} "
                    f"| Current hour: {current_hour.isoformat()}"
                )

                time.sleep(pause_seconds)

            # Move backwards one hour.
            current_hour -= timedelta(hours=1)

            if total == 0:
                print(f"No records for {next_hour.isoformat()}")

    metadata = {
        "dataset_id": DATASET_ID,
        "portal_url": PORTAL_URL,
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "records_downloaded": written,
        "max_records_requested": max_records,
        "page_size": page_size,
        "sort_order": "t_1h DESC",
        "partition": "hourly",
    }

    output.with_suffix(
        output.suffix + ".metadata.json"
    ).write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return written