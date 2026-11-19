from pathlib import Path

import pytest

from openmobility.ingestion.paris_traffic import fetch_records


def test_page_size_is_bounded(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="page_size"):
        fetch_records(tmp_path / "data.jsonl", page_size=101)
