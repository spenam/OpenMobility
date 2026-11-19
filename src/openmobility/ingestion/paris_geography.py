from pathlib import Path

import requests


GEOGRAPHY_URL = (
    "https://opendata.paris.fr/api/explore/v2.1/catalog/"
    "datasets/referentiel-comptages-routiers/exports/geojson"
)


def fetch_geographic_reference(
    output_path: Path,
) -> Path:
    """Download the Paris road-network geographic reference."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(
        GEOGRAPHY_URL,
        timeout=60,
    )
    response.raise_for_status()

    # Check that the response is GeoJSON before saving it.
    payload = response.json()

    if payload.get("type") != "FeatureCollection":
        raise ValueError(
            "The geographic reference response is not a GeoJSON "
            "FeatureCollection."
        )

    if not payload.get("features"):
        raise ValueError("The geographic reference contains no features.")

    output_path.write_text(
        response.text,
        encoding="utf-8",
    )

    return output_path
