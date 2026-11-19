from pathlib import Path
import os
from dotenv import load_dotenv


import folium
import geopandas as gpd
import branca.colormap as cm
import json

load_dotenv()

def create_traffic_map(
    geographic_parquet: Path,
    output_html: Path,
) -> Path:
    """Create an interactive map of traffic by road segment."""

    roads = gpd.read_parquet(geographic_parquet)

    if roads.empty:
        raise ValueError("The geographic dataset is empty.")

    if roads.crs is None:
        raise ValueError("The geographic dataset has no CRS.")

    roads = roads.to_crs("EPSG:4326")

    # Only display segments with available traffic measurements.
    mapped = roads.dropna(subset=["mean_flow"]).copy()

    mapped = mapped[
        mapped.geometry.notna() & ~mapped.geometry.is_empty
    ].copy()

    for column in mapped.columns:
        if column == mapped.geometry.name:
            continue

        if mapped[column].dtype == "datetime64[ns]":
            mapped[column] = mapped[column].astype(str)


    if mapped.empty:
        raise ValueError(
            "No geographic segments matched traffic measurements."
        )

    min_flow = float(mapped["mean_flow"].min())
    max_flow = float(mapped["mean_flow"].max())

    # Avoid a zero-width colour scale.
    if min_flow == max_flow:
        max_flow = min_flow + 1

    colors = cm.linear.YlOrRd_09.scale(min_flow, max_flow)
    colors.caption = "Mean traffic flow (vehicles/hour)"

    center = mapped.geometry.union_all().centroid

    carto_api_key = os.getenv("CARTO_API_KEY")

    if not carto_api_key:
        warning = (
            "[yellow]Warning:[/yellow] CARTO_API_KEY not set. "
            "Using default OpenStreetMap tiles instead of CARTO basemap."
        )
        print(warning)
        center = mapped.geometry.union_all().centroid

        traffic_map = folium.Map(
            location=[center.y, center.x],
            zoom_start=12,
            tiles="CartoDB positron",
        )

    else:
        carto_tiles = (
            "https://{s}.basemaps.cartocdn.com/"
            "light_all/{z}/{x}/{y}.png"
            f"?key={carto_api_key}"
        )

        traffic_map = folium.Map(
            location=[center.y, center.x],
            zoom_start=12,
            tiles=carto_tiles,
            attr=(
                '&copy; <a href="https://www.openstreetmap.org/copyright">'
                'OpenStreetMap</a> contributors '
                '&copy; <a href="https://carto.com/attributions">CARTO</a>'
            ),
            subdomains="abcd",
        )

    def style_function(feature):
        value = feature["properties"].get("mean_flow")

        if value is None:
            return {
                "color": "#808080",
                "weight": 2,
                "opacity": 0.4,
            }

        return {
            "color": colors(value),
            "weight": 4,
            "opacity": 0.85,
        }

    tooltip_fields = [
        field
        for field in [
            "iu_ac",
            "libelle",
            "mean_flow",
            "mean_occupancy",
            "total_rows",
            "missing_flow_rate",
            "missing_occupancy_rate",
        ]
        if field in mapped.columns
    ]

    for col in mapped.columns:
        if col != mapped.geometry.name:
            mapped[col] = mapped[col].map(
                lambda value: value.isoformat()
                if hasattr(value, "isoformat")
                else value
            )

    geojson_data = json.loads(
        mapped.to_json(drop_id=True, na="null")
    )

    folium.GeoJson(
        geojson_data,
        name="Traffic by road segment",
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=tooltip_fields,
            aliases=[
                field.replace("_", " ").title()
                for field in tooltip_fields
            ],
            localize=True,
        ),
    ).add_to(traffic_map)

    colors.add_to(traffic_map)
    folium.LayerControl().add_to(traffic_map)

    output_html.parent.mkdir(parents=True, exist_ok=True)
    traffic_map.save(str(output_html))

    return output_html