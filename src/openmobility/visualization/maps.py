from pathlib import Path

import folium
import geopandas as gpd
import branca.colormap as cm


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

    traffic_map = folium.Map(
        location=[center.y, center.x],
        zoom_start=12,
        tiles="CartoDB positron",
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

    folium.GeoJson(
        mapped,
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