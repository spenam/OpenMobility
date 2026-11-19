from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely import STRtree
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    Point,
)
from shapely.ops import substring


def _line_parts(geometry) -> list[LineString]:
    """Extract usable LineStrings from a geometry."""
    if geometry is None or geometry.is_empty:
        return []

    if isinstance(geometry, LineString):
        return [geometry] if geometry.length > 0 else []

    if isinstance(geometry, MultiLineString):
        return [
            line for line in geometry.geoms
            if not line.is_empty and line.length > 0
        ]

    if isinstance(geometry, GeometryCollection):
        return [
            line
            for item in geometry.geoms
            for line in _line_parts(item)
        ]

    return []


def _union_find_components(
    nodes: set[tuple[int, int]],
    edges: list[tuple[tuple[int, int], tuple[int, int]]],
) -> int:
    """Count connected components in an undirected graph."""
    parent = {node: node for node in nodes}

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left, right):
        root_left = find(left)
        root_right = find(right)

        if root_left != root_right:
            parent[root_right] = root_left

    for left, right in edges:
        union(left, right)

    return len({find(node) for node in nodes})


def _node_key(point: Point, tolerance: float) -> tuple[int, int]:
    """Snap a point to a metric tolerance grid."""
    return (
        round(point.x / tolerance),
        round(point.y / tolerance),
    )


def build_network_topology(
    geojson_path: Path,
    traffic_path: Path,
    output_path: Path,
    *,
    snap_tolerance_m: float = 5.0,
    top_flow_fraction: float = 0.10,
) -> dict:
    """
    Build a road graph from line geometries and summarize traffic coverage.

    The source road features are retained as traffic-bearing segments.
    Intersections split line features into graph edges. Node coordinates
    are snapped to a configurable metric tolerance.

    This is a geometric connectivity model, not a legal or navigability
    model. Grade-separated crossings cannot be distinguished without
    elevation/bridge/tunnel attributes.
    """
    if snap_tolerance_m <= 0:
        raise ValueError("snap_tolerance_m must be positive.")

    if not 0 < top_flow_fraction <= 1:
        raise ValueError("top_flow_fraction must be in (0, 1].")

    roads = gpd.read_file(geojson_path)

    if "iu_ac" not in roads.columns:
        raise ValueError(
            "Road reference must contain the 'iu_ac' identifier."
        )

    roads = roads.loc[
        roads.geometry.notna() & ~roads.geometry.is_empty
    ].copy()

    if roads.empty:
        raise ValueError("No usable road geometries were found.")

    if roads.crs is None:
        roads = roads.set_crs("EPSG:4326")

    # Use a metric CRS for snapping and length calculations.
    roads = roads.to_crs("EPSG:2154")
    roads["iu_ac"] = roads["iu_ac"].astype("string")

    feature_lines: list[tuple[str, LineString]] = []

    for _, row in roads.iterrows():
        identifier = str(row["iu_ac"])

        for line in _line_parts(row.geometry):
            feature_lines.append((identifier, line))

    if not feature_lines:
        raise ValueError(
            "The geographic reference contains no usable line features."
        )

    lines = [line for _, line in feature_lines]
    tree = STRtree(lines)

    # Record intersection points for each source line.
    cut_points: list[list[Point]] = [
        [Point(line.coords[0]), Point(line.coords[-1])]
        for line in lines
    ]

    for index, line in enumerate(lines):
        candidate_indices = tree.query(line, predicate="intersects")

        for candidate in candidate_indices:
            other_index = int(candidate)

            if other_index <= index:
                continue

            other = lines[other_index]
            intersection = line.intersection(other)

            if intersection.is_empty:
                continue

            if isinstance(intersection, Point):
                cut_points[index].append(intersection)
                cut_points[other_index].append(intersection)

            elif hasattr(intersection, "geoms"):
                # For overlapping lines, their boundary points can
                # establish shared graph nodes.
                for part in intersection.geoms:
                    if isinstance(part, Point):
                        cut_points[index].append(part)
                        cut_points[other_index].append(part)
                    elif isinstance(part, LineString):
                        for coordinate in (
                            part.coords[0],
                            part.coords[-1],
                        ):
                            point = Point(coordinate)
                            cut_points[index].append(point)
                            cut_points[other_index].append(point)

    graph_edges = []
    edge_records = []
    node_degrees: Counter = Counter()
    nodes: set[tuple[int, int]] = set()

    for (identifier, line), points in zip(
        feature_lines,
        cut_points,
        strict=True,
    ):
        positions = sorted({
            max(0.0, min(line.length, line.project(point)))
            for point in points
        })

        for start, end in zip(positions, positions[1:]):
            if end - start <= 0.01:
                continue

            segment = substring(line, start, end)

            if segment.is_empty or segment.length <= 0:
                continue

            start_node = _node_key(
                Point(segment.coords[0]),
                snap_tolerance_m,
            )
            end_node = _node_key(
                Point(segment.coords[-1]),
                snap_tolerance_m,
            )

            if start_node == end_node:
                continue

            nodes.update((start_node, end_node))
            node_degrees[start_node] += 1
            node_degrees[end_node] += 1

            graph_edges.append((start_node, end_node))
            edge_records.append({
                "sensor_id": identifier,
                "length_m": float(segment.length),
                "start_node": start_node,
                "end_node": end_node,
            })

    if not graph_edges:
        raise ValueError(
            "No graph edges could be derived from the road geometries."
        )

    # Aggregate observed traffic by sensor/road identifier.
    traffic = pd.read_parquet(traffic_path)

    required = {
        "sensor_id",
        "flow_vehicles_per_hour",
        "occupancy_pct",
    }
    missing = required - set(traffic.columns)

    if missing:
        raise ValueError(
            f"Traffic data is missing columns: {sorted(missing)}"
        )

    traffic["sensor_id"] = traffic["sensor_id"].astype("string")

    traffic_summary = (
        traffic.groupby("sensor_id", dropna=False)
        .agg(
            mean_flow=("flow_vehicles_per_hour", "mean"),
            mean_occupancy=("occupancy_pct", "mean"),
            observation_count=("sensor_id", "size"),
        )
        .reset_index()
    )

    traffic_by_sensor = {
        str(row.sensor_id): row
        for row in traffic_summary.itertuples(index=False)
    }

    matched_edges = 0
    matched_flows = []

    for edge in edge_records:
        traffic_row = traffic_by_sensor.get(edge["sensor_id"])

        if traffic_row is None:
            edge["mean_flow"] = None
            edge["mean_occupancy"] = None
            edge["observation_count"] = 0
            continue

        edge["mean_flow"] = (
            float(traffic_row.mean_flow)
            if pd.notna(traffic_row.mean_flow)
            else None
        )
        edge["mean_occupancy"] = (
            float(traffic_row.mean_occupancy)
            if pd.notna(traffic_row.mean_occupancy)
            else None
        )
        edge["observation_count"] = int(
            traffic_row.observation_count
        )

        matched_edges += 1

        if edge["mean_flow"] is not None:
            matched_flows.append(edge["mean_flow"])

    flow_threshold = (
        float(pd.Series(matched_flows).quantile(
            1 - top_flow_fraction
        ))
        if matched_flows
        else None
    )

    high_flow_edges = [
        edge
        for edge in edge_records
        if (
            flow_threshold is not None
            and edge["mean_flow"] is not None
            and edge["mean_flow"] >= flow_threshold
        )
    ]

    degree_counts = Counter(node_degrees.values())

    components = _union_find_components(nodes, graph_edges)

    summary = {
        "source_features": int(len(roads)),
        "source_line_parts": int(len(feature_lines)),
        "graph_nodes": int(len(nodes)),
        "graph_edges": int(len(graph_edges)),
        "connected_components": int(components),
        "dead_end_nodes": int(degree_counts.get(1, 0)),
        "intersection_nodes": int(
            sum(count for degree, count in degree_counts.items()
                if degree >= 3)
        ),
        "maximum_node_degree": (
            int(max(node_degrees.values()))
            if node_degrees else 0
        ),
        "traffic_sensors": int(len(traffic_summary)),
        "matched_graph_edges": int(matched_edges),
        "unmatched_graph_edges": int(
            len(edge_records) - matched_edges
        ),
        "matched_edge_percentage": round(
            100 * matched_edges / len(edge_records), 2
        ),
        "mean_flow_threshold_for_high_flow_segments": flow_threshold,
        "high_flow_segment_count": len(high_flow_edges),
        "snap_tolerance_m": snap_tolerance_m,
        "top_flow_fraction": top_flow_fraction,
        "high_flow_segments": [
            {
                "sensor_id": edge["sensor_id"],
                "mean_flow": edge["mean_flow"],
                "mean_occupancy": edge["mean_occupancy"],
                "length_m": round(edge["length_m"], 2),
                "start_node_degree": node_degrees[
                    edge["start_node"]
                ],
                "end_node_degree": node_degrees[
                    edge["end_node"]
                ],
            }
            for edge in sorted(
                high_flow_edges,
                key=lambda item: item["mean_flow"] or 0,
                reverse=True,
            )[:50]
        ],
        "limitations": [
            "Geometric crossings may be grade-separated; "
            "bridge and tunnel attributes are not used.",
            "Connectivity depends on the configured endpoint "
            "snapping tolerance.",
            "Traffic is joined by iu_ac/sensor_id identifier "
            "and may not represent every road segment.",
            "High flow is not, by itself, proof of congestion "
            "or a network bottleneck.",
            "Connected components describe geometric connectivity, "
            "not legal vehicle access or travel direction.",
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )

    return {
        **summary,
        "output_path": str(output_path),
    }