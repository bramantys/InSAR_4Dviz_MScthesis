#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_build_footprints.py

InSAR4D RUM Viewer pipeline step 03.

Purpose
-------
Build correct RUM cell footprints from RUM center points.

Important geometry principle
----------------------------
Do NOT make RUM cells by adding degree offsets to lon/lat.

Correct approach:
  1. use x_source / y_source in the configured projected source CRS
  2. build square RUM cells in projected metre coordinates using rum_size_m
  3. transform every corner to WGS84 lon/lat for Cesium-facing products

This is the step that fixes the "skewed RUM cell" problem.

Inputs
------
  prepared_inputs.points_geojson
    Created by step 01. Must contain:
      - geometry.coordinates = [lon, lat]
      - properties.rum_id
      - properties.x_source
      - properties.y_source

Config
------
  user_inputs.source_crs
  user_inputs.rum_size_m
  user_inputs.expected_rum_count

Outputs
-------
  generated_outputs.rum_footprints

Output contract
---------------
GeoJSON FeatureCollection with Polygon features. Each feature contains:
  properties:
    rum_id
    source_row
    x_center
    y_center
    lon_center
    lat_center
    grid_i
    grid_j
    rum_size_m
    area_m2
    all original useful velocity/uncertainty properties

  geometry:
    WGS84 polygon corners, closed ring

Top-level metadata contains:
  bbox_wgs84
  bbox_source
  grid summary
  rum count
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from pyproj import CRS, Transformer
except ImportError as exc:
    raise ImportError(
        "pyproj is required for coordinate transformation. "
        "Install it with: pip install pyproj"
    ) from exc

from pipeline_config import load_resolved_config, resolve_path


# =============================================================================
# ADVANCED SETTINGS
# =============================================================================
# Normal users should edit config/project_config.json, not this script.

ROUND_LON_LAT_DIGITS = 8
ROUND_SOURCE_XY_DIGITS = 4
ROUND_AREA_DIGITS = 3

# Grid index tolerance as a fraction of rum_size_m. This is only for warnings.
GRID_INDEX_RESIDUAL_WARN_FRACTION = 0.15

# Whether to write projected/source CRS corners into properties.
# Useful for diagnostics but makes the JSON larger.
STORE_SOURCE_CORNERS = True

# Whether to preserve most source point properties in each footprint.
# Keep true for downstream horizontal/uncertainty steps.
PRESERVE_SOURCE_PROPERTIES = True


# =============================================================================
# PRINT HELPERS
# =============================================================================

def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def ok(msg: str) -> None:
    print(f"  [OK]   {msg}")


def warn(msg: str) -> None:
    print(f"  [WARN] {msg}")


def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


# =============================================================================
# BASIC HELPERS
# =============================================================================

def safe_float(value: Any, fallback: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return fallback
        out = float(value)
        if not math.isfinite(out):
            return fallback
        return out
    except Exception:
        return fallback


def clean_json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    return value


def load_geojson(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input GeoJSON: {path}")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("type") != "FeatureCollection":
        raise ValueError(f"Expected GeoJSON FeatureCollection: {path}")
    return payload


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))


def bbox_from_xy(xs: List[float], ys: List[float]) -> Dict[str, float]:
    return {
        "min_x": round(min(xs), ROUND_SOURCE_XY_DIGITS),
        "min_y": round(min(ys), ROUND_SOURCE_XY_DIGITS),
        "max_x": round(max(xs), ROUND_SOURCE_XY_DIGITS),
        "max_y": round(max(ys), ROUND_SOURCE_XY_DIGITS),
    }


def bbox_from_lonlat(lons: List[float], lats: List[float]) -> Dict[str, float]:
    return {
        "west": round(min(lons), ROUND_LON_LAT_DIGITS),
        "south": round(min(lats), ROUND_LON_LAT_DIGITS),
        "east": round(max(lons), ROUND_LON_LAT_DIGITS),
        "north": round(max(lats), ROUND_LON_LAT_DIGITS),
    }


def transform_ring_to_wgs84(
    transformer: Transformer,
    ring_xy: List[Tuple[float, float]],
) -> List[List[float]]:
    ring_lonlat: List[List[float]] = []
    for x, y in ring_xy:
        lon, lat = transformer.transform(float(x), float(y))
        ring_lonlat.append([
            round(float(lon), ROUND_LON_LAT_DIGITS),
            round(float(lat), ROUND_LON_LAT_DIGITS),
        ])
    return ring_lonlat


def polygon_area_shoelace(ring_xy: List[Tuple[float, float]]) -> float:
    # Assumes closed or open ring; handles both.
    pts = ring_xy[:-1] if ring_xy and ring_xy[0] == ring_xy[-1] else ring_xy
    if len(pts) < 3:
        return 0.0

    area2 = 0.0
    for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]):
        area2 += x1 * y2 - x2 * y1
    return abs(area2) / 2.0


# =============================================================================
# RUM GEOMETRY
# =============================================================================

def build_square_ring_xy(x: float, y: float, size_m: float) -> List[Tuple[float, float]]:
    """
    Build an axis-aligned RUM square in projected source CRS metres.

    Ring order: SW, SE, NE, NW, SW.
    """
    h = size_m / 2.0
    return [
        (x - h, y - h),
        (x + h, y - h),
        (x + h, y + h),
        (x - h, y + h),
        (x - h, y - h),
    ]


def infer_grid_indices(
    xs: List[float],
    ys: List[float],
    spacing_m: float,
) -> Tuple[List[int], List[int], Dict[str, Any]]:
    """
    Infer grid_i/grid_j by snapping source x/y centers to a spacing-based grid.

    This assumes the RUM grid is aligned with the projected source CRS axes.
    That is appropriate for the current RUM product contract:
      x_rum/y_rum are projected metre coordinates and rum_size_m gives cell size.
    """
    if spacing_m <= 0:
        raise ValueError("spacing_m must be positive")

    min_x = min(xs)
    min_y = min(ys)

    grid_i: List[int] = []
    grid_j: List[int] = []
    residual_x: List[float] = []
    residual_y: List[float] = []

    for x, y in zip(xs, ys):
        i = int(round((x - min_x) / spacing_m))
        j = int(round((y - min_y) / spacing_m))
        snapped_x = min_x + i * spacing_m
        snapped_y = min_y + j * spacing_m

        grid_i.append(i)
        grid_j.append(j)
        residual_x.append(abs(x - snapped_x))
        residual_y.append(abs(y - snapped_y))

    max_rx = max(residual_x) if residual_x else 0.0
    max_ry = max(residual_y) if residual_y else 0.0
    warn_threshold = spacing_m * GRID_INDEX_RESIDUAL_WARN_FRACTION

    summary = {
        "origin_x": round(min_x, ROUND_SOURCE_XY_DIGITS),
        "origin_y": round(min_y, ROUND_SOURCE_XY_DIGITS),
        "spacing_m": spacing_m,
        "min_i": min(grid_i),
        "max_i": max(grid_i),
        "min_j": min(grid_j),
        "max_j": max(grid_j),
        "n_cols": max(grid_i) - min(grid_i) + 1,
        "n_rows": max(grid_j) - min(grid_j) + 1,
        "max_snap_residual_x_m": round(max_rx, ROUND_SOURCE_XY_DIGITS),
        "max_snap_residual_y_m": round(max_ry, ROUND_SOURCE_XY_DIGITS),
        "snap_residual_warning_threshold_m": round(warn_threshold, ROUND_SOURCE_XY_DIGITS),
        "has_large_snap_residual": bool(max(max_rx, max_ry) > warn_threshold),
    }

    return grid_i, grid_j, summary


def extract_centers(features: List[Dict[str, Any]]) -> Tuple[List[float], List[float], List[float], List[float]]:
    xs: List[float] = []
    ys: List[float] = []
    lons: List[float] = []
    lats: List[float] = []

    for idx, feature in enumerate(features):
        props = feature.get("properties") or {}
        geom = feature.get("geometry") or {}
        coords = geom.get("coordinates") or []

        x = safe_float(props.get("x_source"))
        y = safe_float(props.get("y_source"))
        if x is None or y is None:
            raise ValueError(f"Feature {idx} missing x_source/y_source")

        if geom.get("type") != "Point" or len(coords) < 2:
            raise ValueError(f"Feature {idx} has invalid point geometry")

        lon = safe_float(coords[0])
        lat = safe_float(coords[1])
        if lon is None or lat is None:
            raise ValueError(f"Feature {idx} has invalid lon/lat coordinates")

        xs.append(x)
        ys.append(y)
        lons.append(lon)
        lats.append(lat)

    return xs, ys, lons, lats


def expected_count_warning(actual: int, expected: Any) -> None:
    if expected is None:
        return
    try:
        exp = int(expected)
    except Exception:
        warn(f"Expected RUM count is not numeric: {expected}")
        return

    if actual != exp:
        warn(f"RUM count mismatch: actual={actual}, expected={exp}")
    else:
        ok(f"RUM count matches expected value: {actual}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    t_start = time.time()

    cfg = load_resolved_config(__file__)
    project_root = Path(cfg["_resolved"]["project_root"])

    prepared = cfg["prepared_inputs"]
    generated = cfg["generated_outputs"]
    user_inputs = cfg["user_inputs"]

    points_path = resolve_path(project_root, prepared["points_geojson"])
    footprints_path = resolve_path(project_root, generated["rum_footprints"])

    source_crs_value = user_inputs["source_crs"]
    rum_size_m = float(user_inputs["rum_size_m"])
    expected_rum_count = user_inputs.get("expected_rum_count")

    section("Configuration")
    print(f"  Project root       : {project_root}")
    print(f"  Points input       : {points_path}")
    print(f"  Footprints output  : {footprints_path}")
    print(f"  Source CRS         : {source_crs_value}")
    print(f"  RUM size           : {rum_size_m} m")

    if rum_size_m <= 0:
        raise ValueError("rum_size_m must be positive")

    source_crs = CRS.from_user_input(source_crs_value)
    transformer_to_wgs84 = Transformer.from_crs(source_crs, CRS.from_epsg(4326), always_xy=True)

    section("Loading RUM centers")
    points_geojson = load_geojson(points_path)
    point_features = points_geojson.get("features", [])
    if not point_features:
        raise ValueError("No point features found")

    xs, ys, center_lons, center_lats = extract_centers(point_features)

    ok(f"Loaded {len(point_features)} center points")
    expected_count_warning(len(point_features), expected_rum_count)

    source_center_bbox = bbox_from_xy(xs, ys)
    wgs84_center_bbox = bbox_from_lonlat(center_lons, center_lats)
    print(f"  Center bbox source : {source_center_bbox}")
    print(f"  Center bbox WGS84  : {wgs84_center_bbox}")

    section("Inferring grid indices")
    grid_i, grid_j, grid_summary = infer_grid_indices(xs, ys, rum_size_m)

    ok(
        f"Grid inferred: {grid_summary['n_cols']} cols × "
        f"{grid_summary['n_rows']} rows from spacing {rum_size_m} m"
    )
    print(f"  Grid i range       : {grid_summary['min_i']} → {grid_summary['max_i']}")
    print(f"  Grid j range       : {grid_summary['min_j']} → {grid_summary['max_j']}")
    print(f"  Max snap residual  : x={grid_summary['max_snap_residual_x_m']} m, y={grid_summary['max_snap_residual_y_m']} m")

    if grid_summary["has_large_snap_residual"]:
        warn(
            "Large grid snap residual detected. "
            "Check source CRS / rum_size_m / whether the RUM grid is rotated."
        )

    section("Building projected-square WGS84 footprints")
    output_features: List[Dict[str, Any]] = []
    all_corner_x: List[float] = []
    all_corner_y: List[float] = []
    all_corner_lon: List[float] = []
    all_corner_lat: List[float] = []

    occupied_grid = set()
    duplicate_grid_cells = 0

    for idx, feature in enumerate(point_features):
        props_in = feature.get("properties") or {}
        geom = feature.get("geometry") or {}
        coords = geom.get("coordinates") or []

        rum_id = str(props_in.get("rum_id", f"RUM_{idx + 1:06d}"))
        x = xs[idx]
        y = ys[idx]
        lon_center = center_lons[idx]
        lat_center = center_lats[idx]
        gi = grid_i[idx]
        gj = grid_j[idx]

        grid_key = (gi, gj)
        if grid_key in occupied_grid:
            duplicate_grid_cells += 1
        occupied_grid.add(grid_key)

        ring_xy = build_square_ring_xy(x, y, rum_size_m)
        ring_lonlat = transform_ring_to_wgs84(transformer_to_wgs84, ring_xy)
        area_m2 = polygon_area_shoelace(ring_xy)

        for cx, cy in ring_xy[:-1]:
            all_corner_x.append(cx)
            all_corner_y.append(cy)
        for clon, clat in ring_lonlat[:-1]:
            all_corner_lon.append(clon)
            all_corner_lat.append(clat)

        if PRESERVE_SOURCE_PROPERTIES:
            props_out = {
                str(k): clean_json_value(v)
                for k, v in props_in.items()
                if clean_json_value(v) is not None
            }
        else:
            props_out = {}

        props_out.update({
            "rum_id": rum_id,
            "source_row": int(props_in.get("source_row", idx)),
            "x_center": round(x, ROUND_SOURCE_XY_DIGITS),
            "y_center": round(y, ROUND_SOURCE_XY_DIGITS),
            "lon_center": round(lon_center, ROUND_LON_LAT_DIGITS),
            "lat_center": round(lat_center, ROUND_LON_LAT_DIGITS),
            "grid_i": int(gi),
            "grid_j": int(gj),
            "rum_size_m": float(rum_size_m),
            "area_m2": round(area_m2, ROUND_AREA_DIGITS),
        })

        if STORE_SOURCE_CORNERS:
            props_out["source_corners_xy"] = [
                [round(cx, ROUND_SOURCE_XY_DIGITS), round(cy, ROUND_SOURCE_XY_DIGITS)]
                for cx, cy in ring_xy
            ]

        output_features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [ring_lonlat],
            },
            "properties": props_out,
        })

    if duplicate_grid_cells:
        warn(f"Duplicate inferred grid cells found: {duplicate_grid_cells}")
    else:
        ok("No duplicate inferred grid cells")

    footprint_bbox_source = bbox_from_xy(all_corner_x, all_corner_y)
    footprint_bbox_wgs84 = bbox_from_lonlat(all_corner_lon, all_corner_lat)

    ok(f"Built {len(output_features)} WGS84 footprint polygons")
    print(f"  Footprint bbox source : {footprint_bbox_source}")
    print(f"  Footprint bbox WGS84  : {footprint_bbox_wgs84}")

    section("Writing footprint product")
    payload = {
        "type": "FeatureCollection",
        "metadata": {
            "schema": "rum_footprints_v1",
            "source_points_geojson": prepared["points_geojson"],
            "source_crs": source_crs_value,
            "target_crs": "EPSG:4326",
            "rum_size_m": rum_size_m,
            "rum_count": len(output_features),
            "expected_rum_count": expected_rum_count,
            "bbox_source_centers": source_center_bbox,
            "bbox_wgs84_centers": wgs84_center_bbox,
            "bbox_source_footprints": footprint_bbox_source,
            "bbox_wgs84_footprints": footprint_bbox_wgs84,
            "grid": grid_summary,
            "geometry_method": "square_cells_built_in_source_crs_then_transformed_to_wgs84",
        },
        "features": output_features,
    }

    write_json(footprints_path, payload)

    elapsed = time.time() - t_start

    ok(f"Wrote RUM footprints: {footprints_path} ({footprints_path.stat().st_size / 1024 / 1024:.2f} MB)")

    section("Summary")
    ok(f"Step 03 complete in {elapsed:.2f} s")
    print(f"  RUM footprints      : {len(output_features)}")
    print(f"  RUM size            : {rum_size_m} m")
    print(f"  Grid size           : {grid_summary['n_cols']} cols × {grid_summary['n_rows']} rows")
    print(f"  BBox WGS84          : {footprint_bbox_wgs84['west']}, {footprint_bbox_wgs84['south']}, {footprint_bbox_wgs84['east']}, {footprint_bbox_wgs84['north']}")


if __name__ == "__main__":
    main()
