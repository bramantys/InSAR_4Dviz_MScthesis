#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
09_build_blank_cells.py

InSAR4D RUM Viewer pipeline step 09.

Purpose
-------
Detect missing interior RUM grid cells ("blankies") and build a safe blank-cell
product for downstream caps/walls/height-texture steps.

Critical rule
-------------
No blank cells is NOT a failure.

This step ALWAYS writes generated_outputs.blank_cells:

  _internal/data_pipeline/blank_cells.json

If no blank cells are detected, the output is a valid empty FeatureCollection:

{
  "type": "FeatureCollection",
  "metadata": {
    "schema": "blank_cells_v2_model_sigma",
    "blank_count": 0,
    "status": "no_blank_cells_detected"
  },
  "features": []
}

Inputs
------
  generated_outputs.rum_footprints
  generated_outputs.packed_series

Outputs
-------
  generated_outputs.blank_cells

Blank detection concept
-----------------------
The real RUM grid is defined by grid_i/grid_j in rum_footprints.json.
A missing grid cell is considered an interior blank only if it is enclosed by
real cells according to the configured rule.

Default rule:
  row_and_col

Meaning:
  - there is at least one real cell left AND right in the same row
  - there is at least one real cell below AND above in the same column

This avoids filling obvious exterior/coastal/outside-of-domain cells.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

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

# Options:
#   "row_and_col" : conservative, likely true interior holes only
#   "row_or_col"  : more aggressive, fills row/column gaps
INTERIOR_RULE = "row_and_col"

# Neighbors used for interpolation. Radius expands until enough neighbours are
# found or MAX_INTERPOLATION_RADIUS is reached.
MIN_NEIGHBOURS_FOR_INTERPOLATION = 2
MAX_INTERPOLATION_RADIUS = 6
MAX_NEIGHBOURS_USED = 12

# Inverse distance power for interpolation.
IDW_POWER = 2.0

# If no neighbour is found, write zero displacement/sigma instead of failing.
# This should almost never be needed for interior blanks, but keeps the pipeline
# robust.
FALLBACK_TO_ZERO_IF_NO_NEIGHBOURS = True

ROUND_LON_LAT_DIGITS = 8
ROUND_SOURCE_XY_DIGITS = 4
ROUND_SERIES_DIGITS = 4


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


# =============================================================================
# BASIC HELPERS
# =============================================================================

def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))


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


def safe_int(value: Any, fallback: Optional[int] = None) -> Optional[int]:
    try:
        if value is None:
            return fallback
        return int(value)
    except Exception:
        return fallback


def bbox_wgs84_from_ring(ring: List[List[float]]) -> Dict[str, float]:
    lons = [float(p[0]) for p in ring]
    lats = [float(p[1]) for p in ring]
    return {
        "west": round(min(lons), ROUND_LON_LAT_DIGITS),
        "south": round(min(lats), ROUND_LON_LAT_DIGITS),
        "east": round(max(lons), ROUND_LON_LAT_DIGITS),
        "north": round(max(lats), ROUND_LON_LAT_DIGITS),
    }


def bbox_source_from_ring(ring: List[Tuple[float, float]]) -> Dict[str, float]:
    xs = [float(p[0]) for p in ring]
    ys = [float(p[1]) for p in ring]
    return {
        "min_x": round(min(xs), ROUND_SOURCE_XY_DIGITS),
        "min_y": round(min(ys), ROUND_SOURCE_XY_DIGITS),
        "max_x": round(max(xs), ROUND_SOURCE_XY_DIGITS),
        "max_y": round(max(ys), ROUND_SOURCE_XY_DIGITS),
    }


def polygon_area_square(size_m: float) -> float:
    return float(size_m) * float(size_m)


# =============================================================================
# GEOMETRY HELPERS
# =============================================================================

def build_square_ring_xy(x: float, y: float, size_m: float) -> List[Tuple[float, float]]:
    h = size_m / 2.0
    return [
        (x - h, y - h),
        (x + h, y - h),
        (x + h, y + h),
        (x - h, y + h),
        (x - h, y - h),
    ]


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


def transform_point_to_wgs84(transformer: Transformer, x: float, y: float) -> Tuple[float, float]:
    lon, lat = transformer.transform(float(x), float(y))
    return round(float(lon), ROUND_LON_LAT_DIGITS), round(float(lat), ROUND_LON_LAT_DIGITS)


# =============================================================================
# GRID / BLANK DETECTION
# =============================================================================

def load_real_grid(footprints: Dict[str, Any]) -> Tuple[Dict[Tuple[int, int], Dict[str, Any]], Dict[str, Any]]:
    features = footprints.get("features", [])
    if not features:
        raise ValueError("rum_footprints.json contains no features")

    grid: Dict[Tuple[int, int], Dict[str, Any]] = {}

    for idx, feature in enumerate(features):
        props = feature.get("properties") or {}

        rum_id = str(props.get("rum_id", f"RUM_{idx + 1:06d}"))
        gi = safe_int(props.get("grid_i"))
        gj = safe_int(props.get("grid_j"))
        x = safe_float(props.get("x_center"))
        y = safe_float(props.get("y_center"))

        if gi is None or gj is None:
            raise ValueError(f"Footprint {rum_id} missing grid_i/grid_j")
        if x is None or y is None:
            raise ValueError(f"Footprint {rum_id} missing x_center/y_center")

        key = (gi, gj)
        if key in grid:
            raise ValueError(f"Duplicate real grid cell found at {key}")

        grid[key] = {
            "rum_id": rum_id,
            "grid_i": gi,
            "grid_j": gj,
            "x_center": x,
            "y_center": y,
            "lon_center": safe_float(props.get("lon_center")),
            "lat_center": safe_float(props.get("lat_center")),
        }

    metadata = footprints.get("metadata") or {}
    return grid, metadata


def build_row_col_sets(occupied: Set[Tuple[int, int]]) -> Tuple[Dict[int, Set[int]], Dict[int, Set[int]]]:
    rows: Dict[int, Set[int]] = {}
    cols: Dict[int, Set[int]] = {}

    for gi, gj in occupied:
        rows.setdefault(gj, set()).add(gi)
        cols.setdefault(gi, set()).add(gj)

    return rows, cols


def is_interior_missing_cell(
    gi: int,
    gj: int,
    rows: Dict[int, Set[int]],
    cols: Dict[int, Set[int]],
    rule: str,
) -> bool:
    row_values = rows.get(gj, set())
    col_values = cols.get(gi, set())

    has_left = any(i < gi for i in row_values)
    has_right = any(i > gi for i in row_values)
    has_down = any(j < gj for j in col_values)
    has_up = any(j > gj for j in col_values)

    row_span = has_left and has_right
    col_span = has_down and has_up

    if rule == "row_and_col":
        return row_span and col_span
    if rule == "row_or_col":
        return row_span or col_span

    raise ValueError(f"Unknown INTERIOR_RULE: {rule}")


def detect_blank_cells(real_grid: Dict[Tuple[int, int], Dict[str, Any]]) -> List[Tuple[int, int]]:
    occupied = set(real_grid.keys())

    min_i = min(i for i, _ in occupied)
    max_i = max(i for i, _ in occupied)
    min_j = min(j for _, j in occupied)
    max_j = max(j for _, j in occupied)

    rows, cols = build_row_col_sets(occupied)

    blanks: List[Tuple[int, int]] = []

    for gj in range(min_j, max_j + 1):
        for gi in range(min_i, max_i + 1):
            key = (gi, gj)
            if key in occupied:
                continue

            if is_interior_missing_cell(gi, gj, rows, cols, INTERIOR_RULE):
                blanks.append(key)

    return blanks


# =============================================================================
# INTERPOLATION
# =============================================================================

def find_neighbour_keys(
    blank_key: Tuple[int, int],
    real_grid: Dict[Tuple[int, int], Dict[str, Any]],
    max_radius: int,
    min_neighbours: int,
) -> List[Tuple[int, int]]:
    gi0, gj0 = blank_key

    found: List[Tuple[float, Tuple[int, int]]] = []

    for radius in range(1, max_radius + 1):
        found.clear()

        for (gi, gj) in real_grid.keys():
            di = gi - gi0
            dj = gj - gj0
            if max(abs(di), abs(dj)) <= radius:
                dist = math.hypot(di, dj)
                if dist > 0:
                    found.append((dist, (gi, gj)))

        if len(found) >= min_neighbours:
            break

    found.sort(key=lambda item: item[0])
    return [key for _, key in found[:MAX_NEIGHBOURS_USED]]


def weighted_average_series(
    blank_key: Tuple[int, int],
    neighbour_keys: List[Tuple[int, int]],
    real_grid: Dict[Tuple[int, int], Dict[str, Any]],
    packed: Dict[str, Any],
) -> Tuple[List[float], List[float], Dict[str, Any]]:
    """
    Interpolate MODEL and sigma series for blank/no-data cells.

    Blank cells are not measurements. They are only support cells for the
    RUM surface/height texture, so they receive:
      - model_mm: interpolated from neighbouring real RUM model_mm
      - sigma_mm: interpolated from neighbouring real RUM sigma_mm

    measurement_mm is intentionally not produced for blank cells.
    """
    epochs = packed.get("epochs", [])
    epoch_count = len(epochs)

    arrays = packed.get("arrays") or {}
    model_flat = arrays.get("model_mm")
    sigma_flat = arrays.get("sigma_mm")

    if not isinstance(model_flat, list) or not isinstance(sigma_flat, list):
        raise ValueError("packed_series.json missing arrays.model_mm / arrays.sigma_mm")

    meta = packed.get("metadata") or {}
    packed_epoch_count = int(meta.get("epoch_count", epoch_count))

    if packed_epoch_count != epoch_count:
        raise ValueError("Packed epoch count metadata does not match epochs length")

    if not neighbour_keys:
        if FALLBACK_TO_ZERO_IF_NO_NEIGHBOURS:
            return (
                [0.0] * epoch_count,
                [0.0] * epoch_count,
                {
                    "method": "fallback_zero",
                    "neighbour_count": 0,
                    "neighbour_rum_ids": [],
                    "interpolated_roles": ["model_mm", "sigma_mm"],
                },
            )
        raise ValueError(f"No neighbours found for blank cell {blank_key}")

    gi0, gj0 = blank_key

    weighted_model = [0.0] * epoch_count
    weighted_sigma = [0.0] * epoch_count
    weight_sum = 0.0
    neighbour_rum_ids: List[str] = []

    for key in neighbour_keys:
        gi, gj = key
        info = real_grid[key]
        rum_id = info["rum_id"]
        row_index = int(packed["rum_index"][rum_id])

        dist = math.hypot(gi - gi0, gj - gj0)
        if dist <= 0:
            continue

        weight = 1.0 / (dist ** IDW_POWER)
        weight_sum += weight
        neighbour_rum_ids.append(rum_id)

        offset = row_index * epoch_count

        for eidx in range(epoch_count):
            weighted_model[eidx] += float(model_flat[offset + eidx]) * weight
            weighted_sigma[eidx] += float(sigma_flat[offset + eidx]) * weight

    if weight_sum <= 0:
        if FALLBACK_TO_ZERO_IF_NO_NEIGHBOURS:
            return (
                [0.0] * epoch_count,
                [0.0] * epoch_count,
                {
                    "method": "fallback_zero_weight_sum",
                    "neighbour_count": len(neighbour_keys),
                    "neighbour_rum_ids": neighbour_rum_ids,
                    "interpolated_roles": ["model_mm", "sigma_mm"],
                },
            )
        raise ValueError(f"Neighbour weights are zero for blank cell {blank_key}")

    model = [round(v / weight_sum, ROUND_SERIES_DIGITS) for v in weighted_model]
    sigma = [round(max(0.0, s / weight_sum), ROUND_SERIES_DIGITS) for s in weighted_sigma]

    return (
        model,
        sigma,
        {
            "method": "idw_grid_neighbours",
            "idw_power": IDW_POWER,
            "neighbour_count": len(neighbour_rum_ids),
            "neighbour_rum_ids": neighbour_rum_ids,
            "interpolated_roles": ["model_mm", "sigma_mm"],
        },
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    t_start = time.time()

    cfg = load_resolved_config(__file__)
    project_root = Path(cfg["_resolved"]["project_root"])

    generated = cfg["generated_outputs"]
    user_inputs = cfg["user_inputs"]

    footprints_path = resolve_path(project_root, generated["rum_footprints"])
    packed_path = resolve_path(project_root, generated["packed_series"])
    output_path = resolve_path(project_root, generated["blank_cells"])

    source_crs_value = user_inputs["source_crs"]
    rum_size_m = float(user_inputs["rum_size_m"])

    section("Configuration")
    print(f"  Project root        : {project_root}")
    print(f"  Footprints input    : {footprints_path}")
    print(f"  Packed input        : {packed_path}")
    print(f"  Blank output        : {output_path}")
    print(f"  Source CRS          : {source_crs_value}")
    print(f"  RUM size            : {rum_size_m} m")
    print(f"  Interior rule       : {INTERIOR_RULE}")

    source_crs = CRS.from_user_input(source_crs_value)
    transformer_to_wgs84 = Transformer.from_crs(source_crs, CRS.from_epsg(4326), always_xy=True)

    section("Loading inputs")
    footprints = load_json(footprints_path)
    packed = load_json(packed_path)

    real_grid, footprint_meta = load_real_grid(footprints)
    packed_rum_index = packed.get("rum_index") or {}

    if not packed_rum_index:
        raise ValueError("packed_series.json missing rum_index")

    missing_in_packed = sorted(
        info["rum_id"] for info in real_grid.values()
        if info["rum_id"] not in packed_rum_index
    )
    if missing_in_packed:
        raise ValueError(f"Real footprint RUMs missing in packed_series.rum_index; sample={missing_in_packed[:10]}")

    ok(f"Loaded {len(real_grid)} real RUM grid cells")
    ok(f"Loaded packed series with {len(packed_rum_index)} row indices")

    section("Detecting interior blank cells")
    blank_keys = detect_blank_cells(real_grid)

    if not blank_keys:
        ok("No blank cells detected; this is valid")

        payload = {
            "type": "FeatureCollection",
            "metadata": {
                "schema": "blank_cells_v2_model_sigma",
                "status": "no_blank_cells_detected",
                "blank_count": 0,
                "real_rum_count": len(real_grid),
                "interior_rule": INTERIOR_RULE,
                "source_footprints": generated["rum_footprints"],
                "source_packed_series": generated["packed_series"],
                "rum_size_m": rum_size_m,
                "source_crs": source_crs_value,
                "epoch_count": len(packed.get("epochs", [])),
                "epochs": packed.get("epochs", []),
                "roles": {
                    "model_mm": "interpolated blank-cell MODEL series for height texture/caps/walls",
                    "sigma_mm": "interpolated blank-cell uncertainty series",
                    "measurement_mm": "not produced for blank cells because blanks are no-data support cells",
                },
            },
            "features": [],
        }

        write_json(output_path, payload)

        elapsed = time.time() - t_start

        ok(f"Wrote empty blank-cell product: {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")

        section("Summary")
        ok(f"Step 09 complete in {elapsed:.2f} s")
        print(f"  Blank count            : 0")
        print(f"  Status                 : no_blank_cells_detected")
        return

    ok(f"Detected {len(blank_keys)} interior blank cells")

    section("Building blank-cell features")
    grid_meta = footprint_meta.get("grid") or {}
    origin_x = safe_float(grid_meta.get("origin_x"))
    origin_y = safe_float(grid_meta.get("origin_y"))
    spacing_m = safe_float(grid_meta.get("spacing_m"), rum_size_m)

    if origin_x is None or origin_y is None:
        raise ValueError("Footprint metadata grid.origin_x/origin_y is missing")
    if spacing_m is None or spacing_m <= 0:
        raise ValueError("Footprint metadata grid.spacing_m is missing/invalid")

    features: List[Dict[str, Any]] = []
    all_model_values: List[float] = []
    all_sigma_values: List[float] = []
    neighbour_counts: List[int] = []
    fallback_zero_count = 0

    for blank_index, (gi, gj) in enumerate(blank_keys):
        blank_id = f"BLANK_i{gi:04d}_j{gj:04d}"

        x_center = origin_x + gi * spacing_m
        y_center = origin_y + gj * spacing_m
        lon_center, lat_center = transform_point_to_wgs84(transformer_to_wgs84, x_center, y_center)

        ring_xy = build_square_ring_xy(x_center, y_center, rum_size_m)
        ring_lonlat = transform_ring_to_wgs84(transformer_to_wgs84, ring_xy)

        neighbours = find_neighbour_keys(
            blank_key=(gi, gj),
            real_grid=real_grid,
            max_radius=MAX_INTERPOLATION_RADIUS,
            min_neighbours=MIN_NEIGHBOURS_FOR_INTERPOLATION,
        )

        model, sigma, interp_meta = weighted_average_series(
            blank_key=(gi, gj),
            neighbour_keys=neighbours,
            real_grid=real_grid,
            packed=packed,
        )

        if interp_meta["method"].startswith("fallback_zero"):
            fallback_zero_count += 1

        neighbour_counts.append(int(interp_meta["neighbour_count"]))
        all_model_values.extend(model)
        all_sigma_values.extend(sigma)

        props = {
            "blank_id": blank_id,
            "blank_index": blank_index,
            "grid_i": gi,
            "grid_j": gj,
            "x_center": round(x_center, ROUND_SOURCE_XY_DIGITS),
            "y_center": round(y_center, ROUND_SOURCE_XY_DIGITS),
            "lon_center": lon_center,
            "lat_center": lat_center,
            "rum_size_m": rum_size_m,
            "area_m2": round(polygon_area_square(rum_size_m), ROUND_SOURCE_XY_DIGITS),
            "source_corners_xy": [
                [round(x, ROUND_SOURCE_XY_DIGITS), round(y, ROUND_SOURCE_XY_DIGITS)]
                for x, y in ring_xy
            ],
            "bbox_wgs84": bbox_wgs84_from_ring(ring_lonlat),
            "bbox_source": bbox_source_from_ring(ring_xy),
            "interpolation": interp_meta,
            "model_mm": model,
            "sigma_mm": sigma,
        }

        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [ring_lonlat],
            },
            "properties": props,
        })

    if fallback_zero_count:
        warn(f"Blank cells using fallback zero series: {fallback_zero_count}")

    section("Writing blank-cell product")
    blank_count = len(features)

    payload = {
        "type": "FeatureCollection",
        "metadata": {
            "schema": "blank_cells_v1",
            "status": "blank_cells_detected",
            "blank_count": blank_count,
            "real_rum_count": len(real_grid),
            "interior_rule": INTERIOR_RULE,
            "source_footprints": generated["rum_footprints"],
            "source_packed_series": generated["packed_series"],
            "source_crs": source_crs_value,
            "rum_size_m": rum_size_m,
            "grid_origin_x": origin_x,
            "grid_origin_y": origin_y,
            "grid_spacing_m": spacing_m,
            "epoch_count": len(packed.get("epochs", [])),
            "epochs": packed.get("epochs", []),
            "roles": {
                "model_mm": "interpolated blank-cell MODEL series for height texture/caps/walls",
                "sigma_mm": "interpolated blank-cell uncertainty series",
                "measurement_mm": "not produced for blank cells because blanks are no-data support cells",
            },
            "interpolation": {
                "method": "idw_grid_neighbours",
                "min_neighbours": MIN_NEIGHBOURS_FOR_INTERPOLATION,
                "max_radius": MAX_INTERPOLATION_RADIUS,
                "max_neighbours_used": MAX_NEIGHBOURS_USED,
                "idw_power": IDW_POWER,
                "fallback_zero_count": fallback_zero_count,
                "neighbour_count_min": min(neighbour_counts) if neighbour_counts else 0,
                "neighbour_count_max": max(neighbour_counts) if neighbour_counts else 0,
            },
            "summary": {
                "model_min_mm": round(min(all_model_values), ROUND_SERIES_DIGITS) if all_model_values else None,
                "model_max_mm": round(max(all_model_values), ROUND_SERIES_DIGITS) if all_model_values else None,
                "sigma_min_mm": round(min(all_sigma_values), ROUND_SERIES_DIGITS) if all_sigma_values else None,
                "sigma_max_mm": round(max(all_sigma_values), ROUND_SERIES_DIGITS) if all_sigma_values else None,
            },
        },
        "features": features,
    }

    write_json(output_path, payload)

    elapsed = time.time() - t_start

    ok(f"Wrote blank cells: {output_path} ({output_path.stat().st_size / 1024 / 1024:.2f} MB)")

    section("Summary")
    ok(f"Step 09 complete in {elapsed:.2f} s")
    print(f"  Blank count            : {blank_count}")
    print(f"  Real RUM count          : {len(real_grid)}")
    print(f"  Fallback zero count     : {fallback_zero_count}")
    if neighbour_counts:
        print(f"  Neighbours per blank    : {min(neighbour_counts)} to {max(neighbour_counts)}")
    if all_model_values:
        print(f"  Blank MODEL range    : {min(all_model_values):.4f} to {max(all_model_values):.4f} mm")
    if all_sigma_values:
        print(f"  Blank sigma range       : {min(all_sigma_values):.4f} to {max(all_sigma_values):.4f} mm")


if __name__ == "__main__":
    main()
