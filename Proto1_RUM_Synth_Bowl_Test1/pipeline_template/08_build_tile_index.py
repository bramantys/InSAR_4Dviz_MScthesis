#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08_build_tile_index.py

Generic RUM-based InSAR template step.

Purpose
-------
Divide corrected RUM footprints into spatial tiles and write the main
3D Tiles tileset manifest + tile assignment table.

Input:
  config.generated_outputs.rum_footprints

Outputs:
  Data/tiles/tileset.json
  Data/tiles/tile_assignments.json

Notes
-----
This script replaces the old Jakarta-specific phase4_tiler.py.

Important generic fixes:
  - no hardcoded Jakarta latitude;
  - latitude for metre approximation is derived from footprint bbox center;
  - output paths are read from config;
  - tile grid defaults to 8 × 6 but can be overridden in config["tiling"].
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple


# =============================================================================
# PATHS
# =============================================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_DIR / "config" / "project_config.json"


# =============================================================================
# PRINT HELPERS
# =============================================================================

def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def ok(msg: str) -> None:
    print(f"  [OK]   {msg}")


def warn(msg: str) -> None:
    print(f"  [WARN] {msg}")


# =============================================================================
# HELPERS
# =============================================================================

def load_config() -> Dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing config file: {CONFIG_PATH}")
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_project_path(path_value: str) -> Path:
    p = Path(path_value)
    if p.is_absolute():
        return p
    return PROJECT_DIR / p


def deg_to_rad(value_deg: float) -> float:
    return value_deg * math.pi / 180.0


def region_bounding_volume(
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    height_min: float,
    height_max: float,
) -> Dict[str, List[float]]:
    return {
        "region": [
            deg_to_rad(lon_min),
            deg_to_rad(lat_min),
            deg_to_rad(lon_max),
            deg_to_rad(lat_max),
            height_min,
            height_max,
        ]
    }


def compute_geometric_error(
    lon_span_deg: float,
    lat_span_deg: float,
    reference_lat_deg: float,
) -> float:
    lon_m = lon_span_deg * 111_320.0 * max(math.cos(math.radians(reference_lat_deg)), 1e-6)
    lat_m = lat_span_deg * 111_320.0
    return round(math.sqrt(lon_m * lon_m + lat_m * lat_m), 1)


def load_footprints(path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    footprints = data.get("footprints", {})
    if not isinstance(footprints, dict) or not footprints:
        raise ValueError(f"No non-empty 'footprints' dict in {path}")

    return data, footprints


def get_center(fp: Dict[str, Any]) -> Tuple[float, float]:
    center = fp.get("center", [])
    if len(center) < 2:
        raise ValueError("Footprint missing center")
    return float(center[0]), float(center[1])


def get_corners_bbox(footprints: Dict[str, Any]) -> Dict[str, float]:
    lons: List[float] = []
    lats: List[float] = []

    for fp in footprints.values():
        for corner in fp.get("corners", []):
            if len(corner) >= 2:
                lons.append(float(corner[0]))
                lats.append(float(corner[1]))

    if not lons or not lats:
        # Fallback to centers.
        for fp in footprints.values():
            lon, lat = get_center(fp)
            lons.append(lon)
            lats.append(lat)

    return {
        "lon_min": min(lons),
        "lon_max": max(lons),
        "lat_min": min(lats),
        "lat_max": max(lats),
    }


def active_tile_sort_key(key: str) -> Tuple[int, int]:
    col_s, row_s = key.split("_")
    return int(col_s), int(row_s)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    t_start = time.time()

    cfg = load_config()

    generated = cfg.get("generated_outputs", {})
    paths_cfg = cfg.get("paths", {})
    tiling_cfg = cfg.get("tiling", {})

    footprints_path = resolve_project_path(
        generated.get("rum_footprints", "Data/rum_footprints.json")
    )

    tiles_dir = resolve_project_path(
        paths_cfg.get("tiles_dir", "Data/tiles")
    )
    tileset_path = tiles_dir / "tileset.json"
    assignments_path = tiles_dir / "tile_assignments.json"

    tile_grid_cols = int(tiling_cfg.get("tile_grid_cols", 8))
    tile_grid_rows = int(tiling_cfg.get("tile_grid_rows", 6))

    geometric_error_root = float(tiling_cfg.get("geometric_error_root", 5000.0))
    geometric_error_leaf = float(tiling_cfg.get("geometric_error_leaf", 100.0))

    bound_min_height_m = float(tiling_cfg.get("tileset_bound_min_height_m", -1000.0))
    bound_max_height_m = float(tiling_cfg.get("tileset_bound_max_height_m", 10000.0))

    section("Configuration")
    print(f"  Project root         : {PROJECT_DIR}")
    print(f"  Footprints           : {footprints_path}")
    print(f"  Tiles dir            : {tiles_dir}")
    print(f"  Tileset output       : {tileset_path}")
    print(f"  Assignments output   : {assignments_path}")
    print(f"  Tile grid            : {tile_grid_cols} × {tile_grid_rows}")
    print(f"  Height bounds        : {bound_min_height_m} → {bound_max_height_m} m")

    section("Loading footprints")
    if not footprints_path.exists():
        raise FileNotFoundError(f"Missing footprints: {footprints_path}")

    fp_data, footprints = load_footprints(footprints_path)
    bbox = fp_data.get("bbox") or get_corners_bbox(footprints)

    n_rums = len(footprints)
    ok(f"Loaded footprints: {n_rums}")

    lon_min = float(bbox["lon_min"])
    lon_max = float(bbox["lon_max"])
    lat_min = float(bbox["lat_min"])
    lat_max = float(bbox["lat_max"])
    lat_ref = 0.5 * (lat_min + lat_max)

    print(f"  Footprint bbox lon: {lon_min:.6f} → {lon_max:.6f}")
    print(f"  Footprint bbox lat: {lat_min:.6f} → {lat_max:.6f}")
    print(f"  Reference latitude: {lat_ref:.6f}")

    grid_model = fp_data.get("metadata", {}).get("grid_model", {})
    spacing_u = float(grid_model.get("spacing_u", 0.0) or 0.0)
    spacing_v = float(grid_model.get("spacing_v", 0.0) or 0.0)
    half_cell_m = 0.5 * max(spacing_u, spacing_v, float(tiling_cfg.get("fallback_grid_spacing_m", 450.0)))

    ok(f"Using half-cell size for tightened bboxes: {half_cell_m:.3f} m")

    section(f"Building {tile_grid_cols} × {tile_grid_rows} tile grid")

    lon_span = lon_max - lon_min
    lat_span = lat_max - lat_min

    if lon_span <= 0 or lat_span <= 0:
        raise ValueError("Footprint bbox has non-positive lon/lat span")

    tile_lon_step = lon_span / tile_grid_cols
    tile_lat_step = lat_span / tile_grid_rows

    tile_rums: Dict[str, List[str]] = {
        f"{col}_{row}": []
        for row in range(tile_grid_rows)
        for col in range(tile_grid_cols)
    }

    section("Assigning RUMs to tiles")
    t0 = time.time()
    unassigned = 0

    for rum_id, fp in footprints.items():
        try:
            lon, lat = get_center(fp)
        except Exception:
            unassigned += 1
            continue

        col = min(max(int((lon - lon_min) / tile_lon_step), 0), tile_grid_cols - 1)
        row = min(max(int((lat - lat_min) / tile_lat_step), 0), tile_grid_rows - 1)

        tile_rums[f"{col}_{row}"].append(str(rum_id))

    active_tiles = {key: ids for key, ids in tile_rums.items() if ids}
    empty_tiles = [key for key, ids in tile_rums.items() if not ids]
    total_assigned = sum(len(ids) for ids in active_tiles.values())

    ok(f"Assigned {total_assigned} RUMs to {len(active_tiles)} active tiles in {time.time() - t0:.2f}s")
    ok(f"Empty tiles: {len(empty_tiles)}")
    if unassigned:
        warn(f"Unassigned RUMs due to bad centers: {unassigned}")

    counts = sorted(len(ids) for ids in active_tiles.values())
    if counts:
        print(f"  RUMs per tile: min={counts[0]}, median={counts[len(counts)//2]}, max={counts[-1]}")

    section("Tightening tile bounding boxes")

    half_cell_lon = half_cell_m / (111_320.0 * max(math.cos(math.radians(lat_ref)), 1e-6))
    half_cell_lat = half_cell_m / 111_320.0

    tile_bbox: Dict[str, Dict[str, float]] = {}

    for key, rum_ids in active_tiles.items():
        center_lons: List[float] = []
        center_lats: List[float] = []

        # Prefer actual corners for safest bbox.
        corner_lons: List[float] = []
        corner_lats: List[float] = []

        for rid in rum_ids:
            fp = footprints[rid]
            lon, lat = get_center(fp)
            center_lons.append(lon)
            center_lats.append(lat)

            for corner in fp.get("corners", []):
                if len(corner) >= 2:
                    corner_lons.append(float(corner[0]))
                    corner_lats.append(float(corner[1]))

        if corner_lons and corner_lats:
            tile_bbox[key] = {
                "lon_min": min(corner_lons),
                "lon_max": max(corner_lons),
                "lat_min": min(corner_lats),
                "lat_max": max(corner_lats),
            }
        else:
            tile_bbox[key] = {
                "lon_min": min(center_lons) - half_cell_lon,
                "lon_max": max(center_lons) + half_cell_lon,
                "lat_min": min(center_lats) - half_cell_lat,
                "lat_max": max(center_lats) + half_cell_lat,
            }

    ok("Tile bounding boxes built")

    section("Building tileset.json")

    root_lon_min = min(b["lon_min"] for b in tile_bbox.values())
    root_lon_max = max(b["lon_max"] for b in tile_bbox.values())
    root_lat_min = min(b["lat_min"] for b in tile_bbox.values())
    root_lat_max = max(b["lat_max"] for b in tile_bbox.values())

    children: List[Dict[str, Any]] = []

    for key in sorted(active_tiles.keys(), key=active_tile_sort_key):
        b = tile_bbox[key]
        # Keep leaf geometric error fixed for compatibility with previous viewer behavior.
        auto_error = compute_geometric_error(
            b["lon_max"] - b["lon_min"],
            b["lat_max"] - b["lat_min"],
            lat_ref,
        )

        children.append({
            "boundingVolume": region_bounding_volume(
                b["lon_min"],
                b["lon_max"],
                b["lat_min"],
                b["lat_max"],
                bound_min_height_m,
                bound_max_height_m,
            ),
            "geometricError": geometric_error_leaf,
            "refine": "REPLACE",
            "content": {
                "uri": f"{key}.b3dm",
            },
            "metadata": {
                "auto_geometric_error_m": auto_error,
                "rum_count": len(active_tiles[key]),
            },
        })

    tileset = {
        "asset": {
            "version": "1.0",
            "tilesetVersion": "1.0.0",
            "generator": "RUM 4D Template — 08_build_tile_index.py",
        },
        "geometricError": geometric_error_root,
        "root": {
            "boundingVolume": region_bounding_volume(
                root_lon_min,
                root_lon_max,
                root_lat_min,
                root_lat_max,
                bound_min_height_m,
                bound_max_height_m,
            ),
            "geometricError": geometric_error_root,
            "refine": "ADD",
            "children": children,
        },
    }

    ok(f"tileset.json contains 1 root + {len(children)} children")

    section("Writing outputs")

    tiles_dir.mkdir(parents=True, exist_ok=True)

    with tileset_path.open("w", encoding="utf-8") as f:
        json.dump(tileset, f, indent=2)

    assignments = {
        "metadata": {
            "schema_version": "tile_assignments_v1",
            "grid_cols": tile_grid_cols,
            "grid_rows": tile_grid_rows,
            "active_tiles": len(active_tiles),
            "empty_tiles": len(empty_tiles),
            "total_rums": total_assigned,
            "half_cell_m": half_cell_m,
            "source_footprints": str(footprints_path),
            "created_by": "08_build_tile_index.py",
            "created_unix": int(time.time()),
        },
        "tiles": {
            key: {
                "rum_ids": active_tiles[key],
                "rum_count": len(active_tiles[key]),
                "bbox": tile_bbox[key],
                "b3dm_filename": f"{key}.b3dm",
            }
            for key in sorted(active_tiles.keys(), key=active_tile_sort_key)
        },
    }

    with assignments_path.open("w", encoding="utf-8") as f:
        json.dump(assignments, f, separators=(",", ":"))

    ok(f"Written tileset    : {tileset_path} ({tileset_path.stat().st_size / 1024:.1f} KB)")
    ok(f"Written assignments: {assignments_path} ({assignments_path.stat().st_size / 1024:.1f} KB)")

    section("SUMMARY")
    ok("Step 08 complete — tile index and tileset manifest created")
    ok("Next template step: 09_export_epoch_axis.py")
    print(f"  Elapsed: {time.time() - t_start:.2f}s")
    print()


if __name__ == "__main__":
    main()
