#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
07_build_tile_index.py

InSAR4D RUM Viewer pipeline step 07.

Purpose
-------
Build a spatial tile index from RUM footprints.

Inputs
------
  generated_outputs.rum_footprints
    _internal/data_pipeline/rum_footprints.json

  generated_outputs.packed_series
    _internal/data_pipeline/packed_series.json

Outputs
-------
  _internal/data_pipeline/tiles/tile_index.json
  _internal/data_pipeline/tiles/tile_assignments.json

Why this step exists
--------------------
The viewer should not load/build geometry as one giant object. Later B3DM
steps need a tile-by-tile grouping of RUM footprints.

This step groups RUMs by grid_i/grid_j into a configured number of spatial
tiles. It also attaches each RUM's stable row_index from packed_series.json,
which is critical for height texture lookup.

Important
---------
This step does NOT build B3DM geometry yet.
It only builds the spatial index and assignments used by later steps.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pipeline_config import load_resolved_config, resolve_path


# =============================================================================
# ADVANCED SETTINGS
# =============================================================================
# Normal users should edit config/project_config.json, not this script.

TILE_INDEX_FILENAME = "tile_index.json"
TILE_ASSIGNMENTS_FILENAME = "tile_assignments.json"

ROUND_LON_LAT_DIGITS = 8
ROUND_SOURCE_XY_DIGITS = 4

# Keep empty tiles in tile_index for deterministic grid layout. Later geometry
# builders can skip tiles with rum_count = 0.
KEEP_EMPTY_TILES = True


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


def bbox_union_wgs84(bboxes: List[Dict[str, float]]) -> Optional[Dict[str, float]]:
    if not bboxes:
        return None
    return {
        "west": round(min(b["west"] for b in bboxes), ROUND_LON_LAT_DIGITS),
        "south": round(min(b["south"] for b in bboxes), ROUND_LON_LAT_DIGITS),
        "east": round(max(b["east"] for b in bboxes), ROUND_LON_LAT_DIGITS),
        "north": round(max(b["north"] for b in bboxes), ROUND_LON_LAT_DIGITS),
    }


def bbox_union_source(bboxes: List[Dict[str, float]]) -> Optional[Dict[str, float]]:
    if not bboxes:
        return None
    return {
        "min_x": round(min(b["min_x"] for b in bboxes), ROUND_SOURCE_XY_DIGITS),
        "min_y": round(min(b["min_y"] for b in bboxes), ROUND_SOURCE_XY_DIGITS),
        "max_x": round(max(b["max_x"] for b in bboxes), ROUND_SOURCE_XY_DIGITS),
        "max_y": round(max(b["max_y"] for b in bboxes), ROUND_SOURCE_XY_DIGITS),
    }


def polygon_bbox_wgs84(ring: List[List[float]]) -> Dict[str, float]:
    lons = [float(p[0]) for p in ring]
    lats = [float(p[1]) for p in ring]
    return {
        "west": round(min(lons), ROUND_LON_LAT_DIGITS),
        "south": round(min(lats), ROUND_LON_LAT_DIGITS),
        "east": round(max(lons), ROUND_LON_LAT_DIGITS),
        "north": round(max(lats), ROUND_LON_LAT_DIGITS),
    }


def source_corners_bbox(corners: List[List[float]]) -> Dict[str, float]:
    xs = [float(p[0]) for p in corners]
    ys = [float(p[1]) for p in corners]
    return {
        "min_x": round(min(xs), ROUND_SOURCE_XY_DIGITS),
        "min_y": round(min(ys), ROUND_SOURCE_XY_DIGITS),
        "max_x": round(max(xs), ROUND_SOURCE_XY_DIGITS),
        "max_y": round(max(ys), ROUND_SOURCE_XY_DIGITS),
    }


# =============================================================================
# TILE LOGIC
# =============================================================================

def tile_partition_edges(min_value: int, max_value: int, n_parts: int) -> List[Tuple[int, int]]:
    """
    Partition integer index range [min_value, max_value] into n_parts inclusive
    integer ranges.

    Example:
      min=0, max=94, n_parts=8
      -> [(0,11), (12,23), ..., (84,94)]
    """
    if n_parts <= 0:
        raise ValueError("n_parts must be positive")

    total = max_value - min_value + 1
    n_parts = min(n_parts, total)

    edges: List[Tuple[int, int]] = []
    for k in range(n_parts):
        start = min_value + math.floor(k * total / n_parts)
        end = min_value + math.floor((k + 1) * total / n_parts) - 1
        edges.append((start, end))

    return edges


def find_tile_index(value: int, ranges: List[Tuple[int, int]]) -> int:
    for idx, (lo, hi) in enumerate(ranges):
        if lo <= value <= hi:
            return idx
    raise ValueError(f"Grid index {value} outside tile ranges: {ranges}")


def build_feature_records(
    footprints: Dict[str, Any],
    rum_index: Dict[str, int],
) -> List[Dict[str, Any]]:
    features = footprints.get("features", [])
    if not features:
        raise ValueError("rum_footprints.json has no features")

    records: List[Dict[str, Any]] = []
    missing_row_index = 0

    for idx, feature in enumerate(features):
        props = feature.get("properties") or {}
        geom = feature.get("geometry") or {}
        rings = geom.get("coordinates") or []

        rum_id = str(props.get("rum_id", f"RUM_{idx + 1:06d}"))

        if rum_id not in rum_index:
            missing_row_index += 1
            continue

        gi = safe_int(props.get("grid_i"))
        gj = safe_int(props.get("grid_j"))
        if gi is None or gj is None:
            raise ValueError(f"Footprint {rum_id} missing grid_i/grid_j")

        if geom.get("type") != "Polygon" or not rings or not rings[0]:
            raise ValueError(f"Footprint {rum_id} has invalid polygon geometry")

        ring = rings[0]
        bbox_wgs84 = polygon_bbox_wgs84(ring)

        source_corners = props.get("source_corners_xy")
        if source_corners:
            bbox_source = source_corners_bbox(source_corners)
        else:
            x_center = safe_float(props.get("x_center"))
            y_center = safe_float(props.get("y_center"))
            rum_size = safe_float(props.get("rum_size_m"), 0.0)
            if x_center is None or y_center is None:
                raise ValueError(f"Footprint {rum_id} lacks source corners and center x/y")
            h = float(rum_size) / 2.0
            bbox_source = {
                "min_x": round(x_center - h, ROUND_SOURCE_XY_DIGITS),
                "min_y": round(y_center - h, ROUND_SOURCE_XY_DIGITS),
                "max_x": round(x_center + h, ROUND_SOURCE_XY_DIGITS),
                "max_y": round(y_center + h, ROUND_SOURCE_XY_DIGITS),
            }

        records.append({
            "rum_id": rum_id,
            "row_index": int(rum_index[rum_id]),
            "source_row": safe_int(props.get("source_row"), idx),
            "grid_i": int(gi),
            "grid_j": int(gj),
            "bbox_wgs84": bbox_wgs84,
            "bbox_source": bbox_source,
        })

    if missing_row_index:
        raise ValueError(f"Footprints missing packed row_index: {missing_row_index}")

    return records


def build_tiles(
    records: List[Dict[str, Any]],
    tile_cols: int,
    tile_rows: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    if not records:
        raise ValueError("No records to tile")

    min_i = min(r["grid_i"] for r in records)
    max_i = max(r["grid_i"] for r in records)
    min_j = min(r["grid_j"] for r in records)
    max_j = max(r["grid_j"] for r in records)

    i_ranges = tile_partition_edges(min_i, max_i, tile_cols)
    j_ranges = tile_partition_edges(min_j, max_j, tile_rows)

    tile_map: Dict[str, Dict[str, Any]] = {}

    for row_idx, (j0, j1) in enumerate(j_ranges):
        for col_idx, (i0, i1) in enumerate(i_ranges):
            tile_id = f"tile_r{row_idx:02d}_c{col_idx:02d}"
            tile_map[tile_id] = {
                "tile_id": tile_id,
                "tile_row": row_idx,
                "tile_col": col_idx,
                "grid_i_min": i0,
                "grid_i_max": i1,
                "grid_j_min": j0,
                "grid_j_max": j1,
                "rum_ids": [],
                "row_indices": [],
                "records": [],
            }

    for record in records:
        col_idx = find_tile_index(record["grid_i"], i_ranges)
        row_idx = find_tile_index(record["grid_j"], j_ranges)
        tile_id = f"tile_r{row_idx:02d}_c{col_idx:02d}"
        tile = tile_map[tile_id]
        tile["rum_ids"].append(record["rum_id"])
        tile["row_indices"].append(record["row_index"])
        tile["records"].append(record)

    tiles: List[Dict[str, Any]] = []

    for tile_id in sorted(tile_map):
        tile = tile_map[tile_id]
        records_in_tile = tile.pop("records")

        wgs_bboxes = [r["bbox_wgs84"] for r in records_in_tile]
        src_bboxes = [r["bbox_source"] for r in records_in_tile]

        tile["rum_count"] = len(tile["rum_ids"])
        tile["row_indices"] = sorted(tile["row_indices"])
        tile["rum_ids"] = [
            rid for _, rid in sorted(
                zip([r["row_index"] for r in records_in_tile], [r["rum_id"] for r in records_in_tile])
            )
        ]
        tile["bbox_wgs84"] = bbox_union_wgs84(wgs_bboxes)
        tile["bbox_source"] = bbox_union_source(src_bboxes)

        if KEEP_EMPTY_TILES or tile["rum_count"] > 0:
            tiles.append(tile)

    return tiles, {
        "grid_i_min": min_i,
        "grid_i_max": max_i,
        "grid_j_min": min_j,
        "grid_j_max": max_j,
        "tile_cols": len(i_ranges),
        "tile_rows": len(j_ranges),
        "i_ranges": i_ranges,
        "j_ranges": j_ranges,
    }


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    t_start = time.time()

    cfg = load_resolved_config(__file__)
    project_root = Path(cfg["_resolved"]["project_root"])

    generated = cfg["generated_outputs"]
    paths = cfg["paths"]
    tiling = cfg["tiling"]

    footprints_path = resolve_path(project_root, generated["rum_footprints"])
    packed_path = resolve_path(project_root, generated["packed_series"])
    tiles_dir = resolve_path(project_root, paths["tiles_dir"])

    tile_index_path = tiles_dir / TILE_INDEX_FILENAME
    tile_assignments_path = tiles_dir / TILE_ASSIGNMENTS_FILENAME

    tile_cols = int(tiling.get("tile_grid_cols", 8))
    tile_rows = int(tiling.get("tile_grid_rows", 6))

    section("Configuration")
    print(f"  Project root        : {project_root}")
    print(f"  Footprints input    : {footprints_path}")
    print(f"  Packed input        : {packed_path}")
    print(f"  Tile index output   : {tile_index_path}")
    print(f"  Assignments output  : {tile_assignments_path}")
    print(f"  Requested tile grid : {tile_cols} cols × {tile_rows} rows")

    section("Loading inputs")
    footprints = load_json(footprints_path)
    packed = load_json(packed_path)

    rum_index = packed.get("rum_index")
    if not isinstance(rum_index, dict) or not rum_index:
        raise ValueError("packed_series.json missing rum_index")

    packed_meta = packed.get("metadata") or {}
    epoch_count = int(packed_meta.get("epoch_count", len(packed.get("epochs", []))))
    rum_count = int(packed_meta.get("rum_count", len(rum_index)))

    ok(f"Loaded footprints and packed series")
    ok(f"Packed series: {rum_count} RUMs × {epoch_count} epochs")

    section("Building feature records")
    records = build_feature_records(footprints, rum_index)
    ok(f"Prepared {len(records)} footprint records with row indices")

    if len(records) != rum_count:
        warn(f"Record count differs from packed rum_count: records={len(records)}, packed={rum_count}")

    section("Partitioning spatial tiles")
    tiles, grid_summary = build_tiles(
        records=records,
        tile_cols=tile_cols,
        tile_rows=tile_rows,
    )

    non_empty_tiles = [t for t in tiles if t["rum_count"] > 0]
    empty_tiles = [t for t in tiles if t["rum_count"] == 0]

    max_tile_count = max((t["rum_count"] for t in tiles), default=0)
    min_nonempty_count = min((t["rum_count"] for t in non_empty_tiles), default=0)

    ok(f"Built tile index: {len(non_empty_tiles)} non-empty tiles, {len(empty_tiles)} empty tiles")
    print(f"  Tile grid actual     : {grid_summary['tile_cols']} cols × {grid_summary['tile_rows']} rows")
    print(f"  RUMs per non-empty tile: min={min_nonempty_count}, max={max_tile_count}")

    section("Writing tile index products")
    footprint_meta = footprints.get("metadata") or {}

    tile_index = {
        "metadata": {
            "schema": "tile_index_v1",
            "source_footprints": generated["rum_footprints"],
            "source_packed_series": generated["packed_series"],
            "rum_count": len(records),
            "epoch_count": epoch_count,
            "tile_count": len(tiles),
            "non_empty_tile_count": len(non_empty_tiles),
            "empty_tile_count": len(empty_tiles),
            "tile_grid_cols": grid_summary["tile_cols"],
            "tile_grid_rows": grid_summary["tile_rows"],
            "grid": grid_summary,
            "dataset_bbox_wgs84": footprint_meta.get("bbox_wgs84_footprints"),
            "dataset_bbox_source": footprint_meta.get("bbox_source_footprints"),
            "row_index_source": "packed_series.rum_index",
        },
        "tiles": tiles,
    }

    assignments = {
        "metadata": {
            "schema": "tile_assignments_v1",
            "source_tile_index": str(tile_index_path.relative_to(project_root).as_posix()) if tile_index_path.is_relative_to(project_root) else tile_index_path.as_posix(),
            "rum_count": len(records),
            "tile_count": len(tiles),
            "index_formula": "row_index from packed_series.rum_index",
        },
        "assignments": [
            {
                "rum_id": r["rum_id"],
                "row_index": r["row_index"],
                "source_row": r["source_row"],
                "grid_i": r["grid_i"],
                "grid_j": r["grid_j"],
                "tile_id": f"tile_r{find_tile_index(r['grid_j'], grid_summary['j_ranges']):02d}_c{find_tile_index(r['grid_i'], grid_summary['i_ranges']):02d}",
            }
            for r in sorted(records, key=lambda x: x["row_index"])
        ],
    }

    write_json(tile_index_path, tile_index)
    write_json(tile_assignments_path, assignments)

    elapsed = time.time() - t_start

    ok(f"Wrote tile index: {tile_index_path} ({tile_index_path.stat().st_size / 1024:.1f} KB)")
    ok(f"Wrote tile assignments: {tile_assignments_path} ({tile_assignments_path.stat().st_size / 1024:.1f} KB)")

    section("Summary")
    ok(f"Step 07 complete in {elapsed:.2f} s")
    print(f"  RUM count              : {len(records)}")
    print(f"  Tile count             : {len(tiles)}")
    print(f"  Non-empty tiles         : {len(non_empty_tiles)}")
    print(f"  Empty tiles             : {len(empty_tiles)}")
    print(f"  RUMs per tile           : {min_nonempty_count} to {max_tile_count}")


if __name__ == "__main__":
    main()
