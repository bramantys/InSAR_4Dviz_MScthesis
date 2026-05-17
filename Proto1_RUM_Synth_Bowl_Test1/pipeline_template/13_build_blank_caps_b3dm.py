#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
13_build_blank_caps_b3dm.py

Generic RUM-based InSAR template step.

Purpose
-------
Write a separate B3DM tileset for synthetic blank/no-data cells ("blankies").

Inputs:
  config.generated_outputs.blank_cells
  config.generated_outputs.height_meta
  Data/tiles/tile_assignments.json

Outputs:
  Data/tiles_blank/tileset.json
  Data/tiles_blank/<col>_<row>.b3dm
  Data/tiles_blank/blank_tile_assignments.json

Geometry:
  cap-only quads, neutral baked height = 1.0 m.
  Viewer shader moves caps using the same height texture as real RUMs.

Texture row:
  TEXCOORD_0.x = blank_index / (n_rows - 1)
  TEXCOORD_0.y = 1.0
"""

from __future__ import annotations

import json
import math
import struct
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


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


def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


# =============================================================================
# CONFIG / IO HELPERS
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


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# GEODESY / TILE HELPERS
# =============================================================================

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


def wgs84_to_ecef(lon_deg: float, lat_deg: float, h_m: float = 0.0) -> np.ndarray:
    a = 6_378_137.0
    f = 1.0 / 298.257223563
    e2 = 2.0 * f - f * f

    lat = math.radians(max(min(float(lat_deg), 89.999999), -89.999999))
    lon = math.radians(float(lon_deg))
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)

    n = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)

    x = (n + h_m) * cos_lat * math.cos(lon)
    y = (n + h_m) * cos_lat * math.sin(lon)
    z = (n * (1.0 - e2) + h_m) * sin_lat
    return np.array([x, y, z], dtype=np.float64)


def local_up(lon_deg: float, lat_deg: float) -> np.ndarray:
    lat = math.radians(max(min(float(lat_deg), 89.999999), -89.999999))
    lon = math.radians(float(lon_deg))
    return np.array([
        math.cos(lat) * math.cos(lon),
        math.cos(lat) * math.sin(lon),
        math.sin(lat),
    ], dtype=np.float64)


def pad_to(data: bytes, alignment: int, pad_byte: bytes = b"\x00") -> bytes:
    r = len(data) % alignment
    if r == 0:
        return data
    return data + pad_byte * (alignment - r)


def active_tile_sort_key(key: str) -> Tuple[int, int]:
    col_s, row_s = key.split("_")
    return int(col_s), int(row_s)


# =============================================================================
# GLB / B3DM BUILDERS
# =============================================================================

def build_glb(
    positions_f32: np.ndarray,
    normals_f32: np.ndarray,
    batchids_f32: np.ndarray,
    texcoord0_f32: np.ndarray,
    indices_u32: np.ndarray,
) -> bytes:
    n_verts = len(positions_f32)
    n_indices = len(indices_u32)

    pos_bytes = pad_to(positions_f32.tobytes(), 4)
    norm_bytes = pad_to(normals_f32.tobytes(), 4)
    bid_bytes = pad_to(batchids_f32.tobytes(), 4)
    tc0_bytes = pad_to(texcoord0_f32.tobytes(), 4)
    idx_bytes = pad_to(indices_u32.tobytes(), 4)

    bin_buf = pos_bytes + norm_bytes + bid_bytes + tc0_bytes + idx_bytes
    total_bin = len(bin_buf)

    off_pos = 0
    off_norm = off_pos + len(pos_bytes)
    off_bid = off_norm + len(norm_bytes)
    off_tc0 = off_bid + len(bid_bytes)
    off_idx = off_tc0 + len(tc0_bytes)

    pos_min = positions_f32.min(axis=0).tolist()
    pos_max = positions_f32.max(axis=0).tolist()

    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "matrix": [1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 1]}],
        "meshes": [{"primitives": [{
            "attributes": {
                "POSITION": 0,
                "NORMAL": 1,
                "_BATCHID": 2,
                "TEXCOORD_0": 3,
            },
            "indices": 4,
            "mode": 4,
            "material": 0,
        }]}],
        "materials": [{
            "pbrMetallicRoughness": {
                "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 1.0,
            },
            "doubleSided": True,
        }],
        "accessors": [
            {
                "bufferView": 0,
                "byteOffset": 0,
                "componentType": 5126,
                "count": n_verts,
                "type": "VEC3",
                "min": [round(v, 4) for v in pos_min],
                "max": [round(v, 4) for v in pos_max],
            },
            {
                "bufferView": 1,
                "byteOffset": 0,
                "componentType": 5126,
                "count": n_verts,
                "type": "VEC3",
                "min": [-1.0, -1.0, -1.0],
                "max": [1.0, 1.0, 1.0],
            },
            {
                "bufferView": 2,
                "byteOffset": 0,
                "componentType": 5126,
                "count": n_verts,
                "type": "SCALAR",
            },
            {
                "bufferView": 3,
                "byteOffset": 0,
                "componentType": 5126,
                "count": n_verts,
                "type": "VEC2",
            },
            {
                "bufferView": 4,
                "byteOffset": 0,
                "componentType": 5125,
                "count": n_indices,
                "type": "SCALAR",
            },
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": off_pos, "byteLength": len(pos_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": off_norm, "byteLength": len(norm_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": off_bid, "byteLength": len(bid_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": off_tc0, "byteLength": len(tc0_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": off_idx, "byteLength": len(idx_bytes), "target": 34963},
        ],
        "buffers": [{"byteLength": total_bin}],
    }

    json_bytes = pad_to(json.dumps(gltf, separators=(",", ":")).encode("utf-8"), 4, b" ")

    json_type = 0x4E4F534A
    bin_type = 0x004E4942

    json_chunk = struct.pack("<II", len(json_bytes), json_type) + json_bytes
    bin_chunk = struct.pack("<II", total_bin, bin_type) + bin_buf

    total_len = 12 + len(json_chunk) + len(bin_chunk)
    header = b"glTF" + struct.pack("<II", 2, total_len)

    return header + json_chunk + bin_chunk


def build_b3dm(
    blank_ids: List[str],
    blank_cells: Dict[str, Any],
    blank_index: Dict[str, int],
    n_rows: int,
    geometry_top_height_m: float,
) -> bytes:
    n = len(blank_ids)

    all_pos: List[np.ndarray] = []
    all_norm: List[np.ndarray] = []
    all_bid: List[float] = []
    all_tc0: List[List[float]] = []
    all_idx: List[int] = []

    for batch_id, blank_id in enumerate(blank_ids):
        cell = blank_cells[blank_id]
        corners = cell["corners"]

        row_idx = int(blank_index[blank_id])
        row_f = float(row_idx) / float(max(n_rows - 1, 1))

        base_v = len(all_pos)

        for lon, lat in corners:
            pos = wgs84_to_ecef(lon, lat, geometry_top_height_m)
            norm = local_up(lon, lat)
            all_pos.append(pos)
            all_norm.append(norm)
            all_bid.append(float(batch_id))
            all_tc0.append([row_f, 1.0])

        b = base_v
        all_idx.extend([b + 0, b + 1, b + 2, b + 0, b + 2, b + 3])

    pos_array = np.asarray(all_pos, dtype=np.float64)
    rtc = pos_array.mean(axis=0)
    rel_pos = (pos_array - rtc).astype(np.float32)

    glb = build_glb(
        positions_f32=rel_pos,
        normals_f32=np.asarray(all_norm, dtype=np.float32),
        batchids_f32=np.asarray(all_bid, dtype=np.float32),
        texcoord0_f32=np.asarray(all_tc0, dtype=np.float32),
        indices_u32=np.asarray(all_idx, dtype=np.uint32),
    )

    ft_json_obj = {
        "BATCH_LENGTH": n,
        "RTC_CENTER": rtc.tolist(),
    }
    ft_json = pad_to(json.dumps(ft_json_obj, separators=(",", ":")).encode("utf-8"), 8, b" ")

    bt_json_obj = {
        "blank_id": blank_ids,
        "cell_kind": ["blank"] * n,
        "grid_i": [int(blank_cells[bid]["grid_i"]) for bid in blank_ids],
        "grid_j": [int(blank_cells[bid]["grid_j"]) for bid in blank_ids],
        "row_index": [int(blank_index[bid]) for bid in blank_ids],
    }
    bt_json = pad_to(json.dumps(bt_json_obj, separators=(",", ":")).encode("utf-8"), 8, b" ")

    total_size = 28 + len(ft_json) + len(bt_json) + len(glb)

    header = struct.pack(
        "<4sIIIIII",
        b"b3dm",
        1,
        total_size,
        len(ft_json),
        0,
        len(bt_json),
        0,
    )

    return header + ft_json + bt_json + glb


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    t_start = time.time()

    cfg = load_config()

    generated = cfg.get("generated_outputs", {})
    paths_cfg = cfg.get("paths", {})
    tiling_cfg = cfg.get("tiling", {})
    caps_cfg = cfg.get("caps_b3dm", {})

    blank_cells_path = resolve_project_path(
        generated.get("blank_cells", "Data/blank_cells.json")
    )
    height_meta_path = resolve_project_path(
        generated.get("height_meta", "Data/tiles/height_meta.json")
    )

    tiles_dir = resolve_project_path(paths_cfg.get("tiles_dir", "Data/tiles"))
    main_assignments_path = tiles_dir / "tile_assignments.json"

    out_tiles_dir = resolve_project_path(paths_cfg.get("blank_tiles_dir", "Data/tiles_blank"))
    out_tileset_path = out_tiles_dir / "tileset.json"
    out_assignments_path = out_tiles_dir / "blank_tile_assignments.json"

    tile_grid_cols = int(tiling_cfg.get("tile_grid_cols", 8))
    tile_grid_rows = int(tiling_cfg.get("tile_grid_rows", 6))

    geometry_top_height_m = float(caps_cfg.get("geometry_top_height_m", 1.0))
    bound_min_height_m = float(tiling_cfg.get("tileset_bound_min_height_m", -1000.0))
    bound_max_height_m = float(tiling_cfg.get("tileset_bound_max_height_m", 10000.0))

    geometric_error_root = float(tiling_cfg.get("geometric_error_root", 5000.0))
    geometric_error_leaf = float(tiling_cfg.get("geometric_error_leaf", 100.0))

    section("Configuration")
    print(f"  Project root       : {PROJECT_DIR}")
    print(f"  Blank cells        : {blank_cells_path}")
    print(f"  Height meta        : {height_meta_path}")
    print(f"  Main assignments   : {main_assignments_path}")
    print(f"  Output dir         : {out_tiles_dir}")
    print(f"  Output tileset     : {out_tileset_path}")
    print(f"  Output assignments : {out_assignments_path}")
    print(f"  Tile grid          : {tile_grid_cols} × {tile_grid_rows}")
    print(f"  Geometry top height: {geometry_top_height_m} m")

    section("Loading inputs")
    for path in [blank_cells_path, height_meta_path, main_assignments_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required input: {path}")

    blank_data = load_json(blank_cells_path)
    height_meta = load_json(height_meta_path)
    main_assignments = load_json(main_assignments_path)

    blank_cells = blank_data.get("blank_cells", {})
    blank_index = height_meta.get("blank_index", {})
    n_rows = int(height_meta.get("n_rows", height_meta.get("n_rums", 0)))

    if not blank_cells:
        raise ValueError("Blank cells file has no blank_cells")
    if not blank_index or n_rows <= 0:
        raise ValueError("Height meta has no blank_index/n_rows")

    grid_cols = int(main_assignments.get("metadata", {}).get("grid_cols", tile_grid_cols))
    grid_rows = int(main_assignments.get("metadata", {}).get("grid_rows", tile_grid_rows))

    ok(f"Blank cells : {len(blank_cells)}")
    ok(f"Blank index : {len(blank_index)}")
    ok(f"n_rows      : {n_rows}")
    ok(f"Tile grid   : {grid_cols} × {grid_rows}")

    missing_index = [bid for bid in blank_cells if bid not in blank_index]
    if missing_index:
        raise RuntimeError(f"{len(missing_index)} blank cells missing from height_meta.blank_index; sample={missing_index[:5]}")

    section("Assigning blank cells to tile grid")

    lons = [float(cell["center"][0]) for cell in blank_cells.values()]
    lats = [float(cell["center"][1]) for cell in blank_cells.values()]

    lon_min = min(lons)
    lon_max = max(lons)
    lat_min = min(lats)
    lat_max = max(lats)

    corner_lons: List[float] = []
    corner_lats: List[float] = []
    for cell in blank_cells.values():
        for lon, lat in cell.get("corners", []):
            corner_lons.append(float(lon))
            corner_lats.append(float(lat))

    root_lon_min = min(corner_lons)
    root_lon_max = max(corner_lons)
    root_lat_min = min(corner_lats)
    root_lat_max = max(corner_lats)

    lon_span = lon_max - lon_min
    lat_span = lat_max - lat_min
    if lon_span <= 0 or lat_span <= 0:
        raise ValueError("Blank cell bbox has non-positive lon/lat span")

    lon_step = lon_span / grid_cols
    lat_step = lat_span / grid_rows

    tile_blanks: Dict[str, List[str]] = {
        f"{col}_{row}": []
        for col in range(grid_cols)
        for row in range(grid_rows)
    }

    for blank_id, cell in blank_cells.items():
        lon, lat = float(cell["center"][0]), float(cell["center"][1])
        col = min(max(int((lon - lon_min) / lon_step), 0), grid_cols - 1)
        row = min(max(int((lat - lat_min) / lat_step), 0), grid_rows - 1)
        tile_blanks[f"{col}_{row}"].append(str(blank_id))

    active_tiles = {key: ids for key, ids in tile_blanks.items() if ids}
    counts = sorted(len(ids) for ids in active_tiles.values())

    ok(f"Assigned {len(blank_cells)} blank cells to {len(active_tiles)} active blank tiles")
    print(f"  Blank cells per tile: min={counts[0]}, median={counts[len(counts)//2]}, max={counts[-1]}")

    tile_bbox: Dict[str, Dict[str, float]] = {}
    for key, ids in active_tiles.items():
        tlons: List[float] = []
        tlats: List[float] = []

        for bid in ids:
            for lon, lat in blank_cells[bid].get("corners", []):
                tlons.append(float(lon))
                tlats.append(float(lat))

        tile_bbox[key] = {
            "lon_min": min(tlons),
            "lon_max": max(tlons),
            "lat_min": min(tlats),
            "lat_max": max(tlats),
        }

    section(f"Writing {len(active_tiles)} blank cap .b3dm files")
    out_tiles_dir.mkdir(parents=True, exist_ok=True)

    total_bytes = 0
    errors = 0
    t0 = time.time()

    for i, (tile_key, blank_ids) in enumerate(sorted(active_tiles.items(), key=lambda kv: active_tile_sort_key(kv[0]))):
        filename = f"{tile_key}.b3dm"
        out_path = out_tiles_dir / filename

        try:
            data = build_b3dm(
                blank_ids=blank_ids,
                blank_cells=blank_cells,
                blank_index=blank_index,
                n_rows=n_rows,
                geometry_top_height_m=geometry_top_height_m,
            )

            with out_path.open("wb") as f:
                f.write(data)

            total_bytes += len(data)

            if (i + 1) % 10 == 0 or (i + 1) == len(active_tiles):
                print(
                    f"  [{i+1:2d}/{len(active_tiles)}] {filename:<12s} "
                    f"{len(blank_ids):3d} blanks {len(data)/1024:8.1f} KB "
                    f"({time.time() - t0:.1f}s)"
                )
        except Exception as exc:
            errors += 1
            fail(f"{filename}: {exc}")
            import traceback
            traceback.print_exc()

    section("Writing blank tileset.json")

    children: List[Dict[str, Any]] = []
    for key in sorted(active_tiles.keys(), key=active_tile_sort_key):
        b = tile_bbox[key]
        children.append({
            "boundingVolume": region_bounding_volume(
                b["lon_min"], b["lon_max"],
                b["lat_min"], b["lat_max"],
                bound_min_height_m,
                bound_max_height_m,
            ),
            "geometricError": geometric_error_leaf,
            "refine": "REPLACE",
            "content": {
                "uri": f"{key}.b3dm",
            },
        })

    tileset = {
        "asset": {
            "version": "1.0",
            "tilesetVersion": "1.0.0",
            "generator": "RUM 4D Template — 13_build_blank_caps_b3dm.py",
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

    with out_tileset_path.open("w", encoding="utf-8") as f:
        json.dump(tileset, f, indent=2)

    assignments = {
        "metadata": {
            "schema_version": "blank_tile_assignments_v1",
            "grid_cols": grid_cols,
            "grid_rows": grid_rows,
            "active_tiles": len(active_tiles),
            "blank_count": len(blank_cells),
            "source_blank_cells": str(blank_cells_path),
            "source_height_meta": str(height_meta_path),
            "created_by": "13_build_blank_caps_b3dm.py",
            "created_unix": int(time.time()),
        },
        "tiles": {
            key: {
                "blank_ids": active_tiles[key],
                "blank_count": len(active_tiles[key]),
                "bbox": tile_bbox[key],
                "b3dm_filename": f"{key}.b3dm",
            }
            for key in sorted(active_tiles.keys(), key=active_tile_sort_key)
        },
    }

    with out_assignments_path.open("w", encoding="utf-8") as f:
        json.dump(assignments, f, separators=(",", ":"))

    ok(f"Written tileset    : {out_tileset_path} ({out_tileset_path.stat().st_size / 1024:.1f} KB)")
    ok(f"Written assignments: {out_assignments_path} ({out_assignments_path.stat().st_size / 1024:.1f} KB)")

    section("SUMMARY")
    if errors == 0:
        ok(f"Step 13 complete — {len(active_tiles)} blank cap tiles written ({total_bytes/1024/1024:.2f} MB)")
    else:
        warn(f"Step 13 completed with {errors} tile errors")
    ok("Next template step: 14_build_walls_b3dm.py")
    print(f"  Elapsed: {time.time() - t_start:.2f}s")
    print()


if __name__ == "__main__":
    main()
