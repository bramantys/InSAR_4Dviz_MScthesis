#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
14_build_walls_b3dm.py

Generic RUM-based InSAR template step.

Purpose
-------
Write two separate dynamic wall layers:

  1. real-real RUM walls:
      Data/tiles_walls_real/

  2. blank-related walls:
      Data/tiles_walls_blank/

Blank-related walls include:
  - real-blank
  - blank-real
  - blank-blank

This separation lets the viewer toggle:
  - RUM walls
  - blankie walls

Inputs:
  config.generated_outputs.rum_footprints
  config.generated_outputs.blank_cells
  config.generated_outputs.height_meta
  config.generated_outputs.packed_series
  Data/tiles/tile_assignments.json

Outputs:
  Data/tiles_walls_real/tileset.json
  Data/tiles_walls_real/<col>_<row>.b3dm
  Data/tiles_walls_real/wall_tile_assignments.json

  Data/tiles_walls_blank/tileset.json
  Data/tiles_walls_blank/<col>_<row>.b3dm
  Data/tiles_walls_blank/wall_tile_assignments.json

Wall runtime logic in viewer:
  heightA(epoch) = sampled from height_texture row A
  heightB(epoch) = sampled from height_texture row B

  wall bottom = min(heightA, heightB)
  wall top    = max(heightA, heightB)

Geometry:
  neutral wall quad baked between heights 0 m and 1 m.
  TEXCOORD_0.x = rowA / (n_rows - 1)
  TEXCOORD_0.y = rowB / (n_rows - 1)
  TEXCOORD_1.x = wall role: 0.0 lower edge, 1.0 upper edge
"""

from __future__ import annotations

import json
import math
import struct
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
# GEODESY / WALL HELPERS
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


def wall_normal_from_edge(p1: List[float], p2: List[float]) -> np.ndarray:
    lon_mid = 0.5 * (float(p1[0]) + float(p2[0]))
    lat_mid = 0.5 * (float(p1[1]) + float(p2[1]))

    e1 = wgs84_to_ecef(float(p1[0]), float(p1[1]), 0.0)
    e2 = wgs84_to_ecef(float(p2[0]), float(p2[1]), 0.0)
    edge = e2 - e1

    up = local_up(lon_mid, lat_mid)

    normal = np.cross(edge, up)
    n = np.linalg.norm(normal)

    if n < 1e-12:
        normal = np.cross(up, edge)
        n = np.linalg.norm(normal)

    if n < 1e-12:
        return up.astype(np.float64)

    return (normal / n).astype(np.float64)


def pad_to(data: bytes, alignment: int, pad_byte: bytes = b"\x00") -> bytes:
    r = len(data) % alignment
    if r == 0:
        return data
    return data + pad_byte * (alignment - r)


def cell_key(i: int, j: int) -> str:
    return f"{i},{j}"


def edge_points(corners: List[List[float]], side: str) -> Tuple[List[float], List[float]]:
    # corners are NE, NW, SW, SE
    ne, nw, sw, se = corners

    if side == "north":
        return nw, ne
    if side == "east":
        return ne, se
    if side == "south":
        return se, sw
    if side == "west":
        return sw, nw

    raise ValueError(f"Unknown side: {side}")


def midpoint_lonlat(a: List[float], b: List[float]) -> List[float]:
    return [
        (float(a[0]) + float(b[0])) * 0.5,
        (float(a[1]) + float(b[1])) * 0.5,
    ]


def active_tile_sort_key(key: str) -> Tuple[int, int]:
    col_s, row_s = key.split("_")
    return int(col_s), int(row_s)


def nullable_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        out = float(value)
        if not math.isfinite(out):
            return None
        return out
    except Exception:
        return None


# =============================================================================
# GLB / B3DM BUILDERS
# =============================================================================

def build_glb(
    positions_f32: np.ndarray,
    normals_f32: np.ndarray,
    batchids_f32: np.ndarray,
    texcoord0_f32: np.ndarray,
    texcoord1_f32: np.ndarray,
    indices_u32: np.ndarray,
) -> bytes:
    n_verts = len(positions_f32)
    n_indices = len(indices_u32)

    pos_bytes = pad_to(positions_f32.tobytes(), 4)
    norm_bytes = pad_to(normals_f32.tobytes(), 4)
    bid_bytes = pad_to(batchids_f32.tobytes(), 4)
    tc0_bytes = pad_to(texcoord0_f32.tobytes(), 4)
    tc1_bytes = pad_to(texcoord1_f32.tobytes(), 4)
    idx_bytes = pad_to(indices_u32.tobytes(), 4)

    bin_buf = pos_bytes + norm_bytes + bid_bytes + tc0_bytes + tc1_bytes + idx_bytes
    total_bin = len(bin_buf)

    off_pos = 0
    off_norm = off_pos + len(pos_bytes)
    off_bid = off_norm + len(norm_bytes)
    off_tc0 = off_bid + len(bid_bytes)
    off_tc1 = off_tc0 + len(tc0_bytes)
    off_idx = off_tc1 + len(tc1_bytes)

    pos_min = positions_f32.min(axis=0).tolist()
    pos_max = positions_f32.max(axis=0).tolist()

    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        # Same orientation matrix as cap layers.
        "nodes": [{"mesh": 0, "matrix": [1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 1]}],
        "meshes": [{"primitives": [{
            "attributes": {
                "POSITION": 0,
                "NORMAL": 1,
                "_BATCHID": 2,
                "TEXCOORD_0": 3,
                "TEXCOORD_1": 4,
            },
            "indices": 5,
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
                "componentType": 5126,
                "count": n_verts,
                "type": "VEC2",
            },
            {
                "bufferView": 5,
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
            {"buffer": 0, "byteOffset": off_tc1, "byteLength": len(tc1_bytes), "target": 34962},
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
    walls: List[Dict[str, Any]],
    n_rows: int,
    geometry_base_height_m: float,
    geometry_top_height_m: float,
) -> bytes:
    n = len(walls)

    all_pos: List[np.ndarray] = []
    all_norm: List[np.ndarray] = []
    all_bid: List[float] = []
    all_tc0: List[List[float]] = []
    all_tc1: List[List[float]] = []
    all_idx: List[int] = []

    for batch_id, wall in enumerate(walls):
        p1 = wall["p1"]
        p2 = wall["p2"]
        wall_norm = wall_normal_from_edge(p1, p2)

        row_a_f = float(wall["row_a"]) / float(max(n_rows - 1, 1))
        row_b_f = float(wall["row_b"]) / float(max(n_rows - 1, 1))

        base_v = len(all_pos)

        verts = [
            (p1, geometry_base_height_m, 0.0),
            (p2, geometry_base_height_m, 0.0),
            (p2, geometry_top_height_m, 1.0),
            (p1, geometry_top_height_m, 1.0),
        ]

        for lonlat, neutral_h, role in verts:
            lon = float(lonlat[0])
            lat = float(lonlat[1])
            pos = wgs84_to_ecef(lon, lat, neutral_h)
            all_pos.append(pos)
            all_norm.append(wall_norm)
            all_bid.append(float(batch_id))
            all_tc0.append([row_a_f, row_b_f])
            all_tc1.append([role, 0.0])

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
        texcoord1_f32=np.asarray(all_tc1, dtype=np.float32),
        indices_u32=np.asarray(all_idx, dtype=np.uint32),
    )

    ft_json_obj = {
        "BATCH_LENGTH": n,
        "RTC_CENTER": rtc.tolist(),
    }
    ft_json = pad_to(json.dumps(ft_json_obj, separators=(",", ":")).encode("utf-8"), 8, b" ")

    bt_json_obj = {
        "wall_id": [w["wall_id"] for w in walls],
        "cell_a": [w["cell_a"] for w in walls],
        "cell_b": [w["cell_b"] for w in walls],
        "kind_a": [w["kind_a"] for w in walls],
        "kind_b": [w["kind_b"] for w in walls],
        "row_a": [int(w["row_a"]) for w in walls],
        "row_b": [int(w["row_b"]) for w in walls],
        "up_a": [w.get("up_a") for w in walls],
        "up_b": [w.get("up_b") for w in walls],
        "up_high": [w.get("up_high") for w in walls],
        "side": [w["side"] for w in walls],
        "wall_layer": [w["wall_layer"] for w in walls],
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
# WALL LAYER WRITER
# =============================================================================

def write_wall_layer(
    layer_name: str,
    walls: List[Dict[str, Any]],
    out_dir: Path,
    grid_cols: int,
    grid_rows: int,
    n_rows: int,
    geometry_base_height_m: float,
    geometry_top_height_m: float,
    bound_min_height_m: float,
    bound_max_height_m: float,
    geometric_error_root: float,
    geometric_error_leaf: float,
    build_north_walls: bool,
    build_east_walls: bool,
    build_outer_walls: bool,
) -> Dict[str, Any]:
    section(f"Writing layer: {layer_name}")

    if not walls:
        warn(f"No walls for layer {layer_name}; skipping")
        return {
            "layer": layer_name,
            "wall_count": 0,
            "active_tiles": 0,
            "bytes": 0,
            "errors": 0,
        }

    lons = [float(w["center"][0]) for w in walls]
    lats = [float(w["center"][1]) for w in walls]

    lon_min = min(lons)
    lon_max = max(lons)
    lat_min = min(lats)
    lat_max = max(lats)

    root_lons: List[float] = []
    root_lats: List[float] = []

    for w in walls:
        for p in [w["p1"], w["p2"]]:
            root_lons.append(float(p[0]))
            root_lats.append(float(p[1]))

    root_lon_min = min(root_lons)
    root_lon_max = max(root_lons)
    root_lat_min = min(root_lats)
    root_lat_max = max(root_lats)

    lon_span = lon_max - lon_min
    lat_span = lat_max - lat_min

    if lon_span <= 0 or lat_span <= 0:
        # Degenerate layer fallback: one tile.
        grid_cols_eff = 1
        grid_rows_eff = 1
        lon_step = 1.0
        lat_step = 1.0
    else:
        grid_cols_eff = grid_cols
        grid_rows_eff = grid_rows
        lon_step = lon_span / grid_cols_eff
        lat_step = lat_span / grid_rows_eff

    tile_walls: Dict[str, List[Dict[str, Any]]] = {
        f"{col}_{row}": []
        for col in range(grid_cols_eff)
        for row in range(grid_rows_eff)
    }

    for wall in walls:
        lon = float(wall["center"][0])
        lat = float(wall["center"][1])

        if lon_span <= 0 or lat_span <= 0:
            col = 0
            row = 0
        else:
            col = min(max(int((lon - lon_min) / lon_step), 0), grid_cols_eff - 1)
            row = min(max(int((lat - lat_min) / lat_step), 0), grid_rows_eff - 1)

        tile_walls[f"{col}_{row}"].append(wall)

    active_tiles = {key: ws for key, ws in tile_walls.items() if ws}
    counts = sorted(len(ws) for ws in active_tiles.values())

    ok(f"{layer_name}: {len(walls)} walls → {len(active_tiles)} active tiles")
    print(f"  Walls per tile: min={counts[0]}, median={counts[len(counts)//2]}, max={counts[-1]}")

    tile_bbox: Dict[str, Dict[str, float]] = {}

    for key, ws in active_tiles.items():
        tlons: List[float] = []
        tlats: List[float] = []

        for w in ws:
            for p in [w["p1"], w["p2"]]:
                tlons.append(float(p[0]))
                tlats.append(float(p[1]))

        tile_bbox[key] = {
            "lon_min": min(tlons),
            "lon_max": max(tlons),
            "lat_min": min(tlats),
            "lat_max": max(tlats),
        }

    out_dir.mkdir(parents=True, exist_ok=True)

    total_bytes = 0
    errors = 0
    t0 = time.time()

    for i, (tile_key, ws) in enumerate(sorted(active_tiles.items(), key=lambda kv: active_tile_sort_key(kv[0]))):
        filename = f"{tile_key}.b3dm"
        out_path = out_dir / filename

        try:
            data = build_b3dm(
                walls=ws,
                n_rows=n_rows,
                geometry_base_height_m=geometry_base_height_m,
                geometry_top_height_m=geometry_top_height_m,
            )

            with out_path.open("wb") as f:
                f.write(data)

            total_bytes += len(data)

            if (i + 1) % 10 == 0 or (i + 1) == len(active_tiles):
                print(
                    f"  [{i+1:2d}/{len(active_tiles)}] {filename:<12s} "
                    f"{len(ws):4d} walls {len(data)/1024:8.1f} KB "
                    f"({time.time() - t0:.1f}s)"
                )
        except Exception as exc:
            errors += 1
            fail(f"{filename}: {exc}")
            import traceback
            traceback.print_exc()

    children: List[Dict[str, Any]] = []
    for key in sorted(active_tiles.keys(), key=active_tile_sort_key):
        b = tile_bbox[key]
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
            "content": {"uri": f"{key}.b3dm"},
        })

    tileset = {
        "asset": {
            "version": "1.0",
            "tilesetVersion": "1.0.0",
            "generator": f"RUM 4D Template — 14_build_walls_b3dm.py — {layer_name}",
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

    tileset_path = out_dir / "tileset.json"
    with tileset_path.open("w", encoding="utf-8") as f:
        json.dump(tileset, f, indent=2)

    assignments = {
        "metadata": {
            "schema_version": "wall_tile_assignments_v1",
            "layer": layer_name,
            "grid_cols": grid_cols_eff,
            "grid_rows": grid_rows_eff,
            "active_tiles": len(active_tiles),
            "wall_count": len(walls),
            "build_north_walls": build_north_walls,
            "build_east_walls": build_east_walls,
            "build_outer_walls": build_outer_walls,
            "created_by": "14_build_walls_b3dm.py",
            "created_unix": int(time.time()),
        },
        "tiles": {
            key: {
                "wall_count": len(active_tiles[key]),
                "bbox": tile_bbox[key],
                "b3dm_filename": f"{key}.b3dm",
                "wall_ids": [w["wall_id"] for w in active_tiles[key]],
            }
            for key in sorted(active_tiles.keys(), key=active_tile_sort_key)
        },
    }

    assignments_path = out_dir / "wall_tile_assignments.json"
    with assignments_path.open("w", encoding="utf-8") as f:
        json.dump(assignments, f, separators=(",", ":"))

    ok(f"Written tileset    : {tileset_path} ({tileset_path.stat().st_size / 1024:.1f} KB)")
    ok(f"Written assignments: {assignments_path} ({assignments_path.stat().st_size / 1024:.1f} KB)")

    if errors:
        warn(f"{errors} errors in layer {layer_name}")

    return {
        "layer": layer_name,
        "wall_count": len(walls),
        "active_tiles": len(active_tiles),
        "bytes": total_bytes,
        "errors": errors,
    }


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    t_start = time.time()

    cfg = load_config()

    generated = cfg.get("generated_outputs", {})
    paths_cfg = cfg.get("paths", {})
    tiling_cfg = cfg.get("tiling", {})
    walls_cfg = cfg.get("walls_b3dm", {})

    real_footprints_path = resolve_project_path(
        generated.get("rum_footprints", "Data/rum_footprints.json")
    )
    blank_cells_path = resolve_project_path(
        generated.get("blank_cells", "Data/blank_cells.json")
    )
    height_meta_path = resolve_project_path(
        generated.get("height_meta", "Data/tiles/height_meta.json")
    )
    packed_path = resolve_project_path(
        generated.get("packed_series", "Data/packed_series.json")
    )

    tiles_dir = resolve_project_path(paths_cfg.get("tiles_dir", "Data/tiles"))
    main_assignments_path = tiles_dir / "tile_assignments.json"

    out_real_dir = resolve_project_path(paths_cfg.get("real_walls_tiles_dir", "Data/tiles_walls_real"))
    out_blank_dir = resolve_project_path(paths_cfg.get("blank_walls_tiles_dir", "Data/tiles_walls_blank"))

    tile_grid_cols = int(tiling_cfg.get("tile_grid_cols", 8))
    tile_grid_rows = int(tiling_cfg.get("tile_grid_rows", 6))

    geometry_base_height_m = float(walls_cfg.get("geometry_base_height_m", 0.0))
    geometry_top_height_m = float(walls_cfg.get("geometry_top_height_m", 1.0))

    bound_min_height_m = float(tiling_cfg.get("tileset_bound_min_height_m", -1000.0))
    bound_max_height_m = float(tiling_cfg.get("tileset_bound_max_height_m", 10000.0))

    geometric_error_root = float(tiling_cfg.get("geometric_error_root", 5000.0))
    geometric_error_leaf = float(tiling_cfg.get("geometric_error_leaf", 100.0))

    build_north_walls = bool(walls_cfg.get("build_north_walls", True))
    build_east_walls = bool(walls_cfg.get("build_east_walls", True))
    build_outer_walls = bool(walls_cfg.get("build_outer_walls", False))

    section("Configuration")
    print(f"  Project root        : {PROJECT_DIR}")
    print(f"  Real footprints     : {real_footprints_path}")
    print(f"  Blank cells         : {blank_cells_path}")
    print(f"  Height meta         : {height_meta_path}")
    print(f"  Packed series       : {packed_path}")
    print(f"  Main assignments    : {main_assignments_path}")
    print(f"  Real walls output   : {out_real_dir}")
    print(f"  Blank walls output  : {out_blank_dir}")
    print(f"  Tile grid           : {tile_grid_cols} × {tile_grid_rows}")
    print(f"  Build north/east    : {build_north_walls} / {build_east_walls}")
    print(f"  Build outer walls   : {build_outer_walls}")

    section("Loading inputs")
    for path in [real_footprints_path, blank_cells_path, height_meta_path, packed_path, main_assignments_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required input: {path}")

    fp_data = load_json(real_footprints_path)
    blank_data = load_json(blank_cells_path)
    height_meta = load_json(height_meta_path)
    packed = load_json(packed_path)
    main_assignments = load_json(main_assignments_path)

    real_footprints = fp_data.get("footprints", {})
    blank_cells = blank_data.get("blank_cells", {})
    rum_index = height_meta.get("rum_index", {})
    blank_index = height_meta.get("blank_index", {})
    n_rows = int(height_meta.get("n_rows", height_meta.get("n_rums", 0)))
    packed_series = packed.get("series", {})

    grid_cols = int(main_assignments.get("metadata", {}).get("grid_cols", tile_grid_cols))
    grid_rows = int(main_assignments.get("metadata", {}).get("grid_rows", tile_grid_rows))

    if not real_footprints:
        raise ValueError("Real footprints file has no footprints")
    if not blank_cells:
        warn("Blank cells file has no blank_cells; only real-real walls will be generated")
    if not rum_index or n_rows <= 0:
        raise ValueError("Height meta has no rum_index/n_rows")
    if not packed_series:
        raise ValueError("Packed series has no series")

    ok(f"Real cells      : {len(real_footprints)}")
    ok(f"Blank cells     : {len(blank_cells)}")
    ok(f"Height rows     : {n_rows}")
    ok(f"Packed velocities: {len(packed_series)} real RUMs")
    ok(f"Tile grid       : {grid_cols} × {grid_rows}")

    section("Building combined cell topology")

    cells_by_key: Dict[str, Dict[str, Any]] = {}

    for rum_id, fp in real_footprints.items():
        i = int(fp["grid_i"])
        j = int(fp["grid_j"])
        key = cell_key(i, j)

        cells_by_key[key] = {
            "cell_id": str(rum_id),
            "kind": "real",
            "grid_i": i,
            "grid_j": j,
            "corners": fp["corners"],
            "center": fp["center"],
            "row_index": int(rum_index[str(rum_id)]),
            "up": nullable_float(packed_series.get(str(rum_id), {}).get("up")),
        }

    for blank_id, cell in blank_cells.items():
        i = int(cell["grid_i"])
        j = int(cell["grid_j"])
        key = cell_key(i, j)

        cells_by_key[key] = {
            "cell_id": str(blank_id),
            "kind": "blank",
            "grid_i": i,
            "grid_j": j,
            "corners": cell["corners"],
            "center": cell["center"],
            "row_index": int(blank_index[str(blank_id)]),
            "up": None,
        }

    ok(f"Combined cells: {len(cells_by_key)} ({len(real_footprints)} real + {len(blank_cells)} blank)")

    section("Building and splitting height-difference walls")
    walls_real: List[Dict[str, Any]] = []
    walls_blank: List[Dict[str, Any]] = []

    checks_template: List[Tuple[str, int, int]] = []
    if build_north_walls:
        checks_template.append(("north", 0, 1))
    if build_east_walls:
        checks_template.append(("east", 1, 0))

    for key, cell in cells_by_key.items():
        i = int(cell["grid_i"])
        j = int(cell["grid_j"])

        for side, di, dj in checks_template:
            nkey = cell_key(i + di, j + dj)
            neighbor = cells_by_key.get(nkey)

            if neighbor is None:
                # Outer boundary walls intentionally not built unless requested.
                if not build_outer_walls:
                    continue
                # Outer walls would require a synthetic outside row; not implemented by design.
                continue

            p1, p2 = edge_points(cell["corners"], side)
            mid = midpoint_lonlat(p1, p2)

            if cell["kind"] == "real" and neighbor["kind"] == "real":
                wall_layer = "real"
            else:
                wall_layer = "blank"

            up_a = cell.get("up")
            up_b = neighbor.get("up")

            if up_a is not None and up_b is not None:
                up_high = max(float(up_a), float(up_b))
            elif up_a is not None:
                up_high = float(up_a)
            elif up_b is not None:
                up_high = float(up_b)
            else:
                up_high = None

            wall = {
                "wall_id": f"WALL_{cell['cell_id']}__{side}__{neighbor['cell_id']}",
                "cell_a": cell["cell_id"],
                "cell_b": neighbor["cell_id"],
                "kind_a": cell["kind"],
                "kind_b": neighbor["kind"],
                "row_a": int(cell["row_index"]),
                "row_b": int(neighbor["row_index"]),
                "up_a": up_a,
                "up_b": up_b,
                "up_high": up_high,
                "grid_i": i,
                "grid_j": j,
                "side": side,
                "p1": p1,
                "p2": p2,
                "center": mid,
                "wall_layer": wall_layer,
            }

            if wall_layer == "real":
                walls_real.append(wall)
            else:
                walls_blank.append(wall)

    ok(f"Real-real walls      : {len(walls_real)}")
    ok(f"Blank-related walls  : {len(walls_blank)}")
    ok(f"Total internal walls : {len(walls_real) + len(walls_blank)}")

    res_real = write_wall_layer(
        layer_name="real_rum_walls",
        walls=walls_real,
        out_dir=out_real_dir,
        grid_cols=grid_cols,
        grid_rows=grid_rows,
        n_rows=n_rows,
        geometry_base_height_m=geometry_base_height_m,
        geometry_top_height_m=geometry_top_height_m,
        bound_min_height_m=bound_min_height_m,
        bound_max_height_m=bound_max_height_m,
        geometric_error_root=geometric_error_root,
        geometric_error_leaf=geometric_error_leaf,
        build_north_walls=build_north_walls,
        build_east_walls=build_east_walls,
        build_outer_walls=build_outer_walls,
    )

    res_blank = write_wall_layer(
        layer_name="blankie_walls",
        walls=walls_blank,
        out_dir=out_blank_dir,
        grid_cols=grid_cols,
        grid_rows=grid_rows,
        n_rows=n_rows,
        geometry_base_height_m=geometry_base_height_m,
        geometry_top_height_m=geometry_top_height_m,
        bound_min_height_m=bound_min_height_m,
        bound_max_height_m=bound_max_height_m,
        geometric_error_root=geometric_error_root,
        geometric_error_leaf=geometric_error_leaf,
        build_north_walls=build_north_walls,
        build_east_walls=build_east_walls,
        build_outer_walls=build_outer_walls,
    )

    section("SUMMARY")
    ok(
        f"Real RUM wall layer: {res_real['wall_count']} walls, "
        f"{res_real['active_tiles']} tiles, {res_real['bytes']/1024/1024:.2f} MB"
    )
    ok(
        f"Blankie wall layer : {res_blank['wall_count']} walls, "
        f"{res_blank['active_tiles']} tiles, {res_blank['bytes']/1024/1024:.2f} MB"
    )
    ok("Step 14 complete — dynamic wall layers written")
    ok("Next template step: 16_build_horizontal_field.py")
    print(f"  Elapsed: {time.time() - t_start:.2f}s")
    print()


if __name__ == "__main__":
    main()
