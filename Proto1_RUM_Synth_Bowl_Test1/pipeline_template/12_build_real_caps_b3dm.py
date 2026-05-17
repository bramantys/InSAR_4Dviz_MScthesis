#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
12_build_real_caps_b3dm.py

Generic RUM-based InSAR template step.

Purpose
-------
Write real RUM cap geometries as B3DM tiles.

Inputs:
  config.generated_outputs.rum_footprints
  config.generated_outputs.packed_series
  Data/tiles/tile_assignments.json
  config.generated_outputs.height_meta
  Data/tiles/tileset.json

Outputs:
  Data/tiles/<col>_<row>.b3dm
  Data/tiles/tileset.json updated/confirmed to reference .b3dm

Geometry:
  4 vertices per RUM cap, neutral baked height = 1.0 m.
  Runtime viewer shader moves cap vertices to:
    DISPLAY_DATUM_HEIGHT_M + disp_mm(epoch) * vertical_exaggeration

Vertex attributes:
  POSITION
  _BATCHID
  TEXCOORD_0  -> x = texture row fraction, y = is_top
  TEXCOORD_1  -> cap-local UV for procedural uncertainty hatch

TEXCOORD_1 corner order:
  NE=(1,1), NW=(0,1), SW=(0,0), SE=(1,0)
"""

from __future__ import annotations

import json
import math
import struct
import time
from pathlib import Path
from typing import Any, Dict, List

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
# GEODESY / BINARY HELPERS
# =============================================================================

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


def pad_to(data: bytes, alignment: int, pad_byte: bytes = b"\x00") -> bytes:
    r = len(data) % alignment
    if r == 0:
        return data
    return data + pad_byte * (alignment - r)


def update_region_heights(node: Dict[str, Any], h_min: float, h_max: float) -> None:
    bv = node.get("boundingVolume", {})
    region = bv.get("region")
    if isinstance(region, list) and len(region) == 6:
        region[4] = h_min
        region[5] = h_max

    for child in node.get("children", []):
        update_region_heights(child, h_min, h_max)


# =============================================================================
# GLB / B3DM BUILDERS
# =============================================================================

def build_glb(
    positions_f32: np.ndarray,
    batchids_f32: np.ndarray,
    texcoord0_f32: np.ndarray,
    texcoord1_f32: np.ndarray,
    indices_u32: np.ndarray,
) -> bytes:
    n_verts = len(positions_f32)
    n_indices = len(indices_u32)

    pos_bytes = pad_to(positions_f32.tobytes(), 4)
    bid_bytes = pad_to(batchids_f32.tobytes(), 4)
    tc0_bytes = pad_to(texcoord0_f32.tobytes(), 4)
    tc1_bytes = pad_to(texcoord1_f32.tobytes(), 4)
    idx_bytes = pad_to(indices_u32.tobytes(), 4)

    bin_buf = pos_bytes + bid_bytes + tc0_bytes + tc1_bytes + idx_bytes
    total_bin = len(bin_buf)

    off_pos = 0
    off_bid = off_pos + len(pos_bytes)
    off_tc0 = off_bid + len(bid_bytes)
    off_tc1 = off_tc0 + len(tc0_bytes)
    off_idx = off_tc1 + len(tc1_bytes)

    pos_min = positions_f32.min(axis=0).tolist()
    pos_max = positions_f32.max(axis=0).tolist()

    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        # Keep same orientation matrix as previous working cap/wall layers.
        "nodes": [{"mesh": 0, "matrix": [1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 1]}],
        "meshes": [{"primitives": [{
            "attributes": {
                "POSITION": 0,
                "_BATCHID": 1,
                "TEXCOORD_0": 2,
                "TEXCOORD_1": 3,
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
                "type": "SCALAR",
            },
            {
                "bufferView": 2,
                "byteOffset": 0,
                "componentType": 5126,
                "count": n_verts,
                "type": "VEC2",
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
    rum_ids: List[str],
    footprints: Dict[str, Any],
    packed_series: Dict[str, Any],
    rum_index_map: Dict[str, int],
    n_rows: int,
    geometry_top_height_m: float,
) -> bytes:
    n = len(rum_ids)

    all_pos: List[np.ndarray] = []
    all_bid: List[float] = []
    all_tc0: List[List[float]] = []
    all_tc1: List[List[float]] = []
    all_idx: List[int] = []

    cap_uvs = [
        [1.0, 1.0],  # NE
        [0.0, 1.0],  # NW
        [0.0, 0.0],  # SW
        [1.0, 0.0],  # SE
    ]

    for batch_id, rum_id in enumerate(rum_ids):
        fp = footprints[rum_id]
        corners = fp["corners"]

        rum_idx = int(rum_index_map[rum_id])
        rum_f = float(rum_idx) / float(max(n_rows - 1, 1))

        base_v = len(all_pos)

        for corner_idx, (lon, lat) in enumerate(corners):
            pos = wgs84_to_ecef(lon, lat, geometry_top_height_m)
            all_pos.append(pos)
            all_bid.append(float(batch_id))
            all_tc0.append([rum_f, 1.0])
            all_tc1.append(cap_uvs[corner_idx])

        # cap triangles: NE,NW,SW and NE,SW,SE
        b = base_v
        all_idx.extend([b + 0, b + 1, b + 2, b + 0, b + 2, b + 3])

    pos_array = np.asarray(all_pos, dtype=np.float64)
    rtc = pos_array.mean(axis=0)
    rel_pos = (pos_array - rtc).astype(np.float32)

    glb = build_glb(
        positions_f32=rel_pos,
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
        "rum_id": rum_ids,
        "rum_index": [int(rum_index_map[rid]) for rid in rum_ids],
        "up": [packed_series[rid]["up"] for rid in rum_ids],
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
    caps_cfg = cfg.get("caps_b3dm", {})
    tiling_cfg = cfg.get("tiling", {})

    footprints_path = resolve_project_path(
        generated.get("rum_footprints", "Data/rum_footprints.json")
    )
    packed_path = resolve_project_path(
        generated.get("packed_series", "Data/packed_series.json")
    )
    height_meta_path = resolve_project_path(
        generated.get("height_meta", "Data/tiles/height_meta.json")
    )

    tiles_dir = resolve_project_path(paths_cfg.get("tiles_dir", "Data/tiles"))
    assignments_path = tiles_dir / "tile_assignments.json"
    tileset_path = tiles_dir / "tileset.json"

    geometry_top_height_m = float(caps_cfg.get("geometry_top_height_m", 1.0))
    bound_min_height_m = float(tiling_cfg.get("tileset_bound_min_height_m", -1000.0))
    bound_max_height_m = float(tiling_cfg.get("tileset_bound_max_height_m", 10000.0))

    section("Configuration")
    print(f"  Project root       : {PROJECT_DIR}")
    print(f"  Footprints         : {footprints_path}")
    print(f"  Packed series      : {packed_path}")
    print(f"  Tile assignments   : {assignments_path}")
    print(f"  Height meta        : {height_meta_path}")
    print(f"  Tiles dir          : {tiles_dir}")
    print(f"  Tileset            : {tileset_path}")
    print(f"  Geometry top height: {geometry_top_height_m} m")

    section("Loading inputs")
    for path in [footprints_path, packed_path, assignments_path, height_meta_path, tileset_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required input: {path}")

    fp_data = load_json(footprints_path)
    packed_data = load_json(packed_path)
    assignments = load_json(assignments_path)
    height_meta = load_json(height_meta_path)

    footprints = fp_data.get("footprints", {})
    packed_series = packed_data.get("series", {})
    tiles = assignments.get("tiles", {})
    rum_index_map = height_meta.get("rum_index", {})
    n_rows = int(height_meta.get("n_rows", height_meta.get("n_rums", 0)))

    if not footprints:
        raise ValueError("Footprints file has no footprints")
    if not packed_series:
        raise ValueError("Packed series has no series")
    if not tiles:
        raise ValueError("Tile assignments has no tiles")
    if not rum_index_map or n_rows <= 0:
        raise ValueError("Height meta has no rum_index/n_rows")

    ok(f"Footprints       : {len(footprints)} RUMs")
    ok(f"Packed series    : {len(packed_series)} RUMs")
    ok(f"Tile assignments : {len(tiles)} tiles")
    ok(f"Height meta rows : {n_rows}")

    missing_index = [rid for rid in footprints if rid not in rum_index_map]
    if missing_index:
        raise RuntimeError(f"{len(missing_index)} footprints missing from height_meta.rum_index; sample={missing_index[:5]}")

    missing_series = [rid for rid in footprints if rid not in packed_series]
    if missing_series:
        raise RuntimeError(f"{len(missing_series)} footprints missing from packed series; sample={missing_series[:5]}")

    rum_fs = [float(rum_index_map[rid]) / float(max(n_rows - 1, 1)) for rid in footprints.keys()]
    ok(f"Texture row fraction range for real RUMs: {min(rum_fs):.6f} → {max(rum_fs):.6f}")

    section(f"Writing {len(tiles)} real cap .b3dm files")
    tiles_dir.mkdir(parents=True, exist_ok=True)

    total_bytes = 0
    errors = 0
    t0 = time.time()

    for k, (tile_key, tile) in enumerate(sorted(tiles.items(), key=lambda kv: tuple(int(x) for x in kv[0].split("_")))):
        rum_ids = [str(rid) for rid in tile.get("rum_ids", [])]
        out_path = tiles_dir / f"{tile_key}.b3dm"

        try:
            data = build_b3dm(
                rum_ids=rum_ids,
                footprints=footprints,
                packed_series=packed_series,
                rum_index_map=rum_index_map,
                n_rows=n_rows,
                geometry_top_height_m=geometry_top_height_m,
            )

            with out_path.open("wb") as f:
                f.write(data)

            total_bytes += len(data)

            if (k + 1) % 10 == 0 or (k + 1) == len(tiles):
                print(
                    f"  [{k+1:2d}/{len(tiles)}] {tile_key}.b3dm "
                    f"{len(rum_ids):4d} RUMs {len(data)/1024:8.1f} KB "
                    f"({time.time() - t0:.1f}s)"
                )
        except Exception as exc:
            errors += 1
            fail(f"{tile_key}.b3dm: {exc}")
            import traceback
            traceback.print_exc()

    section("Updating tileset.json")
    tileset = load_json(tileset_path)
    root = tileset.get("root", {})
    update_region_heights(root, bound_min_height_m, bound_max_height_m)

    def walk(node: Dict[str, Any]) -> None:
        content = node.get("content")
        if isinstance(content, dict) and "uri" in content:
            uri = str(content["uri"])
            if uri.endswith(".i3dm"):
                content["uri"] = uri[:-5] + ".b3dm"
            elif not uri.endswith(".b3dm") and "_" in uri:
                content["uri"] = Path(uri).with_suffix(".b3dm").as_posix()
        for child in node.get("children", []):
            walk(child)

    walk(root)

    tileset.setdefault("asset", {})
    tileset["asset"]["generator_real_caps"] = "RUM 4D Template — 12_build_real_caps_b3dm.py"

    with tileset_path.open("w", encoding="utf-8") as f:
        json.dump(tileset, f, indent=2)

    ok("tileset.json updated/confirmed: content.uri uses .b3dm")
    ok(f"Tileset height bounds: {bound_min_height_m} → {bound_max_height_m} m")

    section("SUMMARY")
    if errors == 0:
        ok(f"Step 12 complete — {len(tiles)} real cap tiles written ({total_bytes/1024/1024:.2f} MB)")
    else:
        warn(f"Step 12 completed with {errors} tile errors")
    ok("Next template step: 13_build_blank_caps_b3dm.py")
    print(f"  Elapsed: {time.time() - t_start:.2f}s")
    print()


if __name__ == "__main__":
    main()
