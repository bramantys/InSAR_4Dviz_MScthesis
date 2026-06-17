#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
12_build_blank_caps_b3dm.py

InSAR4D RUM Viewer pipeline step 12.

Purpose
-------
Build B3DM tiles for blank/no-data RUM cap polygons.

Inputs
------
  generated_outputs.blank_cells
    _internal/data_pipeline/blank_cells.json

  generated_outputs.height_meta
    _internal/data_pipeline/tiles/height_meta.json

  _internal/data_pipeline/tiles/tile_index.json
    created by Step 07, used as reference tile grid.

Outputs
-------
  generated_outputs.blank_caps_tileset
    _internal/data_pipeline/tiles_blank/tileset.json

  _internal/data_pipeline/tiles_blank/*.b3dm

Critical rule
-------------
No blank cells is NOT a failure.

If blank_count == 0, this step writes a valid empty tileset with no children
and returns OK.

Geometry contract
-----------------
Each blank cap is a flat polygon at viewer.display_datum_height_m.

Each vertex stores:
  POSITION   = local ENU position in tile coordinates
  NORMAL     = [0, 0, 1]
  TEXCOORD_0 = [0.0, row_v]
  _BATCHID   = integer-like float feature id inside the tile

where:
  row_v = (height_texture_blank_row + 0.5) / height_texture_height

Blank row indices are:
  blank_start_row + blank_index

from height_meta.json.

Picking contract
----------------
Each blank cap writes a legacy 3D Tiles batch table with placeholder feature
properties. Cesium scene.pick(...).getProperty("feature_kind") should return
"blankie" for blank caps. The viewer can then show a placeholder popup:
"blankie RUM" with unavailable values.
"""

from __future__ import annotations

import json
import math
import struct
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pipeline_config import load_resolved_config, resolve_path


# =============================================================================
# ADVANCED SETTINGS
# =============================================================================
# Normal users should edit config/project_config.json, not this script.

POSITION_COMPONENT_TYPE = 5126  # FLOAT
NORMAL_COMPONENT_TYPE = 5126    # FLOAT
TEXCOORD_COMPONENT_TYPE = 5126  # FLOAT
BATCHID_COMPONENT_TYPE = 5126   # FLOAT, safest legacy _BATCHID for Cesium B3DM picking
INDEX_COMPONENT_TYPE = 5125     # UNSIGNED_INT

MATERIAL_BASE_COLOR = [1.0, 1.0, 1.0, 0.55]
DOUBLE_SIDED = True
CAP_CLEARANCE_M = 0.05

CLEAN_OLD_BLANK_CAP_B3DM = True

# B3DM feature/batch table. Batch length is set per tile.
# This is what makes scene.pick(...).getProperty("feature_kind") work.

# If blank cells exist, group them by the same tile-row/tile-col ranges as the
# real tile index where possible. This keeps spatial organization consistent.
TILE_INDEX_FILENAME = "tile_index.json"


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
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_binary(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(payload)


def pad_bytes(data: bytes, multiple: int, pad_byte: bytes = b" ") -> bytes:
    remainder = len(data) % multiple
    if remainder == 0:
        return data
    return data + pad_byte * (multiple - remainder)


def rel_path(project_root: Path, path: Path) -> str:
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return path.as_posix()


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


def deg_to_rad(value: float) -> float:
    return float(value) * math.pi / 180.0


# =============================================================================
# WGS84 / ECEF / ENU
# =============================================================================

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


def geodetic_to_ecef(lon_deg: float, lat_deg: float, height_m: float) -> Tuple[float, float, float]:
    lon = deg_to_rad(lon_deg)
    lat = deg_to_rad(lat_deg)

    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)

    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)

    x = (n + height_m) * cos_lat * cos_lon
    y = (n + height_m) * cos_lat * sin_lon
    z = (n * (1.0 - WGS84_E2) + height_m) * sin_lat

    return x, y, z


def enu_basis(lon_deg: float, lat_deg: float) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]:
    lon = deg_to_rad(lon_deg)
    lat = deg_to_rad(lat_deg)

    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)

    east = (-sin_lon, cos_lon, 0.0)
    north = (-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat)
    up = (cos_lat * cos_lon, cos_lat * sin_lon, sin_lat)

    return east, north, up


def dot(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def ecef_to_local_enu(
    ecef: Tuple[float, float, float],
    center_ecef: Tuple[float, float, float],
    east: Tuple[float, float, float],
    north: Tuple[float, float, float],
    up: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    d = (
        ecef[0] - center_ecef[0],
        ecef[1] - center_ecef[1],
        ecef[2] - center_ecef[2],
    )
    return dot(d, east), dot(d, north), dot(d, up)


def enu_to_ecef_transform_column_major(lon_deg: float, lat_deg: float, height_m: float = 0.0) -> List[float]:
    center = geodetic_to_ecef(lon_deg, lat_deg, height_m)
    east, north, up = enu_basis(lon_deg, lat_deg)

    return [
        east[0], east[1], east[2], 0.0,
        north[0], north[1], north[2], 0.0,
        up[0], up[1], up[2], 0.0,
        center[0], center[1], center[2], 1.0,
    ]


# =============================================================================
# GLB / B3DM BUILDING
# =============================================================================

def pack_floats(values: List[float]) -> bytes:
    return struct.pack("<" + "f" * len(values), *values)


def pack_uint32(values: List[int]) -> bytes:
    return struct.pack("<" + "I" * len(values), *values)


def component_min_max_vec3(values: List[float]) -> Tuple[List[float], List[float]]:
    xs = values[0::3]
    ys = values[1::3]
    zs = values[2::3]
    return [min(xs), min(ys), min(zs)], [max(xs), max(ys), max(zs)]


def append_aligned(buffer: bytearray, data: bytes, alignment: int = 4, pad_byte: bytes = b"\x00") -> Tuple[int, int]:
    offset = len(buffer)
    padding = (alignment - (offset % alignment)) % alignment
    if padding:
        buffer.extend(pad_byte * padding)
        offset += padding
    buffer.extend(data)
    return offset, len(data)


def build_glb(
    positions: List[float],
    normals: List[float],
    texcoords: List[float],
    batchids: List[float],
    indices: List[int],
) -> bytes:
    if not positions or not indices:
        raise ValueError("Cannot build GLB with empty positions/indices")

    vertex_count = len(positions) // 3
    index_count = len(indices)

    if len(normals) != vertex_count * 3:
        raise ValueError("normal length mismatch")
    if len(texcoords) != vertex_count * 2:
        raise ValueError("texcoord length mismatch")
    if len(batchids) != vertex_count:
        raise ValueError("batchid length mismatch")

    bin_buffer = bytearray()

    pos_offset, pos_len = append_aligned(bin_buffer, pack_floats(positions), 4)
    normal_offset, normal_len = append_aligned(bin_buffer, pack_floats(normals), 4)
    texcoord_offset, texcoord_len = append_aligned(bin_buffer, pack_floats(texcoords), 4)
    batchid_offset, batchid_len = append_aligned(bin_buffer, pack_floats(batchids), 4)
    index_offset, index_len = append_aligned(bin_buffer, pack_uint32(indices), 4)

    pos_min, pos_max = component_min_max_vec3(positions)

    gltf = {
        "asset": {
            "version": "2.0",
            "generator": "InSAR4D RUM Viewer pipeline step 12",
        },
        "buffers": [{"byteLength": len(bin_buffer)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": pos_offset, "byteLength": pos_len, "target": 34962},
            {"buffer": 0, "byteOffset": normal_offset, "byteLength": normal_len, "target": 34962},
            {"buffer": 0, "byteOffset": texcoord_offset, "byteLength": texcoord_len, "target": 34962},
            {"buffer": 0, "byteOffset": batchid_offset, "byteLength": batchid_len, "target": 34962},
            {"buffer": 0, "byteOffset": index_offset, "byteLength": index_len, "target": 34963},
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": POSITION_COMPONENT_TYPE,
                "count": vertex_count,
                "type": "VEC3",
                "min": pos_min,
                "max": pos_max,
            },
            {
                "bufferView": 1,
                "componentType": NORMAL_COMPONENT_TYPE,
                "count": vertex_count,
                "type": "VEC3",
            },
            {
                "bufferView": 2,
                "componentType": TEXCOORD_COMPONENT_TYPE,
                "count": vertex_count,
                "type": "VEC2",
            },
            {
                "bufferView": 3,
                "componentType": BATCHID_COMPONENT_TYPE,
                "count": vertex_count,
                "type": "SCALAR",
                "min": [min(batchids)],
                "max": [max(batchids)],
            },
            {
                "bufferView": 4,
                "componentType": INDEX_COMPONENT_TYPE,
                "count": index_count,
                "type": "SCALAR",
                "min": [min(indices)],
                "max": [max(indices)],
            },
        ],
        "materials": [
            {
                "pbrMetallicRoughness": {
                    "baseColorFactor": MATERIAL_BASE_COLOR,
                    "metallicFactor": 0.0,
                    "roughnessFactor": 1.0,
                },
                "alphaMode": "BLEND" if MATERIAL_BASE_COLOR[3] < 1.0 else "OPAQUE",
                "doubleSided": DOUBLE_SIDED,
            }
        ],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": 0,
                            "NORMAL": 1,
                            "TEXCOORD_0": 2,
                            "_BATCHID": 3,
                        },
                        "indices": 4,
                        "material": 0,
                        "mode": 4,
                    }
                ]
            }
        ],

        "nodes": [{
            "mesh": 0,
            # z-up → y-up correction matrix (column-major)
            # cancels Cesium's automatic y-up → z-up transform at runtime
            "matrix": [1,0,0,0, 0,0,-1,0, 0,1,0,0, 0,0,0,1],
        }],

        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }

    json_chunk = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    json_chunk = pad_bytes(json_chunk, 4, b" ")
    bin_chunk = pad_bytes(bytes(bin_buffer), 4, b"\x00")

    total_length = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)

    glb = bytearray()
    glb.extend(struct.pack("<4sII", b"glTF", 2, total_length))
    glb.extend(struct.pack("<I4s", len(json_chunk), b"JSON"))
    glb.extend(json_chunk)
    glb.extend(struct.pack("<I4s", len(bin_chunk), b"BIN\x00"))
    glb.extend(bin_chunk)

    return bytes(glb)


def build_b3dm(
    glb: bytes,
    batch_length: int,
    batch_table: Dict[str, List[Any]],
) -> bytes:
    """
    Build legacy B3DM with a per-feature batch table.

    The _BATCHID vertex attribute in the glTF points into this batch table, so
    Cesium feature.getProperty("feature_kind") and friends work after scene.pick().
    """
    feature_table_json = json.dumps(
        {"BATCH_LENGTH": int(batch_length)},
        separators=(",", ":"),
    ).encode("utf-8")
    feature_table_json = pad_bytes(feature_table_json, 8, b" ")

    feature_table_binary = b""

    batch_table_json = json.dumps(
        batch_table,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    batch_table_json = pad_bytes(batch_table_json, 8, b" ")

    batch_table_binary = b""

    byte_length = (
        28
        + len(feature_table_json)
        + len(feature_table_binary)
        + len(batch_table_json)
        + len(batch_table_binary)
        + len(glb)
    )

    header = struct.pack(
        "<4sIIIIII",
        b"b3dm",
        1,
        byte_length,
        len(feature_table_json),
        len(feature_table_binary),
        len(batch_table_json),
        len(batch_table_binary),
    )

    return header + feature_table_json + feature_table_binary + batch_table_json + batch_table_binary + glb


# =============================================================================
# BLANK DATA / TILE GROUPING
# =============================================================================

def tile_center_from_bbox(bbox: Dict[str, float]) -> Tuple[float, float]:
    lon = (float(bbox["west"]) + float(bbox["east"])) / 2.0
    lat = (float(bbox["south"]) + float(bbox["north"])) / 2.0
    return lon, lat


def bounding_region_from_bbox(
    bbox: Dict[str, float],
    min_height_m: float,
    max_height_m: float,
) -> List[float]:
    return [
        deg_to_rad(float(bbox["west"])),
        deg_to_rad(float(bbox["south"])),
        deg_to_rad(float(bbox["east"])),
        deg_to_rad(float(bbox["north"])),
        float(min_height_m),
        float(max_height_m),
    ]


def bbox_union_wgs84(bboxes: List[Dict[str, float]]) -> Dict[str, float]:
    return {
        "west": min(float(b["west"]) for b in bboxes),
        "south": min(float(b["south"]) for b in bboxes),
        "east": max(float(b["east"]) for b in bboxes),
        "north": max(float(b["north"]) for b in bboxes),
    }



def center_from_ring_or_bbox(ring_lonlat: List[List[float]], bbox: Dict[str, float]) -> Tuple[float, float]:
    """Return a robust lon/lat center for popup diagnostics."""
    try:
        corners = ring_lonlat[:-1] if ring_lonlat and ring_lonlat[0] == ring_lonlat[-1] else ring_lonlat
        if corners:
            lon = sum(float(p[0]) for p in corners) / len(corners)
            lat = sum(float(p[1]) for p in corners) / len(corners)
            if math.isfinite(lon) and math.isfinite(lat):
                return lon, lat
    except Exception:
        pass

    return tile_center_from_bbox(bbox)


def build_batch_table_for_blanks(
    records: List[Dict[str, Any]],
    blank_start_row: int,
) -> Dict[str, List[Any]]:
    """
    Legacy B3DM batch table for blank cap picking.

    Blankies are no-data/support cells, not measured RUMs. They therefore get
    a placeholder display id and placeholder values. The viewer can use
    feature_kind/is_blank to show a simple "blankie RUM" popup.
    """
    table: Dict[str, List[Any]] = {
        "feature_kind": [],
        "is_blank": [],
        "rum_id": [],
        "display_id": [],
        "blank_id": [],
        "blank_index": [],
        "row_index": [],
        "height_row": [],
        "grid_i": [],
        "grid_j": [],
        "lon_center": [],
        "lat_center": [],
        "up": [],
        "up_mm_yr": [],
        "east_mm_yr": [],
        "north_mm_yr": [],
        "speed_mm_yr": [],
        "var_up": [],
        "var_east": [],
        "var_north": [],
        "covar_en": [],
    }

    for rec in records:
        blank_index = int(rec["blank_index"])
        row_index = int(blank_start_row) + blank_index

        # Use "-" rather than null for unavailable numeric values because
        # Number(null) becomes 0 in JavaScript; Number("-") becomes NaN and
        # the popup formatter displays it as unavailable.
        unavailable = "-"

        table["feature_kind"].append("blankie")
        table["is_blank"].append(True)
        table["rum_id"].append("blankie RUM")
        table["display_id"].append("blankie RUM")
        table["blank_id"].append(str(rec["blank_id"]))
        table["blank_index"].append(blank_index)
        table["row_index"].append(row_index)
        table["height_row"].append(row_index)
        table["grid_i"].append(int(rec["grid_i"]))
        table["grid_j"].append(int(rec["grid_j"]))
        table["lon_center"].append(safe_float(rec.get("lon_center")))
        table["lat_center"].append(safe_float(rec.get("lat_center")))
        table["up"].append(unavailable)
        table["up_mm_yr"].append(unavailable)
        table["east_mm_yr"].append(unavailable)
        table["north_mm_yr"].append(unavailable)
        table["speed_mm_yr"].append(unavailable)
        table["var_up"].append(unavailable)
        table["var_east"].append(unavailable)
        table["var_north"].append(unavailable)
        table["covar_en"].append(unavailable)

    return table


def find_tile_index(value: int, ranges: List[List[int] | Tuple[int, int]]) -> int:
    for idx, pair in enumerate(ranges):
        lo, hi = int(pair[0]), int(pair[1])
        if lo <= value <= hi:
            return idx
    raise ValueError(f"Grid index {value} outside tile ranges: {ranges}")


def load_blank_records(blank_cells: Dict[str, Any]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    for idx, feature in enumerate(blank_cells.get("features", [])):
        props = feature.get("properties") or {}
        geom = feature.get("geometry") or {}
        rings = geom.get("coordinates") or []

        blank_id = str(props.get("blank_id", f"BLANK_{idx:06d}"))
        blank_index = safe_int(props.get("blank_index"), idx)
        gi = safe_int(props.get("grid_i"))
        gj = safe_int(props.get("grid_j"))

        if gi is None or gj is None:
            raise ValueError(f"Blank cell {blank_id} missing grid_i/grid_j")
        if geom.get("type") != "Polygon" or not rings or len(rings[0]) < 4:
            raise ValueError(f"Blank cell {blank_id} has invalid polygon geometry")

        bbox = props.get("bbox_wgs84")
        if not bbox:
            ring = rings[0]
            lons = [float(p[0]) for p in ring]
            lats = [float(p[1]) for p in ring]
            bbox = {
                "west": min(lons),
                "south": min(lats),
                "east": max(lons),
                "north": max(lats),
            }

        lon_center = safe_float(props.get("lon_center"))
        lat_center = safe_float(props.get("lat_center"))
        if lon_center is None or lat_center is None:
            lon_center, lat_center = center_from_ring_or_bbox(rings[0], bbox)

        records.append({
            "blank_id": blank_id,
            "blank_index": int(blank_index), # type: ignore
            "grid_i": int(gi),
            "grid_j": int(gj),
            "ring_lonlat": rings[0],
            "bbox_wgs84": bbox,
            "lon_center": lon_center,
            "lat_center": lat_center,
        })

    return records


def group_blank_records_by_tile(records: List[Dict[str, Any]], tile_index: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    metadata = tile_index.get("metadata") or {}
    grid = metadata.get("grid") or {}
    i_ranges = grid.get("i_ranges")
    j_ranges = grid.get("j_ranges")

    if not i_ranges or not j_ranges:
        raise ValueError("tile_index metadata.grid.i_ranges/j_ranges missing")

    groups: Dict[str, Dict[str, Any]] = {}

    for rec in records:
        col = find_tile_index(rec["grid_i"], i_ranges)
        row = find_tile_index(rec["grid_j"], j_ranges)
        tile_id = f"blank_tile_r{row:02d}_c{col:02d}"

        group = groups.setdefault(tile_id, {
            "tile_id": tile_id,
            "tile_row": row,
            "tile_col": col,
            "records": [],
        })
        group["records"].append(rec)

    return groups


def add_polygon_to_buffers(
    ring_lonlat: List[List[float]],
    row_v: float,
    batch_id: int,
    center_lon: float,
    center_lat: float,
    datum_height_m: float,
    positions: List[float],
    normals: List[float],
    texcoords: List[float],
    batchids: List[float],
    indices: List[int],
) -> None:
    if len(ring_lonlat) < 4:
        raise ValueError("Blank ring has fewer than 4 coordinates")

    center_ecef = geodetic_to_ecef(center_lon, center_lat, 0.0)
    east, north, up = enu_basis(center_lon, center_lat)

    start_index = len(positions) // 3

    corners = ring_lonlat[:-1] if ring_lonlat[0] == ring_lonlat[-1] else ring_lonlat
    if len(corners) < 4:
        raise ValueError("Blank ring has fewer than 4 unique coordinates")

    for lon, lat in corners:
        ecef = geodetic_to_ecef(float(lon), float(lat), datum_height_m + CAP_CLEARANCE_M)
        local = ecef_to_local_enu(ecef, center_ecef, east, north, up)
        positions.extend([float(local[0]), float(local[1]), float(local[2])])
        normals.extend([0.0, 0.0, 1.0])
        texcoords.extend([0.0, float(row_v)])
        batchids.append(float(batch_id))

    for k in range(1, len(corners) - 1):
        indices.extend([start_index, start_index + k, start_index + k + 1])


def write_empty_tileset(
    path: Path,
    dataset_bbox: Optional[Dict[str, float]],
    bound_min_height_m: float,
    bound_max_height_m: float,
    extras: Dict[str, Any],
) -> None:
    if dataset_bbox:
        root_bv = {"region": bounding_region_from_bbox(dataset_bbox, bound_min_height_m, bound_max_height_m)}
    else:
        # Legal-ish fallback bounding volume. Should only happen if upstream
        # metadata is missing, but keeps zero-blank pathway robust.
        root_bv = {"region": [0, 0, 0, 0, bound_min_height_m, bound_max_height_m]}

    tileset = {
        "asset": {
            "version": "1.0",
            "tilesetVersion": "blank_caps_empty_v1",
            "generator": "InSAR4D RUM Viewer pipeline step 12",
        },
        "geometricError": 0.0,
        "root": {
            "boundingVolume": root_bv,
            "geometricError": 0.0,
            "refine": "ADD",
            "children": [],
        },
        "extras": extras,
    }

    write_json(path, tileset)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    t_start = time.time()

    cfg = load_resolved_config(__file__)
    project_root = Path(cfg["_resolved"]["project_root"])

    generated = cfg["generated_outputs"]
    paths = cfg["paths"]
    viewer = cfg["viewer"]
    caps_cfg = cfg["caps_b3dm"]
    tiling = cfg["tiling"]

    blank_path = resolve_path(project_root, generated["blank_cells"])
    height_meta_path = resolve_path(project_root, generated["height_meta"])
    tiles_dir = resolve_path(project_root, paths["blank_tiles_dir"])
    real_tiles_dir = resolve_path(project_root, paths["tiles_dir"])
    tile_index_path = real_tiles_dir / TILE_INDEX_FILENAME
    tileset_path = resolve_path(project_root, generated["blank_caps_tileset"])

    datum_height_m = float(viewer.get("display_datum_height_m", 1000.0))
    bound_min_height_m = float(caps_cfg.get("bound_min_height_m", -1000.0))
    bound_max_height_m = float(caps_cfg.get("bound_max_height_m", 10000.0))

    section("Configuration")
    print(f"  Project root       : {project_root}")
    print(f"  Blank cells input  : {blank_path}")
    print(f"  Height meta input  : {height_meta_path}")
    print(f"  Tile index input   : {tile_index_path}")
    print(f"  Tileset output     : {tileset_path}")
    print(f"  Display datum      : {datum_height_m} m")

    section("Loading inputs")
    blank_cells = load_json(blank_path)
    height_meta = load_json(height_meta_path)
    tile_index = load_json(tile_index_path)

    blank_meta = blank_cells.get("metadata") or {}
    blank_features = blank_cells.get("features", [])
    blank_count = len(blank_features)

    texture = height_meta.get("texture") or {}
    texture_height = int(texture.get("height", 0))
    row_layout = texture.get("row_layout") or {}
    blank_start_row = row_layout.get("blank_start_row")

    if texture_height <= 0:
        raise ValueError("height_meta texture.height is invalid")

    dataset_bbox = (tile_index.get("metadata") or {}).get("dataset_bbox_wgs84")

    ok(f"Loaded blank-cell product: status={blank_meta.get('status')}, features={blank_count}")
    ok(f"Loaded height meta: texture height={texture_height}")

    if CLEAN_OLD_BLANK_CAP_B3DM:
        removed = 0
        if tiles_dir.exists():
            for old in tiles_dir.glob("blank_tile_r*_c*.b3dm"):
                old.unlink()
                removed += 1
        if removed:
            ok(f"Removed old blank cap B3DM files: {removed}")

    if blank_count == 0:
        section("Writing empty blank tileset")
        extras = {
            "schema": "blank_caps_tileset_v2_pickable_batch",
            "status": "no_blank_cells_detected",
            "source_blank_cells": generated["blank_cells"],
            "source_height_meta": generated["height_meta"],
            "blank_count": 0,
            "tile_count": 0,
            "reason": "blank_cells.json has no features",
        }
        write_empty_tileset(tileset_path, dataset_bbox, bound_min_height_m, bound_max_height_m, extras)

        elapsed = time.time() - t_start

        ok(f"Wrote empty blank caps tileset: {tileset_path} ({tileset_path.stat().st_size / 1024:.1f} KB)")

        section("Summary")
        ok(f"Step 12 complete in {elapsed:.2f} s")
        print("  Blank count            : 0")
        print("  Blank cap tiles         : 0")
        return

    if blank_start_row is None:
        raise ValueError("height_meta row_layout.blank_start_row is missing but blank cells exist")

    blank_start_row = int(blank_start_row)

    section("Preparing blank records")
    records = load_blank_records(blank_cells)
    if len(records) != blank_count:
        raise ValueError(f"Blank record count mismatch: records={len(records)}, features={blank_count}")

    groups = group_blank_records_by_tile(records, tile_index)
    ok(f"Prepared {len(records)} blank records in {len(groups)} tile groups")

    section("Building blank cap B3DM tiles")
    children: List[Dict[str, Any]] = []
    total_vertices = 0
    total_triangles = 0
    total_blanks = 0

    for tile_id in sorted(groups):
        group = groups[tile_id]
        group_records = sorted(group["records"], key=lambda r: r["blank_index"])

        bboxes = [r["bbox_wgs84"] for r in group_records]
        bbox = bbox_union_wgs84(bboxes)
        center_lon, center_lat = tile_center_from_bbox(bbox)

        positions: List[float] = []
        normals: List[float] = []
        texcoords: List[float] = []
        batchids: List[float] = []
        indices: List[int] = []

        for batch_id, rec in enumerate(group_records):
            row_index = blank_start_row + int(rec["blank_index"])
            row_v = (row_index + 0.5) / texture_height

            add_polygon_to_buffers(
                ring_lonlat=rec["ring_lonlat"],
                row_v=row_v,
                batch_id=batch_id,
                center_lon=center_lon,
                center_lat=center_lat,
                datum_height_m=datum_height_m,
                positions=positions,
                normals=normals,
                texcoords=texcoords,
                batchids=batchids,
                indices=indices,
            )

        batch_table = build_batch_table_for_blanks(group_records, blank_start_row)
        glb = build_glb(positions, normals, texcoords, batchids, indices)
        b3dm = build_b3dm(
            glb=glb,
            batch_length=len(group_records),
            batch_table=batch_table,
        )

        b3dm_name = f"{tile_id}.b3dm"
        b3dm_path = tiles_dir / b3dm_name
        write_binary(b3dm_path, b3dm)

        transform = enu_to_ecef_transform_column_major(center_lon, center_lat, 0.0)
        region = bounding_region_from_bbox(bbox, bound_min_height_m, bound_max_height_m)

        children.append({
            "boundingVolume": {
                "region": region,
            },
            "geometricError": float(tiling.get("geometric_error_leaf", 100.0)),
            "refine": "ADD",
            "transform": transform,
            "content": {
                "uri": b3dm_name,
            },
            "extras": {
                "tile_id": tile_id,
                "blank_count": len(group_records),
                "pickable_batch_table": True,
            },
        })

        total_blanks += len(group_records)
        total_vertices += len(positions) // 3
        total_triangles += len(indices) // 3

    ok(f"Built {len(children)} blank cap B3DM tiles")
    ok(f"Total blank caps: {total_blanks}")
    print(f"  Total vertices       : {total_vertices}")
    print(f"  Total triangles      : {total_triangles}")

    section("Writing blank caps tileset")
    root_region = bounding_region_from_bbox(dataset_bbox, bound_min_height_m, bound_max_height_m) if dataset_bbox else [0, 0, 0, 0, bound_min_height_m, bound_max_height_m]

    tileset = {
        "asset": {
            "version": "1.0",
            "tilesetVersion": "blank_caps_v2_pickable_batch",
            "generator": "InSAR4D RUM Viewer pipeline step 12",
        },
        "geometricError": float(tiling.get("geometric_error_root", 5000.0)),
        "root": {
            "boundingVolume": {
                "region": root_region,
            },
            "geometricError": float(tiling.get("geometric_error_root", 5000.0)),
            "refine": "ADD",
            "children": children,
        },
        "extras": {
            "schema": "blank_caps_tileset_v2_pickable_batch",
            "status": "blank_cells_detected",
            "source_blank_cells": generated["blank_cells"],
            "source_height_meta": generated["height_meta"],
            "height_texture": height_meta.get("height_texture"),
            "display_datum_height_m": datum_height_m,
            "texture_height": texture_height,
            "blank_start_row": blank_start_row,
            "row_lookup": "TEXCOORD_0.y = (blank_start_row + blank_index + 0.5) / texture_height",
            "picking_contract": "_BATCHID points to batch table; feature_kind=blankie; rum_id=blankie RUM",
            "blank_count": total_blanks,
            "tile_count": len(children),
            "total_vertices": total_vertices,
            "total_triangles": total_triangles,
        },
    }

    write_json(tileset_path, tileset)

    elapsed = time.time() - t_start

    ok(f"Wrote blank caps tileset: {tileset_path} ({tileset_path.stat().st_size / 1024:.1f} KB)")

    section("Summary")
    ok(f"Step 12 complete in {elapsed:.2f} s")
    print(f"  Blank cap tiles        : {len(children)}")
    print(f"  Blank caps             : {total_blanks}")
    print(f"  Vertices / triangles   : {total_vertices} / {total_triangles}")
    print(f"  Tileset                : {tileset_path}")


if __name__ == "__main__":
    main()
