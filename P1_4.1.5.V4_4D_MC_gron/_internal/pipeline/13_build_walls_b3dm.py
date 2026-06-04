#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
13_build_walls_b3dm.py

InSAR4D RUM Viewer pipeline step 13.

Purpose
-------
Build B3DM wall tiles for real RUM cells and blank/no-data cells.

Inputs
------
  generated_outputs.rum_footprints
  generated_outputs.blank_cells
  generated_outputs.packed_series
  generated_outputs.height_meta
  _internal/data_pipeline/tiles/tile_index.json

Outputs
-------
  generated_outputs.real_walls_tileset
    _internal/data_pipeline/tiles_walls_real/tileset.json

  generated_outputs.blank_walls_tileset
    _internal/data_pipeline/tiles_walls_blank/tileset.json

Geometry / shader contract
--------------------------
Each wall is a quad around a RUM footprint edge.

Each vertex stores:
  POSITION   = local ENU position in tile coordinates
  NORMAL     = approximate side normal
  TEXCOORD_0 = [role, row_v_a]
  TEXCOORD_1 = [0.0, row_v_b]

where:
  role = 0.0 for the lower wall edge
  role = 1.0 for the upper wall edge
  row_v_a = height texture row of one neighbouring cell
  row_v_b = height texture row of the other neighbouring cell

The viewer wall shader samples both neighbouring MODEL rows at the active epoch:
  lower edge -> min(model_mm(row_a), model_mm(row_b))
  upper edge -> max(model_mm(row_a), model_mm(row_b))

Thus walls represent the MODEL height difference between two neighbouring caps.

Important
---------
This step generates only internal neighbour-pair walls:
  - exactly one wall per shared grid edge
  - no outside perimeter walls
  - no duplicate coplanar internal walls
  - real and blank owner cells are written to separate tilesets
  - no blank cells is OK and writes an empty blank-wall tileset
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

TILE_INDEX_FILENAME = "tile_index.json"

POSITION_COMPONENT_TYPE = 5126  # FLOAT
NORMAL_COMPONENT_TYPE = 5126    # FLOAT
TEXCOORD_COMPONENT_TYPE = 5126  # FLOAT
INDEX_COMPONENT_TYPE = 5125     # UNSIGNED_INT

REAL_WALL_MATERIAL_BASE_COLOR = [1.0, 1.0, 1.0, 1.0]
BLANK_WALL_MATERIAL_BASE_COLOR = [1.0, 1.0, 1.0, 0.55]
DOUBLE_SIDED = True
BATCH_LENGTH = 0

CLEAN_OLD_WALL_B3DM = True

# Ring order from Step 03/09 is SW, SE, NE, NW, SW.
# Corresponding sides and neighbour offsets for edges are south, east, north, west.
EDGE_SIDES = ["S", "E", "N", "W"]
EDGE_NEIGHBOUR_OFFSETS = [(0, -1), (1, 0), (0, 1), (-1, 0)]

# Efficient one-wall-per-edge rule:
#   - only the western/southern cell owns an edge
#   - therefore each cell only builds EAST and NORTH sides
#   - if there is no neighbour on that side, no wall is generated
# This gives internal walls without duplicate faces and without outer perimeter walls.
OWNED_EDGE_SIDES = {"E", "N"}


def owns_canonical_side(edge_index: int) -> bool:
    return EDGE_SIDES[edge_index] in OWNED_EDGE_SIDES


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
    r = len(data) % multiple
    return data if r == 0 else data + pad_byte * (multiple - r)


def safe_int(value: Any, fallback: Optional[int] = None) -> Optional[int]:
    try:
        if value is None:
            return fallback
        return int(value)
    except Exception:
        return fallback


def safe_float(value: Any, fallback: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return fallback
        out = float(value)
        return out if math.isfinite(out) else fallback
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
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


def ecef_to_local_enu(
    ecef: Tuple[float, float, float],
    center_ecef: Tuple[float, float, float],
    east: Tuple[float, float, float],
    north: Tuple[float, float, float],
    up: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    d = (ecef[0] - center_ecef[0], ecef[1] - center_ecef[1], ecef[2] - center_ecef[2])
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
# GLB / B3DM
# =============================================================================

def pack_floats(values: List[float]) -> bytes:
    return struct.pack("<" + "f" * len(values), *values)


def pack_uint32(values: List[int]) -> bytes:
    return struct.pack("<" + "I" * len(values), *values)


def append_aligned(buffer: bytearray, data: bytes, alignment: int = 4, pad_byte: bytes = b"\x00") -> Tuple[int, int]:
    offset = len(buffer)
    pad = (alignment - (offset % alignment)) % alignment
    if pad:
        buffer.extend(pad_byte * pad)
        offset += pad
    buffer.extend(data)
    return offset, len(data)


def component_min_max_vec3(values: List[float]) -> Tuple[List[float], List[float]]:
    xs = values[0::3]
    ys = values[1::3]
    zs = values[2::3]
    return [min(xs), min(ys), min(zs)], [max(xs), max(ys), max(zs)]


def build_glb(
    positions: List[float],
    normals: List[float],
    texcoords0: List[float],
    texcoords1: List[float],
    indices: List[int],
    material_color: List[float],
    generator: str,
) -> bytes:
    if not positions or not indices:
        raise ValueError("Cannot build GLB with empty positions/indices")

    vertex_count = len(positions) // 3
    index_count = len(indices)

    if len(normals) != vertex_count * 3:
        raise ValueError("normal length mismatch")
    if len(texcoords0) != vertex_count * 2:
        raise ValueError("texcoord0 length mismatch")
    if len(texcoords1) != vertex_count * 2:
        raise ValueError("texcoord1 length mismatch")

    bin_buffer = bytearray()

    pos_offset, pos_len = append_aligned(bin_buffer, pack_floats(positions), 4)
    norm_offset, norm_len = append_aligned(bin_buffer, pack_floats(normals), 4)
    uv0_offset, uv0_len = append_aligned(bin_buffer, pack_floats(texcoords0), 4)
    uv1_offset, uv1_len = append_aligned(bin_buffer, pack_floats(texcoords1), 4)
    idx_offset, idx_len = append_aligned(bin_buffer, pack_uint32(indices), 4)

    pos_min, pos_max = component_min_max_vec3(positions)

    gltf = {
        "asset": {"version": "2.0", "generator": generator},
        "buffers": [{"byteLength": len(bin_buffer)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": pos_offset, "byteLength": pos_len, "target": 34962},
            {"buffer": 0, "byteOffset": norm_offset, "byteLength": norm_len, "target": 34962},
            {"buffer": 0, "byteOffset": uv0_offset, "byteLength": uv0_len, "target": 34962},
            {"buffer": 0, "byteOffset": uv1_offset, "byteLength": uv1_len, "target": 34962},
            {"buffer": 0, "byteOffset": idx_offset, "byteLength": idx_len, "target": 34963},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": vertex_count, "type": "VEC3", "min": pos_min, "max": pos_max},
            {"bufferView": 1, "componentType": 5126, "count": vertex_count, "type": "VEC3"},
            {"bufferView": 2, "componentType": 5126, "count": vertex_count, "type": "VEC2"},
            {"bufferView": 3, "componentType": 5126, "count": vertex_count, "type": "VEC2"},
            {"bufferView": 4, "componentType": 5125, "count": index_count, "type": "SCALAR", "min": [min(indices)], "max": [max(indices)]},
        ],
        "materials": [
            {
                "pbrMetallicRoughness": {
                    "baseColorFactor": material_color,
                    "metallicFactor": 0.0,
                    "roughnessFactor": 1.0,
                },
                "alphaMode": "BLEND" if material_color[3] < 1.0 else "OPAQUE",
                "doubleSided": DOUBLE_SIDED,
            }
        ],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2, "TEXCOORD_1": 3},
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

    json_chunk = pad_bytes(json.dumps(gltf, separators=(",", ":")).encode("utf-8"), 4, b" ")
    bin_chunk = pad_bytes(bytes(bin_buffer), 4, b"\x00")

    total_length = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
    glb = bytearray()
    glb.extend(struct.pack("<4sII", b"glTF", 2, total_length))
    glb.extend(struct.pack("<I4s", len(json_chunk), b"JSON"))
    glb.extend(json_chunk)
    glb.extend(struct.pack("<I4s", len(bin_chunk), b"BIN\x00"))
    glb.extend(bin_chunk)
    return bytes(glb)


def build_b3dm(glb: bytes) -> bytes:
    ft_json = pad_bytes(json.dumps({"BATCH_LENGTH": BATCH_LENGTH}, separators=(",", ":")).encode("utf-8"), 8, b" ")
    ft_bin = b""
    bt_json = b""
    bt_bin = b""

    byte_length = 28 + len(ft_json) + len(ft_bin) + len(bt_json) + len(bt_bin) + len(glb)
    header = struct.pack("<4sIIIIII", b"b3dm", 1, byte_length, len(ft_json), len(ft_bin), len(bt_json), len(bt_bin))
    return header + ft_json + ft_bin + bt_json + bt_bin + glb


# =============================================================================
# GEOMETRY HELPERS
# =============================================================================

def tile_center_from_bbox(bbox: Dict[str, float]) -> Tuple[float, float]:
    return (float(bbox["west"]) + float(bbox["east"])) / 2.0, (float(bbox["south"]) + float(bbox["north"])) / 2.0


def bounding_region_from_bbox(bbox: Dict[str, float], min_h: float, max_h: float) -> List[float]:
    return [
        deg_to_rad(float(bbox["west"])),
        deg_to_rad(float(bbox["south"])),
        deg_to_rad(float(bbox["east"])),
        deg_to_rad(float(bbox["north"])),
        float(min_h),
        float(max_h),
    ]


def bbox_union_wgs84(bboxes: List[Dict[str, float]]) -> Dict[str, float]:
    return {
        "west": min(float(b["west"]) for b in bboxes),
        "south": min(float(b["south"]) for b in bboxes),
        "east": max(float(b["east"]) for b in bboxes),
        "north": max(float(b["north"]) for b in bboxes),
    }


def bbox_from_ring(ring: List[List[float]]) -> Dict[str, float]:
    lons = [float(p[0]) for p in ring]
    lats = [float(p[1]) for p in ring]
    return {"west": min(lons), "south": min(lats), "east": max(lons), "north": max(lats)}


def normalize2(x: float, y: float) -> Tuple[float, float]:
    n = math.hypot(x, y)
    if n <= 0:
        return 0.0, 1.0
    return x / n, y / n


def local_from_lonlat(
    lon: float,
    lat: float,
    height: float,
    center_lon: float,
    center_lat: float,
    center_ecef: Tuple[float, float, float],
    east: Tuple[float, float, float],
    north: Tuple[float, float, float],
    up: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    ecef = geodetic_to_ecef(lon, lat, height)
    return ecef_to_local_enu(ecef, center_ecef, east, north, up)


def add_wall_pair_quad(
    p0: List[float],
    p1: List[float],
    row_v_a: float,
    row_v_b: float,
    center_lon: float,
    center_lat: float,
    datum_height_m: float,
    positions: List[float],
    normals: List[float],
    texcoords0: List[float],
    texcoords1: List[float],
    indices: List[int],
) -> None:
    """
    Add one dynamic neighbour-pair wall along edge p0->p1.

    Both lower and upper vertices are initially written at the display datum.
    The viewer shader samples both neighbour rows and moves:
      role=0 vertices to min(height_a, height_b)
      role=1 vertices to max(height_a, height_b)

    Vertex order:
      0 lower p0, role=0
      1 lower p1, role=0
      2 upper p1, role=1
      3 upper p0, role=1
    """
    center_ecef = geodetic_to_ecef(center_lon, center_lat, 0.0)
    east, north, up = enu_basis(center_lon, center_lat)

    lower0 = local_from_lonlat(float(p0[0]), float(p0[1]), datum_height_m, center_lon, center_lat, center_ecef, east, north, up)
    lower1 = local_from_lonlat(float(p1[0]), float(p1[1]), datum_height_m, center_lon, center_lat, center_ecef, east, north, up)
    upper1 = local_from_lonlat(float(p1[0]), float(p1[1]), datum_height_m, center_lon, center_lat, center_ecef, east, north, up)
    upper0 = local_from_lonlat(float(p0[0]), float(p0[1]), datum_height_m, center_lon, center_lat, center_ecef, east, north, up)

    start = len(positions) // 3
    verts = [lower0, lower1, upper1, upper0]

    # Approx side normal in local ENU from edge direction.
    dx = lower1[0] - lower0[0]
    dy = lower1[1] - lower0[1]
    nx, ny = normalize2(dy, -dx)

    for idx, v in enumerate(verts):
        positions.extend([float(v[0]), float(v[1]), float(v[2])])
        normals.extend([nx, ny, 0.0])
        role = 0.0 if idx in (0, 1) else 1.0
        texcoords0.extend([role, float(row_v_a)])
        texcoords1.extend([0.0, float(row_v_b)])

    indices.extend([start, start + 1, start + 2, start, start + 2, start + 3])


def cell_edge_points(ring_lonlat: List[List[float]], edge_index: int) -> Tuple[List[float], List[float]]:
    corners = ring_lonlat[:-1] if ring_lonlat and ring_lonlat[0] == ring_lonlat[-1] else ring_lonlat
    if len(corners) < 4:
        raise ValueError("Cell polygon has fewer than 4 corners")
    n = len(corners)
    return corners[edge_index], corners[(edge_index + 1) % n]

# =============================================================================
# INPUT PREP
# =============================================================================

def build_footprint_lookup(footprints: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}

    for idx, feature in enumerate(footprints.get("features", [])):
        props = feature.get("properties") or {}
        rum_id = str(props.get("rum_id", f"RUM_{idx + 1:06d}"))
        geom = feature.get("geometry") or {}
        rings = geom.get("coordinates") or []

        if geom.get("type") != "Polygon" or not rings or len(rings[0]) < 4:
            raise ValueError(f"Invalid footprint geometry for {rum_id}")

        gi = safe_int(props.get("grid_i"))
        gj = safe_int(props.get("grid_j"))
        if gi is None or gj is None:
            raise ValueError(f"Footprint {rum_id} missing grid_i/grid_j")

        lookup[rum_id] = {
            "ring_lonlat": rings[0],
            "bbox_wgs84": bbox_from_ring(rings[0]),
            "grid_i": int(gi),
            "grid_j": int(gj),
        }

    if not lookup:
        raise ValueError("No footprint features found")

    return lookup


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
            raise ValueError(f"Blank {blank_id} missing grid_i/grid_j")
        if geom.get("type") != "Polygon" or not rings or len(rings[0]) < 4:
            raise ValueError(f"Blank {blank_id} invalid polygon")

        records.append({
            "blank_id": blank_id,
            "blank_index": int(blank_index),
            "grid_i": int(gi),
            "grid_j": int(gj),
            "ring_lonlat": rings[0],
            "bbox_wgs84": props.get("bbox_wgs84") or bbox_from_ring(rings[0]),
        })

    return records


def find_tile_index(value: int, ranges: List[List[int] | Tuple[int, int]]) -> int:
    for idx, pair in enumerate(ranges):
        lo, hi = int(pair[0]), int(pair[1])
        if lo <= value <= hi:
            return idx
    raise ValueError(f"Grid index {value} outside tile ranges")


def group_blank_records_by_tile(records: List[Dict[str, Any]], tile_index: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    grid = (tile_index.get("metadata") or {}).get("grid") or {}
    i_ranges = grid.get("i_ranges")
    j_ranges = grid.get("j_ranges")
    if not i_ranges or not j_ranges:
        raise ValueError("tile_index metadata.grid.i_ranges/j_ranges missing")

    groups: Dict[str, Dict[str, Any]] = {}

    for rec in records:
        col = find_tile_index(rec["grid_i"], i_ranges)
        row = find_tile_index(rec["grid_j"], j_ranges)
        tile_id = f"blank_wall_tile_r{row:02d}_c{col:02d}"
        group = groups.setdefault(tile_id, {"tile_id": tile_id, "tile_row": row, "tile_col": col, "records": []})
        group["records"].append(rec)

    return groups


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
        root_bv = {"region": [0, 0, 0, 0, bound_min_height_m, bound_max_height_m]}

    tileset = {
        "asset": {
            "version": "1.0",
            "tilesetVersion": extras.get("tileset_version", "empty_walls_v1"),
            "generator": "InSAR4D RUM Viewer pipeline step 13",
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




def make_cell_record(
    cell_id: str,
    kind: str,
    grid_i: int,
    grid_j: int,
    ring_lonlat: List[List[float]],
    row_idx: int,
    texture_height: int,
    bbox_wgs84: Dict[str, float],
) -> Dict[str, Any]:
    return {
        "cell_id": cell_id,
        "kind": kind,
        "grid_i": int(grid_i),
        "grid_j": int(grid_j),
        "ring_lonlat": ring_lonlat,
        "row_idx": int(row_idx),
        "row_v": (int(row_idx) + 0.5) / float(texture_height),
        "bbox_wgs84": bbox_wgs84,
    }


def build_cell_lookups(
    footprint_lookup: Dict[str, Dict[str, Any]],
    blank_records: List[Dict[str, Any]],
    rum_index: Dict[str, Any],
    texture_height: int,
    blank_start_row: Optional[int],
) -> Tuple[Dict[Tuple[int, int], Dict[str, Any]], Dict[Tuple[int, int], Dict[str, Any]], Dict[Tuple[int, int], Dict[str, Any]]]:
    real_cells: Dict[Tuple[int, int], Dict[str, Any]] = {}
    blank_cells_lookup: Dict[Tuple[int, int], Dict[str, Any]] = {}

    for rum_id, fp in footprint_lookup.items():
        if rum_id not in rum_index:
            continue
        key = (int(fp["grid_i"]), int(fp["grid_j"]))
        real_cells[key] = make_cell_record(
            rum_id,
            "real",
            key[0],
            key[1],
            fp["ring_lonlat"],
            int(rum_index[rum_id]),
            texture_height,
            fp["bbox_wgs84"],
        )

    if blank_start_row is not None:
        for rec in blank_records:
            key = (int(rec["grid_i"]), int(rec["grid_j"]))
            row_idx = int(blank_start_row) + int(rec["blank_index"])
            blank_cells_lookup[key] = make_cell_record(
                rec["blank_id"],
                "blank",
                key[0],
                key[1],
                rec["ring_lonlat"],
                row_idx,
                texture_height,
                rec["bbox_wgs84"],
            )

    union = dict(real_cells)
    union.update(blank_cells_lookup)
    return real_cells, blank_cells_lookup, union


def add_owned_neighbour_pair_walls_for_cell(
    cell: Dict[str, Any],
    union_cells: Dict[Tuple[int, int], Dict[str, Any]],
    center_lon: float,
    center_lat: float,
    datum_height_m: float,
    positions: List[float],
    normals: List[float],
    texcoords0: List[float],
    texcoords1: List[float],
    indices: List[int],
) -> int:
    """Add canonical E/N shared-edge walls owned by this cell.

    Each shared edge is generated once by the western/southern cell.
    Outer perimeter edges are skipped because no neighbour exists there.
    """
    gi = int(cell["grid_i"])
    gj = int(cell["grid_j"])
    count = 0

    for edge_index, (di, dj) in enumerate(EDGE_NEIGHBOUR_OFFSETS):
        if not owns_canonical_side(edge_index):
            continue
        neighbour = union_cells.get((gi + di, gj + dj))
        if neighbour is None:
            continue

        p0, p1 = cell_edge_points(cell["ring_lonlat"], edge_index)
        add_wall_pair_quad(
            p0, p1,
            float(cell["row_v"]),
            float(neighbour["row_v"]),
            center_lon,
            center_lat,
            datum_height_m,
            positions,
            normals,
            texcoords0,
            texcoords1,
            indices,
        )
        count += 1

    return count

# =============================================================================
# REAL WALLS
# =============================================================================

def build_real_walls(
    cfg: Dict[str, Any],
    project_root: Path,
    footprints: Dict[str, Any],
    blank_cells: Dict[str, Any],
    packed: Dict[str, Any],
    height_meta: Dict[str, Any],
    tile_index: Dict[str, Any],
) -> Dict[str, Any]:
    generated = cfg["generated_outputs"]
    paths = cfg["paths"]
    viewer = cfg["viewer"]
    walls_cfg = cfg["walls_b3dm"]
    tiling = cfg["tiling"]

    output_dir = resolve_path(project_root, paths["real_walls_tiles_dir"])
    tileset_path = resolve_path(project_root, generated["real_walls_tileset"])

    if CLEAN_OLD_WALL_B3DM and output_dir.exists():
        for old in output_dir.glob("real_wall_tile_*.b3dm"):
            old.unlink()

    footprint_lookup = build_footprint_lookup(footprints)
    blank_records = load_blank_records(blank_cells)

    rum_index = packed.get("rum_index") or {}
    texture = height_meta.get("texture") or {}
    texture_height = int(texture.get("height", 0))
    row_layout = texture.get("row_layout") or {}
    blank_start_row = row_layout.get("blank_start_row")

    if not rum_index:
        raise ValueError("packed_series missing rum_index")
    if texture_height <= 0:
        raise ValueError("height_meta texture.height invalid")

    real_cells, blank_cells_lookup, union_cells = build_cell_lookups(
        footprint_lookup,
        blank_records,
        rum_index,
        texture_height,
        int(blank_start_row) if blank_start_row is not None else None,
    )
    real_by_id = {cell["cell_id"]: cell for cell in real_cells.values()}

    datum_height_m = float(viewer.get("display_datum_height_m", 1000.0))
    bound_min_height_m = float(walls_cfg.get("bound_min_height_m", -1000.0))
    bound_max_height_m = float(walls_cfg.get("bound_max_height_m", 10000.0))

    children: List[Dict[str, Any]] = []
    total_cells = 0
    total_walls = 0
    total_vertices = 0
    total_triangles = 0

    for tile in tile_index.get("tiles", []):
        rum_ids = tile.get("rum_ids", [])
        if not rum_ids:
            continue

        tile_id = tile["tile_id"].replace("tile_", "real_wall_tile_")
        bbox = tile.get("bbox_wgs84")
        if not bbox:
            continue

        center_lon, center_lat = tile_center_from_bbox(bbox)

        positions: List[float] = []
        normals: List[float] = []
        texcoords0: List[float] = []
        texcoords1: List[float] = []
        indices: List[int] = []

        for rum_id in rum_ids:
            cell = real_by_id.get(str(rum_id))
            if cell is None:
                raise ValueError(f"Missing footprint or row index for {rum_id}")

            wall_count = add_owned_neighbour_pair_walls_for_cell(
                cell,
                union_cells,
                center_lon,
                center_lat,
                datum_height_m,
                positions,
                normals,
                texcoords0,
                texcoords1,
                indices,
            )
            total_walls += wall_count
            total_cells += 1

        if not positions or not indices:
            continue

        glb = build_glb(
            positions, normals, texcoords0, texcoords1, indices,
            REAL_WALL_MATERIAL_BASE_COLOR,
            "InSAR4D RUM Viewer pipeline step 13 real neighbour-pair walls",
        )
        b3dm = build_b3dm(glb)

        b3dm_name = f"{tile_id}.b3dm"
        write_binary(output_dir / b3dm_name, b3dm)

        children.append({
            "boundingVolume": {"region": bounding_region_from_bbox(bbox, bound_min_height_m, bound_max_height_m)},
            "geometricError": float(tiling.get("geometric_error_leaf", 100.0)),
            "refine": "ADD",
            "transform": enu_to_ecef_transform_column_major(center_lon, center_lat, 0.0),
            "content": {"uri": b3dm_name},
            "metadata": {"tile_id": tile_id, "rum_count": len(rum_ids)},
        })

        total_vertices += len(positions) // 3
        total_triangles += len(indices) // 3

    dataset_bbox = (footprints.get("metadata") or {}).get("bbox_wgs84_footprints") or (tile_index.get("metadata") or {}).get("dataset_bbox_wgs84")
    if not dataset_bbox:
        raise ValueError("Cannot determine root bbox for real walls")

    tileset = {
        "asset": {
            "version": "1.0",
            "tilesetVersion": "real_walls_neighbour_pair_v1",
            "generator": "InSAR4D RUM Viewer pipeline step 13",
        },
        "geometricError": float(tiling.get("geometric_error_root", 5000.0)),
        "root": {
            "boundingVolume": {"region": bounding_region_from_bbox(dataset_bbox, bound_min_height_m, bound_max_height_m)},
            "geometricError": float(tiling.get("geometric_error_root", 5000.0)),
            "refine": "ADD",
            "children": children,
        },
        "extras": {
            "schema": "real_walls_tileset_v1",
            "source_footprints": generated["rum_footprints"],
            "source_packed_series": generated["packed_series"],
            "source_height_meta": generated["height_meta"],
            "display_datum_height_m": datum_height_m,
            "texture_height": texture_height,
            "texcoord_contract": "TEXCOORD_0=[role,row_v_a], TEXCOORD_1=[0,row_v_b]; role 0 lower, 1 upper; shader samples both MODEL texture rows and uses min/max model height",
            "wall_policy": "canonical_internal_neighbour_pair; owner builds only E/N shared edges; outer perimeter skipped; one wall per shared edge",
            "cell_count": total_cells,
            "wall_count": total_walls,
            "tile_count": len(children),
            "total_vertices": total_vertices,
            "total_triangles": total_triangles,
        },
    }

    write_json(tileset_path, tileset)

    return {
        "tiles": len(children),
        "cells": total_cells,
        "walls": total_walls,
        "vertices": total_vertices,
        "triangles": total_triangles,
        "tileset": tileset_path,
    }


# =============================================================================
# BLANK WALLS
# =============================================================================

def build_blank_walls(
    cfg: Dict[str, Any],
    project_root: Path,
    footprints: Dict[str, Any],
    blank_cells: Dict[str, Any],
    height_meta: Dict[str, Any],
    tile_index: Dict[str, Any],
) -> Dict[str, Any]:
    generated = cfg["generated_outputs"]
    paths = cfg["paths"]
    viewer = cfg["viewer"]
    walls_cfg = cfg["walls_b3dm"]
    tiling = cfg["tiling"]

    output_dir = resolve_path(project_root, paths["blank_walls_tiles_dir"])
    tileset_path = resolve_path(project_root, generated["blank_walls_tileset"])

    if CLEAN_OLD_WALL_B3DM and output_dir.exists():
        for old in output_dir.glob("blank_wall_tile_*.b3dm"):
            old.unlink()

    records = load_blank_records(blank_cells)
    blank_count = len(records)

    datum_height_m = float(viewer.get("display_datum_height_m", 1000.0))
    bound_min_height_m = float(walls_cfg.get("bound_min_height_m", -1000.0))
    bound_max_height_m = float(walls_cfg.get("bound_max_height_m", 10000.0))
    dataset_bbox = (tile_index.get("metadata") or {}).get("dataset_bbox_wgs84")

    texture = height_meta.get("texture") or {}
    texture_height = int(texture.get("height", 0))
    row_layout = texture.get("row_layout") or {}
    blank_start_row = row_layout.get("blank_start_row")

    if texture_height <= 0:
        raise ValueError("height_meta texture.height invalid")

    if blank_count == 0:
        write_empty_tileset(
            tileset_path,
            dataset_bbox,
            bound_min_height_m,
            bound_max_height_m,
            {
                "schema": "blank_walls_tileset_v1",
                "tileset_version": "blank_walls_empty_v1",
                "status": "no_blank_cells_detected",
                "source_blank_cells": generated["blank_cells"],
                "blank_count": 0,
                "tile_count": 0,
            },
        )
        return {"tiles": 0, "cells": 0, "walls": 0, "vertices": 0, "triangles": 0, "tileset": tileset_path}

    if blank_start_row is None:
        raise ValueError("height_meta row_layout.blank_start_row missing but blank walls requested")

    footprint_lookup = build_footprint_lookup(footprints)
    # Packed rum_index is not loaded in build_blank_walls, so use real row indices
    # from height_meta/packed series is not available here. The caller therefore
    # cannot build mixed blank-real neighbour-pair walls correctly unless it knows
    # real row_v. To keep this function self-contained, read packed_series now.
    packed_path = resolve_path(project_root, generated["packed_series"])
    packed = load_json(packed_path)
    rum_index = packed.get("rum_index") or {}
    if not rum_index:
        raise ValueError("packed_series missing rum_index")

    real_cells, blank_cells_lookup, union_cells = build_cell_lookups(
        footprint_lookup,
        records,
        rum_index,
        texture_height,
        int(blank_start_row),
    )

    groups = group_blank_records_by_tile(records, tile_index)

    children: List[Dict[str, Any]] = []
    total_cells = 0
    total_walls = 0
    total_vertices = 0
    total_triangles = 0

    for tile_id in sorted(groups):
        recs = sorted(groups[tile_id]["records"], key=lambda r: r["blank_index"])
        bbox = bbox_union_wgs84([r["bbox_wgs84"] for r in recs])
        center_lon, center_lat = tile_center_from_bbox(bbox)

        positions: List[float] = []
        normals: List[float] = []
        texcoords0: List[float] = []
        texcoords1: List[float] = []
        indices: List[int] = []

        for rec in recs:
            key = (int(rec["grid_i"]), int(rec["grid_j"]))
            cell = blank_cells_lookup.get(key)
            if cell is None:
                raise ValueError(f"Missing blank cell lookup for {rec.get('blank_id', key)}")

            wall_count = add_owned_neighbour_pair_walls_for_cell(
                cell,
                union_cells,
                center_lon,
                center_lat,
                datum_height_m,
                positions,
                normals,
                texcoords0,
                texcoords1,
                indices,
            )

            total_walls += wall_count
            total_cells += 1

        if not positions or not indices:
            continue

        glb = build_glb(
            positions, normals, texcoords0, texcoords1, indices,
            BLANK_WALL_MATERIAL_BASE_COLOR,
            "InSAR4D RUM Viewer pipeline step 13 blank neighbour-pair walls",
        )
        b3dm = build_b3dm(glb)

        b3dm_name = f"{tile_id}.b3dm"
        write_binary(output_dir / b3dm_name, b3dm)

        children.append({
            "boundingVolume": {"region": bounding_region_from_bbox(bbox, bound_min_height_m, bound_max_height_m)},
            "geometricError": float(tiling.get("geometric_error_leaf", 100.0)),
            "refine": "ADD",
            "transform": enu_to_ecef_transform_column_major(center_lon, center_lat, 0.0),
            "content": {"uri": b3dm_name},
            "metadata": {"tile_id": tile_id, "blank_count": len(recs)},
        })

        total_vertices += len(positions) // 3
        total_triangles += len(indices) // 3

    root_bbox = dataset_bbox or bbox_union_wgs84([r["bbox_wgs84"] for r in records])

    tileset = {
        "asset": {
            "version": "1.0",
            "tilesetVersion": "blank_walls_neighbour_pair_v1",
            "generator": "InSAR4D RUM Viewer pipeline step 13",
        },
        "geometricError": float(tiling.get("geometric_error_root", 5000.0)),
        "root": {
            "boundingVolume": {"region": bounding_region_from_bbox(root_bbox, bound_min_height_m, bound_max_height_m)},
            "geometricError": float(tiling.get("geometric_error_root", 5000.0)),
            "refine": "ADD",
            "children": children,
        },
        "extras": {
            "schema": "blank_walls_tileset_v1",
            "status": "blank_cells_detected",
            "source_blank_cells": generated["blank_cells"],
            "source_height_meta": generated["height_meta"],
            "display_datum_height_m": datum_height_m,
            "texture_height": texture_height,
            "blank_start_row": int(blank_start_row),
            "texcoord_contract": "TEXCOORD_0=[role,row_v_a], TEXCOORD_1=[0,row_v_b]; role 0 lower, 1 upper; shader samples both MODEL texture rows and uses min/max model height",
            "wall_policy": "canonical_internal_neighbour_pair; owner builds only E/N shared edges; outer perimeter skipped; one wall per shared edge",
            "blank_count": total_cells,
            "wall_count": total_walls,
            "tile_count": len(children),
            "total_vertices": total_vertices,
            "total_triangles": total_triangles,
        },
    }

    write_json(tileset_path, tileset)

    return {
        "tiles": len(children),
        "cells": total_cells,
        "walls": total_walls,
        "vertices": total_vertices,
        "triangles": total_triangles,
        "tileset": tileset_path,
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

    footprints_path = resolve_path(project_root, generated["rum_footprints"])
    blank_path = resolve_path(project_root, generated["blank_cells"])
    packed_path = resolve_path(project_root, generated["packed_series"])
    height_meta_path = resolve_path(project_root, generated["height_meta"])
    tile_index_path = resolve_path(project_root, paths["tiles_dir"]) / TILE_INDEX_FILENAME

    section("Configuration")
    print(f"  Project root          : {project_root}")
    print(f"  Footprints input      : {footprints_path}")
    print(f"  Blank cells input     : {blank_path}")
    print(f"  Packed input          : {packed_path}")
    print(f"  Height meta input     : {height_meta_path}")
    print(f"  Tile index input      : {tile_index_path}")
    print(f"  Real walls tileset    : {resolve_path(project_root, generated['real_walls_tileset'])}")
    print(f"  Blank walls tileset   : {resolve_path(project_root, generated['blank_walls_tileset'])}")

    section("Loading inputs")
    footprints = load_json(footprints_path)
    blank_cells = load_json(blank_path)
    packed = load_json(packed_path)
    height_meta = load_json(height_meta_path)
    tile_index = load_json(tile_index_path)

    ok(f"Loaded footprints: {len(footprints.get('features', []))} features")
    ok(f"Loaded blank cells: {len(blank_cells.get('features', []))} features")
    ok(f"Loaded packed series and height metadata")

    section("Building real walls")
    real_summary = build_real_walls(cfg, project_root, footprints, blank_cells, packed, height_meta, tile_index)
    ok(f"Real wall tiles: {real_summary['tiles']}, cells={real_summary['cells']}, walls={real_summary['walls']}")

    section("Building blank walls")
    blank_summary = build_blank_walls(cfg, project_root, footprints, blank_cells, height_meta, tile_index)
    ok(f"Blank wall tiles: {blank_summary['tiles']}, cells={blank_summary['cells']}, walls={blank_summary['walls']}")

    elapsed = time.time() - t_start

    section("Summary")
    ok(f"Step 13 complete in {elapsed:.2f} s")
    print(f"  Real wall tiles        : {real_summary['tiles']}")
    print(f"  Real wall quads        : {real_summary['walls']}")
    print(f"  Blank wall tiles       : {blank_summary['tiles']}")
    print(f"  Blank wall quads       : {blank_summary['walls']}")
    print(f"  Real vertices/triangles: {real_summary['vertices']} / {real_summary['triangles']}")
    print(f"  Blank vertices/triangles: {blank_summary['vertices']} / {blank_summary['triangles']}")


if __name__ == "__main__":
    main()
