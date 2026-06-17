#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
11_build_real_caps_b3dm.py

InSAR4D RUM Viewer pipeline step 11.

Purpose
-------
Build B3DM tiles for real RUM cap polygons.

Inputs
------
  generated_outputs.rum_footprints
    _internal/data_pipeline/rum_footprints.json

  generated_outputs.packed_series
    _internal/data_pipeline/packed_series.json

  _internal/data_pipeline/tiles/tile_index.json
    created by Step 07

  generated_outputs.height_meta
    _internal/data_pipeline/tiles/height_meta.json

Outputs
-------
  generated_outputs.real_caps_tileset
    _internal/data_pipeline/tiles/tileset.json

  _internal/data_pipeline/tiles/*.b3dm

Geometry contract
-----------------
Each square RUM cap contains a configurable checkerboard lattice (selected:
6 x 6) of alternating upward/downward lowpoly truncated-pyramid-capable cells.
Flat rings around the pyramid footprints preserve the velocity colour.

Each vertex stores:
  POSITION   = local ENU position in tile coordinates
  NORMAL     = valid static [0, 0, 1] fallback; runtime shading derives the
               actual deformed face normal from fragment derivatives
  TEXCOORD_0.y = row_v height-texture row lookup
  TEXCOORD_0.x = relief role code:
                 0 = flat/base vertex
                 +/-1..4 = SW/SE/NE/NW top-platform corner
                 sign = upward/downward checkerboard cell
  _BATCHID   = integer-like float feature id inside the tile

The four top-platform vertices begin collapsed at the cell centre. The viewer
separates them when raw sigma exceeds the global p98 display ceiling, creating
a progressively larger square plateau without rebuilding B3DM geometry.

Picking contract
----------------
Each B3DM tile writes a legacy 3D Tiles batch table with per-RUM feature
properties. Cesium scene.pick(...).getProperty("rum_id") should therefore work
on real RUM caps.

Tileset contract
----------------
Each B3DM tile has a tile.transform that maps local ENU coordinates to ECEF.
The glTF coordinates stay small and numerically stable.
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

TILE_INDEX_FILENAME = "tile_index.json"

# Binary/geometry precision.
POSITION_COMPONENT_TYPE = 5126  # FLOAT
NORMAL_COMPONENT_TYPE = 5126    # FLOAT
TEXCOORD_COMPONENT_TYPE = 5126  # FLOAT
BATCHID_COMPONENT_TYPE = 5126   # FLOAT, safest legacy _BATCHID for Cesium B3DM picking
INDEX_COMPONENT_TYPE = 5125     # UNSIGNED_INT

# Material is intentionally simple; final color is normally controlled by
# Cesium custom shaders / height texture / color scale in the viewer.
MATERIAL_BASE_COLOR = [1.0, 1.0, 1.0, 1.0]
DOUBLE_SIDED = True

# A small optional lift to reduce z-fighting with wall/cap boundaries.
CAP_CLEARANCE_M = 0.0

# Selected vertical-uncertainty geometry. Each RUM receives a configurable checkerboard
# of alternating up/down truncated-pyramid-capable cells. The viewer animates
# height and plateau size from the B-channel raw vertical sigma.
CHECKERBOARD_FREQUENCY = 6
PYRAMID_HALF_BASE_RATIO = 0.28

# If true, remove old lowpoly + flat LOD cap B3DM files before writing new ones.
CLEAN_OLD_REAL_CAP_B3DM = True

# Semantic distance LOD. Uncertainty ON uses a coarse checkerboard parent and
# the selected detailed checkerboard child. Uncertainty OFF uses a separate
# true flat-cap tileset, never an SSE clamp on a contentless root.
LOD_COARSE_PREFIX = "coarse_"
LOD_FLAT_PREFIX = "flat_"
LOD_CHILD_GEOMETRIC_ERROR = 0.0
COARSE_ROLE_CODE_OFFSET = 10.0

# B3DM feature/batch table. Batch length is set per tile.
# This is what makes scene.pick(...).getProperty("rum_id") work.


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
        out = int(round(float(value)))
        return out
    except Exception:
        return fallback


def first_present(props: Dict[str, Any], names: Iterable[str], fallback: Any = None) -> Any:
    for name in names:
        if name in props and props[name] is not None and props[name] != "":
            return props[name]
    return fallback


def json_number_or_none(value: Any) -> Optional[float]:
    return safe_float(value, None)


def json_int_or_none(value: Any) -> Optional[int]:
    return safe_int(value, None)


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

    # Cesium Matrix4 arrays are column-major.
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

    pos_bytes = pack_floats(positions)
    normal_bytes = pack_floats(normals)
    texcoord_bytes = pack_floats(texcoords)
    batchid_bytes = pack_floats(batchids)
    index_bytes = pack_uint32(indices)

    pos_offset, pos_len = append_aligned(bin_buffer, pos_bytes, 4)
    normal_offset, normal_len = append_aligned(bin_buffer, normal_bytes, 4)
    texcoord_offset, texcoord_len = append_aligned(bin_buffer, texcoord_bytes, 4)
    batchid_offset, batchid_len = append_aligned(bin_buffer, batchid_bytes, 4)
    index_offset, index_len = append_aligned(bin_buffer, index_bytes, 4)

    pos_min, pos_max = component_min_max_vec3(positions)

    gltf = {
        "asset": {
            "version": "2.0",
            "generator": "InSAR4D RUM Viewer pipeline step 11",
        },
        "buffers": [
            {
                "byteLength": len(bin_buffer),
            }
        ],
        "bufferViews": [
            {
                "buffer": 0,
                "byteOffset": pos_offset,
                "byteLength": pos_len,
                "target": 34962,
            },
            {
                "buffer": 0,
                "byteOffset": normal_offset,
                "byteLength": normal_len,
                "target": 34962,
            },
            {
                "buffer": 0,
                "byteOffset": texcoord_offset,
                "byteLength": texcoord_len,
                "target": 34962,
            },
            {
                "buffer": 0,
                "byteOffset": batchid_offset,
                "byteLength": batchid_len,
                "target": 34962,
            },
            {
                "buffer": 0,
                "byteOffset": index_offset,
                "byteLength": index_len,
                "target": 34963,
            },
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

        "nodes": [
            {
                "mesh": 0,
                # z-up → y-up correction matrix (column-major)
                # cancels Cesium's automatic y-up → z-up transform at runtime
                "matrix": [1,0,0,0, 0,0,-1,0, 0,1,0,0, 0,0,0,1],
            }
        ],

        "scenes": [
            {
                "nodes": [0],
            }
        ],
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
    Cesium feature.getProperty("rum_id") and friends work after scene.pick().
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
# DATA PREP
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

        lookup[rum_id] = {
            "properties": props,
            "ring_lonlat": rings[0],
        }

    if not lookup:
        raise ValueError("No footprint features found")

    return lookup


def build_batch_table_for_rums(
    rum_ids: List[str],
    footprint_lookup: Dict[str, Dict[str, Any]],
    rum_index: Dict[str, Any],
) -> Dict[str, List[Any]]:
    """
    Legacy B3DM batch table for real RUM cap picking.

    Keep this intentionally compact but useful for popups:
      - rum_id and row_index identify the RUM and height texture row
      - grid_i/grid_j support diagnostics and fallback logic
      - up/east/north/covariance values support basic popup metadata
    """
    table: Dict[str, List[Any]] = {
        "rum_id": [],
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

    for rum_id in rum_ids:
        info = footprint_lookup[rum_id]
        props = info.get("properties") or {}
        row_index = int(rum_index[rum_id])

        east = json_number_or_none(first_present(props, ["east_mm_yr", "east", "vel_east", "ve"]))
        north = json_number_or_none(first_present(props, ["north_mm_yr", "north", "vel_north", "vn"]))
        up = json_number_or_none(first_present(props, ["up_mm_yr", "up", "vertical_velocity", "vel_up", "vu"]))

        if east is not None and north is not None:
            speed = math.sqrt(east * east + north * north)
        else:
            speed = json_number_or_none(first_present(props, ["speed_mm_yr", "horizontal_speed_mm_yr", "speed"]))

        table["rum_id"].append(str(rum_id))
        table["row_index"].append(row_index)
        table["height_row"].append(row_index)

        table["grid_i"].append(json_int_or_none(first_present(props, ["grid_i", "i"])))
        table["grid_j"].append(json_int_or_none(first_present(props, ["grid_j", "j"])))
        table["lon_center"].append(json_number_or_none(first_present(props, ["lon_center", "center_lon", "lon", "longitude"])))
        table["lat_center"].append(json_number_or_none(first_present(props, ["lat_center", "center_lat", "lat", "latitude"])))

        # Keep both names because old viewer code often asks for "up"; newer
        # code may prefer the explicit "up_mm_yr".
        table["up"].append(up)
        table["up_mm_yr"].append(up)

        table["east_mm_yr"].append(east)
        table["north_mm_yr"].append(north)
        table["speed_mm_yr"].append(speed)

        table["var_up"].append(json_number_or_none(first_present(props, ["var_up", "variance_up"])))
        table["var_east"].append(json_number_or_none(first_present(props, ["var_east", "variance_east", "var_e"])))
        table["var_north"].append(json_number_or_none(first_present(props, ["var_north", "variance_north", "var_n"])))
        table["covar_en"].append(json_number_or_none(first_present(props, ["covar_en", "cov_en", "covar_east_north"])))

    return table


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


def bilerp_vec3(
    sw: Tuple[float, float, float],
    se: Tuple[float, float, float],
    ne: Tuple[float, float, float],
    nw: Tuple[float, float, float],
    u: float,
    v: float,
) -> Tuple[float, float, float]:
    """Bilinear interpolation on the quadrilateral in SW,SE,NE,NW order."""
    south = tuple(sw[k] * (1.0 - u) + se[k] * u for k in range(3))
    north = tuple(nw[k] * (1.0 - u) + ne[k] * u for k in range(3))
    return tuple(south[k] * (1.0 - v) + north[k] * v for k in range(3))


def add_vertex(
    p: Tuple[float, float, float],
    relief_code: float,
    row_v: float,
    batch_id: int,
    positions: List[float],
    normals: List[float],
    texcoords: List[float],
    batchids: List[float],
) -> int:
    index = len(positions) // 3
    positions.extend([float(p[0]), float(p[1]), float(p[2])])
    # NORMAL is retained as a valid glTF attribute. Dynamic relief shading uses
    # screen derivatives of the deformed surface in the viewer, so it does not
    # rely on this static normal.
    normals.extend([0.0, 0.0, 1.0])
    # TEXCOORD_0.x contract:
    #   0 = fixed flat/base vertex
    #   +/-1..4 = top-platform corner SW,SE,NE,NW; sign = up/down checker cell
    # TEXCOORD_0.y remains the height-texture row lookup.
    texcoords.extend([float(relief_code), float(row_v)])
    batchids.append(float(batch_id))
    return index


def add_quad(indices: List[int], a: int, b: int, c: int, d: int) -> None:
    indices.extend([a, b, c, a, c, d])


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
    checkerboard_frequency: int,
    pyramid_half_base_ratio: float,
    role_code_offset: float = 0.0,
    pyramid_footprint_reference_frequency: int | None = None,
) -> None:
    """Build one lowpoly checkerboard cap for a square RUM footprint."""
    if len(ring_lonlat) < 4:
        raise ValueError("Footprint ring has fewer than 4 coordinates")

    corners_lonlat = ring_lonlat[:-1] if ring_lonlat[0] == ring_lonlat[-1] else ring_lonlat
    if len(corners_lonlat) != 4:
        raise ValueError(
            "Prototype1 lowpoly RUM caps require four-corner square footprints; "
            f"received {len(corners_lonlat)} corners"
        )

    center_ecef = geodetic_to_ecef(center_lon, center_lat, 0.0)
    east, north, up = enu_basis(center_lon, center_lat)

    local_corners: List[Tuple[float, float, float]] = []
    for lon, lat in corners_lonlat:
        ecef = geodetic_to_ecef(float(lon), float(lat), datum_height_m + CAP_CLEARANCE_M)
        local_corners.append(ecef_to_local_enu(ecef, center_ecef, east, north, up))

    # Step 03 emits SW, SE, NE, NW. Preserve that convention.
    sw, se, ne, nw = local_corners
    cells = max(1, int(checkerboard_frequency))
    reference_cells = max(1, int(pyramid_footprint_reference_frequency or cells))
    # Keep the physical pyramid footprint independent of checker spacing when
    # requested. Example: a 4x4 near grid can retain the old 6x6 pyramid base
    # width by using reference_cells=6.
    half_ratio = float(pyramid_half_base_ratio) * cells / reference_cells
    half_ratio = max(0.01, min(0.49, half_ratio))

    for j in range(cells):
        v0 = j / cells
        v1 = (j + 1) / cells
        vc = 0.5 * (v0 + v1)
        iv0 = vc - half_ratio * (v1 - v0)
        iv1 = vc + half_ratio * (v1 - v0)

        for i in range(cells):
            u0 = i / cells
            u1 = (i + 1) / cells
            uc = 0.5 * (u0 + u1)
            iu0 = uc - half_ratio * (u1 - u0)
            iu1 = uc + half_ratio * (u1 - u0)

            outer_pts = [
                bilerp_vec3(sw, se, ne, nw, u0, v0),
                bilerp_vec3(sw, se, ne, nw, u1, v0),
                bilerp_vec3(sw, se, ne, nw, u1, v1),
                bilerp_vec3(sw, se, ne, nw, u0, v1),
            ]
            inner_pts = [
                bilerp_vec3(sw, se, ne, nw, iu0, iv0),
                bilerp_vec3(sw, se, ne, nw, iu1, iv0),
                bilerp_vec3(sw, se, ne, nw, iu1, iv1),
                bilerp_vec3(sw, se, ne, nw, iu0, iv1),
            ]
            center_pt = bilerp_vec3(sw, se, ne, nw, uc, vc)
            sign = 1.0 if (i + j) % 2 == 0 else -1.0

            outer = [
                add_vertex(p, 0.0, row_v, batch_id, positions, normals, texcoords, batchids)
                for p in outer_pts
            ]
            inner = [
                add_vertex(p, 0.0, row_v, batch_id, positions, normals, texcoords, batchids)
                for p in inner_pts
            ]
            top = [
                add_vertex(center_pt, sign * (float(role_code_offset) + float(k + 1)), row_v, batch_id, positions, normals, texcoords, batchids)
                for k in range(4)
            ]

            # Flat ring: preserves quantitative velocity colour between spikes.
            add_quad(indices, outer[0], outer[1], inner[1], inner[0])
            add_quad(indices, outer[1], outer[2], inner[2], inner[1])
            add_quad(indices, outer[2], outer[3], inner[3], inner[2])
            add_quad(indices, outer[3], outer[0], inner[0], inner[3])

            # Four faceted side quads. The top vertices collapse to one apex for
            # sigma <= p98 and separate into a square plateau above p98.
            add_quad(indices, inner[0], inner[1], top[1], top[0])
            add_quad(indices, inner[1], inner[2], top[2], top[1])
            add_quad(indices, inner[2], inner[3], top[3], top[2])
            add_quad(indices, inner[3], inner[0], top[0], top[3])

            # Top platform; degenerate to a point for non-truncated pyramids.
            add_quad(indices, top[0], top[1], top[2], top[3])


def add_flat_polygon_to_buffers(
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
    """Build one lightweight animated flat RUM cap for the far LOD.

    TEXCOORD_0.x is the reserved flat-LOD marker 99 on every vertex. The
    production shader converts that marker to a zero relief role, applies model
    displacement and velocity colour, and uniformly bypasses uncertainty
    relief shading. Picking metadata remains identical to the lowpoly child.
    """
    if len(ring_lonlat) < 4:
        raise ValueError("Footprint ring has fewer than 4 coordinates")

    corners_lonlat = ring_lonlat[:-1] if ring_lonlat[0] == ring_lonlat[-1] else ring_lonlat
    if len(corners_lonlat) != 4:
        raise ValueError(
            "Prototype1 flat LOD caps require four-corner square footprints; "
            f"received {len(corners_lonlat)} corners"
        )

    center_ecef = geodetic_to_ecef(center_lon, center_lat, 0.0)
    east, north, up = enu_basis(center_lon, center_lat)
    local_corners: List[Tuple[float, float, float]] = []
    for lon, lat in corners_lonlat:
        ecef = geodetic_to_ecef(float(lon), float(lat), datum_height_m + CAP_CLEARANCE_M)
        local_corners.append(ecef_to_local_enu(ecef, center_ecef, east, north, up))

    vertices = [
        add_vertex(p, 99.0, row_v, batch_id, positions, normals, texcoords, batchids)
        for p in local_corners
    ]
    # SW, SE, NE, NW
    add_quad(indices, vertices[0], vertices[1], vertices[2], vertices[3])


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
    vunc_cfg = cfg["vertical_uncertainty_encoding"]
    tiling = cfg["tiling"]

    footprints_path = resolve_path(project_root, generated["rum_footprints"])
    packed_path = resolve_path(project_root, generated["packed_series"])
    height_meta_path = resolve_path(project_root, generated["height_meta"])
    relief_tiles_dir = resolve_path(project_root, paths["tiles_dir"])
    flat_tiles_dir = resolve_path(project_root, paths["flat_real_tiles_dir"])
    tile_index_path = relief_tiles_dir / TILE_INDEX_FILENAME
    relief_tileset_path = resolve_path(project_root, generated["real_caps_tileset"])
    flat_tileset_path = resolve_path(project_root, generated["flat_real_caps_tileset"])

    datum_height_m = float(viewer.get("display_datum_height_m", 1000.0))
    near_frequency = int(vunc_cfg.get("checkerboard_frequency_near", vunc_cfg.get("checkerboard_frequency", 4)))
    far_frequency = int(vunc_cfg.get("checkerboard_frequency_far", 2))
    pyramid_half_base_ratio = float(vunc_cfg.get("pyramid_half_base_ratio", PYRAMID_HALF_BASE_RATIO))
    near_footprint_reference_frequency = int(
        vunc_cfg.get("pyramid_footprint_reference_frequency_near", near_frequency)
    )
    far_footprint_reference_frequency = int(
        vunc_cfg.get("pyramid_footprint_reference_frequency_far", far_frequency)
    )
    bound_min_height_m = float(caps_cfg.get("bound_min_height_m", -1000.0))
    bound_max_height_m = float(caps_cfg.get("bound_max_height_m", 10000.0))
    parent_geometric_error = float(vunc_cfg.get("lod_parent_geometric_error_m", tiling.get("geometric_error_leaf", 100.0)))

    if near_frequency <= far_frequency:
        raise ValueError("checkerboard_frequency_near must be greater than checkerboard_frequency_far")

    section("Configuration")
    print(f"  Project root       : {project_root}")
    print(f"  Relief tileset out : {relief_tileset_path}")
    print(f"  Flat tileset out   : {flat_tileset_path}")
    print(f"  Display datum      : {datum_height_m} m")
    print(f"  Semantic LOD       : {far_frequency} × {far_frequency} -> {near_frequency} × {near_frequency}")
    print(f"  Pyramid half ratio : {pyramid_half_base_ratio}")
    print(f"  Near footprint ref : {near_footprint_reference_frequency} × {near_footprint_reference_frequency}")
    print(f"  Far footprint ref  : {far_footprint_reference_frequency} × {far_footprint_reference_frequency}")

    section("Loading inputs")
    footprints = load_json(footprints_path)
    packed = load_json(packed_path)
    height_meta = load_json(height_meta_path)
    tile_index = load_json(tile_index_path)

    footprint_lookup = build_footprint_lookup(footprints)
    rum_index = packed.get("rum_index") or {}
    texture = height_meta.get("texture") or {}
    texture_height = int(texture.get("height", 0))
    sigma_packing = (height_meta.get("packing") or {}).get("sigma") or {}
    sigma_threshold_mm = float(
        sigma_packing.get("sigma_visibility_threshold_mm")
        or (height_meta.get("vertical_uncertainty_encoding") or {}).get("visibility_threshold_mm")
        or 0.0
    )

    if not rum_index:
        raise ValueError("packed_series.json missing rum_index")
    if texture_height <= 0:
        raise ValueError("height_meta.json has invalid texture height")

    tiles = tile_index.get("tiles", [])
    if not isinstance(tiles, list) or not tiles:
        raise ValueError("tile_index.json has no tiles")

    ok(f"Loaded {len(footprint_lookup)} footprints")
    ok(f"Loaded {len(rum_index)} packed row indices")
    ok(f"Loaded tile index with {len(tiles)} tiles")
    print(f"  Relief visibility threshold: {sigma_threshold_mm:.4f} mm")

    relief_tiles_dir.mkdir(parents=True, exist_ok=True)
    flat_tiles_dir.mkdir(parents=True, exist_ok=True)
    if CLEAN_OLD_REAL_CAP_B3DM:
        removed = 0
        for pattern in ("tile_r*_c*.b3dm", f"{LOD_COARSE_PREFIX}tile_r*_c*.b3dm", f"{LOD_FLAT_PREFIX}tile_r*_c*.b3dm"):
            for old in relief_tiles_dir.glob(pattern):
                old.unlink()
                removed += 1
        for old in flat_tiles_dir.glob("*.b3dm"):
            old.unlink()
            removed += 1
        if removed:
            ok(f"Removed old real-cap B3DM files: {removed}")

    section("Building semantic relief LOD and separate flat caps")
    relief_children: List[Dict[str, Any]] = []
    flat_children: List[Dict[str, Any]] = []
    tile_count = 0
    total_rums = 0
    totals = {
        "flat_vertices": 0, "flat_triangles": 0,
        "far_vertices": 0, "far_triangles": 0,
        "near_vertices": 0, "near_triangles": 0,
    }

    for tile in tiles:
        rum_ids = tile.get("rum_ids", [])
        if not rum_ids:
            continue
        tile_id = tile["tile_id"]
        bbox = tile.get("bbox_wgs84")
        if not bbox:
            raise ValueError(f"Tile {tile_id} missing bbox_wgs84")
        center_lon, center_lat = tile_center_from_bbox(bbox)

        buffers = {}
        for name in ("flat", "far", "near"):
            buffers[name] = {"positions": [], "normals": [], "texcoords": [], "batchids": [], "indices": []}
        tile_batch_rum_ids: List[str] = []

        for rum_id in rum_ids:
            if rum_id not in footprint_lookup:
                raise ValueError(f"Tile {tile_id} references missing footprint {rum_id}")
            if rum_id not in rum_index:
                raise ValueError(f"Tile {tile_id} references missing row index {rum_id}")
            row_index = int(rum_index[rum_id])
            row_v = (row_index + 0.5) / texture_height
            batch_id = len(tile_batch_rum_ids)
            tile_batch_rum_ids.append(rum_id)
            ring = footprint_lookup[rum_id]["ring_lonlat"]

            b = buffers["flat"]
            add_flat_polygon_to_buffers(ring, row_v, batch_id, center_lon, center_lat, datum_height_m,
                                        b["positions"], b["normals"], b["texcoords"], b["batchids"], b["indices"])
            b = buffers["far"]
            add_polygon_to_buffers(ring, row_v, batch_id, center_lon, center_lat, datum_height_m,
                                   b["positions"], b["normals"], b["texcoords"], b["batchids"], b["indices"],
                                   far_frequency, pyramid_half_base_ratio, COARSE_ROLE_CODE_OFFSET,
                                   far_footprint_reference_frequency)
            b = buffers["near"]
            add_polygon_to_buffers(ring, row_v, batch_id, center_lon, center_lat, datum_height_m,
                                   b["positions"], b["normals"], b["texcoords"], b["batchids"], b["indices"],
                                   near_frequency, pyramid_half_base_ratio, 0.0,
                                   near_footprint_reference_frequency)

        batch_table = build_batch_table_for_rums(tile_batch_rum_ids, footprint_lookup, rum_index)
        filenames = {
            "flat": f"{LOD_FLAT_PREFIX}{tile_id}.b3dm",
            "far": f"{LOD_COARSE_PREFIX}{tile_id}.b3dm",
            "near": f"{tile_id}.b3dm",
        }
        for name in ("flat", "far", "near"):
            b = buffers[name]
            glb = build_glb(b["positions"], b["normals"], b["texcoords"], b["batchids"], b["indices"])
            payload = build_b3dm(glb=glb, batch_length=len(tile_batch_rum_ids), batch_table=batch_table)
            out_dir = flat_tiles_dir if name == "flat" else relief_tiles_dir
            write_binary(out_dir / filenames[name], payload)
            totals[f"{name}_vertices"] += len(b["positions"]) // 3
            totals[f"{name}_triangles"] += len(b["indices"]) // 3

        transform = enu_to_ecef_transform_column_major(center_lon, center_lat, 0.0)
        region = bounding_region_from_bbox(bbox, bound_min_height_m, bound_max_height_m)
        relief_children.append({
            "boundingVolume": {"region": region},
            "geometricError": parent_geometric_error,
            "refine": "REPLACE",
            "transform": transform,
            "content": {"uri": filenames["far"]},
            "children": [{
                "boundingVolume": {"region": region},
                "geometricError": LOD_CHILD_GEOMETRIC_ERROR,
                "content": {"uri": filenames["near"]},
                "extras": {"tile_id": tile_id, "lod": f"relief_{near_frequency}x{near_frequency}", "rum_count": len(rum_ids)},
            }],
            "extras": {"tile_id": tile_id, "lod": f"relief_{far_frequency}x{far_frequency}", "rum_count": len(rum_ids)},
        })
        flat_children.append({
            "boundingVolume": {"region": region},
            "geometricError": 0.0,
            "transform": transform,
            "content": {"uri": filenames["flat"]},
            "extras": {"tile_id": tile_id, "lod": "true_flat", "rum_count": len(rum_ids)},
        })
        tile_count += 1
        total_rums += len(rum_ids)

    if not relief_children or not flat_children:
        raise ValueError("No non-empty real cap tiles were built")

    ok(f"Built {tile_count} spatial tiles for each cap product")
    ok(f"Total RUM caps: {total_rums}")
    print(f"  Flat vertices / triangles : {totals['flat_vertices']} / {totals['flat_triangles']}")
    print(f"  Far vertices / triangles  : {totals['far_vertices']} / {totals['far_triangles']}")
    print(f"  Near vertices / triangles : {totals['near_vertices']} / {totals['near_triangles']}")

    footprint_meta = footprints.get("metadata") or {}
    dataset_bbox = footprint_meta.get("bbox_wgs84_footprints") or tile_index.get("metadata", {}).get("dataset_bbox_wgs84")
    if not dataset_bbox:
        raise ValueError("Cannot determine dataset bbox for root tileset")
    root_region = bounding_region_from_bbox(dataset_bbox, bound_min_height_m, bound_max_height_m)
    root_error = float(tiling.get("geometric_error_root", 5000.0))

    common_extras = {
        "source_footprints": generated["rum_footprints"],
        "source_packed_series": generated["packed_series"],
        "source_height_meta": generated["height_meta"],
        "height_texture": height_meta.get("height_texture"),
        "display_datum_height_m": datum_height_m,
        "texture_height": texture_height,
        "row_lookup": "TEXCOORD_0.y = (row_index + 0.5) / texture_height",
        "picking": "glTF _BATCHID + B3DM batch table exposes rum_id/row_index/up/grid properties",
        "real_rum_count": total_rums,
        "tile_count": tile_count,
    }

    relief_tileset = {
        "asset": {"version": "1.0", "tilesetVersion": "real_caps_v4_semantic_relief_lod", "generator": "InSAR4D step 11"},
        "geometricError": root_error,
        "root": {"boundingVolume": {"region": root_region}, "geometricError": root_error, "refine": "REPLACE", "children": relief_children},
        "extras": {
            **common_extras,
            "schema": "real_caps_tileset_v4_semantic_relief_lod",
            "lod": {
                "structure": f"relief_{far_frequency}x{far_frequency}_parent_to_{near_frequency}x{near_frequency}_child",
                "refine": "REPLACE",
                "parent_geometric_error_m": parent_geometric_error,
                "child_geometric_error_m": LOD_CHILD_GEOMETRIC_ERROR,
                "coarse_filename_prefix": LOD_COARSE_PREFIX,
                "coarse_role_code_offset": COARSE_ROLE_CODE_OFFSET,
            },
            "vertical_uncertainty_geometry": {
                "type": "lowpoly_checkerboard_spikes",
                "checkerboard_frequency_far": far_frequency,
                "checkerboard_frequency_near": near_frequency,
                "pyramid_half_base_ratio": pyramid_half_base_ratio,
                "pyramid_footprint_reference_frequency_near": near_footprint_reference_frequency,
                "pyramid_footprint_reference_frequency_far": far_footprint_reference_frequency,
                "visibility_threshold_mm": sigma_threshold_mm,
                "visibility_threshold_rule": "current raw sigma >= fixed global percentile threshold",
                "dynamic_plateau": "viewer derives plateau ratio from raw sigma above global p98",
            },
            "far_total_vertices": totals["far_vertices"], "far_total_triangles": totals["far_triangles"],
            "near_total_vertices": totals["near_vertices"], "near_total_triangles": totals["near_triangles"],
        },
    }
    flat_tileset = {
        "asset": {"version": "1.0", "tilesetVersion": "real_caps_v4_true_flat", "generator": "InSAR4D step 11"},
        "geometricError": root_error,
        "root": {"boundingVolume": {"region": root_region}, "geometricError": root_error, "refine": "ADD", "children": flat_children},
        "extras": {
            **common_extras,
            "schema": "real_caps_tileset_v4_true_flat",
            "purpose": "uncertainty relief OFF; animated MODEL displacement and velocity colour only",
            "flat_total_vertices": totals["flat_vertices"], "flat_total_triangles": totals["flat_triangles"],
        },
    }
    write_json(relief_tileset_path, relief_tileset)
    write_json(flat_tileset_path, flat_tileset)

    elapsed = time.time() - t_start
    ok(f"Wrote semantic relief tileset: {relief_tileset_path}")
    ok(f"Wrote true flat tileset: {flat_tileset_path}")
    section("Summary")
    ok(f"Step 11 complete in {elapsed:.2f} s")
    print(f"  Spatial tiles             : {tile_count}")
    print(f"  Real RUM caps             : {total_rums}")
    print(f"  Far relief                : {far_frequency} × {far_frequency}")
    print(f"  Near relief               : {near_frequency} × {near_frequency}")
    print(f"  Relief threshold          : {sigma_threshold_mm:.4f} mm")
    print(f"  Relief tileset            : {relief_tileset_path}")
    print(f"  Flat OFF tileset          : {flat_tileset_path}")


if __name__ == "__main__":
    main()
