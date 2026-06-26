from __future__ import annotations

from pathlib import Path
import json
import math
import shutil
import struct
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


from _proto2_config import load_project_config, output_cesium_dir, output_data_dir, stage_records_dir


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = load_project_config(PROJECT_ROOT)
OUTPUT_DATA = output_data_dir(PROJECT_ROOT, CONFIG)
OUTPUT_CESIUM = output_cesium_dir(PROJECT_ROOT, CONFIG)
RUN_RECORDS = stage_records_dir(PROJECT_ROOT, CONFIG)

# Inputs from previous phases
PHASE09_SUMMARY = OUTPUT_CESIUM / "proto2_cesium_animated_glb_summary.json"
PHASE14_HTML = OUTPUT_CESIUM / "proto2_m1_parcel_color_viewer.html"
PHASE14_SUMMARY = OUTPUT_CESIUM / "proto2_m1_parcel_color_viewer_summary.json"
PHASE14_ASSET_DIR = OUTPUT_CESIUM / "phase14_color_assets"

CAP_VERTICES = OUTPUT_DATA / "parcel_cap_mesh_vertices_indexed.parquet"
CAP_TRIANGLES = OUTPUT_DATA / "parcel_cap_mesh_triangles_indexed.parquet"
FOOTPRINT_VERTICES_CSV = OUTPUT_DATA / "parcel_footprint_vertices.csv"
PARCEL_RENDER_INDEX = OUTPUT_DATA / "parcel_render_index.parquet"
MATRIX_NPZ = OUTPUT_DATA / "parcel_displacement_matrices_float32.npz"

# Outputs
ASSET_DIR = OUTPUT_CESIUM / "phase15_piston_assets"
PISTON_GLB = ASSET_DIR / "proto2_irreversible_piston_mesh.glb"
DISPLAY_TUNING_JSON = OUTPUT_DATA / "parcel_display_tuning.json"
PISTON_SUMMARY_JSON = OUTPUT_DATA / "parcel_piston_mesh_summary.json"

REPORT_TXT_OUT = RUN_RECORDS / "phase15a_irreversible_piston_assets_report.txt"
REPORT_JSON_OUT = RUN_RECORDS / "phase15a_irreversible_piston_assets_report.json"

# Display tuning defaults. These are pipeline constants, not viewer hardcodes.
HEIGHT_SCALE_PER_EXAG_UNIT_M_PER_MM = 0.1
VERTICAL_EXAG_DEFAULT = 10.0
VERTICAL_EXAG_MAX = 300.0
GROUND_HEIGHT_M = 0.0
SAFETY_CLEARANCE_M = 5.0
DATUM_ROUND_STEP_M = 5.0
MIN_DISPLAY_DATUM_HEIGHT_M = 10.0

# Irreversible piston prototype display boost.
# This is intentionally pipeline-owned, not hardcoded in the viewer.
# Slider semantics: 1x = 0.1 m/mm, 10x = 1.0 m/mm, 50x = 5.0 m/mm.
IRREVERSIBLE_PISTON_MOTION_BOOST = 1.0

# Required Phase14 runtime assets copied forward.
PHASE14_RUNTIME_ASSETS = [
    "parcel_displacement_reversible_f32.bin",
    "parcel_displacement_irreversible_f32.bin",
    "parcel_displacement_total_f32.bin",
    "parcel_pick_index.json",
    "parcel_vi_f32.bin",
    "parcel_color_scales.json",
]


WARNINGS: List[str] = []


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    sys.exit(1)


def ok(message: str) -> None:
    print(f"[OK] {message}")


def warn(message: str) -> None:
    WARNINGS.append(message)
    print(f"[WARN] {message}")


def require(path: Path, label: str) -> None:
    if not path.exists():
        fail(f"Missing {label}: {path}")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def pad4_bytes(data: bytes, pad_byte: bytes = b"\x00") -> bytes:
    pad = (-len(data)) % 4
    if pad:
        data += pad_byte * pad
    return data


def ceil_to_step(value: float, step: float) -> float:
    if not math.isfinite(float(value)) or value <= 0 or step <= 0:
        return 0.0
    return float(math.ceil(float(value) / float(step)) * float(step))


def read_matrices_for_tuning() -> Dict[str, np.ndarray]:
    require(MATRIX_NPZ, "displacement matrix NPZ")
    data = np.load(MATRIX_NPZ)
    keys = {k.lower(): k for k in data.files}

    def get_component(*names: str) -> np.ndarray:
        for name in names:
            if name.lower() in keys:
                return np.asarray(data[keys[name.lower()]], dtype=np.float32)
        for key in data.files:
            low = key.lower()
            if any(name.lower() in low for name in names):
                return np.asarray(data[key], dtype=np.float32)
        fail(f"Could not find matrix component in {MATRIX_NPZ}; tried {names}; available keys={data.files}")
        raise AssertionError

    return {
        "reversible": get_component("reversible"),
        "irreversible": get_component("irreversible"),
        "total": get_component("total", "h_spams_final"),
    }


def derive_display_tuning(matrices: Dict[str, np.ndarray]) -> Dict[str, Any]:
    mins = {}
    maxs = {}
    for name, arr in matrices.items():
        finite = np.asarray(arr, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            fail(f"No finite values found in matrix {name}")
        mins[name] = float(np.min(finite))
        maxs[name] = float(np.max(finite))

    # Phase 15 is intentionally an irreversible-only piston architecture test.
    # If we used total/reversible extremes with a 50x display slider, the datum
    # would jump to hundreds of meters and irreversible pistons would become tiny
    # again. So the safe datum is derived from the active/supported component.
    datum_components = ["irreversible"]
    component_motion_boost = {
        "reversible": 1.0,
        "irreversible": float(IRREVERSIBLE_PISTON_MOTION_BOOST),
        "total": 1.0,
    }

    downward_mm_by_component = {}
    boosted_downward_mm_by_component = {}
    for name, min_value in mins.items():
        downward_mm = max(0.0, -float(min_value))
        boost = float(component_motion_boost.get(name, 1.0))
        downward_mm_by_component[name] = downward_mm
        boosted_downward_mm_by_component[name] = downward_mm * boost

    active_boosted = {
        name: boosted_downward_mm_by_component[name]
        for name in datum_components
        if name in boosted_downward_mm_by_component
    }
    if not active_boosted:
        fail(f"No datum components found in displacement matrices: {datum_components}")

    max_downward_mm = max(downward_mm_by_component.values()) if downward_mm_by_component else 0.0
    max_active_boosted_downward_mm = max(active_boosted.values())
    controlling_component = max(active_boosted, key=active_boosted.get) # type: ignore

    max_height_scale = float(VERTICAL_EXAG_MAX) * float(HEIGHT_SCALE_PER_EXAG_UNIT_M_PER_MM)
    default_height_scale = float(VERTICAL_EXAG_DEFAULT) * float(HEIGHT_SCALE_PER_EXAG_UNIT_M_PER_MM)
    max_downward_display_m = max_active_boosted_downward_mm * max_height_scale
    raw_datum = max_downward_display_m + float(SAFETY_CLEARANCE_M)
    display_datum_height_m = max(float(MIN_DISPLAY_DATUM_HEIGHT_M), ceil_to_step(raw_datum, DATUM_ROUND_STEP_M))

    irreversible_downward_mm = downward_mm_by_component.get("irreversible", 0.0)
    irreversible_default_display_m = irreversible_downward_mm * float(IRREVERSIBLE_PISTON_MOTION_BOOST) * default_height_scale
    irreversible_max_display_m = irreversible_downward_mm * float(IRREVERSIBLE_PISTON_MOTION_BOOST) * max_height_scale

    return {
        "schema": "proto2_parcel_display_tuning_v3_extreme_slider_irreversible_scope",
        "purpose": "Pipeline-derived display datum for ground-anchored parcel pistons.",
        "datum_scope": "irreversible_piston_prototype_only",
        "datum_components": datum_components,
        "ground_height_m": float(GROUND_HEIGHT_M),
        "height_scale_per_exag_unit_m_per_mm": float(HEIGHT_SCALE_PER_EXAG_UNIT_M_PER_MM),
        "vertical_exag_default": float(VERTICAL_EXAG_DEFAULT),
        "vertical_exag_max": float(VERTICAL_EXAG_MAX),
        "slider_semantics": "1x = 0.1 m/mm, 10x = 1.0 m/mm, 50x = 5.0 m/mm",
        "default_height_scale_m_per_mm": round(float(default_height_scale), 6),
        "max_height_scale_m_per_mm": round(float(max_height_scale), 6),
        "component_min_mm": {k: round(v, 6) for k, v in mins.items()},
        "component_max_mm": {k: round(v, 6) for k, v in maxs.items()},
        "component_downward_mm": {k: round(v, 6) for k, v in downward_mm_by_component.items()},
        "component_motion_boost": {k: round(float(v), 6) for k, v in component_motion_boost.items()},
        "component_boosted_downward_mm": {k: round(v, 6) for k, v in boosted_downward_mm_by_component.items()},
        "controlling_component_for_datum": controlling_component,
        "max_downward_displacement_mm_all_components": round(float(max_downward_mm), 6),
        "max_active_boosted_downward_displacement_mm": round(float(max_active_boosted_downward_mm), 6),
        "max_downward_display_m_at_slider_max": round(float(max_downward_display_m), 6),
        "irreversible_motion_boost": float(IRREVERSIBLE_PISTON_MOTION_BOOST),
        "irreversible_default_downward_display_m": round(float(irreversible_default_display_m), 6),
        "irreversible_max_downward_display_m": round(float(irreversible_max_display_m), 6),
        "safety_clearance_m": float(SAFETY_CLEARANCE_M),
        "datum_round_step_m": float(DATUM_ROUND_STEP_M),
        "min_display_datum_height_m": float(MIN_DISPLAY_DATUM_HEIGHT_M),
        "display_datum_height_m": round(float(display_datum_height_m), 6),
        "rule": (
            "display_datum_height_m = max(min_datum, ceil_to_step("
            "max(abs(min_irreversible_mm) * irreversible_motion_boost) "
            "* vertical_exag_max * height_scale_per_exag_unit + safety_clearance, datum_round_step))"
        ),
        "note": (
            "Viewer reads datum and slider scale from this product; neither is hardcoded in JavaScript. "
            "This datum is intentionally scoped to the irreversible-only piston prototype. Future total/reversible piston viewers should derive their own mode-appropriate datum."
        ),
    }

# -----------------------------------------------------------------------------
# Local ENU conversion helpers
# -----------------------------------------------------------------------------

def wgs84_to_ecef(lon_deg: np.ndarray, lat_deg: np.ndarray, h_m: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = 6378137.0
    e2 = 6.69437999014e-3
    lon = np.deg2rad(lon_deg.astype(np.float64))
    lat = np.deg2rad(lat_deg.astype(np.float64))
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    N = a / np.sqrt(1.0 - e2 * sin_lat * sin_lat)
    x = (N + h_m) * cos_lat * np.cos(lon)
    y = (N + h_m) * cos_lat * np.sin(lon)
    z = ((1.0 - e2) * N + h_m) * sin_lat
    return x, y, z


def ecef_to_local_enu(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    center_lon_deg: float,
    center_lat_deg: float,
    center_h_m: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cx, cy, cz = wgs84_to_ecef(
        np.array([center_lon_deg], dtype=np.float64),
        np.array([center_lat_deg], dtype=np.float64),
        np.array([center_h_m], dtype=np.float64),
    )
    cx, cy, cz = float(cx[0]), float(cy[0]), float(cz[0])

    lon0 = math.radians(center_lon_deg)
    lat0 = math.radians(center_lat_deg)
    sin_lon, cos_lon = math.sin(lon0), math.cos(lon0)
    sin_lat, cos_lat = math.sin(lat0), math.cos(lat0)

    east = np.array([-sin_lon, cos_lon, 0.0], dtype=np.float64)
    north = np.array([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat], dtype=np.float64)
    up = np.array([cos_lat * cos_lon, cos_lat * sin_lon, sin_lat], dtype=np.float64)

    dx = x - cx
    dy = y - cy
    dz = z - cz

    local_x = dx * east[0] + dy * east[1] + dz * east[2]
    local_y = dx * north[0] + dy * north[1] + dz * north[2]
    local_z = dx * up[0] + dy * up[1] + dz * up[2]
    return local_x, local_y, local_z


def lonlat_to_glb_xy(lon: np.ndarray, lat: np.ndarray, center_lon: float, center_lat: float) -> Tuple[np.ndarray, np.ndarray]:
    x_ecef, y_ecef, z_ecef = wgs84_to_ecef(lon, lat, np.zeros_like(lon, dtype=np.float64))
    local_x, local_y, _ = ecef_to_local_enu(x_ecef, y_ecef, z_ecef, center_lon, center_lat, 0.0)
    # Final Proto2 GLB orientation fix: GLB internal XY uses rotated local ENU.
    # CPU pick index still uses plain ENU, but this mesh passes through the glTF node matrix.
    glb_x = local_y
    glb_y = -local_x
    return glb_x, glb_y


def find_col(df: pd.DataFrame, candidates: Iterable[str], label: str, required: bool = True) -> Optional[str]:
    lower_map = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    if required:
        fail(f"Could not find {label}; tried {list(candidates)}; available columns={list(df.columns)}")
    return None


def read_cap_mesh(center_lon: float, center_lat: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    require(CAP_VERTICES, "indexed cap mesh vertices")
    require(CAP_TRIANGLES, "indexed cap mesh triangles")
    vertices = pd.read_parquet(CAP_VERTICES)
    triangles = pd.read_parquet(CAP_TRIANGLES)

    lon_col = find_col(vertices, ["lon", "longitude", "x_lon"], "cap vertex longitude", required=False)
    lat_col = find_col(vertices, ["lat", "latitude", "y_lat"], "cap vertex latitude", required=False)
    if lon_col and lat_col:
        lon = vertices[lon_col].to_numpy(dtype=np.float64)
        lat = vertices[lat_col].to_numpy(dtype=np.float64)
        glb_x, glb_y = lonlat_to_glb_xy(lon, lat, center_lon, center_lat)
    else:
        # Fallback for already-local products. This is less preferred but keeps the script debuggable.
        x_col = find_col(vertices, ["local_x", "local_x_m", "x_local", "x_m", "x"], "cap local x")
        y_col = find_col(vertices, ["local_y", "local_y_m", "y_local", "y_m", "y"], "cap local y")
        warn(f"Using local XY columns for cap mesh fallback: {x_col}, {y_col}; assuming they are plain ENU and applying GLB rotation.")
        local_x = vertices[x_col].to_numpy(dtype=np.float64)
        local_y = vertices[y_col].to_numpy(dtype=np.float64)
        glb_x = local_y
        glb_y = -local_x

    positions = np.column_stack([glb_x, glb_y, np.zeros_like(glb_x)]).astype("<f4")

    row_col = find_col(vertices, ["displacement_row_index", "row_index", "moving_row", "texture_row"], "cap displacement row")
    disp_row = vertices[row_col].to_numpy(dtype=np.float32)

    has_col = find_col(vertices, ["has_displacement", "is_moving", "moving"], "cap has displacement flag", required=False)
    if has_col:
        has_disp = vertices[has_col].astype(bool).to_numpy()
    else:
        has_disp = np.isfinite(disp_row) & (disp_row >= 0)

    tex0 = np.empty((len(vertices), 2), dtype="<f4")
    tex0[:, 0] = disp_row.astype("<f4")
    tex0[:, 1] = has_disp.astype(np.float32)

    tex1 = np.empty((len(vertices), 2), dtype="<f4")
    tex1[:, 0] = 1.0  # piston_t: top follows animated height
    tex1[:, 1] = 0.0  # wall_flag: cap

    colors = np.empty((len(vertices), 4), dtype=np.uint8)
    colors[has_disp] = np.array([47, 128, 237, 255], dtype=np.uint8)
    colors[~has_disp] = np.array([184, 184, 184, 160], dtype=np.uint8)

    tri_cols = [find_col(triangles, [name], f"triangle column {name}") for name in ["v0", "v1", "v2"]]
    indices = triangles[tri_cols].to_numpy(dtype="<u4").reshape(-1)

    if int(indices.min()) < 0 or int(indices.max()) >= len(vertices):
        fail("Cap triangle indices reference vertices out of range")

    stats = {
        "cap_vertices": int(len(vertices)),
        "cap_triangles": int(len(indices) // 3),
        "moving_cap_vertices": int(has_disp.sum()),
        "blank_cap_vertices": int((~has_disp).sum()),
    }
    return positions, colors, tex0, tex1, indices, stats # type: ignore


def load_render_row_lookup() -> Dict[str, Tuple[float, float]]:
    require(PARCEL_RENDER_INDEX, "parcel render index")
    df = pd.read_parquet(PARCEL_RENDER_INDEX)
    parcel_col = find_col(df, ["parcel_id", "int_id", "pnt_id", "source_parcel_id"], "render parcel id")
    row_col = find_col(df, ["displacement_row_index", "row_index", "moving_row", "texture_row"], "render displacement row")
    has_col = find_col(df, ["has_displacement", "is_moving", "moving"], "render has displacement", required=False)

    out: Dict[str, Tuple[float, float]] = {}
    for _, r in df.iterrows():
        key = str(r[parcel_col]) # type: ignore
        row = float(r[row_col]) if pd.notna(r[row_col]) else -1.0 # type: ignore
        if has_col:
            has = 1.0 if bool(r[has_col]) else 0.0
        else:
            has = 1.0 if row >= 0.0 else 0.0
        out[key] = (row, has)
    return out


def ring_key_columns(df: pd.DataFrame) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    parcel_col = find_col(df, ["parcel_id", "int_id", "pnt_id", "source_parcel_id"], "footprint parcel id")
    part_col = find_col(df, ["part_index", "part_id", "geometry_part_index", "part"], "footprint part", required=False)
    ring_col = find_col(df, ["ring_index", "ring_id", "ring", "interior_ring_index"], "footprint ring", required=False)
    order_col = find_col(df, ["vertex_index", "vertex_id", "point_index", "coord_index", "sequence", "order"], "footprint vertex order", required=False)
    return parcel_col, part_col, ring_col, order_col # type: ignore


def read_wall_mesh(center_lon: float, center_lat: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    require(FOOTPRINT_VERTICES_CSV, "parcel footprint vertices CSV")
    df = pd.read_csv(FOOTPRINT_VERTICES_CSV)
    parcel_col, part_col, ring_col, order_col = ring_key_columns(df)

    lon_col = find_col(df, ["lon", "longitude", "x_lon"], "footprint longitude", required=False)
    lat_col = find_col(df, ["lat", "latitude", "y_lat"], "footprint latitude", required=False)
    use_lonlat = lon_col is not None and lat_col is not None

    if not use_lonlat:
        x_col = find_col(df, ["local_x", "local_x_m", "x_local", "x_m", "x"], "footprint local x")
        y_col = find_col(df, ["local_y", "local_y_m", "y_local", "y_m", "y"], "footprint local y")
        warn(f"Using local XY columns for footprint fallback: {x_col}, {y_col}; assuming plain ENU and applying GLB rotation.")
    else:
        x_col = y_col = None

    row_lookup = load_render_row_lookup()

    group_cols = [parcel_col]
    if part_col:
        group_cols.append(part_col)
    if ring_col:
        group_cols.append(ring_col)

    pos_chunks: List[np.ndarray] = []
    color_chunks: List[np.ndarray] = []
    tex0_chunks: List[np.ndarray] = []
    tex1_chunks: List[np.ndarray] = []
    index_chunks: List[np.ndarray] = []

    vertex_offset = 0
    segment_count = 0
    skipped_segments = 0
    moving_segments = 0
    blank_segments = 0

    grouped = df.groupby(group_cols, sort=False, dropna=False)
    for key, g in grouped:
        if order_col:
            g = g.sort_values(order_col)
        parcel_id = key[0] if isinstance(key, tuple) else key
        parcel_key = str(parcel_id)
        row, has = row_lookup.get(parcel_key, (-1.0, 0.0))

        if use_lonlat:
            lon = g[lon_col].to_numpy(dtype=np.float64)
            lat = g[lat_col].to_numpy(dtype=np.float64)
            gx, gy = lonlat_to_glb_xy(lon, lat, center_lon, center_lat)
        else:
            local_x = g[x_col].to_numpy(dtype=np.float64)
            local_y = g[y_col].to_numpy(dtype=np.float64)
            gx = local_y
            gy = -local_x

        n = len(gx)
        if n < 2:
            continue

        # Avoid a duplicate closing segment if the ring already repeats first vertex at the end.
        closed = bool(np.hypot(gx[0] - gx[-1], gy[0] - gy[-1]) < 1e-6)
        last_i = n - 1 if closed else n

        for i in range(last_i):
            j = i + 1
            if j >= n:
                j = 0
            ax, ay = float(gx[i]), float(gy[i])
            bx, by = float(gx[j]), float(gy[j])
            if not all(math.isfinite(v) for v in [ax, ay, bx, by]):
                skipped_segments += 1
                continue
            if math.hypot(ax - bx, ay - by) < 1e-6:
                skipped_segments += 1
                continue

            # Four vertices per segment: bottom A, bottom B, top B, top A.
            pos = np.array([
                [ax, ay, 0.0],
                [bx, by, 0.0],
                [bx, by, 0.0],
                [ax, ay, 0.0],
            ], dtype="<f4")
            pos_chunks.append(pos)

            c = np.array([47, 128, 237, 255] if has > 0.5 else [184, 184, 184, 160], dtype=np.uint8)
            color_chunks.append(np.tile(c, (4, 1)))

            tex0 = np.array([[row, has], [row, has], [row, has], [row, has]], dtype="<f4")
            tex0_chunks.append(tex0)
            tex1 = np.array([[0.0, 1.0], [0.0, 1.0], [1.0, 1.0], [1.0, 1.0]], dtype="<f4")
            tex1_chunks.append(tex1)

            idx = np.array([
                vertex_offset + 0, vertex_offset + 1, vertex_offset + 2,
                vertex_offset + 0, vertex_offset + 2, vertex_offset + 3,
            ], dtype="<u4")
            index_chunks.append(idx)
            vertex_offset += 4
            segment_count += 1
            if has > 0.5:
                moving_segments += 1
            else:
                blank_segments += 1

    if not pos_chunks:
        fail("No wall geometry was generated from footprint vertices")

    positions = np.vstack(pos_chunks).astype("<f4")
    colors = np.vstack(color_chunks).astype(np.uint8)
    tex0 = np.vstack(tex0_chunks).astype("<f4")
    tex1 = np.vstack(tex1_chunks).astype("<f4")
    indices = np.concatenate(index_chunks).astype("<u4")

    stats = {
        "wall_segments": int(segment_count),
        "wall_triangles": int(len(indices) // 3),
        "wall_vertices": int(len(positions)),
        "moving_wall_segments": int(moving_segments),
        "blank_wall_segments": int(blank_segments),
        "skipped_segments": int(skipped_segments),
    }
    return positions, colors, tex0, tex1, indices, stats


# -----------------------------------------------------------------------------
# GLB writer with TEXCOORD_1 piston role attributes
# -----------------------------------------------------------------------------

def build_glb(positions_f32: np.ndarray, colors_u8: np.ndarray, tex0_f32: np.ndarray, tex1_f32: np.ndarray, indices_u32: np.ndarray) -> bytes:
    if positions_f32.dtype != np.dtype("<f4"):
        fail("positions_f32 must be little-endian float32")
    if colors_u8.dtype != np.dtype("uint8"):
        fail("colors_u8 must be uint8")
    if tex0_f32.dtype != np.dtype("<f4"):
        fail("tex0_f32 must be little-endian float32")
    if tex1_f32.dtype != np.dtype("<f4"):
        fail("tex1_f32 must be little-endian float32")
    if indices_u32.dtype != np.dtype("<u4"):
        fail("indices_u32 must be little-endian uint32")

    vertex_count = int(positions_f32.shape[0])
    index_count = int(indices_u32.size)

    chunks: List[bytes] = []
    buffer_views: List[Dict[str, Any]] = []
    accessors: List[Dict[str, Any]] = []
    byte_offset = 0

    def add_buffer_view(data_bytes: bytes, target: int) -> int:
        nonlocal byte_offset
        aligned_offset = (byte_offset + 3) // 4 * 4
        padding_needed = aligned_offset - byte_offset
        if padding_needed:
            chunks.append(b"\x00" * padding_needed)
            byte_offset = aligned_offset
        view_index = len(buffer_views)
        buffer_views.append({"buffer": 0, "byteOffset": byte_offset, "byteLength": len(data_bytes), "target": target})
        chunks.append(data_bytes)
        byte_offset += len(data_bytes)
        return view_index

    pos_view = add_buffer_view(positions_f32.tobytes(order="C"), 34962)
    color_view = add_buffer_view(colors_u8.tobytes(order="C"), 34962)
    tex0_view = add_buffer_view(tex0_f32.tobytes(order="C"), 34962)
    tex1_view = add_buffer_view(tex1_f32.tobytes(order="C"), 34962)
    index_view = add_buffer_view(indices_u32.tobytes(order="C"), 34963)
    bin_chunk = pad4_bytes(b"".join(chunks), b"\x00")

    pos_min = positions_f32.min(axis=0).astype(float).tolist()
    pos_max = positions_f32.max(axis=0).astype(float).tolist()

    pos_accessor = len(accessors)
    accessors.append({"bufferView": pos_view, "byteOffset": 0, "componentType": 5126, "count": vertex_count, "type": "VEC3", "min": pos_min, "max": pos_max})
    color_accessor = len(accessors)
    accessors.append({"bufferView": color_view, "byteOffset": 0, "componentType": 5121, "count": vertex_count, "type": "VEC4", "normalized": True})
    tex0_accessor = len(accessors)
    accessors.append({"bufferView": tex0_view, "byteOffset": 0, "componentType": 5126, "count": vertex_count, "type": "VEC2"})
    tex1_accessor = len(accessors)
    accessors.append({"bufferView": tex1_view, "byteOffset": 0, "componentType": 5126, "count": vertex_count, "type": "VEC2"})
    index_accessor = len(accessors)
    accessors.append({"bufferView": index_view, "byteOffset": 0, "componentType": 5125, "count": index_count, "type": "SCALAR", "min": [int(indices_u32.min())], "max": [int(indices_u32.max())]})

    gltf = {
        "asset": {"version": "2.0", "generator": "Proto2 Phase 15 irreversible piston mesh exporter"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{
            "mesh": 0,
            "name": "proto2_irreversible_piston_mesh",
            "matrix": [
                1, 0, 0, 0,
                0, 0, -1, 0,
                0, 1, 0, 0,
                0, 0, 0, 1,
            ],
        }],
        "meshes": [{
            "name": "proto2_irreversible_piston_mesh",
            "primitives": [{
                "attributes": {
                    "POSITION": pos_accessor,
                    "COLOR_0": color_accessor,
                    "TEXCOORD_0": tex0_accessor,
                    "TEXCOORD_1": tex1_accessor,
                },
                "indices": index_accessor,
                "material": 0,
                "mode": 4,
            }],
        }],
        "materials": [{
            "name": "piston_vertex_material",
            "doubleSided": True,
            "alphaMode": "OPAQUE",
            "pbrMetallicRoughness": {
                "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 1.0,
            },
        }],
        "buffers": [{"byteLength": len(bin_chunk)}],
        "bufferViews": buffer_views,
        "accessors": accessors,
    }

    json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    json_chunk = pad4_bytes(json_bytes, b" ")
    total_length = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
    header = struct.pack("<4sII", b"glTF", 2, total_length)
    json_header = struct.pack("<I4s", len(json_chunk), b"JSON")
    bin_header = struct.pack("<I4s", len(bin_chunk), b"BIN\x00")
    return header + json_header + json_chunk + bin_header + bin_chunk


def main() -> None:
    OUTPUT_DATA.mkdir(parents=True, exist_ok=True)
    OUTPUT_CESIUM.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    RUN_RECORDS.mkdir(parents=True, exist_ok=True)

    print("\n=== PROTO2 PHASE 15A: IRREVERSIBLE PISTON ASSETS ===")
    print(f"Project root: {PROJECT_ROOT}")

    require(PHASE09_SUMMARY, "Phase09 summary")
    require(PHASE14_HTML, "Phase14 color viewer HTML")
    require(PHASE14_SUMMARY, "Phase14 color viewer summary")
    require(PHASE14_ASSET_DIR, "Phase14 asset directory")
    for name in PHASE14_RUNTIME_ASSETS:
        require(PHASE14_ASSET_DIR / name, f"Phase14 runtime asset {name}")

    phase09 = json.loads(PHASE09_SUMMARY.read_text(encoding="utf-8"))
    center_lon = float(phase09["center_lon"])
    center_lat = float(phase09["center_lat"])

    print("\nDeriving display datum tuning...")
    matrices = read_matrices_for_tuning()
    display_tuning = derive_display_tuning(matrices)
    write_json(DISPLAY_TUNING_JSON, display_tuning)
    ok(f"wrote {DISPLAY_TUNING_JSON}")
    ok(f"display datum height = {display_tuning['display_datum_height_m']} m")

    print("\nBuilding cap mesh section...")
    cap_pos, cap_col, cap_tex0, cap_tex1, cap_idx, cap_stats = read_cap_mesh(center_lon, center_lat) # type: ignore
    ok(f"cap vertices={cap_stats['cap_vertices']:,}, cap triangles={cap_stats['cap_triangles']:,}")

    print("\nBuilding piston wall mesh section...")
    wall_pos, wall_col, wall_tex0, wall_tex1, wall_idx, wall_stats = read_wall_mesh(center_lon, center_lat)
    ok(f"wall vertices={wall_stats['wall_vertices']:,}, wall triangles={wall_stats['wall_triangles']:,}, skipped={wall_stats['skipped_segments']:,}")

    print("\nCombining mesh...")
    vertex_offset = np.uint32(cap_pos.shape[0])
    wall_idx_shifted = (wall_idx + vertex_offset).astype("<u4")
    positions = np.vstack([cap_pos, wall_pos]).astype("<f4")
    colors = np.vstack([cap_col, wall_col]).astype(np.uint8)
    tex0 = np.vstack([cap_tex0, wall_tex0]).astype("<f4")
    tex1 = np.vstack([cap_tex1, wall_tex1]).astype("<f4")
    indices = np.concatenate([cap_idx, wall_idx_shifted]).astype("<u4") # type: ignore

    if int(indices.min()) < 0 or int(indices.max()) >= len(positions):
        fail("Combined indices reference vertices out of range")

    ok(f"combined vertices={len(positions):,}, triangles={len(indices)//3:,}")

    print("\nWriting GLB...")
    PISTON_GLB.write_bytes(build_glb(positions, colors, tex0, tex1, indices))
    ok(f"wrote {PISTON_GLB} ({PISTON_GLB.stat().st_size / (1024*1024):.2f} MB)")

    print("\nCopying Phase14 runtime assets into Phase15 asset directory...")
    copied = []
    for name in PHASE14_RUNTIME_ASSETS:
        src = PHASE14_ASSET_DIR / name
        dst = ASSET_DIR / name
        shutil.copy2(src, dst)
        copied.append(name)
        ok(f"copied {name}")

    summary = {
        "product": "proto2_irreversible_piston_assets",
        "purpose": "Ground-anchored 3D parcel piston prototype for irreversible/drowning mode.",
        "center_lon": center_lon,
        "center_lat": center_lat,
        "center_height_m": float(display_tuning["display_datum_height_m"]),
        "ground_height_m": float(display_tuning["ground_height_m"]),
        "display_tuning": display_tuning,
        "geometry_contract": {
            "TEXCOORD_0.x": "displacement_row_index; -1 for blank/no displacement",
            "TEXCOORD_0.y": "has_displacement flag",
            "TEXCOORD_1.x": "piston_t: 0 = ground/base vertex, 1 = animated top/cap vertex",
            "TEXCOORD_1.y": "wall_flag: 0 = cap, 1 = wall",
            "model_matrix_height": "center_height_m = display_datum_height_m; shader places base at ground_height_m - display_datum_height_m",
            "glb_xy_orientation": "Phase09/12 GLB correction: x=local_north, y=-local_east; CPU pick index remains plain ENU",
        },
        "cap_stats": cap_stats,
        "wall_stats": wall_stats,
        "combined": {
            "vertices": int(len(positions)),
            "triangles": int(len(indices) // 3),
            "indices": int(len(indices)),
            "glb_size_mb": float(PISTON_GLB.stat().st_size / (1024 * 1024)),
        },
        "outputs": {
            "glb": str(PISTON_GLB),
            "display_tuning": str(DISPLAY_TUNING_JSON),
            "piston_summary": str(PISTON_SUMMARY_JSON),
            "asset_dir": str(ASSET_DIR),
        },
        "copied_phase14_assets": copied,
        "warnings": WARNINGS,
    }
    write_json(PISTON_SUMMARY_JSON, summary)
    write_json(REPORT_JSON_OUT, summary)
    REPORT_TXT_OUT.write_text(
        "PROTO2 PHASE 15A: IRREVERSIBLE PISTON ASSETS\n"
        f"Project root: {PROJECT_ROOT}\n"
        f"Display datum height: {display_tuning['display_datum_height_m']} m\n"
        f"Ground height: {display_tuning['ground_height_m']} m\n"
        f"Cap vertices: {cap_stats['cap_vertices']:,}\n"
        f"Wall vertices: {wall_stats['wall_vertices']:,}\n"
        f"Combined vertices: {len(positions):,}\n"
        f"Combined triangles: {len(indices)//3:,}\n"
        f"GLB: {PISTON_GLB}\n"
        f"Warnings: {len(WARNINGS)}\n",
        encoding="utf-8",
    )

    print("\n=== PHASE 15A RESULT: PASS ===")


if __name__ == "__main__":
    main()
