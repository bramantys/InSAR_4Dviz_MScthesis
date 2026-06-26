#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

from _pass3_common import (
    Pass3Error,
    atomic_write_bytes,
    clean_stage_area,
    file_record,
    print_pass,
    project_root_from,
    read_json,
    require,
    stage_root,
    write_json,
)


from _proto2_config import load_project_config, output_data_dir
import _glb_cap_support as cap_support
import _glb_piston_support as piston_support
import _wall_glb_support as wall_support
import _opaque_glb_support as opaque_support

def component_stats(arr: np.ndarray) -> Dict[str, Any]:
    """
    Preserve the accepted historical metadata semantics exactly.

    The runtime arrays are float32.  The accepted summary calculated min,
    max and mean on those float32 values directly.  Promoting to float64
    before the aggregate changes the reported mean slightly even though the
    binary runtime asset is byte-identical.
    """
    finite_mask = np.isfinite(arr)
    values = arr[finite_mask]
    if values.size == 0:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "finite_count": 0,
            "nan_count": int(arr.size),
        }
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "finite_count": int(finite_mask.sum()),
        "nan_count": int(np.isnan(arr).sum()),
    }


def read_matrix(path: Path, rows: int, cols: int) -> np.ndarray:
    arr = np.fromfile(path, dtype="<f4")
    expected = rows * cols
    if arr.size != expected:
        raise Pass3Error(f"Unexpected float32 count in {path}: {arr.size:,} != {expected:,}")
    return arr.reshape(rows, cols)


def build_cap_glb(project_root: Path, out_path: Path) -> Dict[str, Any]:
    legacy = cap_support
    config = load_project_config(project_root)
    output_data = output_data_dir(project_root, config)
    vertices_candidate = output_data / "parcel_cap_mesh_vertices_indexed.parquet"
    if not vertices_candidate.exists():
        raise Pass3Error(
            "Missing indexed cap vertices: "
            f"{vertices_candidate}\n"
            "The bonestock template does not retain generated intermediates. "
            "Run the full pipeline so stages 00-05 rebuild them first."
        )
    vertices_path = vertices_candidate
    triangles_path = require(output_data / "parcel_cap_mesh_triangles_indexed.parquet", "indexed cap triangles")
    animation_manifest_path = require(output_data / "parcel_animation_manifest.json", "animation manifest")
    rev_path = require(output_data / "parcel_displacement_reversible_f32.bin", "reversible float32 array")
    irr_path = require(output_data / "parcel_displacement_irreversible_f32.bin", "irreversible float32 array")
    total_path = require(output_data / "parcel_displacement_total_f32.bin", "MC total float32 array")
    deterministic_total_path = require(output_data / "parcel_displacement_deterministic_total_f32.bin", "deterministic decomposition-total float32 array")

    animation_manifest = read_json(animation_manifest_path)
    shape = animation_manifest.get("shape") or animation_manifest.get("matrix_shape") or {}
    rows = int(shape.get("moving_parcels") or shape.get("rows") or shape.get("n_rows") or 0)
    epochs = int(shape.get("epochs") or shape.get("columns") or shape.get("n_epochs") or 0)
    if rows <= 0 or epochs <= 0:
        raise Pass3Error(f"Could not resolve animation shape from {animation_manifest_path}")

    reversible = read_matrix(rev_path, rows, epochs)
    irreversible = read_matrix(irr_path, rows, epochs)
    total = read_matrix(total_path, rows, epochs)
    deterministic_total = read_matrix(deterministic_total_path, rows, epochs)

    # V4.1 Total is the MC central estimate.  It must not be forced to equal
    # the separately supplied deterministic component decomposition.  Check
    # that decomposition only where all deterministic component cells exist.
    deterministic_mask = np.isfinite(reversible) & np.isfinite(irreversible) & np.isfinite(deterministic_total)
    if not deterministic_mask.any():
        raise Pass3Error("No finite deterministic component cells are available for decomposition validation")
    max_total_diff = float(np.abs(deterministic_total[deterministic_mask] - (reversible[deterministic_mask] + irreversible[deterministic_mask])).max())
    if max_total_diff > 1e-4:
        raise Pass3Error(f"deterministic total != reversible + irreversible; max diff={max_total_diff:.9g}")

    vertices = pd.read_parquet(vertices_path)
    triangles = pd.read_parquet(triangles_path)
    required_vertex_cols = ["global_vertex_index", "lon", "lat", "has_displacement", "displacement_row_index"]
    required_triangle_cols = ["global_triangle_index", "v0", "v1", "v2"]
    missing_v = [c for c in required_vertex_cols if c not in vertices.columns]
    missing_t = [c for c in required_triangle_cols if c not in triangles.columns]
    if missing_v or missing_t:
        raise Pass3Error(f"Missing mesh columns; vertices={missing_v}, triangles={missing_t}")

    vertices = vertices.sort_values("global_vertex_index").reset_index(drop=True)
    triangles = triangles.sort_values("global_triangle_index").reset_index(drop=True)
    if not np.array_equal(vertices["global_vertex_index"].to_numpy(np.int64), np.arange(len(vertices), dtype=np.int64)):
        raise Pass3Error("global_vertex_index is not contiguous")
    if not np.array_equal(triangles["global_triangle_index"].to_numpy(np.int64), np.arange(len(triangles), dtype=np.int64)):
        raise Pass3Error("global_triangle_index is not contiguous")

    lon = vertices["lon"].to_numpy(np.float64)
    lat = vertices["lat"].to_numpy(np.float64)
    west, east = float(lon.min()), float(lon.max())
    south, north = float(lat.min()), float(lat.max())
    center_lon = 0.5 * (west + east)
    center_lat = 0.5 * (south + north)
    static_height = float(getattr(legacy, "STATIC_HEIGHT_OFFSET_M", 4.0))

    x_ecef, y_ecef, z_ecef = legacy.wgs84_to_ecef(lon, lat, np.full_like(lon, static_height))
    local_x, local_y, local_z = legacy.ecef_to_local_enu(
        x_ecef, y_ecef, z_ecef, center_lon, center_lat, static_height
    )
    positions = np.column_stack([local_y, -local_x, local_z]).astype("<f4")

    disp_row = vertices["displacement_row_index"].to_numpy(np.float32)
    has_disp = vertices["has_displacement"].astype(bool).to_numpy()
    if np.nanmin(disp_row) < -1 or np.nanmax(disp_row) >= rows:
        raise Pass3Error("displacement_row_index is outside animation matrix")

    texcoord = np.empty((len(vertices), 2), dtype="<f4")
    texcoord[:, 0] = disp_row.astype("<f4")
    texcoord[:, 1] = has_disp.astype(np.float32)
    colors = np.empty((len(vertices), 4), dtype=np.uint8)
    colors[has_disp] = np.array([47, 128, 237, 215], dtype=np.uint8)
    colors[~has_disp] = np.array([184, 184, 184, 90], dtype=np.uint8)
    indices = triangles[["v0", "v1", "v2"]].to_numpy(dtype="<u4").reshape(-1)
    if int(indices.min()) < 0 or int(indices.max()) >= len(vertices):
        raise Pass3Error("Cap indices reference vertices out of range")

    glb = legacy.build_glb(positions, colors, texcoord, indices)
    atomic_write_bytes(out_path, glb)

    # Preserve the accepted historical metadata contract literally.
    #
    # Historical note:
    # POSITION is [north, -east, up] after the 90-degree orientation fix.
    # The historical export stored POSITION[:, 0] under the key "east_west"
    # and POSITION[:, 1] under "north_south".  Those labels are therefore
    # historical render-axis labels, not clean geographic semantics.
    #
    # Pass 3 is a strict parity pass, so we preserve that accepted contract
    # exactly.  A future schema migration may rename these fields deliberately.
    # The span calculation also remains on float32 POSITION values, matching
    # the accepted camera-height precision.
    span_east_west = float(positions[:, 0].max() - positions[:, 0].min())
    span_north_south = float(positions[:, 1].max() - positions[:, 1].min())
    epoch = animation_manifest.get("epoch") or {}
    start = str(epoch.get("start") or animation_manifest.get("epoch_start") or "")
    end = str(epoch.get("end") or animation_manifest.get("epoch_end") or "")
    labels = epoch.get("labels") if isinstance(epoch, dict) else None
    if not isinstance(labels, list) or len(labels) != epochs:
        try:
            labels = pd.date_range(start=start, end=end, periods=epochs).strftime("%Y-%m-%d").tolist()
        except Exception:
            labels = [f"epoch {i}" for i in range(epochs)]

    return {
        "vertices": int(len(vertices)),
        "triangles": int(len(triangles)),
        "indices": int(indices.size),
        "moving_vertices": int(has_disp.sum()),
        "blank_vertices": int((~has_disp).sum()),
        "moving_parcels": rows,
        "epochs": epochs,
        "epoch_start": start,
        "epoch_end": end,
        "epoch_labels": labels,
        "stats": {
            "reversible": component_stats(reversible),
            "irreversible": component_stats(irreversible),
            "deterministic_total": component_stats(deterministic_total),
            "total": component_stats(total),
        },
        "max_total_diff": max_total_diff,
        "max_deterministic_component_diff": max_total_diff,
        "total_product": "monte_carlo_mean_t",
        "center_lon": center_lon,
        "center_lat": center_lat,
        "center_height_m": static_height,
        "camera_height_m": max(span_east_west, span_north_south) * 2.2,
        "bounds_wgs84": {"west": west, "south": south, "east": east, "north": north},
        "local_span_m": {
            "east_west": span_east_west,
            "north_south": span_north_south,
        },
    }



def ceil_to_step(value: float, step: float) -> float:
    """Mirror the viewer's ceilToDatumStep() exactly for positive values."""
    if not math.isfinite(value) or value <= 0.0 or not math.isfinite(step) or step <= 0.0:
        return 0.0
    return math.ceil(value / step) * step


def derive_uncertainty_diagnostic_static_bounds(
    *,
    total: np.ndarray,
    sigma_h: np.ndarray,
    reversible: np.ndarray,
    irreversible: np.ndarray,
    row_index: int,
    piston_module: Any,
) -> Dict[str, Any]:
    """Return a conservative static Z envelope for the 90647 diagnostic GLB.

    Cesium derives a Model's bounding volume from static glTF POSITION data,
    while the viewer's uncertainty shader overwrites ``positionMC.z`` every
    frame. A one-parcel GLB therefore needs static sentinel vertices spanning
    every Z value the shader may create. This function mirrors the active
    Total-mode runtime controls:

    - vertical exaggeration slider: 0 .. ``vertical_exag_max`` in whole steps;
    - Total reference epoch: any loaded epoch;
    - spike relief slider: 0 .. 500x;
    - Total-mode datum rule used by the viewer.

    The result is intentionally diagnostic-only for parcel 90647. The
    all-parcel rollout should retain the same calculation per combined GLB,
    not multiply this asset into one Model per parcel.
    """
    if total.ndim != 2 or sigma_h.ndim != 2 or total.shape != sigma_h.shape:
        raise Pass3Error(f"Total/sigma shape mismatch for uncertainty bounds: total={total.shape}, sigma={sigma_h.shape}")
    if row_index < 0 or row_index >= total.shape[0]:
        raise Pass3Error(f"Diagnostic row {row_index} is outside Total/sigma matrices")

    total_series = np.asarray(total[row_index], dtype=np.float64)
    sigma_series = np.asarray(sigma_h[row_index], dtype=np.float64)
    finite_total = total_series[np.isfinite(total_series)]
    finite_sigma = sigma_series[np.isfinite(sigma_series)]
    if finite_total.size == 0:
        raise Pass3Error(f"Diagnostic row {row_index} has no finite Total samples")
    if finite_sigma.size == 0 or np.any(finite_sigma < 0.0):
        raise Pass3Error(f"Diagnostic row {row_index} has invalid sigma samples")

    # Stage 08 derives this same tuning product. Re-deriving it here gives
    # Stage 06 the exact model-origin height that the assembled viewer later
    # stores as META.center_height_m.
    display_tuning = piston_module.derive_display_tuning({
        "reversible": reversible,
        "irreversible": irreversible,
        "total": total,
    })

    height_scale_per_exag = float(display_tuning["height_scale_per_exag_unit_m_per_mm"])
    vertical_exag_max = int(round(float(display_tuning["vertical_exag_max"])))
    total_downward_mm = float(display_tuning["component_downward_mm"]["total"])
    min_datum_m = float(display_tuning["min_display_datum_height_m"])
    datum_step_m = float(display_tuning["datum_round_step_m"])
    safety_clearance_m = float(display_tuning["safety_clearance_m"])
    model_origin_height_m = float(display_tuning["display_datum_height_m"])

    # Must remain in sync with the Total-mode HTML controls / shader.
    uncertainty_base_m_per_mm = 0.1
    uncertainty_scale_max = 500.0
    static_bound_padding_m = 25.0

    # The viewer lets users re-reference Total to any loaded epoch. The
    # greatest possible displayed signal is consequently max(series)-min(series)
    # in either direction, not simply the raw series min/max.
    total_reference_span_mm = float(finite_total.max() - finite_total.min())
    total_display_min_mm = -total_reference_span_mm
    total_display_max_mm = total_reference_span_mm
    sigma_max_mm = float(finite_sigma.max())
    relief_amplitude_max_m = sigma_max_mm * uncertainty_base_m_per_mm * uncertainty_scale_max

    z_min = math.inf
    z_max = -math.inf
    for exag in range(0, vertical_exag_max + 1):
        height_scale = float(exag) * height_scale_per_exag
        active_total_datum_m = max(
            min_datum_m,
            ceil_to_step(total_downward_mm * height_scale + safety_clearance_m, datum_step_m),
        )
        datum_local_z = active_total_datum_m - model_origin_height_m
        z_min = min(z_min, datum_local_z + total_display_min_mm * height_scale - relief_amplitude_max_m)
        z_max = max(z_max, datum_local_z + total_display_max_mm * height_scale + relief_amplitude_max_m)

    if not math.isfinite(z_min) or not math.isfinite(z_max) or z_min >= z_max:
        raise Pass3Error(f"Could not derive a valid uncertainty static bound: z=[{z_min}, {z_max}]")

    static_z_min = float(math.floor(z_min - static_bound_padding_m))
    static_z_max = float(math.ceil(z_max + static_bound_padding_m))
    return {
        "method": "degenerate_static_bound_sentinels_v1",
        "scope": "parcel_90647_diagnostic_only",
        "model_origin_height_m": model_origin_height_m,
        "vertical_exag_min": 0,
        "vertical_exag_max": vertical_exag_max,
        "height_scale_per_exag_unit_m_per_mm": height_scale_per_exag,
        "total_downward_mm_global": total_downward_mm,
        "total_reference_span_mm_parcel": total_reference_span_mm,
        "sigma_max_mm_parcel": sigma_max_mm,
        "uncertainty_base_m_per_mm": uncertainty_base_m_per_mm,
        "uncertainty_scale_max": uncertainty_scale_max,
        "relief_amplitude_max_m": relief_amplitude_max_m,
        "padding_m": static_bound_padding_m,
        "static_z_min_m": static_z_min,
        "static_z_max_m": static_z_max,
    }

def build_uncertainty_cap_glb(project_root: Path, cap_summary: Dict[str, Any], out_path: Path) -> Dict[str, Any]:
    """Build the one-parcel real-geometry uncertainty diagnostic.

    This is deliberately limited to parcel 90647. It validates the actual Cesium
    geometry recipe before any city-wide renderer is attempted:
    - 10 m global/map-aligned lattice
    - 3.5 m four-sided up pyramids
    - 4.5 m inverted dimples
    - true holes beneath dimples, so they cannot be hidden by a flat cap
    - static degenerate bound sentinels covering all shader-driven Z motion
    - no expensive all-parcel polygon boolean loop

    The sentinel vertices do not render. They only expand the glTF POSITION
    accessor/bounding sphere so Cesium's depth-frustum partitioning includes
    the surface after the viewer shader rewrites positionMC.z.
    """
    target_parcel_id = 90647
    spacing_m = 10.0
    up_feature_size_m = 3.5
    down_feature_size_m = 4.5

    cap_module = cap_support
    piston_module = piston_support

    config = load_project_config(project_root)
    output_data = output_data_dir(project_root, config)
    footprint_path = require(output_data / "parcel_footprint_vertices.csv", "parcel footprint vertices")
    vertices_path = require(output_data / "parcel_cap_mesh_vertices_indexed.parquet", "indexed cap vertices")

    footprint_df = pd.read_csv(footprint_path)
    footprint_df = footprint_df.loc[footprint_df["parcel_id"].astype(str) == str(target_parcel_id)].copy()
    if footprint_df.empty:
        raise Pass3Error(f"Diagnostic parcel {target_parcel_id} is absent from parcel_footprint_vertices.csv")

    vdf = pd.read_parquet(vertices_path)
    target_vertices = vdf.loc[
        (vdf["parcel_id"].astype(str) == str(target_parcel_id)) & vdf["has_displacement"].astype(bool)
    ].copy()
    if target_vertices.empty:
        raise Pass3Error(f"Diagnostic parcel {target_parcel_id} has no moving-mesh rows")
    row_values = target_vertices["displacement_row_index"].dropna().astype(int).unique().tolist()
    if len(row_values) != 1:
        raise Pass3Error(f"Diagnostic parcel {target_parcel_id} must map to one displacement row; got {row_values}")
    row_index = int(row_values[0])

    animation_manifest = read_json(require(output_data / "parcel_animation_manifest.json", "animation manifest"))
    shape = animation_manifest.get("shape") or animation_manifest.get("matrix_shape") or {}
    rows = int(shape.get("moving_parcels") or shape.get("rows") or shape.get("n_rows") or 0)
    epochs = int(shape.get("epochs") or shape.get("columns") or shape.get("n_epochs") or 0)
    if rows <= 0 or epochs <= 0:
        raise Pass3Error("Could not resolve animation shape for uncertainty-bound calculation")
    if row_index >= rows:
        raise Pass3Error(f"Diagnostic row {row_index} is outside animation matrix row count {rows}")

    reversible = read_matrix(require(output_data / "parcel_displacement_reversible_f32.bin", "reversible float32 array"), rows, epochs)
    irreversible = read_matrix(require(output_data / "parcel_displacement_irreversible_f32.bin", "irreversible float32 array"), rows, epochs)
    total = read_matrix(require(output_data / "parcel_displacement_total_f32.bin", "MC Total float32 array"), rows, epochs)
    sigma_h = read_matrix(require(output_data / "parcel_displacement_sigma_h_f32.bin", "sigmaH float32 array"), rows, epochs)
    static_bounds = derive_uncertainty_diagnostic_static_bounds(
        total=total,
        sigma_h=sigma_h,
        reversible=reversible,
        irreversible=irreversible,
        row_index=row_index,
        piston_module=piston_module,
    )

    center_lon = float(cap_summary["center_lon"])
    center_lat = float(cap_summary["center_lat"])
    static_height = float(cap_summary["center_height_m"])
    lon = footprint_df["lon"].to_numpy(dtype=np.float64)
    lat = footprint_df["lat"].to_numpy(dtype=np.float64)
    h = np.full_like(lon, static_height, dtype=np.float64)
    ecef_x, ecef_y, ecef_z = cap_module.wgs84_to_ecef(lon, lat, h)
    local_x, local_y, _ = cap_module.ecef_to_local_enu(
        ecef_x, ecef_y, ecef_z, center_lon, center_lat, static_height
    )
    footprint_df["local_x"] = local_x
    footprint_df["local_y"] = local_y
    sentinel_anchor_x = float(np.mean(local_x))
    sentinel_anchor_y = float(np.mean(local_y))

    def polygon_from_part(part_df: pd.DataFrame) -> Polygon:
        exterior = None
        holes: List[List[tuple[float, float]]] = []
        for (ring_type, _ring_index), ring_df in part_df.groupby(["ring_type", "ring_index"], sort=False):
            ring_df = ring_df.sort_values("vertex_index")
            coords = [(float(x), float(y)) for x, y in ring_df[["local_x", "local_y"]].to_numpy()]
            if len(coords) >= 2 and coords[0] == coords[-1]:
                coords = coords[:-1]
            if len(coords) < 3:
                continue
            if str(ring_type).lower() == "exterior":
                exterior = coords
            else:
                holes.append(coords)
        if exterior is None:
            raise Pass3Error(f"Diagnostic parcel {target_parcel_id} part has no exterior ring")
        poly = Polygon(exterior, holes)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or not isinstance(poly, Polygon):
            raise Pass3Error(f"Diagnostic parcel {target_parcel_id} part did not form one valid polygon")
        return poly

    def iter_polygons(geom):
        if geom is None or geom.is_empty:
            return []
        if isinstance(geom, Polygon):
            return [geom]
        if isinstance(geom, MultiPolygon):
            return [g for g in geom.geoms if not g.is_empty]
        return []

    def earcut_indices(poly: Polygon) -> tuple[np.ndarray, np.ndarray]:
        try:
            import mapbox_earcut as earcut
        except ImportError as exc:
            raise Pass3Error("mapbox-earcut is required for the real-spike diagnostic mesh") from exc

        def open_ring(coords):
            out = [(float(x), float(y)) for x, y in list(coords)]
            if len(out) >= 2 and out[0] == out[-1]:
                out = out[:-1]
            return out

        rings = [open_ring(poly.exterior.coords)] + [open_ring(r.coords) for r in poly.interiors]
        rings = [ring for ring in rings if len(ring) >= 3]
        if not rings:
            return np.empty((0, 2), dtype=np.float64), np.empty((0, 3), dtype=np.uint32)
        xy = np.asarray([pt for ring in rings for pt in ring], dtype=np.float64)
        ring_ends = np.cumsum([len(ring) for ring in rings], dtype=np.uint32)
        try:
            tri = earcut.triangulate_float64(xy, ring_ends)
        except TypeError:
            tri = earcut.triangulate_float64(xy.flatten(), ring_ends)
        tri = np.asarray(tri, dtype=np.uint32)
        return xy, tri.reshape((-1, 3))

    pos_rows: List[List[float]] = []
    tex0_rows: List[List[float]] = []
    tex1_rows: List[List[float]] = []
    idx_rows: List[int] = []

    def push_vertex(x: float, y: float, relief: float) -> int:
        idx = len(pos_rows)
        # Match the main cap GLB’s local orientation exactly.
        pos_rows.append([float(y), float(-x), float(relief)])
        tex0_rows.append([float(row_index), 1.0])
        tex1_rows.append([float(relief), 0.0])
        return idx

    def add_static_bound_sentinel(z: float) -> None:
        """Add one zero-area triangle that expands only the static GLB bounds.

        All three vertices occupy exactly the same point, so rasterization emits
        no fragments. TEXCOORD_1 relief is deliberately zero: even if a driver
        reaches the degenerate primitive, the custom shader leaves it at the
        flat diagnostic surface rather than producing a visible rogue spike.
        """
        sentinel_indices: List[int] = []
        for _ in range(3):
            idx = len(pos_rows)
            pos_rows.append([sentinel_anchor_y, -sentinel_anchor_x, float(z)])
            tex0_rows.append([float(row_index), 1.0])
            tex1_rows.append([0.0, 0.0])
            sentinel_indices.append(idx)
        idx_rows.extend(sentinel_indices)

    def add_tri(a: tuple[float, float, float], b: tuple[float, float, float], c: tuple[float, float, float]) -> None:
        ia = push_vertex(*a)
        ib = push_vertex(*b)
        ic = push_vertex(*c)
        idx_rows.extend([ia, ib, ic])

    def add_flat_geometry(geom) -> int:
        face_count = 0
        for poly in iter_polygons(geom):
            xy, tri = earcut_indices(poly)
            for i0, i1, i2 in tri:
                a, b, c = xy[int(i0)], xy[int(i1)], xy[int(i2)]
                add_tri((a[0], a[1], 0.0), (b[0], b[1], 0.0), (c[0], c[1], 0.0))
                face_count += 1
        return face_count

    def add_pyramid(square: Polygon, relief: float) -> int:
        coords = list(square.exterior.coords)
        if len(coords) >= 2 and coords[0] == coords[-1]:
            coords = coords[:-1]
        if len(coords) != 4:
            raise Pass3Error("Diagnostic feature footprint must remain a four-sided square")
        centroid = square.centroid
        apex = (float(centroid.x), float(centroid.y), float(relief))
        for i in range(4):
            a = coords[i]
            b = coords[(i + 1) % 4]
            if relief >= 0.0:
                add_tri((a[0], a[1], 0.0), (b[0], b[1], 0.0), apex)
            else:
                add_tri((b[0], b[1], 0.0), (a[0], a[1], 0.0), apex)
        return 4

    total_flat_triangles = 0
    up_features = 0
    down_features = 0
    part_count = 0

    for _part_index, part_df in footprint_df.groupby("part_index", sort=False):
        parcel_poly = polygon_from_part(part_df)
        minx, miny, maxx, maxy = parcel_poly.bounds
        up_squares: List[Polygon] = []
        down_squares: List[Polygon] = []
        for ix in range(int(math.floor(minx / spacing_m)), int(math.ceil(maxx / spacing_m)) + 1):
            cx = ix * spacing_m
            for iy in range(int(math.floor(miny / spacing_m)), int(math.ceil(maxy / spacing_m)) + 1):
                cy = iy * spacing_m
                is_up = ((ix + iy) % 2 == 0)
                size = up_feature_size_m if is_up else down_feature_size_m
                half = size * 0.5
                square = Polygon([
                    (cx - half, cy - half),
                    (cx + half, cy - half),
                    (cx + half, cy + half),
                    (cx - half, cy + half),
                ])
                # Keep a deliberately flat boundary remainder: every feature is a true square,
                # never a clipped partial pseudo-pyramid.
                if not parcel_poly.covers(square):
                    continue
                if is_up:
                    up_squares.append(square)
                else:
                    down_squares.append(square)

        dimple_holes = unary_union(down_squares) if down_squares else None
        flat_surface = parcel_poly.difference(dimple_holes) if dimple_holes is not None else parcel_poly
        total_flat_triangles += add_flat_geometry(flat_surface)
        for square in up_squares:
            add_pyramid(square, +1.0)
        for square in down_squares:
            add_pyramid(square, -1.0)
        up_features += len(up_squares)
        down_features += len(down_squares)
        part_count += 1

    if not pos_rows or not idx_rows:
        raise Pass3Error("Diagnostic uncertainty cap geometry was empty")

    visible_triangles = len(idx_rows) // 3
    add_static_bound_sentinel(float(static_bounds["static_z_min_m"]))
    add_static_bound_sentinel(float(static_bounds["static_z_max_m"]))
    sentinel_vertices = 6
    sentinel_triangles = 2

    positions = np.asarray(pos_rows, dtype="<f4")
    if float(positions[:, 2].min()) > float(static_bounds["static_z_min_m"]) or float(positions[:, 2].max()) < float(static_bounds["static_z_max_m"]):
        raise Pass3Error("Uncertainty bound sentinels did not survive POSITION construction")
    colors = np.tile(np.array([[255, 255, 255, 255]], dtype=np.uint8), (len(positions), 1))
    tex0 = np.asarray(tex0_rows, dtype="<f4")
    tex1 = np.asarray(tex1_rows, dtype="<f4")
    indices = np.asarray(idx_rows, dtype="<u4")
    atomic_write_bytes(out_path, piston_module.build_glb(positions, colors, tex0, tex1, indices))

    return {
        "scope": "single_parcel_real_geometry_diagnostic",
        "parcel_id": target_parcel_id,
        "displacement_row_index": row_index,
        "spacing_m": spacing_m,
        "up_feature_size_m": up_feature_size_m,
        "down_feature_size_m": down_feature_size_m,
        "parts": part_count,
        "up_features": up_features,
        "down_features": down_features,
        "flat_triangles": total_flat_triangles,
        "visible_triangles": int(visible_triangles),
        "sentinel_vertices": sentinel_vertices,
        "sentinel_triangles": sentinel_triangles,
        "static_bounds": static_bounds,
        "vertices": int(len(positions)),
        "triangles": int(len(indices) // 3),
        "indices": int(len(indices)),
        "glb_bytes": int(out_path.stat().st_size),
    }


def build_piston_and_aux(project_root: Path, cap_summary: Dict[str, Any], paths: Dict[str, Path]) -> Dict[str, Any]:
    piston_module = piston_support
    wall_module = wall_support
    opaque_module = opaque_support

    center_lon = float(cap_summary["center_lon"])
    center_lat = float(cap_summary["center_lat"])
    cap_pos, cap_col, cap_tex0, cap_tex1, cap_idx, cap_stats = piston_module.read_cap_mesh(center_lon, center_lat)
    wall_pos, wall_col, wall_tex0, wall_tex1, wall_idx, wall_stats = piston_module.read_wall_mesh(center_lon, center_lat)

    vertex_offset = np.uint32(cap_pos.shape[0])
    positions = np.vstack([cap_pos, wall_pos]).astype("<f4")
    colors = np.vstack([cap_col, wall_col]).astype(np.uint8)
    tex0 = np.vstack([cap_tex0, wall_tex0]).astype("<f4")
    tex1 = np.vstack([cap_tex1, wall_tex1]).astype("<f4")
    indices = np.concatenate([cap_idx, (wall_idx + vertex_offset).astype("<u4")]).astype("<u4")
    if int(indices.min()) < 0 or int(indices.max()) >= len(positions):
        raise Pass3Error("Combined piston indices reference vertices out of range")

    atomic_write_bytes(paths["pistons"], piston_module.build_glb(positions, colors, tex0, tex1, indices))
    wall_summary = wall_module.build_wall_only_glb_from_piston(paths["pistons"], paths["walls"])
    opaque_summary = opaque_module.build_opaque_datum_cap_glb(paths["caps"], paths["opaque_datum_caps"])

    return {
        "cap_stats": cap_stats,
        "wall_stats": wall_stats,
        "combined": {
            "vertices": int(len(positions)),
            "triangles": int(len(indices) // 3),
            "indices": int(len(indices)),
        },
        "wall_split": wall_summary,
        "opaque_datum": opaque_summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build staged Proto2 runtime geometry using accepted algorithms.")
    parser.parse_args()

    project_root = project_root_from(__file__)
    geometry_dir = clean_stage_area(project_root, "geometry")
    paths = {
        "caps": geometry_dir / "parcel_caps.glb",
        "pistons": geometry_dir / "parcel_pistons.glb",
        "walls": geometry_dir / "parcel_walls.glb",
        "opaque_datum_caps": geometry_dir / "parcel_datum_caps_opaque.glb",
    }

    print("\n=== PROTO2 STAGE 06: BUILD MAIN RUNTIME GEOMETRY ===")
    cap_summary = build_cap_glb(project_root, paths["caps"])
    aux_summary = build_piston_and_aux(project_root, cap_summary, paths)
    summary = {
        "schema": "proto2_runtime_geometry_build_v5_3",
        "algorithm_source": [
            "_glb_cap_support.py",
            "_glb_piston_support.py",
            "_wall_glb_support.py",
            "_opaque_glb_support.py",
        ],
        "cap": cap_summary,
        "piston": aux_summary,
        "outputs": {key: file_record(path, project_root) for key, path in paths.items()},
    }
    report = stage_root(project_root) / "geometry_build_summary.json"
    write_json(report, summary)
    print_pass("STAGE 06 RESULT", report)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Pass3Error as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        raise SystemExit(1)
