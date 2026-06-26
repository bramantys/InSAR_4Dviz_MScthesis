#!/usr/bin/env python3
"""Stage 06b — build full-field tiled uncertainty-cap GLBs for two locked LOD families.

This is the production uncertainty geometry stage. It consumes the locked
20 m detail and 40 m overview layout audits, parcel footprints, row mapping,
and animation arrays.

For every moving parcel part, the output retains a flat cap remainder. Downward
uncertainty facets become true holes in that remainder; upward facets are real
pyramids. The output is split into a fixed 4 × 4 map tile grid. Every tile gets
its own static shader-motion sentinels for Cesium bounds/frustum stability.

The companion Gate 3 viewer renders exactly one LOD family at a time and hides
only the normal moving Total caps while the full-field uncertainty layer is
ready and enabled. Normal Total blank caps and walls are left alone.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import GeometryCollection, MultiPolygon, Point, Polygon
from shapely.ops import unary_union

from _pass3_common import (
    Pass3Error,
    stage_root,
    semantic_root,
    atomic_write_bytes,
    file_record,
    print_pass,
    project_root_from,
    read_json,
    require,
    write_json,
)
from _proto2_config import load_project_config, output_data_dir
import _glb_cap_support as cap_support
import _glb_piston_support as piston_support


DELIVERY_ID = "PROTO2_UNCERTAINTY_LOD_TILES_V5_3"
SCRIPT_ID = "PROTO2_BUILD_UNCERTAINTY_LOD_TILES"
SCHEMA = "proto2_uncertainty_lod_tiles_v5_1"
METRIC_CRS = "EPSG:28992"
DETAIL_SPACING_M = 20.0
OVERVIEW_SPACING_M = 40.0
READABLE_MIN_BASE_M = 2.5
UNCERTAINTY_BASE_M_PER_MM = 1.0
UNCERTAINTY_SCALE_MAX = 50.0
STATIC_BOUND_PADDING_M = 25.0
TILE_COLUMNS = 4
TILE_ROWS = 4

# Feature centres travel through a metric -> WGS84 GeoJSON -> metric round trip.
# Some carriers intentionally touch a parcel boundary, so a microscopic centre
# drift can make the re-created full square fail the audit's 0.1 µm fit check.
# Never clip a carrier: at most 5 mm of UNIFORM side reduction is allowed,
# preserving a true square. A larger mismatch remains a hard failure.
RECONSTRUCTION_MAX_SIDE_SHRINK_M = 0.005
RECONSTRUCTION_MAX_CENTER_NUDGE_M = 0.005
RECONSTRUCTION_FIT_BUFFER_M = 1.0e-7
RECONSTRUCTION_BINARY_STEPS = 24
RECONSTRUCTION_NUDGE_DIRECTIONS = 16


@dataclass(frozen=True)
class LodFamily:
    key: str
    label: str
    spacing_m: float
    audit_stage: str
    output_subdir: str


@dataclass(frozen=True)
class FeaturePoint:
    sign: int
    side_m: float
    point_metric: Any


@dataclass
class TileCounts:
    part_count: int = 0
    fallback_flat_parts: int = 0
    nonrenderable_flat_parts: int = 0
    up_features: int = 0
    down_features: int = 0
    flat_triangles: int = 0
    precision_recovered_carriers: int = 0
    max_precision_side_shrink_m: float = 0.0
    precision_nudged_carriers: int = 0
    max_precision_center_nudge_m: float = 0.0


FAMILIES = (
    LodFamily(
        key="detail",
        label="Detail · 20 m",
        spacing_m=DETAIL_SPACING_M,
        audit_stage="uncertainty_layout_detail_20m",
        output_subdir="detail_20m",
    ),
    LodFamily(
        key="overview",
        label="Overview · 40 m",
        spacing_m=OVERVIEW_SPACING_M,
        audit_stage="uncertainty_layout_overview_40m",
        output_subdir="overview_40m",
    ),
)


# -----------------------------------------------------------------------------
# Generic geometry / mesh helpers
# -----------------------------------------------------------------------------

def _read_matrix(path: Path, rows: int, cols: int) -> np.ndarray:
    arr = np.fromfile(path, dtype="<f4")
    expected = rows * cols
    if arr.size != expected:
        raise Pass3Error(f"Unexpected float32 count in {path}: {arr.size:,} != {expected:,}")
    return arr.reshape(rows, cols)


def _ceil_to_step(value: float, step: float) -> float:
    if not math.isfinite(value) or value <= 0.0 or not math.isfinite(step) or step <= 0.0:
        return 0.0
    return math.ceil(value / step) * step


def _iter_polygons(geom: Any) -> list[Polygon]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return [poly for poly in geom.geoms if not poly.is_empty]
    if isinstance(geom, GeometryCollection):
        output: list[Polygon] = []
        for item in geom.geoms:
            output.extend(_iter_polygons(item))
        return output
    return []


def _safe_polygon(geom: Any, context: str) -> Polygon:
    if not isinstance(geom, Polygon):
        raise Pass3Error(f"{context}: expected Polygon, got {getattr(geom, 'geom_type', type(geom).__name__)}")
    poly = geom
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or not isinstance(poly, Polygon):
        raise Pass3Error(f"{context}: could not resolve one valid Polygon")
    return poly


def _open_ring(coords: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    ring = [(float(x), float(y)) for x, y in coords]
    if len(ring) >= 2 and ring[0] == ring[-1]:
        ring = ring[:-1]
    return ring


def _earcut_indices(poly: Polygon) -> tuple[np.ndarray, np.ndarray]:
    try:
        import mapbox_earcut as earcut
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise Pass3Error("Gate 3 needs mapbox-earcut in the normal Proto2 environment") from exc

    rings = [_open_ring(poly.exterior.coords)] + [_open_ring(ring.coords) for ring in poly.interiors]
    rings = [ring for ring in rings if len(ring) >= 3]
    if not rings:
        return np.empty((0, 2), dtype=np.float64), np.empty((0, 3), dtype=np.uint32)
    xy = np.asarray([point for ring in rings for point in ring], dtype=np.float64)
    ring_ends = np.cumsum([len(ring) for ring in rings], dtype=np.uint32)
    try:
        tri = earcut.triangulate_float64(xy, ring_ends)
    except TypeError:
        tri = earcut.triangulate_float64(xy.flatten(), ring_ends)
    tri = np.asarray(tri, dtype=np.uint32)
    return xy, tri.reshape((-1, 3))


def _metric_polygon_to_local(
    poly_metric: Polygon,
    *,
    to_wgs84: Transformer,
    cap_module: Any,
    center_lon: float,
    center_lat: float,
    static_height: float,
) -> Polygon:
    def transform_ring(coords: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
        values = list(coords)
        xs = np.asarray([float(x) for x, _ in values], dtype=np.float64)
        ys = np.asarray([float(y) for _, y in values], dtype=np.float64)
        lon, lat = to_wgs84.transform(xs, ys)
        h = np.full_like(np.asarray(lon, dtype=np.float64), static_height, dtype=np.float64)
        ecef_x, ecef_y, ecef_z = cap_module.wgs84_to_ecef(lon, lat, h)
        local_x, local_y, _local_z = cap_module.ecef_to_local_enu(
            ecef_x, ecef_y, ecef_z, center_lon, center_lat, static_height
        )
        return [(float(x), float(y)) for x, y in zip(local_x, local_y)]

    exterior = transform_ring(poly_metric.exterior.coords)
    holes = [transform_ring(ring.coords) for ring in poly_metric.interiors]
    local = Polygon(exterior, holes)
    if not local.is_valid:
        local = local.buffer(0)
    if local.is_empty or not isinstance(local, Polygon):
        raise Pass3Error("Metric-to-ENU transform did not preserve one Polygon")
    return local


def _ring_without_closure(coords: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    values = [(float(x), float(y)) for x, y in coords]
    if len(values) >= 2 and values[0] == values[-1]:
        values = values[:-1]
    return values


def _major_axis_angle_deg(poly: Polygon) -> float:
    rect = poly.minimum_rotated_rectangle
    coords = _ring_without_closure(rect.exterior.coords)
    if len(coords) < 2:
        return 0.0
    best_len = -1.0
    best_angle = 0.0
    for index, (x0, y0) in enumerate(coords):
        x1, y1 = coords[(index + 1) % len(coords)]
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        if length > best_len:
            best_len = length
            best_angle = math.degrees(math.atan2(dy, dx))
    while best_angle <= -90.0:
        best_angle += 180.0
    while best_angle > 90.0:
        best_angle -= 180.0
    return best_angle


def _square_from_point_metric(point_geom: Any, side_m: float, angle_deg: float) -> Polygon:
    from shapely import affinity
    from shapely.geometry import box

    half = 0.5 * float(side_m)
    square = box(
        float(point_geom.x) - half,
        float(point_geom.y) - half,
        float(point_geom.x) + half,
        float(point_geom.y) + half,
    )
    return affinity.rotate(square, angle_deg, origin=(float(point_geom.x), float(point_geom.y)), use_radians=False)


def _reconstruct_accepted_square(
    *,
    point_geom: Any,
    requested_side_m: float,
    angle_deg: float,
    source_part: Polygon,
    context: str,
) -> tuple[Polygon, float, float]:
    """Recreate one audited square without accepting clipping.

    The audit emits feature centres as WGS84 GeoJSON. On reconstruction the
    inverse CRS round-trip can move a boundary-touching centre by a few tiny
    fractions of a millimetre. First use the exact recorded centre and side.
    Then try a bounded centre recovery, keeping the full square intact. Only
    after that may the side be uniformly reduced by at most 5 mm. Anything
    outside those microscopic windows remains a hard failure.
    """
    requested = float(requested_side_m)
    if requested <= 0.0:
        raise Pass3Error(f"{context}: non-positive requested carrier side {requested}")

    base_x = float(point_geom.x)
    base_y = float(point_geom.y)

    def candidate(side: float, x: float = base_x, y: float = base_y) -> Polygon:
        return _square_from_point_metric(Point(x, y), side, angle_deg)

    def fits(square: Polygon) -> bool:
        return bool(source_part.covers(square.buffer(-RECONSTRUCTION_FIT_BUFFER_M)))

    full = candidate(requested)
    if fits(full):
        return full, 0.0, 0.0

    # Preserve the recorded full size whenever a CRS round-trip displaced the
    # centre by only a few millimetres. Test the source-part interior direction
    # first, then a small radial fan so a concave/irregular parcel still has a
    # deterministic opportunity to recover its accepted square.
    target = source_part.representative_point()
    tx = float(target.x) - base_x
    ty = float(target.y) - base_y
    tlen = math.hypot(tx, ty)
    directions: list[tuple[float, float]] = []
    if tlen > 1.0e-12:
        directions.append((tx / tlen, ty / tlen))
    for index in range(RECONSTRUCTION_NUDGE_DIRECTIONS):
        theta = 2.0 * math.pi * float(index) / float(RECONSTRUCTION_NUDGE_DIRECTIONS)
        directions.append((math.cos(theta), math.sin(theta)))
    # Keep ordering stable while removing an occasional duplicate interior ray.
    unique_directions: list[tuple[float, float]] = []
    for dx, dy in directions:
        if not any(abs(dx - ox) < 1.0e-12 and abs(dy - oy) < 1.0e-12 for ox, oy in unique_directions):
            unique_directions.append((dx, dy))
    for radius in (
        RECONSTRUCTION_MAX_CENTER_NUDGE_M * 0.25,
        RECONSTRUCTION_MAX_CENTER_NUDGE_M * 0.50,
        RECONSTRUCTION_MAX_CENTER_NUDGE_M,
    ):
        for dx, dy in unique_directions:
            shifted = candidate(requested, base_x + dx * radius, base_y + dy * radius)
            if fits(shifted):
                return shifted, 0.0, float(radius)

    lower_side = requested - RECONSTRUCTION_MAX_SIDE_SHRINK_M
    if lower_side < READABLE_MIN_BASE_M - 1.0e-9:
        raise Pass3Error(
            f"{context}: reconstructed {requested:.6f} m carrier misses after the allowed "
            f"{RECONSTRUCTION_MAX_CENTER_NUDGE_M * 1000.0:.1f} mm centre-recovery window; "
            f"shrinking would violate the {READABLE_MIN_BASE_M:.3f} m readable minimum"
        )
    lower = candidate(lower_side)
    if not fits(lower):
        raise Pass3Error(
            f"{context}: reconstructed carrier misses its source part by more than the allowed "
            f"{RECONSTRUCTION_MAX_SIDE_SHRINK_M * 1000.0:.1f} mm precision window"
        )

    lo = lower_side
    hi = requested
    for _ in range(RECONSTRUCTION_BINARY_STEPS):
        mid = 0.5 * (lo + hi)
        if fits(candidate(mid)):
            lo = mid
        else:
            hi = mid
    repaired = candidate(lo)
    if not fits(repaired):  # defensive: protects against an unexpected GEOS edge case
        raise Pass3Error(f"{context}: precision recovery failed its final fit check")
    return repaired, requested - lo, 0.0


# -----------------------------------------------------------------------------
# Gate inputs and static bound context
# -----------------------------------------------------------------------------

def _audit_paths(project_root: Path, family: LodFamily) -> tuple[Path, Path, Path]:
    audit_dir = stage_root(project_root) / family.audit_stage
    return (
        audit_dir / "uncertainty_layout_summary.json",
        audit_dir / "uncertainty_layout_part_audit.csv",
        audit_dir / "uncertainty_layout_feature_centres.geojson",
    )


def _load_family_audit(project_root: Path, family: LodFamily) -> tuple[dict[str, Any], pd.DataFrame, gpd.GeoDataFrame]:
    summary_path, part_path, points_path = _audit_paths(project_root, family)
    summary = read_json(require(summary_path, f"{family.label} layout summary"))
    schema = str(summary.get("schema") or "")
    if not schema.startswith("proto2_uncertainty_adaptive_layout_"):
        raise Pass3Error(f"{family.label}: required readable-row audit schema, got {schema!r}")
    settings = summary.get("settings") or {}
    spacing = float(settings.get("target_spacing_m", -1.0))
    if abs(spacing - family.spacing_m) > 1.0e-8:
        raise Pass3Error(f"{family.label}: expected {family.spacing_m:.0f} m, found {spacing:g} m")
    readable = float(settings.get("readable_min_feature_size_m", -1.0))
    if readable < READABLE_MIN_BASE_M - 1.0e-8:
        raise Pass3Error(f"{family.label}: readable minimum below {READABLE_MIN_BASE_M} m")
    if int(summary.get("parts_below_readable_base", -1)) != 0:
        raise Pass3Error(f"{family.label}: readable-carrier guard did not pass")

    part_df = pd.read_csv(require(part_path, f"{family.label} part audit"))
    required_part = {"parcel_id", "part_index", "feature_count", "below_readable_base"}
    missing_part = sorted(required_part - set(part_df.columns))
    if missing_part:
        raise Pass3Error(f"{family.label}: part audit missing columns: {missing_part}")
    if part_df["below_readable_base"].astype(bool).any():
        raise Pass3Error(f"{family.label}: part audit contains sub-readable feature(s)")

    points = gpd.read_file(require(points_path, f"{family.label} feature centres"))
    if points.crs is None:
        raise Pass3Error(f"{family.label}: feature centres have no CRS")
    points = points.to_crs(METRIC_CRS)
    if not points.empty:
        points["parcel_id"] = points["parcel_id"].astype(int)
        points["part_index"] = points["part_index"].astype(int)
        points["sign"] = points["sign"].astype(int)
        points["side_m"] = points["side_m"].astype(float)
        if float(points["side_m"].min()) < READABLE_MIN_BASE_M - 1.0e-8:
            raise Pass3Error(f"{family.label}: feature centre list contains a sub-readable carrier")
    return summary, part_df, points


def _load_moving_parts_and_rows(project_root: Path) -> tuple[gpd.GeoDataFrame, dict[int, int], pd.DataFrame]:
    config = load_project_config(project_root)
    output_data = output_data_dir(project_root, config)
    parts_path = require(output_data / "parcel_footprints_parts.parquet", "parcel footprint parts")
    vertices_path = require(output_data / "parcel_cap_mesh_vertices_indexed.parquet", "indexed cap vertices")

    parts = gpd.read_parquet(parts_path)
    if parts.crs is None:
        raise Pass3Error("parcel_footprints_parts.parquet has no CRS")
    required = {"parcel_id", "footprint_id", "part_index", "has_displacement", "geometry"}
    missing = sorted(required - set(parts.columns))
    if missing:
        raise Pass3Error(f"Parcel footprint parts missing: {missing}")
    parts = parts.loc[parts["has_displacement"].astype(bool)].copy().to_crs(METRIC_CRS)
    if parts.empty:
        raise Pass3Error("No moving footprint parts are available")
    parts["parcel_id"] = parts["parcel_id"].astype(int)
    parts["part_index"] = parts["part_index"].astype(int)
    parts["footprint_id"] = parts["footprint_id"].astype(str)

    vertices = pd.read_parquet(vertices_path, columns=["parcel_id", "has_displacement", "displacement_row_index", "lon", "lat"])
    needed = {"parcel_id", "has_displacement", "displacement_row_index", "lon", "lat"}
    missing_v = sorted(needed - set(vertices.columns))
    if missing_v:
        raise Pass3Error(f"Indexed cap vertices missing: {missing_v}")
    moving_vertices = vertices.loc[vertices["has_displacement"].astype(bool)].copy()
    rows: dict[int, int] = {}
    for parcel_id, group in moving_vertices.groupby(moving_vertices["parcel_id"].astype(int)):
        unique = sorted(group["displacement_row_index"].dropna().astype(int).unique().tolist())
        if len(unique) != 1:
            raise Pass3Error(f"Parcel {parcel_id} has unexpected displacement row mapping: {unique}")
        rows[int(parcel_id)] = int(unique[0])
    missing_rows = sorted(set(parts["parcel_id"].astype(int)) - set(rows))
    if missing_rows:
        raise Pass3Error(f"{len(missing_rows)} moving parcel(s) have no runtime row mapping")
    return parts, rows, vertices


def _load_animation_context(project_root: Path, piston_module: Any) -> dict[str, Any]:
    config = load_project_config(project_root)
    output_data = output_data_dir(project_root, config)
    manifest = read_json(require(output_data / "parcel_animation_manifest.json", "animation manifest"))
    shape = manifest.get("shape") or manifest.get("matrix_shape") or {}
    matrix_rows = int(shape.get("moving_parcels") or shape.get("rows") or shape.get("n_rows") or 0)
    epochs = int(shape.get("epochs") or shape.get("columns") or shape.get("n_epochs") or 0)
    if matrix_rows <= 0 or epochs <= 0:
        raise Pass3Error("Could not resolve animation matrix shape")
    reversible = _read_matrix(require(output_data / "parcel_displacement_reversible_f32.bin", "reversible runtime array"), matrix_rows, epochs)
    irreversible = _read_matrix(require(output_data / "parcel_displacement_irreversible_f32.bin", "irreversible runtime array"), matrix_rows, epochs)
    total = _read_matrix(require(output_data / "parcel_displacement_total_f32.bin", "MC Total runtime array"), matrix_rows, epochs)
    sigma_h = _read_matrix(require(output_data / "parcel_displacement_sigma_h_f32.bin", "sigma_h runtime array"), matrix_rows, epochs)

    tuning = piston_module.derive_display_tuning({
        "reversible": reversible,
        "irreversible": irreversible,
        "total": total,
    })
    return {
        "rows": matrix_rows,
        "epochs": epochs,
        "total_span_by_row": np.nanmax(total, axis=1).astype(np.float64) - np.nanmin(total, axis=1).astype(np.float64),
        "sigma_max_by_row": np.nanmax(sigma_h, axis=1).astype(np.float64),
        "height_scale_per_exag": float(tuning["height_scale_per_exag_unit_m_per_mm"]),
        "vertical_exag_max": int(round(float(tuning["vertical_exag_max"]))),
        "total_downward_mm": float(tuning["component_downward_mm"]["total"]),
        "min_datum_m": float(tuning["min_display_datum_height_m"]),
        "datum_step_m": float(tuning["datum_round_step_m"]),
        "safety_clearance_m": float(tuning["safety_clearance_m"]),
        "model_origin_height_m": float(tuning["display_datum_height_m"]),
    }


def _derive_static_bounds(animation: dict[str, Any], row_indices: Iterable[int]) -> dict[str, Any]:
    rows = np.asarray(sorted({int(row) for row in row_indices}), dtype=np.int64)
    if rows.size == 0:
        raise Pass3Error("Tile has no moving runtime rows")
    if int(rows.min()) < 0 or int(rows.max()) >= int(animation["rows"]):
        raise Pass3Error("Tile row index lies outside runtime matrix bounds")
    total_span = float(np.nanmax(np.asarray(animation["total_span_by_row"])[rows]))
    sigma_max = float(np.nanmax(np.asarray(animation["sigma_max_by_row"])[rows]))
    if not (math.isfinite(total_span) and math.isfinite(sigma_max) and sigma_max >= 0.0):
        raise Pass3Error("Tile has invalid total/sigma static-bound inputs")

    relief_max = sigma_max * UNCERTAINTY_BASE_M_PER_MM * UNCERTAINTY_SCALE_MAX
    z_min = math.inf
    z_max = -math.inf
    for exag in range(int(animation["vertical_exag_max"]) + 1):
        scale = float(exag) * float(animation["height_scale_per_exag"])
        datum = max(
            float(animation["min_datum_m"]),
            _ceil_to_step(float(animation["total_downward_mm"]) * scale + float(animation["safety_clearance_m"]), float(animation["datum_step_m"])),
        )
        local_datum = datum - float(animation["model_origin_height_m"])
        z_min = min(z_min, local_datum - total_span * scale - relief_max)
        z_max = max(z_max, local_datum + total_span * scale + relief_max)
    if not math.isfinite(z_min) or not math.isfinite(z_max) or z_min >= z_max:
        raise Pass3Error("Could not derive static shader bounds for tile")
    return {
        "method": "degenerate_static_bound_sentinels_v1",
        "total_reference_span_mm_tile_max": total_span,
        "sigma_max_mm_tile_max": sigma_max,
        "uncertainty_base_m_per_mm": UNCERTAINTY_BASE_M_PER_MM,
        "uncertainty_scale_max": UNCERTAINTY_SCALE_MAX,
        "relief_amplitude_max_m": relief_max,
        "padding_m": STATIC_BOUND_PADDING_M,
        "static_z_min_m": float(math.floor(z_min - STATIC_BOUND_PADDING_M)),
        "static_z_max_m": float(math.ceil(z_max + STATIC_BOUND_PADDING_M)),
    }


# -----------------------------------------------------------------------------
# Tile and family builders
# -----------------------------------------------------------------------------

def _feature_lookup(points: gpd.GeoDataFrame) -> dict[tuple[int, int], list[FeaturePoint]]:
    result: dict[tuple[int, int], list[FeaturePoint]] = defaultdict(list)
    for item in points.itertuples():
        side = float(item.side_m)
        if side < READABLE_MIN_BASE_M - 1.0e-8:
            raise Pass3Error(f"Feature below readable minimum: parcel={item.parcel_id}, side={side}")
        key = (int(item.parcel_id), int(item.part_index))
        result[key].append(FeaturePoint(
            sign=1 if int(item.sign) >= 0 else -1,
            side_m=side,
            point_metric=item.geometry,
        ))
    return result


def _assign_tiles(parts: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, dict[str, dict[str, Any]]]:
    minx, miny, maxx, maxy = [float(v) for v in parts.total_bounds]
    span_x = max(maxx - minx, 1.0)
    span_y = max(maxy - miny, 1.0)
    work = parts.copy()
    tile_ids: list[str] = []
    tile_x: list[int] = []
    tile_y: list[int] = []
    for geom in work.geometry:
        p = geom.representative_point()
        ix = min(TILE_COLUMNS - 1, max(0, int(math.floor((float(p.x) - minx) / span_x * TILE_COLUMNS))))
        iy = min(TILE_ROWS - 1, max(0, int(math.floor((float(p.y) - miny) / span_y * TILE_ROWS))))
        tile_x.append(ix)
        tile_y.append(iy)
        tile_ids.append(f"x{ix}_y{iy}")
    work["_uncertainty_tile_id"] = tile_ids
    work["_uncertainty_tile_x"] = tile_x
    work["_uncertainty_tile_y"] = tile_y

    infos: dict[str, dict[str, Any]] = {}
    for iy in range(TILE_ROWS):
        for ix in range(TILE_COLUMNS):
            tile_id = f"x{ix}_y{iy}"
            x0 = minx + span_x * ix / TILE_COLUMNS
            x1 = minx + span_x * (ix + 1) / TILE_COLUMNS
            y0 = miny + span_y * iy / TILE_ROWS
            y1 = miny + span_y * (iy + 1) / TILE_ROWS
            infos[tile_id] = {
                "tile_id": tile_id,
                "tile_x": ix,
                "tile_y": iy,
                "metric_bbox": [x0, y0, x1, y1],
            }
    return work, infos


def _build_one_tile(
    *,
    tile_parts: gpd.GeoDataFrame,
    tile_info: dict[str, Any],
    family: LodFamily,
    output_path: Path,
    features_by_part: dict[tuple[int, int], list[FeaturePoint]],
    rows_by_parcel: dict[int, int],
    animation: dict[str, Any],
    cap_module: Any,
    piston_module: Any,
    to_wgs84: Transformer,
    center_lon: float,
    center_lat: float,
    static_height: float,
) -> dict[str, Any]:
    positions: list[list[float]] = []
    tex0: list[list[float]] = []
    tex1: list[list[float]] = []
    indices: list[int] = []
    counts = TileCounts()
    row_indices: set[int] = set()

    def push_vertex(x: float, y: float, relief: float, row_index: int) -> int:
        vertex_index = len(positions)
        # Matches the main cap model orientation: [north, -east, up].
        positions.append([float(y), float(-x), float(relief)])
        tex0.append([float(row_index), 1.0])
        tex1.append([float(relief), 0.0])
        return vertex_index

    def add_triangle(a: tuple[float, float, float], b: tuple[float, float, float], c: tuple[float, float, float], row_index: int) -> None:
        indices.extend([push_vertex(*a, row_index), push_vertex(*b, row_index), push_vertex(*c, row_index)])

    def add_flat(local_geom: Any, row_index: int) -> int:
        triangle_count = 0
        for poly in _iter_polygons(local_geom):
            xy, tri = _earcut_indices(poly)
            for i0, i1, i2 in tri:
                a, b, c = xy[int(i0)], xy[int(i1)], xy[int(i2)]
                add_triangle(
                    (float(a[0]), float(a[1]), 0.0),
                    (float(b[0]), float(b[1]), 0.0),
                    (float(c[0]), float(c[1]), 0.0),
                    row_index,
                )
                triangle_count += 1
        return triangle_count

    def add_pyramid(local_square: Polygon, relief: float, row_index: int) -> None:
        coords = _open_ring(local_square.exterior.coords)
        if len(coords) != 4:
            raise Pass3Error("Uncertainty carrier must remain a four-sided square")
        cx, cy = float(local_square.centroid.x), float(local_square.centroid.y)
        apex = (cx, cy, float(relief))
        for index in range(4):
            a = coords[index]
            b = coords[(index + 1) % 4]
            if relief >= 0.0:
                add_triangle((a[0], a[1], 0.0), (b[0], b[1], 0.0), apex, row_index)
            else:
                add_triangle((b[0], b[1], 0.0), (a[0], a[1], 0.0), apex, row_index)

    for part in tile_parts.itertuples():
        parcel_id = int(part.parcel_id)
        part_index = int(part.part_index)
        row_index = int(rows_by_parcel[parcel_id])
        row_indices.add(row_index)
        counts.part_count += 1
        part_poly_metric = _safe_polygon(part.geometry, f"tile {tile_info['tile_id']} parcel {parcel_id} part {part_index}")
        angle = _major_axis_angle_deg(part_poly_metric)
        source_features = features_by_part.get((parcel_id, part_index), [])

        feature_specs: list[tuple[Polygon, int, float]] = []
        for feature in source_features:
            square, side_shrink_m, center_nudge_m = _reconstruct_accepted_square(
                point_geom=feature.point_metric,
                requested_side_m=feature.side_m,
                angle_deg=angle,
                source_part=part_poly_metric,
                context=(
                    f"{family.label} carrier parcel={parcel_id}, part={part_index}, "
                    f"requested_side={feature.side_m:.6f} m"
                ),
            )
            if side_shrink_m > 0.0 or center_nudge_m > 0.0:
                counts.precision_recovered_carriers += 1
            if side_shrink_m > 0.0:
                counts.max_precision_side_shrink_m = max(counts.max_precision_side_shrink_m, float(side_shrink_m))
            if center_nudge_m > 0.0:
                counts.precision_nudged_carriers += 1
                counts.max_precision_center_nudge_m = max(counts.max_precision_center_nudge_m, float(center_nudge_m))
            feature_specs.append((square, feature.sign, feature.side_m))

        if not feature_specs:
            counts.fallback_flat_parts += 1

        down_squares = [square for square, sign, _ in feature_specs if sign < 0]
        dimple_holes = unary_union(down_squares) if down_squares else None
        flat_metric = part_poly_metric.difference(dimple_holes) if dimple_holes is not None else part_poly_metric
        flat_triangles = 0
        for flat_piece in _iter_polygons(flat_metric):
            local_flat = _metric_polygon_to_local(
                flat_piece,
                to_wgs84=to_wgs84,
                cap_module=cap_module,
                center_lon=center_lon,
                center_lat=center_lat,
                static_height=static_height,
            )
            flat_triangles += add_flat(local_flat, row_index)
        counts.flat_triangles += flat_triangles
        if flat_triangles == 0:
            counts.nonrenderable_flat_parts += 1

        for square_metric, sign, _side in feature_specs:
            local_square = _metric_polygon_to_local(
                square_metric,
                to_wgs84=to_wgs84,
                cap_module=cap_module,
                center_lon=center_lon,
                center_lat=center_lat,
                static_height=static_height,
            )
            add_pyramid(local_square, float(sign), row_index)
            if sign > 0:
                counts.up_features += 1
            else:
                counts.down_features += 1

    if not positions or not indices:
        raise Pass3Error(f"Tile {tile_info['tile_id']} in {family.label} produced no renderable geometry")

    visible_triangles = len(indices) // 3
    static_bounds = _derive_static_bounds(animation, row_indices)
    base_positions = np.asarray(positions, dtype=np.float64)
    sentinel_x = float(-np.mean(base_positions[:, 1]))
    sentinel_y = float(np.mean(base_positions[:, 0]))
    sentinel_row = int(min(row_indices))
    for z in (float(static_bounds["static_z_min_m"]), float(static_bounds["static_z_max_m"])):
        sentinel_indices: list[int] = []
        for _ in range(3):
            sentinel_indices.append(push_vertex(sentinel_x, sentinel_y, z, sentinel_row))
            tex1[-1] = [0.0, 0.0]
        indices.extend(sentinel_indices)

    positions_arr = np.asarray(positions, dtype="<f4")
    tex0_arr = np.asarray(tex0, dtype="<f4")
    tex1_arr = np.asarray(tex1, dtype="<f4")
    colors_arr = np.tile(np.asarray([[255, 255, 255, 255]], dtype=np.uint8), (len(positions_arr), 1))
    indices_arr = np.asarray(indices, dtype="<u4")
    if float(positions_arr[:, 2].min()) > float(static_bounds["static_z_min_m"]) or float(positions_arr[:, 2].max()) < float(static_bounds["static_z_max_m"]):
        raise Pass3Error(f"Tile {tile_info['tile_id']}: static-bound sentinels did not survive POSITION construction")

    atomic_write_bytes(output_path, piston_module.build_glb(positions_arr, colors_arr, tex0_arr, tex1_arr, indices_arr))
    return {
        **tile_info,
        "spacing_m": family.spacing_m,
        "lod_key": family.key,
        "part_count": counts.part_count,
        "fallback_flat_parts": counts.fallback_flat_parts,
        "nonrenderable_flat_parts": counts.nonrenderable_flat_parts,
        "up_features": counts.up_features,
        "down_features": counts.down_features,
        "feature_count": counts.up_features + counts.down_features,
        "flat_triangles": counts.flat_triangles,
        "precision_recovered_carriers": counts.precision_recovered_carriers,
        "max_precision_side_shrink_m": float(counts.max_precision_side_shrink_m),
        "precision_nudged_carriers": counts.precision_nudged_carriers,
        "max_precision_center_nudge_m": float(counts.max_precision_center_nudge_m),
        "visible_triangles": int(visible_triangles),
        "sentinel_triangles": 2,
        "vertices": int(len(positions_arr)),
        "triangles": int(len(indices_arr) // 3),
        "indices": int(len(indices_arr)),
        "runtime_rows": int(len(row_indices)),
        "static_bounds": static_bounds,
        "glb_bytes": int(output_path.stat().st_size),
        "relative_glb_path": output_path.name,
    }


def _build_family(
    *,
    project_root: Path,
    family: LodFamily,
    parts: gpd.GeoDataFrame,
    tile_infos: dict[str, dict[str, Any]],
    rows_by_parcel: dict[int, int],
    animation: dict[str, Any],
    cap_module: Any,
    piston_module: Any,
    to_wgs84: Transformer,
    center_lon: float,
    center_lat: float,
    static_height: float,
    output_root: Path,
) -> dict[str, Any]:
    summary, part_audit, feature_points = _load_family_audit(project_root, family)
    _ = part_audit  # validated; manifest provenance stays in summary below.
    features_by_part = _feature_lookup(feature_points)

    family_dir = output_root / family.output_subdir
    if family_dir.exists():
        shutil.rmtree(family_dir)
    family_dir.mkdir(parents=True, exist_ok=True)

    tile_records: list[dict[str, Any]] = []
    ordered_ids = [f"x{ix}_y{iy}" for iy in range(TILE_ROWS) for ix in range(TILE_COLUMNS)]
    for ordinal, tile_id in enumerate(ordered_ids, start=1):
        tile_parts = parts.loc[parts["_uncertainty_tile_id"] == tile_id].copy()
        if tile_parts.empty:
            continue
        feature_count = sum(len(features_by_part.get((int(row.parcel_id), int(row.part_index)), [])) for row in tile_parts.itertuples())
        print(
            f"[{family.key} {ordinal:02d}/{len(ordered_ids):02d}] {tile_id}: "
            f"parts={len(tile_parts):,}, carriers={feature_count:,}",
            flush=True,
        )
        glb_path = family_dir / f"uncertainty_{family.key}_{tile_id}.glb"
        record = _build_one_tile(
            tile_parts=tile_parts,
            tile_info=tile_infos[tile_id],
            family=family,
            output_path=glb_path,
            features_by_part=features_by_part,
            rows_by_parcel=rows_by_parcel,
            animation=animation,
            cap_module=cap_module,
            piston_module=piston_module,
            to_wgs84=to_wgs84,
            center_lon=center_lon,
            center_lat=center_lat,
            static_height=static_height,
        )
        record["url"] = f"_internal/data_pipeline/runtime/geometry/uncertainty_lod/{family.output_subdir}/{glb_path.name}"
        tile_records.append(record)

    if not tile_records:
        raise Pass3Error(f"{family.label}: no non-empty tiles were built")
    totals = {
        "tile_count": len(tile_records),
        "parts": sum(int(item["part_count"]) for item in tile_records),
        "fallback_flat_parts": sum(int(item["fallback_flat_parts"]) for item in tile_records),
        "nonrenderable_flat_parts": sum(int(item["nonrenderable_flat_parts"]) for item in tile_records),
        "features": sum(int(item["feature_count"]) for item in tile_records),
        "flat_triangles": sum(int(item["flat_triangles"]) for item in tile_records),
        "precision_recovered_carriers": sum(int(item.get("precision_recovered_carriers", 0)) for item in tile_records),
        "max_precision_side_shrink_m": max((float(item.get("max_precision_side_shrink_m", 0.0)) for item in tile_records), default=0.0),
        "precision_nudged_carriers": sum(int(item.get("precision_nudged_carriers", 0)) for item in tile_records),
        "max_precision_center_nudge_m": max((float(item.get("max_precision_center_nudge_m", 0.0)) for item in tile_records), default=0.0),
        "visible_triangles": sum(int(item["visible_triangles"]) for item in tile_records),
        "triangles": sum(int(item["triangles"]) for item in tile_records),
        "vertices": sum(int(item["vertices"]) for item in tile_records),
        "bytes": sum(int(item["glb_bytes"]) for item in tile_records),
    }
    return {
        "key": family.key,
        "label": family.label,
        "spacing_m": family.spacing_m,
        "audit": {
            "delivery_id": summary.get("delivery_id"),
            "script_id": summary.get("script_id"),
            "schema": summary.get("schema"),
            "stage": family.audit_stage,
            "parts_below_readable_base": summary.get("parts_below_readable_base"),
        },
        "totals": totals,
        "tiles": tile_records,
    }


# -----------------------------------------------------------------------------
# Runtime and main
# -----------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Build full-field 20 m detail + 40 m overview uncertainty LOD GLB tile families.")
    parser.add_argument("--tiles-x", type=int, default=TILE_COLUMNS, help="Fixed tile-column count (must remain 4 for this delivery).")
    parser.add_argument("--tiles-y", type=int, default=TILE_ROWS, help="Fixed tile-row count (must remain 4 for this delivery).")
    args = parser.parse_args()
    if int(args.tiles_x) != TILE_COLUMNS or int(args.tiles_y) != TILE_ROWS:
        raise Pass3Error("The production uncertainty build is fixed to a 4 × 4 spatial tile grid")

    project_root = project_root_from(__file__)
    cap_module = cap_support
    piston_module = piston_support
    parts, rows_by_parcel, vertices = _load_moving_parts_and_rows(project_root)
    parts, tile_infos = _assign_tiles(parts)
    animation = _load_animation_context(project_root, piston_module)
    to_wgs84 = Transformer.from_crs(METRIC_CRS, "EPSG:4326", always_xy=True)
    center_lon = 0.5 * (float(vertices["lon"].min()) + float(vertices["lon"].max()))
    center_lat = 0.5 * (float(vertices["lat"].min()) + float(vertices["lat"].max()))
    static_height = float(getattr(cap_module, "STATIC_HEIGHT_OFFSET_M", 4.0))

    output_root = stage_root(project_root) / "geometry" / "uncertainty_lod"
    output_root.mkdir(parents=True, exist_ok=True)

    print("\n=== PROTO2 STAGE 08: BUILD UNCERTAINTY LOD RUNTIME GEOMETRY ===", flush=True)
    print(f"Delivery ID  : {DELIVERY_ID}", flush=True)
    print(f"Script ID    : {SCRIPT_ID}", flush=True)
    print(f"Project root : {project_root}", flush=True)
    print(f"Moving parts : {len(parts):,}", flush=True)
    print(f"Tile grid    : {TILE_COLUMNS} × {TILE_ROWS}", flush=True)
    print("Families     : detail 20 m + overview 40 m", flush=True)
    print("Normal caps  : untouched by builder; the final viewer hides moving Total caps only when the field layer is ready.", flush=True)

    family_results: dict[str, Any] = {}
    for family in FAMILIES:
        print(f"\n--- Building {family.label} ---", flush=True)
        family_results[family.key] = _build_family(
            project_root=project_root,
            family=family,
            parts=parts,
            tile_infos=tile_infos,
            rows_by_parcel=rows_by_parcel,
            animation=animation,
            cap_module=cap_module,
            piston_module=piston_module,
            to_wgs84=to_wgs84,
            center_lon=center_lon,
            center_lat=center_lat,
            static_height=static_height,
            output_root=output_root,
        )

    manifest: dict[str, Any] = {
        "delivery_id": DELIVERY_ID,
        "script_id": SCRIPT_ID,
        "schema": SCHEMA,
        "scope": "production_full_field_uncertainty_lod_assets_only_no_viewer_file_modified",
        "tile_grid": {"columns": TILE_COLUMNS, "rows": TILE_ROWS},
        "model_origin": {
            "center_lon": center_lon,
            "center_lat": center_lat,
            "static_height_m": static_height,
        },
        "layout_contract": {
            "detail_spacing_m": DETAIL_SPACING_M,
            "overview_spacing_m": OVERVIEW_SPACING_M,
            "readable_min_feature_base_m": READABLE_MIN_BASE_M,
            "up_base_m": 3.5,
            "down_base_m": 4.5,
            "uncertainty_base_m_per_mm": UNCERTAINTY_BASE_M_PER_MM,
            "uncertainty_slider_max": UNCERTAINTY_SCALE_MAX,
            "lod_switching": "exclusive: the viewer shows detail OR overview, never both families together",
        },
        "viewer_lod_config": {
            "enabled": True,
            "thresholds": {
                "detail_enter_height_m": 10000.0,
                "overview_enter_height_m": 13000.0,
                "default_lod": "overview",
                "note": "Hysteresis: keep the current family between 10 km and 13 km camera height.",
            },
            "configuration_transport": "embedded_in_assembled_viewer_bootstrap",
        },
        "lod_families": family_results,
        "notes": [
            "Every moving part is assigned to exactly one spatial tile by its representative point; parts are never cut at tile borders.",
            "Each tile retains the coloured flat remainder and uses true dimple holes for downward carriers.",
            "Fallback-flat parts retain flat geometry where triangulable. Degenerate sliver fragments may produce zero visible triangles and are reported per family.",
            "Static bound sentinels are invisible zero-area triangles used only for Cesium culling/frustum stability.",
            "Carrier reconstruction uses an exact fit first. A centre that shifted during the WGS84 GeoJSON round trip may be nudged inward by at most 5 mm while keeping its full square. Only then may the square be uniformly reduced by at most 5 mm; every recovery is counted in the manifest. Larger discrepancies fail the build.",
            "This is the production uncertainty tile builder used by the normal numbered pipeline.",
        ],
    }
    manifest_path = output_root / "uncertainty_lod_manifest.json"
    manifest["outputs"] = {
        "manifest": {"path": "_internal/data_pipeline/runtime/geometry/uncertainty_lod/uncertainty_lod_manifest.json"},
        "detail_tiles": [file_record(output_root / FAMILIES[0].output_subdir / Path(tile["url"]).name, project_root) for tile in family_results["detail"]["tiles"]],
        "overview_tiles": [file_record(output_root / FAMILIES[1].output_subdir / Path(tile["url"]).name, project_root) for tile in family_results["overview"]["tiles"]],
    }
    write_json(manifest_path, manifest)

    for key, family in family_results.items():
        totals = family["totals"]
        print(
            f"{key:8s}: tiles={totals['tile_count']}, features={totals['features']:,}, "
            f"visible triangles={totals['visible_triangles']:,}, GLBs={totals['bytes'] / (1024 * 1024):.1f} MiB, "
            f"precision recoveries={totals.get('precision_recovered_carriers', 0):,} "
            f"(nudged={totals.get('precision_nudged_carriers', 0):,}, "
            f"max nudge={totals.get('max_precision_center_nudge_m', 0.0) * 1000.0:.3f} mm, "
            f"max shrink={totals.get('max_precision_side_shrink_m', 0.0) * 1000.0:.3f} mm)",
            flush=True,
        )
    print_pass("STAGE 08 RESULT", manifest_path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Pass3Error as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        raise SystemExit(1)
