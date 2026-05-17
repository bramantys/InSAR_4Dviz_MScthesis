#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_build_footprints.py

Generic RUM-based InSAR template step.

Purpose
-------
Build corrected WGS84 RUM cell footprints from point centers.

Input:
  - points GeoJSON from config.prepared_inputs.points_geojson
  - base/prepared vertical epoch JSON from config.prepared_inputs.vertical_epoch_json_without_enhanced_sigma
    or config.prepared_inputs.vertical_epoch_json

Output:
  - config.generated_outputs.rum_footprints

Important
---------
This script infers the RUM grid axis and spacing from the point cloud itself.
It should therefore work for Jakarta, Groningen, and other RUM-based InSAR
products, provided the input points are already in WGS84 and have stable rum_id.

The epoch JSON is only used to attach source_up_mm_yr to footprints when
available. Footprint geometry is inferred from the point cloud.
"""

from __future__ import annotations

import json
import math
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
# CONFIG HELPERS
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


def safe_float(value: Any, fallback: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return fallback
        out = float(value)
        if not math.isfinite(out):
            return fallback
        return out
    except (TypeError, ValueError):
        return fallback


def median(values: List[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


# =============================================================================
# GEODESY
# =============================================================================

def wgs84_to_ecef(lon_deg: float, lat_deg: float, h_m: float = 0.0) -> np.ndarray:
    """WGS84 geodetic → ECEF Cartesian3 [m]."""
    a = 6_378_137.0
    f = 1.0 / 298.257223563
    e2 = 2 * f - f * f

    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)

    n = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)

    x = (n + h_m) * cos_lat * math.cos(lon)
    y = (n + h_m) * cos_lat * math.sin(lon)
    z = (n * (1.0 - e2) + h_m) * sin_lat
    return np.array([x, y, z], dtype=np.float64)


def build_enu_frame(ecef_origin: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Build local ENU frame at ecef_origin."""
    x, y, z = ecef_origin
    lon = math.atan2(y, x)
    lat = math.atan2(z, math.sqrt(x * x + y * y))

    east = np.array([-math.sin(lon), math.cos(lon), 0.0], dtype=np.float64)
    north = np.array([
        -math.sin(lat) * math.cos(lon),
        -math.sin(lat) * math.sin(lon),
         math.cos(lat),
    ], dtype=np.float64)
    up = np.array([
        math.cos(lat) * math.cos(lon),
        math.cos(lat) * math.sin(lon),
        math.sin(lat),
    ], dtype=np.float64)

    frame = np.eye(4, dtype=np.float64)
    frame[0:3, 0] = east
    frame[0:3, 1] = north
    frame[0:3, 2] = up
    frame[0:3, 3] = ecef_origin

    frame_inv = np.linalg.inv(frame)
    return frame, frame_inv


def ecef_to_enu(ecef_pt: np.ndarray, frame_inv: np.ndarray) -> Tuple[float, float, float]:
    h = np.array([ecef_pt[0], ecef_pt[1], ecef_pt[2], 1.0], dtype=np.float64)
    local = frame_inv @ h
    return float(local[0]), float(local[1]), float(local[2])


def enu_to_ecef(east: float, north: float, up: float, frame: np.ndarray) -> np.ndarray:
    h = np.array([east, north, up, 1.0], dtype=np.float64)
    world = frame @ h
    return world[0:3]


def ecef_to_wgs84(ecef: np.ndarray) -> Tuple[float, float, float]:
    """ECEF → WGS84 lon/lat/h. Iterative latitude solve."""
    x, y, z = ecef

    a = 6_378_137.0
    f = 1.0 / 298.257223563
    e2 = 2 * f - f * f

    lon = math.atan2(y, x)
    p = math.sqrt(x * x + y * y)
    lat = math.atan2(z, p * (1.0 - e2))

    for _ in range(5):
        sin_lat = math.sin(lat)
        n = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
        lat = math.atan2(z + e2 * n * sin_lat, p)

    sin_lat = math.sin(lat)
    n = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)

    if abs(math.cos(lat)) > 1e-10:
        h = p / math.cos(lat) - n
    else:
        h = abs(z) / max(abs(sin_lat), 1e-10) - n * (1.0 - e2)

    return math.degrees(lon), math.degrees(lat), float(h)


# =============================================================================
# GRID INFERENCE
# =============================================================================

def rotate_enu_to_grid(east: float, north: float, axis_angle_rad: float) -> Tuple[float, float]:
    c = math.cos(axis_angle_rad)
    s = math.sin(axis_angle_rad)
    return east * c + north * s, -east * s + north * c


def rotate_grid_to_enu(u: float, v: float, axis_angle_rad: float) -> Tuple[float, float]:
    c = math.cos(axis_angle_rad)
    s = math.sin(axis_angle_rad)
    return u * c - v * s, u * s + v * c


def wrap_to_period(value: float, period: float) -> float:
    wrapped = value % period
    if wrapped < 0:
        wrapped += period
    return wrapped


def estimate_periodic_offset(values: List[float], spacing: float) -> float:
    if not values or spacing <= 0:
        return 0.0

    sum_x = 0.0
    sum_y = 0.0

    for value in values:
        wrapped = wrap_to_period(value, spacing)
        angle = (2.0 * math.pi * wrapped) / spacing
        sum_x += math.cos(angle)
        sum_y += math.sin(angle)

    mean_angle = math.atan2(sum_y, sum_x)
    if mean_angle < 0:
        mean_angle += 2.0 * math.pi

    return (mean_angle / (2.0 * math.pi)) * spacing


def estimate_grid_axis_angle(neighbor_vectors: List[Tuple[float, float]]) -> float:
    """
    Estimate dominant square-grid axis angle using the 4θ circular mean trick.
    """
    if not neighbor_vectors:
        return 0.0

    sum_x = 0.0
    sum_y = 0.0

    for dx, dy in neighbor_vectors:
        theta = math.atan2(dy, dx)
        sum_x += math.cos(4.0 * theta)
        sum_y += math.sin(4.0 * theta)

    return 0.25 * math.atan2(sum_y, sum_x)


def build_rum_grid_model(
    enu_points: List[Tuple[float, float]],
    nominal_spacing_m: float,
    search_neighbors: int,
    axis_tolerance_deg: float,
    spacing_ratio_min: float,
    spacing_ratio_max: float,
) -> Dict[str, float]:
    n_points = len(enu_points)
    if n_points < 4:
        raise ValueError("Need at least 4 points to infer a RUM grid")

    axis_tol_tan = math.tan(math.radians(axis_tolerance_deg))

    nearest_distances: List[float] = []
    neighbor_vectors: List[Tuple[float, float]] = []

    for i in range(n_points):
        ei, ni = enu_points[i]
        candidates: List[Tuple[float, float, float]] = []

        for j in range(n_points):
            if i == j:
                continue
            dx = enu_points[j][0] - ei
            dy = enu_points[j][1] - ni
            d = math.hypot(dx, dy)
            if d > 0:
                candidates.append((d, dx, dy))

        candidates.sort(key=lambda row: row[0])

        if candidates:
            nearest_distances.append(candidates[0][0])

        for k in range(min(search_neighbors, len(candidates))):
            neighbor_vectors.append((candidates[k][1], candidates[k][2]))

    nn_spacing = median(nearest_distances) or nominal_spacing_m

    short_vectors = [
        (dx, dy)
        for dx, dy in neighbor_vectors
        if math.hypot(dx, dy) <= nn_spacing * 1.8
    ]
    vectors_for_axis = short_vectors if short_vectors else neighbor_vectors

    axis_angle_rad = estimate_grid_axis_angle(vectors_for_axis)

    spacing_u_candidates: List[float] = []
    spacing_v_candidates: List[float] = []

    for dx, dy in vectors_for_axis:
        u, v = rotate_enu_to_grid(dx, dy, axis_angle_rad)
        au = abs(u)
        av = abs(v)

        if (
            nn_spacing * spacing_ratio_min <= au <= nn_spacing * spacing_ratio_max
            and av <= au * axis_tol_tan
        ):
            spacing_u_candidates.append(au)

        if (
            nn_spacing * spacing_ratio_min <= av <= nn_spacing * spacing_ratio_max
            and au <= av * axis_tol_tan
        ):
            spacing_v_candidates.append(av)

    spacing_u = median(spacing_u_candidates) or nn_spacing
    spacing_v = median(spacing_v_candidates) or nn_spacing

    uv_points = [rotate_enu_to_grid(e, n, axis_angle_rad) for e, n in enu_points]

    offset_u = estimate_periodic_offset([uv[0] for uv in uv_points], spacing_u)
    offset_v = estimate_periodic_offset([uv[1] for uv in uv_points], spacing_v)

    return {
        "axis_angle_rad": axis_angle_rad,
        "axis_angle_deg": math.degrees(axis_angle_rad),
        "spacing_u": spacing_u,
        "spacing_v": spacing_v,
        "offset_u": offset_u,
        "offset_v": offset_v,
        "nearest_neighbor_spacing_m": nn_spacing,
        "nominal_spacing_m": nominal_spacing_m,
    }


def compute_grid_indices(east: float, north: float, grid_model: Dict[str, float]) -> Tuple[int, int]:
    u, v = rotate_enu_to_grid(east, north, grid_model["axis_angle_rad"])
    i = round((u - grid_model["offset_u"]) / grid_model["spacing_u"])
    j = round((v - grid_model["offset_v"]) / grid_model["spacing_v"])
    return int(i), int(j)


def compute_corners_from_grid(
    i: int,
    j: int,
    grid_model: Dict[str, float],
    frame: np.ndarray,
    height_m: float = 0.0,
) -> List[List[float]]:
    m = grid_model

    left = m["offset_u"] + (i - 0.5) * m["spacing_u"]
    right = m["offset_u"] + (i + 0.5) * m["spacing_u"]
    bottom = m["offset_v"] + (j - 0.5) * m["spacing_v"]
    top = m["offset_v"] + (j + 0.5) * m["spacing_v"]

    ne_e, ne_n = rotate_grid_to_enu(right, top, m["axis_angle_rad"])
    nw_e, nw_n = rotate_grid_to_enu(left, top, m["axis_angle_rad"])
    sw_e, sw_n = rotate_grid_to_enu(left, bottom, m["axis_angle_rad"])
    se_e, se_n = rotate_grid_to_enu(right, bottom, m["axis_angle_rad"])

    corners_enu = [(ne_e, ne_n), (nw_e, nw_n), (sw_e, sw_n), (se_e, se_n)]

    corners_wgs84: List[List[float]] = []
    for ce, cn in corners_enu:
        ecef = enu_to_ecef(ce, cn, height_m, frame)
        lon, lat, _ = ecef_to_wgs84(ecef)
        corners_wgs84.append([lon, lat])

    return corners_wgs84


# =============================================================================
# INPUT LOADERS
# =============================================================================

def load_geojson(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if data.get("type") != "FeatureCollection":
        raise ValueError(f"Expected GeoJSON FeatureCollection, got: {data.get('type')}")

    return data


def load_epoch_series(path: Path) -> Dict[str, Any]:
    if not path.exists():
        warn(f"Epoch JSON not found; source_up_mm_yr will be read from GeoJSON when possible: {path}")
        return {}

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    series = data.get("series", {})
    if not isinstance(series, dict):
        warn(f"Epoch JSON has no dict-like 'series': {path}")
        return {}

    return series


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    r = 6_378_137.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2
    )
    return 2.0 * r * math.asin(math.sqrt(a))


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    cfg = load_config()

    prepared = cfg.get("prepared_inputs", {})
    generated = cfg.get("generated_outputs", {})
    expected = cfg.get("expected_counts", {})

    points_path = resolve_project_path(
        prepared.get("points_geojson", "Data/points_wgs84_with_rumid.geojson")
    )

    # Footprints should be based on the base linear epoch file at this stage.
    epoch_json_candidates = [
        prepared.get("vertical_epoch_json_without_enhanced_sigma"),
        prepared.get("vertical_epoch_json"),
    ]
    epoch_path = None
    for candidate in epoch_json_candidates:
        if candidate:
            p = resolve_project_path(candidate)
            if p.exists():
                epoch_path = p
                break
    if epoch_path is None and epoch_json_candidates[0]:
        epoch_path = resolve_project_path(epoch_json_candidates[0])

    output_path = resolve_project_path(
        generated.get("rum_footprints", "Data/rum_footprints.json")
    )

    nominal_spacing = float(expected.get("grid_spacing_m_nominal", 450.0))
    search_neighbors = int(expected.get("grid_axis_search_neighbors", 8))
    axis_tolerance_deg = float(expected.get("grid_axis_tolerance_deg", 15.0))
    spacing_ratio_min = float(expected.get("grid_spacing_ratio_min", 0.45))
    spacing_ratio_max = float(expected.get("grid_spacing_ratio_max", 1.60))

    section("Configuration")
    print(f"  Project root      : {PROJECT_DIR}")
    print(f"  Points GeoJSON    : {points_path}")
    print(f"  Epoch JSON        : {epoch_path}")
    print(f"  Output footprints : {output_path}")
    print(f"  Nominal spacing   : {nominal_spacing:.3f} m")

    section("Loading inputs")
    if not points_path.exists():
        raise FileNotFoundError(f"Missing points GeoJSON: {points_path}")

    geojson = load_geojson(points_path)
    features = geojson.get("features", [])
    ok(f"Loaded point features: {len(features)}")

    epoch_series = load_epoch_series(epoch_path) if epoch_path is not None else {}
    if epoch_series:
        ok(f"Loaded epoch series: {len(epoch_series)} RUMs")

    section("Extracting valid RUM centers")

    rum_data: List[Dict[str, Any]] = []
    skipped = 0

    for feature in features:
        props = feature.get("properties") or {}
        geom = feature.get("geometry") or {}
        coords = geom.get("coordinates") or []

        rum_id = props.get("rum_id")
        if not rum_id or len(coords) < 2:
            skipped += 1
            continue

        lon = safe_float(coords[0])
        lat = safe_float(coords[1])

        if lon is None or lat is None:
            skipped += 1
            continue

        rum_data.append({
            "rum_id": str(rum_id),
            "lon": lon,
            "lat": lat,
            "x_rum": props.get("x_rum"),
            "y_rum": props.get("y_rum"),
            "source_up_geojson": safe_float(props.get("up")),
        })

    ok(f"Valid RUM centers: {len(rum_data)}")
    if skipped:
        warn(f"Skipped invalid/missing center features: {skipped}")

    if len(rum_data) < 4:
        raise ValueError("Too few valid RUM centers to build footprints")

    section("Building shared ENU frame")

    centroid_lon = sum(r["lon"] for r in rum_data) / len(rum_data)
    centroid_lat = sum(r["lat"] for r in rum_data) / len(rum_data)

    origin_ecef = wgs84_to_ecef(centroid_lon, centroid_lat, 0.0)
    frame, frame_inv = build_enu_frame(origin_ecef)

    ok(f"Centroid lon={centroid_lon:.6f}, lat={centroid_lat:.6f}")

    section("Converting centers to local ENU")

    t0 = time.time()
    for r in rum_data:
        ecef = wgs84_to_ecef(r["lon"], r["lat"], 0.0)
        east, north, _ = ecef_to_enu(ecef, frame_inv)
        r["east"] = east
        r["north"] = north
    ok(f"Converted {len(rum_data)} centers in {time.time() - t0:.2f}s")

    section("Inferring RUM grid model")

    t0 = time.time()
    enu_points = [(float(r["east"]), float(r["north"])) for r in rum_data]
    grid_model = build_rum_grid_model(
        enu_points=enu_points,
        nominal_spacing_m=nominal_spacing,
        search_neighbors=search_neighbors,
        axis_tolerance_deg=axis_tolerance_deg,
        spacing_ratio_min=spacing_ratio_min,
        spacing_ratio_max=spacing_ratio_max,
    )

    ok(f"Grid axis angle      : {grid_model['axis_angle_deg']:.6f}°")
    ok(f"Nearest-neighbor dist: {grid_model['nearest_neighbor_spacing_m']:.3f} m")
    ok(f"Spacing U            : {grid_model['spacing_u']:.3f} m")
    ok(f"Spacing V            : {grid_model['spacing_v']:.3f} m")
    ok(f"Offset U             : {grid_model['offset_u']:.3f} m")
    ok(f"Offset V             : {grid_model['offset_v']:.3f} m")
    ok(f"Inference time       : {time.time() - t0:.2f}s")

    for name, val in [("U", grid_model["spacing_u"]), ("V", grid_model["spacing_v"])]:
        diff = abs(val - nominal_spacing)
        if diff <= max(30.0, nominal_spacing * 0.10):
            ok(f"Spacing {name} close to nominal ({diff:.2f} m difference)")
        else:
            warn(f"Spacing {name} differs from nominal by {diff:.2f} m")

    section("Computing grid-aligned footprints")

    t0 = time.time()

    footprints: Dict[str, Any] = {}
    occupied: Dict[Tuple[int, int], str] = {}
    collisions = 0

    for r in rum_data:
        rum_id = r["rum_id"]
        i, j = compute_grid_indices(float(r["east"]), float(r["north"]), grid_model)

        key = (i, j)
        if key in occupied:
            collisions += 1
        occupied[key] = rum_id

        corners = compute_corners_from_grid(i, j, grid_model, frame, height_m=0.0)

        source_up = None
        if rum_id in epoch_series:
            source_up = safe_float(epoch_series[rum_id].get("source_up_mm_yr"))
        if source_up is None:
            source_up = r.get("source_up_geojson")

        footprints[rum_id] = {
            "center": [r["lon"], r["lat"]],
            "corners": corners,
            "grid_i": i,
            "grid_j": j,
            "source_up_mm_yr": source_up,
            "x_rum": r.get("x_rum"),
            "y_rum": r.get("y_rum"),
        }

    ok(f"Computed footprints: {len(footprints)} in {time.time() - t0:.2f}s")
    if collisions:
        warn(f"Grid collisions detected: {collisions}")
    else:
        ok("Zero grid collisions")

    section("Geometry spot check")

    expected_diag = math.sqrt(grid_model["spacing_u"] ** 2 + grid_model["spacing_v"] ** 2)
    sample_ids = sorted(footprints.keys())[:5]

    for rid in sample_ids:
        fp = footprints[rid]
        ne, nw, sw, se = fp["corners"]
        diag = haversine_m(ne[0], ne[1], sw[0], sw[1])
        diff = abs(diag - expected_diag)
        status = "OK" if diff < max(10.0, expected_diag * 0.03) else "WARN"
        print(
            f"  [{status}] {rid:<24s} "
            f"grid=({fp['grid_i']},{fp['grid_j']}) "
            f"diag={diag:.2f} m expected={expected_diag:.2f} m diff={diff:.2f} m"
        )

    section("Coverage bounding box")

    all_center_lons = [fp["center"][0] for fp in footprints.values()]
    all_center_lats = [fp["center"][1] for fp in footprints.values()]
    all_corner_lons = []
    all_corner_lats = []

    for fp in footprints.values():
        for lon, lat in fp["corners"]:
            all_corner_lons.append(lon)
            all_corner_lats.append(lat)

    center_lon_min = min(all_center_lons)
    center_lon_max = max(all_center_lons)
    center_lat_min = min(all_center_lats)
    center_lat_max = max(all_center_lats)

    bbox = {
        "lon_min": min(all_corner_lons),
        "lon_max": max(all_corner_lons),
        "lat_min": min(all_corner_lats),
        "lat_max": max(all_corner_lats),
    }

    print(f"  Center lon: {center_lon_min:.6f} → {center_lon_max:.6f}")
    print(f"  Center lat: {center_lat_min:.6f} → {center_lat_max:.6f}")
    print(f"  Corner lon: {bbox['lon_min']:.6f} → {bbox['lon_max']:.6f}")
    print(f"  Corner lat: {bbox['lat_min']:.6f} → {bbox['lat_max']:.6f}")

    section("Writing output")

    output = {
        "metadata": {
            "schema_version": "footprints_v1",
            "rum_count": len(footprints),
            "grid_model": grid_model,
            "corner_order": "NE, NW, SW, SE",
            "coordinate_system": "WGS84 lon/lat degrees",
            "source_points_geojson": str(points_path),
            "source_epoch_json": str(epoch_path) if epoch_path else None,
            "created_by": "03_build_footprints.py",
            "notes": [
                "Grid axis and cell footprints are inferred from WGS84 point centers converted to a local ENU frame.",
                "Footprints use corrected inferred grid geometry, not raw point-to-point polygon geometry."
            ],
        },
        "bbox": bbox,
        "footprints": footprints,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, separators=(",", ":"))

    ok(f"Written: {output_path} ({output_path.stat().st_size / 1024 / 1024:.2f} MB)")

    section("SUMMARY")
    ok("Step 03 complete — corrected RUM footprints created")
    ok("Next template step: 04_enhance_vertical_sigma_optional.py")
    print()


if __name__ == "__main__":
    main()
