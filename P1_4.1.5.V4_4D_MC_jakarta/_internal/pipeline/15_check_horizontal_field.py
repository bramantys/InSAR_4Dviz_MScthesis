#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
15_check_horizontal_field.py

InSAR4D RUM Viewer pipeline step 15.

Purpose
-------
Validate and summarize the horizontal velocity field created by Step 14.

Input
-----
  generated_outputs.horizontal_field
    _internal/data_pipeline/horizontal_field.json

Output
------
  generated_outputs.horizontal_debug_vectors
    _internal/data_pipeline/horizontal_debug_vectors.geojson

What this step checks
---------------------
- horizontal_field.json exists and has records
- row_index values are unique and mostly continuous
- lon/lat centers are valid
- east/north/up/speed values are finite
- speed matches sqrt(east^2 + north^2)
- unit_east/unit_north is valid for nonzero speeds
- azimuth is finite and in [0, 360)
- covariance fields are present/finite when marked available
- var_east and var_north are non-negative
- horizontal covariance determinant is non-negative:
    det = var_east * var_north - covar_en^2

Important note
--------------
This step does NOT build confidence ellipses.

The confidence ellipse calculation is rebuilt later in Step 16/17. We do not
copy old pipeline 19 scaling because the previous version may have had a /100
or unit conversion mistake. This step only flags suspicious covariance scale.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pipeline_config import load_resolved_config, resolve_path


# =============================================================================
# ADVANCED SETTINGS
# =============================================================================

ROUND_COORD_DIGITS = 8
ROUND_VECTOR_DIGITS = 6
ROUND_SUMMARY_DIGITS = 6

# Speed consistency tolerance in mm/yr.
SPEED_ABS_TOL = 1e-5

# Unit vector tolerance.
UNIT_VECTOR_ABS_TOL = 5e-4

# Very small horizontal speeds make direction/unit-vector checks unstable,
# especially after Step 14 rounds east/north/speed/unit values for JSON.
# These records are still valid, but their direction is visually weak.
UNIT_VECTOR_VALIDATE_MIN_SPEED_MM_YR = 0.05

# Heuristic covariance-scale warnings only; never fail on these alone.
# If the velocity covariance is in (mm/yr)^2, sqrt(var) is std in mm/yr.
SUSPICIOUS_STD_P50_TINY_MM_YR = 0.001
SUSPICIOUS_STD_P50_HUGE_MM_YR = 1000.0

# Debug vector GeoJSON sampling.
# Use stride to keep file light. Top fastest vectors are always included.
DEBUG_GRID_STRIDE = 3
INCLUDE_TOP_N_SPEEDS = 50
MAX_DEBUG_VECTOR_LENGTH_M = 450.0
SPEED_FOR_MAX_VECTOR_P98 = True


# =============================================================================
# PRINT HELPERS
# =============================================================================

WARNINGS: List[str] = []


def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def ok(msg: str) -> None:
    print(f"  [OK]   {msg}")


def warn(msg: str) -> None:
    WARNINGS.append(msg)
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
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))


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


def percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * p / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    return xs[f] * (c - k) + xs[c] * (k - f)


def minmax(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    return min(values), max(values)


def approx_equal(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


def meters_per_degree_lat() -> float:
    return 111320.0


def meters_per_degree_lon(lat_deg: float) -> float:
    return 111320.0 * math.cos(math.radians(lat_deg))


def vector_endpoint_lonlat(
    lon: float,
    lat: float,
    east_mm_yr: float,
    north_mm_yr: float,
    speed_scale_mm_yr: float,
) -> Tuple[float, float]:
    """
    Create a short debug vector endpoint in lon/lat.

    This is not used for final arrows. It only creates a lightweight diagnostic
    GeoJSON. Final arrows are rebuilt as B3DM in Step 17.
    """
    speed = math.hypot(east_mm_yr, north_mm_yr)
    if speed <= 0 or speed_scale_mm_yr <= 0:
        return lon, lat

    scale = min(1.0, speed / speed_scale_mm_yr)
    length_m = scale * MAX_DEBUG_VECTOR_LENGTH_M

    ue = east_mm_yr / speed
    un = north_mm_yr / speed

    d_east_m = ue * length_m
    d_north_m = un * length_m

    m_per_lon = meters_per_degree_lon(lat)
    m_per_lat = meters_per_degree_lat()

    if abs(m_per_lon) < 1e-9:
        return lon, lat

    lon2 = lon + d_east_m / m_per_lon
    lat2 = lat + d_north_m / m_per_lat

    return round(lon2, ROUND_COORD_DIGITS), round(lat2, ROUND_COORD_DIGITS)


# =============================================================================
# VALIDATION
# =============================================================================

def validate_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        raise ValueError("horizontal_field.json has no records")

    row_indices: List[int] = []
    speeds: List[float] = []
    east_values: List[float] = []
    north_values: List[float] = []
    up_values: List[float] = []
    std_east_values: List[float] = []
    std_north_values: List[float] = []
    det_values: List[float] = []

    invalid_lonlat = 0
    invalid_vector = 0
    speed_mismatch = 0
    unit_mismatch = 0
    weak_direction_count = 0
    invalid_azimuth = 0
    duplicate_row_indices = 0
    covariance_available = 0
    covariance_missing_when_available = 0
    negative_variance = 0
    invalid_covariance_det = 0
    nonfinite_covariance = 0

    seen_rows = set()

    for idx, rec in enumerate(records):
        row_index = safe_int(rec.get("row_index"))
        if row_index is None:
            raise ValueError(f"Record {idx} missing row_index")
        if row_index in seen_rows:
            duplicate_row_indices += 1
        seen_rows.add(row_index)
        row_indices.append(row_index)

        lon = safe_float(rec.get("lon_center"))
        lat = safe_float(rec.get("lat_center"))
        if lon is None or lat is None or not (-180 <= lon <= 180) or not (-90 <= lat <= 90):
            invalid_lonlat += 1

        east = safe_float(rec.get("east_mm_yr"))
        north = safe_float(rec.get("north_mm_yr"))
        up = safe_float(rec.get("up_mm_yr"))
        speed = safe_float(rec.get("speed_mm_yr"))
        ux = safe_float(rec.get("unit_east"))
        uy = safe_float(rec.get("unit_north"))
        az = safe_float(rec.get("azimuth_deg_clockwise_from_north"))

        if east is None or north is None or up is None or speed is None or ux is None or uy is None:
            invalid_vector += 1
            continue

        expected_speed = math.hypot(east, north)
        if not approx_equal(speed, expected_speed, SPEED_ABS_TOL):
            speed_mismatch += 1

        if speed > UNIT_VECTOR_VALIDATE_MIN_SPEED_MM_YR:
            if not approx_equal(ux, east / speed, UNIT_VECTOR_ABS_TOL):
                unit_mismatch += 1
            if not approx_equal(uy, north / speed, UNIT_VECTOR_ABS_TOL):
                unit_mismatch += 1
        else:
            # Direction is mathematically valid but visually/diagnostically weak.
            # Do not hard-fail near-zero vectors; they are common in stable areas.
            weak_direction_count += 1

        if az is None or not (0.0 <= az < 360.0):
            invalid_azimuth += 1

        speeds.append(speed)
        east_values.append(east)
        north_values.append(north)
        up_values.append(up)

        cov = rec.get("covariance") or {}
        if cov.get("available"):
            covariance_available += 1

            var_e = safe_float(cov.get("var_east"))
            var_n = safe_float(cov.get("var_north"))
            cov_en = safe_float(cov.get("covar_en"))

            if var_e is None or var_n is None or cov_en is None:
                covariance_missing_when_available += 1
                continue

            if var_e < 0 or var_n < 0:
                negative_variance += 1
                continue

            det = var_e * var_n - cov_en * cov_en

            if not math.isfinite(det):
                nonfinite_covariance += 1
                continue

            if det < -1e-9:
                invalid_covariance_det += 1

            det_values.append(det)
            std_east_values.append(math.sqrt(max(0.0, var_e)))
            std_north_values.append(math.sqrt(max(0.0, var_n)))

    if duplicate_row_indices:
        raise ValueError(f"Duplicate row_index values in horizontal field: {duplicate_row_indices}")
    if invalid_lonlat:
        raise ValueError(f"Invalid lon/lat centers in horizontal field: {invalid_lonlat}")
    if invalid_vector:
        raise ValueError(f"Invalid/nonfinite velocity vector records: {invalid_vector}")
    if speed_mismatch:
        raise ValueError(f"speed_mm_yr mismatch with sqrt(east^2+north^2): {speed_mismatch}")
    if unit_mismatch:
        raise ValueError(
            f"unit_east/unit_north mismatch for strong-enough vectors: {unit_mismatch}. "
            f"Near-zero vectors below {UNIT_VECTOR_VALIDATE_MIN_SPEED_MM_YR} mm/yr are ignored."
        )

    if weak_direction_count:
        warn(
            f"Near-zero horizontal vectors skipped in unit-vector validation: {weak_direction_count} "
            f"(speed <= {UNIT_VECTOR_VALIDATE_MIN_SPEED_MM_YR} mm/yr)"
        )
    if invalid_azimuth:
        raise ValueError(f"Invalid azimuth values: {invalid_azimuth}")

    if covariance_missing_when_available:
        raise ValueError(f"Covariance marked available but fields missing: {covariance_missing_when_available}")
    if negative_variance:
        raise ValueError(f"Negative horizontal variances found: {negative_variance}")
    if nonfinite_covariance:
        raise ValueError(f"Nonfinite covariance determinant found: {nonfinite_covariance}")
    if invalid_covariance_det:
        raise ValueError(
            f"Invalid horizontal covariance matrices with negative determinant: {invalid_covariance_det}. "
            "Check var/covar units and source columns."
        )

    # Heuristic covariance scale warnings. These are exactly where old /100 or
    # unit mistakes might reveal themselves. Do not fail automatically.
    std_p50_e = percentile(std_east_values, 50)
    std_p50_n = percentile(std_north_values, 50)

    if std_p50_e is not None and std_p50_n is not None:
        std_p50 = (std_p50_e + std_p50_n) / 2.0
        if std_p50 < SUSPICIOUS_STD_P50_TINY_MM_YR:
            warn(
                "Horizontal covariance standard deviation median is extremely tiny "
                f"({std_p50:.6g} mm/yr). Check for accidental /100 or unit scaling."
            )
        elif std_p50 > SUSPICIOUS_STD_P50_HUGE_MM_YR:
            warn(
                "Horizontal covariance standard deviation median is extremely large "
                f"({std_p50:.6g} mm/yr). Check covariance units."
            )

    row_min, row_max = minmax(row_indices)
    expected_continuous_count = int(row_max - row_min + 1) if row_min is not None and row_max is not None else None
    row_continuous = expected_continuous_count == len(row_indices)

    if not row_continuous:
        warn(
            f"row_index values are not continuous: min={row_min}, max={row_max}, "
            f"count={len(row_indices)}"
        )
    else:
        ok("row_index values are unique and continuous")

    summary = {
        "record_count": len(records),
        "row_index_min": row_min,
        "row_index_max": row_max,
        "row_index_continuous": row_continuous,
        "east_min_mm_yr": min(east_values),
        "east_p50_mm_yr": percentile(east_values, 50),
        "east_max_mm_yr": max(east_values),
        "north_min_mm_yr": min(north_values),
        "north_p50_mm_yr": percentile(north_values, 50),
        "north_max_mm_yr": max(north_values),
        "up_min_mm_yr": min(up_values),
        "up_p50_mm_yr": percentile(up_values, 50),
        "up_max_mm_yr": max(up_values),
        "speed_min_mm_yr": min(speeds),
        "speed_p02_mm_yr": percentile(speeds, 2),
        "speed_p50_mm_yr": percentile(speeds, 50),
        "speed_p98_mm_yr": percentile(speeds, 98),
        "speed_max_mm_yr": max(speeds),
        "weak_direction_count": weak_direction_count,
        "unit_vector_validation_min_speed_mm_yr": UNIT_VECTOR_VALIDATE_MIN_SPEED_MM_YR,
        "covariance_available_count": covariance_available,
        "covariance_available_fraction": covariance_available / len(records) if records else 0.0,
        "std_east_p50_mm_yr": std_p50_e,
        "std_north_p50_mm_yr": std_p50_n,
        "det_min": min(det_values) if det_values else None,
        "det_p50": percentile(det_values, 50) if det_values else None,
        "det_max": max(det_values) if det_values else None,
    }

    return summary


def build_debug_vectors(records: List[Dict[str, Any]], speed_scale: float) -> Dict[str, Any]:
    top_rows = {
        int(rec["row_index"])
        for rec in sorted(records, key=lambda r: float(r.get("speed_mm_yr", 0.0)), reverse=True)[:INCLUDE_TOP_N_SPEEDS]
    }

    features: List[Dict[str, Any]] = []

    for rec in records:
        row_index = int(rec["row_index"])
        gi = safe_int(rec.get("grid_i"), 0) or 0
        gj = safe_int(rec.get("grid_j"), 0) or 0

        include_by_grid = (gi % DEBUG_GRID_STRIDE == 0 and gj % DEBUG_GRID_STRIDE == 0)
        include_by_top_speed = row_index in top_rows

        if not include_by_grid and not include_by_top_speed:
            continue

        lon = float(rec["lon_center"])
        lat = float(rec["lat_center"])
        east = float(rec["east_mm_yr"])
        north = float(rec["north_mm_yr"])

        lon2, lat2 = vector_endpoint_lonlat(lon, lat, east, north, speed_scale)

        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [round(lon, ROUND_COORD_DIGITS), round(lat, ROUND_COORD_DIGITS)],
                    [lon2, lat2],
                ],
            },
            "properties": {
                "rum_id": rec["rum_id"],
                "row_index": row_index,
                "grid_i": rec.get("grid_i"),
                "grid_j": rec.get("grid_j"),
                "east_mm_yr": rec.get("east_mm_yr"),
                "north_mm_yr": rec.get("north_mm_yr"),
                "speed_mm_yr": rec.get("speed_mm_yr"),
                "azimuth_deg_clockwise_from_north": rec.get("azimuth_deg_clockwise_from_north"),
                "debug_include_reason": "top_speed" if include_by_top_speed else "grid_stride",
            },
        })

    return {
        "type": "FeatureCollection",
        "metadata": {
            "schema": "horizontal_debug_vectors_v1",
            "purpose": "diagnostic_only_not_final_arrows",
            "debug_grid_stride": DEBUG_GRID_STRIDE,
            "include_top_n_speeds": INCLUDE_TOP_N_SPEEDS,
            "max_debug_vector_length_m": MAX_DEBUG_VECTOR_LENGTH_M,
            "speed_scale_mm_yr": speed_scale,
            "feature_count": len(features),
        },
        "features": features,
    }


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    t_start = time.time()

    cfg = load_resolved_config(__file__)
    project_root = Path(cfg["_resolved"]["project_root"])

    generated = cfg["generated_outputs"]

    input_path = resolve_path(project_root, generated["horizontal_field"])
    output_path = resolve_path(project_root, generated["horizontal_debug_vectors"])

    section("Configuration")
    print(f"  Project root          : {project_root}")
    print(f"  Horizontal input      : {input_path}")
    print(f"  Debug vector output   : {output_path}")

    section("Loading horizontal field")
    hfield = load_json(input_path)

    records = hfield.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("horizontal_field.json has no records")

    ok(f"Loaded horizontal field: {len(records)} records")

    section("Validating horizontal field")
    summary = validate_records(records)

    ok("Horizontal vector values are finite and internally consistent")
    ok(
        f"Covariance available for {summary['covariance_available_count']} / "
        f"{summary['record_count']} records"
    )

    section("Writing debug vectors")
    speed_scale = summary["speed_p98_mm_yr"] if SPEED_FOR_MAX_VECTOR_P98 else summary["speed_max_mm_yr"]
    if speed_scale is None or speed_scale <= 0:
        speed_scale = 1.0
        warn("Speed scale was zero/invalid; using 1.0 mm/yr for debug vectors")

    debug_vectors = build_debug_vectors(records, float(speed_scale))
    write_json(output_path, debug_vectors)

    ok(f"Wrote debug vectors: {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")
    ok(f"Debug vector count: {len(debug_vectors['features'])}")

    elapsed = time.time() - t_start

    section("Summary")
    ok(f"Step 15 complete in {elapsed:.2f} s")
    print(f"  Records                : {summary['record_count']}")
    print(f"  Speed range            : {summary['speed_min_mm_yr']:.6f} to {summary['speed_max_mm_yr']:.6f} mm/yr")
    print(f"  Speed p50/p98          : {summary['speed_p50_mm_yr']:.6f} / {summary['speed_p98_mm_yr']:.6f} mm/yr")
    print(f"  Covariance available   : {summary['covariance_available_count']} / {summary['record_count']}")
    print(f"  Weak direction skipped : {summary['weak_direction_count']}")
    print(f"  Std east/north p50     : {summary['std_east_p50_mm_yr']} / {summary['std_north_p50_mm_yr']} mm/yr")
    print(f"  Warnings               : {len(WARNINGS)}")


if __name__ == "__main__":
    main()
