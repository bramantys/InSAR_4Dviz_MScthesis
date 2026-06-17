#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
14_build_horizontal_field.py

InSAR4D RUM Viewer pipeline step 14.

Purpose
-------
Build the static horizontal velocity field aligned to corrected RUM footprints.

Inputs
------
  generated_outputs.rum_footprints
    _internal/data_pipeline/rum_footprints.json

  generated_outputs.packed_series
    _internal/data_pipeline/packed_series.json

Outputs
-------
  generated_outputs.horizontal_field
    _internal/data_pipeline/horizontal_field.json

  generated_outputs.horizontal_particle_field
    _internal/data_pipeline/horizontal_particle_field.json

Why this step exists
--------------------
The horizontal velocity field should use the corrected footprint centers from
Step 03, not approximate/old center locations.

The full horizontal_field product feeds:
  - dev arrows
  - confidence ellipses
  - uncertainty checks

The lightweight horizontal_particle_field product feeds:
  - canvas horizontal particles only

Main contract
-------------
Each record includes:
  rum_id
  row_index
  grid_i/grid_j
  x_center/y_center
  lon_center/lat_center
  east_mm_yr
  north_mm_yr
  up_mm_yr
  speed_mm_yr
  azimuth_deg_clockwise_from_north
  covariance fields when available
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
ROUND_SOURCE_XY_DIGITS = 4
ROUND_VECTOR_DIGITS = 6
ROUND_ANGLE_DIGITS = 3

# Speeds below this are still kept, but unit vector and azimuth are considered
# visually weak/unstable.
NEAR_ZERO_SPEED_MM_YR = 1e-9


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
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))


def safe_float(value: Any, fallback: Optional[float] = 0.0) -> Optional[float]:
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


def azimuth_from_east_north(east: float, north: float) -> float:
    """
    Return azimuth in degrees clockwise from north.
      northward = 0
      eastward  = 90
      southward = 180
      westward  = 270
    """
    angle = math.degrees(math.atan2(east, north))
    return (angle + 360.0) % 360.0


def unit_vector(east: float, north: float, speed: float) -> Tuple[float, float]:
    if speed <= NEAR_ZERO_SPEED_MM_YR:
        return 0.0, 0.0
    return east / speed, north / speed


# =============================================================================
# FIELD BUILDING
# =============================================================================

def get_schema_fields(cfg: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, str]]:
    source_schema = cfg.get("source_schema", {})
    source_inputs = cfg.get("source_inputs", {})

    vel = source_schema.get("velocity_fields") or source_inputs.get("source_velocity_fields") or {}
    unc = source_schema.get("uncertainty_fields") or source_inputs.get("source_variance_fields") or {}

    return vel, unc


def build_horizontal_records(
    footprints: Dict[str, Any],
    packed: Dict[str, Any],
    velocity_fields: Dict[str, str],
    uncertainty_fields: Dict[str, str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    features = footprints.get("features", [])
    if not features:
        raise ValueError("rum_footprints.json contains no features")

    rum_index = packed.get("rum_index") or {}
    if not rum_index:
        raise ValueError("packed_series.json missing rum_index")

    east_col = velocity_fields.get("east", "east")
    north_col = velocity_fields.get("north", "north")
    up_col = velocity_fields.get("up", "up")

    var_east_col = uncertainty_fields.get("var_east", "var_east")
    var_north_col = uncertainty_fields.get("var_north", "var_north")
    var_up_col = uncertainty_fields.get("var_up", "var_up")
    covar_en_col = uncertainty_fields.get("covar_en", "covar_en")
    covar_eu_col = uncertainty_fields.get("covar_eu", "covar_eu")
    covar_nu_col = uncertainty_fields.get("covar_nu", "covar_nu")

    records: List[Dict[str, Any]] = []
    missing_velocity = 0
    missing_row_index = 0
    covariance_available = 0

    speeds: List[float] = []
    east_values: List[float] = []
    north_values: List[float] = []
    up_values: List[float] = []

    for idx, feature in enumerate(features):
        props = feature.get("properties") or {}

        rum_id = str(props.get("rum_id", f"RUM_{idx + 1:06d}"))

        if rum_id not in rum_index:
            missing_row_index += 1
            continue

        east = safe_float(props.get(east_col), None)
        north = safe_float(props.get(north_col), None)
        up = safe_float(props.get(up_col), None)

        if east is None or north is None:
            missing_velocity += 1
            east = 0.0 if east is None else east
            north = 0.0 if north is None else north

        if up is None:
            up = 0.0

        speed = math.hypot(east, north)
        ux, uy = unit_vector(east, north, speed)
        azimuth = azimuth_from_east_north(east, north)

        var_east = safe_float(props.get(var_east_col), None)
        var_north = safe_float(props.get(var_north_col), None)
        var_up = safe_float(props.get(var_up_col), None)
        covar_en = safe_float(props.get(covar_en_col), None)
        covar_eu = safe_float(props.get(covar_eu_col), None)
        covar_nu = safe_float(props.get(covar_nu_col), None)

        has_covariance = (
            var_east is not None
            and var_north is not None
            and covar_en is not None
        )
        if has_covariance:
            covariance_available += 1

        rec = {
            "rum_id": rum_id,
            "row_index": int(rum_index[rum_id]),
            "source_row": safe_int(props.get("source_row"), idx),
            "grid_i": safe_int(props.get("grid_i")),
            "grid_j": safe_int(props.get("grid_j")),
            "x_center": round(float(safe_float(props.get("x_center"), 0.0)), ROUND_SOURCE_XY_DIGITS),
            "y_center": round(float(safe_float(props.get("y_center"), 0.0)), ROUND_SOURCE_XY_DIGITS),
            "lon_center": round(float(safe_float(props.get("lon_center"), 0.0)), ROUND_COORD_DIGITS),
            "lat_center": round(float(safe_float(props.get("lat_center"), 0.0)), ROUND_COORD_DIGITS),
            "east_mm_yr": round(float(east), ROUND_VECTOR_DIGITS),
            "north_mm_yr": round(float(north), ROUND_VECTOR_DIGITS),
            "up_mm_yr": round(float(up), ROUND_VECTOR_DIGITS),
            "speed_mm_yr": round(float(speed), ROUND_VECTOR_DIGITS),
            "unit_east": round(float(ux), ROUND_VECTOR_DIGITS),
            "unit_north": round(float(uy), ROUND_VECTOR_DIGITS),
            "azimuth_deg_clockwise_from_north": round(float(azimuth), ROUND_ANGLE_DIGITS),
            "covariance": {
                "var_east": var_east,
                "var_north": var_north,
                "var_up": var_up,
                "covar_en": covar_en,
                "covar_eu": covar_eu,
                "covar_nu": covar_nu,
                "available": bool(has_covariance),
            },
        }

        records.append(rec)
        speeds.append(speed)
        east_values.append(east)
        north_values.append(north)
        up_values.append(up)

    if missing_row_index:
        raise ValueError(f"Footprints missing packed row_index: {missing_row_index}")

    records.sort(key=lambda r: r["row_index"])

    summary = {
        "record_count": len(records),
        "missing_velocity_count": missing_velocity,
        "covariance_available_count": covariance_available,
        "covariance_available_fraction": covariance_available / len(records) if records else 0.0,
        "east_min_mm_yr": min(east_values) if east_values else None,
        "east_max_mm_yr": max(east_values) if east_values else None,
        "north_min_mm_yr": min(north_values) if north_values else None,
        "north_max_mm_yr": max(north_values) if north_values else None,
        "up_min_mm_yr": min(up_values) if up_values else None,
        "up_max_mm_yr": max(up_values) if up_values else None,
        "speed_min_mm_yr": min(speeds) if speeds else None,
        "speed_p02_mm_yr": percentile(speeds, 2),
        "speed_p50_mm_yr": percentile(speeds, 50),
        "speed_p98_mm_yr": percentile(speeds, 98),
        "speed_max_mm_yr": max(speeds) if speeds else None,
    }

    return records, summary


def particle_field_output_path(project_root: Path, generated: Dict[str, Any], full_output_path: Path) -> Path:
    """Return configured particle-field path, or a stable sibling fallback."""
    configured = generated.get("horizontal_particle_field")
    if configured:
        return resolve_path(project_root, configured)
    return full_output_path.with_name("horizontal_particle_field.json")


def build_particle_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build the lightweight particle-only field.

    Important separation:
      - horizontal_field.json remains the rich diagnostic/product for checks,
        arrows, ellipses, colour-scale derivation, and documentation.
      - horizontal_particle_field.json is only for canvas particle advection.

    The particle renderer does not need rum_id or popup metadata anymore because
    real/blank RUM popup selection now comes from pickable B3DM cap batch tables.
    """
    particle_records: List[Dict[str, Any]] = []

    for rec in records:
        cov = rec.get("covariance") or {}
        particle_records.append({
            "grid_i": rec.get("grid_i"),
            "grid_j": rec.get("grid_j"),
            "lon": rec.get("lon_center"),
            "lat": rec.get("lat_center"),
            "height_row": rec.get("row_index"),
            "east_mm_yr": rec.get("east_mm_yr"),
            "north_mm_yr": rec.get("north_mm_yr"),
            "speed_mm_yr": rec.get("speed_mm_yr"),
            "var_east": cov.get("var_east"),
            "var_north": cov.get("var_north"),
            "covar_en": cov.get("covar_en"),
        })

    return particle_records


def build_particle_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    speeds = [
        float(r["speed_mm_yr"])
        for r in records
        if safe_float(r.get("speed_mm_yr"), None) is not None
    ]
    return {
        "record_count": len(records),
        "speed_min_mm_yr": min(speeds) if speeds else None,
        "speed_p50_mm_yr": percentile(speeds, 50),
        "speed_p75_mm_yr": percentile(speeds, 75),
        "speed_p95_mm_yr": percentile(speeds, 95),
        "speed_p995_mm_yr": percentile(speeds, 99.5),
        "speed_max_mm_yr": max(speeds) if speeds else None,
        "speed_mm_yr": {
            "p50": percentile(speeds, 50),
            "p75": percentile(speeds, 75),
            "p95": percentile(speeds, 95),
            "p995": percentile(speeds, 99.5),
        },
    }


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    t_start = time.time()

    cfg = load_resolved_config(__file__)
    project_root = Path(cfg["_resolved"]["project_root"])

    generated = cfg["generated_outputs"]

    footprints_path = resolve_path(project_root, generated["rum_footprints"])
    packed_path = resolve_path(project_root, generated["packed_series"])
    output_path = resolve_path(project_root, generated["horizontal_field"])
    particle_output_path = particle_field_output_path(project_root, generated, output_path)

    velocity_fields, uncertainty_fields = get_schema_fields(cfg)

    section("Configuration")
    print(f"  Project root          : {project_root}")
    print(f"  Footprints input      : {footprints_path}")
    print(f"  Packed input          : {packed_path}")
    print(f"  Horizontal output     : {output_path}")
    print(f"  Particle output       : {particle_output_path}")
    print(f"  East/North/Up fields  : {velocity_fields.get('east', 'east')}, {velocity_fields.get('north', 'north')}, {velocity_fields.get('up', 'up')}")

    section("Loading inputs")
    footprints = load_json(footprints_path)
    packed = load_json(packed_path)

    ok(f"Loaded footprints: {len(footprints.get('features', []))} features")
    ok(f"Loaded packed row index: {len(packed.get('rum_index', {}))} RUMs")

    section("Building horizontal records")
    records, summary = build_horizontal_records(
        footprints=footprints,
        packed=packed,
        velocity_fields=velocity_fields,
        uncertainty_fields=uncertainty_fields,
    )

    if summary["missing_velocity_count"]:
        warn(f"Missing east/north velocity values replaced with 0: {summary['missing_velocity_count']}")
    else:
        ok("All horizontal velocity values available")

    ok(
        f"Covariance available for {summary['covariance_available_count']} / "
        f"{summary['record_count']} records"
    )

    section("Writing horizontal field")
    footprint_meta = footprints.get("metadata") or {}
    packed_meta = packed.get("metadata") or {}

    payload = {
        "metadata": {
            "schema": "horizontal_field_v1",
            "source_footprints": generated["rum_footprints"],
            "source_packed_series": generated["packed_series"],
            "record_count": len(records),
            "row_order": "sorted_by_packed_series_row_index",
            "units": {
                "east": "mm/yr",
                "north": "mm/yr",
                "up": "mm/yr",
                "speed": "mm/yr",
                "covariance": "(mm/yr)^2",
            },
            "field_names": {
                "east": velocity_fields.get("east", "east"),
                "north": velocity_fields.get("north", "north"),
                "up": velocity_fields.get("up", "up"),
                **uncertainty_fields,
            },
            "bbox_wgs84": footprint_meta.get("bbox_wgs84_footprints"),
            "bbox_source": footprint_meta.get("bbox_source_footprints"),
            "grid": footprint_meta.get("grid"),
            "rum_count": packed_meta.get("rum_count", len(records)),
            "summary": summary,
        },
        "records": records,
    }

    write_json(output_path, payload)

    particle_records = build_particle_records(records)
    particle_stats = build_particle_stats(particle_records)
    particle_payload = {
        "metadata": {
            "schema": "horizontal_particle_field_v1",
            "source_horizontal_field": generated["horizontal_field"],
            "purpose": "lightweight canvas particle advection field only; popup metadata lives in pickable B3DM cap batch tables",
            "record_count": len(particle_records),
            "row_order": "sorted_by_packed_series_row_index",
            "units": {
                "east": "mm/yr",
                "north": "mm/yr",
                "speed": "mm/yr",
                "covariance": "(mm/yr)^2",
            },
            "kept_fields": [
                "grid_i", "grid_j", "lon", "lat", "height_row",
                "east_mm_yr", "north_mm_yr", "speed_mm_yr",
                "var_east", "var_north", "covar_en",
            ],
            "dropped_for_lightweight_runtime": [
                "rum_id", "source_row", "x_center", "y_center", "up_mm_yr",
                "unit_east", "unit_north", "azimuth_deg_clockwise_from_north",
                "var_up", "covar_eu", "covar_nu", "popup_metadata",
            ],
            "bbox_wgs84": footprint_meta.get("bbox_wgs84_footprints"),
            "grid": footprint_meta.get("grid"),
            "summary": particle_stats,
        },
        "stats": particle_stats,
        "records": particle_records,
    }

    write_json(particle_output_path, particle_payload)

    elapsed = time.time() - t_start

    ok(f"Wrote horizontal field: {output_path} ({output_path.stat().st_size / 1024 / 1024:.2f} MB)")
    ok(f"Wrote particle field: {particle_output_path} ({particle_output_path.stat().st_size / 1024 / 1024:.2f} MB)")

    section("Summary")
    ok(f"Step 14 complete in {elapsed:.2f} s")
    print(f"  Records                : {summary['record_count']}")
    print(f"  East range             : {summary['east_min_mm_yr']:.4f} to {summary['east_max_mm_yr']:.4f} mm/yr")
    print(f"  North range            : {summary['north_min_mm_yr']:.4f} to {summary['north_max_mm_yr']:.4f} mm/yr")
    print(f"  Speed range            : {summary['speed_min_mm_yr']:.4f} to {summary['speed_max_mm_yr']:.4f} mm/yr")
    print(f"  Speed p50/p98          : {summary['speed_p50_mm_yr']:.4f} / {summary['speed_p98_mm_yr']:.4f} mm/yr")
    print(f"  Covariance availability: {summary['covariance_available_count']} / {summary['record_count']}")
    print(f"  Particle field records : {len(particle_records)}")
    if output_path.exists() and particle_output_path.exists() and output_path.stat().st_size > 0:
        ratio = particle_output_path.stat().st_size / output_path.stat().st_size
        print(f"  Particle/full size     : {ratio:.3f}x")


if __name__ == "__main__":
    main()
