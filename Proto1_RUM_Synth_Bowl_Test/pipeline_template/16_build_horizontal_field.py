#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
16_build_horizontal_field.py

Generic RUM-based InSAR template step.

Purpose
-------
Build the static horizontal velocity field used by the viewer's particle layer.

Inputs:
  config.prepared_inputs.points_geojson
      source velocity/covariance fields, usually:
        east, north, up
        var_east, var_north, covar_en
        rum_id

  config.generated_outputs.rum_footprints
      corrected geometry/topology:
        grid_i, grid_j
        corners
        center

Output:
  config.generated_outputs.horizontal_field

Important
---------
The output uses corrected Phase 1/Step 03 footprint centers, not raw source
point geometry. This keeps horizontal particles aligned with the corrected
RUM caps and walls.

Horizontal velocity is static for now:
  east_mm_yr, north_mm_yr, speed_mm_yr

Covariance fields are preserved for particle uncertainty shimmer:
  var_east, var_north, covar_en
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


def as_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        v = float(value)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return default


def stats(values: List[float]) -> Optional[Dict[str, float]]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return None

    return {
        "min": float(np.min(arr)),
        "p05": float(np.percentile(arr, 5)),
        "median": float(np.percentile(arr, 50)),
        "mean": float(np.mean(arr)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def print_stats(name: str, values: List[float], unit: str = "") -> Optional[Dict[str, float]]:
    s = stats(values)
    if s is None:
        warn(f"No finite values for {name}")
        return None

    suffix = f" {unit}" if unit else ""
    print(
        f"  {name:<20s} "
        f"min={s['min']:9.3f} p05={s['p05']:9.3f} "
        f"med={s['median']:9.3f} mean={s['mean']:9.3f} "
        f"p95={s['p95']:9.3f} p99={s['p99']:9.3f} "
        f"max={s['max']:9.3f}{suffix}"
    )
    return s


# =============================================================================
# GEOMETRY HELPERS
# =============================================================================

def get_grid(fp: Dict[str, Any]) -> Tuple[int, int]:
    if "grid_i" in fp and "grid_j" in fp:
        return int(fp["grid_i"]), int(fp["grid_j"])

    grid = fp.get("grid")
    if isinstance(grid, list) and len(grid) >= 2:
        return int(grid[0]), int(grid[1])

    raise KeyError("Footprint has no grid_i/grid_j or grid field")


def get_center_lonlat(fp: Dict[str, Any]) -> Tuple[float, float]:
    center = fp.get("center")
    if isinstance(center, list) and len(center) >= 2:
        lon = as_float(center[0])
        lat = as_float(center[1])
        if lon is not None and lat is not None:
            return lon, lat

    for lon_key, lat_key in [
        ("center_lon", "center_lat"),
        ("lon", "lat"),
        ("centroid_lon", "centroid_lat"),
    ]:
        if lon_key in fp and lat_key in fp:
            lon = as_float(fp.get(lon_key))
            lat = as_float(fp.get(lat_key))
            if lon is not None and lat is not None:
                return lon, lat

    corners = fp.get("corners", [])
    if corners and len(corners) >= 4:
        lon = float(np.mean([float(c[0]) for c in corners]))
        lat = float(np.mean([float(c[1]) for c in corners]))
        return lon, lat

    raise KeyError("Cannot infer footprint center lon/lat")


def get_bbox_from_corners(footprints: Dict[str, Any]) -> Dict[str, float]:
    lons: List[float] = []
    lats: List[float] = []

    for fp in footprints.values():
        for c in fp.get("corners", []):
            if len(c) >= 2:
                lon = as_float(c[0])
                lat = as_float(c[1])
                if lon is not None and lat is not None:
                    lons.append(lon)
                    lats.append(lat)

    if not lons or not lats:
        for fp in footprints.values():
            lon, lat = get_center_lonlat(fp)
            lons.append(lon)
            lats.append(lat)

    return {
        "west": min(lons),
        "south": min(lats),
        "east": max(lons),
        "north": max(lats),
    }


def make_rum_id_from_xy(props: Dict[str, Any], x_field: str, y_field: str) -> Optional[str]:
    x = as_float(props.get(x_field))
    y = as_float(props.get(y_field))

    if x is None or y is None:
        return None

    return f"RUM_{int(round(x))}_{int(round(y))}"


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    t_start = time.time()

    cfg = load_config()

    prepared = cfg.get("prepared_inputs", {})
    generated = cfg.get("generated_outputs", {})
    source_inputs = cfg.get("source_inputs", {})

    coord_fields = source_inputs.get("source_coordinate_fields", {})
    velocity_fields = source_inputs.get("source_velocity_fields", {})
    variance_fields = source_inputs.get("source_variance_fields", {})

    x_field = str(coord_fields.get("x", "x_rum"))
    y_field = str(coord_fields.get("y", "y_rum"))

    east_field = str(velocity_fields.get("east", "east"))
    north_field = str(velocity_fields.get("north", "north"))
    up_field = str(velocity_fields.get("up", "up"))

    var_east_field = str(variance_fields.get("var_east", "var_east"))
    var_north_field = str(variance_fields.get("var_north", "var_north"))
    var_up_field = str(variance_fields.get("var_up", "var_up"))
    covar_en_field = str(variance_fields.get("covar_en", "covar_en"))
    covar_eu_field = str(variance_fields.get("covar_eu", "covar_eu"))
    covar_nu_field = str(variance_fields.get("covar_nu", "covar_nu"))

    points_path = resolve_project_path(
        prepared.get("points_geojson", "Data/points_wgs84_with_rumid.geojson")
    )
    footprints_path = resolve_project_path(
        generated.get("rum_footprints", "Data/rum_footprints.json")
    )
    output_path = resolve_project_path(
        generated.get("horizontal_field", "Data/horizontal_field.json")
    )

    section("Configuration")
    print(f"  Project root     : {PROJECT_DIR}")
    print(f"  Points GeoJSON   : {points_path}")
    print(f"  Footprints       : {footprints_path}")
    print(f"  Output field     : {output_path}")
    print(f"  Velocity fields  : east={east_field}, north={north_field}, up={up_field}")
    print(f"  Covariance fields: {var_east_field}, {var_north_field}, {covar_en_field}")
    print(f"  Coordinate fields: x={x_field}, y={y_field}")

    section("Loading inputs")
    if not points_path.exists():
        raise FileNotFoundError(f"Missing points GeoJSON: {points_path}")
    if not footprints_path.exists():
        raise FileNotFoundError(f"Missing footprints: {footprints_path}")

    points_geojson = load_json(points_path)
    features = points_geojson.get("features", [])

    fp_data = load_json(footprints_path)
    footprints = fp_data.get("footprints", {})

    if not features:
        raise ValueError("Points GeoJSON has no features")
    if not footprints:
        raise ValueError("Footprints JSON has no footprints")

    ok(f"Loaded point features: {len(features)}")
    ok(f"Loaded footprints    : {len(footprints)}")

    section("Indexing velocity source by rum_id")

    velocity_by_rum: Dict[str, Dict[str, Any]] = {}
    missing_rum_id = 0
    missing_velocity = 0

    covariance_presence = {
        var_east_field: 0,
        var_north_field: 0,
        covar_en_field: 0,
    }

    for ft in features:
        props = ft.get("properties") or {}
        geom = ft.get("geometry") or {}
        coords = geom.get("coordinates") or []

        rum_id = props.get("rum_id")
        if not rum_id:
            rum_id = make_rum_id_from_xy(props, x_field, y_field)

        if not rum_id:
            missing_rum_id += 1
            continue

        east = as_float(props.get(east_field))
        north = as_float(props.get(north_field))
        up = as_float(props.get(up_field))

        if east is None or north is None:
            missing_velocity += 1
            continue

        for key in covariance_presence:
            if as_float(props.get(key)) is not None:
                covariance_presence[key] += 1

        source_lon = as_float(props.get("lon"))
        source_lat = as_float(props.get("lat"))

        if source_lon is None and len(coords) >= 2:
            source_lon = as_float(coords[0])
        if source_lat is None and len(coords) >= 2:
            source_lat = as_float(coords[1])

        velocity_by_rum[str(rum_id)] = {
            "east": east,
            "north": north,
            "up": up,
            "var_east": as_float(props.get(var_east_field)),
            "var_north": as_float(props.get(var_north_field)),
            "var_up": as_float(props.get(var_up_field)),
            "covar_en": as_float(props.get(covar_en_field)),
            "covar_eu": as_float(props.get(covar_eu_field)),
            "covar_nu": as_float(props.get(covar_nu_field)),
            "source_lon": source_lon,
            "source_lat": source_lat,
            "x_rum": as_float(props.get(x_field)),
            "y_rum": as_float(props.get(y_field)),
        }

    ok(f"Velocity records indexed: {len(velocity_by_rum)}")
    if missing_rum_id:
        warn(f"{missing_rum_id} source rows had no rum_id and no usable x/y")
    if missing_velocity:
        warn(f"{missing_velocity} source rows had missing east/north and were skipped")

    section("Covariance field availability in source")
    for key, present in covariance_presence.items():
        print(f"  {key:<20s}: {present:5d}/{len(features):5d} numeric")
        if present == len(features):
            ok(f"{key} complete")
        elif present == 0:
            warn(f"{key} absent or non-numeric")
        else:
            warn(f"{key} partially available")

    section("Joining velocities to corrected RUM footprints")

    cells: List[Dict[str, Any]] = []
    missing_velocity_for_fp: List[str] = []
    bad_footprints: List[str] = []

    for rum_id in sorted(footprints.keys()):
        fp = footprints[rum_id]
        src = velocity_by_rum.get(str(rum_id))

        if src is None:
            missing_velocity_for_fp.append(str(rum_id))
            continue

        try:
            grid_i, grid_j = get_grid(fp)
            center_lon, center_lat = get_center_lonlat(fp)
        except Exception:
            bad_footprints.append(str(rum_id))
            continue

        east = float(src["east"])
        north = float(src["north"])
        speed = math.sqrt(east * east + north * north)

        cells.append({
            "rum_id": str(rum_id),
            "grid_i": grid_i,
            "grid_j": grid_j,

            # Corrected geometry from Step 03, not raw source point.
            "lon": center_lon,
            "lat": center_lat,

            "east_mm_yr": east,
            "north_mm_yr": north,
            "speed_mm_yr": speed,

            "up_mm_yr": src["up"],

            "var_east": src["var_east"],
            "var_north": src["var_north"],
            "var_up": src["var_up"],
            "covar_en": src["covar_en"],
            "covar_eu": src["covar_eu"],
            "covar_nu": src["covar_nu"],

            "source_lon": src["source_lon"],
            "source_lat": src["source_lat"],
            "x_rum": src["x_rum"],
            "y_rum": src["y_rum"],
        })

    ok(f"Joined cells: {len(cells)}")
    if missing_velocity_for_fp:
        warn(f"Footprints without velocity: {len(missing_velocity_for_fp)}")
        print(f"  First missing: {missing_velocity_for_fp[:5]}")
    if bad_footprints:
        warn(f"Footprints with bad geometry/grid: {len(bad_footprints)}")
        print(f"  First bad: {bad_footprints[:5]}")

    if not cells:
        raise RuntimeError("No horizontal field cells were built")

    section("Grid/domain metadata")

    grid_i_vals = [int(c["grid_i"]) for c in cells]
    grid_j_vals = [int(c["grid_j"]) for c in cells]

    i_min, i_max = min(grid_i_vals), max(grid_i_vals)
    j_min, j_max = min(grid_j_vals), max(grid_j_vals)

    bbox = get_bbox_from_corners(footprints)

    ok(f"grid_i range: {i_min} → {i_max}")
    ok(f"grid_j range: {j_min} → {j_max}")
    ok(f"bbox lon: {bbox['west']:.6f} → {bbox['east']:.6f}")
    ok(f"bbox lat: {bbox['south']:.6f} → {bbox['north']:.6f}")

    grid_lookup = {
        f"{int(c['grid_i'])},{int(c['grid_j'])}": idx
        for idx, c in enumerate(cells)
    }

    if len(grid_lookup) == len(cells):
        ok("No duplicate grid locations in horizontal field")
    else:
        warn(f"Duplicate grid locations possible: lookup={len(grid_lookup)}, cells={len(cells)}")

    section("Velocity statistics")

    east_vals = [float(c["east_mm_yr"]) for c in cells]
    north_vals = [float(c["north_mm_yr"]) for c in cells]
    speed_vals = [float(c["speed_mm_yr"]) for c in cells]
    up_vals = [float(c["up_mm_yr"]) for c in cells if c["up_mm_yr"] is not None]

    east_stats = print_stats("east_mm_yr", east_vals, " mm/yr")
    north_stats = print_stats("north_mm_yr", north_vals, " mm/yr")
    speed_stats = print_stats("speed_mm_yr", speed_vals, " mm/yr")
    up_stats = print_stats("up_mm_yr_ref", up_vals, " mm/yr")

    if speed_stats is None:
        raise RuntimeError("Cannot compute speed statistics")

    section("Covariance validity quick check")

    var_e = np.array([as_float(c.get("var_east"), np.nan) for c in cells], dtype=np.float64)
    var_n = np.array([as_float(c.get("var_north"), np.nan) for c in cells], dtype=np.float64)
    cov_en = np.array([as_float(c.get("covar_en"), np.nan) for c in cells], dtype=np.float64)

    finite_cov = np.isfinite(var_e) & np.isfinite(var_n) & np.isfinite(cov_en)
    ok(f"Finite EN covariance records: {int(finite_cov.sum())}/{len(cells)}")

    if finite_cov.any():
        neg_var = finite_cov & ((var_e < 0) | (var_n < 0))
        det = var_e * var_n - cov_en * cov_en
        bad_psd = finite_cov & (det < -1e-9)

        if np.any(neg_var):
            warn(f"Negative horizontal variance records: {int(np.sum(neg_var))}")
        else:
            ok("No negative horizontal variances")

        if np.any(bad_psd):
            warn(f"Non-PSD horizontal covariance records: {int(np.sum(bad_psd))}")
        else:
            ok("Horizontal covariance matrices PSD within tolerance")
    else:
        warn("No finite EN covariance records; uncertainty shimmer will not work")

    section("Writing horizontal field")

    output = {
        "metadata": {
            "schema": "horizontal_field_v1",
            "description": "Static horizontal velocity field for 4D Cesium particle visualization",
            "source_points": str(points_path),
            "source_footprints": str(footprints_path),
            "geometry_source": "Step 03 corrected RUM footprints",
            "velocity_source": f"{points_path.name} properties {east_field}/{north_field}",
            "units": {
                "east_mm_yr": "mm/year",
                "north_mm_yr": "mm/year",
                "speed_mm_yr": "mm/year",
                "up_mm_yr": "mm/year",
                "var_east": "source variance, likely (mm/year)^2",
                "var_north": "source variance, likely (mm/year)^2",
                "var_up": "source variance, likely (mm/year)^2",
                "covar_en": "source covariance, likely (mm/year)^2",
                "covar_eu": "source covariance, likely (mm/year)^2",
                "covar_nu": "source covariance, likely (mm/year)^2",
            },
            "particle_interpretation": (
                "Particles visualize horizontal velocity direction and relative magnitude; "
                "they are not physical material particles."
            ),
            "created_by": "16_build_horizontal_field.py",
            "created_unix": int(time.time()),
        },
        "bbox": bbox,
        "grid": {
            "i_min": i_min,
            "i_max": i_max,
            "j_min": j_min,
            "j_max": j_max,
            "valid_count": len(cells),
            "lookup": grid_lookup,
        },
        "stats": {
            "east_mm_yr": east_stats,
            "north_mm_yr": north_stats,
            "speed_mm_yr": speed_stats,
            "up_mm_yr_reference": up_stats,
            "speed_p95_mm_yr": float(speed_stats["p95"]),
            "speed_p99_mm_yr": float(speed_stats["p99"]),
            "speed_max_mm_yr": float(speed_stats["max"]),
        },
        "cells": cells,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, separators=(",", ":"))

    ok(f"Written: {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")

    section("SUMMARY")
    ok("Step 16 complete — static horizontal field package created")
    ok("Next template step: 17_check_horizontal_field.py")
    print(f"  Elapsed: {time.time() - t_start:.2f}s")
    print()


if __name__ == "__main__":
    main()
