#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
17_check_horizontal_field.py

Generic RUM-based InSAR template diagnostic step.

Purpose
-------
Validate Data/horizontal_field.json before using it in the viewer particle layer.

Reads:
  config.generated_outputs.horizontal_field

Writes:
  Data/horizontal_debug_vectors.geojson

The debug GeoJSON contains sampled vector line segments using corrected
RUM centers. It is for GIS/Cesium sanity inspection only, not the final
particle visualization.
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter
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
# HELPERS
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


def stats(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("No finite values for stats")
    return {
        "min": float(np.min(arr)),
        "p01": float(np.percentile(arr, 1)),
        "p05": float(np.percentile(arr, 5)),
        "median": float(np.percentile(arr, 50)),
        "mean": float(np.mean(arr)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def print_stats(name: str, values: List[float], unit: str = "") -> Dict[str, float]:
    s = stats(values)
    suffix = f" {unit}" if unit else ""
    print(
        f"  {name:<18s} "
        f"min={s['min']:9.3f} p01={s['p01']:9.3f} p05={s['p05']:9.3f} "
        f"med={s['median']:9.3f} mean={s['mean']:9.3f} "
        f"p95={s['p95']:9.3f} p99={s['p99']:9.3f} max={s['max']:9.3f}{suffix}"
    )
    return s


def offset_lonlat_m(lon: float, lat: float, east_m: float, north_m: float) -> Tuple[float, float]:
    lat_rad = math.radians(lat)
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * max(math.cos(lat_rad), 1e-9)
    return lon + east_m / m_per_deg_lon, lat + north_m / m_per_deg_lat


def vector_endpoint(
    lon: float,
    lat: float,
    east_mm_yr: float,
    north_mm_yr: float,
    speed_p95: float,
    max_debug_vector_length_m: float,
) -> Tuple[float, float]:
    speed = math.sqrt(east_mm_yr * east_mm_yr + north_mm_yr * north_mm_yr)
    if speed <= 1e-12:
        return lon, lat

    length_m = min(
        max_debug_vector_length_m,
        max_debug_vector_length_m * speed / max(speed_p95, 1e-9),
    )
    unit_e = east_mm_yr / speed
    unit_n = north_mm_yr / speed

    return offset_lonlat_m(lon, lat, unit_e * length_m, unit_n * length_m)


def make_arrowhead(
    lon0: float,
    lat0: float,
    lon1: float,
    lat1: float,
    east_mm_yr: float,
    north_mm_yr: float,
    length_m: float,
) -> List[List[List[float]]]:
    speed = math.sqrt(east_mm_yr * east_mm_yr + north_mm_yr * north_mm_yr)
    if speed <= 1e-12:
        return []

    ue = east_mm_yr / speed
    un = north_mm_yr / speed

    angle = math.radians(150.0)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    arm1_e = ue * cos_a - un * sin_a
    arm1_n = ue * sin_a + un * cos_a

    arm2_e = ue * cos_a + un * sin_a
    arm2_n = -ue * sin_a + un * cos_a

    a1 = offset_lonlat_m(lon1, lat1, arm1_e * length_m, arm1_n * length_m)
    a2 = offset_lonlat_m(lon1, lat1, arm2_e * length_m, arm2_n * length_m)

    return [
        [[lon1, lat1], [a1[0], a1[1]]],
        [[lon1, lat1], [a2[0], a2[1]]],
    ]


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    t_start = time.time()

    cfg = load_config()
    generated = cfg.get("generated_outputs", {})
    horizontal_cfg = cfg.get("horizontal_debug", {})

    field_path = resolve_project_path(
        generated.get("horizontal_field", "Data/horizontal_field.json")
    )
    out_debug_geojson = resolve_project_path(
        generated.get("horizontal_debug_vectors", "Data/horizontal_debug_vectors.geojson")
    )

    max_debug_vector_length_m = float(horizontal_cfg.get("max_debug_vector_length_m", 350.0))
    debug_grid_stride = int(horizontal_cfg.get("debug_grid_stride", 2))
    include_top_n_speeds = int(horizontal_cfg.get("include_top_n_speeds", 50))
    arrowhead_length_m = float(horizontal_cfg.get("arrowhead_length_m", 70.0))

    section("Configuration")
    print(f"  Project root       : {PROJECT_DIR}")
    print(f"  Horizontal field   : {field_path}")
    print(f"  Debug vector output: {out_debug_geojson}")
    print(f"  Max vector length  : {max_debug_vector_length_m} m")
    print(f"  Debug grid stride  : {debug_grid_stride}")
    print(f"  Include top speeds : {include_top_n_speeds}")

    section("Loading horizontal field")
    if not field_path.exists():
        raise FileNotFoundError(f"Missing horizontal field: {field_path}")

    field = load_json(field_path)
    cells = field.get("cells", [])
    grid = field.get("grid", {})
    bbox = field.get("bbox", {})
    lookup = grid.get("lookup", {})

    if not cells:
        raise ValueError("horizontal_field.json has no cells")

    ok(f"Cells loaded  : {len(cells)}")
    ok(f"Lookup entries: {len(lookup)}")
    print(
        f"  bbox: lon {bbox.get('west'):.6f} → {bbox.get('east'):.6f}, "
        f"lat {bbox.get('south'):.6f} → {bbox.get('north'):.6f}"
    )

    section("Basic integrity checks")

    rum_ids = [str(c.get("rum_id")) for c in cells]
    duplicate_rums = [rid for rid, n in Counter(rum_ids).items() if n > 1]
    if duplicate_rums:
        fail(f"Duplicate RUM IDs: {len(duplicate_rums)}")
        print(f"  First duplicates: {duplicate_rums[:5]}")
    else:
        ok("No duplicate RUM IDs")

    grid_keys = [(int(c["grid_i"]), int(c["grid_j"])) for c in cells]
    duplicate_grid = [g for g, n in Counter(grid_keys).items() if n > 1]
    if duplicate_grid:
        fail(f"Duplicate grid cells: {len(duplicate_grid)}")
        print(f"  First duplicates: {duplicate_grid[:5]}")
    else:
        ok("No duplicate grid cells")

    bad_values = []
    required_numeric = ["lon", "lat", "east_mm_yr", "north_mm_yr", "speed_mm_yr"]
    for c in cells:
        for key in required_numeric:
            v = as_float(c.get(key))
            if v is None:
                bad_values.append((c.get("rum_id"), key, c.get(key)))

    if bad_values:
        fail(f"Bad numeric lon/lat/vector/speed values: {len(bad_values)}")
        print(f"  First bad values: {bad_values[:5]}")
    else:
        ok("No bad numeric lon/lat/vector/speed values")

    section("Grid occupancy")

    i_vals = [int(c["grid_i"]) for c in cells]
    j_vals = [int(c["grid_j"]) for c in cells]
    i_min, i_max = min(i_vals), max(i_vals)
    j_min, j_max = min(j_vals), max(j_vals)

    possible_cells = (i_max - i_min + 1) * (j_max - j_min + 1)
    occupancy = len(cells) / max(possible_cells, 1) * 100.0

    print(f"  grid_i: {i_min} → {i_max} ({i_max - i_min + 1} columns)")
    print(f"  grid_j: {j_min} → {j_max} ({j_max - j_min + 1} rows)")
    print(f"  bounding rectangle cells: {possible_cells}")
    print(f"  valid RUM cells: {len(cells)}")
    print(f"  occupancy: {occupancy:.1f}%")

    cell_set = set(grid_keys)
    n4_counts = []
    n8_counts = []

    for i, j in grid_keys:
        n4 = 0
        for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            if (i + di, j + dj) in cell_set:
                n4 += 1
        n4_counts.append(n4)

        n8 = 0
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                if (i + di, j + dj) in cell_set:
                    n8 += 1
        n8_counts.append(n8)

    print("  4-neighbor count distribution:")
    for k, count in sorted(Counter(n4_counts).items()):
        print(f"    {k} neighbors: {count:4d} cells")

    print("  8-neighbor count distribution:")
    for k, count in sorted(Counter(n8_counts).items()):
        print(f"    {k} neighbors: {count:4d} cells")

    section("Velocity statistics")

    east = [float(c["east_mm_yr"]) for c in cells]
    north = [float(c["north_mm_yr"]) for c in cells]
    speed = [float(c["speed_mm_yr"]) for c in cells]

    east_stats = print_stats("east", east, "mm/yr")
    north_stats = print_stats("north", north, "mm/yr")
    speed_stats = print_stats("speed", speed, "mm/yr")

    if speed_stats["p95"] <= 0:
        warn("p95 speed is zero/negative; particle speed scaling would fail")
    else:
        ok(f"Recommended first particle speed normalization: p95 = {speed_stats['p95']:.3f} mm/yr")

    section("Horizontal covariance sanity")

    var_e = np.array([as_float(c.get("var_east"), np.nan) for c in cells], dtype=float)
    var_n = np.array([as_float(c.get("var_north"), np.nan) for c in cells], dtype=float)
    cov = np.array([as_float(c.get("covar_en"), np.nan) for c in cells], dtype=float)

    finite = np.isfinite(var_e) & np.isfinite(var_n) & np.isfinite(cov)
    ok(f"Finite covariance records: {int(finite.sum())}/{len(cells)}")

    if np.any(finite):
        det = var_e * var_n - cov * cov
        neg_var = finite & ((var_e < 0) | (var_n < 0))
        bad_psd = finite & (det < -1e-9)

        if np.any(neg_var):
            warn(f"Negative horizontal variance records: {int(np.sum(neg_var))}")
        else:
            ok("No negative horizontal variances")

        if np.any(bad_psd):
            warn(f"Non-PSD horizontal covariance records: {int(np.sum(bad_psd))}")
        else:
            ok("Covariance matrices are positive semi-definite within tolerance")
    else:
        warn("No finite covariance records; uncertainty shimmer cannot be derived")

    section("Outlier vectors")

    cells_sorted = sorted(cells, key=lambda c: float(c["speed_mm_yr"]), reverse=True)
    for rank, c in enumerate(cells_sorted[:12], start=1):
        print(
            f"  #{rank:02d} {str(c['rum_id']):<24s} "
            f"grid=({int(c['grid_i']):>4},{int(c['grid_j']):>4}) "
            f"speed={float(c['speed_mm_yr']):8.3f} mm/yr "
            f"east={float(c['east_mm_yr']):9.3f} "
            f"north={float(c['north_mm_yr']):9.3f} "
            f"lon={float(c['lon']):.6f} lat={float(c['lat']):.6f}"
        )

    section("Writing debug vector GeoJSON")

    speed_p95 = float(speed_stats["p95"])
    top_ids = set(str(c["rum_id"]) for c in cells_sorted[:include_top_n_speeds])

    features: List[Dict[str, Any]] = []
    exported = 0

    for c in cells:
        grid_i = int(c["grid_i"])
        grid_j = int(c["grid_j"])

        include_by_stride = (
            (grid_i - i_min) % debug_grid_stride == 0
            and (grid_j - j_min) % debug_grid_stride == 0
        )
        include_by_speed = str(c["rum_id"]) in top_ids

        if not (include_by_stride or include_by_speed):
            continue

        lon0 = float(c["lon"])
        lat0 = float(c["lat"])
        east_v = float(c["east_mm_yr"])
        north_v = float(c["north_mm_yr"])
        speed_v = float(c["speed_mm_yr"])

        lon1, lat1 = vector_endpoint(
            lon0,
            lat0,
            east_v,
            north_v,
            speed_p95,
            max_debug_vector_length_m,
        )

        if abs(lon0 - lon1) < 1e-15 and abs(lat0 - lat1) < 1e-15:
            continue

        debug_length_m = min(
            max_debug_vector_length_m,
            max_debug_vector_length_m * speed_v / max(speed_p95, 1e-9),
        )

        features.append({
            "type": "Feature",
            "properties": {
                "rum_id": c["rum_id"],
                "kind": "shaft",
                "grid_i": grid_i,
                "grid_j": grid_j,
                "east_mm_yr": east_v,
                "north_mm_yr": north_v,
                "speed_mm_yr": speed_v,
                "debug_length_m": debug_length_m,
                "included_by": "top_speed" if include_by_speed and not include_by_stride else "stride",
            },
            "geometry": {
                "type": "LineString",
                "coordinates": [[lon0, lat0], [lon1, lat1]],
            },
        })
        exported += 1

        for arm in make_arrowhead(
            lon0,
            lat0,
            lon1,
            lat1,
            east_v,
            north_v,
            arrowhead_length_m,
        ):
            features.append({
                "type": "Feature",
                "properties": {
                    "rum_id": c["rum_id"],
                    "kind": "arrowhead",
                    "grid_i": grid_i,
                    "grid_j": grid_j,
                    "speed_mm_yr": speed_v,
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": arm,
                },
            })

    geojson = {
        "type": "FeatureCollection",
        "metadata": {
            "source": str(field_path),
            "purpose": "debug horizontal velocity vectors; not final visualization",
            "max_debug_vector_length_m": max_debug_vector_length_m,
            "debug_grid_stride": debug_grid_stride,
            "include_top_n_speeds": include_top_n_speeds,
            "arrowhead_length_m": arrowhead_length_m,
            "speed_p95_mm_yr": speed_p95,
            "created_by": "17_check_horizontal_field.py",
            "created_unix": int(time.time()),
        },
        "features": features,
    }

    out_debug_geojson.parent.mkdir(parents=True, exist_ok=True)
    with out_debug_geojson.open("w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False, separators=(",", ":"))

    ok(f"Exported vector shafts: {exported}")
    ok(f"GeoJSON features incl. arrowheads: {len(features)}")
    ok(f"Written: {out_debug_geojson} ({out_debug_geojson.stat().st_size / 1024:.1f} KB)")

    section("Interpretation")
    print("  Check debug vectors in GIS/Cesium if needed:")
    print("    1. Vectors should point in plausible regional directions.")
    print("    2. No obvious lon/lat flip or east/north swap.")
    print("    3. Outlier vectors should be inspected, not automatically removed.")
    print("    4. This debug arrow layer is NOT the final particle visualization.")

    section("SUMMARY")
    ok("Step 17 complete — horizontal field sanity check passed and debug vectors exported")
    ok("Next template step: 18_check_horizontal_uncertainty.py")
    print(f"  Elapsed: {time.time() - t_start:.2f}s")
    print()


if __name__ == "__main__":
    main()
