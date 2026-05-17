#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_validate_prepared_inputs.py

Generic RUM-based InSAR template validation step.

Purpose
-------
Validate that the prepared viewer inputs are internally consistent before
packing series, building blank cells, textures, and 3D tiles.

Inputs from config:
  - prepared_inputs.points_geojson
  - prepared_inputs.vertical_epoch_json
  - generated_outputs.rum_footprints

Checks:
  1. File existence
  2. GeoJSON feature count
  3. Epoch JSON schema and epoch count
  4. RUM ID matching between points, epoch JSON, and footprints
  5. Coordinate range / optional bbox check
  6. Grid spacing sanity from footprints
  7. Epoch value and sigma sanity
  8. Footprint geometry sanity

This replaces the old Jakarta-specific phase0_validate.py.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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


def summarize_arr(values: List[float]) -> Optional[Dict[str, float]]:
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
        "max": float(np.max(arr)),
    }


def print_summary(name: str, values: List[float], unit: str = "") -> None:
    s = summarize_arr(values)
    if not s:
        warn(f"{name}: no finite values")
        return

    suffix = f" {unit}" if unit else ""
    print(
        f"  {name:<22s} "
        f"min={s['min']:10.4f} p05={s['p05']:10.4f} "
        f"med={s['median']:10.4f} mean={s['mean']:10.4f} "
        f"p95={s['p95']:10.4f} max={s['max']:10.4f}{suffix}"
    )


def get_geojson_ids_and_coords(geojson: Dict[str, Any]) -> Tuple[Set[str], List[float], List[float], int]:
    ids: Set[str] = set()
    lons: List[float] = []
    lats: List[float] = []
    missing = 0

    for feature in geojson.get("features", []):
        props = feature.get("properties") or {}
        geom = feature.get("geometry") or {}
        coords = geom.get("coordinates") or []

        rum_id = props.get("rum_id")
        lon = safe_float(coords[0]) if len(coords) >= 2 else None
        lat = safe_float(coords[1]) if len(coords) >= 2 else None

        if not rum_id:
            missing += 1
        else:
            ids.add(str(rum_id))

        if lon is not None and lat is not None:
            lons.append(lon)
            lats.append(lat)

    return ids, lons, lats, missing


def parse_numeric_array(value: Any) -> List[float]:
    if isinstance(value, str):
        if not value.strip():
            return []
        return [float(x) for x in value.split(",")]
    if isinstance(value, list):
        return [float(x) for x in value]
    return []


def get_epoch_series(epoch_data: Dict[str, Any]) -> Dict[str, Any]:
    series = epoch_data.get("series", {})
    if isinstance(series, dict):
        return series

    # Older possible format: RUMs at top-level
    ignore = {"metadata", "epochs", "epoch_decimal_year", "epoch_unix"}
    return {
        k: v
        for k, v in epoch_data.items()
        if k not in ignore and isinstance(v, dict)
    }


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    cfg = load_config()

    prepared = cfg.get("prepared_inputs", {})
    generated = cfg.get("generated_outputs", {})
    expected = cfg.get("expected_counts", {})
    bbox_check = cfg.get("bbox_check", {})

    points_path = resolve_project_path(
        prepared.get("points_geojson", "Data/points_wgs84_with_rumid.geojson")
    )
    epoch_path = resolve_project_path(
        prepared.get("vertical_epoch_json", "Data/vertical_epochs.json")
    )
    footprints_path = resolve_project_path(
        generated.get("rum_footprints", "Data/rum_footprints.json")
    )

    expected_rum_count = expected.get("rum_count")
    expected_epoch_count = expected.get("epoch_count")
    expected_spacing = float(expected.get("grid_spacing_m_nominal", 450.0))
    spacing_tolerance = float(expected.get("grid_spacing_tolerance_m", max(50.0, expected_spacing * 0.15)))

    errors = 0

    section("Configuration")
    print(f"  Project root     : {PROJECT_DIR}")
    print(f"  Points GeoJSON   : {points_path}")
    print(f"  Epoch JSON       : {epoch_path}")
    print(f"  Footprints JSON  : {footprints_path}")
    print(f"  Expected RUMs    : {expected_rum_count}")
    print(f"  Expected epochs  : {expected_epoch_count}")
    print(f"  Expected spacing : {expected_spacing:.3f} m ± {spacing_tolerance:.3f} m")

    section("1. File existence")
    for path in [points_path, epoch_path, footprints_path]:
        if path.exists():
            ok(f"{path} ({path.stat().st_size / 1024 / 1024:.2f} MB)")
        else:
            fail(f"Missing: {path}")
            errors += 1

    if errors:
        section("SUMMARY")
        fail("Validation aborted because required files are missing")
        sys.exit(1)

    section("2. Loading files")
    with points_path.open("r", encoding="utf-8") as f:
        geojson = json.load(f)
    with epoch_path.open("r", encoding="utf-8") as f:
        epoch_data = json.load(f)
    with footprints_path.open("r", encoding="utf-8") as f:
        fp_data = json.load(f)

    features = geojson.get("features", [])
    epochs = epoch_data.get("epochs", [])
    epoch_unix = epoch_data.get("epoch_unix", [])
    rum_series = get_epoch_series(epoch_data)
    footprints = fp_data.get("footprints", {})

    ok(f"GeoJSON features : {len(features)}")
    ok(f"Epoch count      : {len(epochs)}")
    ok(f"Epoch series RUMs: {len(rum_series)}")
    ok(f"Footprints       : {len(footprints)}")

    section("3. Count checks")
    if expected_rum_count is not None:
        expected_rum_count = int(expected_rum_count)
        for label, count in [
            ("GeoJSON features", len(features)),
            ("Epoch series RUMs", len(rum_series)),
            ("Footprints", len(footprints)),
        ]:
            if count == expected_rum_count:
                ok(f"{label}: {count}")
            else:
                warn(f"{label}: {count}, expected {expected_rum_count}")

    if expected_epoch_count is not None:
        expected_epoch_count = int(expected_epoch_count)
        if len(epochs) == expected_epoch_count:
            ok(f"Epoch count: {len(epochs)}")
        else:
            warn(f"Epoch count: {len(epochs)}, expected {expected_epoch_count}")

    if epoch_unix:
        if len(epoch_unix) == len(epochs):
            ok("epoch_unix length matches epochs")
        else:
            warn(f"epoch_unix length {len(epoch_unix)} != epochs length {len(epochs)}")
    else:
        warn("epoch_unix missing or empty; downstream scripts may reconstruct or fail")

    section("4. RUM ID matching")
    geo_ids, lons, lats, missing_geo_ids = get_geojson_ids_and_coords(geojson)
    epoch_ids = set(str(k) for k in rum_series.keys())
    fp_ids = set(str(k) for k in footprints.keys())

    if missing_geo_ids:
        warn(f"GeoJSON features missing rum_id: {missing_geo_ids}")
        errors += missing_geo_ids
    else:
        ok("All GeoJSON features have rum_id")

    matched_all = geo_ids & epoch_ids & fp_ids
    ok(f"Matched across all three inputs: {len(matched_all)}")

    checks = [
        ("GeoJSON only vs epoch", geo_ids - epoch_ids),
        ("Epoch only vs GeoJSON", epoch_ids - geo_ids),
        ("GeoJSON only vs footprints", geo_ids - fp_ids),
        ("Footprints only vs GeoJSON", fp_ids - geo_ids),
        ("Epoch only vs footprints", epoch_ids - fp_ids),
        ("Footprints only vs epoch", fp_ids - epoch_ids),
    ]

    for label, diff in checks:
        if diff:
            warn(f"{label}: {len(diff)}")
            print(f"    sample: {sorted(diff)[:5]}")
        else:
            ok(f"{label}: 0")

    section("5. Coordinate range")
    if lons and lats:
        print(f"  Lon range: {min(lons):.6f} → {max(lons):.6f}")
        print(f"  Lat range: {min(lats):.6f} → {max(lats):.6f}")

        if bool(bbox_check.get("enabled", False)):
            lon_min = safe_float(bbox_check.get("lon_min"))
            lon_max = safe_float(bbox_check.get("lon_max"))
            lat_min = safe_float(bbox_check.get("lat_min"))
            lat_max = safe_float(bbox_check.get("lat_max"))

            if (
                lon_min is not None
                and lon_max is not None
                and lat_min is not None
                and lat_max is not None
            ):
                lon_min_f = float(lon_min)
                lon_max_f = float(lon_max)
                lat_min_f = float(lat_min)
                lat_max_f = float(lat_max)

                bad = 0
                for lon, lat in zip(lons, lats):
                    if not (lon_min_f <= lon <= lon_max_f and lat_min_f <= lat <= lat_max_f):
                        bad += 1

                print(f"  Config bbox: lon {lon_min_f} → {lon_max_f}, lat {lat_min_f} → {lat_max_f}")
                if bad == 0:
                    ok("All point coordinates inside configured bbox")
                else:
                    warn(f"{bad} points outside configured bbox")
            else:
                warn("bbox_check.enabled=true but bbox values are incomplete")
    else:
        warn("No valid point coordinates found")

    section("6. Grid spacing / topology sanity")
    grid_i = []
    grid_j = []
    diag_samples = []

    grid_keys = set()

    for rum_id, fp in footprints.items():
        gi = fp.get("grid_i")
        gj = fp.get("grid_j")
        if gi is not None and gj is not None:
            grid_i.append(int(gi))
            grid_j.append(int(gj))
            grid_keys.add((int(gi), int(gj)))

        corners = fp.get("corners", [])
        if len(corners) >= 4:
            ne = corners[0]
            sw = corners[2]
            try:
                diag_samples.append(haversine_m(ne[0], ne[1], sw[0], sw[1]))
            except Exception:
                pass

    if grid_i and grid_j:
        ok(f"grid_i range: {min(grid_i)} → {max(grid_i)}")
        ok(f"grid_j range: {min(grid_j)} → {max(grid_j)}")
        if len(grid_keys) == len(footprints):
            ok("No duplicate grid cells in footprints")
        else:
            warn(f"Duplicate grid cells possible: unique={len(grid_keys)}, footprints={len(footprints)}")

    if diag_samples:
        median_diag = float(np.percentile(np.asarray(diag_samples), 50))
        expected_diag = math.sqrt(2.0) * expected_spacing
        print(f"  Median footprint diagonal: {median_diag:.3f} m")
        print(f"  Nominal expected diagonal: {expected_diag:.3f} m")
        if abs(median_diag - expected_diag) <= spacing_tolerance * math.sqrt(2.0):
            ok("Footprint diagonal close to nominal spacing")
        else:
            warn("Footprint diagonal differs from nominal spacing more than tolerance")

    grid_model = fp_data.get("metadata", {}).get("grid_model", {})
    if grid_model:
        spacing_u = safe_float(grid_model.get("spacing_u"))
        spacing_v = safe_float(grid_model.get("spacing_v"))
        axis_deg = safe_float(grid_model.get("axis_angle_deg"))
        print(f"  Inferred grid spacing U/V: {spacing_u:.3f} / {spacing_v:.3f} m")
        print(f"  Inferred grid axis angle : {axis_deg:.6f}°")

    section("7. Epoch value and sigma sanity")
    sample_ids = sorted(list(matched_all))[:5]
    bad_series_length = 0
    all_first_values = []
    all_last_values = []
    sigma_samples = []

    for rid in epoch_ids:
        entry = rum_series[rid]
        v = parse_numeric_array(entry.get("vertical_mm", []))
        s = parse_numeric_array(entry.get("sigma_mm", []))
        if len(v) != len(epochs) or len(s) != len(epochs):
            bad_series_length += 1
            continue

        if v:
            all_first_values.append(v[0])
            all_last_values.append(v[-1])
        if s:
            sigma_samples.extend(s[::max(1, len(s)//20)])  # sample for speed

    if bad_series_length:
        fail(f"RUM series with wrong vertical/sigma length: {bad_series_length}")
        errors += bad_series_length
    else:
        ok("All checked RUM series lengths match epoch count")

    for rid in sample_ids:
        entry = rum_series[rid]
        v = parse_numeric_array(entry.get("vertical_mm", []))
        s = parse_numeric_array(entry.get("sigma_mm", []))
        up = safe_float(entry.get("source_up_mm_yr"))
        print(
            f"  {rid:<24s} "
            f"n={len(v):3d} "
            f"v=[{min(v):8.2f},{max(v):8.2f}] mm "
            f"s=[{min(s):8.3f},{max(s):8.3f}] mm "
            f"up={up if up is not None else float('nan'):8.3f} mm/yr"
        )

    if all_first_values:
        print_summary("epoch0 vertical", all_first_values, "mm")
    if all_last_values:
        print_summary("last vertical", all_last_values, "mm")
    if sigma_samples:
        print_summary("sigma sampled", sigma_samples, "mm")

    epoch0_abs_max = max(abs(v) for v in all_first_values) if all_first_values else float("nan")
    if math.isfinite(epoch0_abs_max):
        if epoch0_abs_max < 1e-6:
            ok("Epoch 0 displacement is zero for all RUMs")
        else:
            warn(f"Epoch 0 displacement not zero everywhere; max abs={epoch0_abs_max:.6f} mm")

    section("8. Footprint geometry sanity")
    bad_corners = 0
    bad_center = 0

    for rid, fp in footprints.items():
        center = fp.get("center", [])
        corners = fp.get("corners", [])

        if len(center) < 2:
            bad_center += 1
        if len(corners) != 4:
            bad_corners += 1
            continue

        for c in corners:
            if len(c) < 2 or safe_float(c[0]) is None or safe_float(c[1]) is None:
                bad_corners += 1
                break

    if bad_center:
        fail(f"Footprints with missing/bad center: {bad_center}")
        errors += bad_center
    else:
        ok("All footprints have center coordinates")

    if bad_corners:
        fail(f"Footprints with missing/bad corners: {bad_corners}")
        errors += bad_corners
    else:
        ok("All footprints have 4 valid corners")

    section("SUMMARY")
    if errors == 0:
        ok("ALL VALIDATION CHECKS PASSED — ready for Step 06 pack vertical series")
    else:
        fail(f"{errors} error(s) found — fix before continuing")
        sys.exit(1)

    print()


if __name__ == "__main__":
    main()
