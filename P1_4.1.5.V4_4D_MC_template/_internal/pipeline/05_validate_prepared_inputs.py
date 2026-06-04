#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_validate_prepared_inputs.py

InSAR4D RUM Viewer pipeline step 05.

Purpose
-------
Validate the prepared data products created by steps 01–04.

This step is a gate/check step. It should not modify the main data products.

Validated inputs
----------------
1. points_wgs84_with_rumid.geojson
2. vertical_epochs.json
3. rum_footprints.json

Checks
------
- required files exist
- GeoJSON structure is valid enough for the pipeline
- RUM count matches expected count, or warns if not
- RUM IDs match across points, footprints, and vertical series
- epoch count matches config-derived expected count
- measurement/model/sigma arrays are finite and have correct length
- footprint polygons are closed and finite
- grid_i/grid_j exists and has no duplicate occupied cells
- bbox metadata exists
- source x/y coordinates are finite
- sigma is non-negative

Output
------
  _internal/data_pipeline/validation_report.json

The runner captures console output in run_records/latest_run.log.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from pipeline_config import load_resolved_config, resolve_path


# =============================================================================
# ADVANCED SETTINGS
# =============================================================================
# Normal users should edit config/project_config.json, not this script.

FAIL_ON_ID_SET_MISMATCH = True
FAIL_ON_DUPLICATE_GRID_CELLS = True
FAIL_ON_INVALID_GEOMETRY = True
FAIL_ON_INVALID_VERTICAL_SERIES = True

# Expected RUM count and RUM size mismatch are warnings by default,
# because the user explicitly wanted these to be non-terminating.
EXPECTED_COUNT_MISMATCH_IS_WARNING = True

ROUND_SUMMARY_DIGITS = 4
VALIDATION_REPORT_NAME = "validation_report.json"


# =============================================================================
# PRINT HELPERS
# =============================================================================

WARNINGS: List[str] = []
ERRORS: List[str] = []


def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def ok(msg: str) -> None:
    print(f"  [OK]   {msg}")


def warn(msg: str) -> None:
    WARNINGS.append(msg)
    print(f"  [WARN] {msg}")


def fail(msg: str) -> None:
    ERRORS.append(msg)
    print(f"  [FAIL] {msg}")


# =============================================================================
# BASIC HELPERS
# =============================================================================

def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


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


def minmax(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    return min(values), max(values)


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


def require_feature_collection(payload: Dict[str, Any], name: str) -> List[Dict[str, Any]]:
    if payload.get("type") != "FeatureCollection":
        raise ValueError(f"{name} is not a GeoJSON FeatureCollection")
    features = payload.get("features")
    if not isinstance(features, list) or not features:
        raise ValueError(f"{name} contains no features")
    return features


def get_props(feature: Dict[str, Any]) -> Dict[str, Any]:
    props = feature.get("properties")
    return props if isinstance(props, dict) else {}


def get_rum_id(props: Dict[str, Any], idx: int, context: str) -> str:
    rid = props.get("rum_id")
    if rid is None or str(rid).strip() == "":
        raise ValueError(f"{context} feature {idx} missing rum_id")
    return str(rid)


def set_sample(values: Iterable[str], n: int = 8) -> List[str]:
    return sorted(list(values))[:n]


# =============================================================================
# POINT VALIDATION
# =============================================================================

def validate_points(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    ids: List[str] = []
    lon_values: List[float] = []
    lat_values: List[float] = []
    x_values: List[float] = []
    y_values: List[float] = []

    invalid_geom = 0
    invalid_source_xy = 0

    for idx, feature in enumerate(features):
        props = get_props(feature)
        rid = get_rum_id(props, idx, "point")
        ids.append(rid)

        geom = feature.get("geometry") or {}
        coords = geom.get("coordinates") or []

        if geom.get("type") != "Point" or len(coords) < 2:
            invalid_geom += 1
        else:
            lon = safe_float(coords[0])
            lat = safe_float(coords[1])
            if lon is None or lat is None or not (-180 <= lon <= 180) or not (-90 <= lat <= 90):
                invalid_geom += 1
            else:
                lon_values.append(lon)
                lat_values.append(lat)

        x = safe_float(props.get("x_source"))
        y = safe_float(props.get("y_source"))
        if x is None or y is None:
            invalid_source_xy += 1
        else:
            x_values.append(x)
            y_values.append(y)

    duplicate_ids = len(ids) - len(set(ids))

    if duplicate_ids:
        raise ValueError(f"Point product has duplicate rum_id values: {duplicate_ids}")

    if invalid_geom:
        msg = f"Point product has invalid point geometries: {invalid_geom}"
        if FAIL_ON_INVALID_GEOMETRY:
            raise ValueError(msg)
        warn(msg)

    if invalid_source_xy:
        msg = f"Point product has missing/invalid x_source/y_source: {invalid_source_xy}"
        if FAIL_ON_INVALID_GEOMETRY:
            raise ValueError(msg)
        warn(msg)

    ok(f"Point product valid: {len(ids)} points, unique rum_id values")

    return {
        "ids": set(ids),
        "count": len(ids),
        "bbox_wgs84": {
            "west": round(min(lon_values), ROUND_SUMMARY_DIGITS),
            "south": round(min(lat_values), ROUND_SUMMARY_DIGITS),
            "east": round(max(lon_values), ROUND_SUMMARY_DIGITS),
            "north": round(max(lat_values), ROUND_SUMMARY_DIGITS),
        },
        "bbox_source": {
            "min_x": round(min(x_values), ROUND_SUMMARY_DIGITS),
            "min_y": round(min(y_values), ROUND_SUMMARY_DIGITS),
            "max_x": round(max(x_values), ROUND_SUMMARY_DIGITS),
            "max_y": round(max(y_values), ROUND_SUMMARY_DIGITS),
        },
    }


# =============================================================================
# FOOTPRINT VALIDATION
# =============================================================================

def validate_footprints(payload: Dict[str, Any], features: List[Dict[str, Any]]) -> Dict[str, Any]:
    ids: List[str] = []
    grid_cells: Set[Tuple[int, int]] = set()
    duplicate_grid = 0
    invalid_geom = 0
    invalid_grid = 0
    rum_sizes: List[float] = []
    areas: List[float] = []

    for idx, feature in enumerate(features):
        props = get_props(feature)
        rid = get_rum_id(props, idx, "footprint")
        ids.append(rid)

        gi = props.get("grid_i")
        gj = props.get("grid_j")
        if gi is None or gj is None:
            invalid_grid += 1
        else:
            try:
                cell = (int(gi), int(gj))
                if cell in grid_cells:
                    duplicate_grid += 1
                grid_cells.add(cell)
            except Exception:
                invalid_grid += 1

        rs = safe_float(props.get("rum_size_m"))
        if rs is not None:
            rum_sizes.append(rs)

        area = safe_float(props.get("area_m2"))
        if area is not None:
            areas.append(area)

        geom = feature.get("geometry") or {}
        rings = geom.get("coordinates") or []
        if geom.get("type") != "Polygon" or not rings or not isinstance(rings[0], list):
            invalid_geom += 1
            continue

        ring = rings[0]
        if len(ring) < 4:
            invalid_geom += 1
            continue

        if ring[0] != ring[-1]:
            invalid_geom += 1
            continue

        for coord in ring:
            if not isinstance(coord, list) or len(coord) < 2:
                invalid_geom += 1
                break
            lon = safe_float(coord[0])
            lat = safe_float(coord[1])
            if lon is None or lat is None or not (-180 <= lon <= 180) or not (-90 <= lat <= 90):
                invalid_geom += 1
                break

    duplicate_ids = len(ids) - len(set(ids))
    if duplicate_ids:
        raise ValueError(f"Footprint product has duplicate rum_id values: {duplicate_ids}")

    if invalid_grid:
        raise ValueError(f"Footprint product has invalid/missing grid_i/grid_j: {invalid_grid}")

    if duplicate_grid:
        msg = f"Footprint product has duplicate grid cells: {duplicate_grid}"
        if FAIL_ON_DUPLICATE_GRID_CELLS:
            raise ValueError(msg)
        warn(msg)

    if invalid_geom:
        msg = f"Footprint product has invalid polygon geometries: {invalid_geom}"
        if FAIL_ON_INVALID_GEOMETRY:
            raise ValueError(msg)
        warn(msg)

    metadata = payload.get("metadata") or {}
    required_meta = [
        "bbox_wgs84_footprints",
        "bbox_source_footprints",
        "grid",
        "rum_size_m",
        "rum_count",
    ]
    missing_meta = [key for key in required_meta if key not in metadata]
    if missing_meta:
        warn(f"Footprint metadata missing keys: {missing_meta}")

    ok(f"Footprint product valid: {len(ids)} polygons, {len(grid_cells)} occupied grid cells")

    area_min, area_max = minmax(areas)
    size_min, size_max = minmax(rum_sizes)

    return {
        "ids": set(ids),
        "count": len(ids),
        "grid_cell_count": len(grid_cells),
        "grid_i_min": min(i for i, _ in grid_cells),
        "grid_i_max": max(i for i, _ in grid_cells),
        "grid_j_min": min(j for _, j in grid_cells),
        "grid_j_max": max(j for _, j in grid_cells),
        "rum_size_min_m": size_min,
        "rum_size_max_m": size_max,
        "area_min_m2": area_min,
        "area_max_m2": area_max,
        "metadata": metadata,
    }


# =============================================================================
# EPOCH VALIDATION
# =============================================================================

def validate_epochs(payload: Dict[str, Any], expected_epoch_count: int) -> Dict[str, Any]:
    epochs = payload.get("epochs", [])
    epoch_decimal_year = payload.get("epoch_decimal_year", [])
    epoch_unix = payload.get("epoch_unix", [])
    series = payload.get("series", {})

    if not isinstance(epochs, list) or not epochs:
        raise ValueError("Vertical epoch product has no epochs")
    if not isinstance(series, dict) or not series:
        raise ValueError("Vertical epoch product has no series")

    if len(epochs) != expected_epoch_count:
        raise ValueError(f"Epoch count mismatch: actual={len(epochs)}, expected={expected_epoch_count}")

    if epoch_decimal_year and len(epoch_decimal_year) != len(epochs):
        raise ValueError("epoch_decimal_year length does not match epochs")
    if epoch_unix and len(epoch_unix) != len(epochs):
        raise ValueError("epoch_unix length does not match epochs")

    bad_series = 0
    bad_sigma = 0

    measurement_values: List[float] = []
    model_values: List[float] = []
    sigma_values: List[float] = []
    up_values: List[float] = []

    for rum_id, item in series.items():
        measurement = item.get("measurement_mm")
        model = item.get("model_mm")
        sigma = item.get("sigma_mm")

        if not isinstance(measurement, list) or len(measurement) != len(epochs):
            bad_series += 1
            continue
        if not isinstance(model, list) or len(model) != len(epochs):
            bad_series += 1
            continue
        if not isinstance(sigma, list) or len(sigma) != len(epochs):
            bad_series += 1
            continue

        for v in measurement:
            vf = safe_float(v)
            if vf is None:
                bad_series += 1
                break
            measurement_values.append(vf)

        for v in model:
            vf = safe_float(v)
            if vf is None:
                bad_series += 1
                break
            model_values.append(vf)

        for s in sigma:
            sf = safe_float(s)
            if sf is None or sf < 0:
                bad_sigma += 1
                break
            sigma_values.append(sf)

        up = safe_float(item.get("source_up_mm_yr"))
        if up is not None:
            up_values.append(up)

    if bad_series:
        msg = f"Invalid measurement/model/sigma series arrays: {bad_series}"
        if FAIL_ON_INVALID_VERTICAL_SERIES:
            raise ValueError(msg)
        warn(msg)

    if bad_sigma:
        msg = f"Invalid negative/nonfinite sigma arrays: {bad_sigma}"
        if FAIL_ON_INVALID_VERTICAL_SERIES:
            raise ValueError(msg)
        warn(msg)

    ok(f"Vertical epoch product valid: {len(series)} RUMs × {len(epochs)} epochs")
    ok("Validated MEASUREMENT, MODEL, and sigma arrays")

    measurement_min, measurement_max = minmax(measurement_values)
    model_min, model_max = minmax(model_values)
    s_min, s_max = minmax(sigma_values)
    up_min, up_max = minmax(up_values)

    return {
        "ids": set(str(k) for k in series.keys()),
        "rum_count": len(series),
        "epoch_count": len(epochs),
        "first_epoch": epochs[0],
        "last_epoch": epochs[-1],

        "measurement_min_mm": measurement_min,
        "measurement_p02_mm": percentile(measurement_values, 2),
        "measurement_p50_mm": percentile(measurement_values, 50),
        "measurement_p98_mm": percentile(measurement_values, 98),
        "measurement_max_mm": measurement_max,

        "model_min_mm": model_min,
        "model_p02_mm": percentile(model_values, 2),
        "model_p50_mm": percentile(model_values, 50),
        "model_p98_mm": percentile(model_values, 98),
        "model_max_mm": model_max,

        "sigma_min_mm": s_min,
        "sigma_p02_mm": percentile(sigma_values, 2),
        "sigma_p50_mm": percentile(sigma_values, 50),
        "sigma_p98_mm": percentile(sigma_values, 98),
        "sigma_max_mm": s_max,

        "up_min_mm_yr": up_min,
        "up_max_mm_yr": up_max,
        "metadata": payload.get("metadata") or {},
    }


# =============================================================================
# CROSS-CHECKS
# =============================================================================

def check_id_sets(point_ids: Set[str], footprint_ids: Set[str], epoch_ids: Set[str]) -> None:
    all_equal = point_ids == footprint_ids == epoch_ids

    if all_equal:
        ok("RUM ID sets match across points, footprints, and epoch series")
        return

    p_not_f = point_ids - footprint_ids
    f_not_p = footprint_ids - point_ids
    p_not_e = point_ids - epoch_ids
    e_not_p = epoch_ids - point_ids

    msg = (
        "RUM ID set mismatch. "
        f"points-not-footprints={len(p_not_f)}, footprints-not-points={len(f_not_p)}, "
        f"points-not-epochs={len(p_not_e)}, epochs-not-points={len(e_not_p)}. "
        f"samples: p-not-f={set_sample(p_not_f)}, f-not-p={set_sample(f_not_p)}, "
        f"p-not-e={set_sample(p_not_e)}, e-not-p={set_sample(e_not_p)}"
    )

    if FAIL_ON_ID_SET_MISMATCH:
        raise ValueError(msg)
    warn(msg)


def check_expected_count(actual_count: int, expected_count: Any) -> None:
    if expected_count is None:
        warn("No expected_rum_count provided; count mismatch check skipped")
        return

    expected = int(expected_count)
    if actual_count == expected:
        ok(f"Actual RUM count matches expected count: {actual_count}")
        return

    msg = f"Actual RUM count differs from expected: actual={actual_count}, expected={expected}"
    if EXPECTED_COUNT_MISMATCH_IS_WARNING:
        warn(msg)
    else:
        raise ValueError(msg)


def check_rum_size(footprint_summary: Dict[str, Any], expected_rum_size_m: float) -> None:
    size_min = footprint_summary.get("rum_size_min_m")
    size_max = footprint_summary.get("rum_size_max_m")

    if size_min is None or size_max is None:
        warn("Cannot validate footprint rum_size_m because values are missing")
        return

    tol = max(1e-6, expected_rum_size_m * 0.001)
    if abs(size_min - expected_rum_size_m) <= tol and abs(size_max - expected_rum_size_m) <= tol:
        ok(f"Footprint rum_size_m matches config: {expected_rum_size_m} m")
    else:
        warn(
            f"Footprint rum_size_m differs from config: "
            f"config={expected_rum_size_m}, footprint_min={size_min}, footprint_max={size_max}"
        )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    t_start = time.time()

    cfg = load_resolved_config(__file__)
    project_root = Path(cfg["_resolved"]["project_root"])

    prepared = cfg["prepared_inputs"]
    generated = cfg["generated_outputs"]
    expected = cfg["expected_counts"]

    points_path = resolve_path(project_root, prepared["points_geojson"])
    epochs_path = resolve_path(project_root, prepared["vertical_epoch_json"])
    footprints_path = resolve_path(project_root, generated["rum_footprints"])

    validation_report_path = Path(cfg["_resolved"]["pipeline_output_dir"]) / VALIDATION_REPORT_NAME

    expected_rum_count = expected.get("rum_count")
    expected_epoch_count = int(expected.get("epoch_count"))
    expected_rum_size_m = float(expected.get("grid_spacing_m_nominal"))

    section("Configuration")
    print(f"  Project root          : {project_root}")
    print(f"  Points input          : {points_path}")
    print(f"  Epoch input           : {epochs_path}")
    print(f"  Footprints input      : {footprints_path}")
    print(f"  Validation report     : {validation_report_path}")
    print(f"  Expected RUM count    : {expected_rum_count}")
    print(f"  Expected epoch count  : {expected_epoch_count}")
    print(f"  Expected RUM size     : {expected_rum_size_m} m")

    report: Dict[str, Any] = {
        "schema": "validation_report_v2_measurement_model",
        "status": "UNKNOWN",
        "warnings": WARNINGS,
        "errors": ERRORS,
        "inputs": {
            "points_geojson": prepared["points_geojson"],
            "vertical_epoch_json": prepared["vertical_epoch_json"],
            "rum_footprints": generated["rum_footprints"],
        },
        "summary": {},
    }

    try:
        section("Checking required files")
        for label, path in [
            ("points", points_path),
            ("epochs", epochs_path),
            ("footprints", footprints_path),
        ]:
            if not path.exists():
                raise FileNotFoundError(f"Missing {label} file: {path}")
            ok(f"{label} file exists: {path.stat().st_size / 1024 / 1024:.2f} MB")

        section("Loading products")
        points_payload = load_json(points_path)
        epochs_payload = load_json(epochs_path)
        footprints_payload = load_json(footprints_path)

        point_features = require_feature_collection(points_payload, "points")
        footprint_features = require_feature_collection(footprints_payload, "footprints")

        ok("JSON products loaded")

        section("Validating points")
        point_summary = validate_points(point_features)

        section("Validating footprints")
        footprint_summary = validate_footprints(footprints_payload, footprint_features)

        section("Validating epoch series")
        epoch_summary = validate_epochs(epochs_payload, expected_epoch_count)

        section("Cross-checking products")
        check_expected_count(point_summary["count"], expected_rum_count)
        check_rum_size(footprint_summary, expected_rum_size_m)
        check_id_sets(point_summary["ids"], footprint_summary["ids"], epoch_summary["ids"])

        if point_summary["count"] != footprint_summary["count"]:
            raise ValueError(
                f"Point/footprint count mismatch: "
                f"points={point_summary['count']}, footprints={footprint_summary['count']}"
            )

        if point_summary["count"] != epoch_summary["rum_count"]:
            raise ValueError(
                f"Point/epoch series count mismatch: "
                f"points={point_summary['count']}, epoch_series={epoch_summary['rum_count']}"
            )

        ok("Cross-product counts are consistent")

        report["summary"] = {
            "rum_count": point_summary["count"],
            "epoch_count": epoch_summary["epoch_count"],
            "first_epoch": epoch_summary["first_epoch"],
            "last_epoch": epoch_summary["last_epoch"],
            "point_bbox_wgs84": point_summary["bbox_wgs84"],
            "point_bbox_source": point_summary["bbox_source"],
            "footprint_bbox_wgs84": footprint_summary["metadata"].get("bbox_wgs84_footprints"),
            "footprint_bbox_source": footprint_summary["metadata"].get("bbox_source_footprints"),
            "grid": footprint_summary["metadata"].get("grid"),
            "measurement_min_mm": round(epoch_summary["measurement_min_mm"], ROUND_SUMMARY_DIGITS),
            "measurement_p02_mm": round(epoch_summary["measurement_p02_mm"], ROUND_SUMMARY_DIGITS),
            "measurement_p50_mm": round(epoch_summary["measurement_p50_mm"], ROUND_SUMMARY_DIGITS),
            "measurement_p98_mm": round(epoch_summary["measurement_p98_mm"], ROUND_SUMMARY_DIGITS),
            "measurement_max_mm": round(epoch_summary["measurement_max_mm"], ROUND_SUMMARY_DIGITS),
            "model_min_mm": round(epoch_summary["model_min_mm"], ROUND_SUMMARY_DIGITS),
            "model_p02_mm": round(epoch_summary["model_p02_mm"], ROUND_SUMMARY_DIGITS),
            "model_p50_mm": round(epoch_summary["model_p50_mm"], ROUND_SUMMARY_DIGITS),
            "model_p98_mm": round(epoch_summary["model_p98_mm"], ROUND_SUMMARY_DIGITS),
            "model_max_mm": round(epoch_summary["model_max_mm"], ROUND_SUMMARY_DIGITS),
            "sigma_min_mm": round(epoch_summary["sigma_min_mm"], ROUND_SUMMARY_DIGITS),
            "sigma_p02_mm": round(epoch_summary["sigma_p02_mm"], ROUND_SUMMARY_DIGITS),
            "sigma_p50_mm": round(epoch_summary["sigma_p50_mm"], ROUND_SUMMARY_DIGITS),
            "sigma_p98_mm": round(epoch_summary["sigma_p98_mm"], ROUND_SUMMARY_DIGITS),
            "sigma_max_mm": round(epoch_summary["sigma_max_mm"], ROUND_SUMMARY_DIGITS),
            "up_min_mm_yr": round(epoch_summary["up_min_mm_yr"], ROUND_SUMMARY_DIGITS),
            "up_max_mm_yr": round(epoch_summary["up_max_mm_yr"], ROUND_SUMMARY_DIGITS),
            "warnings_count": len(WARNINGS),
            "errors_count": len(ERRORS),
        }

        report["status"] = "OK" if not WARNINGS else "WARN"

    except Exception as exc:
        fail(str(exc))
        report["status"] = "FAIL"
        report["summary"]["warnings_count"] = len(WARNINGS)
        report["summary"]["errors_count"] = len(ERRORS)

    section("Writing validation report")
    write_json(validation_report_path, report)
    ok(f"Wrote validation report: {validation_report_path} ({validation_report_path.stat().st_size / 1024:.1f} KB)")

    elapsed = time.time() - t_start

    section("Summary")
    if report["status"] == "OK":
        ok(f"Step 05 validation passed in {elapsed:.2f} s")
    elif report["status"] == "WARN":
        warn(f"Step 05 validation passed with warnings in {elapsed:.2f} s")
    else:
        fail(f"Step 05 validation failed in {elapsed:.2f} s")

    if report.get("summary"):
        s = report["summary"]
        print(f"  RUM count              : {s.get('rum_count')}")
        print(f"  Epoch count            : {s.get('epoch_count')}")
        print(f"  MEASUREMENT range      : {s.get('measurement_min_mm')} to {s.get('measurement_max_mm')} mm")
        print(f"  MODEL range            : {s.get('model_min_mm')} to {s.get('model_max_mm')} mm")
        print(f"  Sigma range            : {s.get('sigma_min_mm')} to {s.get('sigma_max_mm')} mm")
        print(f"  Warnings               : {len(WARNINGS)}")
        print(f"  Errors                 : {len(ERRORS)}")

    if report["status"] == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
