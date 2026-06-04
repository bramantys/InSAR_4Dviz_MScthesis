#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
16_check_horizontal_uncertainty.py

InSAR4D RUM Viewer pipeline step 16.

Purpose
-------
Analyze horizontal velocity uncertainty/covariance and prepare safe uncertainty
parameters for Step 17 confidence ellipses.

Input
-----
  generated_outputs.horizontal_field
    _internal/data_pipeline/horizontal_field.json

Output
------
  generated_outputs.horizontal_uncertainty_check
    _internal/data_pipeline/horizontal_uncertainty_check.json

Critical note about units
-------------------------
This step assumes the source covariance terms are in:

  (mm/yr)^2

So:
  std_east_mm_yr  = sqrt(var_east)
  std_north_mm_yr = sqrt(var_north)

And confidence ellipse axes are:

  axis_mm_yr = sqrt(eigenvalue) * confidence_scale

There is intentionally NO division by 100 here.

If the original source data ever uses different units, the correct place to
handle that is config/source schema or an explicit covariance_unit_scale,
not hidden in the ellipse math.

Covariance matrix
-----------------
For each RUM:

  C = [[var_east,  covar_en],
       [covar_en, var_north]]

Eigenvalues are variance along major/minor axes.

Confidence scale
----------------
For a 2D Gaussian:
  1-sigma ellipse scale      = 1.0
  95% confidence scale       = sqrt(chi2.ppf(0.95, df=2)) ≈ 2.44774683

To avoid scipy dependency, this script uses the closed form for df=2:
  chi2_ppf(p, df=2) = -2 ln(1-p)
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

ROUND_DIGITS = 6
ROUND_ANGLE_DIGITS = 3

CONFIDENCE_PROBABILITY = 0.95

# Eigenvalues slightly below zero can occur from floating point noise.
NEGATIVE_EIGENVALUE_TOL = 1e-12

# Heuristic warning thresholds only.
SUSPICIOUS_MEDIAN_STD_TINY_MM_YR = 0.001
SUSPICIOUS_MEDIAN_STD_HUGE_MM_YR = 1000.0
SUSPICIOUS_AXIS_RATIO_HUGE = 1000.0

# SNR/signal-to-uncertainty denominator floor.
STD_FLOOR_MM_YR = 1e-12


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


def round_or_none(value: Optional[float], digits: int = ROUND_DIGITS) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


def chi2_scale_2d(probability: float) -> float:
    """
    For chi-square with df=2:
      CDF(x) = 1 - exp(-x/2)
      PPF(p) = -2 ln(1-p)
    """
    p = min(0.999999999, max(1e-12, float(probability)))
    return math.sqrt(-2.0 * math.log(1.0 - p))


# =============================================================================
# EIGEN / ELLIPSE MATH
# =============================================================================

def covariance_eigen_2x2(var_e: float, var_n: float, cov_en: float) -> Dict[str, float]:
    """
    Eigen decomposition for symmetric 2x2 covariance matrix:
      [[var_e, cov_en],
       [cov_en, var_n]]

    Coordinates:
      x = east
      y = north

    Returned angle is degrees counter-clockwise from east axis.
    """
    a = float(var_e)
    c = float(var_n)
    b = float(cov_en)

    trace_half = 0.5 * (a + c)
    diff_half = 0.5 * (a - c)
    root = math.sqrt(diff_half * diff_half + b * b)

    lam_major = trace_half + root
    lam_minor = trace_half - root

    if lam_minor < 0 and abs(lam_minor) <= NEGATIVE_EIGENVALUE_TOL:
        lam_minor = 0.0
    if lam_major < 0 and abs(lam_major) <= NEGATIVE_EIGENVALUE_TOL:
        lam_major = 0.0

    if lam_major < 0 or lam_minor < 0:
        raise ValueError(
            f"Negative covariance eigenvalue: major={lam_major}, minor={lam_minor}, "
            f"var_e={var_e}, var_n={var_n}, cov_en={cov_en}"
        )

    # Major-axis eigenvector. Stable branches for diagonal matrices.
    if abs(b) > 1e-18 or abs(lam_major - a) > 1e-18:
        vx = b
        vy = lam_major - a
    else:
        if a >= c:
            vx, vy = 1.0, 0.0
        else:
            vx, vy = 0.0, 1.0

    norm = math.hypot(vx, vy)
    if norm <= 0:
        vx, vy = 1.0, 0.0
    else:
        vx /= norm
        vy /= norm

    angle_east_ccw = math.degrees(math.atan2(vy, vx))
    # Normalize to [0, 180) because ellipse major axis has no arrow direction.
    angle_east_ccw = angle_east_ccw % 180.0

    # Convert to azimuth-like orientation clockwise from north, also [0, 180).
    # east_ccw 0   = east-west axis -> north_clockwise 90
    # east_ccw 90  = north-south axis -> north_clockwise 0
    angle_north_cw = (90.0 - angle_east_ccw) % 180.0

    return {
        "lambda_major": lam_major,
        "lambda_minor": lam_minor,
        "std_major": math.sqrt(lam_major),
        "std_minor": math.sqrt(lam_minor),
        "angle_major_deg_ccw_from_east": angle_east_ccw,
        "angle_major_deg_clockwise_from_north": angle_north_cw,
    }


def analyze_record(rec: Dict[str, Any], confidence_scale: float) -> Optional[Dict[str, Any]]:
    cov = rec.get("covariance") or {}
    if not cov.get("available"):
        return None

    var_e = safe_float(cov.get("var_east"))
    var_n = safe_float(cov.get("var_north"))
    cov_en = safe_float(cov.get("covar_en"))

    if var_e is None or var_n is None or cov_en is None:
        raise ValueError(f"Covariance marked available but missing fields for {rec.get('rum_id')}")

    if var_e < 0 or var_n < 0:
        raise ValueError(f"Negative variance for {rec.get('rum_id')}")

    det = var_e * var_n - cov_en * cov_en
    if det < -NEGATIVE_EIGENVALUE_TOL:
        raise ValueError(
            f"Invalid covariance determinant for {rec.get('rum_id')}: "
            f"det={det}, var_e={var_e}, var_n={var_n}, cov_en={cov_en}"
        )

    eig = covariance_eigen_2x2(var_e, var_n, cov_en)

    speed = safe_float(rec.get("speed_mm_yr"), 0.0) or 0.0
    std_major = eig["std_major"]
    std_minor = eig["std_minor"]

    axis_major = std_major * confidence_scale
    axis_minor = std_minor * confidence_scale

    # Signal-to-noise style indicators. Not a formal hypothesis test.
    snr_major = speed / max(std_major, STD_FLOOR_MM_YR)
    snr_minor = speed / max(std_minor, STD_FLOOR_MM_YR)
    axis_ratio = axis_major / max(axis_minor, STD_FLOOR_MM_YR)

    return {
        "rum_id": rec.get("rum_id"),
        "row_index": rec.get("row_index"),
        "grid_i": rec.get("grid_i"),
        "grid_j": rec.get("grid_j"),
        "lon_center": rec.get("lon_center"),
        "lat_center": rec.get("lat_center"),
        "east_mm_yr": rec.get("east_mm_yr"),
        "north_mm_yr": rec.get("north_mm_yr"),
        "speed_mm_yr": round(float(speed), ROUND_DIGITS),
        "var_east": round(float(var_e), ROUND_DIGITS),
        "var_north": round(float(var_n), ROUND_DIGITS),
        "covar_en": round(float(cov_en), ROUND_DIGITS),
        "determinant": round(float(det), ROUND_DIGITS),
        "std_east_mm_yr": round(math.sqrt(max(0.0, var_e)), ROUND_DIGITS),
        "std_north_mm_yr": round(math.sqrt(max(0.0, var_n)), ROUND_DIGITS),
        "std_major_1sigma_mm_yr": round(std_major, ROUND_DIGITS),
        "std_minor_1sigma_mm_yr": round(std_minor, ROUND_DIGITS),
        "ellipse_major_mm_yr": round(axis_major, ROUND_DIGITS),
        "ellipse_minor_mm_yr": round(axis_minor, ROUND_DIGITS),
        "ellipse_angle_deg_ccw_from_east": round(eig["angle_major_deg_ccw_from_east"], ROUND_ANGLE_DIGITS),
        "ellipse_angle_deg_clockwise_from_north": round(eig["angle_major_deg_clockwise_from_north"], ROUND_ANGLE_DIGITS),
        "axis_ratio": round(axis_ratio, ROUND_DIGITS),
        "speed_over_std_major": round(snr_major, ROUND_DIGITS),
        "speed_over_std_minor": round(snr_minor, ROUND_DIGITS),
    }


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def analyze_uncertainty(records: List[Dict[str, Any]], confidence_scale: float) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    unavailable_count = 0

    for rec in records:
        result = analyze_record(rec, confidence_scale)
        if result is None:
            unavailable_count += 1
        else:
            results.append(result)

    if not results:
        raise ValueError("No records have available horizontal covariance")

    std_east = [r["std_east_mm_yr"] for r in results]
    std_north = [r["std_north_mm_yr"] for r in results]
    std_major = [r["std_major_1sigma_mm_yr"] for r in results]
    std_minor = [r["std_minor_1sigma_mm_yr"] for r in results]
    ell_major = [r["ellipse_major_mm_yr"] for r in results]
    ell_minor = [r["ellipse_minor_mm_yr"] for r in results]
    axis_ratio = [r["axis_ratio"] for r in results]
    speed_over_major = [r["speed_over_std_major"] for r in results]
    determinants = [r["determinant"] for r in results]

    std_major_p50 = percentile(std_major, 50)
    std_minor_p50 = percentile(std_minor, 50)
    axis_ratio_p98 = percentile(axis_ratio, 98)

    if std_major_p50 is not None and std_major_p50 < SUSPICIOUS_MEDIAN_STD_TINY_MM_YR:
        warn(
            "Median horizontal major-axis standard deviation is extremely tiny "
            f"({std_major_p50:.6g} mm/yr). Check covariance units; do not hide-scale with /100."
        )

    if std_major_p50 is not None and std_major_p50 > SUSPICIOUS_MEDIAN_STD_HUGE_MM_YR:
        warn(
            "Median horizontal major-axis standard deviation is extremely large "
            f"({std_major_p50:.6g} mm/yr). Check covariance units."
        )

    if axis_ratio_p98 is not None and axis_ratio_p98 > SUSPICIOUS_AXIS_RATIO_HUGE:
        warn(
            "Very elongated uncertainty ellipses detected "
            f"(axis ratio p98={axis_ratio_p98:.3g}). Check covariance quality."
        )

    summary = {
        "record_count": len(records),
        "available_covariance_count": len(results),
        "unavailable_covariance_count": unavailable_count,
        "confidence_probability": CONFIDENCE_PROBABILITY,
        "confidence_scale": confidence_scale,
        "unit_assumption": "covariance in (mm/yr)^2; ellipse axes in mm/yr; no hidden /100 scaling",
        "std_east_min_mm_yr": min(std_east),
        "std_east_p50_mm_yr": percentile(std_east, 50),
        "std_east_p98_mm_yr": percentile(std_east, 98),
        "std_east_max_mm_yr": max(std_east),
        "std_north_min_mm_yr": min(std_north),
        "std_north_p50_mm_yr": percentile(std_north, 50),
        "std_north_p98_mm_yr": percentile(std_north, 98),
        "std_north_max_mm_yr": max(std_north),
        "std_major_min_1sigma_mm_yr": min(std_major),
        "std_major_p50_1sigma_mm_yr": percentile(std_major, 50),
        "std_major_p98_1sigma_mm_yr": percentile(std_major, 98),
        "std_major_max_1sigma_mm_yr": max(std_major),
        "std_minor_min_1sigma_mm_yr": min(std_minor),
        "std_minor_p50_1sigma_mm_yr": percentile(std_minor, 50),
        "std_minor_p98_1sigma_mm_yr": percentile(std_minor, 98),
        "std_minor_max_1sigma_mm_yr": max(std_minor),
        "ellipse_major_p50_mm_yr": percentile(ell_major, 50),
        "ellipse_major_p98_mm_yr": percentile(ell_major, 98),
        "ellipse_major_max_mm_yr": max(ell_major),
        "ellipse_minor_p50_mm_yr": percentile(ell_minor, 50),
        "ellipse_minor_p98_mm_yr": percentile(ell_minor, 98),
        "ellipse_minor_max_mm_yr": max(ell_minor),
        "axis_ratio_p50": percentile(axis_ratio, 50),
        "axis_ratio_p98": percentile(axis_ratio, 98),
        "axis_ratio_max": max(axis_ratio),
        "speed_over_std_major_p50": percentile(speed_over_major, 50),
        "speed_over_std_major_p98": percentile(speed_over_major, 98),
        "determinant_min": min(determinants),
        "determinant_p50": percentile(determinants, 50),
        "determinant_max": max(determinants),
        "warnings_count": len(WARNINGS),
    }

    return results, summary


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    t_start = time.time()

    cfg = load_resolved_config(__file__)
    project_root = Path(cfg["_resolved"]["project_root"])

    generated = cfg["generated_outputs"]

    input_path = resolve_path(project_root, generated["horizontal_field"])
    output_path = resolve_path(project_root, generated["horizontal_uncertainty_check"])

    confidence_scale = chi2_scale_2d(CONFIDENCE_PROBABILITY)

    section("Configuration")
    print(f"  Project root           : {project_root}")
    print(f"  Horizontal input       : {input_path}")
    print(f"  Uncertainty output     : {output_path}")
    print(f"  Confidence probability : {CONFIDENCE_PROBABILITY}")
    print(f"  Confidence scale       : {confidence_scale:.8f}")
    print("  Unit assumption        : covariance=(mm/yr)^2, axes=mm/yr, no /100 scaling")

    section("Loading horizontal field")
    hfield = load_json(input_path)
    records = hfield.get("records")

    if not isinstance(records, list) or not records:
        raise ValueError("horizontal_field.json has no records")

    ok(f"Loaded horizontal field: {len(records)} records")

    section("Analyzing covariance / ellipse parameters")
    results, summary = analyze_uncertainty(records, confidence_scale)

    ok(f"Computed ellipse parameters for {len(results)} records")
    ok("Covariance eigenvalue analysis complete")

    section("Writing uncertainty check product")
    payload = {
        "metadata": {
            "schema": "horizontal_uncertainty_check_v1",
            "source_horizontal_field": generated["horizontal_field"],
            "purpose": "validated covariance/eigen parameters for Step 17 B3DM confidence ellipses",
            "unit_assumption": summary["unit_assumption"],
            "confidence_probability": CONFIDENCE_PROBABILITY,
            "confidence_scale": confidence_scale,
            "angle_convention": {
                "ellipse_angle_deg_ccw_from_east": "mathematical angle in EN plane, [0, 180)",
                "ellipse_angle_deg_clockwise_from_north": "azimuth-style orientation, [0, 180)",
            },
            "old_pipeline_warning": "Do not copy hidden /100 scaling from old pipeline 19; this product keeps covariance units explicit.",
            "summary": summary,
            "warnings": WARNINGS,
        },
        "records": results,
    }

    write_json(output_path, payload)

    elapsed = time.time() - t_start

    ok(f"Wrote horizontal uncertainty check: {output_path} ({output_path.stat().st_size / 1024 / 1024:.2f} MB)")

    section("Summary")
    ok(f"Step 16 complete in {elapsed:.2f} s")
    print(f"  Records with covariance: {summary['available_covariance_count']} / {summary['record_count']}")
    print(f"  1σ major p50/p98       : {summary['std_major_p50_1sigma_mm_yr']:.6f} / {summary['std_major_p98_1sigma_mm_yr']:.6f} mm/yr")
    print(f"  95% ellipse major p50  : {summary['ellipse_major_p50_mm_yr']:.6f} mm/yr")
    print(f"  Axis ratio p50/p98     : {summary['axis_ratio_p50']:.6f} / {summary['axis_ratio_p98']:.6f}")
    print(f"  Speed/std-major p50    : {summary['speed_over_std_major_p50']:.6f}")
    print(f"  Warnings               : {len(WARNINGS)}")


if __name__ == "__main__":
    main()
