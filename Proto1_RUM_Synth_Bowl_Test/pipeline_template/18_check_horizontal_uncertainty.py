#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
18_check_horizontal_uncertainty.py

Generic RUM-based InSAR template diagnostic step.

Purpose
-------
Inspect horizontal uncertainty/covariance fields used by the particle shimmer
visualization.

Reads:
  config.generated_outputs.horizontal_field

Writes:
  Data/horizontal_uncertainty_check.json

Core quantities:
  - sigma_east_mm_yr
  - sigma_north_mm_yr
  - covariance PSD validity
  - principal horizontal uncertainty axes
  - approximate direction uncertainty sigma_theta_deg

Interpretation
--------------
The viewer keeps the mean particle path unchanged. Uncertainty should only
affect render-side trail shimmer/diffuseness/wobble.

For a velocity vector v = [east, north], and covariance matrix C, the
approximate angular variance is:

  var(theta) = u_perp.T C u_perp / speed^2

where u_perp is perpendicular to the velocity direction.

A speed floor is applied so nearly-zero horizontal velocities do not produce
unbounded angular uncertainty.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def summarize(values: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return {
            "count": 0,
            "min": float("nan"),
            "p05": float("nan"),
            "p25": float("nan"),
            "median": float("nan"),
            "mean": float("nan"),
            "p75": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "max": float("nan"),
        }

    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "mean": float(np.mean(arr)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def print_summary(name: str, values: np.ndarray, unit: str = "") -> Dict[str, float]:
    s = summarize(values)
    suffix = f" {unit}" if unit else ""
    print(
        f"  {name:<24s} "
        f"n={s['count']:5d} "
        f"min={s['min']:9.3f} p25={s['p25']:9.3f} "
        f"med={s['median']:9.3f} p75={s['p75']:9.3f} "
        f"p90={s['p90']:9.3f} p95={s['p95']:9.3f} "
        f"max={s['max']:9.3f}{suffix}"
    )
    return s


def principal_axes_2x2(var_e: float, var_n: float, cov_en: float) -> Dict[str, float]:
    c = np.array([[var_e, cov_en], [cov_en, var_n]], dtype=np.float64)
    vals, vecs = np.linalg.eigh(c)

    # ascending eigenvalues from eigh
    lam_minor = max(float(vals[0]), 0.0)
    lam_major = max(float(vals[1]), 0.0)

    v_major = vecs[:, 1]
    angle_deg = math.degrees(math.atan2(float(v_major[1]), float(v_major[0])))

    # Keep angle in [0,180), because ellipse axis has no direction sign.
    angle_deg = angle_deg % 180.0

    return {
        "sigma_major_mm_yr": math.sqrt(lam_major),
        "sigma_minor_mm_yr": math.sqrt(lam_minor),
        "major_axis_angle_deg_from_east": angle_deg,
    }


def angular_uncertainty_deg(
    east: float,
    north: float,
    var_e: float,
    var_n: float,
    cov_en: float,
    speed_floor: float,
) -> float:
    speed = math.sqrt(east * east + north * north)
    speed_eff = max(speed, speed_floor)

    if speed <= 1e-12:
        # Direction is physically undefined. Use eastward proxy direction;
        # speed floor prevents explosive values.
        ue = 1.0
        un = 0.0
    else:
        ue = east / speed
        un = north / speed

    # Perpendicular unit vector to mean velocity direction.
    pe = -un
    pn = ue

    var_perp = pe * pe * var_e + pn * pn * var_n + 2.0 * pe * pn * cov_en
    var_perp = max(var_perp, 0.0)

    sigma_theta_rad = math.sqrt(var_perp) / speed_eff
    return math.degrees(sigma_theta_rad)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    t_start = time.time()

    cfg = load_config()
    generated = cfg.get("generated_outputs", {})
    hunc_cfg = cfg.get("horizontal_uncertainty", {})

    field_path = resolve_project_path(
        generated.get("horizontal_field", "Data/horizontal_field.json")
    )
    out_path = resolve_project_path(
        generated.get("horizontal_uncertainty_check", "Data/horizontal_uncertainty_check.json")
    )

    speed_floor_mm_yr = float(hunc_cfg.get("speed_floor_mm_yr", 0.50))
    preferred_shimmer_strength = float(hunc_cfg.get("preferred_shimmer_strength", 0.50))
    max_recommended_wobble_deg = float(hunc_cfg.get("max_recommended_wobble_deg", 35.0))

    section("Configuration")
    print(f"  Project root             : {PROJECT_DIR}")
    print(f"  Horizontal field          : {field_path}")
    print(f"  Output report             : {out_path}")
    print(f"  Speed floor               : {speed_floor_mm_yr:.3f} mm/yr")
    print(f"  Preferred shimmer strength: {preferred_shimmer_strength:.2f}")
    print(f"  Max recommended wobble    : {max_recommended_wobble_deg:.1f}°")

    section("Loading horizontal field")
    if not field_path.exists():
        raise FileNotFoundError(f"Missing horizontal field: {field_path}")

    field = load_json(field_path)
    cells = field.get("cells", [])

    if not cells:
        raise ValueError("horizontal_field.json has no cells")

    ok(f"Loaded cells: {len(cells)}")

    section("Extracting velocity and covariance arrays")

    east = np.array([as_float(c.get("east_mm_yr"), np.nan) for c in cells], dtype=np.float64)
    north = np.array([as_float(c.get("north_mm_yr"), np.nan) for c in cells], dtype=np.float64)
    speed = np.array([as_float(c.get("speed_mm_yr"), np.nan) for c in cells], dtype=np.float64)

    var_e = np.array([as_float(c.get("var_east"), np.nan) for c in cells], dtype=np.float64)
    var_n = np.array([as_float(c.get("var_north"), np.nan) for c in cells], dtype=np.float64)
    cov_en = np.array([as_float(c.get("covar_en"), np.nan) for c in cells], dtype=np.float64)

    finite = (
        np.isfinite(east)
        & np.isfinite(north)
        & np.isfinite(speed)
        & np.isfinite(var_e)
        & np.isfinite(var_n)
        & np.isfinite(cov_en)
    )

    ok(f"Finite velocity + EN covariance records: {int(np.sum(finite))}/{len(cells)}")

    if not np.any(finite):
        raise RuntimeError("No finite horizontal covariance records available")

    east_f = east[finite]
    north_f = north[finite]
    speed_f = speed[finite]
    var_e_f = var_e[finite]
    var_n_f = var_n[finite]
    cov_en_f = cov_en[finite]

    section("Covariance matrix validity")

    det = var_e_f * var_n_f - cov_en_f * cov_en_f
    trace = var_e_f + var_n_f

    negative_var = (var_e_f < 0.0) | (var_n_f < 0.0)
    non_psd = det < -1e-9

    if np.any(negative_var):
        warn(f"Negative variance records: {int(np.sum(negative_var))}")
    else:
        ok("No negative variances")

    if np.any(non_psd):
        warn(f"Non-PSD records: {int(np.sum(non_psd))}")
    else:
        ok("All covariance matrices PSD within tolerance")

    print_summary("determinant", det, "(mm/yr)^4")
    print_summary("trace", trace, "(mm/yr)^2")

    section("Sigma component distributions")

    sigma_e = np.sqrt(np.maximum(var_e_f, 0.0))
    sigma_n = np.sqrt(np.maximum(var_n_f, 0.0))

    stats_sigma_e = print_summary("sigma_east", sigma_e, "mm/yr")
    stats_sigma_n = print_summary("sigma_north", sigma_n, "mm/yr")
    stats_speed = print_summary("speed", speed_f, "mm/yr")

    section("Principal uncertainty ellipse axes")

    sigma_major = np.zeros_like(speed_f)
    sigma_minor = np.zeros_like(speed_f)
    axis_angle = np.zeros_like(speed_f)

    for i in range(len(speed_f)):
        axes = principal_axes_2x2(
            var_e=float(var_e_f[i]),
            var_n=float(var_n_f[i]),
            cov_en=float(cov_en_f[i]),
        )
        sigma_major[i] = axes["sigma_major_mm_yr"]
        sigma_minor[i] = axes["sigma_minor_mm_yr"]
        axis_angle[i] = axes["major_axis_angle_deg_from_east"]

    stats_major = print_summary("sigma_major", sigma_major, "mm/yr")
    stats_minor = print_summary("sigma_minor", sigma_minor, "mm/yr")
    stats_axis = print_summary("major axis angle", axis_angle, "deg from east")

    section("Direction uncertainty for particle shimmer")

    sigma_theta = np.zeros_like(speed_f)

    for i in range(len(speed_f)):
        sigma_theta[i] = angular_uncertainty_deg(
            east=float(east_f[i]),
            north=float(north_f[i]),
            var_e=float(var_e_f[i]),
            var_n=float(var_n_f[i]),
            cov_en=float(cov_en_f[i]),
            speed_floor=speed_floor_mm_yr,
        )

    stats_theta = print_summary("sigma_theta", sigma_theta, "deg")

    low_speed_mask = speed_f < speed_floor_mm_yr
    high_unc_mask = sigma_theta > max_recommended_wobble_deg

    print(f"  Cells below speed floor: {int(np.sum(low_speed_mask))}/{len(speed_f)}")
    print(f"  Cells above {max_recommended_wobble_deg:.1f}° sigma_theta: {int(np.sum(high_unc_mask))}/{len(speed_f)}")

    if stats_theta["p95"] <= max_recommended_wobble_deg:
        ok(f"P95 sigma_theta = {stats_theta['p95']:.2f}° ≤ {max_recommended_wobble_deg:.1f}°")
    else:
        warn(f"P95 sigma_theta = {stats_theta['p95']:.2f}° > {max_recommended_wobble_deg:.1f}°")

    section("Suggested viewer interpretation")
    print("  Mean particle path:")
    print("    Keep unchanged: use east_mm_yr and north_mm_yr only.")
    print()
    print("  Uncertainty channel:")
    print("    Use covariance only for render-side shimmer/diffuseness.")
    print(f"    Suggested default uncertainty shimmer strength: {preferred_shimmer_strength:.2f}")
    print(f"    Suggested speed floor for angle uncertainty: {speed_floor_mm_yr:.2f} mm/yr")
    print()
    print("  Useful reference:")
    print(f"    sigma_theta P75 ≈ {stats_theta['p75']:.1f}°")
    print(f"    sigma_theta P90 ≈ {stats_theta['p90']:.1f}°")
    print(f"    sigma_theta P95 ≈ {stats_theta['p95']:.1f}°")

    section("Outlier uncertainty cells")

    original_indices = np.where(finite)[0]
    order = np.argsort(-sigma_theta)

    for rank, local_idx in enumerate(order[:12], start=1):
        global_idx = int(original_indices[int(local_idx)])
        cell = cells[global_idx]
        print(
            f"  #{rank:02d} {str(cell.get('rum_id')):<24s} "
            f"grid=({int(cell.get('grid_i')):>4},{int(cell.get('grid_j')):>4}) "
            f"speed={speed_f[local_idx]:7.3f} mm/yr "
            f"sigma_theta={sigma_theta[local_idx]:7.2f}° "
            f"sigma_major={sigma_major[local_idx]:7.3f} "
            f"sigma_minor={sigma_minor[local_idx]:7.3f}"
        )

    section("Writing report")

    report = {
        "schema_version": "horizontal_uncertainty_check_v1",
        "source_horizontal_field": str(field_path),
        "speed_floor_mm_yr": speed_floor_mm_yr,
        "preferred_shimmer_strength": preferred_shimmer_strength,
        "max_recommended_wobble_deg": max_recommended_wobble_deg,
        "counts": {
            "cells_total": len(cells),
            "finite_velocity_covariance": int(np.sum(finite)),
            "negative_variance_records": int(np.sum(negative_var)),
            "non_psd_records": int(np.sum(non_psd)),
            "below_speed_floor": int(np.sum(low_speed_mask)),
            "above_max_recommended_wobble": int(np.sum(high_unc_mask)),
        },
        "stats": {
            "speed_mm_yr": stats_speed,
            "sigma_east_mm_yr": stats_sigma_e,
            "sigma_north_mm_yr": stats_sigma_n,
            "sigma_major_mm_yr": stats_major,
            "sigma_minor_mm_yr": stats_minor,
            "major_axis_angle_deg_from_east": stats_axis,
            "sigma_theta_deg": stats_theta,
        },
        "viewer_recommendation": {
            "mean_particle_path": "Use east_mm_yr and north_mm_yr unchanged.",
            "uncertainty_visualization": "Use covariance only for render-side shimmer/diffuseness/wobble.",
            "default_uncertainty_strength": preferred_shimmer_strength,
            "speed_floor_mm_yr": speed_floor_mm_yr,
        },
        "created_by": "18_check_horizontal_uncertainty.py",
        "created_unix": int(time.time()),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    ok(f"Written: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")

    section("SUMMARY")
    ok("Step 18 complete — horizontal uncertainty diagnostics finished")
    ok("Template rebuild pipeline is now complete through vertical + horizontal products")
    print(f"  Elapsed: {time.time() - t_start:.2f}s")
    print()


if __name__ == "__main__":
    main()
