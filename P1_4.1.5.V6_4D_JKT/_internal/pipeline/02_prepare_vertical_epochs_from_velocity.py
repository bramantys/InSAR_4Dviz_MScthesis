#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_prepare_vertical_epochs_from_velocity.py

InSAR4D RUM Viewer pipeline step 02.

Purpose
-------
Create synthetic vertical epoch products from a velocity-only RUM product.

The source RUM product is velocity-only:
  - east  [mm/yr]
  - north [mm/yr]
  - up    [mm/yr]

This step creates two explicit vertical roles:

  MEASUREMENT
    Stored as series[rum_id]["measurement_mm"].
    Intended for trendline / popup / labels.
    Controlled by:
      epoch_generation.vertical_measurement_behavior = linear | sinusoidal
      epoch_generation.vertical_measurement_noise_sigma_mm

  MODEL
    Stored as series[rum_id]["model_mm"].
    Intended for RUM height / choropleth / height texture / walls / blankies.
    Controlled by:
      epoch_generation.vertical_model = linear | sinusoidal

Sigma/uncertainty:
  Stored as series[rum_id]["sigma_mm"].
  This is only the base propagated sigma at Step 02.
  Step 04 later enhances/replaces sigma_mm using uncertainty_quality.

Inputs
------
  prepared_inputs.points_geojson
    GeoJSON points created by Step 01.

Outputs
-------
  prepared_inputs.vertical_epoch_json_without_enhanced_sigma
    Base synthetic epoch product.

  prepared_inputs.vertical_epoch_json
    Same as base at this stage, so downstream steps can run before Step 04.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import random
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from pipeline_config import load_resolved_config, resolve_path


# =============================================================================
# ADVANCED SETTINGS
# =============================================================================
# Normal users should edit config/project_config.json, not this script.

ROUND_VERTICAL_DIGITS = 4
ROUND_SIGMA_DIGITS = 4
ROUND_DECIMAL_YEAR_DIGITS = 8

# If var_up is not present or unusable, this base sigma velocity is used.
# Step 04 later replaces/enhances sigma using the selected quality preset.
FALLBACK_SIGMA_UP_MM_YR = 0.0

# The displacement product includes the last configured date. If interval_days
# does not land exactly on the end date, the end date is appended.
INCLUDE_EXACT_END_DATE = True


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
# TIME HELPERS
# =============================================================================

def parse_date(value: str) -> dt.date:
    return dt.date.fromisoformat(str(value)[:10])


def build_epoch_dates(start_date: str, end_date: str, interval_days: int) -> List[dt.date]:
    start = parse_date(start_date)
    end = parse_date(end_date)

    if end < start:
        raise ValueError(f"end_date {end} is before start_date {start}")
    if interval_days <= 0:
        raise ValueError("interval_days must be positive")

    epochs: List[dt.date] = []
    current = start
    step = dt.timedelta(days=int(interval_days))

    while current <= end:
        epochs.append(current)
        current += step

    if INCLUDE_EXACT_END_DATE and epochs[-1] != end:
        epochs.append(end)

    return epochs


def decimal_year(d: dt.date) -> float:
    year_start = dt.date(d.year, 1, 1)
    next_year_start = dt.date(d.year + 1, 1, 1)
    return d.year + (d - year_start).days / (next_year_start - year_start).days


def unix_time(d: dt.date) -> float:
    return dt.datetime(d.year, d.month, d.day, tzinfo=dt.timezone.utc).timestamp()


def elapsed_years_from_start(epoch_dates: List[dt.date]) -> List[float]:
    if not epoch_dates:
        return []
    t0 = epoch_dates[0]
    return [(d - t0).days / 365.25 for d in epoch_dates]


def elapsed_days_from_start(epoch_dates: List[dt.date]) -> List[float]:
    if not epoch_dates:
        return []
    t0 = epoch_dates[0]
    return [float((d - t0).days) for d in epoch_dates]


# =============================================================================
# DATA HELPERS
# =============================================================================

def safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        if value is None:
            return fallback
        out = float(value)
        if not math.isfinite(out):
            return fallback
        return out
    except Exception:
        return fallback


def safe_bool(value: Any, fallback: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return fallback
    if isinstance(value, (int, float)):
        return bool(value)

    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return fallback


def positive_sqrt(value: Any, fallback: float = FALLBACK_SIGMA_UP_MM_YR) -> float:
    v = safe_float(value, fallback=float("nan"))
    if not math.isfinite(v) or v < 0:
        return fallback
    return math.sqrt(v)


def load_geojson(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing points GeoJSON: {path}")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("type") != "FeatureCollection":
        raise ValueError(f"Expected GeoJSON FeatureCollection: {path}")
    return payload


def get_features(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    features = payload.get("features", [])
    if not isinstance(features, list) or not features:
        raise ValueError("GeoJSON has no features")
    return features


def get_rum_id(props: Dict[str, Any], index: int) -> str:
    rid = props.get("rum_id")
    if rid is None or str(rid).strip() == "":
        return f"RUM_{index + 1:06d}"
    return str(rid)


def stable_noise_seed(base_seed: int, rum_id: str) -> int:
    """Stable per-RUM seed, independent of Python's randomized hash()."""
    digest = hashlib.sha256(f"{base_seed}:{rum_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


# =============================================================================
# VERTICAL SERIES MODELS
# =============================================================================

def linear_displacement(up_mm_yr: float, elapsed_years: List[float]) -> List[float]:
    return [float(up_mm_yr) * t for t in elapsed_years]


def sinusoidal_displacement(
    up_mm_yr: float,
    elapsed_years: List[float],
    elapsed_days: List[float],
    amplitude_mm: float,
    period_days: float,
    phase_days: float,
    start_at_zero: bool,
) -> List[float]:
    if period_days <= 0:
        raise ValueError("sinusoidal period_days must be positive")

    base = linear_displacement(up_mm_yr, elapsed_years)

    omega = 2.0 * math.pi / period_days
    theta0 = omega * (0.0 - phase_days)
    s0 = math.sin(theta0)

    out: List[float] = []
    for b, t_days in zip(base, elapsed_days):
        seasonal = amplitude_mm * math.sin(omega * (t_days - phase_days))
        if start_at_zero:
            seasonal -= amplitude_mm * s0
        out.append(b + seasonal)

    return out


def build_behavior_series(
    behavior: str,
    behavior_cfg: Dict[str, Any],
    up_mm_yr: float,
    elapsed_years: List[float],
    elapsed_days: List[float],
    default_start_at_zero: bool,
) -> List[float]:
    behavior = str(behavior).strip().lower()

    if behavior == "linear":
        return linear_displacement(up_mm_yr, elapsed_years)

    if behavior == "sinusoidal":
        start_at_zero = safe_bool(
            behavior_cfg.get("start_displacement_at_zero"),
            fallback=default_start_at_zero,
        )
        return sinusoidal_displacement(
            up_mm_yr=up_mm_yr,
            elapsed_years=elapsed_years,
            elapsed_days=elapsed_days,
            amplitude_mm=safe_float(behavior_cfg.get("amplitude_mm"), 5.0),
            period_days=safe_float(behavior_cfg.get("period_days"), 365.25),
            phase_days=safe_float(behavior_cfg.get("phase_days"), 45.0),
            start_at_zero=start_at_zero,
        )

    raise ValueError(f"Unknown vertical behavior/model: {behavior!r}")


def add_measurement_noise(
    measurement_mm: List[float],
    noise_sigma_mm: float,
    outlier_probability: float,
    outlier_sigma_mm: float,
    keep_first_epoch_zero: bool,
    rng: random.Random,
) -> Tuple[List[float], Dict[str, int]]:
    """
    Add optional synthetic observation-like noise to MEASUREMENT only.

    MODEL must stay clean and is never passed through this function.
    """
    base_noise = max(0.0, float(noise_sigma_mm))
    outlier_prob = max(0.0, min(1.0, float(outlier_probability)))
    outlier_sigma = max(0.0, float(outlier_sigma_mm))

    out: List[float] = []
    stats = {
        "epochs_with_gaussian_noise": 0,
        "epochs_with_outliers": 0,
        "first_epoch_locked_zero": 0,
    }

    for i, value in enumerate(measurement_mm):
        if i == 0 and keep_first_epoch_zero:
            out.append(0.0)
            stats["first_epoch_locked_zero"] += 1
            continue

        noisy = float(value)

        if base_noise > 0.0:
            noisy += rng.gauss(0.0, base_noise)
            stats["epochs_with_gaussian_noise"] += 1

        if outlier_prob > 0.0 and outlier_sigma > 0.0 and rng.random() < outlier_prob:
            noisy += rng.gauss(0.0, outlier_sigma)
            stats["epochs_with_outliers"] += 1

        out.append(noisy)

    return out, stats


def build_sigma_series(sigma_up_mm_yr: float, elapsed_years: List[float]) -> List[float]:
    """
    Base propagation for a velocity-only product.

    If up velocity uncertainty is expressed as mm/yr, displacement uncertainty
    grows approximately with elapsed time:
      sigma_displacement_mm = sigma_velocity_mm_yr * elapsed_years

    Step 04 later enhances/replaces this for visual uncertainty/hatch behavior.
    """
    sigma = max(0.0, float(sigma_up_mm_yr))
    return [sigma * t for t in elapsed_years]


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    t_start = time.time()

    cfg = load_resolved_config(__file__)
    project_root = Path(cfg["_resolved"]["project_root"])

    prepared = cfg["prepared_inputs"]
    source_schema = cfg.get("source_schema", {})
    source_inputs = cfg.get("source_inputs", {})
    epoch_cfg = cfg["epoch_generation"]

    points_path = resolve_path(project_root, prepared["points_geojson"])
    output_base_path = resolve_path(project_root, prepared["vertical_epoch_json_without_enhanced_sigma"])
    output_sigma_path = resolve_path(project_root, prepared["vertical_epoch_json"])

    velocity_fields = source_schema.get("velocity_fields") or source_inputs.get("source_velocity_fields", {})
    uncertainty_fields = source_schema.get("uncertainty_fields") or source_inputs.get("source_variance_fields", {})

    up_col = velocity_fields.get("up", "up")
    var_up_col = uncertainty_fields.get("var_up", "var_up")

    start_date = epoch_cfg["default_start_date"]
    end_date = epoch_cfg["default_end_date"]
    interval_days = int(epoch_cfg["default_interval_days"])

    measurement_behavior = str(epoch_cfg["vertical_measurement_behavior"]).lower()
    measurement_cfg = epoch_cfg.get("vertical_measurement") or {}
    measurement_noise_label = str(epoch_cfg.get("vertical_measurement_noise", "unknown"))
    measurement_noise_sigma_mm = safe_float(
        epoch_cfg.get("measurement_noise_sigma_mm", epoch_cfg.get("vertical_measurement_noise_sigma_mm")),
        0.0,
    )
    measurement_outlier_probability = safe_float(epoch_cfg.get("measurement_outlier_probability"), 0.0)
    measurement_outlier_sigma_mm = safe_float(epoch_cfg.get("measurement_outlier_sigma_mm"), 0.0)
    measurement_noise_keep_first_epoch_zero = safe_bool(
        epoch_cfg.get("measurement_noise_keep_first_epoch_zero"),
        True,
    )

    vertical_model = str(epoch_cfg["vertical_model"]).lower()
    model_cfg = epoch_cfg.get("vertical_model_config") or {}

    start_at_zero = safe_bool(epoch_cfg.get("start_displacement_at_zero"), True)
    round_digits = int(epoch_cfg.get("round_digits", ROUND_VERTICAL_DIGITS))
    random_seed = int(epoch_cfg.get("random_seed", 42))

    section("Configuration")
    print(f"  Project root             : {project_root}")
    print(f"  Points GeoJSON           : {points_path}")
    print(f"  Base epoch output        : {output_base_path}")
    print(f"  Working epoch output     : {output_sigma_path}")
    print(f"  Up velocity field        : {up_col}")
    print(f"  Up variance field        : {var_up_col}")
    print(f"  Date range               : {start_date} → {end_date}")
    print(f"  Interval                 : {interval_days} days")
    print(f"  MEASUREMENT behavior     : {measurement_behavior}")
    print(f"  MEASUREMENT noise        : {measurement_noise_label} ({measurement_noise_sigma_mm} mm)")
    print(f"  MEASUREMENT outlier p    : {measurement_outlier_probability}")
    print(f"  MEASUREMENT outlier σ    : {measurement_outlier_sigma_mm} mm")
    print(f"  MODEL behavior           : {vertical_model}")
    print(f"  Random seed              : {random_seed}")

    if measurement_behavior not in {"linear", "sinusoidal"}:
        raise ValueError("vertical_measurement_behavior must be 'linear' or 'sinusoidal'")
    if vertical_model not in {"linear", "sinusoidal"}:
        raise ValueError("vertical_model must be 'linear' or 'sinusoidal'")

    if measurement_behavior == "sinusoidal":
        print(f"  MEASUREMENT sinus amp    : {safe_float(measurement_cfg.get('amplitude_mm'), 5.0)} mm")
        print(f"  MEASUREMENT sinus period : {safe_float(measurement_cfg.get('period_days'), 365.25)} days")
        print(f"  MEASUREMENT sinus phase  : {safe_float(measurement_cfg.get('phase_days'), 45.0)} days")

    if vertical_model == "sinusoidal":
        print(f"  MODEL sinus amp          : {safe_float(model_cfg.get('amplitude_mm'), 5.0)} mm")
        print(f"  MODEL sinus period       : {safe_float(model_cfg.get('period_days'), 365.25)} days")
        print(f"  MODEL sinus phase        : {safe_float(model_cfg.get('phase_days'), 45.0)} days")

    section("Building time axis")
    epoch_dates = build_epoch_dates(start_date, end_date, interval_days)
    epochs = [d.isoformat() for d in epoch_dates]
    epoch_decimal_year = [round(decimal_year(d), ROUND_DECIMAL_YEAR_DIGITS) for d in epoch_dates]
    epoch_unix = [unix_time(d) for d in epoch_dates]
    elapsed_years = elapsed_years_from_start(epoch_dates)
    elapsed_days = elapsed_days_from_start(epoch_dates)

    ok(f"Built {len(epochs)} epochs")
    print(f"  First epoch              : {epochs[0]}")
    print(f"  Last epoch               : {epochs[-1]}")
    if len(epochs) > 1:
        print(f"  First interval           : {(epoch_dates[1] - epoch_dates[0]).days} days")

    section("Loading RUM point product")
    point_data = load_geojson(points_path)
    features = get_features(point_data)
    ok(f"Loaded {len(features)} RUM point features")

    section("Generating synthetic vertical MEASUREMENT and MODEL series")
    series: Dict[str, Dict[str, Any]] = {}

    missing_up = 0
    missing_var_up = 0
    duplicate_ids = 0
    seen_ids = set()

    measurement_min = math.inf
    measurement_max = -math.inf
    model_min = math.inf
    model_max = -math.inf
    sigma_min = math.inf
    sigma_max = -math.inf
    up_min = math.inf
    up_max = -math.inf

    total_gaussian_noise_epochs = 0
    total_outlier_epochs = 0

    for idx, feature in enumerate(features):
        props = feature.get("properties") or {}

        rum_id = get_rum_id(props, idx)
        if rum_id in seen_ids:
            duplicate_ids += 1
            rum_id = f"{rum_id}__dup_{idx + 1:06d}"
        seen_ids.add(rum_id)

        if up_col not in props:
            missing_up += 1
        up_mm_yr = safe_float(props.get(up_col), 0.0)

        if var_up_col not in props:
            missing_var_up += 1
        sigma_up_mm_yr = positive_sqrt(props.get(var_up_col), FALLBACK_SIGMA_UP_MM_YR)

        measurement_clean = build_behavior_series(
            behavior=measurement_behavior,
            behavior_cfg=measurement_cfg,
            up_mm_yr=up_mm_yr,
            elapsed_years=elapsed_years,
            elapsed_days=elapsed_days,
            default_start_at_zero=start_at_zero,
        )

        model = build_behavior_series(
            behavior=vertical_model,
            behavior_cfg=model_cfg,
            up_mm_yr=up_mm_yr,
            elapsed_years=elapsed_years,
            elapsed_days=elapsed_days,
            default_start_at_zero=start_at_zero,
        )

        rng = random.Random(stable_noise_seed(random_seed, rum_id))
        measurement, noise_stats = add_measurement_noise(
            measurement_mm=measurement_clean,
            noise_sigma_mm=measurement_noise_sigma_mm,
            outlier_probability=measurement_outlier_probability,
            outlier_sigma_mm=measurement_outlier_sigma_mm,
            keep_first_epoch_zero=measurement_noise_keep_first_epoch_zero,
            rng=rng,
        )
        total_gaussian_noise_epochs += noise_stats["epochs_with_gaussian_noise"]
        total_outlier_epochs += noise_stats["epochs_with_outliers"]

        sigma = build_sigma_series(sigma_up_mm_yr, elapsed_years)

        measurement = [round(float(v), round_digits) for v in measurement]
        model = [round(float(v), round_digits) for v in model]
        sigma = [round(float(s), ROUND_SIGMA_DIGITS) for s in sigma]

        if measurement:
            measurement_min = min(measurement_min, min(measurement))
            measurement_max = max(measurement_max, max(measurement))
        if model:
            model_min = min(model_min, min(model))
            model_max = max(model_max, max(model))
        if sigma:
            sigma_min = min(sigma_min, min(sigma))
            sigma_max = max(sigma_max, max(sigma))

        up_min = min(up_min, up_mm_yr)
        up_max = max(up_max, up_mm_yr)

        series[rum_id] = {
            "measurement_mm": measurement,
            "model_mm": model,
            "sigma_mm": sigma,
            "source_up_mm_yr": round(up_mm_yr, ROUND_VERTICAL_DIGITS),
            "source_sigma_up_mm_yr": round(sigma_up_mm_yr, ROUND_SIGMA_DIGITS),
            "measurement_behavior": measurement_behavior,
            "measurement_noise": measurement_noise_label,
            "vertical_model": vertical_model,
        }

    if duplicate_ids:
        warn(f"Duplicate RUM IDs were made unique: {duplicate_ids}")

    if missing_up:
        warn(f"Features missing '{up_col}' field; fallback up=0 used: {missing_up}")

    if missing_var_up:
        warn(f"Features missing '{var_up_col}' field; fallback sigma used: {missing_var_up}")

    if not series:
        raise ValueError("No RUM series generated")

    ok(f"Generated synthetic series for {len(series)} RUMs × {len(epochs)} epochs")
    ok(f"Applied Gaussian measurement noise to {total_gaussian_noise_epochs} epoch values")
    ok(f"Applied synthetic measurement outliers to {total_outlier_epochs} epoch values")

    section("Writing epoch products")
    metadata = {
        "schema": "vertical_epochs_v2_measurement_model",
        "source": "velocity_only_rum_product",
        "source_points_geojson": prepared["points_geojson"],
        "roles": {
            "measurement_mm": "synthetic measurement series for trendline/popup/labelling",
            "model_mm": "synthetic model series for RUM height/choropleth/walls/blankies",
            "sigma_mm": "base uncertainty series; Step 04 enhances/replaces this",
        },
        "measurement": {
            "behavior": measurement_behavior,
            "noise": measurement_noise_label,
            "noise_sigma_mm": measurement_noise_sigma_mm,
            "outlier_probability": measurement_outlier_probability,
            "outlier_sigma_mm": measurement_outlier_sigma_mm,
            "noise_keep_first_epoch_zero": measurement_noise_keep_first_epoch_zero,
            "config": measurement_cfg,
        },
        "model": {
            "behavior": vertical_model,
            "config": model_cfg,
        },
        "start_date": start_date,
        "end_date": end_date,
        "interval_days": interval_days,
        "epoch_count": len(epochs),
        "rum_count": len(series),
        "vertical_unit": "mm",
        "velocity_unit": "mm/yr",
        "sigma_unit": "mm",
        "up_velocity_field": up_col,
        "up_variance_field": var_up_col,
        "random_seed": random_seed,
        "summary": {
            "source_up_min_mm_yr": round(up_min, ROUND_VERTICAL_DIGITS),
            "source_up_max_mm_yr": round(up_max, ROUND_VERTICAL_DIGITS),
            "measurement_min_mm": round(measurement_min, round_digits),
            "measurement_max_mm": round(measurement_max, round_digits),
            "model_min_mm": round(model_min, round_digits),
            "model_max_mm": round(model_max, round_digits),
            "sigma_min_mm": round(sigma_min, ROUND_SIGMA_DIGITS),
            "sigma_max_mm": round(sigma_max, ROUND_SIGMA_DIGITS),
            "measurement_noise_epoch_values": total_gaussian_noise_epochs,
            "measurement_outlier_epoch_values": total_outlier_epochs,
        },
    }

    epoch_product = {
        "metadata": metadata,
        "epochs": epochs,
        "epoch_decimal_year": epoch_decimal_year,
        "epoch_unix": epoch_unix,
        "series": series,
    }

    output_base_path.parent.mkdir(parents=True, exist_ok=True)
    with output_base_path.open("w", encoding="utf-8") as f:
        json.dump(epoch_product, f, ensure_ascii=False, separators=(",", ":"))
    ok(f"Wrote base synthetic epochs: {output_base_path} ({output_base_path.stat().st_size / 1024 / 1024:.2f} MB)")

    # At this stage the working epoch product is identical to the base product.
    # Step 04 may later replace/enhance sigma and overwrite vertical_epoch_json.
    if output_sigma_path != output_base_path:
        shutil.copyfile(output_base_path, output_sigma_path)
        ok(f"Wrote working epoch product: {output_sigma_path} ({output_sigma_path.stat().st_size / 1024 / 1024:.2f} MB)")

    elapsed = time.time() - t_start

    section("Summary")
    ok(f"Step 02 complete in {elapsed:.2f} s")
    print(f"  RUM count                : {len(series)}")
    print(f"  Epoch count              : {len(epochs)}")
    print(f"  Up velocity range        : {up_min:.4f} to {up_max:.4f} mm/yr")
    print(f"  MEASUREMENT range        : {measurement_min:.4f} to {measurement_max:.4f} mm")
    print(f"  MODEL range              : {model_min:.4f} to {model_max:.4f} mm")
    print(f"  Sigma range              : {sigma_min:.4f} to {sigma_max:.4f} mm")
    print(f"  MEASUREMENT behavior     : {measurement_behavior}")
    print(f"  MEASUREMENT noise        : {measurement_noise_label} ({measurement_noise_sigma_mm} mm)")
    print(f"  MODEL behavior           : {vertical_model}")


if __name__ == "__main__":
    main()
