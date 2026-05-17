#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_prepare_vertical_epochs_from_velocity.py

Generic RUM-based InSAR template step.

Purpose
-------
Convert a velocity-based RUM product into a viewer-ready vertical epoch JSON.

Typical input:
  Data/<project>_points_wgs84_with_rumid.geojson

Required per-feature properties:
  rum_id
  up        vertical velocity [mm/yr]

Optional per-feature properties:
  var_up    vertical velocity variance [(mm/yr)^2]

Outputs:
  Main dense linear epoch product, configured by:
    config/project_config.json -> prepared_inputs.vertical_epoch_json_without_enhanced_sigma

The output schema is the expected downstream contract:
  epochs
  epoch_decimal_year
  epoch_unix
  series[rum_id].vertical_mm
  series[rum_id].sigma_mm
  series[rum_id].source_up_mm_yr
  series[rum_id].source_sigma_up_mm_yr

Notes
-----
This script is designed for velocity-only products.
If a future dataset already has real epoch time series, use a different
adapter script that maps that real time series into the same output contract.
"""

from __future__ import annotations

import calendar
import datetime as dt
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional


# =============================================================================
# PATHS
# =============================================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_DIR / "config" / "project_config.json"


# =============================================================================
# HELPERS
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


def parse_date(value: str) -> dt.date:
    return dt.date.fromisoformat(value)


def date_to_unix(date: dt.date) -> float:
    return dt.datetime(date.year, date.month, date.day, tzinfo=dt.timezone.utc).timestamp()


def decimal_year(date: dt.date) -> float:
    year_start = dt.date(date.year, 1, 1)
    next_year_start = dt.date(date.year + 1, 1, 1)
    return date.year + (date - year_start).days / (next_year_start - year_start).days


def elapsed_years(start: dt.date, date: dt.date) -> float:
    return (date - start).days / 365.25


def add_months(date: dt.date, months: int) -> dt.date:
    month0 = date.month - 1 + months
    year = date.year + month0 // 12
    month = month0 % 12 + 1
    day = min(date.day, calendar.monthrange(year, month)[1])
    return dt.date(year, month, day)


def build_epochs_by_days(start: dt.date, end: dt.date, interval_days: int) -> List[dt.date]:
    if interval_days <= 0:
        raise ValueError("interval_days must be positive")

    epochs: List[dt.date] = []
    current = start
    step = dt.timedelta(days=interval_days)

    while current <= end:
        epochs.append(current)
        current += step

    if epochs[-1] != end:
        epochs.append(end)

    return epochs


def build_epochs_by_months(start: dt.date, end: dt.date, interval_months: int) -> List[dt.date]:
    if interval_months <= 0:
        raise ValueError("interval_months must be positive")

    epochs: List[dt.date] = []
    current = start

    while current <= end:
        epochs.append(current)
        current = add_months(current, interval_months)

    if epochs[-1] != end:
        epochs.append(end)

    return epochs


def make_rum_id(props: Dict[str, Any]) -> str:
    x = safe_float(props.get("x_rum"))
    y = safe_float(props.get("y_rum"))

    if x is not None and y is not None:
        return f"RUM_{int(round(x))}_{int(round(y))}"

    lon = safe_float(props.get("lon"), 0.0) or 0.0
    lat = safe_float(props.get("lat"), 0.0) or 0.0
    return f"RUM_lon{lon:.6f}_lat{lat:.6f}".replace("-", "m").replace(".", "p")


def load_geojson(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if data.get("type") != "FeatureCollection":
        raise ValueError(f"Expected GeoJSON FeatureCollection, got: {data.get('type')}")

    return data


def write_json(path: Path, data: Dict[str, Any], pretty: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        if pretty:
            json.dump(data, f, ensure_ascii=False, indent=2)
        else:
            json.dump(data, f, ensure_ascii=False, separators=(",", ":"))


def get_sigma_up_mm_yr(props: Dict[str, Any], var_up_field: str) -> float:
    var_up = safe_float(props.get(var_up_field), 0.0) or 0.0
    return math.sqrt(max(0.0, var_up))


def noise_from_sigma(rng: random.Random, sigma: float, fraction: float) -> float:
    if sigma <= 0.0 or fraction <= 0.0:
        return 0.0
    return rng.gauss(0.0, sigma * fraction)


def derive_optional_sparse_output(main_output: Path) -> Path:
    stem = main_output.stem
    if "1b_linear_14d" in stem:
        return main_output.with_name(stem.replace("1b_linear_14d", "1a_linear_3mo") + main_output.suffix)
    return main_output.with_name(stem + "_debug_3mo" + main_output.suffix)


# =============================================================================
# EPOCH GENERATION
# =============================================================================

def build_linear_epoch_payload(
    geojson: Dict[str, Any],
    epochs: List[dt.date],
    start_date: dt.date,
    dataset_name: str,
    dataset_title: str,
    up_field: str,
    var_up_field: str,
    rng_seed: int,
    noise_fraction_of_sigma: float,
    round_digits: int,
) -> Dict[str, Any]:
    rng = random.Random(rng_seed)

    series: Dict[str, Any] = {}
    skipped_no_id = 0
    skipped_no_up = 0

    duplicate_counts: Dict[str, int] = {}

    for feature in geojson.get("features", []):
        props = feature.get("properties") or {}

        rum_id = props.get("rum_id") or make_rum_id(props)
        rum_id = str(rum_id)

        duplicate_counts[rum_id] = duplicate_counts.get(rum_id, 0) + 1
        if duplicate_counts[rum_id] > 1:
            rum_id = f"{rum_id}_dup{duplicate_counts[rum_id]}"

        if not rum_id:
            skipped_no_id += 1
            continue

        up = safe_float(props.get(up_field))
        if up is None:
            skipped_no_up += 1
            continue

        sigma_up = get_sigma_up_mm_yr(props, var_up_field)

        vertical_mm: List[float] = []
        sigma_mm: List[float] = []

        for epoch in epochs:
            t = elapsed_years(start_date, epoch)
            sigma = sigma_up * t
            disp = up * t + noise_from_sigma(rng, sigma, noise_fraction_of_sigma)

            vertical_mm.append(round(disp, round_digits))
            sigma_mm.append(round(sigma, round_digits))

        series[rum_id] = {
            "vertical_mm": vertical_mm,
            "sigma_mm": sigma_mm,
            "source_up_mm_yr": round(up, round_digits),
            "source_sigma_up_mm_yr": round(sigma_up, round_digits),
        }

    return {
        "metadata": {
            "schema_version": "0.3",
            "dataset_name": dataset_name,
            "dataset_title": dataset_title,
            "dataset_type": "synthetic_linear_from_velocity",
            "description": "Synthetic cumulative vertical displacement generated from source vertical velocity.",
            "id_property": "rum_id",
            "source_velocity_property": up_field,
            "source_variance_property": var_up_field,
            "units": {
                "source_up": "mm/yr",
                "source_sigma_up": "mm/yr",
                "vertical_mm": "mm",
                "sigma_mm": "mm"
            },
            "start_date": start_date.isoformat(),
            "end_date": epochs[-1].isoformat(),
            "epoch_count": len(epochs),
            "rum_count": len(series),
            "skipped_no_id_count": skipped_no_id,
            "skipped_no_up_count": skipped_no_up,
            "random_seed": rng_seed,
            "noise_fraction_of_sigma": noise_fraction_of_sigma,
            "assumptions": [
                "Cumulative displacement is zero at the first epoch.",
                "Each real RUM follows a linear trend based on its source up velocity.",
                "sigma_mm is initially derived from sqrt(var_up) multiplied by elapsed years.",
                "If var_up is unavailable or zero, sigma_mm is zero until an optional uncertainty enhancement step.",
                "Optional sigma enhancement may later replace sigma_mm without changing vertical_mm."
            ],
        },
        "epochs": [e.isoformat() for e in epochs],
        "epoch_decimal_year": [round(decimal_year(e), 8) for e in epochs],
        "epoch_unix": [date_to_unix(e) for e in epochs],
        "series": series,
    }


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    cfg = load_config()

    project_cfg = cfg.get("project", {})
    prepared = cfg.get("prepared_inputs", {})
    source_inputs = cfg.get("source_inputs", {})
    epoch_cfg = cfg.get("epoch_generation", {})

    points_path = resolve_project_path(
        prepared.get("points_geojson", "Data/points_wgs84_with_rumid.geojson")
    )
    main_output = resolve_project_path(
        prepared.get("vertical_epoch_json_without_enhanced_sigma", "Data/vertical_epochs_linear_14d.json")
    )
    sparse_output = derive_optional_sparse_output(main_output)

    dataset_id = str(project_cfg.get("dataset_id", "rum_project"))
    dataset_title = str(project_cfg.get("dataset_title", dataset_id))

    velocity_fields = source_inputs.get("source_velocity_fields", {})
    variance_fields = source_inputs.get("source_variance_fields", {})

    up_field = str(velocity_fields.get("up", "up"))
    var_up_field = str(variance_fields.get("var_up", "var_up"))

    start_date = parse_date(str(epoch_cfg.get("default_start_date", "2014-01-01")))
    end_date = parse_date(str(epoch_cfg.get("default_end_date", "2025-01-01")))
    interval_days = int(epoch_cfg.get("default_interval_days", 14))
    interval_months_debug = int(epoch_cfg.get("default_interval_months_debug", 3))

    rng_seed = int(epoch_cfg.get("random_seed", 42))
    round_digits = int(epoch_cfg.get("round_digits", 4))
    noise_fraction = float(epoch_cfg.get("linear_noise_fraction_of_sigma", 0.10))

    if end_date <= start_date:
        raise ValueError("default_end_date must be after default_start_date")

    section("Configuration")
    print(f"  Project root       : {PROJECT_DIR}")
    print(f"  Points GeoJSON     : {points_path}")
    print(f"  Main output        : {main_output}")
    print(f"  Sparse debug output: {sparse_output}")
    print(f"  Dataset ID         : {dataset_id}")
    print(f"  Up field           : {up_field}")
    print(f"  Var-up field       : {var_up_field}")
    print(f"  Date range         : {start_date} → {end_date}")
    print(f"  Main interval      : {interval_days} days")
    print(f"  Sparse interval    : {interval_months_debug} months")
    print(f"  Noise fraction     : {noise_fraction}")

    section("Loading points GeoJSON")
    if not points_path.exists():
        raise FileNotFoundError(f"Missing points GeoJSON: {points_path}")

    geojson = load_geojson(points_path)
    features = geojson.get("features", [])
    ok(f"Loaded features: {len(features)}")

    missing_up = 0
    missing_var_up = 0
    for feature in features:
        props = feature.get("properties") or {}
        if safe_float(props.get(up_field)) is None:
            missing_up += 1
        if safe_float(props.get(var_up_field)) is None:
            missing_var_up += 1

    if missing_up:
        warn(f"{missing_up} features do not have numeric '{up_field}' and will be skipped")
    else:
        ok(f"All features have numeric '{up_field}'")

    if missing_var_up:
        warn(f"{missing_var_up} features do not have numeric '{var_up_field}'; sigma may be zero for them")
    else:
        ok(f"All features have numeric '{var_up_field}'")

    section("Building epoch axes")
    main_epochs = build_epochs_by_days(start_date, end_date, interval_days)
    sparse_epochs = build_epochs_by_months(start_date, end_date, interval_months_debug)

    ok(f"Main dense epochs : {len(main_epochs)} ({main_epochs[0]} → {main_epochs[-1]})")
    ok(f"Sparse debug epochs: {len(sparse_epochs)} ({sparse_epochs[0]} → {sparse_epochs[-1]})")

    section("Generating main dense linear epoch product")
    main_payload = build_linear_epoch_payload(
        geojson=geojson,
        epochs=main_epochs,
        start_date=start_date,
        dataset_name=f"{dataset_id}_linear_{interval_days}d",
        dataset_title=dataset_title,
        up_field=up_field,
        var_up_field=var_up_field,
        rng_seed=rng_seed,
        noise_fraction_of_sigma=noise_fraction,
        round_digits=round_digits,
    )

    write_json(main_output, main_payload, pretty=False)
    ok(f"Written: {main_output} ({main_output.stat().st_size / 1024 / 1024:.2f} MB)")
    ok(f"RUMs: {len(main_payload['series'])}, epochs: {len(main_payload['epochs'])}")

    section("Generating optional sparse debug epoch product")
    sparse_payload = build_linear_epoch_payload(
        geojson=geojson,
        epochs=sparse_epochs,
        start_date=start_date,
        dataset_name=f"{dataset_id}_linear_{interval_months_debug}mo",
        dataset_title=dataset_title,
        up_field=up_field,
        var_up_field=var_up_field,
        rng_seed=rng_seed,
        noise_fraction_of_sigma=noise_fraction,
        round_digits=round_digits,
    )

    write_json(sparse_output, sparse_payload, pretty=False)
    ok(f"Written: {sparse_output} ({sparse_output.stat().st_size / 1024 / 1024:.2f} MB)")
    ok(f"RUMs: {len(sparse_payload['series'])}, epochs: {len(sparse_payload['epochs'])}")

    section("Spot check")
    sample_ids = list(main_payload["series"].keys())[:5]
    for rid in sample_ids:
        entry = main_payload["series"][rid]
        vals = entry["vertical_mm"]
        sig = entry["sigma_mm"]
        print(
            f"  {rid:<24s} "
            f"up={entry['source_up_mm_yr']:8.3f} mm/yr  "
            f"v0={vals[0]:8.3f} mm  vLast={vals[-1]:8.3f} mm  "
            f"sLast={sig[-1]:8.3f} mm"
        )

    section("SUMMARY")
    ok("Step 02 complete — velocity-based product converted to synthetic vertical epochs")
    ok("Next template step: patch/run 03_build_footprints.py")
    print()


if __name__ == "__main__":
    main()
