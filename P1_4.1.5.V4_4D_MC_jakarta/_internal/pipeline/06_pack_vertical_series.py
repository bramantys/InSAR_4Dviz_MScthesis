#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_pack_vertical_series.py

InSAR4D RUM Viewer pipeline step 06.

Purpose
-------
Convert the validated vertical epoch product into a deterministic packed series
product for downstream tile/texture generation.

Input
-----
  prepared_inputs.vertical_epoch_json
    _internal/data_pipeline/vertical_epochs.json

Output
------
  generated_outputs.packed_series
    _internal/data_pipeline/packed_series.json

Why this step exists
--------------------
The epoch product from Step 02/04 is human-readable and keyed by rum_id:

  series[rum_id]["measurement_mm"] = [...]
  series[rum_id]["model_mm"] = [...]
  series[rum_id]["sigma_mm"] = [...]

For height textures and B3DM tiles, we need a stable row index:

  rum_id -> row_index
  row_index -> packed measurement/model/sigma arrays

Packed array roles
------------------
  arrays.measurement_mm
    Synthetic MEASUREMENT series for trendline / popup / labelling.

  arrays.model_mm
    Synthetic MODEL series for RUM height / choropleth / height texture /
    caps / walls / blankies.

  arrays.sigma_mm
    Enhanced uncertainty / sigma series from Step 04.

This step establishes the stable row order and writes flat row-major arrays.
"""

from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pipeline_config import load_resolved_config, resolve_path


# =============================================================================
# ADVANCED SETTINGS
# =============================================================================
# Normal users should edit config/project_config.json, not this script.

ROUND_VERTICAL_DIGITS = 4
ROUND_SIGMA_DIGITS = 4
ROUND_SOURCE_VELOCITY_DIGITS = 6

# Keep rum_order deterministic. Natural-ish sorting keeps RUM_2 before RUM_10.
SORT_RUM_IDS = True

# Fail if any series arrays are malformed. This should already be checked by
# Step 05, but Step 06 is strict because it creates the final row-index contract.
FAIL_ON_MALFORMED_SERIES = True


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


def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


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


def finite_or_fail(value: Any, label: str) -> float:
    try:
        out = float(value)
    except Exception as exc:
        raise ValueError(f"Non-numeric {label}: {value!r}") from exc
    if not math.isfinite(out):
        raise ValueError(f"Non-finite {label}: {value!r}")
    return out


def natural_key(text: str) -> List[Any]:
    parts = re.split(r"(\d+)", str(text))
    out: List[Any] = []
    for part in parts:
        out.append(int(part) if part.isdigit() else part.lower())
    return out


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


# =============================================================================
# PACKING
# =============================================================================

def validate_epoch_product(payload: Dict[str, Any]) -> Tuple[List[str], List[float], List[float], Dict[str, Any]]:
    epochs = payload.get("epochs")
    epoch_decimal_year = payload.get("epoch_decimal_year")
    epoch_unix = payload.get("epoch_unix")
    series = payload.get("series")

    if not isinstance(epochs, list) or not epochs:
        raise ValueError("vertical epoch product has no epochs")
    if not isinstance(series, dict) or not series:
        raise ValueError("vertical epoch product has no series")

    if epoch_decimal_year is None:
        epoch_decimal_year = []
    if epoch_unix is None:
        epoch_unix = []

    if epoch_decimal_year and len(epoch_decimal_year) != len(epochs):
        raise ValueError("epoch_decimal_year length does not match epochs")
    if epoch_unix and len(epoch_unix) != len(epochs):
        raise ValueError("epoch_unix length does not match epochs")

    return epochs, epoch_decimal_year, epoch_unix, series


def require_series_array(item: Dict[str, Any], key: str, rum_id: str, epoch_count: int) -> List[Any]:
    values = item.get(key)

    if not isinstance(values, list) or len(values) != epoch_count:
        if FAIL_ON_MALFORMED_SERIES:
            raise ValueError(
                f"Malformed {key} for {rum_id}: expected length {epoch_count}, "
                f"got {len(values) if isinstance(values, list) else 'not-list'}"
            )
        return [0.0] * epoch_count

    return values


def append_series_values(
    flat: List[float],
    values: List[Any],
    label_key: str,
    rum_id: str,
    round_digits: int,
) -> None:
    for epoch_index, value in enumerate(values):
        flat.append(
            round(
                finite_or_fail(value, f"{label_key}[{rum_id}][{epoch_index}]"),
                round_digits,
            )
        )


def pack_series(
    series: Dict[str, Any],
    rum_order: List[str],
    epoch_count: int,
) -> Tuple[Dict[str, List[float]], Dict[str, List[Any]], Dict[str, Any]]:
    measurement_flat: List[float] = []
    model_flat: List[float] = []
    sigma_flat: List[float] = []

    source_up: List[float] = []
    source_sigma_up: List[float] = []
    measurement_behavior: List[str] = []
    measurement_noise: List[str] = []
    vertical_model: List[str] = []
    sigma_topology_class: List[str] = []
    sigma_neighbour_count: List[int] = []

    malformed_count = 0

    for row_index, rum_id in enumerate(rum_order):
        item = series.get(rum_id)
        if not isinstance(item, dict):
            malformed_count += 1
            if FAIL_ON_MALFORMED_SERIES:
                raise ValueError(f"Malformed series entry for {rum_id}")
            item = {}

        measurement = require_series_array(item, "measurement_mm", rum_id, epoch_count)
        model = require_series_array(item, "model_mm", rum_id, epoch_count)
        sigma = require_series_array(item, "sigma_mm", rum_id, epoch_count)

        append_series_values(
            flat=measurement_flat,
            values=measurement,
            label_key="measurement_mm",
            rum_id=rum_id,
            round_digits=ROUND_VERTICAL_DIGITS,
        )

        append_series_values(
            flat=model_flat,
            values=model,
            label_key="model_mm",
            rum_id=rum_id,
            round_digits=ROUND_VERTICAL_DIGITS,
        )

        for epoch_index, value in enumerate(sigma):
            sf = finite_or_fail(value, f"sigma_mm[{rum_id}][{epoch_index}]")
            if sf < 0:
                raise ValueError(f"Negative sigma_mm[{rum_id}][{epoch_index}]: {sf}")
            sigma_flat.append(round(sf, ROUND_SIGMA_DIGITS))

        source_up.append(round(safe_float(item.get("source_up_mm_yr"), 0.0), ROUND_SOURCE_VELOCITY_DIGITS))
        source_sigma_up.append(round(safe_float(item.get("source_sigma_up_mm_yr"), 0.0), ROUND_SOURCE_VELOCITY_DIGITS))
        measurement_behavior.append(str(item.get("measurement_behavior", "")))
        measurement_noise.append(str(item.get("measurement_noise", "")))
        vertical_model.append(str(item.get("vertical_model", "")))

        sigma_model = item.get("sigma_model") or {}
        sigma_topology_class.append(str(sigma_model.get("topology_class", "")))
        sigma_neighbour_count.append(int(safe_float(sigma_model.get("neighbour_count"), -1)))

    arrays = {
        "measurement_mm": measurement_flat,
        "model_mm": model_flat,
        "sigma_mm": sigma_flat,
    }

    per_rum = {
        "source_up_mm_yr": source_up,
        "source_sigma_up_mm_yr": source_sigma_up,
        "measurement_behavior": measurement_behavior,
        "measurement_noise": measurement_noise,
        "vertical_model": vertical_model,
        "sigma_topology_class": sigma_topology_class,
        "sigma_neighbour_count": sigma_neighbour_count,
    }

    pack_summary = {
        "malformed_count": malformed_count,
        "measurement_mm_role": "trendline_popup_labelling",
        "model_mm_role": "rum_height_choropleth_height_texture_caps_walls_blankies",
        "sigma_mm_role": "uncertainty_snr_hatch_visualization",
    }

    return arrays, per_rum, pack_summary


def summarize_arrays(
    measurement_flat: List[float],
    model_flat: List[float],
    sigma_flat: List[float],
    source_up: List[float],
) -> Dict[str, Any]:
    return {
        "measurement_min_mm": min(measurement_flat),
        "measurement_p02_mm": percentile(measurement_flat, 2),
        "measurement_p50_mm": percentile(measurement_flat, 50),
        "measurement_p98_mm": percentile(measurement_flat, 98),
        "measurement_max_mm": max(measurement_flat),

        "model_min_mm": min(model_flat),
        "model_p02_mm": percentile(model_flat, 2),
        "model_p50_mm": percentile(model_flat, 50),
        "model_p98_mm": percentile(model_flat, 98),
        "model_max_mm": max(model_flat),

        "sigma_min_mm": min(sigma_flat),
        "sigma_p02_mm": percentile(sigma_flat, 2),
        "sigma_p50_mm": percentile(sigma_flat, 50),
        "sigma_p98_mm": percentile(sigma_flat, 98),
        "sigma_max_mm": max(sigma_flat),

        "source_up_min_mm_yr": min(source_up) if source_up else None,
        "source_up_max_mm_yr": max(source_up) if source_up else None,
    }


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    t_start = time.time()

    cfg = load_resolved_config(__file__)
    project_root = Path(cfg["_resolved"]["project_root"])

    prepared = cfg["prepared_inputs"]
    generated = cfg["generated_outputs"]

    input_path = resolve_path(project_root, prepared["vertical_epoch_json"])
    output_path = resolve_path(project_root, generated["packed_series"])

    section("Configuration")
    print(f"  Project root       : {project_root}")
    print(f"  Epoch input        : {input_path}")
    print(f"  Packed output      : {output_path}")

    section("Loading vertical epoch product")
    epoch_product = load_json(input_path)
    epochs, epoch_decimal_year, epoch_unix, series = validate_epoch_product(epoch_product)

    rum_ids = list(series.keys())
    rum_order = sorted(rum_ids, key=natural_key) if SORT_RUM_IDS else rum_ids
    rum_index = {rum_id: idx for idx, rum_id in enumerate(rum_order)}

    epoch_count = len(epochs)
    rum_count = len(rum_order)

    ok(f"Loaded {rum_count} RUM series × {epoch_count} epochs")
    ok("Built deterministic rum_order and rum_index")

    section("Packing arrays")
    arrays, per_rum, pack_summary = pack_series(
        series=series,
        rum_order=rum_order,
        epoch_count=epoch_count,
    )

    expected_len = rum_count * epoch_count

    for key in ["measurement_mm", "model_mm", "sigma_mm"]:
        if len(arrays[key]) != expected_len:
            raise ValueError(
                f"Packed {key} array length mismatch: "
                f"actual={len(arrays[key])}, expected={expected_len}"
            )
        ok(f"Packed {key} flat array length: {len(arrays[key])}")

    section("Building metadata summary")
    summary = summarize_arrays(
        measurement_flat=arrays["measurement_mm"],
        model_flat=arrays["model_mm"],
        sigma_flat=arrays["sigma_mm"],
        source_up=per_rum["source_up_mm_yr"],
    )

    source_metadata = epoch_product.get("metadata", {})

    payload = {
        "metadata": {
            "schema": "packed_series_v3_measurement_model",
            "source_epoch_product": prepared["vertical_epoch_json"],
            "source_schema": source_metadata.get("schema"),
            "rum_count": rum_count,
            "epoch_count": epoch_count,
            "array_length": expected_len,
            "row_major_order": "rum_order_then_epoch",
            "index_formula": "flat_index = rum_index * epoch_count + epoch_index",
            "sort_rum_ids": SORT_RUM_IDS,
            "vertical_unit": "mm",
            "sigma_unit": "mm",
            "velocity_unit": "mm/yr",
            "vertical_roles": {
                "arrays.measurement_mm": "synthetic_measurement_series_for_trendline_popup_labelling",
                "arrays.model_mm": "synthetic_model_series_for_rum_height_choropleth_height_texture_caps_walls_blankies",
                "arrays.sigma_mm": "uncertainty_series_for_snr_hatch_visualization",
            },
            "source_epoch_metadata": {
                "measurement": source_metadata.get("measurement"),
                "model": source_metadata.get("model"),
                "sigma_enhancement": source_metadata.get("sigma_enhancement"),
            },
            "summary": summary,
            "pack_summary": pack_summary,
        },
        "epochs": epochs,
        "epoch_decimal_year": epoch_decimal_year,
        "epoch_unix": epoch_unix,
        "rum_order": rum_order,
        "rum_index": rum_index,
        "arrays": arrays,
        "per_rum": per_rum,
    }

    section("Writing packed series")
    write_json(output_path, payload)

    elapsed = time.time() - t_start

    ok(f"Wrote packed series: {output_path} ({output_path.stat().st_size / 1024 / 1024:.2f} MB)")

    section("Summary")
    ok(f"Step 06 complete in {elapsed:.2f} s")
    print(f"  RUM count              : {rum_count}")
    print(f"  Epoch count            : {epoch_count}")
    print(f"  Packed array length    : {expected_len}")
    print(f"  MEASUREMENT range      : {summary['measurement_min_mm']:.4f} to {summary['measurement_max_mm']:.4f} mm")
    print(f"  MODEL range            : {summary['model_min_mm']:.4f} to {summary['model_max_mm']:.4f} mm")
    print(f"  Sigma range            : {summary['sigma_min_mm']:.4f} to {summary['sigma_max_mm']:.4f} mm")
    print(f"  Source up range        : {summary['source_up_min_mm_yr']:.4f} to {summary['source_up_max_mm_yr']:.4f} mm/yr")


if __name__ == "__main__":
    main()
