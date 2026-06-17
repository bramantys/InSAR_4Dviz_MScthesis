#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10_build_height_texture.py

InSAR4D RUM Viewer pipeline step 10.

Purpose
-------
Build the MODEL/sigma texture used by Cesium/B3DM shaders.

Inputs
------
  generated_outputs.packed_series
    _internal/data_pipeline/packed_series.json

  generated_outputs.blank_cells
    _internal/data_pipeline/blank_cells.json

Outputs
-------
  generated_outputs.height_texture
    _internal/data_pipeline/tiles/height_texture.png

  generated_outputs.height_meta
    _internal/data_pipeline/tiles/height_meta.json

Texture layout
--------------
Width  = epoch_count
Height = real_rum_count + blank_cell_count

Rows:
  0 .. real_rum_count-1
    real RUM rows in packed_series.rum_order order

  real_rum_count .. real_rum_count+blank_cell_count-1
    blank cell rows in blank_cells feature order

Columns:
  0 .. epoch_count-1
    epoch index

Channels:
  R = high byte of normalized MODEL uint16
  G = low byte of normalized MODEL uint16
  B = normalized sigma uint8

MODEL packing:
  v_norm = clamp((model_mm - v_min_mm) / (v_max_mm - v_min_mm), 0, 1)
  v_u16  = round(v_norm * 65535)
  R      = v_u16 >> 8
  G      = v_u16 & 255

Sigma packing:
  B stores the full raw vertical sigma range against the dataset maximum.
  The global real-RUM p98 is exported separately as the visual relief-height
  ceiling. Above-p98 values therefore remain recoverable for flat-top size.

Important
---------
This texture is not the color map itself. It is a data texture used by the
viewer to animate cap/wall MODEL heights and vertical-uncertainty relief.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from PIL import Image
except ImportError as exc:
    raise ImportError(
        "Pillow is required to write PNG textures. Install it with: pip install pillow"
    ) from exc

from pipeline_config import load_resolved_config, resolve_path


# =============================================================================
# ADVANCED SETTINGS
# =============================================================================
# Normal users should edit config/project_config.json, not this script.

ROUND_SUMMARY_DIGITS = 4

# The B channel stores the complete raw vertical sigma range. The visual p98
# ceiling is exported separately and applied by the viewer to the lowpoly
# relief. This preserves above-p98 exceedance for the flat-top encoding.
SIGMA_SCALE_MODE = "max"

# Minimum sigma scale to avoid division by zero.
SIGMA_SCALE_MIN_MM = 1.0

# Include blank cells in the data texture when blank_cells.json has features.
INCLUDE_BLANK_CELLS = True


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
        json.dump(payload, f, ensure_ascii=False, indent=2)


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


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


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


def minmax(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    return min(values), max(values)


def rel_path(project_root: Path, path: Path) -> str:
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return path.as_posix()


# =============================================================================
# INPUT EXTRACTION
# =============================================================================

def validate_packed_series(packed: Dict[str, Any]) -> Dict[str, Any]:
    metadata = packed.get("metadata") or {}
    epochs = packed.get("epochs")
    rum_order = packed.get("rum_order")
    rum_index = packed.get("rum_index")
    arrays = packed.get("arrays") or {}

    model = arrays.get("model_mm")
    sigma = arrays.get("sigma_mm")

    if not isinstance(epochs, list) or not epochs:
        raise ValueError("packed_series.json has no epochs")
    if not isinstance(rum_order, list) or not rum_order:
        raise ValueError("packed_series.json has no rum_order")
    if not isinstance(rum_index, dict) or not rum_index:
        raise ValueError("packed_series.json has no rum_index")
    if not isinstance(model, list) or not isinstance(sigma, list):
        raise ValueError("packed_series.json missing arrays.model_mm / arrays.sigma_mm")

    rum_count = int(metadata.get("rum_count", len(rum_order)))
    epoch_count = int(metadata.get("epoch_count", len(epochs)))
    expected_len = rum_count * epoch_count

    if len(model) != expected_len:
        raise ValueError(f"model_mm length mismatch: actual={len(model)}, expected={expected_len}")
    if len(sigma) != expected_len:
        raise ValueError(f"sigma_mm length mismatch: actual={len(sigma)}, expected={expected_len}")
    if len(rum_order) != rum_count:
        raise ValueError(f"rum_order length mismatch: actual={len(rum_order)}, expected={rum_count}")

    return {
        "metadata": metadata,
        "epochs": epochs,
        "rum_order": rum_order,
        "rum_index": rum_index,
        "model": [safe_float(v, 0.0) for v in model],
        "sigma": [max(0.0, safe_float(s, 0.0)) for s in sigma],
        "rum_count": rum_count,
        "epoch_count": epoch_count,
    }


def extract_blank_rows(blank_cells: Dict[str, Any], epoch_count: int) -> Tuple[List[str], List[float], List[float], List[Dict[str, Any]]]:
    """
    Return blank row ids and flat row-major MODEL/sigma arrays.

    If there are no blank features, returns empty lists.
    """
    features = blank_cells.get("features", [])
    if not isinstance(features, list):
        raise ValueError("blank_cells.json features is not a list")

    blank_ids: List[str] = []
    model_flat: List[float] = []
    sigma_flat: List[float] = []
    row_metadata: List[Dict[str, Any]] = []

    for idx, feature in enumerate(features):
        props = feature.get("properties") or {}
        blank_id = str(props.get("blank_id", f"BLANK_{idx:06d}"))

        model = props.get("model_mm")
        sigma = props.get("sigma_mm")

        if not isinstance(model, list) or len(model) != epoch_count:
            raise ValueError(
                f"Blank cell {blank_id} model_mm length mismatch: "
                f"actual={len(model) if isinstance(model, list) else 'not-list'}, expected={epoch_count}"
            )
        if not isinstance(sigma, list) or len(sigma) != epoch_count:
            raise ValueError(
                f"Blank cell {blank_id} sigma_mm length mismatch: "
                f"actual={len(sigma) if isinstance(sigma, list) else 'not-list'}, expected={epoch_count}"
            )

        blank_ids.append(blank_id)
        model_flat.extend([safe_float(v, 0.0) for v in model])
        sigma_flat.extend([max(0.0, safe_float(s, 0.0)) for s in sigma])

        row_metadata.append({
            "blank_id": blank_id,
            "blank_index": int(props.get("blank_index", idx)),
            "grid_i": props.get("grid_i"),
            "grid_j": props.get("grid_j"),
            "lon_center": props.get("lon_center"),
            "lat_center": props.get("lat_center"),
        })

    return blank_ids, model_flat, sigma_flat, row_metadata


# =============================================================================
# TEXTURE PACKING
# =============================================================================

def choose_sigma_max(sigma_values: List[float]) -> float:
    """Return the raw-sigma storage maximum for the 8-bit B channel."""
    if not sigma_values:
        return SIGMA_SCALE_MIN_MM
    return max(SIGMA_SCALE_MIN_MM, float(max(sigma_values)))


def pack_displacement_to_rg(value_mm: float, v_min: float, v_max: float) -> Tuple[int, int, bool]:
    if v_max <= v_min:
        raise ValueError("v_max_mm must be greater than v_min_mm")

    clipped = value_mm < v_min or value_mm > v_max
    norm = clamp01((float(value_mm) - v_min) / (v_max - v_min))
    u16 = int(round(norm * 65535.0))
    u16 = max(0, min(65535, u16))

    r = (u16 >> 8) & 255
    g = u16 & 255
    return r, g, clipped


def pack_sigma_to_b(sigma_mm: float, sigma_max_mm: float) -> Tuple[int, bool]:
    clipped = sigma_mm > sigma_max_mm
    norm = clamp01(float(sigma_mm) / sigma_max_mm)
    b = int(round(norm * 255.0))
    return max(0, min(255, b)), clipped


def build_texture_rgb(
    model_values: List[float],
    sigma_values: List[float],
    width: int,
    height: int,
    v_min_mm: float,
    v_max_mm: float,
    sigma_max_mm: float,
) -> Tuple[Image.Image, Dict[str, Any]]:
    expected_len = width * height

    if len(model_values) != expected_len:
        raise ValueError(f"model_values length mismatch: actual={len(model_values)}, expected={expected_len}")
    if len(sigma_values) != expected_len:
        raise ValueError(f"sigma_values length mismatch: actual={len(sigma_values)}, expected={expected_len}")

    pixels: List[Tuple[int, int, int]] = []
    model_clip_count = 0
    sigma_clip_count = 0

    for v, s in zip(model_values, sigma_values):
        r, g, v_clipped = pack_displacement_to_rg(v, v_min_mm, v_max_mm)
        b, s_clipped = pack_sigma_to_b(s, sigma_max_mm)

        if v_clipped:
            model_clip_count += 1
        if s_clipped:
            sigma_clip_count += 1

        pixels.append((r, g, b))

    img = Image.new("RGB", (width, height))
    img.putdata(pixels)

    stats = {
        "pixel_count": expected_len,
        "model_clip_count": model_clip_count,
        "model_clip_fraction": model_clip_count / expected_len if expected_len else 0.0,
        "sigma_clip_count": sigma_clip_count,
        "sigma_clip_fraction": sigma_clip_count / expected_len if expected_len else 0.0,
    }

    return img, stats


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    t_start = time.time()

    cfg = load_resolved_config(__file__)
    project_root = Path(cfg["_resolved"]["project_root"])

    generated = cfg["generated_outputs"]
    height_cfg = cfg["height_texture"]
    vunc_cfg = cfg["vertical_uncertainty_encoding"]
    rum_size_m = float(cfg.get("expected_counts", {}).get("grid_spacing_m_nominal", 450.0))
    checkerboard_frequency_near = int(vunc_cfg.get("checkerboard_frequency_near", vunc_cfg.get("checkerboard_frequency", 4)))
    checkerboard_frequency_far = int(vunc_cfg.get("checkerboard_frequency_far", 2))
    pyramid_half_base_ratio = float(vunc_cfg.get("pyramid_half_base_ratio", 0.28))
    pyramid_footprint_reference_frequency_near = int(
        vunc_cfg.get("pyramid_footprint_reference_frequency_near", checkerboard_frequency_near)
    )
    pyramid_footprint_reference_frequency_far = int(
        vunc_cfg.get("pyramid_footprint_reference_frequency_far", checkerboard_frequency_far)
    )
    visibility_threshold_mode = str(vunc_cfg.get("visibility_threshold_mode", "global_percentile")).strip().lower()
    visibility_threshold_percentile = float(vunc_cfg.get("visibility_threshold_percentile", 50.0))

    packed_path = resolve_path(project_root, generated["packed_series"])
    blank_path = resolve_path(project_root, generated["blank_cells"])
    texture_path = resolve_path(project_root, generated["height_texture"])
    meta_path = resolve_path(project_root, generated["height_meta"])

    v_min_mm = float(height_cfg.get("v_min_mm", -1500.0))
    v_max_mm = float(height_cfg.get("v_max_mm", 1500.0))

    section("Configuration")
    print(f"  Project root       : {project_root}")
    print(f"  Packed input       : {packed_path}")
    print(f"  Blank input        : {blank_path}")
    print(f"  Texture output     : {texture_path}")
    print(f"  Metadata output    : {meta_path}")
    print(f"  MODEL packing range: {v_min_mm} to {v_max_mm} mm")
    print(f"  Sigma storage mode : raw max in B channel")

    if v_max_mm <= v_min_mm:
        raise ValueError("height_texture.v_max_mm must be greater than v_min_mm")

    section("Loading inputs")
    packed = load_json(packed_path)
    packed_data = validate_packed_series(packed)

    blank_cells = load_json(blank_path)
    blank_meta = blank_cells.get("metadata") or {}
    blank_status = blank_meta.get("status", "unknown")

    ok(f"Loaded packed series: {packed_data['rum_count']} RUM rows × {packed_data['epoch_count']} epochs")
    ok(f"Loaded blank-cell product: status={blank_status}, features={len(blank_cells.get('features', []))}")

    section("Preparing texture rows")
    real_model = packed_data["model"]
    real_sigma = packed_data["sigma"]
    real_rum_count = packed_data["rum_count"]
    epoch_count = packed_data["epoch_count"]

    blank_ids: List[str] = []
    blank_model: List[float] = []
    blank_sigma: List[float] = []
    blank_row_metadata: List[Dict[str, Any]] = []

    if INCLUDE_BLANK_CELLS:
        blank_ids, blank_model, blank_sigma, blank_row_metadata = extract_blank_rows(blank_cells, epoch_count)
    else:
        warn("INCLUDE_BLANK_CELLS=False; blank rows will not be included in texture")

    blank_count = len(blank_ids)

    all_model = real_model + blank_model
    all_sigma = real_sigma + blank_sigma

    texture_width = epoch_count
    texture_height = real_rum_count + blank_count

    if texture_height <= 0:
        raise ValueError("Texture height is zero")

    ok(f"Texture dimensions: {texture_width} × {texture_height}")
    ok(f"Rows: real={real_rum_count}, blank={blank_count}")

    section("Computing packing scales")
    model_min_actual, model_max_actual = minmax(all_model)
    sigma_min_actual, sigma_max_actual = minmax(all_sigma)
    sigma_max_p98 = percentile(real_sigma, 98)
    if visibility_threshold_mode != "global_percentile":
        raise ValueError(
            "vertical_uncertainty_encoding.visibility_threshold_mode currently supports only global_percentile"
        )
    if not 0.0 <= visibility_threshold_percentile <= 100.0:
        raise ValueError("visibility_threshold_percentile must be between 0 and 100")
    sigma_visibility_threshold = percentile(real_sigma, visibility_threshold_percentile)
    sigma_pack_max = choose_sigma_max(all_sigma)

    print(f"  Actual MODEL range    : {model_min_actual:.4f} to {model_max_actual:.4f} mm")
    print(f"  Actual sigma range    : {sigma_min_actual:.4f} to {sigma_max_actual:.4f} mm")
    print(f"  Sigma p98 (real RUMs) : {sigma_max_p98:.4f} mm")
    print(f"  Sigma storage max     : {sigma_pack_max:.4f} mm")
    print(f"  Sigma display p98     : {sigma_max_p98:.4f} mm")
    print(f"  Sigma relief threshold: P{visibility_threshold_percentile:g} = {sigma_visibility_threshold:.4f} mm")

    if model_min_actual < v_min_mm or model_max_actual > v_max_mm:
        warn(
            "MODEL values exceed configured packing range; clipping will occur. "
            f"actual={model_min_actual:.4f}..{model_max_actual:.4f}, "
            f"configured={v_min_mm:.4f}..{v_max_mm:.4f}"
        )

    section("Building texture")
    image, packing_stats = build_texture_rgb(
        model_values=all_model,
        sigma_values=all_sigma,
        width=texture_width,
        height=texture_height,
        v_min_mm=v_min_mm,
        v_max_mm=v_max_mm,
        sigma_max_mm=sigma_pack_max,
    )

    texture_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(texture_path)

    if packing_stats["model_clip_count"]:
        warn(f"MODEL clipped pixels: {packing_stats['model_clip_count']} ({packing_stats['model_clip_fraction']:.4%})")
    else:
        ok("No MODEL clipping")

    if packing_stats["sigma_clip_count"]:
        warn(f"Sigma clipped pixels: {packing_stats['sigma_clip_count']} ({packing_stats['sigma_clip_fraction']:.4%})")
    else:
        ok("No sigma clipping")

    section("Writing height metadata")
    meta = {
        "schema": "height_meta_v2_model_sigma",
        "source_packed_series": generated["packed_series"],
        "source_blank_cells": generated["blank_cells"],
        "height_texture": rel_path(project_root, texture_path),
        "texture": {
            "width": texture_width,
            "height": texture_height,
            "format": "RGB",
            "row_count": texture_height,
            "epoch_count": epoch_count,
            "real_rum_count": real_rum_count,
            "blank_count": blank_count,
            "row_layout": {
                "real_start_row": 0,
                "real_end_row_inclusive": real_rum_count - 1 if real_rum_count else None,
                "blank_start_row": real_rum_count if blank_count else None,
                "blank_end_row_inclusive": real_rum_count + blank_count - 1 if blank_count else None,
            },
        },
        "packing": {
            "type": "uint16_rg_plus_sigma_uint8_b",
            "channels": {
                "R": "model_u16_high_byte",
                "G": "model_u16_low_byte",
                "B": "sigma_u8_normalized",
            },
            "model": {
                "unit": "mm",
                "v_min_mm": v_min_mm,
                "v_max_mm": v_max_mm,
                "decode_formula": "model_mm = v_min_mm + ((R*256 + G)/65535) * (v_max_mm - v_min_mm)",
            },
            "sigma": {
                "unit": "mm",
                "sigma_storage_max_mm": sigma_pack_max,
                "sigma_display_p98_mm": sigma_max_p98,
                "sigma_visibility_threshold_mm": sigma_visibility_threshold,
                "sigma_visibility_threshold_mode": visibility_threshold_mode,
                "sigma_visibility_threshold_percentile": visibility_threshold_percentile,
                "sigma_max_mm": sigma_pack_max,
                "sigma_scale_mode": "raw_max_storage_with_separate_p98_display_ceiling",
                "decode_formula": "raw_sigma_mm = (B/255) * sigma_storage_max_mm",
                "display_height_formula": "display_sigma_mm = min(raw_sigma_mm, sigma_display_p98_mm)",
                "plateau_ratio_formula": "max(0, 1 - sigma_display_p98_mm/raw_sigma_mm)",
            },
        },
        "epochs": packed_data["epochs"],
        "epoch_decimal_year": packed.get("epoch_decimal_year", []),
        "epoch_unix": packed.get("epoch_unix", []),
        "rum_order": packed_data["rum_order"],
        "rum_index": packed_data["rum_index"],
        "blank_order": blank_ids,
        "blank_row_metadata": blank_row_metadata,
        "vertical_uncertainty_encoding": {
            **vunc_cfg,
            "checkerboard_frequency": checkerboard_frequency_near,
            "checkerboard_frequency_near": checkerboard_frequency_near,
            "checkerboard_frequency_far": checkerboard_frequency_far,
            "pyramid_half_base_m": rum_size_m / pyramid_footprint_reference_frequency_near * pyramid_half_base_ratio,
            "pyramid_half_base_near_m": rum_size_m / pyramid_footprint_reference_frequency_near * pyramid_half_base_ratio,
            "pyramid_half_base_far_m": rum_size_m / pyramid_footprint_reference_frequency_far * pyramid_half_base_ratio,
            "pyramid_footprint_reference_frequency_near": pyramid_footprint_reference_frequency_near,
            "pyramid_footprint_reference_frequency_far": pyramid_footprint_reference_frequency_far,
            "visibility_threshold_mm": sigma_visibility_threshold,
            "visibility_threshold_population": "real_rums_all_epochs",
            "rum_size_m": rum_size_m,
            "texture_channel": "B_raw_vertical_sigma",
            "horizontal_uncertainty_affected": False,
        },
        "summary": {
            "model_min_actual_mm": round(model_min_actual, ROUND_SUMMARY_DIGITS),
            "model_max_actual_mm": round(model_max_actual, ROUND_SUMMARY_DIGITS),
            "model_p02_actual_mm": round(percentile(all_model, 2), ROUND_SUMMARY_DIGITS),
            "model_p50_actual_mm": round(percentile(all_model, 50), ROUND_SUMMARY_DIGITS),
            "model_p98_actual_mm": round(percentile(all_model, 98), ROUND_SUMMARY_DIGITS),
            "sigma_min_actual_mm": round(sigma_min_actual, ROUND_SUMMARY_DIGITS),
            "sigma_max_actual_mm": round(sigma_max_actual, ROUND_SUMMARY_DIGITS),
            "sigma_p02_actual_mm": round(percentile(all_sigma, 2), ROUND_SUMMARY_DIGITS),
            "sigma_p50_actual_mm": round(percentile(all_sigma, 50), ROUND_SUMMARY_DIGITS),
            "sigma_p98_actual_mm": round(sigma_max_p98, ROUND_SUMMARY_DIGITS),
            "sigma_visibility_threshold_mm": round(sigma_visibility_threshold, ROUND_SUMMARY_DIGITS),
            "sigma_visibility_threshold_percentile": visibility_threshold_percentile,
            "sigma_visibility_retained_fraction": sum(1 for value in real_sigma if value >= sigma_visibility_threshold) / len(real_sigma),
            "sigma_p98_population": "real_rums_all_epochs",
            "model_clip_count": packing_stats["model_clip_count"],
            "model_clip_fraction": packing_stats["model_clip_fraction"],
            "sigma_clip_count": packing_stats["sigma_clip_count"],
            "sigma_clip_fraction": packing_stats["sigma_clip_fraction"],
        },
    }

    write_json(meta_path, meta)

    elapsed = time.time() - t_start

    ok(f"Wrote height texture: {texture_path} ({texture_path.stat().st_size / 1024:.1f} KB)")
    ok(f"Wrote height metadata: {meta_path} ({meta_path.stat().st_size / 1024:.1f} KB)")

    section("Summary")
    ok(f"Step 10 complete in {elapsed:.2f} s")
    print(f"  Texture size           : {texture_width} × {texture_height}")
    print(f"  Real / blank rows      : {real_rum_count} / {blank_count}")
    print(f"  MODEL actual range     : {model_min_actual:.4f} to {model_max_actual:.4f} mm")
    print(f"  Packing vmin/vmax      : {v_min_mm:.4f} to {v_max_mm:.4f} mm")
    print(f"  Sigma actual range     : {sigma_min_actual:.4f} to {sigma_max_actual:.4f} mm")
    print(f"  Sigma packing max      : {sigma_pack_max:.4f} mm")


if __name__ == "__main__":
    main()
