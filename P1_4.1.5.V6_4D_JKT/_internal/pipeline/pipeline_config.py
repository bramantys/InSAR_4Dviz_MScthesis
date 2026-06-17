#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline_config.py

Shared configuration/path resolver for the InSAR4D RUM Viewer pipeline.

This module is NOT a numbered pipeline step. It is imported by:
  - 00_run_pipeline.py
  - 01..18 processing scripts

Purpose
-------
Keep config/project_config.json simple for users, while exposing a resolved
old-style/internal config structure to the existing pipeline scripts.
"""

from __future__ import annotations

import copy
import datetime as dt
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# ADVANCED DEFAULTS
# =============================================================================
# Normal users should edit config/project_config.json only.
# These defaults define the internal pipeline contract and fallback behavior.

DEFAULT_SOURCE_SCHEMA: Dict[str, Dict[str, str]] = {
    "coordinate_fields": {"x": "x_rum", "y": "y_rum"},
    "velocity_fields": {"east": "east", "north": "north", "up": "up"},
    "uncertainty_fields": {
        "var_east": "var_east",
        "var_north": "var_north",
        "var_up": "var_up",
        "covar_en": "covar_en",
        "covar_eu": "covar_eu",
        "covar_nu": "covar_nu",
    },
}

DEFAULT_TILING: Dict[str, Any] = {
    "tile_grid_cols": 8,
    "tile_grid_rows": 6,
    "geometric_error_root": 5000.0,
    "geometric_error_leaf": 100.0,
    "tileset_bound_min_height_m": -1000.0,
    "tileset_bound_max_height_m": 10000.0,
}

DEFAULT_BLANK_CELLS: Dict[str, Any] = {
    "fill_by_row_spans": True,
    "fill_by_col_spans": True,
    "max_fill_passes": 25,
    "outlier_rejection_mm": 100.0,
    "fallback_to_zero": True,
    "enable_block_flattening": False,
    "blank_flatten_block_size": 3,
    "blank_flatten_min_blanks": 5,
}

DEFAULT_HEIGHT_TEXTURE: Dict[str, Any] = {
    "v_min_mm": -1500.0,
    "v_max_mm": 1500.0,
    "packing": "uint16_rg8_plus_sigma_b8",
    "include_blank_cells": True,
    "snr_thin_threshold": 3.0,
    "snr_thick_threshold": 1.0,
}

DEFAULT_SERIES_PACKING: Dict[str, Any] = {
    "vertical_decimals": 2,
    "sigma_decimals": 3,
    "epoch_decimal_decimals": 8,
}


DEFAULT_VERTICAL_UNCERTAINTY_ENCODING: Dict[str, Any] = {
    "type": "lowpoly_checkerboard_spikes",
    "checkerboard_frequency": 4,
    "checkerboard_frequency_near": 4,
    "checkerboard_frequency_far": 2,
    "pyramid_half_base_ratio": 0.28,
    "pyramid_footprint_reference_frequency_near": 6,
    "pyramid_footprint_reference_frequency_far": 2,
    "sigma_multiplier": 1.0,
    "relief_gain": 1.0,
    "cue_mode": "neutral_slope_shade",
    "cue_strength": 1.0,
    "height_ceiling": "global_sigma_p98",
    "above_ceiling": "truncated_square_plateau",
    "visibility_threshold_mode": "global_percentile",
    "visibility_threshold_percentile": 50.0,
    "lod_parent_geometric_error_m": 100.0,
    "lod_on_maximum_screen_space_error": 12.0,
    "lod_force_lowpoly_maximum_screen_space_error": 0.1,
    "lod_refine": "REPLACE",
    "lod_strategy": "semantic_relief_2x2_to_4x4_fixed_6x6_footprint",
}

DEFAULT_CAPS_B3DM: Dict[str, Any] = {
    "geometry_top_height_m": 1.0,
    "bound_min_height_m": -1000.0,
    "bound_max_height_m": 10000.0,
}

DEFAULT_WALLS_B3DM: Dict[str, Any] = {
    "geometry_base_height_m": 0.0,
    "geometry_top_height_m": 1.0,
    "build_north_walls": True,
    "build_east_walls": True,
    "build_outer_walls": False,
    "bound_min_height_m": -1000.0,
    "bound_max_height_m": 10000.0,
}

DEFAULT_HORIZONTAL_DEBUG: Dict[str, Any] = {
    "max_debug_vector_length_m": 350.0,
    "debug_grid_stride": 2,
    "include_top_n_speeds": 50,
    "arrowhead_length_m": 70.0,
}

DEFAULT_HORIZONTAL_UNCERTAINTY: Dict[str, Any] = {
    "speed_floor_mm_yr": 0.50,
    "preferred_shimmer_strength": 0.50,
    "max_recommended_wobble_deg": 35.0,
}

DEFAULT_HORIZONTAL_DEV_LAYERS: Dict[str, Any] = {
    "enabled": True,
    "minimum_speed_mm_yr": 0.2,

    # Automatic scaling is the preferred default.  The manual scale values below
    # remain available when auto_scale is set to false in project_config.json.
    "auto_scale": True,
    "arrow_scale_m_per_mm_yr": 22.5,
    "ellipse_scale_m_per_mm_yr": 22.5,
    "arrow_reference_percentile": 99.5,
    "ellipse_reference_percentile": 99.5,

    # Horizontal glyph layout.  P99.5 speed maps to arrow_max_length_rum_fraction
    # of the RUM size; P99.5 ellipse major axis maps to half of
    # ellipse_max_diameter_rum_fraction of the RUM size.
    "arrow_max_length_rum_fraction": 0.80,
    "ellipse_max_diameter_rum_fraction": 0.75,

    # Arrow anchoring: 0.75 means tail=0%, RUM centre=75%, arrowhead=100%.
    # This lets the confidence ellipse sit at the arrowhead without pushing the
    # whole glyph too far outside the source RUM.
    "arrow_anchor_fraction_at_rum_center": 0.75,

    # Central horizontal uncertainty convention.
    # Fallback only: normal users set these knobs in config/project_config.json.
    # Default viewer rule: show static glyphs when speed exceeds 1σ major
    # uncertainty, and draw confidence ellipses at 2σ.  This keeps the rule
    # project/template-friendly because each dataset brings its own sigma.
    "visibility_sigma_multiplier": 1.0,
    "ellipse_sigma_multiplier": 2.0,
    "ellipse_label": "2σ",
    "visibility_label": "1σ",

    # Backward-compatible aliases used by older configs/scripts.
    "confidence_ellipse_sigma": 2.0,
    "arrow_significance_sigma": 1.0,

    "ellipse_points": 64,
    "ellipse_match_arrow_filter": True,
    "arrow_significance_filter": True,

    # By default the ellipse axes use the same horizontal scale as arrows, so
    # vector magnitude and uncertainty radius are visually comparable.
    # Use "auto_percentile_by_rum_size" to restore the older separate ellipse
    # auto-scaling behaviour.
    "ellipse_scale_mode": "same_as_arrow",
    "ellipse_visual_scale_factor": 1.0,

    "arrowhead_angle_deg": 28.0,
    "arrowhead_frac": 0.22,
    "arrowhead_min_m": 35.0,
    "arrowhead_max_m": 120.0,
    "clearance_above_cap_m": 5.0,
}

DEFAULT_VIEWER_AUTO_TUNING: Dict[str, Any] = {
    "enabled": True,
    "viewer_resources_dirname": "viewer_resources",
    "color_scale": {
        "mode": "robust_percentile",
        "lower_percentile": 2.0,
        "upper_percentile": 98.0,
        "round_to_mm": 10.0,
        "symmetric": False,
    },
    "vertical_exaggeration": {
        "mode": "target_visible_height",
        "target_visible_height_m": 350.0,
        "min_m_per_mm": 5.0,
        "max_m_per_mm": 50.0,
    },
    "camera": {
        "mode": "auto_from_dataset_bbox",
        "height_multiplier": 2.2,
        "min_height_m": 3000.0,
        "max_height_m": 250000.0,
    },
}

DEFAULT_VIEWER: Dict[str, Any] = {
    "dev_viewer": "viz1_dev.html",
    "audience_viewer": "viz1_viewer.html",
    "display_datum_height_m": 1000.0,
    "wall_base_height_m": 999.0,
    "vertical_exaggeration_m_per_mm": 5.0,
}

DEFAULT_PIPELINE_BEHAVIOR: Dict[str, Any] = {
    "clean_pipeline_output_before_run": True,
    "overwrite_existing_outputs": True,
    "stop_on_first_error": True,
    "rum_count_mismatch_action": "warn",
    "rum_size_mismatch_action": "warn",
    "write_human_log": True,
    "write_manifest_json": True,
    "copy_latest_run_records": True,
    "print_step_summary_to_console": True,
}

# Old Step 04-compatible sigma defaults. User chooses only high/medium/low.
SIGMA_QUALITY_PRESETS: Dict[str, Dict[str, Any]] = {
    # These presets control the synthetic epoch-dependent vertical uncertainty
    # generated by Step 04. They deliberately preserve the source sigma ranking
    # while adding smooth, spatially coherent temporal variation for the
    # visualization demonstration.
    "high": {
        "random_seed": 42,
        "source_sigma_floor_mm": 0.12,
        "source_sigma_scale_mm": 1.55,
        "source_sigma_cap_ratio": 1.50,
        "topology_edge_factor": 1.12,
        "topology_isolated_factor": 1.28,
        "seasonal_primary_fraction": 0.18,
        "seasonal_secondary_fraction": 0.08,
        "smooth_noise_fraction": 0.06,
        "spatial_jitter_fraction": 0.08,
        "episode_1_amplitude_mm": 1.00,
        "episode_2_amplitude_mm": 1.30,
        "episode_3_amplitude_mm": 0.90,
        "episode_4_amplitude_mm": 3.60,
        "edge_neighbour_threshold": 5,
        "isolated_neighbour_threshold": 2,
        "sigma_floor_mm": 0.05,
    },
    "medium": {
        "random_seed": 42,
        "source_sigma_floor_mm": 0.20,
        "source_sigma_scale_mm": 2.40,
        "source_sigma_cap_ratio": 1.60,
        "topology_edge_factor": 1.18,
        "topology_isolated_factor": 1.38,
        "seasonal_primary_fraction": 0.24,
        "seasonal_secondary_fraction": 0.11,
        "smooth_noise_fraction": 0.08,
        "spatial_jitter_fraction": 0.10,
        "episode_1_amplitude_mm": 1.60,
        "episode_2_amplitude_mm": 2.10,
        "episode_3_amplitude_mm": 1.40,
        "episode_4_amplitude_mm": 5.20,
        "edge_neighbour_threshold": 5,
        "isolated_neighbour_threshold": 2,
        "sigma_floor_mm": 0.08,
    },
    "low": {
        "random_seed": 42,
        "source_sigma_floor_mm": 0.35,
        "source_sigma_scale_mm": 3.80,
        "source_sigma_cap_ratio": 1.75,
        "topology_edge_factor": 1.25,
        "topology_isolated_factor": 1.55,
        "seasonal_primary_fraction": 0.30,
        "seasonal_secondary_fraction": 0.15,
        "smooth_noise_fraction": 0.11,
        "spatial_jitter_fraction": 0.14,
        "episode_1_amplitude_mm": 2.50,
        "episode_2_amplitude_mm": 3.30,
        "episode_3_amplitude_mm": 2.20,
        "episode_4_amplitude_mm": 7.50,
        "edge_neighbour_threshold": 5,
        "isolated_neighbour_threshold": 2,
        "sigma_floor_mm": 0.12,
    },
}


# Step 02-compatible synthetic epoch defaults.
# These are fallback values only. project_config.json can override them through
# top-level "synthetic_epoch_presets".
DEFAULT_SYNTHETIC_EPOCH_PRESETS: Dict[str, Any] = {
    "vertical_measurement_noise_mm": {
        "low": 2.0,
        "medium": 5.0,
        "high": 8.0,
    },
    "vertical_measurement_behavior": {
        "linear": {
            "base_trend": "from_velocity",
            "velocity_component": "up",
            "start_displacement_at_zero": True,
            "output_unit": "mm",
        },
        "sinusoidal": {
            "base_trend": "from_velocity",
            "velocity_component": "up",
            "amplitude_mm": 5.0,
            "period_days": 365.25,
            "phase_days": 45.0,
            "apply_to": "all_rums",
            "start_displacement_at_zero": True,
            "output_unit": "mm",
        },
    },
    "vertical_model": {
        "linear": {
            "base_trend": "from_velocity",
            "velocity_component": "up",
            "start_displacement_at_zero": True,
            "output_unit": "mm",
        },
        "sinusoidal": {
            "base_trend": "from_velocity",
            "velocity_component": "up",
            "amplitude_mm": 5.0,
            "period_days": 365.25,
            "phase_days": 45.0,
            "apply_to": "all_rums",
            "start_displacement_at_zero": True,
            "output_unit": "mm",
        },
    },
}


# =============================================================================
# BASIC HELPERS
# =============================================================================

def deep_update(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a deep-merged copy of base updated by override."""
    out = copy.deepcopy(base)
    if not override:
        return out
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def rel_to_root(project_root: Path, path: Path) -> str:
    """Return a clean project-relative POSIX path if possible."""
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return path.as_posix()


def resolve_path(project_root: Path, path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else project_root / path


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Dict[str, Any], pretty: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        if pretty:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        else:
            json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))


# Backward-compatible helper name for scripts that want this function.
def resolve_project_path(path_value: str | Path, project_root: Optional[str | Path] = None) -> Path:
    root = Path(project_root) if project_root is not None else find_project_root(__file__)
    return resolve_path(root, path_value)


# =============================================================================
# PROJECT ROOT / CONFIG DISCOVERY
# =============================================================================

def find_project_root(start_file: str | Path) -> Path:
    """Walk upward from start_file until config/project_config.json is found."""
    start = Path(start_file).resolve()
    current = start if start.is_dir() else start.parent
    for candidate in [current, *current.parents]:
        config_path = candidate / "config" / "project_config.json"
        if config_path.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find project root from {start}. Expected config/project_config.json in this folder or a parent."
    )


def get_config_path(project_root: Path) -> Path:
    return project_root / "config" / "project_config.json"


def load_user_config(script_file: str | Path) -> Tuple[Path, Path, Dict[str, Any]]:
    project_root = find_project_root(script_file)
    config_path = get_config_path(project_root)
    user_cfg = load_json(config_path)
    return project_root, config_path, user_cfg


# =============================================================================
# SOURCE / DATASET DERIVATION
# =============================================================================

def derive_project_key(source_file: str | Path) -> str:
    """Derive a compact project key from '<project>_enu_estimates.csv/json/pkl'."""
    stem = Path(source_file).stem
    for suffix in ["_enu_estimates", "_enu_estimate", "_3d_estimates", "_estimates"]:
        if stem.lower().endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def derive_companion_path(source_path: Path, suffix: str) -> Path:
    return source_path.with_suffix(suffix)


def expected_epoch_count(start_date: str, end_date: str, interval_days: int) -> int:
    """Match Step 02 behavior: include start; append end if interval does not land exactly."""
    start = dt.date.fromisoformat(start_date)
    end = dt.date.fromisoformat(end_date)
    if end < start:
        raise ValueError(f"end_date {end_date} is before start_date {start_date}")
    if interval_days <= 0:
        raise ValueError("interval_days must be positive")

    epochs: List[dt.date] = []
    current = start
    step = dt.timedelta(days=int(interval_days))
    while current <= end:
        epochs.append(current)
        current += step
    if epochs[-1] != end:
        epochs.append(end)
    return len(epochs)


def parse_rum_size(value: Any, fallback: float) -> float:
    """Accept numeric values or metadata strings like '450x450m' / '500 x 500 m'."""
    if value is None:
        return float(fallback)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    nums = re.findall(r"[-+]?\d*\.?\d+", str(value))
    if nums:
        return float(nums[0])
    return float(fallback)


def normalized_choice(value: Any, valid: set[str], fallback: str) -> str:
    """Normalize a simple config choice and fall back if empty/unknown."""
    text = str(value if value is not None else fallback).strip().lower()
    return text if text in valid else fallback


def get_nested(mapping: Dict[str, Any], *keys: str, fallback: Any = None) -> Any:
    cur: Any = mapping
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return fallback
        cur = cur[key]
    return cur


def resolve_vertical_epoch_controls(
    synth: Dict[str, Any],
    project_presets: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Resolve the new vertical synthetic-epoch contract.

    Human-facing choices:
      - vertical_measurement_behavior: linear | sinusoidal
      - vertical_measurement_noise   : low | medium | high
      - vertical_model               : linear | sinusoidal
      - uncertainty_quality          : high | medium | low

    Precedence:
      1. user_inputs.synthetic_epochs explicit fields
      2. project_config.json synthetic_epoch_presets
      3. pipeline defaults
    """
    warnings: List[str] = []

    presets = deep_update(DEFAULT_SYNTHETIC_EPOCH_PRESETS, project_presets or {})

    measurement_behavior = str(
        synth.get("vertical_measurement_behavior", "linear")
    ).strip().lower()

    measurement_behavior = normalized_choice(
        measurement_behavior,
        {"linear", "sinusoidal"},
        "linear",
    )

    vertical_model = normalized_choice(
        synth.get("vertical_model", "linear"),
        {"linear", "sinusoidal"},
        "linear",
    )

    measurement_noise = normalized_choice(
        synth.get("vertical_measurement_noise", "low"),
        {"low", "medium", "high"},
        "low",
    )

    noise_presets = get_nested(presets, "vertical_measurement_noise_mm", fallback={}) or {}
    measurement_noise_sigma_mm = float(noise_presets.get(measurement_noise, 0.0))

    measurement_cfg = copy.deepcopy(
        get_nested(presets, "vertical_measurement_behavior", measurement_behavior, fallback={}) or {}
    )
    model_cfg = copy.deepcopy(
        get_nested(presets, "vertical_model", vertical_model, fallback={}) or {}
    )

    # Direct advanced overrides. These are optional; normal users should usually
    # change only the simple choices above.
    if measurement_behavior == "sinusoidal":
        if "sinusoidal_amplitude_mm" in synth:
            measurement_cfg["amplitude_mm"] = float(synth["sinusoidal_amplitude_mm"])
        if "sinusoidal_period_days" in synth:
            measurement_cfg["period_days"] = float(synth["sinusoidal_period_days"])
        if "sinusoidal_phase_days" in synth:
            measurement_cfg["phase_days"] = float(synth["sinusoidal_phase_days"])
        if "sinusoidal_apply_to" in synth:
            measurement_cfg["apply_to"] = str(synth["sinusoidal_apply_to"])

    if "vertical_measurement_noise_sigma_mm" in synth:
        measurement_noise_sigma_mm = float(synth["vertical_measurement_noise_sigma_mm"])

    outlier_probability = float(synth.get("measurement_outlier_probability", 0.0))
    outlier_sigma_mm = float(synth.get("measurement_outlier_sigma_mm", 0.0))

    return {
        "vertical_measurement_behavior": measurement_behavior,
        "vertical_measurement_noise": measurement_noise,
        "vertical_measurement_noise_sigma_mm": measurement_noise_sigma_mm,
        "vertical_measurement": measurement_cfg,
        "vertical_model": vertical_model,
        "vertical_model_config": model_cfg,
        "measurement_outlier_probability": outlier_probability,
        "measurement_outlier_sigma_mm": outlier_sigma_mm,
        "measurement_noise_keep_first_epoch_zero": bool(
            synth.get("measurement_noise_keep_first_epoch_zero", True)
        ),
    }, warnings


def validate_user_inputs(user_inputs: Dict[str, Any], project_root: Path) -> List[str]:
    warnings: List[str] = []
    required = ["source_file", "source_crs", "rum_size_m", "expected_rum_count", "synthetic_epochs"]
    for key in required:
        if key not in user_inputs:
            raise KeyError(f"Missing required config.user_inputs.{key}")

    source_path = resolve_path(project_root, user_inputs["source_file"])
    if not source_path.exists():
        raise FileNotFoundError(f"Configured source_file does not exist: {source_path}")

    if not str(user_inputs.get("source_crs", "")).strip():
        raise ValueError("source_crs is required, e.g. EPSG:23830")

    rum_size = float(user_inputs["rum_size_m"])
    if not math.isfinite(rum_size) or rum_size <= 0:
        raise ValueError("rum_size_m must be a positive number")

    expected_rum_count = user_inputs.get("expected_rum_count")
    if expected_rum_count is not None:
        n = int(expected_rum_count)
        if n <= 0:
            raise ValueError("expected_rum_count must be positive or null")

    epoch_cfg = user_inputs["synthetic_epochs"]

    measurement_behavior = str(epoch_cfg.get("vertical_measurement_behavior", "")).lower()
    if measurement_behavior not in {"linear", "sinusoidal"}:
        raise ValueError("synthetic_epochs.vertical_measurement_behavior must be 'linear' or 'sinusoidal'")

    vertical_model = str(epoch_cfg.get("vertical_model", "linear")).lower()
    if vertical_model not in {"linear", "sinusoidal"}:
        raise ValueError("synthetic_epochs.vertical_model must be 'linear' or 'sinusoidal'")

    measurement_noise = str(epoch_cfg.get("vertical_measurement_noise", "low")).lower()
    if measurement_noise not in {"low", "medium", "high"}:
        raise ValueError("synthetic_epochs.vertical_measurement_noise must be low, medium, or high")

    quality = str(epoch_cfg.get("uncertainty_quality", "")).lower()
    if quality not in SIGMA_QUALITY_PRESETS:
        raise ValueError("synthetic_epochs.uncertainty_quality must be high, medium, or low")

    dt.date.fromisoformat(epoch_cfg["start_date"])
    dt.date.fromisoformat(epoch_cfg["end_date"])
    interval_days = int(epoch_cfg["interval_days"])
    if interval_days <= 0:
        raise ValueError("synthetic_epochs.interval_days must be positive")

    return warnings


# =============================================================================
# RESOLVED CONFIG BUILDER
# =============================================================================

def build_resolved_config(user_cfg: Dict[str, Any], project_root: Path) -> Dict[str, Any]:
    """Convert simple user config into a resolved/internal config."""
    cfg = copy.deepcopy(user_cfg)
    user_inputs = cfg.get("user_inputs")
    if not isinstance(user_inputs, dict):
        raise KeyError("project_config.json must contain a user_inputs block for schema v0.3.")

    validation_warnings = validate_user_inputs(user_inputs, project_root)

    source_path = resolve_path(project_root, user_inputs["source_file"])
    source_rel = rel_to_root(project_root, source_path)
    source_suffix = source_path.suffix.lower()

    project_key = derive_project_key(source_path)
    project_title = cfg.get("project", {}).get("dataset_title") or f"{project_key.title()} 4D RUM Viewer"
    project_label = cfg.get("project", {}).get("dataset_label_short") or project_key.title()

    source_csv = source_path if source_suffix == ".csv" else derive_companion_path(source_path, ".csv")
    source_json = source_path if source_suffix in {".json", ".geojson"} else derive_companion_path(source_path, ".json")
    source_pkl = source_path if source_suffix in {".pkl", ".pickle"} else derive_companion_path(source_path, ".pkl")
    metadata_json = derive_companion_path(source_path, ".json")

    rum_size_m = float(user_inputs["rum_size_m"])
    expected_rums = user_inputs.get("expected_rum_count")

    synth = user_inputs["synthetic_epochs"]
    start_date = str(synth["start_date"])
    end_date = str(synth["end_date"])
    interval_days = int(synth["interval_days"])
    epoch_count = expected_epoch_count(start_date, end_date, interval_days)
    vertical_controls, vertical_control_warnings = resolve_vertical_epoch_controls(
        synth=synth,
        project_presets=cfg.get("synthetic_epoch_presets"),
    )
    validation_warnings.extend(vertical_control_warnings)

    input_data_dir = resolve_path(project_root, cfg.get("paths", {}).get("input_data_dir", "data"))
    internal_dir = resolve_path(project_root, cfg.get("paths", {}).get("internal_dir", "_internal"))
    pipeline_dir = resolve_path(project_root, cfg.get("paths", {}).get("pipeline_dir", "_internal/pipeline"))
    output_dir = resolve_path(project_root, cfg.get("paths", {}).get("pipeline_output_dir", "_internal/data_pipeline"))
    cesium_dir = resolve_path(project_root, cfg.get("paths", {}).get("cesium_dir", "_internal/cesium"))
    run_records_dir = resolve_path(project_root, cfg.get("paths", {}).get("run_records_dir", "run_records"))

    tiles_dir = output_dir / "tiles"
    flat_real_tiles_dir = output_dir / "tiles_flat_real"
    blank_tiles_dir = output_dir / "tiles_blank"
    real_walls_tiles_dir = output_dir / "tiles_walls_real"
    blank_walls_tiles_dir = output_dir / "tiles_walls_blank"
    horizontal_dev_dir = output_dir / "horizontal_dev"
    viewer_resources_dir = output_dir / DEFAULT_VIEWER_AUTO_TUNING["viewer_resources_dirname"]

    # Standard generated files.
    points_plain = output_dir / "points_wgs84.geojson"
    points_with_id = output_dir / "points_wgs84_with_rumid.geojson"
    vertical_base = output_dir / "vertical_epochs_base.json"
    vertical_sigma = output_dir / "vertical_epochs.json"
    footprints = output_dir / "rum_footprints.json"
    packed_series = output_dir / "packed_series.json"
    blank_cells = output_dir / "blank_cells.json"

    height_texture = tiles_dir / "height_texture.png"
    height_meta = tiles_dir / "height_meta.json"
    epoch_axis = tiles_dir / "epoch_axis.json"

    horizontal_field = output_dir / "horizontal_field.json"
    horizontal_particle_field = output_dir / "horizontal_particle_field.json"
    horizontal_debug_vectors = output_dir / "horizontal_debug_vectors.geojson"
    horizontal_uncertainty_check = output_dir / "horizontal_uncertainty_check.json"

    horizontal_arrows_tileset = horizontal_dev_dir / "arrows" / "tileset.json"
    horizontal_ellipses_tileset = horizontal_dev_dir / "ellipses" / "tileset.json"

    # Transitional outputs if old GeoJSON/entity-style Step 17 still exists during migration.
    horizontal_arrows_wgs84 = horizontal_dev_dir / "horizontal_arrows_wgs84.geojson"
    horizontal_ellipses_wgs84 = horizontal_dev_dir / "horizontal_confidence_ellipses_wgs84.geojson"

    viewer_tuning = viewer_resources_dir / "viewer_tuning.json"

    source_schema = deep_update(DEFAULT_SOURCE_SCHEMA, cfg.get("source_schema"))
    viewer_cfg = deep_update(DEFAULT_VIEWER, cfg.get("viewer"))
    height_cfg = deep_update(DEFAULT_HEIGHT_TEXTURE, cfg.get("height_texture") or viewer_cfg.get("height_texture"))
    tiling_cfg = deep_update(DEFAULT_TILING, cfg.get("tiling"))
    blank_cfg = deep_update(DEFAULT_BLANK_CELLS, cfg.get("blank_cells"))
    caps_cfg = deep_update(DEFAULT_CAPS_B3DM, cfg.get("caps_b3dm"))
    vertical_uncertainty_cfg = deep_update(
        DEFAULT_VERTICAL_UNCERTAINTY_ENCODING,
        cfg.get("vertical_uncertainty_encoding"),
    )
    walls_cfg = deep_update(DEFAULT_WALLS_B3DM, cfg.get("walls_b3dm"))
    packing_cfg = deep_update(DEFAULT_SERIES_PACKING, cfg.get("series_packing"))
    hdebug_cfg = deep_update(DEFAULT_HORIZONTAL_DEBUG, cfg.get("horizontal_debug"))
    hunc_cfg = deep_update(DEFAULT_HORIZONTAL_UNCERTAINTY, cfg.get("horizontal_uncertainty"))
    hdev_cfg = deep_update(DEFAULT_HORIZONTAL_DEV_LAYERS, cfg.get("horizontal_dev_layers"))
    tuning_cfg = deep_update(DEFAULT_VIEWER_AUTO_TUNING, cfg.get("viewer_auto_tuning"))
    behavior_cfg = deep_update(DEFAULT_PIPELINE_BEHAVIOR, cfg.get("pipeline_behavior"))

    quality = str(synth["uncertainty_quality"]).lower()
    sigma_cfg = deep_update(SIGMA_QUALITY_PRESETS[quality], cfg.get("sigma_enhancement"))

    cfg["project"] = deep_update(
        {
            "dataset_id": project_key,
            "dataset_title": project_title,
            "dataset_label_short": project_label,
            "source_stem": source_path.stem,
        },
        cfg.get("project"),
    )

    cfg["paths"] = {
        **cfg.get("paths", {}),
        "input_data_dir": rel_to_root(project_root, input_data_dir),
        "internal_dir": rel_to_root(project_root, internal_dir),
        "pipeline_dir": rel_to_root(project_root, pipeline_dir),
        "pipeline_output_dir": rel_to_root(project_root, output_dir),
        "cesium_dir": rel_to_root(project_root, cesium_dir),
        "run_records_dir": rel_to_root(project_root, run_records_dir),
        "data_dir": rel_to_root(project_root, output_dir),
        "raw_source_dir": rel_to_root(project_root, input_data_dir),
        "tiles_dir": rel_to_root(project_root, tiles_dir),
        "flat_real_tiles_dir": rel_to_root(project_root, flat_real_tiles_dir),
        "blank_tiles_dir": rel_to_root(project_root, blank_tiles_dir),
        "real_walls_tiles_dir": rel_to_root(project_root, real_walls_tiles_dir),
        "blank_walls_tiles_dir": rel_to_root(project_root, blank_walls_tiles_dir),
        "horizontal_dev_dir": rel_to_root(project_root, horizontal_dev_dir),
        "viewer_resources_dir": rel_to_root(project_root, viewer_resources_dir),
    }

    cfg["source_inputs"] = {
        "primary_source_csv": rel_to_root(project_root, source_csv),
        "primary_source_json": rel_to_root(project_root, source_json),
        "primary_source_pkl": rel_to_root(project_root, source_pkl),
        "selected_source_file": source_rel,
        "selected_source_extension": source_suffix,
        "metadata_json": rel_to_root(project_root, metadata_json),
        "metadata_json_exists": metadata_json.exists(),
        "source_crs": str(user_inputs["source_crs"]),
        "source_coordinate_fields": source_schema["coordinate_fields"],
        "source_velocity_fields": source_schema["velocity_fields"],
        "source_variance_fields": {
            "var_east": source_schema["uncertainty_fields"]["var_east"],
            "var_north": source_schema["uncertainty_fields"]["var_north"],
            "var_up": source_schema["uncertainty_fields"]["var_up"],
            "covar_en": source_schema["uncertainty_fields"]["covar_en"],
            "covar_eu": source_schema["uncertainty_fields"]["covar_eu"],
            "covar_nu": source_schema["uncertainty_fields"]["covar_nu"],
        },
    }

    cfg["prepared_inputs"] = {
        "plain_points_geojson": rel_to_root(project_root, points_plain),
        "points_geojson": rel_to_root(project_root, points_with_id),
        "vertical_epoch_json_without_enhanced_sigma": rel_to_root(project_root, vertical_base),
        "vertical_epoch_json": rel_to_root(project_root, vertical_sigma),
    }

    cfg["generated_outputs"] = {
        "rum_footprints": rel_to_root(project_root, footprints),
        "packed_series": rel_to_root(project_root, packed_series),
        "blank_cells": rel_to_root(project_root, blank_cells),
        "height_texture": rel_to_root(project_root, height_texture),
        "height_meta": rel_to_root(project_root, height_meta),
        "epoch_axis": rel_to_root(project_root, epoch_axis),
        "horizontal_field": rel_to_root(project_root, horizontal_field),
        "horizontal_particle_field": rel_to_root(project_root, horizontal_particle_field),
        "horizontal_debug_vectors": rel_to_root(project_root, horizontal_debug_vectors),
        "horizontal_uncertainty_check": rel_to_root(project_root, horizontal_uncertainty_check),
        "horizontal_arrows_tileset": rel_to_root(project_root, horizontal_arrows_tileset),
        "horizontal_ellipses_tileset": rel_to_root(project_root, horizontal_ellipses_tileset),
        "horizontal_arrows_wgs84": rel_to_root(project_root, horizontal_arrows_wgs84),
        "horizontal_confidence_ellipses_wgs84": rel_to_root(project_root, horizontal_ellipses_wgs84),
        "viewer_tuning": rel_to_root(project_root, viewer_tuning),
        "real_caps_tileset": rel_to_root(project_root, tiles_dir / "tileset.json"),
        "flat_real_caps_tileset": rel_to_root(project_root, flat_real_tiles_dir / "tileset.json"),
        "blank_caps_tileset": rel_to_root(project_root, blank_tiles_dir / "tileset.json"),
        "real_walls_tileset": rel_to_root(project_root, real_walls_tiles_dir / "tileset.json"),
        "blank_walls_tileset": rel_to_root(project_root, blank_walls_tiles_dir / "tileset.json"),
    }

    cfg["expected_counts"] = {
        "rum_count": expected_rums,
        "epoch_count": epoch_count,
        "grid_spacing_m_nominal": rum_size_m,
        "grid_spacing_tolerance_m": max(50.0, rum_size_m * 0.15),
    }

    measurement_cfg = vertical_controls["vertical_measurement"]
    model_cfg = vertical_controls["vertical_model_config"]

    cfg["epoch_generation"] = {
        "input_mode": "velocity_only",
        "default_start_date": start_date,
        "default_end_date": end_date,
        "default_interval_days": interval_days,
        "default_interval_months_debug": 3,

        # Explicit vertical contract.
        "vertical_measurement_behavior": vertical_controls["vertical_measurement_behavior"],
        "vertical_measurement_noise": vertical_controls["vertical_measurement_noise"],
        "vertical_measurement_noise_sigma_mm": vertical_controls["vertical_measurement_noise_sigma_mm"],
        "vertical_measurement": measurement_cfg,

        "vertical_model": vertical_controls["vertical_model"],
        "vertical_model_config": model_cfg,

        # Step 04 uncertainty/SNR-ish quality, independent from measurement noise.
        "uncertainty_quality": quality,

        "start_displacement_at_zero": True,
        "output_unit": "mm",

        "random_seed": int(synth.get("random_seed", 42)),
        "measurement_noise_sigma_mm": vertical_controls["vertical_measurement_noise_sigma_mm"],
        "measurement_outlier_probability": vertical_controls["measurement_outlier_probability"],
        "measurement_outlier_sigma_mm": vertical_controls["measurement_outlier_sigma_mm"],
        "measurement_noise_keep_first_epoch_zero": vertical_controls["measurement_noise_keep_first_epoch_zero"],

        "round_digits": int(synth.get("round_digits", 4)),
    }


    cfg["sigma_enhancement"] = sigma_cfg
    cfg["tiling"] = tiling_cfg
    cfg["blank_cells"] = blank_cfg
    cfg["height_texture"] = height_cfg
    cfg["series_packing"] = packing_cfg
    cfg["caps_b3dm"] = caps_cfg
    cfg["vertical_uncertainty_encoding"] = vertical_uncertainty_cfg
    cfg["walls_b3dm"] = walls_cfg
    cfg["horizontal_debug"] = hdebug_cfg
    cfg["horizontal_uncertainty"] = hunc_cfg
    cfg["horizontal_dev_layers"] = hdev_cfg
    cfg["viewer_auto_tuning"] = tuning_cfg
    cfg["pipeline_behavior"] = behavior_cfg
    cfg["viewer"] = viewer_cfg

    cfg["_resolved"] = {
        "project_root": project_root.as_posix(),
        "config_path": (project_root / "config" / "project_config.json").as_posix(),
        "project_key": project_key,
        "source_file": source_path.as_posix(),
        "source_file_rel": source_rel,
        "source_extension": source_suffix,
        "source_stem": source_path.stem,
        "metadata_json": metadata_json.as_posix(),
        "metadata_json_exists": metadata_json.exists(),
        "pipeline_output_dir": output_dir.as_posix(),
        "run_records_dir": run_records_dir.as_posix(),
        "viewer_resources_dir": viewer_resources_dir.as_posix(),
        "validation_warnings": validation_warnings,
    }

    return cfg


def load_resolved_config(script_file: str | Path) -> Dict[str, Any]:
    project_root, _config_path, user_cfg = load_user_config(script_file)
    return build_resolved_config(user_cfg, project_root)


# =============================================================================
# DIRECTORY / CLEANUP HELPERS
# =============================================================================

def ensure_standard_dirs(cfg: Dict[str, Any]) -> None:
    project_root = Path(cfg["_resolved"]["project_root"])
    for key in ["pipeline_output_dir", "run_records_dir", "viewer_resources_dir"]:
        ensure_dir(cfg["_resolved"][key])

    for rel_path in [
        cfg["paths"]["tiles_dir"],
        cfg["paths"]["flat_real_tiles_dir"],
        cfg["paths"]["blank_tiles_dir"],
        cfg["paths"]["real_walls_tiles_dir"],
        cfg["paths"]["blank_walls_tiles_dir"],
        cfg["paths"]["horizontal_dev_dir"],
    ]:
        ensure_dir(resolve_path(project_root, rel_path))

    ensure_dir(resolve_path(project_root, cfg["paths"]["run_records_dir"]) / "archive")


def clean_pipeline_output(cfg: Dict[str, Any]) -> None:
    """Remove and recreate _internal/data_pipeline if configured."""
    if not bool(cfg.get("pipeline_behavior", {}).get("clean_pipeline_output_before_run", True)):
        return
    output_dir = Path(cfg["_resolved"]["pipeline_output_dir"])
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_standard_dirs(cfg)


# =============================================================================
# LOG / MANIFEST HELPERS
# =============================================================================

def build_run_metadata(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Small metadata block for 00_run_pipeline.py logs/manifests."""
    ui = cfg["user_inputs"]
    synth = ui["synthetic_epochs"]
    project_root = Path(cfg["_resolved"]["project_root"])
    return {
        "schema_version": cfg.get("schema_version"),
        "project_key": cfg["_resolved"]["project_key"],
        "dataset_title": cfg.get("project", {}).get("dataset_title"),
        "source_file": cfg["_resolved"]["source_file_rel"],
        "source_crs": ui["source_crs"],
        "rum_size_m": ui["rum_size_m"],
        "expected_rum_count": ui["expected_rum_count"],
        "synthetic_epochs": {
            "vertical_measurement_behavior": cfg["epoch_generation"]["vertical_measurement_behavior"],
            "vertical_measurement_noise": cfg["epoch_generation"]["vertical_measurement_noise"],
            "vertical_measurement_noise_sigma_mm": cfg["epoch_generation"]["vertical_measurement_noise_sigma_mm"],
            "vertical_model": cfg["epoch_generation"]["vertical_model"],
            "uncertainty_quality": synth["uncertainty_quality"],
            "start_date": synth["start_date"],
            "end_date": synth["end_date"],
            "interval_days": synth["interval_days"],
            "expected_epoch_count": cfg["expected_counts"]["epoch_count"],
        },
        "paths": {
            "pipeline_output_dir": rel_to_root(project_root, Path(cfg["_resolved"]["pipeline_output_dir"])),
            "run_records_dir": rel_to_root(project_root, Path(cfg["_resolved"]["run_records_dir"])),
            "viewer_tuning": cfg["generated_outputs"]["viewer_tuning"],
        },
    }


def collect_existing_outputs(cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Return existence/size summary for generated_outputs. Useful for manifest."""
    project_root = Path(cfg["_resolved"]["project_root"])
    outputs: Dict[str, Dict[str, Any]] = {}
    for key, rel_path in cfg.get("generated_outputs", {}).items():
        path = resolve_path(project_root, rel_path)
        outputs[key] = {
            "path": rel_path,
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
        }
    return outputs


# =============================================================================
# DEBUG CLI
# =============================================================================

def main() -> None:
    """Manual check: python _internal/pipeline/pipeline_config.py"""
    cfg = load_resolved_config(__file__)
    ensure_standard_dirs(cfg)
    meta = build_run_metadata(cfg)

    print("=" * 72)
    print("pipeline_config.py — resolved configuration check")
    print("=" * 72)
    print(f"Project root      : {cfg['_resolved']['project_root']}")
    print(f"Config path       : {cfg['_resolved']['config_path']}")
    print(f"Project key       : {cfg['_resolved']['project_key']}")
    print(f"Source file       : {meta['source_file']}")
    print(f"Source CRS        : {meta['source_crs']}")
    print(f"RUM size          : {meta['rum_size_m']} m")
    print(f"Expected RUMs     : {meta['expected_rum_count']}")
    print(f"Measurement       : {meta['synthetic_epochs']['vertical_measurement_behavior']}")
    print(f"Measurement noise : {meta['synthetic_epochs']['vertical_measurement_noise']} "
          f"({meta['synthetic_epochs']['vertical_measurement_noise_sigma_mm']} mm)")
    print(f"Vertical model    : {meta['synthetic_epochs']['vertical_model']}")
    print(f"Uncertainty       : {meta['synthetic_epochs']['uncertainty_quality']}")
    print(f"Epoch count       : {meta['synthetic_epochs']['expected_epoch_count']}")
    print(f"Output dir        : {meta['paths']['pipeline_output_dir']}")
    print(f"Run records       : {meta['paths']['run_records_dir']}")
    print(f"Viewer tuning     : {meta['paths']['viewer_tuning']}")
    print("\nPrepared inputs:")
    for key, value in cfg["prepared_inputs"].items():
        print(f"  {key:<42s} {value}")
    print("\nGenerated outputs:")
    for key, value in cfg["generated_outputs"].items():
        print(f"  {key:<42s} {value}")
    print("\nOK")


if __name__ == "__main__":
    main()
