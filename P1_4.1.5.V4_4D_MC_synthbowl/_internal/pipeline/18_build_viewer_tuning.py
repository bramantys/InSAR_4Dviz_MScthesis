#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
18_build_viewer_tuning.py

InSAR4D RUM Viewer pipeline step 18.

Purpose
-------
Build a single data-derived viewer tuning file.

Output
------
  generated_outputs.viewer_tuning
    _internal/data_pipeline/viewer_resources/viewer_tuning.json

Why this step exists
--------------------
The viewer should not hard-code Jakarta/Groningen-specific camera positions,
color ranges, exaggeration, tileset paths, texture paths, or layer defaults.

This step gathers the final pipeline products and writes a viewer-facing JSON
that both viewer tiers can load:

  - Viz4dRUM_dev.html
  - Viz4dRUM_output.html

Design principle
----------------
Data-derived defaults first, viewer override second.

The viewer can still override values in HTML/JS, but this file should provide
sane defaults for a new dataset without manually editing camera/scale knobs.
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
# Normal users should edit config/project_config.json, not this script.

ROUND_COORD_DIGITS = 8
ROUND_NUMERIC_DIGITS = 6

# Camera defaults.
CAMERA_RANGE_MULTIPLIER = 2.4
CAMERA_MIN_RANGE_M = 25000.0
CAMERA_MAX_RANGE_M = 350000.0
CAMERA_DEFAULT_HEADING_DEG = 0.0
CAMERA_DEFAULT_PITCH_DEG = -50.0
CAMERA_DEFAULT_ROLL_DEG = 0.0

# Vertical exaggeration heuristics based on absolute displacement range.
# This is intentionally simple and predictable.
VERTICAL_EXAG_TARGETS = [
    # max_abs_displacement_mm, recommended_m_per_mm
    # Viewer wording: 1x means 1 mm displacement = 1 m display height.
    (20.0, 60.0),
    (50.0, 40.0),
    (150.0, 30.0),
    (500.0, 10.0),
    (999999999.0, 5.0),
]

# If displacement is extremely tiny, avoid overreacting.
VERTICAL_EXAG_MIN = 1.0
VERTICAL_EXAG_MAX = 80.0

# Adaptive diverging colour scale for vertical velocity in mm/yr.
# Zero remains the physical divider: negative = subsidence, positive = uplift.
COLOR_SCALE_UNCERTAINTY_PERCENTILE = 75.0
COLOR_SCALE_PERCENTILE = 98.0
COLOR_SCALE_EXTREME_PERCENTILE = 99.5
COLOR_SCALE_NEAR_ZERO_STEP_MM_YR = 0.5
COLOR_SCALE_MIN_ACTIVE_FRACTION = 0.01
COLOR_SCALE_FALLBACK_LIMIT_MM_YR = 10.0
COLOR_SCALE_MIN_SPAN_MM_YR = 2.0
COLOR_SCALE_ZERO_POSITION_DAMPING = 0.60
COLOR_SCALE_ZERO_POSITION_MIN_PCT = 35.0
COLOR_SCALE_ZERO_POSITION_MAX_PCT = 72.0
COLOR_SCALE_STABLE_BAND_WIDTH_PCT = 16.0

# Viewer display defaults. These are copied to viewer_tuning.json so HTML stays dataset-neutral.
DEFAULT_2D_GLOBAL_OPACITY = 0.70
DEFAULT_3D_GLOBAL_OPACITY = 1.00
DEFAULT_HEIGHT_TEXTURE_ROW_FLIP = True
DEFAULT_BLANK_CAP_COLOR_RGB = [0.28, 0.28, 0.28]
DEFAULT_BLANK_CAP_ALPHA = 0.55
DEFAULT_REAL_WALL_DARKEN = 0.72
DEFAULT_REAL_WALL_ALPHA = 1.00
DEFAULT_BLANK_WALL_COLOR_RGB = [0.22, 0.22, 0.22]
DEFAULT_BLANK_WALL_DARKEN = 0.62
DEFAULT_BLANK_WALL_ALPHA = 0.60
DEFAULT_WALL_SHADE_MIN = 0.45
DEFAULT_WALL_SHADE_MAX = 0.95
DEFAULT_WALL_LIGHT_DIR_EC = [0.25, 0.35, 0.90]

# Horizontal dynamic particle viewer/runtime defaults.
# Scientific sigma/filtering lives in project_config; these are presentation knobs.
DEFAULT_H_PARTICLES_ENABLED_INITIAL = True
DEFAULT_H_PARTICLE_ENGINE_MODE = "primitive_points"
DEFAULT_H_PARTICLE_COUNT = 5000
DEFAULT_H_PARTICLE_SPEED_MULTIPLIER = 1.5
DEFAULT_H_PARTICLE_TRAIL_PERSISTENCE = 0.92
DEFAULT_H_PARTICLE_SIZE_MULTIPLIER = 1.0
DEFAULT_H_PARTICLE_OPACITY = 1.0
DEFAULT_H_PARTICLE_BASE_MPS = 1800.0
DEFAULT_H_PARTICLE_SURFACE_OFFSET_M = 20.0
DEFAULT_H_PARTICLE_STALL_SPEED_MM_YR = 0.05
DEFAULT_H_PARTICLE_MAX_TRAIL_SCREEN_JUMP_PX = 120.0
DEFAULT_H_PARTICLE_CAMERA_STABLE_DELAY_MS = 1
DEFAULT_H_PARTICLE_SAMPLER_MODE = "conservative_v2"

DEFAULT_H_PRIMITIVE_POINTS_PIXEL_SIZE = 5.0
DEFAULT_H_PRIMITIVE_POINTS_OUTLINE_WIDTH = 1.0
DEFAULT_H_PRIMITIVE_POINTS_COLOR_RGB = [0.02, 0.02, 0.02]
DEFAULT_H_PRIMITIVE_POINTS_OUTLINE_RGB = [1.0, 1.0, 1.0]
DEFAULT_H_PRIMITIVE_POINTS_DEBUG_ENABLED = False
DEFAULT_H_PRIMITIVE_POINTS_DEBUG_LOG_INTERVAL_MS = 15000

# Canvas fallback / old shimmer visual defaults. These stay available as a comparison baseline.
DEFAULT_H_CANVAS_UNCERTAINTY_ENABLED_INITIAL = True
DEFAULT_H_CANVAS_UNCERTAINTY_STRENGTH = 0.5
DEFAULT_H_CANVAS_UNCERTAINTY_SPEED_FLOOR_MM_YR = 0.50
DEFAULT_H_CANVAS_UNCERTAINTY_THETA_LOW_DEG = 8.0
DEFAULT_H_CANVAS_UNCERTAINTY_THETA_HIGH_DEG = 32.5
DEFAULT_H_CANVAS_UNCERTAINTY_MAX_WOBBLE_PX = 5.0
DEFAULT_H_CANVAS_UNCERTAINTY_FREQ_MIN_HZ = 0.7
DEFAULT_H_CANVAS_UNCERTAINTY_FREQ_MAX_HZ = 1.6

# Static horizontal arrow/ellipse viewer style.
# Scientific threshold/scaling/placement lives in project_config + Step 17.
# These are viewer presentation uniforms exported by Step 18.
DEFAULT_HORIZONTAL_STATIC_ARROW_COLOR_RGBA = [0.00, 0.00, 0.00, 1.00]
DEFAULT_HORIZONTAL_STATIC_ELLIPSE_COLOR_RGBA = [0.00, 0.00, 0.00, 1.00]
# Shader-side additive colour lift only; not a true halo/glow.
DEFAULT_HORIZONTAL_STATIC_ARROW_LIFT_RGB = [0.00, 0.00, 0.00]
DEFAULT_HORIZONTAL_STATIC_ELLIPSE_LIFT_RGB = [0.00, 0.00, 0.00]
DEFAULT_HORIZONTAL_STATIC_ARROW_OPACITY = 1.00
DEFAULT_HORIZONTAL_STATIC_ELLIPSE_OPACITY = 1.00

# 11-step vertical velocity palette.
# The adaptive VALUE positions are still calculated by Step 18.
# Only the colour ramp changes here.
#
# Contract:
#   0: far-far negative / clipped subsidence, sign-specific P99.5
#   1: far negative, sign-specific P98
#   2: mid negative, halfway between P98 and -tau
#   3: near negative shoulder, just outside -tau
#   4: -tau / stable boundary low, WHITE
#   5: zero reference, WHITE
#   6: +tau / stable boundary high, WHITE
#   7: near positive shoulder, just outside +tau
#   8: mid positive, halfway between +tau and P98
#   9: far positive, sign-specific P98
#  10: far-far positive / clipped uplift, sign-specific P99.5
VERTICAL_COLOR_PALETTE_NAME = "RdBu_11"

VERTICAL_COLOR_PALETTES_11 = {
    "RdBu_11": [
        "#67001f",  # far-far negative / P99.5 clipped subsidence
        "#b2182b",  # far negative / P98 subsidence
        "#ef8a62",  # mid negative
        "#fddbc7",  # near negative shoulder
        "#f7f7f7",  # -tau / stable boundary low
        "#f7f7f7",  # zero reference
        "#f7f7f7",  # +tau / stable boundary high
        "#d1e5f0",  # near positive shoulder
        "#67a9cf",  # mid positive
        "#2166ac",  # far positive / P98 uplift
        "#053061",  # far-far positive / P99.5 clipped uplift
    ],
}


def get_vertical_palette_11() -> List[str]:
    palette = VERTICAL_COLOR_PALETTES_11.get(VERTICAL_COLOR_PALETTE_NAME)
    if palette is None:
        warn(f"Unknown VERTICAL_COLOR_PALETTE_NAME={VERTICAL_COLOR_PALETTE_NAME!r}; using RdBu_11")
        palette = VERTICAL_COLOR_PALETTES_11["RdBu_11"]
    if len(palette) != 11:
        warn(f"Palette {VERTICAL_COLOR_PALETTE_NAME!r} does not have 11 colours; using RdBu_11")
        palette = VERTICAL_COLOR_PALETTES_11["RdBu_11"]
    return list(palette)

# Layer defaults.
DEFAULT_SHOW_REAL_CAPS = True
DEFAULT_SHOW_BLANK_CAPS = True
DEFAULT_SHOW_REAL_WALLS = True
DEFAULT_SHOW_BLANK_WALLS = True
DEFAULT_SHOW_HORIZONTAL_ARROWS_DEV = True
DEFAULT_SHOW_HORIZONTAL_ELLIPSES_DEV = True
DEFAULT_SHOW_HORIZONTAL_ARROWS_OUTPUT = False
DEFAULT_SHOW_HORIZONTAL_ELLIPSES_OUTPUT = False

# If product paths do not exist, keep them in JSON but mark exists=false.
STRICT_REQUIRE_FINAL_PRODUCTS = False


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

def load_json_optional(path: Path, label: str) -> Optional[Dict[str, Any]]:
    if not path.exists():
        warn(f"Missing optional product: {label} -> {path}")
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def rel_path(project_root: Path, path: Path) -> str:
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return path.as_posix()


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


def clamp(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


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


def round_or_none(value: Optional[float], digits: int = ROUND_NUMERIC_DIGITS) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


def deg_to_rad(value: float) -> float:
    return float(value) * math.pi / 180.0


def meters_per_degree_lat() -> float:
    return 111320.0


def meters_per_degree_lon(lat_deg: float) -> float:
    return 111320.0 * math.cos(math.radians(lat_deg))


def bbox_center(bbox: Dict[str, float]) -> Tuple[float, float]:
    lon = (float(bbox["west"]) + float(bbox["east"])) / 2.0
    lat = (float(bbox["south"]) + float(bbox["north"])) / 2.0
    return lon, lat


def bbox_width_height_m(bbox: Dict[str, float]) -> Tuple[float, float]:
    center_lon, center_lat = bbox_center(bbox)
    width_m = abs(float(bbox["east"]) - float(bbox["west"])) * meters_per_degree_lon(center_lat)
    height_m = abs(float(bbox["north"]) - float(bbox["south"])) * meters_per_degree_lat()
    return width_m, height_m


def ceil_to_step(value: float, step: float) -> float:
    """Round a positive value upward to the nearest step."""
    if not math.isfinite(float(value)) or value <= 0 or step <= 0:
        return 0.0
    return float(math.ceil(float(value) / float(step)) * float(step))


def round_near_zero_threshold(tau_raw: float) -> float:
    """Round the near-zero threshold upward to the nearest configured step."""
    return ceil_to_step(tau_raw, COLOR_SCALE_NEAR_ZERO_STEP_MM_YR)


def round_p98_limit(limit_raw: float) -> float:
    """
    Round a positive P98 colour limit upward to a readable value.

    Rule:
      <10 mm/yr   -> nearest 1 mm/yr
      10-20       -> nearest 2 mm/yr
      20-100      -> nearest 5 mm/yr
      >100        -> nearest 10 mm/yr
    """
    if not math.isfinite(float(limit_raw)) or limit_raw <= 0:
        return 0.0

    if limit_raw < 10.0:
        step = 1.0
    elif limit_raw < 20.0:
        step = 2.0
    elif limit_raw <= 100.0:
        step = 5.0
    else:
        step = 10.0

    return ceil_to_step(limit_raw, step)



def derive_extreme_limit(base_limit: float, extreme_raw: float, tau: float) -> float:
    """
    Derive a far-far colour limit, normally from sign-specific P99.5.

    If P99.5 rounds to the same value as P98, nudge it outward so the last
    colour stop still has visible span in the legend and shader ramp.
    """
    base = max(float(base_limit), float(tau), 1e-9)

    if math.isfinite(float(extreme_raw)) and extreme_raw > 0.0:
        extreme = round_p98_limit(float(extreme_raw))
    else:
        extreme = 0.0

    if extreme <= base:
        extreme = round_p98_limit(base * 1.15)

    if extreme <= base:
        extreme = base + max(float(tau), 0.5)

    return float(extreme)

def fmt_limit(value: float) -> str:
    """Readable numeric string for legend labels."""
    v = float(value)
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:g}"


def diverging_position_pct(value: float, vmin: float, vmax: float) -> float:
    """
    Map a velocity value to a 0-100 diverging legend position.

    vmin -> 0, zero -> 50, vmax -> 100.
    """
    value = max(vmin, min(vmax, float(value)))
    if value < 0.0 and vmin < 0.0:
        return 50.0 * (value - vmin) / (0.0 - vmin)
    if value > 0.0 and vmax > 0.0:
        return 50.0 + 50.0 * value / vmax
    return 50.0


def interp(a: float, b: float, f: float) -> float:
    return float(a) + (float(b) - float(a)) * float(f)


def add_color_stop(stops: List[Dict[str, Any]], value: float, color: str, role: str) -> None:
    """Append a stop while avoiding exact duplicate values."""
    value = round(float(value), ROUND_NUMERIC_DIGITS)
    for existing in stops:
        if abs(float(existing["value_mm_yr"]) - value) < 1e-9:
            existing["color"] = color
            existing["role"] = role
            return
    stops.append({"value_mm_yr": value, "color": color, "role": role})


def build_adaptive_vertical_stops_11(
    L_sub: float,
    L_up: float,
    tau: float,
    L_sub_extreme: Optional[float] = None,
    L_up_extreme: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Build an 11-stop adaptive vertical velocity colour scale.

    Value contract:
      - far-far negative: -L_sub_extreme, sign-specific P99.5
      - far negative:     -L_sub, sign-specific P98
      - mid negative:     halfway between -L_sub and -tau
      - near negative:    just outside -tau
      - stable low:       -tau, white
      - zero:             0, white
      - stable high:      +tau, white
      - near positive:    just outside +tau
      - mid positive:     halfway between +tau and +L_up
      - far positive:     +L_up, sign-specific P98
      - far-far positive: +L_up_extreme, sign-specific P99.5
    """
    L_sub = max(float(L_sub), float(tau), 1e-9)
    L_up = max(float(L_up), float(tau), 1e-9)
    tau = max(float(tau), 0.0)

    if L_sub_extreme is None:
        L_sub_extreme = derive_extreme_limit(L_sub, 0.0, tau)
    if L_up_extreme is None:
        L_up_extreme = derive_extreme_limit(L_up, 0.0, tau)

    L_sub_extreme = max(float(L_sub_extreme), L_sub)
    L_up_extreme = max(float(L_up_extreme), L_up)

    c = get_vertical_palette_11()

    # Keep the asymmetric visual layout from the previous versions, but add
    # two shoulder stops around the white stable plateau.
    zero_raw = 100.0 * L_sub / max(L_sub + L_up, 1e-9)
    zero_damped = 50.0 + COLOR_SCALE_ZERO_POSITION_DAMPING * (zero_raw - 50.0)
    zero_pct = clamp(zero_damped, COLOR_SCALE_ZERO_POSITION_MIN_PCT, COLOR_SCALE_ZERO_POSITION_MAX_PCT)
    stable_width = clamp(COLOR_SCALE_STABLE_BAND_WIDTH_PCT, 10.0, 22.0)
    stable_left = clamp(zero_pct - stable_width / 2.0, 2.0, 96.0)
    stable_right = clamp(zero_pct + stable_width / 2.0, 4.0, 98.0)
    if stable_right <= stable_left:
        stable_right = min(98.0, stable_left + stable_width)

    neg_mid_mag = interp(L_sub, tau, 0.50)
    pos_mid_value = interp(tau, L_up, 0.50)

    # Shoulder values just outside the stable band. For tau=1 this gives
    # approximately -3 and +2, matching the Jakarta visual target.
    near_neg_mag = min(L_sub, max(3.0 * tau, tau + 2.0 * COLOR_SCALE_NEAR_ZERO_STEP_MM_YR))
    near_pos_value = min(L_up, max(2.0 * tau, tau + 2.0 * COLOR_SCALE_NEAR_ZERO_STEP_MM_YR))

    # Convert actual values into visual legend percentages on each side.
    # This keeps colourbar positions consistent even when zero is asymmetric.
    def neg_pct_for_mag(mag: float) -> float:
        denom = max(L_sub_extreme - tau, 1e-9)
        return stable_left * clamp((L_sub_extreme - float(mag)) / denom, 0.0, 1.0)

    def pos_pct_for_value(value: float) -> float:
        denom = max(L_up_extreme - tau, 1e-9)
        return stable_right + (100.0 - stable_right) * clamp((float(value) - tau) / denom, 0.0, 1.0)

    stops = [
        {"value_mm_yr": -L_sub_extreme, "color": c[0],  "role": "clipped_extreme_subsidence", "position_pct": 0.0},
        {"value_mm_yr": -L_sub,         "color": c[1],  "role": "far_subsidence",             "position_pct": neg_pct_for_mag(L_sub)},
        {"value_mm_yr": -neg_mid_mag,   "color": c[2],  "role": "moderate_subsidence",        "position_pct": neg_pct_for_mag(neg_mid_mag)},
        {"value_mm_yr": -near_neg_mag,  "color": c[3],  "role": "near_stable_subsidence",     "position_pct": neg_pct_for_mag(near_neg_mag)},
        {"value_mm_yr": -tau,           "color": c[4],  "role": "stable_boundary_low",        "position_pct": stable_left},
        {"value_mm_yr": 0.0,            "color": c[5],  "role": "zero_reference",             "position_pct": zero_pct},
        {"value_mm_yr": tau,            "color": c[6],  "role": "stable_boundary_high",       "position_pct": stable_right},
        {"value_mm_yr": near_pos_value, "color": c[7],  "role": "near_stable_uplift",         "position_pct": pos_pct_for_value(near_pos_value)},
        {"value_mm_yr": pos_mid_value,  "color": c[8],  "role": "moderate_uplift",            "position_pct": pos_pct_for_value(pos_mid_value)},
        {"value_mm_yr": L_up,           "color": c[9],  "role": "far_uplift",                 "position_pct": pos_pct_for_value(L_up)},
        {"value_mm_yr": L_up_extreme,   "color": c[10], "role": "clipped_extreme_uplift",     "position_pct": 100.0},
    ]

    return sorted(
        [
            {
                "value_mm_yr": round(float(s["value_mm_yr"]), ROUND_NUMERIC_DIGITS),
                "color": s["color"],
                "role": s["role"],
                "position_pct": round(float(s["position_pct"]), 3),
            }
            for s in stops
        ],
        key=lambda item: item["value_mm_yr"],
    )


def build_adaptive_legend(
    L_sub: float,
    L_up: float,
    tau: float,
    L_sub_extreme: Optional[float] = None,
    L_up_extreme: Optional[float] = None,
) -> Dict[str, Any]:
    stops = build_adaptive_vertical_stops_11(L_sub, L_up, tau, L_sub_extreme, L_up_extreme)
    by_role = {s["role"]: s for s in stops}
    zero_pct = by_role.get("zero_reference", {}).get("position_pct", 50.0)
    stable_left = by_role.get("stable_boundary_low", {}).get("position_pct", 42.0)
    stable_right = by_role.get("stable_boundary_high", {}).get("position_pct", 58.0)
    far_sub = by_role.get("far_subsidence", {})
    neg_mid = by_role.get("moderate_subsidence", {})
    pos_mid = by_role.get("moderate_uplift", {})
    far_up = by_role.get("far_uplift", {})

    # Seven readable labels for the compact UI bar.
    # Do not label the 0% and 100% extreme P99.5 caps; they remain in
    # color_stops only as dark end caps. The visible far labels sit at the
    # P98 far_subsidence/far_uplift positions instead.
    far_sub_value = float(far_sub.get("value_mm_yr", -float(L_sub)))
    far_up_value = float(far_up.get("value_mm_yr", float(L_up)))
    labels = [
        {
            "value_mm_yr": far_sub_value,
            "label": f"≤−{fmt_limit(abs(far_sub_value))}",
            "position_pct": far_sub.get("position_pct", 6.0),
            "role": "far_minus_label",
        },
        {
            "value_mm_yr": neg_mid.get("value_mm_yr", -float(L_sub)),
            "label": f"−{fmt_limit(abs(float(neg_mid.get('value_mm_yr', -float(L_sub)))))}",
            "position_pct": neg_mid.get("position_pct", stable_left * 0.5),
            "role": "mid_minus_label",
        },
        {"value_mm_yr": -float(tau), "label": f"−{fmt_limit(tau)}", "position_pct": stable_left, "role": "stable_min_label"},
        {"value_mm_yr": 0.0, "label": "0", "position_pct": zero_pct, "role": "zero_label"},
        {"value_mm_yr": float(tau), "label": f"+{fmt_limit(tau)}", "position_pct": stable_right, "role": "stable_plus_label"},
        {
            "value_mm_yr": pos_mid.get("value_mm_yr", float(L_up)),
            "label": f"+{fmt_limit(float(pos_mid.get('value_mm_yr', float(L_up))))}",
            "position_pct": pos_mid.get("position_pct", stable_right + (100.0 - stable_right) * 0.5),
            "role": "mid_plus_label",
        },
        {
            "value_mm_yr": far_up_value,
            "label": f"≥+{fmt_limit(far_up_value)}",
            "position_pct": far_up.get("position_pct", 90.0),
            "role": "far_plus_label",
        },
    ]

    return {
        "title": "Vertical Velocity [mm/yr]",
        "unit": "mm/yr",
        "labels": [
            {
                "value_mm_yr": round(float(item["value_mm_yr"]), ROUND_NUMERIC_DIGITS),
                "label": item["label"],
                "position_pct": round(float(item["position_pct"]), 3),
                "role": item.get("role"),
            }
            for item in labels
        ],
        "readability_note": (
            "Seven labels are shown for readability: P98 far minus, mid minus, stable min, zero, stable plus, mid plus, P98 far plus. "
            "P99.5 extremes and shoulder stops remain encoded in color_stops but are not labelled at 0% or 100%."
        ),
    }


def adaptive_legend_layout(
    L_sub: float,
    L_up: float,
    tau: float,
    L_sub_extreme: Optional[float] = None,
    L_up_extreme: Optional[float] = None,
) -> Dict[str, Any]:
    stops = build_adaptive_vertical_stops_11(L_sub, L_up, tau, L_sub_extreme, L_up_extreme)
    by_role = {s["role"]: s for s in stops}
    raw_zero = 100.0 * float(L_sub) / max(float(L_sub) + float(L_up), 1e-9)
    damped_zero = 50.0 + COLOR_SCALE_ZERO_POSITION_DAMPING * (raw_zero - 50.0)
    return {
        "mode": "asymmetric_zero_with_stable_plateau_damped_11step",
        "zero_position_pct": by_role.get("zero_reference", {}).get("position_pct"),
        "raw_zero_position_pct": round(raw_zero, 3),
        "damped_zero_position_pct": round(damped_zero, 3),
        "zero_position_damping": COLOR_SCALE_ZERO_POSITION_DAMPING,
        "zero_position_min_pct": COLOR_SCALE_ZERO_POSITION_MIN_PCT,
        "zero_position_max_pct": COLOR_SCALE_ZERO_POSITION_MAX_PCT,
        "stable_band_width_pct": COLOR_SCALE_STABLE_BAND_WIDTH_PCT,
        "stable_left_pct": by_role.get("stable_boundary_low", {}).get("position_pct"),
        "stable_right_pct": by_role.get("stable_boundary_high", {}).get("position_pct"),
        "far_subsidence_pct": by_role.get("far_subsidence", {}).get("position_pct"),
        "near_stable_subsidence_pct": by_role.get("near_stable_subsidence", {}).get("position_pct"),
        "near_stable_uplift_pct": by_role.get("near_stable_uplift", {}).get("position_pct"),
        "far_uplift_pct": by_role.get("far_uplift", {}).get("position_pct"),
        "extreme_subsidence_pct": by_role.get("clipped_extreme_subsidence", {}).get("position_pct"),
        "extreme_uplift_pct": by_role.get("clipped_extreme_uplift", {}).get("position_pct"),
        "note": "Legend positions are visual; shader colours remain value-based. P98 and shoulder stops add colour richness around a strict white stable plateau; P99.5 controls the clipped end colours.",
    }


def choose_vertical_exaggeration(max_abs_displacement_mm: float) -> float:
    for threshold, exag in VERTICAL_EXAG_TARGETS:
        if max_abs_displacement_mm <= threshold:
            return max(VERTICAL_EXAG_MIN, min(VERTICAL_EXAG_MAX, exag))
    return 10.0


# =============================================================================
# PRODUCT SUMMARIES
# =============================================================================

def product_ref(project_root: Path, path: Path) -> Dict[str, Any]:
    return {
        "path": rel_path(project_root, path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
    }


def extract_dataset_bbox(
    footprints: Optional[Dict[str, Any]],
    tile_index: Optional[Dict[str, Any]],
    validation: Optional[Dict[str, Any]],
) -> Optional[Dict[str, float]]:
    candidates = []

    if footprints:
        meta = footprints.get("metadata") or {}
        candidates.append(meta.get("bbox_wgs84_footprints"))
        candidates.append(meta.get("bbox_wgs84_centers"))

    if tile_index:
        meta = tile_index.get("metadata") or {}
        candidates.append(meta.get("dataset_bbox_wgs84"))

    if validation:
        summary = validation.get("summary") or {}
        candidates.append(summary.get("footprint_bbox_wgs84"))
        candidates.append(summary.get("point_bbox_wgs84"))

    for c in candidates:
        if isinstance(c, dict) and all(k in c for k in ["west", "south", "east", "north"]):
            return {
                "west": round(float(c["west"]), ROUND_COORD_DIGITS),
                "south": round(float(c["south"]), ROUND_COORD_DIGITS),
                "east": round(float(c["east"]), ROUND_COORD_DIGITS),
                "north": round(float(c["north"]), ROUND_COORD_DIGITS),
            }

    return None


def extract_epoch_defaults(epoch_axis: Optional[Dict[str, Any]], height_meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if epoch_axis:
        meta = epoch_axis.get("metadata") or {}
        epochs = epoch_axis.get("epochs", [])
        return {
            "epoch_count": meta.get("epoch_count", len(epochs)),
            "first_epoch": meta.get("first_epoch", epochs[0] if epochs else None),
            "last_epoch": meta.get("last_epoch", epochs[-1] if epochs else None),
            "default_epoch_index": meta.get("default_epoch_index", max(0, len(epochs) - 1) if epochs else 0),
            "default_epoch": meta.get("default_epoch", epochs[-1] if epochs else None),
            "epochs_path_hint": "tiles/epoch_axis.json",
        }

    if height_meta:
        epochs = height_meta.get("epochs", [])
        return {
            "epoch_count": len(epochs),
            "first_epoch": epochs[0] if epochs else None,
            "last_epoch": epochs[-1] if epochs else None,
            "default_epoch_index": max(0, len(epochs) - 1) if epochs else 0,
            "default_epoch": epochs[-1] if epochs else None,
            "epochs_path_hint": "tiles/epoch_axis.json",
        }

    return {
        "epoch_count": None,
        "first_epoch": None,
        "last_epoch": None,
        "default_epoch_index": 0,
        "default_epoch": None,
        "epochs_path_hint": "tiles/epoch_axis.json",
    }


def extract_vertical_summary(
    packed: Optional[Dict[str, Any]],
    height_meta: Optional[Dict[str, Any]],
    validation: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Extract vertical MEASUREMENT / MODEL / sigma summaries.

    Clean contract:
      - measurement_mm lives in packed_series and is for trendline / popup / labelling.
      - model_mm is packed into the height texture and is for RUM height / caps / walls.
      - sigma_mm is packed into the height texture B channel for uncertainty/hatch display.

    Prefer height_meta for MODEL and sigma because it includes blank rows and
    final texture packing. Prefer packed metadata for MEASUREMENT because blank
    cells intentionally do not have measurement_mm.
    """
    out: Dict[str, Any] = {
        "measurement": {},
        "model": {},
        "sigma": {},
        "texture": {},
    }

    if packed:
        psummary = (packed.get("metadata") or {}).get("summary") or {}
        out["measurement"].update({
            "min_mm": psummary.get("measurement_min_mm"),
            "p02_mm": psummary.get("measurement_p02_mm"),
            "p50_mm": psummary.get("measurement_p50_mm"),
            "p98_mm": psummary.get("measurement_p98_mm"),
            "max_mm": psummary.get("measurement_max_mm"),
            "source": "packed_series.metadata.summary",
            "role": "trendline_popup_labelling",
        })
        out["model"].update({
            "min_mm": psummary.get("model_min_mm"),
            "p02_mm": psummary.get("model_p02_mm"),
            "p50_mm": psummary.get("model_p50_mm"),
            "p98_mm": psummary.get("model_p98_mm"),
            "max_mm": psummary.get("model_max_mm"),
            "source": "packed_series.metadata.summary",
            "role": "rum_height_choropleth_caps_walls_blankies",
        })
        out["sigma"].update({
            "min_mm": psummary.get("sigma_min_mm"),
            "p02_mm": psummary.get("sigma_p02_mm"),
            "p50_mm": psummary.get("sigma_p50_mm"),
            "p98_mm": psummary.get("sigma_p98_mm"),
            "max_mm": psummary.get("sigma_max_mm"),
            "source": "packed_series.metadata.summary",
            "role": "uncertainty_snr_hatch_visualization",
        })

    # Height meta is the most important source for MODEL/sigma display because
    # it is the exact texture actually sampled by caps/walls/arrows.
    if height_meta:
        summary = height_meta.get("summary") or {}
        packing = height_meta.get("packing") or {}
        model_pack = packing.get("model") or {}
        sigma_pack = packing.get("sigma") or {}

        out["model"].update({
            "min_mm": summary.get("model_min_actual_mm"),
            "p02_mm": summary.get("model_p02_actual_mm"),
            "p50_mm": summary.get("model_p50_actual_mm"),
            "p98_mm": summary.get("model_p98_actual_mm"),
            "max_mm": summary.get("model_max_actual_mm"),
            "clip_fraction": summary.get("model_clip_fraction"),
            "source": "height_meta.summary",
            "role": "rum_height_choropleth_caps_walls_blankies",
        })
        out["sigma"].update({
            "min_mm": summary.get("sigma_min_actual_mm"),
            "p02_mm": summary.get("sigma_p02_actual_mm"),
            "p50_mm": summary.get("sigma_p50_actual_mm"),
            "p98_mm": summary.get("sigma_p98_actual_mm"),
            "max_mm": summary.get("sigma_max_actual_mm"),
            "clip_fraction": summary.get("sigma_clip_fraction"),
            "packing_max_mm": sigma_pack.get("sigma_max_mm"),
            "source": "height_meta.summary",
            "role": "uncertainty_snr_hatch_visualization",
        })
        out["texture"].update({
            "model_v_min_mm": model_pack.get("v_min_mm"),
            "model_v_max_mm": model_pack.get("v_max_mm"),
            "sigma_max_mm": sigma_pack.get("sigma_max_mm"),
            "height_texture_source": "model_mm_plus_sigma_mm",
            "source": "height_meta.packing",
        })

    # Validation report is a fallback only.
    if validation and not out["model"].get("min_mm"):
        summary = validation.get("summary") or {}
        out["measurement"].update({
            "min_mm": summary.get("measurement_min_mm"),
            "p02_mm": summary.get("measurement_p02_mm"),
            "p50_mm": summary.get("measurement_p50_mm"),
            "p98_mm": summary.get("measurement_p98_mm"),
            "max_mm": summary.get("measurement_max_mm"),
            "source": "validation_report.summary",
            "role": "trendline_popup_labelling",
        })
        out["model"].update({
            "min_mm": summary.get("model_min_mm"),
            "p02_mm": summary.get("model_p02_mm"),
            "p50_mm": summary.get("model_p50_mm"),
            "p98_mm": summary.get("model_p98_mm"),
            "max_mm": summary.get("model_max_mm"),
            "source": "validation_report.summary",
            "role": "rum_height_choropleth_caps_walls_blankies",
        })
        out["sigma"].update({
            "min_mm": summary.get("sigma_min_mm"),
            "p02_mm": summary.get("sigma_p02_mm"),
            "p50_mm": summary.get("sigma_p50_mm"),
            "p98_mm": summary.get("sigma_p98_mm"),
            "max_mm": summary.get("sigma_max_mm"),
            "source": "validation_report.summary",
            "role": "uncertainty_snr_hatch_visualization",
        })

    return out


def extract_velocity_summary(horizontal_field: Optional[Dict[str, Any]], packed: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if horizontal_field:
        summary = ((horizontal_field.get("metadata") or {}).get("summary")) or {}
        records = horizontal_field.get("records", [])
        up_values = [safe_float(r.get("up_mm_yr")) for r in records]
        up_values = [v for v in up_values if v is not None]
        return {
            "east_min_mm_yr": summary.get("east_min_mm_yr"),
            "east_max_mm_yr": summary.get("east_max_mm_yr"),
            "north_min_mm_yr": summary.get("north_min_mm_yr"),
            "north_max_mm_yr": summary.get("north_max_mm_yr"),
            "speed_min_mm_yr": summary.get("speed_min_mm_yr"),
            "speed_p50_mm_yr": summary.get("speed_p50_mm_yr"),
            "speed_p98_mm_yr": summary.get("speed_p98_mm_yr"),
            "speed_max_mm_yr": summary.get("speed_max_mm_yr"),
            "up_min_mm_yr": min(up_values) if up_values else summary.get("up_min_mm_yr"),
            "up_p02_mm_yr": percentile(up_values, 2) if up_values else None,
            "up_p50_mm_yr": percentile(up_values, 50) if up_values else None,
            "up_p98_mm_yr": percentile(up_values, 98) if up_values else None,
            "up_max_mm_yr": max(up_values) if up_values else summary.get("up_max_mm_yr"),
        }

    if packed:
        per_rum = packed.get("per_rum") or {}
        up_values = [safe_float(v) for v in per_rum.get("source_up_mm_yr", [])]
        up_values = [v for v in up_values if v is not None]
        if up_values:
            return {
                "up_min_mm_yr": min(up_values),
                "up_p02_mm_yr": percentile(up_values, 2),
                "up_p50_mm_yr": percentile(up_values, 50),
                "up_p98_mm_yr": percentile(up_values, 98),
                "up_max_mm_yr": max(up_values),
            }

    return {}


def extract_blank_summary(blank_cells: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not blank_cells:
        return {"blank_count": None, "status": "unknown"}

    meta = blank_cells.get("metadata") or {}
    return {
        "blank_count": meta.get("blank_count", len(blank_cells.get("features", []))),
        "status": meta.get("status", "unknown"),
        "interior_rule": meta.get("interior_rule"),
        "real_rum_count": meta.get("real_rum_count"),
    }


def derive_color_scale(
    horizontal_field: Optional[Dict[str, Any]],
    velocity_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Derive adaptive diverging vertical velocity colour scale.

    Rule:
      tau = ceil_0.5(P75(2 * sqrt(var_up)))
      L_sub = readable_round(P98(abs(up[up < -tau])))
      L_up  = readable_round(P98(up[up > +tau]))

    The colour scale is centred at 0 mm/yr and the two sides are scaled
    independently. Values beyond the limits are clipped to the darkest end
    colours, but their actual values remain in the data/popup.
    """
    records = (horizontal_field or {}).get("records") or []

    up_values: List[float] = []
    var_up_values: List[float] = []

    for rec in records:
        up = safe_float(rec.get("up_mm_yr"))
        cov = rec.get("covariance") or {}
        var_up = safe_float(cov.get("var_up"))

        if up is None or var_up is None or var_up < 0.0:
            continue

        up_values.append(float(up))
        var_up_values.append(float(var_up))

    # Fallback: keep Step 18 usable even if horizontal_field or var_up is missing.
    if not up_values or not var_up_values:
        up_p02 = safe_float(velocity_summary.get("up_p02_mm_yr"))
        up_p98 = safe_float(velocity_summary.get("up_p98_mm_yr"))
        up_min = safe_float(velocity_summary.get("up_min_mm_yr"))
        up_max = safe_float(velocity_summary.get("up_max_mm_yr"))

        lower = up_p02 if up_p02 is not None else up_min
        upper = up_p98 if up_p98 is not None else up_max
        if lower is None or upper is None:
            lower, upper = -COLOR_SCALE_FALLBACK_LIMIT_MM_YR, COLOR_SCALE_FALLBACK_LIMIT_MM_YR

        L_sub = round_p98_limit(abs(float(lower)))
        L_up = round_p98_limit(abs(float(upper)))
        if L_sub <= 0:
            L_sub = COLOR_SCALE_FALLBACK_LIMIT_MM_YR
        if L_up <= 0:
            L_up = COLOR_SCALE_FALLBACK_LIMIT_MM_YR
        tau = min(0.5, L_sub, L_up)
        L_sub_extreme = derive_extreme_limit(L_sub, abs(float(up_min)) if up_min is not None else 0.0, tau)
        L_up_extreme = derive_extreme_limit(L_up, abs(float(up_max)) if up_max is not None else 0.0, tau)

        return {
            "mode": "adaptive_asymmetric_11step_fallback_no_var_up",
            "velocity_field": "up_mm_yr",
            "unit": "mm/yr",
            "recommended_min_mm_yr": -float(L_sub_extreme),
            "recommended_center_mm_yr": 0.0,
            "recommended_max_mm_yr": float(L_up_extreme),
            "near_zero_threshold_mm_yr": float(tau),
            "L_sub_mm_yr": float(L_sub),
            "L_up_mm_yr": float(L_up),
            "L_sub_extreme_mm_yr": float(L_sub_extreme),
            "L_up_extreme_mm_yr": float(L_up_extreme),
            "palette_name": VERTICAL_COLOR_PALETTE_NAME,
            "palette_colours": get_vertical_palette_11(),
            "color_stop_count": 11,
            "color_stops": build_adaptive_vertical_stops_11(L_sub, L_up, tau, L_sub_extreme, L_up_extreme),
            "legend_layout": adaptive_legend_layout(L_sub, L_up, tau, L_sub_extreme, L_up_extreme),
            "legend": build_adaptive_legend(L_sub, L_up, tau, L_sub_extreme, L_up_extreme),
            "data_p02_mm_yr": round_or_none(up_p02),
            "data_p98_mm_yr": round_or_none(up_p98),
            "data_min_mm_yr": round_or_none(up_min),
            "data_max_mm_yr": round_or_none(up_max),
            "note": "Fallback because horizontal_field.records covariance.var_up was missing or unusable.",
        }

    n_valid = len(up_values)
    two_sigma = [2.0 * math.sqrt(v) for v in var_up_values if v >= 0.0]
    tau_raw = percentile(two_sigma, COLOR_SCALE_UNCERTAINTY_PERCENTILE)
    tau = round_near_zero_threshold(tau_raw or 0.0)
    if tau <= 0.0:
        tau = COLOR_SCALE_NEAR_ZERO_STEP_MM_YR

    sub_values = [v for v in up_values if v < -tau]
    uplift_values = [v for v in up_values if v > tau]

    n_sub = len(sub_values)
    n_up = len(uplift_values)

    neg_active = (n_sub / max(n_valid, 1)) >= COLOR_SCALE_MIN_ACTIVE_FRACTION
    pos_active = (n_up / max(n_valid, 1)) >= COLOR_SCALE_MIN_ACTIVE_FRACTION

    if neg_active:
        abs_sub_values = [abs(v) for v in sub_values]
        L_sub_raw = percentile(abs_sub_values, COLOR_SCALE_PERCENTILE) or 0.0
        L_sub_far_raw = percentile(abs_sub_values, COLOR_SCALE_EXTREME_PERCENTILE) or L_sub_raw
        L_sub = round_p98_limit(L_sub_raw)
    else:
        L_sub_raw = 0.0
        L_sub_far_raw = 0.0
        L_sub = 0.0

    if pos_active:
        L_up_raw = percentile(uplift_values, COLOR_SCALE_PERCENTILE) or 0.0
        L_up_far_raw = percentile(uplift_values, COLOR_SCALE_EXTREME_PERCENTILE) or L_up_raw
        L_up = round_p98_limit(L_up_raw)
    else:
        L_up_raw = 0.0
        L_up_far_raw = 0.0
        L_up = 0.0

    # Keep the diverging scale valid for one-sided or mostly stable datasets.
    if neg_active and not pos_active:
        # Keep an uplift side for a valid diverging legend, but do not mirror the
        # full subsidence range. This preserves the visual message that the data
        # are subsidence-dominant.
        L_up = max(5.0 * tau, tau, COLOR_SCALE_MIN_SPAN_MM_YR)
    elif pos_active and not neg_active:
        # Symmetric idea for uplift-dominant datasets.
        L_sub = max(5.0 * tau, tau, COLOR_SCALE_MIN_SPAN_MM_YR)
    elif not neg_active and not pos_active:
        L_sub = max(5.0 * tau, tau, COLOR_SCALE_FALLBACK_LIMIT_MM_YR)
        L_up = max(5.0 * tau, tau, COLOR_SCALE_FALLBACK_LIMIT_MM_YR)

    # Final safety.
    L_sub = max(float(L_sub), float(tau), 1e-9)
    L_up = max(float(L_up), float(tau), 1e-9)
    L_sub_extreme = derive_extreme_limit(L_sub, L_sub_far_raw, tau)
    L_up_extreme = derive_extreme_limit(L_up, L_up_far_raw, tau)

    vmin = -L_sub_extreme
    vmax = L_up_extreme

    return {
        "mode": "adaptive_asymmetric_11step",
        "velocity_field": "up_mm_yr",
        "variance_field": "covariance.var_up",
        "unit": "mm/yr",
        "interpretation": {
            "negative": "subsidence",
            "zero": "stable / no vertical motion reference",
            "positive": "uplift",
            "red_side": "subsidence",
            "blue_side": "uplift",
            "stable_band": f"|up| ≤ {fmt_limit(tau)} mm/yr is the near-zero/stable reference band",
            "palette": VERTICAL_COLOR_PALETTE_NAME,
        },
        "recommended_min_mm_yr": round(float(vmin), ROUND_NUMERIC_DIGITS),
        "recommended_center_mm_yr": 0.0,
        "recommended_max_mm_yr": round(float(vmax), ROUND_NUMERIC_DIGITS),
        "near_zero_threshold_mm_yr": round(float(tau), ROUND_NUMERIC_DIGITS),
        "L_sub_raw_mm_yr": round_or_none(L_sub_raw),
        "L_up_raw_mm_yr": round_or_none(L_up_raw),
        "L_sub_far_raw_mm_yr": round_or_none(L_sub_far_raw),
        "L_up_far_raw_mm_yr": round_or_none(L_up_far_raw),
        "L_sub_mm_yr": round(float(L_sub), ROUND_NUMERIC_DIGITS),
        "L_up_mm_yr": round(float(L_up), ROUND_NUMERIC_DIGITS),
        "L_sub_extreme_mm_yr": round(float(L_sub_extreme), ROUND_NUMERIC_DIGITS),
        "L_up_extreme_mm_yr": round(float(L_up_extreme), ROUND_NUMERIC_DIGITS),
        "tau_raw_mm_yr": round_or_none(tau_raw),
        "tau_mm_yr": round(float(tau), ROUND_NUMERIC_DIGITS),
        "neg_active": bool(neg_active),
        "pos_active": bool(pos_active),
        "n_valid": int(n_valid),
        "n_sub": int(n_sub),
        "n_up": int(n_up),
        "uncertainty_percentile": COLOR_SCALE_UNCERTAINTY_PERCENTILE,
        "colour_percentile": COLOR_SCALE_PERCENTILE,
        "extreme_colour_percentile": COLOR_SCALE_EXTREME_PERCENTILE,
        "min_active_fraction": COLOR_SCALE_MIN_ACTIVE_FRACTION,
        "palette_name": VERTICAL_COLOR_PALETTE_NAME,
        "palette_colours": get_vertical_palette_11(),
        "color_stop_count": 11,
        "data_min_mm_yr": round_or_none(min(up_values)),
        "data_p02_mm_yr": round_or_none(percentile(up_values, 2)),
        "data_p50_mm_yr": round_or_none(percentile(up_values, 50)),
        "data_p98_mm_yr": round_or_none(percentile(up_values, 98)),
        "data_max_mm_yr": round_or_none(max(up_values)),
        "color_stops": build_adaptive_vertical_stops_11(L_sub, L_up, tau, L_sub_extreme, L_up_extreme),
        "legend_layout": adaptive_legend_layout(L_sub, L_up, tau, L_sub_extreme, L_up_extreme),
        "legend": build_adaptive_legend(L_sub, L_up, tau, L_sub_extreme, L_up_extreme),
        "clipping": {
            "low": f"up <= -{fmt_limit(L_sub_extreme)} mm/yr uses darkest red",
            "high": f"up >= +{fmt_limit(L_up_extreme)} mm/yr uses darkest blue",
            "actual_values_remain_available": True,
        },
        "note": (
            "Adaptive asymmetric 11-step scale. Near-zero band uses P75(2*sqrt(var_up)); "
            "subsidence/uplift far stops use sign-specific P98 and clipped extreme stops use P99.5. "
            "The legend zero is allowed to move off-centre and -tau/0/+tau are explicit colour stops. "
            "Limits are visualisation limits, not hazard thresholds."
        ),
    }

def derive_vertical_exaggeration(vertical_summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Derive vertical exaggeration from MODEL displacement range.

    MEASUREMENT can be noisy/sinusoidal and is meant for trendline display.
    It must not control RUM height exaggeration.
    """
    model_summary = vertical_summary.get("model") or {}
    vmin = safe_float(model_summary.get("min_mm"))
    vmax = safe_float(model_summary.get("max_mm"))

    if vmin is None or vmax is None:
        return {
            "recommended": 5.0,
            "recommended_m_per_mm": 5.0,
            "slider_min": 0.0,
            "slider_max": 10.0,
            "slider_step": 0.25,
            "unit": "m_per_mm",
            "ui_label": "Vertical exaggeration",
            "interpretation": "1x means 1 mm MODEL displacement = 1 m display height",
            "data_role": "model_mm",
            "reason": "fallback_no_model_summary",
        }

    max_abs = max(abs(vmin), abs(vmax))
    exag = choose_vertical_exaggeration(max_abs)
    slider_max = clamp(max(10.0, exag * 2.0), 10.0, VERTICAL_EXAG_MAX)
    slider_step = 0.25 if slider_max <= 20.0 else 1.0

    return {
        "recommended": exag,
        "recommended_m_per_mm": exag,
        "slider_min": 0.0,
        "slider_max": slider_max,
        "slider_step": slider_step,
        "unit": "m_per_mm",
        "ui_label": "Vertical exaggeration",
        "interpretation": "1x means 1 mm MODEL displacement = 1 m display height",
        "data_role": "model_mm",
        "max_abs_model_displacement_mm": round_or_none(max_abs),
        "model_min_mm": round_or_none(vmin),
        "model_max_mm": round_or_none(vmax),
        "reason": "heuristic_from_absolute_model_displacement_range",
        "note": "Uses MODEL range only. MEASUREMENT noise/sinusoid is for trendline and does not affect RUM height exaggeration.",
    }


def derive_camera(bbox: Optional[Dict[str, float]]) -> Dict[str, Any]:
    if not bbox:
        warn("Cannot derive camera because dataset bbox is missing")
        return {
            "available": False,
            "center_lon": None,
            "center_lat": None,
            "range_m": CAMERA_MIN_RANGE_M,
            "heading_deg": CAMERA_DEFAULT_HEADING_DEG,
            "pitch_deg": CAMERA_DEFAULT_PITCH_DEG,
            "roll_deg": CAMERA_DEFAULT_ROLL_DEG,
        }

    center_lon, center_lat = bbox_center(bbox)
    width_m, height_m = bbox_width_height_m(bbox)
    diag_m = math.hypot(width_m, height_m)
    range_m = clamp(diag_m * CAMERA_RANGE_MULTIPLIER, CAMERA_MIN_RANGE_M, CAMERA_MAX_RANGE_M)

    return {
        "available": True,
        "bbox_wgs84": bbox,
        "center_lon": round(center_lon, ROUND_COORD_DIGITS),
        "center_lat": round(center_lat, ROUND_COORD_DIGITS),
        "width_m": round(width_m, ROUND_NUMERIC_DIGITS),
        "height_m": round(height_m, ROUND_NUMERIC_DIGITS),
        "diagonal_m": round(diag_m, ROUND_NUMERIC_DIGITS),
        "range_m": round(range_m, ROUND_NUMERIC_DIGITS),
        "heading_deg": CAMERA_DEFAULT_HEADING_DEG,
        "pitch_deg": CAMERA_DEFAULT_PITCH_DEG,
        "roll_deg": CAMERA_DEFAULT_ROLL_DEG,
    }




def _nested_get(mapping: Dict[str, Any], path: Iterable[str], fallback: Any = None) -> Any:
    cur: Any = mapping
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return fallback
        cur = cur[key]
    return cur


def _format_sigma_label(multiplier: Any, fallback: str) -> str:
    value = safe_float(multiplier, None)
    if value is None:
        return fallback
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value))}σ"
    return f"{value:g}σ"


def extract_static_horizontal_glyph_tuning(
    cfg: Dict[str, Any],
    arrows_tileset: Optional[Dict[str, Any]],
    ellipses_tileset: Optional[Dict[str, Any]],
    products: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Build viewer-facing static horizontal glyph contract.

    Step 17 owns the scientific geometry: uncertainty threshold, arrow scaling,
    arrow anchoring, and ellipse placement.  Step 18 only exports that contract
    to viewer_tuning and adds presentation colours/lift for CustomShader use.
    """
    hdev = cfg.get("horizontal_dev_layers") or {}

    arrow_scaling = _nested_get(arrows_tileset or {}, ["root", "extras", "scaling"])
    if arrow_scaling is None:
        arrow_scaling = _nested_get(arrows_tileset or {}, ["extras", "scaling"])
    ellipse_scaling = _nested_get(ellipses_tileset or {}, ["root", "extras", "scaling"])
    if ellipse_scaling is None:
        ellipse_scaling = _nested_get(ellipses_tileset or {}, ["extras", "scaling"])
    scaling = arrow_scaling if isinstance(arrow_scaling, dict) else {}
    if not scaling and isinstance(ellipse_scaling, dict):
        scaling = ellipse_scaling

    visibility_sigma = safe_float(
        scaling.get("visibility_sigma_multiplier") if scaling else None,
        safe_float(hdev.get("visibility_sigma_multiplier"), 1.0),
    )
    ellipse_sigma = safe_float(
        scaling.get("ellipse_sigma_multiplier") if scaling else None,
        safe_float(hdev.get("ellipse_sigma_multiplier"), 2.0),
    )

    visibility_label = str(
        (scaling.get("visibility_label") if scaling else None)
        or hdev.get("visibility_label")
        or _format_sigma_label(visibility_sigma, "1σ")
    )
    ellipse_label = str(
        (scaling.get("ellipse_label") if scaling else None)
        or hdev.get("ellipse_label")
        or _format_sigma_label(ellipse_sigma, "2σ")
    )

    arrow_ref_pct = safe_float(
        scaling.get("arrow_reference_percentile") if scaling else None,
        safe_float(hdev.get("arrow_reference_percentile"), 99.5),
    )
    arrow_max_length_m = safe_float(scaling.get("arrow_max_length_m") if scaling else None, None)
    rum_size_m = safe_float(scaling.get("rum_size_m") if scaling else None, safe_float(cfg.get("rum_size_m"), None))
    arrow_max_fraction = None
    if arrow_max_length_m is not None and rum_size_m is not None and rum_size_m > 0:
        arrow_max_fraction = arrow_max_length_m / rum_size_m
    arrow_max_fraction = safe_float(arrow_max_fraction, safe_float(hdev.get("arrow_max_length_rum_fraction"), 0.80))

    return {
        "arrows_tileset": products["horizontal_arrows_tileset"]["path"],
        "ellipses_tileset": products["horizontal_ellipses_tileset"]["path"],
        "particle_field": products["horizontal_particle_field"]["path"],
        "vertical_follow_contract": "B3DM TEXCOORD_0.y = row_v; shader samples same MODEL/sigma texture as caps",
        "ellipse_scaling_note": "Ellipse axes are from Step16 covariance eigenvalues in mm/yr; no hidden /100 scaling.",
        "static_glyph_contract": {
            "source": "Step17 tileset extras when available; project_config/pipeline defaults as fallback",
            "minimum_speed_mm_yr": safe_float(scaling.get("minimum_speed_mm_yr") if scaling else None, safe_float(hdev.get("minimum_speed_mm_yr"), 0.2)),
            "visibility_sigma_multiplier": visibility_sigma,
            "visibility_label": visibility_label,
            "ellipse_sigma_multiplier": ellipse_sigma,
            "ellipse_label": ellipse_label,
            "arrow_reference_percentile": arrow_ref_pct,
            "arrow_speed_ref_mm_yr": safe_float(scaling.get("arrow_speed_ref_mm_yr") if scaling else None, None),
            "arrow_max_length_m": arrow_max_length_m,
            "arrow_max_length_rum_fraction": round(float(arrow_max_fraction), 6) if arrow_max_fraction is not None else None,
            "arrow_scale_m_per_mm_yr": safe_float(scaling.get("arrow_scale_m_per_mm_yr") if scaling else None, safe_float(hdev.get("arrow_scale_m_per_mm_yr"), None)),
            "arrow_anchor_fraction_at_rum_center": safe_float(scaling.get("arrow_anchor_fraction_at_rum_center") if scaling else None, safe_float(hdev.get("arrow_anchor_fraction_at_rum_center"), 0.75)),
            "ellipse_center_placement": (scaling.get("ellipse_center_placement") if scaling else None) or "arrowhead",
            "ellipse_reference_percentile": safe_float(scaling.get("ellipse_reference_percentile") if scaling else None, safe_float(hdev.get("ellipse_reference_percentile"), 99.5)),
            "ellipse_max_diameter_m": safe_float(scaling.get("ellipse_max_diameter_m") if scaling else None, None),
            "ellipse_max_diameter_rum_fraction": safe_float(hdev.get("ellipse_max_diameter_rum_fraction"), 0.75),
            "ellipse_scale_m_per_mm_yr": safe_float(scaling.get("ellipse_scale_m_per_mm_yr") if scaling else None, safe_float(hdev.get("ellipse_scale_m_per_mm_yr"), None)),
            "ellipse_scale_mode": (scaling.get("ellipse_scale_mode") if scaling else None) or str(hdev.get("ellipse_scale_mode", "same_as_arrow")),
            "rule_summary": f"show arrows/ellipses when speed ≥ {visibility_label} major uncertainty; draw ellipse axes as {ellipse_label}; place ellipse at arrowhead; RUM centre at configured arrow anchor fraction",
        },
        "glyph_style": {
            "source": "18_build_viewer_tuning.py presentation constants",
            "arrow_color_rgba": DEFAULT_HORIZONTAL_STATIC_ARROW_COLOR_RGBA,
            "ellipse_color_rgba": DEFAULT_HORIZONTAL_STATIC_ELLIPSE_COLOR_RGBA,
            "arrow_lift_rgb": DEFAULT_HORIZONTAL_STATIC_ARROW_LIFT_RGB,
            "ellipse_lift_rgb": DEFAULT_HORIZONTAL_STATIC_ELLIPSE_LIFT_RGB,
            "arrow_opacity": DEFAULT_HORIZONTAL_STATIC_ARROW_OPACITY,
            "ellipse_opacity": DEFAULT_HORIZONTAL_STATIC_ELLIPSE_OPACITY,
            "note": "Lift is shader-side additive colour, not true glow/halo.",
        },
    }

# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    t_start = time.time()

    cfg = load_resolved_config(__file__)
    project_root = Path(cfg["_resolved"]["project_root"])

    generated = cfg["generated_outputs"]
    paths = cfg["paths"]
    user_inputs = cfg["user_inputs"]

    viewer_tuning_path = resolve_path(project_root, generated["viewer_tuning"])

    product_paths = {
        "points_geojson": resolve_path(project_root, cfg["prepared_inputs"]["points_geojson"]),
        "vertical_epochs": resolve_path(project_root, cfg["prepared_inputs"]["vertical_epoch_json"]),
        "rum_footprints": resolve_path(project_root, generated["rum_footprints"]),
        "packed_series": resolve_path(project_root, generated["packed_series"]),
        "blank_cells": resolve_path(project_root, generated["blank_cells"]),
        "height_texture": resolve_path(project_root, generated["height_texture"]),
        "height_meta": resolve_path(project_root, generated["height_meta"]),
        "epoch_axis": resolve_path(project_root, generated["epoch_axis"]),
        "tile_index": resolve_path(project_root, paths["tiles_dir"]) / "tile_index.json",
        "real_caps_tileset": resolve_path(project_root, generated["real_caps_tileset"]),
        "blank_caps_tileset": resolve_path(project_root, generated["blank_caps_tileset"]),
        "real_walls_tileset": resolve_path(project_root, generated["real_walls_tileset"]),
        "blank_walls_tileset": resolve_path(project_root, generated["blank_walls_tileset"]),
        "horizontal_field": resolve_path(project_root, generated["horizontal_field"]),
        "horizontal_particle_field": resolve_path(project_root, generated["horizontal_particle_field"]),
        "horizontal_uncertainty_check": resolve_path(project_root, generated["horizontal_uncertainty_check"]),
        "horizontal_arrows_tileset": resolve_path(project_root, generated["horizontal_arrows_tileset"]),
        "horizontal_ellipses_tileset": resolve_path(project_root, generated["horizontal_ellipses_tileset"]),
        "validation_report": Path(cfg["_resolved"]["pipeline_output_dir"]) / "validation_report.json",
    }

    section("Configuration")
    print(f"  Project root       : {project_root}")
    print(f"  Viewer tuning out  : {viewer_tuning_path}")
    print(f"  Dataset title      : {user_inputs.get('dataset_title')}")
    print(f"  Project key        : {cfg['_resolved'].get('project_key')}")

    section("Loading final products")
    footprints = load_json_optional(product_paths["rum_footprints"], "rum_footprints")
    packed = load_json_optional(product_paths["packed_series"], "packed_series")
    blank_cells = load_json_optional(product_paths["blank_cells"], "blank_cells")
    height_meta = load_json_optional(product_paths["height_meta"], "height_meta")
    epoch_axis = load_json_optional(product_paths["epoch_axis"], "epoch_axis")
    tile_index = load_json_optional(product_paths["tile_index"], "tile_index")
    horizontal_field = load_json_optional(product_paths["horizontal_field"], "horizontal_field")
    horizontal_unc = load_json_optional(product_paths["horizontal_uncertainty_check"], "horizontal_uncertainty_check")
    horizontal_arrows_tileset = load_json_optional(product_paths["horizontal_arrows_tileset"], "horizontal_arrows_tileset")
    horizontal_ellipses_tileset = load_json_optional(product_paths["horizontal_ellipses_tileset"], "horizontal_ellipses_tileset")
    validation = load_json_optional(product_paths["validation_report"], "validation_report")

    required_for_complete = [
        "real_caps_tileset",
        "blank_caps_tileset",
        "real_walls_tileset",
        "blank_walls_tileset",
        "height_texture",
        "height_meta",
        "epoch_axis",
    ]

    missing_required = [
        key for key in required_for_complete
        if not product_paths[key].exists()
    ]

    if missing_required:
        msg = f"Missing final viewer products: {missing_required}"
        if STRICT_REQUIRE_FINAL_PRODUCTS:
            raise FileNotFoundError(msg)
        warn(msg)
    else:
        ok("All core viewer products exist")

    # Build product path references before deriving horizontal static tuning, because
    # that helper exports links to the generated arrow/ellipse/particle products.
    products = {
        key: product_ref(project_root, path)
        for key, path in product_paths.items()
    }

    section("Deriving viewer knobs")
    dataset_bbox = extract_dataset_bbox(footprints, tile_index, validation)
    camera = derive_camera(dataset_bbox)
    epoch_defaults = extract_epoch_defaults(epoch_axis, height_meta)
    vertical_summary = extract_vertical_summary(packed, height_meta, validation)
    velocity_summary = extract_velocity_summary(horizontal_field, packed)
    blank_summary = extract_blank_summary(blank_cells)
    color_scale = derive_color_scale(horizontal_field, velocity_summary)
    vertical_exag = derive_vertical_exaggeration(vertical_summary)
    horizontal_static_tuning = extract_static_horizontal_glyph_tuning(
        cfg=cfg,
        arrows_tileset=horizontal_arrows_tileset,
        ellipses_tileset=horizontal_ellipses_tileset,
        products=products,
    )

    ok("Derived camera, epoch, color-scale, vertical-exaggeration, and horizontal glyph defaults")

    if blank_summary.get("blank_count") == 0:
        ok("Blank layer default can stay available; source has zero blank cells")
    elif blank_summary.get("blank_count"):
        ok(f"Blank cells detected: {blank_summary.get('blank_count')}")

    section("Writing viewer tuning JSON")

    viewer_tuning = {
        "schema": "viewer_tuning_v2_measurement_model",
        "generated_by": "18_build_viewer_tuning.py",
        "project": {
            "project_key": cfg["_resolved"].get("project_key"),
            "dataset_title": user_inputs.get("dataset_title"),
            "source_file": user_inputs.get("source_file"),
            "source_crs": user_inputs.get("source_crs"),
            "rum_size_m": user_inputs.get("rum_size_m"),
            "expected_rum_count": user_inputs.get("expected_rum_count"),
            "epoch_generation": user_inputs.get("synthetic_epochs"),
        },
        "products": products,
        "camera": camera,
        "time": epoch_defaults,
        "data_summary": {
            "bbox_wgs84": dataset_bbox,
            "vertical": vertical_summary,
            "velocity": velocity_summary,
            "blank_cells": blank_summary,
            "horizontal_uncertainty": ((horizontal_unc or {}).get("metadata") or {}).get("summary"),
            "validation": (validation or {}).get("summary"),
        },
        "visual_defaults": {
            "vertical_exaggeration": vertical_exag,
            "color_scale": color_scale,
            "height_texture": {
                "row_flip": DEFAULT_HEIGHT_TEXTURE_ROW_FLIP,
                "source_role": "model_mm_plus_sigma_mm",
                "note": "PNG/WebGL row orientation correction. Viewer should sample 1.0 - row_v when true.",
            },
            "view_modes": {
                "global_opacity": {
                    "2d": DEFAULT_2D_GLOBAL_OPACITY,
                    "3d": DEFAULT_3D_GLOBAL_OPACITY,
                },
            },
            "layer_style": {
                "blank_caps": {
                    "color_rgb": DEFAULT_BLANK_CAP_COLOR_RGB,
                    "alpha": DEFAULT_BLANK_CAP_ALPHA,
                },
                "real_walls": {
                    "darken": DEFAULT_REAL_WALL_DARKEN,
                    "alpha": DEFAULT_REAL_WALL_ALPHA,
                },
                "blank_walls": {
                    "color_rgb": DEFAULT_BLANK_WALL_COLOR_RGB,
                    "darken": DEFAULT_BLANK_WALL_DARKEN,
                    "alpha": DEFAULT_BLANK_WALL_ALPHA,
                },
                "wall_shading": {
                    "shade_min": DEFAULT_WALL_SHADE_MIN,
                    "shade_max": DEFAULT_WALL_SHADE_MAX,
                    "light_dir_ec": DEFAULT_WALL_LIGHT_DIR_EC,
                },
            },
            "sigma": {
                "texture_sigma_max_mm": (vertical_summary.get("texture") or {}).get("sigma_max_mm"),
                "sigma_clip_fraction": (vertical_summary.get("sigma") or {}).get("clip_fraction"),
                "interpretation": "vertical uncertainty / hatch channel; p98 clipping is acceptable for visual contrast",
            },
            "layer_visibility": {
                "dev": {
                    "real_caps": DEFAULT_SHOW_REAL_CAPS,
                    "blank_caps": DEFAULT_SHOW_BLANK_CAPS,
                    "real_walls": DEFAULT_SHOW_REAL_WALLS,
                    "blank_walls": DEFAULT_SHOW_BLANK_WALLS,
                    "horizontal_arrows": DEFAULT_SHOW_HORIZONTAL_ARROWS_DEV,
                    "horizontal_ellipses": DEFAULT_SHOW_HORIZONTAL_ELLIPSES_DEV,
                },
                "output": {
                    "real_caps": DEFAULT_SHOW_REAL_CAPS,
                    "blank_caps": DEFAULT_SHOW_BLANK_CAPS,
                    "real_walls": DEFAULT_SHOW_REAL_WALLS,
                    "blank_walls": DEFAULT_SHOW_BLANK_WALLS,
                    "horizontal_arrows": DEFAULT_SHOW_HORIZONTAL_ARROWS_OUTPUT,
                    "horizontal_ellipses": DEFAULT_SHOW_HORIZONTAL_ELLIPSES_OUTPUT,
                },
            },
            "horizontal_particles": {
                "enabled_initial": DEFAULT_H_PARTICLES_ENABLED_INITIAL,
                "default_engine_mode": DEFAULT_H_PARTICLE_ENGINE_MODE,
                "particle_count": DEFAULT_H_PARTICLE_COUNT,
                "speed_multiplier": DEFAULT_H_PARTICLE_SPEED_MULTIPLIER,
                "trail_persistence": DEFAULT_H_PARTICLE_TRAIL_PERSISTENCE,
                "size_multiplier": DEFAULT_H_PARTICLE_SIZE_MULTIPLIER,
                "opacity": DEFAULT_H_PARTICLE_OPACITY,
                "base_mps": DEFAULT_H_PARTICLE_BASE_MPS,
                "surface_offset_m": DEFAULT_H_PARTICLE_SURFACE_OFFSET_M,
                "stall_speed_mm_yr": DEFAULT_H_PARTICLE_STALL_SPEED_MM_YR,
                "max_trail_screen_jump_px": DEFAULT_H_PARTICLE_MAX_TRAIL_SCREEN_JUMP_PX,
                "camera_stable_delay_ms": DEFAULT_H_PARTICLE_CAMERA_STABLE_DELAY_MS,
                "sampler_mode": DEFAULT_H_PARTICLE_SAMPLER_MODE,
                "primitive_points": {
                    "pixel_size": DEFAULT_H_PRIMITIVE_POINTS_PIXEL_SIZE,
                    "outline_width": DEFAULT_H_PRIMITIVE_POINTS_OUTLINE_WIDTH,
                    "color_rgb": DEFAULT_H_PRIMITIVE_POINTS_COLOR_RGB,
                    "outline_rgb": DEFAULT_H_PRIMITIVE_POINTS_OUTLINE_RGB,
                    "debug_enabled": DEFAULT_H_PRIMITIVE_POINTS_DEBUG_ENABLED,
                    "debug_log_interval_ms": DEFAULT_H_PRIMITIVE_POINTS_DEBUG_LOG_INTERVAL_MS,
                    "depth_test": "enabled; disableDepthTestDistance=0",
                    "render_contract": "Cesium PointPrimitiveCollection dots in the WebGL scene; depth-tested; no trails yet",
                },
                "canvas": {
                    "uncertainty_enabled_initial": DEFAULT_H_CANVAS_UNCERTAINTY_ENABLED_INITIAL,
                    "uncertainty_strength": DEFAULT_H_CANVAS_UNCERTAINTY_STRENGTH,
                    "uncertainty_speed_floor_mm_yr": DEFAULT_H_CANVAS_UNCERTAINTY_SPEED_FLOOR_MM_YR,
                    "uncertainty_theta_low_deg": DEFAULT_H_CANVAS_UNCERTAINTY_THETA_LOW_DEG,
                    "uncertainty_theta_high_deg": DEFAULT_H_CANVAS_UNCERTAINTY_THETA_HIGH_DEG,
                    "uncertainty_max_wobble_px": DEFAULT_H_CANVAS_UNCERTAINTY_MAX_WOBBLE_PX,
                    "uncertainty_freq_min_hz": DEFAULT_H_CANVAS_UNCERTAINTY_FREQ_MIN_HZ,
                    "uncertainty_freq_max_hz": DEFAULT_H_CANVAS_UNCERTAINTY_FREQ_MAX_HZ,
                    "render_contract": "screen-space canvas trails; fallback/comparison only; no true depth test",
                },
                "height_contract": "particle Z = display datum + MODEL surface displacement at particle XY/current epoch + surface_offset_m",
            },
            "horizontal_layers": horizontal_static_tuning,
            "vertical_series_contract": {
                "measurement_array": "packed_series.arrays.measurement_mm",
                "model_array": "packed_series.arrays.model_mm",
                "sigma_array": "packed_series.arrays.sigma_mm",
                "trendline_role": "measurement_mm",
                "height_texture_role": "model_mm",
                "uncertainty_role": "sigma_mm",
            },
        },
        "viewer_resource_paths": {
            "packed_series": products["packed_series"]["path"],
            "height_texture": products["height_texture"]["path"],
            "height_meta": products["height_meta"]["path"],
            "epoch_axis": products["epoch_axis"]["path"],
            "real_caps_tileset": products["real_caps_tileset"]["path"],
            "blank_caps_tileset": products["blank_caps_tileset"]["path"],
            "real_walls_tileset": products["real_walls_tileset"]["path"],
            "blank_walls_tileset": products["blank_walls_tileset"]["path"],
            "horizontal_arrows_tileset": products["horizontal_arrows_tileset"]["path"],
            "horizontal_ellipses_tileset": products["horizontal_ellipses_tileset"]["path"],
            "horizontal_particle_field": products["horizontal_particle_field"]["path"],
        },
        "warnings": WARNINGS,
    }

    write_json(viewer_tuning_path, viewer_tuning)

    elapsed = time.time() - t_start

    ok(f"Wrote viewer tuning: {viewer_tuning_path} ({viewer_tuning_path.stat().st_size / 1024:.1f} KB)")

    section("Summary")
    ok(f"Step 18 complete in {elapsed:.2f} s")
    print(f"  Camera center          : {camera.get('center_lon')}, {camera.get('center_lat')}")
    print(f"  Camera range           : {camera.get('range_m')} m")
    print(f"  Model exaggeration     : {vertical_exag.get('recommended')}x")
    print(
        "  Color scale            : "
        f"{color_scale.get('recommended_min_mm_yr')} to "
        f"{color_scale.get('recommended_max_mm_yr')} mm/yr "
        f"(center={color_scale.get('recommended_center_mm_yr')}, "
        f"tau={color_scale.get('near_zero_threshold_mm_yr')})"
    )
    print(f"  Default epoch          : {epoch_defaults.get('default_epoch')}")
    print(f"  Blank count            : {blank_summary.get('blank_count')}")
    print(f"  Warnings               : {len(WARNINGS)}")


if __name__ == "__main__":
    main()
