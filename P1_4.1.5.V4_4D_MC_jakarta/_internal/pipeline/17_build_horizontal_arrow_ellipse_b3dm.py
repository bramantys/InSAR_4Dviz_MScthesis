#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
17_build_horizontal_arrow_ellipse_b3dm.py

InSAR4D RUM Viewer pipeline step 17.

Purpose
-------
Build B3DM tilesets for horizontal velocity arrows and confidence ellipses.

Inputs
------
  generated_outputs.horizontal_field
    _internal/data_pipeline/horizontal_field.json

  generated_outputs.horizontal_uncertainty_check
    _internal/data_pipeline/horizontal_uncertainty_check.json

  generated_outputs.height_meta
    _internal/data_pipeline/tiles/height_meta.json

  _internal/data_pipeline/tiles/tile_index.json

Outputs
-------
  generated_outputs.horizontal_arrows_tileset
    _internal/data_pipeline/horizontal_dev/arrows/tileset.json

  generated_outputs.horizontal_ellipses_tileset
    _internal/data_pipeline/horizontal_dev/ellipses/tileset.json

Important design decisions
--------------------------
1. Arrows and confidence ellipses are B3DM, not Cesium entities/GeoJSON.
   This avoids the laggy entity-heavy approach.

2. Arrows and ellipses move vertically with the RUM caps.
   Every vertex stores:
     TEXCOORD_0.x = 1.0
     TEXCOORD_0.y = row_v

   where:
     row_v = (row_index + 0.5) / height_texture_height

   The viewer shader can sample the same height texture used by caps.

3. Confidence ellipse math is NOT copied from old pipeline 19.
   Step 16 already computed ellipse axes from covariance eigenvalues using:
     covariance unit = (mm/yr)^2
     ellipse axis unit = mm/yr
     no hidden /100 scaling

   Step 17 only scales those axis values from mm/yr into display metres:
     axis_m = axis_mm_yr * ELLIPSE_SCALE_M_PER_MM_YR
"""

from __future__ import annotations

import json
import math
import struct
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pipeline_config import load_resolved_config, resolve_path


# =============================================================================
# ADVANCED SETTINGS
# =============================================================================

TILE_INDEX_FILENAME = "tile_index.json"

POSITION_COMPONENT_TYPE = 5126
NORMAL_COMPONENT_TYPE = 5126
TEXCOORD_COMPONENT_TYPE = 5126
INDEX_COMPONENT_TYPE = 5125

BATCH_LENGTH = 0
DOUBLE_SIDED = True

ARROW_MATERIAL_BASE_COLOR = [1.0, 0.72, 0.12, 1.0]
ELLIPSE_MATERIAL_BASE_COLOR = [0.00, 0.95, 0.85, 1.0]

# If config does not define these, use safe defaults.
# These are fallback values only; normal operation derives horizontal glyph
# scaling from the dataset distribution and RUM size.
DEFAULT_MINIMUM_SPEED_MM_YR = 0.2
DEFAULT_ARROW_SCALE_M_PER_MM_YR = 22.5
DEFAULT_ELLIPSE_SCALE_M_PER_MM_YR = 22.5

# Automatic sizing defaults.
# Goal: make horizontal glyphs readable across datasets without hand-tuning.
DEFAULT_AUTO_SCALE_ENABLED = True
DEFAULT_RUM_SIZE_M = 450.0
DEFAULT_ARROW_REFERENCE_PERCENTILE = 99.5
DEFAULT_ELLIPSE_REFERENCE_PERCENTILE = 99.5
DEFAULT_ARROW_MAX_LENGTH_RUM_FRACTION = 0.90
DEFAULT_ELLIPSE_MAX_DIAMETER_RUM_FRACTION = 0.90
DEFAULT_ARROW_SIGNIFICANCE_SIGMA = 1.0
DEFAULT_ARROW_MIN_SPEED_REF_MM_YR = 0.05
DEFAULT_MINIMUM_SPEED_PERCENTILE = 5.0
DEFAULT_ELLIPSE_MIN_AXIS_REF_MM_YR = 0.05

# Geometry style, expressed as RUM-size fractions unless explicitly overridden.
DEFAULT_ARROW_SHAFT_WIDTH_FRACTION = 0.045
DEFAULT_ARROW_SHAFT_WIDTH_MIN_RUM_FRACTION = 0.010
DEFAULT_ARROW_SHAFT_WIDTH_MAX_RUM_FRACTION = 0.040
DEFAULT_ARROWHEAD_FRACTION = 0.22
DEFAULT_ARROWHEAD_MIN_RUM_FRACTION = 0.060
DEFAULT_ARROWHEAD_MAX_RUM_FRACTION = 0.250

DEFAULT_ELLIPSE_LINE_WIDTH_RUM_FRACTION = 0.010
DEFAULT_ELLIPSE_AXIS_MIN_RUM_FRACTION = 0.004

# Arrow / ellipse placement contract.
# 0.75 means: tail=0%, RUM centre=75%, arrowhead=100%.
# Therefore the arrowhead sits slightly downstream of the RUM centre, and
# the uncertainty ellipse is centred at the arrowhead.
DEFAULT_ARROW_ANCHOR_FRACTION_AT_RUM_CENTER = 0.75
DEFAULT_ELLIPSE_CENTER_PLACEMENT = "arrowhead"
DEFAULT_ELLIPSE_SCALE_MODE = "same_as_arrow"

# Ellipse clipping policy.
#   none               : default; preserve raw scaled axes, even if the ellipse protrudes outside the RUM.
#   uniform            : cap the semi-major axis and scale both axes by the same factor, preserving aspect ratio.
#   legacy_independent : old behavior; cap major/minor separately (kept only for backward debugging).
DEFAULT_ELLIPSE_CLIP_MODE = "none"

# Clean old files in output dirs before writing.
CLEAN_OLD_OUTPUTS = True


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


def write_binary(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(payload)


def pad_bytes(data: bytes, multiple: int, pad_byte: bytes = b" ") -> bytes:
    r = len(data) % multiple
    return data if r == 0 else data + pad_byte * (multiple - r)


def safe_float(value: Any, fallback: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return fallback
        out = float(value)
        return out if math.isfinite(out) else fallback
    except Exception:
        return fallback


def deg_to_rad(value: float) -> float:
    return float(value) * math.pi / 180.0


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def normalize_ellipse_clip_mode(value: Any) -> str:
    """Return a safe ellipse clip mode.

    The default is intentionally "none" so confidence ellipses preserve
    their covariance-derived shape and magnitude. The previous independent
    clipping could turn elongated ellipses into circles when both axes hit
    the display cap.
    """
    mode = str(value if value is not None else DEFAULT_ELLIPSE_CLIP_MODE).strip().lower()
    aliases = {
        "off": "none",
        "false": "none",
        "no": "none",
        "none": "none",
        "unclipped": "none",
        "preserve": "uniform",
        "preserve_aspect": "uniform",
        "aspect": "uniform",
        "uniform": "uniform",
        "scale": "uniform",
        "legacy": "legacy_independent",
        "independent": "legacy_independent",
        "legacy_independent": "legacy_independent",
    }
    return aliases.get(mode, DEFAULT_ELLIPSE_CLIP_MODE)


def normalize_ellipse_scale_mode(value: Any) -> str:
    """Return a safe ellipse display-scale mode.

    same_as_arrow keeps the same metres-per-(mm/yr) scale as the arrows.
    independent keeps the older behaviour where ellipse scale is derived from
    its own reference percentile and RUM fraction.
    """
    mode = str(value if value is not None else DEFAULT_ELLIPSE_SCALE_MODE).strip().lower()
    aliases = {
        "same": "same_as_arrow",
        "same_as_arrow": "same_as_arrow",
        "arrow": "same_as_arrow",
        "arrow_scale": "same_as_arrow",
        "same_scale": "same_as_arrow",
        "independent": "independent",
        "ellipse": "independent",
        "ellipse_auto": "independent",
        "separate": "independent",
    }
    return aliases.get(mode, DEFAULT_ELLIPSE_SCALE_MODE)


def scale_ellipse_axes_for_display(
    major_mm_yr: float,
    minor_mm_yr: float,
    ellipse_scale_m_per_mm_yr: float,
    ellipse_axis_min_m: float,
    ellipse_axis_max_m: float,
    ellipse_clip_mode: str,
) -> Tuple[float, float]:
    """Convert ellipse axes from mm/yr to display metres.

    By default this does not clip the axes. Optional uniform clipping preserves
    ellipse aspect ratio. Legacy independent clipping is retained only so old
    screenshots can be reproduced if needed.
    """
    raw_major_m = float(major_mm_yr) * float(ellipse_scale_m_per_mm_yr)
    raw_minor_m = float(minor_mm_yr) * float(ellipse_scale_m_per_mm_yr)

    if raw_major_m <= 0.0 or raw_minor_m <= 0.0:
        return raw_major_m, raw_minor_m

    mode = normalize_ellipse_clip_mode(ellipse_clip_mode)

    if mode == "legacy_independent":
        return (
            clamp(raw_major_m, ellipse_axis_min_m, ellipse_axis_max_m),
            clamp(raw_minor_m, ellipse_axis_min_m, ellipse_axis_max_m),
        )

    if mode == "uniform":
        factor = 1.0
        if ellipse_axis_max_m > 0.0 and raw_major_m > ellipse_axis_max_m:
            factor = min(factor, ellipse_axis_max_m / raw_major_m)
        # Keep tiny ellipses visible without changing their aspect ratio.
        if ellipse_axis_min_m > 0.0 and raw_major_m * factor < ellipse_axis_min_m:
            factor = max(factor, ellipse_axis_min_m / raw_major_m)
        return raw_major_m * factor, raw_minor_m * factor

    # Default: no clipping and no independent min/max forcing. This keeps
    # the covariance shape honest, even when the ellipse protrudes outside a RUM.
    return raw_major_m, raw_minor_m


def percentile(values: List[float], p: float) -> Optional[float]:
    vals = sorted(v for v in values if math.isfinite(float(v)))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    k = (len(vals) - 1) * float(p) / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return vals[int(k)]
    return vals[f] * (c - k) + vals[c] * (k - f)


def bool_from_config(value: Any, fallback: bool = False) -> bool:
    if value is None:
        return fallback
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return fallback


def nested_get(mapping: Dict[str, Any], path: Iterable[str], fallback: Any = None) -> Any:
    cur: Any = mapping
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return fallback
        cur = cur[key]
    return cur


def _median(values: List[float]) -> Optional[float]:
    xs = sorted(float(v) for v in values if safe_float(v, None) is not None and float(v) > 0.0)
    if not xs:
        return None
    n = len(xs)
    mid = n // 2
    if n % 2 == 1:
        return xs[mid]
    return 0.5 * (xs[mid - 1] + xs[mid])


def _records_to_local_xy_m(records: List[Dict[str, Any]]) -> Optional[List[Tuple[float, float]]]:
    """
    Return record centre coordinates in local metres.

    Priority:
      1. projected/local XY fields, already in metres
      2. lon/lat centre fields converted to approximate local EN metres
    """
    if not records:
        return None

    projected_xy_candidates = [
        ("x", "y"),
        ("x_m", "y_m"),
        ("x_rum", "y_rum"),
        ("center_x", "center_y"),
        ("cx", "cy"),
        ("easting", "northing"),
        ("east_m", "north_m"),
        ("x_local_m", "y_local_m"),
        ("source_x", "source_y"),
    ]
    lonlat_candidates = [
        ("lon_center", "lat_center"),
        ("longitude", "latitude"),
        ("lon", "lat"),
    ]

    for x_key, y_key in projected_xy_candidates:
        if x_key not in records[0] or y_key not in records[0]:
            continue
        pts: List[Tuple[float, float]] = []
        for rec in records:
            x = safe_float(rec.get(x_key), None)
            y = safe_float(rec.get(y_key), None)
            if x is not None and y is not None:
                pts.append((float(x), float(y)))
        if len(pts) >= 2:
            return pts

    for lon_key, lat_key in lonlat_candidates:
        if lon_key not in records[0] or lat_key not in records[0]:
            continue
        lonlat: List[Tuple[float, float]] = []
        for rec in records:
            lon = safe_float(rec.get(lon_key), None)
            lat = safe_float(rec.get(lat_key), None)
            if lon is not None and lat is not None:
                lonlat.append((float(lon), float(lat)))
        if len(lonlat) < 2:
            continue

        lon0 = sum(p[0] for p in lonlat) / len(lonlat)
        lat0 = sum(p[1] for p in lonlat) / len(lonlat)
        lat0_rad = math.radians(lat0)
        earth_radius = 6378137.0
        pts = []
        for lon, lat in lonlat:
            x = earth_radius * math.cos(lat0_rad) * math.radians(lon - lon0)
            y = earth_radius * math.radians(lat - lat0)
            pts.append((x, y))
        return pts

    return None


def _nearest_neighbor_spacing_m(points: List[Tuple[float, float]]) -> Optional[float]:
    if len(points) < 2:
        return None

    try:
        import numpy as np
        from scipy.spatial import cKDTree

        arr = np.asarray(points, dtype=float)
        tree = cKDTree(arr)
        dists, _ = tree.query(arr, k=2)
        nn = dists[:, 1] # type: ignore
        nn = nn[np.isfinite(nn) & (nn > 0.0)]
        if nn.size:
            return float(np.median(nn))
    except Exception:
        pass

    # Fallback when scipy is unavailable: sample up to ~2500 points.
    n = len(points)
    step = max(1, n // 2500)
    sample = points[::step]
    dists: List[float] = []

    for x, y in sample:
        best2 = None
        for x2, y2 in points:
            dx = x2 - x
            dy = y2 - y
            d2 = dx * dx + dy * dy
            if d2 <= 0.0:
                continue
            if best2 is None or d2 < best2:
                best2 = d2
        if best2 is not None and best2 > 0.0:
            dists.append(math.sqrt(best2))

    return _median(dists)


def _infer_rum_size_from_horizontal_records(records: List[Dict[str, Any]]) -> Optional[float]:
    """
    Infer RUM spacing from actual horizontal records.

    Preferred method:
      use neighbouring grid_i/grid_j records and centre coordinates.

    Fallback method:
      use median nearest-neighbour distance between record centres.
    """
    points = _records_to_local_xy_m(records)
    if points is None or len(points) < 2:
        return None

    if "grid_i" in records[0] and "grid_j" in records[0]:
        by_grid: Dict[Tuple[int, int], Tuple[float, float]] = {}
        for rec, pt in zip(records, points):
            i = safe_float(rec.get("grid_i"), None)
            j = safe_float(rec.get("grid_j"), None)
            if i is None or j is None:
                continue
            by_grid[(int(round(i)), int(round(j)))] = pt

        dists: List[float] = []
        for (i, j), (x, y) in by_grid.items():
            for nb in ((i + 1, j), (i, j + 1)):
                if nb not in by_grid:
                    continue
                x2, y2 = by_grid[nb]
                d = math.hypot(x2 - x, y2 - y)
                if math.isfinite(d) and d > 0.0:
                    dists.append(d)

        out = _median(dists)
        if out is not None and out > 0.0:
            return float(out)

    out = _nearest_neighbor_spacing_m(points)
    if out is not None and out > 0.0:
        return float(out)
    return None


def derive_rum_size_m(
    cfg: Dict[str, Any],
    hdev: Dict[str, Any],
    tile_index: Dict[str, Any],
    horizontal_records: Optional[List[Dict[str, Any]]] = None,
) -> float:
    candidates = [
        hdev.get("rum_size_m"),
        cfg.get("rum_size_m"),
        cfg.get("cell_size_m"),
        cfg.get("grid_spacing_m"),
        nested_get(cfg, ["project", "rum_size_m"]),
        nested_get(cfg, ["project", "cell_size_m"]),
        nested_get(cfg, ["project", "grid_spacing_m"]),
        nested_get(cfg, ["dataset", "rum_size_m"]),
        nested_get(cfg, ["dataset", "cell_size_m"]),
        nested_get(cfg, ["dataset", "grid_spacing_m"]),
        nested_get(cfg, ["grid", "rum_size_m"]),
        nested_get(cfg, ["grid", "cell_size_m"]),
        nested_get(cfg, ["grid", "spacing_m"]),
        nested_get(tile_index, ["metadata", "rum_size_m"]),
        nested_get(tile_index, ["metadata", "cell_size_m"]),
        nested_get(tile_index, ["metadata", "grid_spacing_m"]),
        nested_get(tile_index, ["metadata", "grid", "spacing_m"]),
    ]

    for value in candidates:
        out = safe_float(value, None)
        if out is not None and out > 0.0:
            print("  RUM size source            : config/tile_index")
            return float(out)

    inferred = _infer_rum_size_from_horizontal_records(horizontal_records or [])
    if inferred is not None and inferred > 0.0:
        print("  RUM size source            : horizontal_records_inferred")
        return float(inferred)

    warn(
        f"Could not find rum_size_m in config/tile index/records; "
        f"using fallback {DEFAULT_RUM_SIZE_M} m"
    )
    print("  RUM size source            : fallback")
    return DEFAULT_RUM_SIZE_M


def derive_horizontal_visual_scaling(
    cfg: Dict[str, Any],
    hdev: Dict[str, Any],
    h_lookup: Dict[str, Dict[str, Any]],
    u_lookup: Dict[str, Dict[str, Any]],
    tile_index: Dict[str, Any],
    horizontal_records: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    rum_size_m = derive_rum_size_m(cfg, hdev, tile_index, horizontal_records)
    auto_scale = bool_from_config(hdev.get("auto_scale", hdev.get("auto_horizontal_scaling")), DEFAULT_AUTO_SCALE_ENABLED)

    speeds = [safe_float(rec.get("speed_mm_yr"), None) for rec in h_lookup.values()]
    speeds = [v for v in speeds if v is not None and v > 0.0]

    ellipse_major = [safe_float(rec.get("ellipse_major_mm_yr"), None) for rec in u_lookup.values()]
    ellipse_major = [v for v in ellipse_major if v is not None and v > 0.0]

    arrow_ref_percentile = float(hdev.get("arrow_reference_percentile", DEFAULT_ARROW_REFERENCE_PERCENTILE))
    ellipse_ref_percentile = float(hdev.get("ellipse_reference_percentile", DEFAULT_ELLIPSE_REFERENCE_PERCENTILE))

    arrow_speed_ref = percentile(speeds, arrow_ref_percentile) or DEFAULT_ARROW_MIN_SPEED_REF_MM_YR
    arrow_speed_ref = max(float(arrow_speed_ref), DEFAULT_ARROW_MIN_SPEED_REF_MM_YR)

    ellipse_major_ref = percentile(ellipse_major, ellipse_ref_percentile) or DEFAULT_ELLIPSE_MIN_AXIS_REF_MM_YR
    ellipse_major_ref = max(float(ellipse_major_ref), DEFAULT_ELLIPSE_MIN_AXIS_REF_MM_YR)

    arrow_max_length_m = float(hdev.get(
        "arrow_max_length_m",
        rum_size_m * float(hdev.get("arrow_max_length_rum_fraction", DEFAULT_ARROW_MAX_LENGTH_RUM_FRACTION)),
    ))

    ellipse_max_diameter_m = float(hdev.get(
        "ellipse_max_diameter_m",
        rum_size_m * float(hdev.get("ellipse_max_diameter_rum_fraction", DEFAULT_ELLIPSE_MAX_DIAMETER_RUM_FRACTION)),
    ))
    ellipse_axis_max_m = ellipse_max_diameter_m / 2.0

    ellipse_scale_mode = normalize_ellipse_scale_mode(hdev.get("ellipse_scale_mode", DEFAULT_ELLIPSE_SCALE_MODE))

    if auto_scale:
        arrow_scale = arrow_max_length_m / arrow_speed_ref
        if ellipse_scale_mode == "same_as_arrow":
            ellipse_scale = arrow_scale
            scale_mode = "auto_percentile_by_rum_size_same_as_arrow"
        else:
            ellipse_scale = ellipse_axis_max_m / ellipse_major_ref
            scale_mode = "auto_percentile_by_rum_size_independent_ellipse"
    else:
        arrow_scale = float(hdev.get("arrow_scale_m_per_mm_yr", DEFAULT_ARROW_SCALE_M_PER_MM_YR))
        if ellipse_scale_mode == "same_as_arrow":
            ellipse_scale = arrow_scale
        else:
            ellipse_scale = float(hdev.get("ellipse_scale_m_per_mm_yr", arrow_scale if arrow_scale else DEFAULT_ELLIPSE_SCALE_M_PER_MM_YR))
        scale_mode = "manual_config"

    arrowhead_frac = float(hdev.get("arrowhead_frac", DEFAULT_ARROWHEAD_FRACTION))
    arrowhead_min_m = float(hdev.get(
        "arrowhead_min_m",
        rum_size_m * float(hdev.get("arrowhead_min_rum_fraction", DEFAULT_ARROWHEAD_MIN_RUM_FRACTION)),
    ))
    arrowhead_max_m = float(hdev.get(
        "arrowhead_max_m",
        rum_size_m * float(hdev.get("arrowhead_max_rum_fraction", DEFAULT_ARROWHEAD_MAX_RUM_FRACTION)),
    ))

    shaft_width_fraction = float(hdev.get("arrow_shaft_width_fraction", DEFAULT_ARROW_SHAFT_WIDTH_FRACTION))
    shaft_width_min_m = float(hdev.get(
        "arrow_shaft_width_min_m",
        rum_size_m * float(hdev.get("arrow_shaft_width_min_rum_fraction", DEFAULT_ARROW_SHAFT_WIDTH_MIN_RUM_FRACTION)),
    ))
    shaft_width_max_m = float(hdev.get(
        "arrow_shaft_width_max_m",
        rum_size_m * float(hdev.get("arrow_shaft_width_max_rum_fraction", DEFAULT_ARROW_SHAFT_WIDTH_MAX_RUM_FRACTION)),
    ))

    ellipse_line_width_m = float(hdev.get(
        "ellipse_line_width_m",
        rum_size_m * float(hdev.get("ellipse_line_width_rum_fraction", DEFAULT_ELLIPSE_LINE_WIDTH_RUM_FRACTION)),
    ))
    ellipse_axis_min_m = float(hdev.get(
        "ellipse_axis_min_m",
        max(1.0, rum_size_m * float(hdev.get("ellipse_axis_min_rum_fraction", DEFAULT_ELLIPSE_AXIS_MIN_RUM_FRACTION))),
    ))

    minimum_speed_percentile = float(hdev.get("minimum_speed_percentile", DEFAULT_MINIMUM_SPEED_PERCENTILE))
    minimum_speed_auto = percentile(speeds, minimum_speed_percentile) or DEFAULT_MINIMUM_SPEED_MM_YR
    minimum_speed_auto = max(DEFAULT_ARROW_MIN_SPEED_REF_MM_YR, float(minimum_speed_auto))
    minimum_speed_mm_yr = float(hdev.get("minimum_speed_mm_yr", minimum_speed_auto))
    arrow_significance_sigma = float(hdev.get("arrow_significance_sigma", DEFAULT_ARROW_SIGNIFICANCE_SIGMA))
    arrow_significance_filter = bool_from_config(hdev.get("arrow_significance_filter"), True)
    ellipse_match_arrow_filter = bool_from_config(hdev.get("ellipse_match_arrow_filter"), True)
    ellipse_clip_mode = normalize_ellipse_clip_mode(hdev.get("ellipse_clip_mode", DEFAULT_ELLIPSE_CLIP_MODE))
    arrow_anchor_fraction_at_rum_center = clamp(
        safe_float(hdev.get("arrow_anchor_fraction_at_rum_center"), DEFAULT_ARROW_ANCHOR_FRACTION_AT_RUM_CENTER)
        or DEFAULT_ARROW_ANCHOR_FRACTION_AT_RUM_CENTER,
        0.0,
        1.0,
    )
    ellipse_center_placement = str(hdev.get("ellipse_center_placement", DEFAULT_ELLIPSE_CENTER_PLACEMENT)).strip().lower()
    if ellipse_center_placement not in {"arrowhead", "rum_center"}:
        ellipse_center_placement = DEFAULT_ELLIPSE_CENTER_PLACEMENT

    return {
        "mode": scale_mode,
        "auto_scale": auto_scale,
        "rum_size_m": rum_size_m,
        "minimum_speed_mm_yr": minimum_speed_mm_yr,
        "minimum_speed_percentile": minimum_speed_percentile,
        "minimum_speed_auto_mm_yr": minimum_speed_auto,
        "arrow_significance_filter": arrow_significance_filter,
        "arrow_significance_sigma": arrow_significance_sigma,
        "ellipse_match_arrow_filter": ellipse_match_arrow_filter,
        "ellipse_clip_mode": ellipse_clip_mode,
        "ellipse_scale_mode": ellipse_scale_mode,
        "ellipse_center_placement": ellipse_center_placement,
        "arrow_anchor_fraction_at_rum_center": arrow_anchor_fraction_at_rum_center,
        "arrow_reference_percentile": arrow_ref_percentile,
        "arrow_speed_ref_mm_yr": arrow_speed_ref,
        "arrow_max_length_m": arrow_max_length_m,
        "arrow_scale_m_per_mm_yr": arrow_scale,
        "arrow_shaft_width_fraction": shaft_width_fraction,
        "arrow_shaft_width_min_m": shaft_width_min_m,
        "arrow_shaft_width_max_m": shaft_width_max_m,
        "arrowhead_frac": arrowhead_frac,
        "arrowhead_min_m": arrowhead_min_m,
        "arrowhead_max_m": arrowhead_max_m,
        "ellipse_reference_percentile": ellipse_ref_percentile,
        "ellipse_major_ref_mm_yr": ellipse_major_ref,
        "ellipse_max_diameter_m": ellipse_max_diameter_m,
        "ellipse_axis_max_m": ellipse_axis_max_m,
        "ellipse_axis_min_m": ellipse_axis_min_m,
        "ellipse_scale_m_per_mm_yr": ellipse_scale,
        "ellipse_line_width_m": ellipse_line_width_m,
        "note": (
            "Arrow scale maps a robust high-percentile horizontal speed to the configured RUM-size fraction. "
            "Ellipse scale mode can follow the arrow scale or use an independent ellipse percentile scale. "
            "Default ellipse_clip_mode=none preserves covariance aspect ratio and allows large uncertainty to protrude. "
            "Default placement uses the configured arrow anchor fraction and centres the ellipse at the arrowhead."
        ),
    }


def arrow_filter_reason(
    rum_id: str,
    hrec: Dict[str, Any],
    u_lookup: Dict[str, Dict[str, Any]],
    scaling: Dict[str, Any],
) -> Optional[str]:
    speed = safe_float(hrec.get("speed_mm_yr"), 0.0) or 0.0
    if speed < float(scaling["minimum_speed_mm_yr"]):
        return "low_speed"

    if not scaling.get("arrow_significance_filter", True):
        return None

    urec = u_lookup.get(str(rum_id))
    if urec is None:
        return "missing_uncertainty"

    std_major = safe_float(urec.get("std_major_1sigma_mm_yr"), None)
    if std_major is None:
        return "missing_uncertainty"

    threshold = float(scaling["arrow_significance_sigma"]) * float(std_major)
    if speed < threshold:
        return "insignificant_vs_uncertainty"

    return None


# =============================================================================
# WGS84 / ECEF / ENU
# =============================================================================

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


def geodetic_to_ecef(lon_deg: float, lat_deg: float, height_m: float) -> Tuple[float, float, float]:
    lon = deg_to_rad(lon_deg)
    lat = deg_to_rad(lat_deg)

    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)

    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)

    x = (n + height_m) * cos_lat * cos_lon
    y = (n + height_m) * cos_lat * sin_lon
    z = (n * (1.0 - WGS84_E2) + height_m) * sin_lat
    return x, y, z


def enu_basis(lon_deg: float, lat_deg: float) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]:
    lon = deg_to_rad(lon_deg)
    lat = deg_to_rad(lat_deg)

    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)

    east = (-sin_lon, cos_lon, 0.0)
    north = (-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat)
    up = (cos_lat * cos_lon, cos_lat * sin_lon, sin_lat)
    return east, north, up


def dot(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


def ecef_to_local_enu(
    ecef: Tuple[float, float, float],
    center_ecef: Tuple[float, float, float],
    east: Tuple[float, float, float],
    north: Tuple[float, float, float],
    up: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    d = (ecef[0]-center_ecef[0], ecef[1]-center_ecef[1], ecef[2]-center_ecef[2])
    return dot(d, east), dot(d, north), dot(d, up)


def enu_to_ecef_transform_column_major(lon_deg: float, lat_deg: float, height_m: float = 0.0) -> List[float]:
    center = geodetic_to_ecef(lon_deg, lat_deg, height_m)
    east, north, up = enu_basis(lon_deg, lat_deg)
    return [
        east[0], east[1], east[2], 0.0,
        north[0], north[1], north[2], 0.0,
        up[0], up[1], up[2], 0.0,
        center[0], center[1], center[2], 1.0,
    ]


def local_from_lonlat(
    lon: float,
    lat: float,
    height: float,
    tile_center_lon: float,
    tile_center_lat: float,
    center_ecef: Tuple[float, float, float],
    east: Tuple[float, float, float],
    north: Tuple[float, float, float],
    up: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    ecef = geodetic_to_ecef(lon, lat, height)
    return ecef_to_local_enu(ecef, center_ecef, east, north, up)


# =============================================================================
# GLB / B3DM
# =============================================================================

def pack_floats(values: List[float]) -> bytes:
    return struct.pack("<" + "f" * len(values), *values)


def pack_uint32(values: List[int]) -> bytes:
    return struct.pack("<" + "I" * len(values), *values)


def append_aligned(buffer: bytearray, data: bytes, alignment: int = 4, pad_byte: bytes = b"\x00") -> Tuple[int, int]:
    offset = len(buffer)
    pad = (alignment - (offset % alignment)) % alignment
    if pad:
        buffer.extend(pad_byte * pad)
        offset += pad
    buffer.extend(data)
    return offset, len(data)


def component_min_max_vec3(values: List[float]) -> Tuple[List[float], List[float]]:
    xs = values[0::3]
    ys = values[1::3]
    zs = values[2::3]
    return [min(xs), min(ys), min(zs)], [max(xs), max(ys), max(zs)]


def build_glb(
    positions: List[float],
    normals: List[float],
    texcoords: List[float],
    indices: List[int],
    material_color: List[float],
    generator: str,
) -> bytes:
    if not positions or not indices:
        raise ValueError("Cannot build GLB with empty positions/indices")

    vertex_count = len(positions) // 3
    index_count = len(indices)

    if len(normals) != vertex_count * 3:
        raise ValueError("normal length mismatch")
    if len(texcoords) != vertex_count * 2:
        raise ValueError("texcoord length mismatch")

    bin_buffer = bytearray()
    pos_offset, pos_len = append_aligned(bin_buffer, pack_floats(positions), 4)
    norm_offset, norm_len = append_aligned(bin_buffer, pack_floats(normals), 4)
    uv_offset, uv_len = append_aligned(bin_buffer, pack_floats(texcoords), 4)
    idx_offset, idx_len = append_aligned(bin_buffer, pack_uint32(indices), 4)

    pos_min, pos_max = component_min_max_vec3(positions)

    gltf = {
        "asset": {"version": "2.0", "generator": generator},
        "buffers": [{"byteLength": len(bin_buffer)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": pos_offset, "byteLength": pos_len, "target": 34962},
            {"buffer": 0, "byteOffset": norm_offset, "byteLength": norm_len, "target": 34962},
            {"buffer": 0, "byteOffset": uv_offset, "byteLength": uv_len, "target": 34962},
            {"buffer": 0, "byteOffset": idx_offset, "byteLength": idx_len, "target": 34963},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": vertex_count, "type": "VEC3", "min": pos_min, "max": pos_max},
            {"bufferView": 1, "componentType": 5126, "count": vertex_count, "type": "VEC3"},
            {"bufferView": 2, "componentType": 5126, "count": vertex_count, "type": "VEC2"},
            {"bufferView": 3, "componentType": 5125, "count": index_count, "type": "SCALAR", "min": [min(indices)], "max": [max(indices)]},
        ],
        "materials": [
            {
                "pbrMetallicRoughness": {
                    "baseColorFactor": material_color,
                    "metallicFactor": 0.0,
                    "roughnessFactor": 1.0,
                },
                "alphaMode": "BLEND" if material_color[3] < 1.0 else "OPAQUE",
                "doubleSided": DOUBLE_SIDED,
            }
        ],
        "meshes": [
            {"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2}, "indices": 3, "material": 0, "mode": 4}]}
        ],
        "nodes": [
            {
                "mesh": 0,
                # z-up → y-up correction matrix (column-major)
                # cancels Cesium's automatic y-up → z-up transform at runtime
                "matrix": [1,0,0,0, 0,0,-1,0, 0,1,0,0, 0,0,0,1],
            }
        ],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }

    json_chunk = pad_bytes(json.dumps(gltf, separators=(",", ":")).encode("utf-8"), 4, b" ")
    bin_chunk = pad_bytes(bytes(bin_buffer), 4, b"\x00")

    total_length = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
    glb = bytearray()
    glb.extend(struct.pack("<4sII", b"glTF", 2, total_length))
    glb.extend(struct.pack("<I4s", len(json_chunk), b"JSON"))
    glb.extend(json_chunk)
    glb.extend(struct.pack("<I4s", len(bin_chunk), b"BIN\x00"))
    glb.extend(bin_chunk)
    return bytes(glb)


def build_b3dm(glb: bytes) -> bytes:
    ft_json = pad_bytes(json.dumps({"BATCH_LENGTH": BATCH_LENGTH}, separators=(",", ":")).encode("utf-8"), 8, b" ")
    byte_length = 28 + len(ft_json) + len(glb)
    header = struct.pack("<4sIIIIII", b"b3dm", 1, byte_length, len(ft_json), 0, 0, 0)
    return header + ft_json + glb


# =============================================================================
# TILE / RECORD HELPERS
# =============================================================================

def tile_center_from_bbox(bbox: Dict[str, float]) -> Tuple[float, float]:
    return (float(bbox["west"]) + float(bbox["east"])) / 2.0, (float(bbox["south"]) + float(bbox["north"])) / 2.0


def bounding_region_from_bbox(bbox: Dict[str, float], min_h: float, max_h: float) -> List[float]:
    return [
        deg_to_rad(float(bbox["west"])),
        deg_to_rad(float(bbox["south"])),
        deg_to_rad(float(bbox["east"])),
        deg_to_rad(float(bbox["north"])),
        float(min_h),
        float(max_h),
    ]


def bbox_union_wgs84(bboxes: List[Dict[str, float]]) -> Dict[str, float]:
    return {
        "west": min(float(b["west"]) for b in bboxes),
        "south": min(float(b["south"]) for b in bboxes),
        "east": max(float(b["east"]) for b in bboxes),
        "north": max(float(b["north"]) for b in bboxes),
    }


def point_bbox(lon: float, lat: float, radius_deg: float = 0.001) -> Dict[str, float]:
    return {"west": lon-radius_deg, "south": lat-radius_deg, "east": lon+radius_deg, "north": lat+radius_deg}


def build_record_lookup(horizontal_field: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    lookup = {}
    for rec in horizontal_field.get("records", []):
        lookup[str(rec["rum_id"])] = rec
    if not lookup:
        raise ValueError("horizontal_field has no records")
    return lookup


def build_uncertainty_lookup(uncertainty: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    lookup = {}
    for rec in uncertainty.get("records", []):
        lookup[str(rec["rum_id"])] = rec
    if not lookup:
        raise ValueError("horizontal_uncertainty_check has no records")
    return lookup


# =============================================================================
# GEOMETRY: ARROWS
# =============================================================================

def add_vertex(
    local_xy: Tuple[float, float],
    base_local: Tuple[float, float, float],
    row_v: float,
    positions: List[float],
    normals: List[float],
    texcoords: List[float],
) -> int:
    idx = len(positions) // 3
    positions.extend([
        float(base_local[0] + local_xy[0]),
        float(base_local[1] + local_xy[1]),
        float(base_local[2]),
    ])
    normals.extend([0.0, 0.0, 1.0])
    # x=1 means "top/move-with-cap" role, y=row lookup.
    texcoords.extend([1.0, float(row_v)])
    return idx


def add_arrow(
    rec: Dict[str, Any],
    row_v: float,
    center_lon: float,
    center_lat: float,
    datum_height_m: float,
    clearance_m: float,
    arrow_scale_m_per_mm_yr: float,
    arrowhead_frac: float,
    arrowhead_min_m: float,
    arrowhead_max_m: float,
    arrow_max_length_m: float,
    arrow_anchor_fraction_at_rum_center: float,
    shaft_width_fraction: float,
    shaft_width_min_m: float,
    shaft_width_max_m: float,
    positions: List[float],
    normals: List[float],
    texcoords: List[float],
    indices: List[int],
) -> bool:
    speed = float(rec["speed_mm_yr"])
    if speed <= 0:
        return False

    ue = float(rec["unit_east"])
    un = float(rec["unit_north"])
    if math.hypot(ue, un) <= 0:
        return False

    lon = float(rec["lon_center"])
    lat = float(rec["lat_center"])

    center_ecef = geodetic_to_ecef(center_lon, center_lat, 0.0)
    east, north, up = enu_basis(center_lon, center_lat)
    base_local = local_from_lonlat(lon, lat, datum_height_m + clearance_m, center_lon, center_lat, center_ecef, east, north, up)

    length_raw = speed * arrow_scale_m_per_mm_yr
    length = min(length_raw, arrow_max_length_m) if arrow_max_length_m > 0 else length_raw
    if length <= 0:
        return False

    head_len = clamp(length * arrowhead_frac, arrowhead_min_m, arrowhead_max_m)
    if head_len > length * 0.65:
        head_len = length * 0.65

    shaft_len = max(0.0, length - head_len)
    shaft_half_w = clamp(length * shaft_width_fraction, shaft_width_min_m, shaft_width_max_m)
    head_half_w = max(shaft_half_w * 2.4, head_len * 0.45)

    # Direction and perpendicular in tile-local EN plane.
    ux, uy = ue, un
    px, py = -uy, ux

    # Place the RUM centre at a configurable fraction along the arrow.
    # anchor=0.75 means tail=0%, RUM centre=75%, arrowhead=100%, so the
    # arrow is backed up and the arrowhead sits slightly downstream.
    anchor = clamp(arrow_anchor_fraction_at_rum_center, 0.0, 1.0)
    arrow_origin_shift = -anchor * length

    def xy(a: float, b: float) -> Tuple[float, float]:
        aa = a + arrow_origin_shift
        return ux * aa + px * b, uy * aa + py * b

    tail_l = xy(0.0, shaft_half_w)
    tail_r = xy(0.0, -shaft_half_w)
    shaft_l = xy(shaft_len, shaft_half_w)
    shaft_r = xy(shaft_len, -shaft_half_w)
    head_l = xy(shaft_len, head_half_w)
    tip = xy(length, 0.0)
    head_r = xy(shaft_len, -head_half_w)

    # Shaft rectangle.
    i0 = add_vertex(tail_l, base_local, row_v, positions, normals, texcoords)
    i1 = add_vertex(tail_r, base_local, row_v, positions, normals, texcoords)
    i2 = add_vertex(shaft_r, base_local, row_v, positions, normals, texcoords)
    i3 = add_vertex(shaft_l, base_local, row_v, positions, normals, texcoords)
    indices.extend([i0, i1, i2, i0, i2, i3])

    # Head triangle.
    j0 = add_vertex(head_l, base_local, row_v, positions, normals, texcoords)
    j1 = add_vertex(tip, base_local, row_v, positions, normals, texcoords)
    j2 = add_vertex(head_r, base_local, row_v, positions, normals, texcoords)
    indices.extend([j0, j1, j2])

    return True


# =============================================================================
# GEOMETRY: ELLIPSES
# =============================================================================

def add_ellipse_ring(
    rec: Dict[str, Any],
    row_v: float,
    center_lon: float,
    center_lat: float,
    datum_height_m: float,
    clearance_m: float,
    ellipse_scale_m_per_mm_yr: float,
    ellipse_points: int,
    ellipse_line_width_m: float,
    ellipse_axis_min_m: float,
    ellipse_axis_max_m: float,
    ellipse_clip_mode: str,
    ellipse_center_placement: str,
    arrow_scale_m_per_mm_yr: float,
    arrow_max_length_m: float,
    arrow_anchor_fraction_at_rum_center: float,
    positions: List[float],
    normals: List[float],
    texcoords: List[float],
    indices: List[int],
) -> bool:
    major_mm_yr = safe_float(rec.get("ellipse_major_mm_yr"), None)
    minor_mm_yr = safe_float(rec.get("ellipse_minor_mm_yr"), None)
    angle_deg = safe_float(rec.get("ellipse_angle_deg_ccw_from_east"), None)

    if major_mm_yr is None or minor_mm_yr is None or angle_deg is None:
        return False

    major_m, minor_m = scale_ellipse_axes_for_display(
        major_mm_yr=major_mm_yr,
        minor_mm_yr=minor_mm_yr,
        ellipse_scale_m_per_mm_yr=ellipse_scale_m_per_mm_yr,
        ellipse_axis_min_m=ellipse_axis_min_m,
        ellipse_axis_max_m=ellipse_axis_max_m,
        ellipse_clip_mode=ellipse_clip_mode,
    )

    if major_m <= 0 or minor_m <= 0:
        return False

    lon = float(rec["lon_center"])
    lat = float(rec["lat_center"])

    center_ecef = geodetic_to_ecef(center_lon, center_lat, 0.0)
    east, north, up = enu_basis(center_lon, center_lat)
    base_local = local_from_lonlat(lon, lat, datum_height_m + clearance_m, center_lon, center_lat, center_ecef, east, north, up)

    # Optional placement: centre the ellipse at the velocity arrowhead rather
    # than at the RUM centre. This uses the exact same scaled/capped arrow
    # length and configured anchor rule as add_arrow().
    ellipse_offset_x = 0.0
    ellipse_offset_y = 0.0
    placement = str(ellipse_center_placement or DEFAULT_ELLIPSE_CENTER_PLACEMENT).strip().lower()
    if placement == "arrowhead":
        speed = safe_float(rec.get("speed_mm_yr"), 0.0) or 0.0
        ue = safe_float(rec.get("unit_east"), 0.0) or 0.0
        un = safe_float(rec.get("unit_north"), 0.0) or 0.0
        if speed > 0.0 and math.hypot(ue, un) > 0.0:
            length_raw = speed * float(arrow_scale_m_per_mm_yr)
            length = min(length_raw, float(arrow_max_length_m)) if float(arrow_max_length_m) > 0.0 else length_raw
            anchor = clamp(arrow_anchor_fraction_at_rum_center, 0.0, 1.0)
            head_offset = (1.0 - anchor) * max(0.0, length)
            ellipse_offset_x = ue * head_offset
            ellipse_offset_y = un * head_offset

    phi = math.radians(angle_deg)
    c = math.cos(phi)
    s = math.sin(phi)

    half_w = max(0.5, ellipse_line_width_m / 2.0)
    outer_major = major_m + half_w
    outer_minor = minor_m + half_w
    inner_major = max(0.1, major_m - half_w)
    inner_minor = max(0.1, minor_m - half_w)

    n = max(12, int(ellipse_points))
    outer_indices: List[int] = []
    inner_indices: List[int] = []

    for k in range(n):
        t = 2.0 * math.pi * k / n
        ct = math.cos(t)
        st = math.sin(t)

        xo = outer_major * ct
        yo = outer_minor * st
        xi = inner_major * ct
        yi = inner_minor * st

        # Rotate from ellipse local axes into EN plane.
        xro = xo * c - yo * s
        yro = xo * s + yo * c
        xri = xi * c - yi * s
        yri = xi * s + yi * c

        outer_indices.append(add_vertex((xro + ellipse_offset_x, yro + ellipse_offset_y), base_local, row_v, positions, normals, texcoords))
        inner_indices.append(add_vertex((xri + ellipse_offset_x, yri + ellipse_offset_y), base_local, row_v, positions, normals, texcoords))

    for k in range(n):
        ko = outer_indices[k]
        kn = outer_indices[(k + 1) % n]
        io = inner_indices[k]
        inn = inner_indices[(k + 1) % n]

        indices.extend([ko, kn, inn, ko, inn, io])

    return True


# =============================================================================
# TILESET WRITING
# =============================================================================

def write_empty_tileset(
    path: Path,
    dataset_bbox: Dict[str, float],
    min_h: float,
    max_h: float,
    extras: Dict[str, Any],
) -> None:
    tileset = {
        "asset": {
            "version": "1.0",
            "tilesetVersion": extras.get("tileset_version", "empty_v1"),
            "generator": "InSAR4D RUM Viewer pipeline step 17",
        },
        "geometricError": 0.0,
        "root": {
            "boundingVolume": {"region": bounding_region_from_bbox(dataset_bbox, min_h, max_h)},
            "geometricError": 0.0,
            "refine": "ADD",
            "children": [],
        },
        "extras": extras,
    }
    write_json(path, tileset)


def build_layer_tileset(
    layer_name: str,
    output_dir: Path,
    tileset_path: Path,
    tile_index: Dict[str, Any],
    h_lookup: Dict[str, Dict[str, Any]],
    u_lookup: Dict[str, Dict[str, Any]],
    height_meta: Dict[str, Any],
    datum_height_m: float,
    clearance_m: float,
    scaling: Dict[str, Any],
    ellipse_points: int,
    min_bound_h: float,
    max_bound_h: float,
    geometric_error_root: float,
    geometric_error_leaf: float,
) -> Dict[str, Any]:
    texture = height_meta.get("texture") or {}
    texture_height = int(texture.get("height", 0))
    if texture_height <= 0:
        raise ValueError("height_meta texture.height invalid")

    if CLEAN_OLD_OUTPUTS and output_dir.exists():
        for old in output_dir.glob("*.b3dm"):
            old.unlink()

    dataset_bbox = (tile_index.get("metadata") or {}).get("dataset_bbox_wgs84")
    if not dataset_bbox:
        raise ValueError("tile_index missing dataset bbox")

    children: List[Dict[str, Any]] = []
    built_features = 0
    built_tiles = 0
    total_vertices = 0
    total_triangles = 0
    skipped_low_speed = 0
    skipped_missing_uncertainty = 0
    skipped_insignificant = 0

    for tile in tile_index.get("tiles", []):
        rum_ids = tile.get("rum_ids", [])
        if not rum_ids:
            continue

        bbox = tile.get("bbox_wgs84")
        if not bbox:
            continue

        center_lon, center_lat = tile_center_from_bbox(bbox)

        positions: List[float] = []
        normals: List[float] = []
        texcoords: List[float] = []
        indices: List[int] = []

        tile_feature_count = 0

        for rum_id in rum_ids:
            hrec = h_lookup.get(rum_id)
            if hrec is None:
                continue

            speed = float(hrec.get("speed_mm_yr", 0.0))

            row_index = int(hrec["row_index"])
            row_v = (row_index + 0.5) / texture_height

            if layer_name == "arrows":
                reason = arrow_filter_reason(str(rum_id), hrec, u_lookup, scaling)
                if reason == "low_speed":
                    skipped_low_speed += 1
                    continue
                if reason == "missing_uncertainty":
                    skipped_missing_uncertainty += 1
                    continue
                if reason == "insignificant_vs_uncertainty":
                    skipped_insignificant += 1
                    continue

                built = add_arrow(
                    rec=hrec,
                    row_v=row_v,
                    center_lon=center_lon,
                    center_lat=center_lat,
                    datum_height_m=datum_height_m,
                    clearance_m=clearance_m,
                    arrow_scale_m_per_mm_yr=float(scaling["arrow_scale_m_per_mm_yr"]),
                    arrowhead_frac=float(scaling["arrowhead_frac"]),
                    arrowhead_min_m=float(scaling["arrowhead_min_m"]),
                    arrowhead_max_m=float(scaling["arrowhead_max_m"]),
                    arrow_max_length_m=float(scaling["arrow_max_length_m"]),
                    arrow_anchor_fraction_at_rum_center=float(scaling["arrow_anchor_fraction_at_rum_center"]),
                    shaft_width_fraction=float(scaling["arrow_shaft_width_fraction"]),
                    shaft_width_min_m=float(scaling["arrow_shaft_width_min_m"]),
                    shaft_width_max_m=float(scaling["arrow_shaft_width_max_m"]),
                    positions=positions,
                    normals=normals,
                    texcoords=texcoords,
                    indices=indices,
                )
            else:
                if scaling.get("ellipse_match_arrow_filter", True):
                    reason = arrow_filter_reason(str(rum_id), hrec, u_lookup, scaling)
                    if reason == "low_speed":
                        skipped_low_speed += 1
                        continue
                    if reason == "missing_uncertainty":
                        skipped_missing_uncertainty += 1
                        continue
                    if reason == "insignificant_vs_uncertainty":
                        skipped_insignificant += 1
                        continue

                urec = u_lookup.get(rum_id)
                if urec is None:
                    skipped_missing_uncertainty += 1
                    continue

                # Merge center/row fields from horizontal field into uncertainty record.
                merged = dict(urec)
                merged["lon_center"] = hrec["lon_center"]
                merged["lat_center"] = hrec["lat_center"]
                merged["speed_mm_yr"] = hrec.get("speed_mm_yr", 0.0)
                merged["unit_east"] = hrec.get("unit_east", 0.0)
                merged["unit_north"] = hrec.get("unit_north", 0.0)

                built = add_ellipse_ring(
                    rec=merged,
                    row_v=row_v,
                    center_lon=center_lon,
                    center_lat=center_lat,
                    datum_height_m=datum_height_m,
                    clearance_m=clearance_m,
                    ellipse_scale_m_per_mm_yr=float(scaling["ellipse_scale_m_per_mm_yr"]),
                    ellipse_points=ellipse_points,
                    ellipse_line_width_m=float(scaling["ellipse_line_width_m"]),
                    ellipse_axis_min_m=float(scaling["ellipse_axis_min_m"]),
                    ellipse_axis_max_m=float(scaling["ellipse_axis_max_m"]),
                    ellipse_clip_mode=str(scaling.get("ellipse_clip_mode", DEFAULT_ELLIPSE_CLIP_MODE)),
                    ellipse_center_placement=str(scaling.get("ellipse_center_placement", DEFAULT_ELLIPSE_CENTER_PLACEMENT)),
                    arrow_scale_m_per_mm_yr=float(scaling["arrow_scale_m_per_mm_yr"]),
                    arrow_max_length_m=float(scaling["arrow_max_length_m"]),
                    arrow_anchor_fraction_at_rum_center=float(scaling["arrow_anchor_fraction_at_rum_center"]),
                    positions=positions,
                    normals=normals,
                    texcoords=texcoords,
                    indices=indices,
                )

            if built:
                built_features += 1
                tile_feature_count += 1

        if tile_feature_count == 0:
            continue

        material = ARROW_MATERIAL_BASE_COLOR if layer_name == "arrows" else ELLIPSE_MATERIAL_BASE_COLOR
        generator = f"InSAR4D RUM Viewer pipeline step 17 {layer_name}"

        glb = build_glb(positions, normals, texcoords, indices, material, generator)
        b3dm = build_b3dm(glb)

        b3dm_name = f"{layer_name}_{tile['tile_id']}.b3dm"
        b3dm_path = output_dir / b3dm_name
        write_binary(b3dm_path, b3dm)

        children.append({
            "boundingVolume": {"region": bounding_region_from_bbox(bbox, min_bound_h, max_bound_h)},
            "geometricError": geometric_error_leaf,
            "refine": "ADD",
            "transform": enu_to_ecef_transform_column_major(center_lon, center_lat, 0.0),
            "content": {"uri": b3dm_name},
            "metadata": {
                "tile_id": tile["tile_id"],
                "feature_count": tile_feature_count,
            },
        })

        built_tiles += 1
        total_vertices += len(positions) // 3
        total_triangles += len(indices) // 3

    if built_tiles == 0:
        write_empty_tileset(
            tileset_path,
            dataset_bbox,
            min_bound_h,
            max_bound_h,
            {
                "schema": f"horizontal_{layer_name}_tileset_v1",
                "tileset_version": f"horizontal_{layer_name}_empty_v1",
                "status": "no_features_after_filtering",
                "minimum_speed_mm_yr": scaling["minimum_speed_mm_yr"],
                "feature_count": 0,
                "tile_count": 0,
                "texture_height": texture_height,
                "vertical_follow_contract": "TEXCOORD_0.x=1.0, TEXCOORD_0.y=row_v",
                "scaling": scaling,
            },
        )
    else:
        tileset = {
            "asset": {
                "version": "1.0",
                "tilesetVersion": f"horizontal_{layer_name}_v1",
                "generator": "InSAR4D RUM Viewer pipeline step 17",
            },
            "geometricError": geometric_error_root,
            "root": {
                "boundingVolume": {"region": bounding_region_from_bbox(dataset_bbox, min_bound_h, max_bound_h)},
                "geometricError": geometric_error_root,
                "refine": "ADD",
                "children": children,
            },
            "extras": {
                "schema": f"horizontal_{layer_name}_tileset_v1",
                "status": "features_built",
                "layer_name": layer_name,
                "minimum_speed_mm_yr": scaling["minimum_speed_mm_yr"],
                "feature_count": built_features,
                "tile_count": built_tiles,
                "total_vertices": total_vertices,
                "total_triangles": total_triangles,
                "texture_height": texture_height,
                "display_datum_height_m": datum_height_m,
                "clearance_above_cap_m": clearance_m,
                "vertical_follow_contract": "TEXCOORD_0.x=1.0 means move with cap; TEXCOORD_0.y=(row_index+0.5)/texture_height",
                "scaling": {
                    **scaling,
                    "ellipse_unit_source": "Step16 ellipse axes in mm/yr; no hidden /100 scaling",
                },
                "skipped_low_speed": skipped_low_speed,
                "skipped_insignificant_vs_uncertainty": skipped_insignificant,
                "skipped_missing_uncertainty": skipped_missing_uncertainty,
            },
        }
        write_json(tileset_path, tileset)

    return {
        "layer": layer_name,
        "tiles": built_tiles,
        "features": built_features,
        "vertices": total_vertices,
        "triangles": total_triangles,
        "skipped_low_speed": skipped_low_speed,
        "skipped_insignificant": skipped_insignificant,
        "skipped_missing_uncertainty": skipped_missing_uncertainty,
        "tileset": tileset_path,
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
    viewer = cfg["viewer"]
    hdev = cfg["horizontal_dev_layers"]
    tiling = cfg["tiling"]

    hfield_path = resolve_path(project_root, generated["horizontal_field"])
    uncertainty_path = resolve_path(project_root, generated["horizontal_uncertainty_check"])
    height_meta_path = resolve_path(project_root, generated["height_meta"])
    tile_index_path = resolve_path(project_root, paths["tiles_dir"]) / TILE_INDEX_FILENAME

    arrows_tileset_path = resolve_path(project_root, generated["horizontal_arrows_tileset"])
    ellipses_tileset_path = resolve_path(project_root, generated["horizontal_ellipses_tileset"])
    arrows_dir = arrows_tileset_path.parent
    ellipses_dir = ellipses_tileset_path.parent

    datum_height_m = float(viewer.get("display_datum_height_m", 1000.0))
    clearance_m = float(hdev.get("clearance_above_cap_m", 5.0))

    ellipse_points = int(hdev.get("ellipse_points", 64))

    min_bound_h = float(tiling.get("tileset_bound_min_height_m", -1000.0))
    max_bound_h = float(tiling.get("tileset_bound_max_height_m", 10000.0))
    ge_root = float(tiling.get("geometric_error_root", 5000.0))
    ge_leaf = float(tiling.get("geometric_error_leaf", 100.0))

    section("Configuration")
    print(f"  Project root              : {project_root}")
    print(f"  Horizontal input          : {hfield_path}")
    print(f"  Uncertainty input         : {uncertainty_path}")
    print(f"  Height meta input         : {height_meta_path}")
    print(f"  Tile index input          : {tile_index_path}")
    print(f"  Arrows tileset output     : {arrows_tileset_path}")
    print(f"  Ellipses tileset output   : {ellipses_tileset_path}")
    print("  Ellipse scaling note      : Step16 axes in mm/yr, no hidden /100")

    section("Loading inputs")
    hfield = load_json(hfield_path)
    uncertainty = load_json(uncertainty_path)
    height_meta = load_json(height_meta_path)
    tile_index = load_json(tile_index_path)

    h_lookup = build_record_lookup(hfield)
    u_lookup = build_uncertainty_lookup(uncertainty)

    ok(f"Loaded horizontal records: {len(h_lookup)}")
    ok(f"Loaded uncertainty records: {len(u_lookup)}")
    ok(f"Loaded tile index: {len(tile_index.get('tiles', []))} tiles")

    section("Deriving horizontal visual scale")
    scaling = derive_horizontal_visual_scaling(
        cfg=cfg,
        hdev=hdev,
        h_lookup=h_lookup,
        u_lookup=u_lookup,
        tile_index=tile_index,
        horizontal_records=hfield.get("records", []),
    )
    print(f"  Scaling mode              : {scaling['mode']}")
    print(f"  RUM size                  : {scaling['rum_size_m']:.3f} m")
    print(f"  Arrow reference           : P{scaling['arrow_reference_percentile']} speed = {scaling['arrow_speed_ref_mm_yr']:.6f} mm/yr")
    print(f"  Arrow max length          : {scaling['arrow_max_length_m']:.3f} m")
    print(f"  Arrow scale               : {scaling['arrow_scale_m_per_mm_yr']:.6f} m per mm/yr")
    print(f"  Arrow anchor fraction     : {scaling['arrow_anchor_fraction_at_rum_center']:.3f} (tail=0, head=1)")
    print(f"  Arrow visibility          : speed >= {scaling['minimum_speed_mm_yr']:.6f} mm/yr" + (
        f" and speed >= {scaling['arrow_significance_sigma']:.2f} × 1σ_major" if scaling['arrow_significance_filter'] else ""
    ))
    print(f"  Ellipse reference         : P{scaling['ellipse_reference_percentile']} major axis = {scaling['ellipse_major_ref_mm_yr']:.6f} mm/yr")
    print(f"  Ellipse max diameter      : {scaling['ellipse_max_diameter_m']:.3f} m")
    print(f"  Ellipse scale mode        : {scaling['ellipse_scale_mode']}")
    print(f"  Ellipse centre placement  : {scaling['ellipse_center_placement']}")
    print(f"  Ellipse clip mode         : {scaling['ellipse_clip_mode']}")
    print(f"  Ellipse scale             : {scaling['ellipse_scale_m_per_mm_yr']:.6f} m per mm/yr")

    section("Building arrow B3DM tileset")
    arrow_summary = build_layer_tileset(
        layer_name="arrows",
        output_dir=arrows_dir,
        tileset_path=arrows_tileset_path,
        tile_index=tile_index,
        h_lookup=h_lookup,
        u_lookup=u_lookup,
        height_meta=height_meta,
        datum_height_m=datum_height_m,
        clearance_m=clearance_m,
        scaling=scaling,
        ellipse_points=ellipse_points,
        min_bound_h=min_bound_h,
        max_bound_h=max_bound_h,
        geometric_error_root=ge_root,
        geometric_error_leaf=ge_leaf,
    )
    ok(f"Arrow tiles: {arrow_summary['tiles']}, arrows={arrow_summary['features']}")

    section("Building confidence ellipse B3DM tileset")
    ellipse_summary = build_layer_tileset(
        layer_name="ellipses",
        output_dir=ellipses_dir,
        tileset_path=ellipses_tileset_path,
        tile_index=tile_index,
        h_lookup=h_lookup,
        u_lookup=u_lookup,
        height_meta=height_meta,
        datum_height_m=datum_height_m,
        clearance_m=clearance_m,
        scaling=scaling,
        ellipse_points=ellipse_points,
        min_bound_h=min_bound_h,
        max_bound_h=max_bound_h,
        geometric_error_root=ge_root,
        geometric_error_leaf=ge_leaf,
    )
    ok(f"Ellipse tiles: {ellipse_summary['tiles']}, ellipses={ellipse_summary['features']}")

    elapsed = time.time() - t_start

    section("Summary")
    ok(f"Step 17 complete in {elapsed:.2f} s")
    print(f"  Arrow tiles/features      : {arrow_summary['tiles']} / {arrow_summary['features']}")
    print(f"  Ellipse tiles/features    : {ellipse_summary['tiles']} / {ellipse_summary['features']}")
    print(f"  Arrow vertices/triangles  : {arrow_summary['vertices']} / {arrow_summary['triangles']}")
    print(f"  Ellipse vertices/triangles: {ellipse_summary['vertices']} / {ellipse_summary['triangles']}")
    print(f"  Skipped low-speed records : {arrow_summary['skipped_low_speed']}")
    print(f"  Skipped insignificant     : {arrow_summary['skipped_insignificant']}")
    print(f"  Arrows tileset            : {arrows_tileset_path}")
    print(f"  Ellipses tileset          : {ellipses_tileset_path}")


if __name__ == "__main__":
    main()
