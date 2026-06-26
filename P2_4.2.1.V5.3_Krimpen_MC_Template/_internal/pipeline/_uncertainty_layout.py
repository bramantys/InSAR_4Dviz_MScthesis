#!/usr/bin/env python3
"""Shared production engine for adaptive parcel-aware uncertainty layouts.

This is deliberately *not* a runtime-geometry builder.  It inspects every
moving parcel part and proposes the feature carrier layout that a later GLB
builder should use:

  grid_2d        wide part: parcel-local checkerboard grid
  centreline_row slender part: alternating up/down row along major axis
  compact_pair   small part: one adaptive up/down pair
  fallback_flat  too small/awkward: no fake spike geometry; retain flat cap

The engine writes a selected layout audit inside the disposable build stage.
It never edits the published runtime or viewer.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    import geopandas as gpd
    from shapely import affinity
    from shapely.geometry import GeometryCollection, LineString, MultiLineString, Polygon, box, mapping
except ImportError as exc:  # pragma: no cover - dependency environment specific
    raise SystemExit(
        "[FAIL] The uncertainty layout engine needs geopandas and shapely. "
        "Install the normal Proto2 geospatial environment first."
    ) from exc

from _pass3_common import Pass3Error, clean_stage_area, file_record, print_pass, project_root_from, write_json
from _proto2_config import load_project_config, output_data_dir


DELIVERY_ID_DEFAULT = "PROTO2_UNCERTAINTY_LAYOUTS_V5_1"
SCRIPT_ID_DEFAULT = "PROTO2_ADAPTIVE_UNCERTAINTY_LAYOUT"
SCHEMA = "proto2_uncertainty_adaptive_layout_v5_1_readable_rows"
METRIC_CRS_EPSG = 28992  # same metric diagnostic CRS used by Phase 03/04


@dataclass(frozen=True)
class Settings:
    target_spacing_m: float
    up_feature_size_m: float
    down_feature_size_m: float
    min_feature_size_m: float
    compact_spacing_factor: float
    grid_min_rows: int
    grid_min_columns: int
    centreline_offset_samples: int


@dataclass
class LayoutPlan:
    parcel_id: int
    footprint_id: str
    part_index: int
    layout_type: str
    layout_reason: str
    local_angle_deg: float
    long_axis_m: float
    short_axis_m: float
    aspect_ratio: float
    area_m2: float
    perimeter_m: float
    has_holes: bool
    source_geometry_type: str
    features_local: list[dict[str, Any]]
    local_polygon: Polygon
    local_origin: tuple[float, float]
    local_rotation_deg: float


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------

def _safe_polygon(geom: Any) -> Polygon:
    if isinstance(geom, Polygon):
        poly = geom
    else:
        raise Pass3Error(f"Expected Polygon footprint part, got {getattr(geom, 'geom_type', type(geom).__name__)}")
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or not isinstance(poly, Polygon):
        raise Pass3Error("Footprint part did not resolve to one valid Polygon")
    return poly


def _ring_without_closure(coords: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    out = [(float(x), float(y)) for x, y in coords]
    if len(out) >= 2 and out[0] == out[-1]:
        out = out[:-1]
    return out


def _major_axis_angle_deg(poly: Polygon) -> float:
    """Return angle of longest edge of the minimum rotated rectangle."""
    rect = poly.minimum_rotated_rectangle
    coords = _ring_without_closure(rect.exterior.coords)
    if len(coords) < 2:
        return 0.0
    best_len = -1.0
    best_angle = 0.0
    for i, (x0, y0) in enumerate(coords):
        x1, y1 = coords[(i + 1) % len(coords)]
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        if length > best_len:
            best_len = length
            best_angle = math.degrees(math.atan2(dy, dx))
    # Axis direction has 180° symmetry. Keep a stable half-turn representation.
    while best_angle <= -90.0:
        best_angle += 180.0
    while best_angle > 90.0:
        best_angle -= 180.0
    return best_angle


def _localize(poly: Polygon) -> tuple[Polygon, tuple[float, float], float]:
    """Rotate a metric parcel so its major axis lies on local +X."""
    origin = (float(poly.centroid.x), float(poly.centroid.y))
    angle_deg = _major_axis_angle_deg(poly)
    local = affinity.rotate(poly, -angle_deg, origin=origin, use_radians=False)
    return _safe_polygon(local), origin, angle_deg


def _restore_local_geometry(local_geom: Any, origin: tuple[float, float], angle_deg: float) -> Any:
    return affinity.rotate(local_geom, angle_deg, origin=origin, use_radians=False)


def _rect_dimensions(poly: Polygon) -> tuple[float, float]:
    rect = poly.minimum_rotated_rectangle
    coords = _ring_without_closure(rect.exterior.coords)
    if len(coords) < 4:
        minx, miny, maxx, maxy = poly.bounds
        a, b = abs(maxx - minx), abs(maxy - miny)
        return max(a, b), min(a, b)
    lengths: list[float] = []
    for i, (x0, y0) in enumerate(coords):
        x1, y1 = coords[(i + 1) % len(coords)]
        lengths.append(math.hypot(x1 - x0, y1 - y0))
    long_axis = max(lengths)
    short_axis = min(lengths)
    return long_axis, short_axis


def _square(cx: float, cy: float, side_m: float) -> Polygon:
    half = 0.5 * float(side_m)
    return box(cx - half, cy - half, cx + half, cy + half)


def _contains_square(poly: Polygon, cx: float, cy: float, side_m: float) -> bool:
    if side_m <= 0.0:
        return False
    # A tiny tolerance avoids false rejections from coordinate-rounding on a
    # square that exactly touches a parcel edge. Never buffer outward.
    return bool(poly.covers(_square(cx, cy, side_m).buffer(-1.0e-7)))


def _candidate_sizes(desired: float, minimum: float) -> list[float]:
    desired = float(desired)
    minimum = float(minimum)
    if desired <= minimum:
        return [minimum]
    values = np.linspace(desired, minimum, 12)
    # Keep distinct values after float rounding.
    out: list[float] = []
    for value in values:
        rounded = round(float(value), 3)
        if not out or abs(out[-1] - rounded) > 1.0e-9:
            out.append(rounded)
    return out


def _largest_fitting_size(poly: Polygon, cx: float, cy: float, desired: float, minimum: float) -> float | None:
    for side in _candidate_sizes(desired, minimum):
        if _contains_square(poly, cx, cy, side):
            return side
    return None


def _aligned_centres(low: float, high: float, step: float, anchor: float) -> list[float]:
    """Centres placed symmetrically around an anchor, never global map origin."""
    if high < low or step <= 0:
        return []
    k_min = math.floor((low - anchor) / step) - 1
    k_max = math.ceil((high - anchor) / step) + 1
    values = [anchor + k * step for k in range(k_min, k_max + 1)]
    return [value for value in values if low - 1.0e-7 <= value <= high + 1.0e-7]


def _flatten_lines(geom: Any) -> list[LineString]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom]
    if isinstance(geom, MultiLineString):
        return [line for line in geom.geoms if not line.is_empty]
    if isinstance(geom, GeometryCollection):
        lines: list[LineString] = []
        for item in geom.geoms:
            lines.extend(_flatten_lines(item))
        return lines
    return []


def _longest_horizontal_segment(poly: Polygon, y: float) -> LineString | None:
    minx, _miny, maxx, _maxy = poly.bounds
    pad = max(2.0, (maxx - minx) * 0.05)
    cut = LineString([(minx - pad, y), (maxx + pad, y)])
    lines = _flatten_lines(poly.intersection(cut))
    if not lines:
        return None
    return max(lines, key=lambda line: float(line.length))


def _centered_positions(
    start: float,
    end: float,
    spacing: float,
    minimum_count: int,
    edge_inset: float,
) -> list[float]:
    """Centres evenly spaced inside a segment, with room for full carriers."""
    length = float(end - start)
    inset = max(0.0, float(edge_inset))
    usable = length - 2.0 * inset
    if usable <= 0.0 or spacing <= 0.0:
        return []
    count = int(math.floor(usable / spacing)) + 1
    if count < minimum_count:
        return []
    used = (count - 1) * spacing
    lead = 0.5 * (usable - used)
    return [start + inset + lead + i * spacing for i in range(count)]


# -----------------------------------------------------------------------------
# Layout generators
# -----------------------------------------------------------------------------

def _make_feature(cx: float, cy: float, side: float, sign: int, ordinal: int, source: str) -> dict[str, Any]:
    return {
        "ordinal": int(ordinal),
        "sign": int(1 if sign >= 0 else -1),
        "feature_type": "up" if sign >= 0 else "down",
        "center_x_m": float(cx),
        "center_y_m": float(cy),
        "side_m": float(side),
        "source": source,
        "geometry": _square(cx, cy, side),
    }


def _grid_plan(poly: Polygon, settings: Settings) -> list[dict[str, Any]]:
    minx, miny, maxx, maxy = poly.bounds
    anchor_x = float(poly.centroid.x)
    anchor_y = float(poly.centroid.y)
    xs = _aligned_centres(minx, maxx, settings.target_spacing_m, anchor_x)
    ys = _aligned_centres(miny, maxy, settings.target_spacing_m, anchor_y)
    features: list[dict[str, Any]] = []
    valid_rows: set[int] = set()
    valid_cols: set[int] = set()
    ordinal = 0
    for iy, cy in enumerate(ys):
        for ix, cx in enumerate(xs):
            sign = 1 if ((ix + iy) % 2 == 0) else -1
            desired = settings.up_feature_size_m if sign > 0 else settings.down_feature_size_m
            if not _contains_square(poly, cx, cy, desired):
                continue
            features.append(_make_feature(cx, cy, desired, sign, ordinal, "grid_2d"))
            ordinal += 1
            valid_rows.add(iy)
            valid_cols.add(ix)
    if len(valid_rows) < settings.grid_min_rows or len(valid_cols) < settings.grid_min_columns:
        return []
    return features


def _centreline_candidates(poly: Polygon, sample_count: int) -> list[float]:
    minx, miny, maxx, maxy = poly.bounds
    representative_y = float(poly.representative_point().y)
    centroid_y = float(poly.centroid.y)
    centre_y = 0.5 * (miny + maxy)
    # Coordinates stay in the original metric frame after rotation; zero is
    # therefore not generally the parcel centreline.
    values = [representative_y, centroid_y, centre_y]
    if sample_count > 1 and maxy > miny:
        # Include interior cross-sections. Avoid the exact polygon boundary.
        values.extend(float(v) for v in np.linspace(miny + 0.08 * (maxy - miny), maxy - 0.08 * (maxy - miny), sample_count))
    unique: list[float] = []
    for value in values:
        if miny - 1e-7 <= value <= maxy + 1e-7 and all(abs(value - old) > 1e-5 for old in unique):
            unique.append(value)
    return unique


def _row_for_segment(
    poly: Polygon,
    segment: LineString,
    spacing: float,
    minimum_count: int,
    settings: Settings,
    source: str,
) -> list[dict[str, Any]]:
    coords = list(segment.coords)
    if len(coords) < 2:
        return []
    x0, y0 = coords[0]
    x1, y1 = coords[-1]
    if abs(y1 - y0) > 1e-5:
        return []
    start, end = sorted((float(x0), float(x1)))
    positions = _centered_positions(
        start,
        end,
        spacing,
        minimum_count,
        0.5 * max(settings.up_feature_size_m, settings.down_feature_size_m),
    )
    if not positions:
        return []

    # Find one common carrier size for this row. This avoids a weird sawtooth
    # of different base sizes while still allowing a genuinely narrow parcel to
    # receive a smaller, honest feature carrier.
    maxima: list[float] = []
    for index, cx in enumerate(positions):
        sign = 1 if index % 2 == 0 else -1
        desired = settings.up_feature_size_m if sign > 0 else settings.down_feature_size_m
        fit = _largest_fitting_size(poly, cx, y0, desired, settings.min_feature_size_m)
        if fit is None:
            return []
        maxima.append(fit)
    common = min(maxima)
    if common < settings.min_feature_size_m - 1.0e-8:
        return []

    features: list[dict[str, Any]] = []
    for ordinal, cx in enumerate(positions):
        sign = 1 if ordinal % 2 == 0 else -1
        # Preserve the up/down visual difference where the parcel permits it.
        desired = settings.up_feature_size_m if sign > 0 else settings.down_feature_size_m
        side = min(desired, common)
        if not _contains_square(poly, cx, y0, side):
            return []
        features.append(_make_feature(cx, y0, side, sign, ordinal, source))
    return features


def _centreline_plan(poly: Polygon, settings: Settings) -> list[dict[str, Any]]:
    candidates: list[tuple[tuple[int, float, float, float], list[dict[str, Any]]]] = []
    for y in _centreline_candidates(poly, settings.centreline_offset_samples):
        segment = _longest_horizontal_segment(poly, y)
        if segment is None:
            continue
        features = _row_for_segment(
            poly,
            segment,
            settings.target_spacing_m,
            2,
            settings,
            "centreline_row",
        )
        if not features:
            continue
        # Every row already meets the readable minimum. Prefer the row with
        # more carriers, then its largest common readable carrier, then support
        # length, then centrality. This prevents a very long but pinched row
        # from beating an equally useful, visually stronger row.
        local_cross_centre = 0.5 * (poly.bounds[1] + poly.bounds[3])
        common_side = min(float(feature["side_m"]) for feature in features)
        key = (
            len(features),
            common_side,
            float(segment.length),
            -abs(float(y) - local_cross_centre),
        )
        candidates.append((key, features))
    if not candidates:
        return []
    return max(candidates, key=lambda item: item[0])[1]


def _compact_pair_plan(poly: Polygon, settings: Settings) -> list[dict[str, Any]]:
    candidates: list[tuple[tuple[float, float], list[dict[str, Any]]]] = []
    for y in _centreline_candidates(poly, settings.centreline_offset_samples):
        segment = _longest_horizontal_segment(poly, y)
        if segment is None or segment.length <= 0.0:
            continue
        compact_spacing = min(
            settings.target_spacing_m * settings.compact_spacing_factor,
            float(segment.length) * 0.48,
        )
        compact_spacing = max(compact_spacing, settings.min_feature_size_m * 2.2)
        features = _row_for_segment(
            poly,
            segment,
            compact_spacing,
            2,
            settings,
            "compact_pair",
        )
        if len(features) != 2:
            continue
        local_cross_centre = 0.5 * (poly.bounds[1] + poly.bounds[3])
        key = (float(segment.length), -abs(float(y) - local_cross_centre))
        candidates.append((key, features))
    if not candidates:
        return []
    return max(candidates, key=lambda item: item[0])[1]


def make_layout_plan(row: pd.Series, metric_geom: Polygon, settings: Settings) -> LayoutPlan:
    poly = _safe_polygon(metric_geom)
    local_poly, origin, angle_deg = _localize(poly)
    long_axis_m, short_axis_m = _rect_dimensions(poly)
    area_m2 = float(poly.area)
    perimeter_m = float(poly.length)
    aspect_ratio = long_axis_m / max(short_axis_m, 1.0e-6)

    features = _grid_plan(local_poly, settings)
    if features:
        layout_type = "grid_2d"
        reason = "At least two valid rows and two valid columns fit inside the parcel-local footprint."
    else:
        features = _centreline_plan(local_poly, settings)
        if features:
            layout_type = "centreline_row"
            reason = "No honest two-dimensional grid fit; alternating carrier row follows the parcel major axis."
        else:
            features = _compact_pair_plan(local_poly, settings)
            if features:
                layout_type = "compact_pair"
                reason = "Parcel is too small or narrow for target spacing; one scaled, alternating up/down pair fits honestly."
            else:
                layout_type = "fallback_flat"
                reason = "No pair of full square carriers can fit without clipping; retain a flat uncertainty cap and mark fallback later."

    return LayoutPlan(
        parcel_id=int(row["parcel_id"]),
        footprint_id=str(row["footprint_id"]),
        part_index=int(row["part_index"]),
        layout_type=layout_type,
        layout_reason=reason,
        local_angle_deg=float(angle_deg),
        long_axis_m=float(long_axis_m),
        short_axis_m=float(short_axis_m),
        aspect_ratio=float(aspect_ratio),
        area_m2=area_m2,
        perimeter_m=perimeter_m,
        has_holes=bool(len(poly.interiors) > 0),
        source_geometry_type=str(row.get("source_geometry_type", "Polygon")),
        features_local=features,
        local_polygon=local_poly,
        local_origin=origin,
        local_rotation_deg=float(angle_deg),
    )


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------

def _part_row(plan: LayoutPlan) -> dict[str, Any]:
    features = plan.features_local
    signs = [int(feature["sign"]) for feature in features]
    sides = [float(feature["side_m"]) for feature in features]
    return {
        "parcel_id": plan.parcel_id,
        "footprint_id": plan.footprint_id,
        "part_index": plan.part_index,
        "layout_type": plan.layout_type,
        "layout_reason": plan.layout_reason,
        "feature_count": int(len(features)),
        "up_feature_count": int(sum(sign > 0 for sign in signs)),
        "down_feature_count": int(sum(sign < 0 for sign in signs)),
        "feature_side_min_m": float(min(sides)) if sides else None,
        "feature_side_max_m": float(max(sides)) if sides else None,
        "feature_side_mean_m": float(np.mean(sides)) if sides else None,
        "readable_min_feature_size_m": float(SETTINGS.min_feature_size_m),
        "below_readable_base": bool(
            sides and min(sides) < SETTINGS.min_feature_size_m - 1.0e-8
        ),
        "target_spacing_m": float(SETTINGS.target_spacing_m),
        "long_axis_m": plan.long_axis_m,
        "short_axis_m": plan.short_axis_m,
        "aspect_ratio": plan.aspect_ratio,
        "area_m2": plan.area_m2,
        "perimeter_m": plan.perimeter_m,
        "has_holes": plan.has_holes,
        "source_geometry_type": plan.source_geometry_type,
        # Actual real spike/dimple faces are four triangles per carrier.
        "spike_face_triangles": int(len(features) * 4),
        # Conservative planning number only: a later flat-remainder
        # triangulation around dimple holes adds edges/triangles. Do not treat
        # this as final GLB geometry count.
        "conservative_added_triangles": int(len(features) * 8),
    }


def _write_feature_point_geojson(plans: list[LayoutPlan], _parts_metric: gpd.GeoDataFrame, out_path: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for plan in plans:
        for feature in plan.features_local:
            world = _restore_local_geometry(feature["geometry"], plan.local_origin, plan.local_rotation_deg)
            center = world.centroid
            records.append({
                "parcel_id": plan.parcel_id,
                "footprint_id": plan.footprint_id,
                "part_index": plan.part_index,
                "layout_type": plan.layout_type,
                "feature_type": feature["feature_type"],
                "sign": int(feature["sign"]),
                "side_m": float(feature["side_m"]),
                "ordinal": int(feature["ordinal"]),
                "geometry": center,
            })
    if not records:
        out_path.write_text(
            json.dumps({"type": "FeatureCollection", "features": []}, ensure_ascii=False),
            encoding="utf-8",
        )
        return {"features": 0, "path": str(out_path)}
    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=f"EPSG:{METRIC_CRS_EPSG}").to_crs(epsg=4326)
    gdf.to_file(out_path, driver="GeoJSON")
    return {"features": int(len(gdf)), "path": str(out_path)}


def _svg_path_for_polygon(poly: Polygon, transform, *, fill: str, stroke: str, stroke_width: float, opacity: float = 1.0) -> str:
    def p(coords: Iterable[tuple[float, float]]) -> str:
        values = list(coords)
        out: list[str] = []
        for x, y in values:
            sx, sy = transform(float(x), float(y))
            out.append(f"{sx:.2f},{sy:.2f}")
        return " ".join(out)
    fragments = [f'<polygon points="{p(poly.exterior.coords)}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width:.2f}" opacity="{opacity:.3f}" fill-rule="evenodd"/>']
    for ring in poly.interiors:
        fragments.append(f'<polygon points="{p(ring.coords)}" fill="#ffffff" stroke="{stroke}" stroke-width="{stroke_width:.2f}" opacity="1.0"/>')
    return "\n".join(fragments)


def _gallery_card(plan: LayoutPlan, width: int = 380, height: int = 300) -> str:
    poly = plan.local_polygon
    all_geoms = [poly] + [feature["geometry"] for feature in plan.features_local]
    minx = min(geom.bounds[0] for geom in all_geoms)
    miny = min(geom.bounds[1] for geom in all_geoms)
    maxx = max(geom.bounds[2] for geom in all_geoms)
    maxy = max(geom.bounds[3] for geom in all_geoms)
    span_x = max(maxx - minx, 1.0)
    span_y = max(maxy - miny, 1.0)
    pad = 22.0
    scale = min((width - 2.0 * pad) / span_x, (height - 2.0 * pad) / span_y)

    def transform(x: float, y: float) -> tuple[float, float]:
        return pad + (x - minx) * scale, height - pad - (y - miny) * scale

    exterior = _svg_path_for_polygon(poly, transform, fill="#edf1f5", stroke="#263238", stroke_width=1.4)
    feature_svg: list[str] = []
    for feature in plan.features_local:
        world = feature["geometry"]
        fill = "#c58b63" if feature["sign"] > 0 else "#617f9f"
        stroke = "#5b3b26" if feature["sign"] > 0 else "#314b66"
        feature_svg.append(_svg_path_for_polygon(world, transform, fill=fill, stroke=stroke, stroke_width=0.7, opacity=0.88))
        cx, cy = transform(feature["center_x_m"], feature["center_y_m"])
        glyph = "▲" if feature["sign"] > 0 else "▼"
        feature_svg.append(f'<text x="{cx:.2f}" y="{cy + 3.2:.2f}" text-anchor="middle" font-size="10" fill="#ffffff">{glyph}</text>')

    title = f"Parcel {plan.parcel_id} · part {plan.part_index}"
    meta = (
        f"{plan.layout_type} · {len(plan.features_local)} carrier(s) · "
        f"{plan.long_axis_m:.0f} × {plan.short_axis_m:.0f} m · aspect {plan.aspect_ratio:.1f}"
    )
    return f"""
<section class=\"card\">
  <h2>{html.escape(title)}</h2>
  <div class=\"meta\">{html.escape(meta)}</div>
  <svg viewBox=\"0 0 {width} {height}\" role=\"img\" aria-label=\"{html.escape(title)}\">
    <rect x=\"0\" y=\"0\" width=\"{width}\" height=\"{height}\" fill=\"#f9fafb\"/>
    {exterior}
    {''.join(feature_svg)}
  </svg>
  <p>{html.escape(plan.layout_reason)}</p>
</section>
"""


def _choose_representative_plans(plans: list[LayoutPlan]) -> list[LayoutPlan]:
    selected: list[LayoutPlan] = []
    used: set[tuple[int, int]] = set()

    def add(plan: LayoutPlan | None) -> None:
        if plan is None:
            return
        key = (plan.parcel_id, plan.part_index)
        if key not in used:
            selected.append(plan)
            used.add(key)

    add(next((p for p in plans if p.parcel_id == 90647), None))

    by_type: dict[str, list[LayoutPlan]] = {}
    for plan in plans:
        by_type.setdefault(plan.layout_type, []).append(plan)

    for layout_type in ["grid_2d", "centreline_row", "compact_pair", "fallback_flat"]:
        group = by_type.get(layout_type, [])
        if not group:
            continue
        if layout_type == "grid_2d":
            counts = sorted(len(p.features_local) for p in group)
            target = counts[len(counts) // 2]
            add(min(group, key=lambda p: abs(len(p.features_local) - target)))
        elif layout_type == "centreline_row":
            add(max(group, key=lambda p: p.aspect_ratio))
            add(max(group, key=lambda p: len(p.features_local)))
        elif layout_type == "compact_pair":
            add(min(group, key=lambda p: p.area_m2))
        else:
            add(min(group, key=lambda p: p.area_m2))

    with_holes = [p for p in plans if p.has_holes]
    if with_holes:
        add(max(with_holes, key=lambda p: p.perimeter_m))

    multiparts = Counter(p.parcel_id for p in plans)
    multipart_plans = [p for p in plans if multiparts[p.parcel_id] > 1]
    if multipart_plans:
        add(max(multipart_plans, key=lambda p: p.aspect_ratio))

    add(max(plans, key=lambda p: len(p.features_local), default=None))
    add(min((p for p in plans if p.features_local), key=lambda p: p.area_m2, default=None))
    return selected[:12]


def _write_gallery(plans: list[LayoutPlan], summary: dict[str, Any], out_path: Path) -> None:
    cards = "\n".join(_gallery_card(plan) for plan in _choose_representative_plans(plans))
    counts = summary["layout_counts"]
    out_path.write_text(
        f"""<!doctype html>
<html lang=\"en\"><head><meta charset=\"utf-8\"/>
<title>Proto2 adaptive uncertainty-layout audit</title>
<style>
body {{ margin: 24px; background: #f4f6f8; color: #192129; font-family: system-ui, sans-serif; }}
h1 {{ margin: 0 0 6px; }} .lead {{ max-width: 1040px; line-height: 1.45; color:#455a64; }}
.summary {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:10px; margin:18px 0 26px; }}
.metric {{ background:#fff; border:1px solid #d8dee4; border-radius:10px; padding:12px; }}
.metric b {{ display:block; font-size:22px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(390px,1fr)); gap:16px; }}
.card {{ background:#fff; border:1px solid #d8dee4; border-radius:12px; padding:13px; box-shadow:0 2px 7px rgba(30,55,70,.08); }}
h2 {{ font-size:15px; margin:0; }} .meta {{ margin:4px 0 9px; color:#607d8b; font-size:12px; }}
svg {{ width:100%; border:1px solid #d8dee4; border-radius:8px; }} p {{ min-height:34px; font-size:12px; line-height:1.4; color:#455a64; }}
code {{ background:#eef2f5; padding:1px 4px; border-radius:4px; }}
</style></head><body>
<h1>Adaptive parcel-aware uncertainty layout — Gate 1 v2</h1>
<p class=\"lead\">Delivery: <code>{summary['delivery_id']}</code>. Target spacing: <b>{summary['settings']['target_spacing_m']} m</b>. Readable carrier minimum: <b>{summary['settings']['readable_min_feature_size_m']} m</b>. Every card uses a parcel-local major-axis frame. Orange squares are upward pyramid carriers; blue squares are downward dimple carriers. The viewer has not been changed: this is an audit of the future GLB layout logic only.</p>
<div class=\"summary\">
  <div class=\"metric\"><span>Moving parts</span><b>{summary['moving_parts']:,}</b></div>
  <div class=\"metric\"><span>Grid 2D</span><b>{counts.get('grid_2d',0):,}</b></div>
  <div class=\"metric\"><span>Centreline rows</span><b>{counts.get('centreline_row',0):,}</b></div>
  <div class=\"metric\"><span>Compact pairs</span><b>{counts.get('compact_pair',0):,}</b></div>
  <div class=\"metric\"><span>Flat fallbacks</span><b>{counts.get('fallback_flat',0):,}</b></div>
  <div class=\"metric\"><span>Total carriers</span><b>{summary['total_features']:,}</b></div>
  <div class=\"metric\"><span>Below readable min</span><b>{summary['parts_below_readable_base']:,}</b></div>
</div>
<div class=\"grid\">{cards}</div>
</body></html>""",
        encoding="utf-8",
    )


def _write_representative_feature_geojson(plans: list[LayoutPlan], out_path: Path) -> int:
    selected = _choose_representative_plans(plans)
    features: list[dict[str, Any]] = []
    for plan in selected:
        for feature in plan.features_local:
            world = _restore_local_geometry(feature["geometry"], plan.local_origin, plan.local_rotation_deg)
            features.append({
                "type": "Feature",
                "properties": {
                    "parcel_id": plan.parcel_id,
                    "footprint_id": plan.footprint_id,
                    "part_index": plan.part_index,
                    "layout_type": plan.layout_type,
                    "feature_type": feature["feature_type"],
                    "sign": feature["sign"],
                    "side_m": feature["side_m"],
                },
                "geometry": mapping(world),
            })
    # The geometry is metric EPSG:28992. GeoJSON's CRS member is deprecated,
    # but retained as a practical audit hint for QGIS users.
    payload = {
        "type": "FeatureCollection",
        "name": "uncertainty_layout_representative_features_metric",
        "crs": {"type": "name", "properties": {"name": "EPSG:28992"}},
        "features": features,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return len(features)


def _parcel_summary(part_df: pd.DataFrame) -> pd.DataFrame:
    aggregations: dict[str, Any] = {
        "feature_count": "sum",
        "up_feature_count": "sum",
        "down_feature_count": "sum",
        "spike_face_triangles": "sum",
        "conservative_added_triangles": "sum",
        "area_m2": "sum",
        "perimeter_m": "sum",
        "has_holes": "max",
    }
    grouped = part_df.groupby("parcel_id", as_index=False).agg(aggregations)
    layout_order = {"grid_2d": 0, "centreline_row": 1, "compact_pair": 2, "fallback_flat": 3}
    work = part_df[["parcel_id", "layout_type"]].copy()
    work["layout_rank"] = work["layout_type"].map(layout_order).fillna(99)
    representative = work.sort_values(["parcel_id", "layout_rank"]).drop_duplicates("parcel_id")
    grouped = grouped.merge(representative[["parcel_id", "layout_type"]], on="parcel_id", how="left")
    grouped["part_count"] = part_df.groupby("parcel_id").size().reindex(grouped["parcel_id"]).to_numpy()
    fallback_counts = (
        part_df.assign(is_fallback=part_df["layout_type"].eq("fallback_flat"))
        .groupby("parcel_id")["is_fallback"].sum()
        .reindex(grouped["parcel_id"])
        .fillna(0)
        .astype(int)
        .to_numpy()
    )
    grouped["fallback_part_count"] = fallback_counts
    grouped["has_any_fallback"] = grouped["fallback_part_count"].gt(0)
    return grouped.sort_values("parcel_id").reset_index(drop=True)


# This global is set by main so _part_row can remain intentionally compact.
SETTINGS: Settings


def _run_v2_self_test() -> None:
    """Synthetic checks proving the 2.5 m readability guard is active."""
    settings = Settings(
        target_spacing_m=20.0,
        up_feature_size_m=3.5,
        down_feature_size_m=4.5,
        min_feature_size_m=2.5,
        compact_spacing_factor=0.55,
        grid_min_rows=2,
        grid_min_columns=2,
        centreline_offset_samples=13,
    )

    # A wide parcel must receive a proper 2D layout at readable bases.
    wide = box(-60.0, -40.0, 60.0, 40.0)
    wide_features = _grid_plan(wide, settings)
    if not wide_features or min(feature["side_m"] for feature in wide_features) < 2.5:
        raise Pass3Error("V2 self-test failed: wide parcel did not retain readable grid carriers.")

    # A long narrow corridor should receive a single row, but never <2.5 m.
    slender = box(-120.0, -2.0, 120.0, 2.0)
    slender_features = _centreline_plan(slender, settings)
    if not slender_features or min(feature["side_m"] for feature in slender_features) < 2.5:
        raise Pass3Error("V2 self-test failed: slender parcel did not retain readable centreline carriers.")

    # A corridor narrower than a 2.5 m carrier must not emit fake tiny carriers.
    too_narrow = box(-120.0, -1.1, 120.0, 1.1)
    if _centreline_plan(too_narrow, settings) or _compact_pair_plan(too_narrow, settings):
        raise Pass3Error("V2 self-test failed: sub-readable corridor emitted a carrier.")

    print("[SELF-TEST] PASS — enforces a 2.5 m readable carrier minimum.", flush=True)



def main() -> int:
    parser = argparse.ArgumentParser(description="Build one adaptive parcel-aware uncertainty layout audit.")
    parser.add_argument("--spacing", type=float, default=20.0, help="Target carrier spacing in metres (default: 20).")
    parser.add_argument("--up-base", type=float, default=3.5, help="Preferred upward-pyramid square base in metres.")
    parser.add_argument("--down-base", type=float, default=4.5, help="Preferred downward-dimple square base in metres.")
    parser.add_argument(
        "--min-base", "--readable-min-base", dest="min_base", type=float, default=2.5,
        help="Minimum readable full-square carrier in metres (default: 2.5).",
    )
    parser.add_argument("--centreline-samples", type=int, default=13, help="Cross-axis candidates tested for slender parcels.")
    parser.add_argument("--stage-name", default="uncertainty_layout_detail_20m", help="Disposable build-stage folder name.")
    parser.add_argument("--lod-role", default="detail", choices=["detail", "overview"], help="LOD role documented in the audit.")
    parser.add_argument("--delivery-id", default=DELIVERY_ID_DEFAULT)
    parser.add_argument("--script-id", default=SCRIPT_ID_DEFAULT)
    parser.add_argument("--self-test", action="store_true", help="Run v2 synthetic layout checks only; do not read project data.")
    args = parser.parse_args()

    if args.spacing <= 0 or args.up_base <= 0 or args.down_base <= 0 or args.min_base <= 0:
        raise Pass3Error("Spacing and feature sizes must be positive")
    if args.min_base > min(args.up_base, args.down_base):
        raise Pass3Error("--min-base cannot exceed both preferred feature bases")

    global SETTINGS
    SETTINGS = Settings(
        target_spacing_m=float(args.spacing),
        up_feature_size_m=float(args.up_base),
        down_feature_size_m=float(args.down_base),
        min_feature_size_m=float(args.min_base),
        compact_spacing_factor=0.55,
        grid_min_rows=2,
        grid_min_columns=2,
        centreline_offset_samples=max(3, int(args.centreline_samples)),
    )

    if args.self_test:
        _run_v2_self_test()
        return 0

    project_root = project_root_from(__file__)
    pipeline_dir = Path(__file__).resolve().parent
    config = load_project_config(project_root)
    output_data = output_data_dir(project_root, config)
    parts_path = output_data / "parcel_footprints_parts.parquet"
    if not parts_path.exists():
        raise Pass3Error(
            f"Missing {parts_path}. Run the normal pipeline through Phase 03 first "
            "(or complete a normal full pipeline run)."
        )

    if not str(args.stage_name).replace("_", "").isalnum():
        raise Pass3Error("--stage-name may contain only letters, numbers, and underscores")
    stage_dir = clean_stage_area(project_root, str(args.stage_name))
    print("\n=== PROTO2: ADAPTIVE UNCERTAINTY LAYOUT AUDIT ===")
    print(f"Delivery ID  : {args.delivery_id}")
    print(f"Script ID    : {args.script_id}")
    print(f"Project root : {project_root}")
    print(f"Target space : {SETTINGS.target_spacing_m:.1f} m")
    print(f"Readable base: {SETTINGS.min_feature_size_m:.1f} m minimum")
    print(f"Source       : {parts_path}")

    parts = gpd.read_parquet(parts_path)
    if parts.crs is None:
        raise Pass3Error("parcel_footprints_parts.parquet has no CRS")
    parts_metric = parts.to_crs(epsg=METRIC_CRS_EPSG)
    required = {"parcel_id", "footprint_id", "part_index", "has_displacement", "geometry"}
    missing = required - set(parts_metric.columns)
    if missing:
        raise Pass3Error(f"Footprint parts missing required columns: {sorted(missing)}")

    moving = parts_metric.loc[parts_metric["has_displacement"].astype(bool)].copy()
    if moving.empty:
        raise Pass3Error("No moving parcel parts are available for the uncertainty audit")
    print(f"Moving parts : {len(moving):,}")

    plans: list[LayoutPlan] = []
    failures: list[dict[str, Any]] = []
    for index, row in moving.iterrows():
        try:
            plans.append(make_layout_plan(row, _safe_polygon(row.geometry), SETTINGS))
        except Exception as exc:
            failures.append({
                "parcel_id": int(row.get("parcel_id", -1)),
                "footprint_id": str(row.get("footprint_id", "?")),
                "error": str(exc),
            })

    if failures:
        failure_path = stage_dir / "uncertainty_layout_failures.json"
        write_json(failure_path, failures)
        raise Pass3Error(f"Layout audit failed for {len(failures)} parcel part(s). See {failure_path}")

    part_df = pd.DataFrame([_part_row(plan) for plan in plans]).sort_values(["parcel_id", "part_index"]).reset_index(drop=True)
    parcel_df = _parcel_summary(part_df)

    layout_counts = {str(k): int(v) for k, v in Counter(part_df["layout_type"]).items()}
    total_features = int(part_df["feature_count"].sum())
    total_spike_triangles = int(part_df["spike_face_triangles"].sum())
    conservative_added = int(part_df["conservative_added_triangles"].sum())
    fallback_parts = int((part_df["layout_type"] == "fallback_flat").sum())
    fallback_parcels = int((parcel_df["feature_count"] == 0).sum())
    parcels_with_any_fallback_part = int(parcel_df["has_any_fallback"].sum())
    parts_below_readable_base = int(part_df["below_readable_base"].astype(bool).sum())
    if parts_below_readable_base:
        raise Pass3Error(
            f"Readable-carrier guard failed: {parts_below_readable_base} part(s) contain a carrier below "
            f"{SETTINGS.min_feature_size_m:.2f} m. No audit output was accepted."
        )

    part_csv = stage_dir / "uncertainty_layout_part_audit.csv"
    parcel_csv = stage_dir / "uncertainty_layout_parcel_audit.csv"
    points_geojson = stage_dir / "uncertainty_layout_feature_centres.geojson"
    representative_geojson = stage_dir / "uncertainty_layout_representative_features_metric.geojson"
    gallery_html = stage_dir / "uncertainty_layout_gallery.html"

    part_df.to_csv(part_csv, index=False)
    parcel_df.to_csv(parcel_csv, index=False)
    point_info = _write_feature_point_geojson(plans, parts_metric, points_geojson)
    representative_count = _write_representative_feature_geojson(plans, representative_geojson)

    summary: dict[str, Any] = {
        "delivery_id": args.delivery_id,
        "script_id": args.script_id,
        "lod_role": args.lod_role,
        "schema": SCHEMA,
        "scope": "production_layout_audit_only_no_glb_no_viewer_change",
        "settings": {
            "target_spacing_m": SETTINGS.target_spacing_m,
            "up_feature_size_m": SETTINGS.up_feature_size_m,
            "down_feature_size_m": SETTINGS.down_feature_size_m,
            "min_feature_size_m": SETTINGS.min_feature_size_m,
            "readable_min_feature_size_m": SETTINGS.min_feature_size_m,
            "compact_spacing_factor": SETTINGS.compact_spacing_factor,
            "grid_min_rows": SETTINGS.grid_min_rows,
            "grid_min_columns": SETTINGS.grid_min_columns,
            "centreline_offset_samples": SETTINGS.centreline_offset_samples,
            "metric_crs": f"EPSG:{METRIC_CRS_EPSG}",
        },
        "moving_parts": int(len(plans)),
        "moving_parcels": int(parcel_df["parcel_id"].nunique()),
        "layout_counts": layout_counts,
        "total_features": total_features,
        "total_spike_face_triangles": total_spike_triangles,
        "conservative_added_triangle_budget": conservative_added,
        "fallback_parts": fallback_parts,
        "fallback_parcels": fallback_parcels,
        "parcels_with_any_fallback_part": parcels_with_any_fallback_part,
        "parts_below_readable_base": parts_below_readable_base,
        "feature_count_distribution": {
            "min": int(part_df["feature_count"].min()),
            "p25": float(part_df["feature_count"].quantile(0.25)),
            "median": float(part_df["feature_count"].median()),
            "p75": float(part_df["feature_count"].quantile(0.75)),
            "p95": float(part_df["feature_count"].quantile(0.95)),
            "max": int(part_df["feature_count"].max()),
        },
        "notes": [
            "This layout audit does not build any GLB or edit the viewer.",
            "Feature spacing and placement are display carriers only; they do not encode local changes in parcel uncertainty.",
            "Every later real-geometry implementation must keep the entire coloured flat parcel remainder, including fallback-flat parts.",
            "Triangle counts are planning figures only. Final flat-remainder triangulation around dimple holes must report its actual count separately.",
            "Hard guard: no accepted part may contain a carrier below the readable minimum base.",
        ],
    }
    _write_gallery(plans, summary, gallery_html)
    summary["outputs"] = {
        "part_audit_csv": file_record(part_csv, project_root),
        "parcel_audit_csv": file_record(parcel_csv, project_root),
        "feature_centres_geojson": file_record(points_geojson, project_root),
        "representative_features_metric_geojson": file_record(representative_geojson, project_root),
        "gallery_html": file_record(gallery_html, project_root),
        "representative_feature_polygons": representative_count,
        "full_feature_centres": point_info["features"],
    }
    summary_path = stage_dir / "uncertainty_layout_summary.json"
    write_json(summary_path, summary)

    print("\n--- Layout result ---")
    for layout_type in ["grid_2d", "centreline_row", "compact_pair", "fallback_flat"]:
        print(f"{layout_type:16s}: {layout_counts.get(layout_type, 0):,}")
    print(f"Carrier features : {total_features:,}")
    print(f"Spike faces      : {total_spike_triangles:,} triangles")
    print(f"Below readable base: {parts_below_readable_base:,}")
    print(f"Fallback parcels : {fallback_parcels:,}")
    print(f"Gallery          : {gallery_html}")
    print_pass("UNCERTAINTY LAYOUT RESULT", summary_path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Pass3Error as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        raise SystemExit(1)
