# -*- coding: utf-8 -*-
import json
import math
from pathlib import Path

import numpy as np
from pyproj import Transformer


ROOT = Path(__file__).resolve().parents[1]

HFIELD = ROOT / "Data" / "horizontal_field.json"
OUT_ARROWS = ROOT / "Data" / "horizontal_arrows_wgs84.geojson"
OUT_ELLIPSES = ROOT / "Data" / "horizontal_confidence_ellipses_wgs84.geojson"

SOURCE_CRS = "EPSG:23830"
TARGET_CRS = "EPSG:4326"

# Design decision:
# 20 mm/yr = 450 m = one RUM width

SCALING = 22.5  # m per mm/yr

ELLIPSE_SCALE_MULT = 5.0
ELLIPSE_SCALING = SCALING * ELLIPSE_SCALE_MULT

MIN_SPEED_MM_YR = 0.2
N_STD = 2.0
ELLIPSE_POINTS = 64

# Arrowhead settings
ARROWHEAD_ANGLE_DEG = 28.0
ARROWHEAD_FRAC = 0.22
ARROWHEAD_MIN_M = 35.0
ARROWHEAD_MAX_M = 120.0

to_wgs84 = Transformer.from_crs(SOURCE_CRS, TARGET_CRS, always_xy=True)
to_projected = Transformer.from_crs(TARGET_CRS, SOURCE_CRS, always_xy=True)


RUM_DISPLAY_DATUM_M = 1000.0
DIAGNOSTIC_CLEARANCE_M = 5.0
DIAGNOSTIC_HEIGHT_M = RUM_DISPLAY_DATUM_M + DIAGNOSTIC_CLEARANCE_M

def xy_to_lonlat(x, y):
    lon, lat = to_wgs84.transform(float(x), float(y))
    return [float(lon), float(lat), DIAGNOSTIC_HEIGHT_M]


def lonlat_to_xy(lon, lat):
    x, y = to_projected.transform(float(lon), float(lat))
    return float(x), float(y)


def get_any(obj, names, default=None):
    for name in names:
        if name in obj and obj[name] is not None:
            return obj[name]
    return default


def as_float(value, default=0.0):
    try:
        n = float(value)
        if math.isfinite(n):
            return n
    except Exception:
        pass
    return float(default)


def as_optional_float(value):
    try:
        n = float(value)
        if math.isfinite(n):
            return n
    except Exception:
        pass
    return None


def line_feature(coords, props):
    return {
        "type": "Feature",
        "properties": props,
        "geometry": {
            "type": "LineString",
            "coordinates": coords
        }
    }


def ellipse_coords_projected(cx, cy, cov_scaled):
    eigvals, eigvecs = np.linalg.eigh(cov_scaled)
    order = eigvals.argsort()[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)
    eigvecs = eigvecs[:, order]

    # Alex-style: width/height = N_STD * sigma.
    # Direct coordinate drawing needs semi-axis = half width/height.
    semi_major = 0.5 * N_STD * math.sqrt(eigvals[0])
    semi_minor = 0.5 * N_STD * math.sqrt(eigvals[1])

    theta = np.linspace(0, 2 * np.pi, ELLIPSE_POINTS, endpoint=True)
    unit = np.vstack([np.cos(theta), np.sin(theta)])
    axes = np.diag([semi_major, semi_minor])
    pts = eigvecs @ axes @ unit

    xs = cx + pts[0, :]
    ys = cy + pts[1, :]

    coords = [xy_to_lonlat(x, y) for x, y in zip(xs, ys)]

    if coords[0] != coords[-1]:
        coords.append(coords[0])

    angle_deg = math.degrees(math.atan2(eigvecs[1, 0], eigvecs[0, 0]))

    return coords, semi_major, semi_minor, angle_deg


def build_centered_arrow_projected(x0, y0, east, north, props):
    """
    Shaft center = RUM center.
    Arrow tip = forward end.
    """
    dx = east * SCALING
    dy = north * SCALING

    x_start = x0 - 0.5 * dx
    y_start = y0 - 0.5 * dy

    x_end = x0 + 0.5 * dx
    y_end = y0 + 0.5 * dy

    out = []

    out.append(
        line_feature(
            [xy_to_lonlat(x_start, y_start), xy_to_lonlat(x_end, y_end)],
            {**props, "arrow_part": "shaft"}
        )
    )

    length = math.hypot(dx, dy)
    if length <= 1e-9:
        return out, x_end, y_end

    head_len = max(
        ARROWHEAD_MIN_M,
        min(ARROWHEAD_MAX_M, ARROWHEAD_FRAC * length)
    )

    heading = math.atan2(dy, dx)
    head_angle = math.radians(ARROWHEAD_ANGLE_DEG)

    # Two head strokes from tip backward
    xh1 = x_end - head_len * math.cos(heading - head_angle)
    yh1 = y_end - head_len * math.sin(heading - head_angle)

    xh2 = x_end - head_len * math.cos(heading + head_angle)
    yh2 = y_end - head_len * math.sin(heading + head_angle)

    out.append(
        line_feature(
            [xy_to_lonlat(x_end, y_end), xy_to_lonlat(xh1, yh1)],
            {**props, "arrow_part": "head_left"}
        )
    )

    out.append(
        line_feature(
            [xy_to_lonlat(x_end, y_end), xy_to_lonlat(xh2, yh2)],
            {**props, "arrow_part": "head_right"}
        )
    )

    return out, x_end, y_end


def get_horizontal_records(data):
    """
    Supports both possible layouts:
    - {"cells": [...]}
    - {"features": [...]}
    - GeoJSON-like {"features": [{"properties": {...}, "geometry": ...}]}
    """
    if isinstance(data, dict):
        if isinstance(data.get("cells"), list):
            return data["cells"], "cells"

        if isinstance(data.get("features"), list):
            records = []
            for f in data["features"]:
                if isinstance(f, dict) and "properties" in f:
                    props = dict(f.get("properties") or {})
                    geom = f.get("geometry") or {}
                    coords = geom.get("coordinates")
                    if (
                        isinstance(coords, list)
                        and len(coords) >= 2
                        and "lon" not in props
                        and "lat" not in props
                    ):
                        props["lon"] = coords[0]
                        props["lat"] = coords[1]
                    records.append(props)
                else:
                    records.append(f)
            return records, "features"

    raise ValueError("horizontal_field.json has neither 'cells' nor 'features'")


def main():
    print("=" * 72)
    print("Step 19 - Build horizontal arrows and confidence ellipses")
    print("=" * 72)

    print(f"Input horizontal field : {HFIELD}")
    print(f"Output arrows          : {OUT_ARROWS}")
    print(f"Output ellipses        : {OUT_ELLIPSES}")
    print(f"Source CRS             : {SOURCE_CRS}")
    print(f"SCALING                : {SCALING} m per mm/yr")
    print(f"MIN_SPEED_MM_YR        : {MIN_SPEED_MM_YR}")
    print(f"N_STD                  : {N_STD}")

    if not HFIELD.exists():
        raise FileNotFoundError(HFIELD)

    data = json.loads(HFIELD.read_text(encoding="utf-8"))
    records, layout = get_horizontal_records(data)

    print(f"Detected layout        : {layout}")
    print(f"Input records          : {len(records)}")

    arrow_features = []
    ellipse_features = []

    kept = 0
    skipped_speed = 0
    skipped_missing_position = 0
    skipped_bad = 0

    for r in records:
        if not isinstance(r, dict):
            skipped_bad += 1
            continue

        rum_id = str(get_any(r, ["rum_id", "rumId", "RUM_ID", "id"], ""))

        east = as_float(get_any(r, ["east_mm_yr", "east", "E", "ve"], 0.0))
        north = as_float(get_any(r, ["north_mm_yr", "north", "N", "vn"], 0.0))
        speed = as_float(
            get_any(r, ["speed_mm_yr", "speed", "horizontal_speed"], None),
            math.hypot(east, north)
        )

        if speed < MIN_SPEED_MM_YR:
            skipped_speed += 1
            continue

        x0_raw = get_any(r, ["x_rum", "x", "easting", "E_rum"], None)
        y0_raw = get_any(r, ["y_rum", "y", "northing", "N_rum"], None)

        if x0_raw is not None and y0_raw is not None:
            x0_opt = as_optional_float(x0_raw)
            y0_opt = as_optional_float(y0_raw)

            if x0_opt is None or y0_opt is None:
                skipped_missing_position += 1
                continue

            x0 = x0_opt
            y0 = y0_opt

        else:
            lon_raw = get_any(r, ["lon", "longitude"], None)
            lat_raw = get_any(r, ["lat", "latitude"], None)

            if lon_raw is None or lat_raw is None:
                skipped_missing_position += 1
                continue

            x0, y0 = lonlat_to_xy(lon_raw, lat_raw)

        var_east = as_float(get_any(r, ["var_east", "variance_east"], 0.0))
        var_north = as_float(get_any(r, ["var_north", "variance_north"], 0.0))
        covar_en = as_float(get_any(r, ["covar_en", "cov_en", "covar_east_north"], 0.0))

        props = {
            "rum_id": rum_id,
            "east_mm_yr": east,
            "north_mm_yr": north,
            "speed_mm_yr": speed,
            "scale_m_per_mm_yr": SCALING,
            "scale_rule": "20 mm/yr = 450 m",
            "min_speed_mm_yr": MIN_SPEED_MM_YR,
            "n_std": N_STD,
            "arrow_centered_on_rum": True
        }

        arrow_parts, x_tip, y_tip = build_centered_arrow_projected(
            x0, y0, east, north, props
        )
        arrow_features.extend(arrow_parts)

        q_scaled = np.array([
            [var_east * ELLIPSE_SCALING**2, covar_en * ELLIPSE_SCALING**2],
            [covar_en * ELLIPSE_SCALING**2, var_north * ELLIPSE_SCALING**2]
        ])

        ell_coords, semi_major, semi_minor, angle_deg = ellipse_coords_projected(
            x_tip, y_tip, q_scaled
        )

        ellipse_features.append(
            line_feature(
                ell_coords,
                {
                    **props,
                    "kind": "horizontal_2sigma_confidence_ellipse",
                    "ellipse_scale_mult": ELLIPSE_SCALE_MULT,
                    "ellipse_scale_rule": "visual exaggeration applied to uncertainty ellipse only",
                    "semi_major_m": semi_major,
                    "semi_minor_m": semi_minor,
                    "angle_deg_from_east": angle_deg
                }
            )
        )

        kept += 1

    arrows_fc = {
        "type": "FeatureCollection",
        "name": "horizontal_arrows_wgs84",
        "features": arrow_features
    }

    ellipses_fc = {
        "type": "FeatureCollection",
        "name": "horizontal_confidence_ellipses_wgs84",
        "features": ellipse_features
    }

    OUT_ARROWS.write_text(json.dumps(arrows_fc), encoding="utf-8")
    OUT_ELLIPSES.write_text(json.dumps(ellipses_fc), encoding="utf-8")

    print("-" * 72)
    print(f"RUM vectors kept                       : {kept}")
    print(f"Skipped below speed threshold          : {skipped_speed}")
    print(f"Skipped missing position               : {skipped_missing_position}")
    print(f"Skipped bad records                    : {skipped_bad}")
    print(f"Arrow line features written            : {len(arrow_features)}")
    print(f"Ellipse features written               : {len(ellipse_features)}")
    print("[OK] Step 19 complete")
    print("=" * 72)


if __name__ == "__main__":
    main()