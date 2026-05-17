#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10_build_blank_cells.py

Generic RUM-based InSAR template step.

Purpose
-------
Create synthetic "blankie" cells for missing interior RUM grid cells.

Inputs:
  config.generated_outputs.rum_footprints
  config.generated_outputs.packed_series

Output:
  config.generated_outputs.blank_cells

Method
------
1. Use grid_i/grid_j from corrected RUM footprints.
2. Detect missing interior cells using row spans and column spans.
3. Compute blank footprints using the inferred grid model.
4. Interpolate blank vertical time series using robust neighbor median.
5. Write blank cells with vertical_mm arrays.

Blank sigma is deliberately not generated here.
The height texture step later sets blank sigma channel to zero/unused.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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


# =============================================================================
# CONFIG HELPERS
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


# =============================================================================
# GEODESY HELPERS
# =============================================================================

def wgs84_to_ecef(lon_deg: float, lat_deg: float, h_m: float = 0.0) -> np.ndarray:
    a = 6_378_137.0
    f = 1.0 / 298.257223563
    e2 = 2.0 * f - f * f

    lat = math.radians(lon_lat_clip(lat_deg))
    lon = math.radians(lon_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)

    n = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)

    x = (n + h_m) * cos_lat * math.cos(lon)
    y = (n + h_m) * cos_lat * math.sin(lon)
    z = (n * (1.0 - e2) + h_m) * sin_lat
    return np.array([x, y, z], dtype=np.float64)


def lon_lat_clip(lat_deg: float) -> float:
    # Avoid pathological tangent/cos issues near poles for generic projects.
    return max(min(float(lat_deg), 89.999999), -89.999999)


def build_enu_frame(ecef_origin: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x, y, z = ecef_origin
    lon = math.atan2(y, x)
    lat = math.atan2(z, math.sqrt(x * x + y * y))

    east = np.array([-math.sin(lon), math.cos(lon), 0.0], dtype=np.float64)
    north = np.array([
        -math.sin(lat) * math.cos(lon),
        -math.sin(lat) * math.sin(lon),
         math.cos(lat),
    ], dtype=np.float64)
    up = np.array([
        math.cos(lat) * math.cos(lon),
        math.cos(lat) * math.sin(lon),
        math.sin(lat),
    ], dtype=np.float64)

    frame = np.eye(4, dtype=np.float64)
    frame[0:3, 0] = east
    frame[0:3, 1] = north
    frame[0:3, 2] = up
    frame[0:3, 3] = ecef_origin

    return frame, np.linalg.inv(frame)


def enu_to_ecef(east: float, north: float, up: float, frame: np.ndarray) -> np.ndarray:
    h = np.array([east, north, up, 1.0], dtype=np.float64)
    return (frame @ h)[0:3]


def ecef_to_wgs84(ecef: np.ndarray) -> Tuple[float, float, float]:
    x, y, z = ecef

    a = 6_378_137.0
    f = 1.0 / 298.257223563
    e2 = 2.0 * f - f * f

    lon = math.atan2(y, x)
    p = math.sqrt(x * x + y * y)
    lat = math.atan2(z, p * (1.0 - e2))

    for _ in range(5):
        sin_lat = math.sin(lat)
        n = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
        lat = math.atan2(z + e2 * n * sin_lat, p)

    sin_lat = math.sin(lat)
    n = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)

    if abs(math.cos(lat)) > 1e-10:
        h = p / math.cos(lat) - n
    else:
        h = abs(z) / max(abs(sin_lat), 1e-10) - n * (1.0 - e2)

    return math.degrees(lon), math.degrees(lat), float(h)


def rotate_grid_to_enu(u: float, v: float, axis_angle_rad: float) -> Tuple[float, float]:
    c = math.cos(axis_angle_rad)
    s = math.sin(axis_angle_rad)
    return u * c - v * s, u * s + v * c


def compute_corners_from_grid(
    i: int,
    j: int,
    grid_model: Dict[str, Any],
    frame: np.ndarray,
) -> List[List[float]]:
    m = grid_model

    left = float(m["offset_u"]) + (i - 0.5) * float(m["spacing_u"])
    right = float(m["offset_u"]) + (i + 0.5) * float(m["spacing_u"])
    bottom = float(m["offset_v"]) + (j - 0.5) * float(m["spacing_v"])
    top = float(m["offset_v"]) + (j + 0.5) * float(m["spacing_v"])
    axis = float(m["axis_angle_rad"])

    ne_e, ne_n = rotate_grid_to_enu(right, top, axis)
    nw_e, nw_n = rotate_grid_to_enu(left, top, axis)
    sw_e, sw_n = rotate_grid_to_enu(left, bottom, axis)
    se_e, se_n = rotate_grid_to_enu(right, bottom, axis)

    corners = []
    for east, north in [(ne_e, ne_n), (nw_e, nw_n), (sw_e, sw_n), (se_e, se_n)]:
        ecef = enu_to_ecef(east, north, 0.0, frame)
        lon, lat, _ = ecef_to_wgs84(ecef)
        corners.append([lon, lat])

    return corners


# =============================================================================
# SERIES / GRID HELPERS
# =============================================================================

def parse_packed_array(value: Any) -> np.ndarray:
    if isinstance(value, str):
        if not value.strip():
            return np.array([], dtype=np.float32)
        return np.array(value.split(","), dtype=np.float32)
    if isinstance(value, list):
        return np.array(value, dtype=np.float32)
    return np.array([], dtype=np.float32)


def robust_median_stack(arrays: List[np.ndarray], outlier_rejection_mm: float) -> np.ndarray:
    if len(arrays) == 1:
        return arrays[0].copy()

    stack = np.vstack(arrays).astype(np.float32)
    med = np.median(stack, axis=0)

    diff = np.abs(stack - med[None, :])
    mask = diff <= float(outlier_rejection_mm)

    out = np.empty(stack.shape[1], dtype=np.float32)
    for k in range(stack.shape[1]):
        vals = stack[:, k][mask[:, k]]
        if vals.size == 0:
            out[k] = med[k]
        else:
            out[k] = float(np.median(vals))
    return out


def cell_key(i: int, j: int) -> str:
    return f"{i},{j}"


def parse_cell_key(key: str) -> Tuple[int, int]:
    i_s, j_s = key.split(",")
    return int(i_s), int(j_s)


def neighbor_keys(i: int, j: int) -> List[str]:
    return [
        cell_key(i, j + 1),
        cell_key(i + 1, j),
        cell_key(i, j - 1),
        cell_key(i - 1, j),
    ]


def add_blank_candidate(blank_keys: Set[str], real_by_key: Dict[str, str], i: int, j: int) -> None:
    key = cell_key(i, j)
    if key not in real_by_key:
        blank_keys.add(key)


def maybe_apply_block_flattening(
    blank_series: Dict[str, np.ndarray],
    blank_ij: Dict[str, Tuple[int, int]],
    real_by_key: Dict[str, str],
    block_size: int,
    min_blanks: int,
) -> None:
    if block_size <= 0:
        return

    blocks: Dict[str, Dict[str, Any]] = {}

    for key, (i, j) in blank_ij.items():
        bi = math.floor(i / block_size)
        bj = math.floor(j / block_size)
        bkey = f"{bi},{bj}"
        blocks.setdefault(bkey, {"keys": [], "real_count": 0})
        blocks[bkey]["keys"].append(key)

    for key in real_by_key:
        i, j = parse_cell_key(key)
        bi = math.floor(i / block_size)
        bj = math.floor(j / block_size)
        bkey = f"{bi},{bj}"
        if bkey in blocks:
            blocks[bkey]["real_count"] += 1

    for block in blocks.values():
        keys = [k for k in block["keys"] if k in blank_series]
        if len(keys) < min_blanks:
            continue
        if block["real_count"] > 0:
            continue

        stack = np.vstack([blank_series[k] for k in keys])
        flat = np.median(stack, axis=0).astype(np.float32)
        for k in keys:
            blank_series[k] = flat.copy()


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    t_start = time.time()

    cfg = load_config()

    generated = cfg.get("generated_outputs", {})
    blank_cfg = cfg.get("blank_cells", {})

    footprints_path = resolve_project_path(
        generated.get("rum_footprints", "Data/rum_footprints.json")
    )
    packed_path = resolve_project_path(
        generated.get("packed_series", "Data/packed_series.json")
    )
    output_path = resolve_project_path(
        generated.get("blank_cells", "Data/blank_cells.json")
    )

    fill_by_row_spans = bool(blank_cfg.get("fill_by_row_spans", True))
    fill_by_col_spans = bool(blank_cfg.get("fill_by_col_spans", True))
    max_fill_passes = int(blank_cfg.get("max_fill_passes", 25))
    outlier_rejection_mm = float(blank_cfg.get("outlier_rejection_mm", 100.0))
    fallback_to_zero = bool(blank_cfg.get("fallback_to_zero", True))
    enable_block_flattening = bool(blank_cfg.get("enable_block_flattening", False))
    block_flatten_size = int(blank_cfg.get("blank_flatten_block_size", 3))
    block_flatten_min_blanks = int(blank_cfg.get("blank_flatten_min_blanks", 5))

    section("Configuration")
    print(f"  Project root       : {PROJECT_DIR}")
    print(f"  Footprints         : {footprints_path}")
    print(f"  Packed series      : {packed_path}")
    print(f"  Blank output       : {output_path}")
    print(f"  Fill by row spans  : {fill_by_row_spans}")
    print(f"  Fill by col spans  : {fill_by_col_spans}")
    print(f"  Max fill passes    : {max_fill_passes}")
    print(f"  Fallback to zero   : {fallback_to_zero}")

    section("Loading inputs")
    if not footprints_path.exists():
        raise FileNotFoundError(f"Missing footprints: {footprints_path}")
    if not packed_path.exists():
        raise FileNotFoundError(f"Missing packed series: {packed_path}")

    with footprints_path.open("r", encoding="utf-8") as f:
        fp_data = json.load(f)
    with packed_path.open("r", encoding="utf-8") as f:
        packed = json.load(f)

    footprints = fp_data.get("footprints", {})
    grid_model = fp_data.get("metadata", {}).get("grid_model", {})
    epochs = packed.get("epochs", [])
    epoch_decimal_year = packed.get("epoch_decimal_year", [])
    epoch_unix = packed.get("epoch_unix", [])
    real_series_raw = packed.get("series", {})

    if not footprints:
        raise ValueError("Footprints file has no footprints")
    if not grid_model:
        raise ValueError("Footprints metadata has no grid_model")
    if not epochs:
        raise ValueError("Packed series has no epochs")
    if not real_series_raw:
        raise ValueError("Packed series has no series")

    n_epochs = len(epochs)
    ok(f"Loaded real footprints: {len(footprints)}")
    ok(f"Loaded packed series  : {len(real_series_raw)} RUMs × {n_epochs} epochs")

    section("Rebuilding ENU frame from real centers")
    lons = [float(fp["center"][0]) for fp in footprints.values()]
    lats = [float(fp["center"][1]) for fp in footprints.values()]
    centroid_lon = sum(lons) / len(lons)
    centroid_lat = sum(lats) / len(lats)

    origin_ecef = wgs84_to_ecef(centroid_lon, centroid_lat, 0.0)
    frame, _ = build_enu_frame(origin_ecef)

    ok(f"Centroid lon={centroid_lon:.6f}, lat={centroid_lat:.6f}")
    ok(
        f"Grid axis={float(grid_model['axis_angle_deg']):.6f}°, "
        f"spacing=({float(grid_model['spacing_u']):.3f}, {float(grid_model['spacing_v']):.3f}) m"
    )

    section("Detecting missing interior cells")
    real_by_key: Dict[str, str] = {}
    real_ij: Dict[str, Tuple[int, int]] = {}

    malformed_grid = 0
    for rum_id, fp in footprints.items():
        if "grid_i" not in fp or "grid_j" not in fp:
            malformed_grid += 1
            continue

        i = int(fp["grid_i"])
        j = int(fp["grid_j"])
        key = cell_key(i, j)

        real_by_key[key] = str(rum_id)
        real_ij[str(rum_id)] = (i, j)

    if malformed_grid:
        warn(f"Footprints missing grid_i/grid_j: {malformed_grid}")

    blank_keys: Set[str] = set()

    if fill_by_row_spans:
        row_spans: Dict[int, List[int]] = {}
        for i, j in real_ij.values():
            if j not in row_spans:
                row_spans[j] = [i, i]
            else:
                row_spans[j][0] = min(row_spans[j][0], i)
                row_spans[j][1] = max(row_spans[j][1], i)

        for j, (min_i, max_i) in row_spans.items():
            for i in range(min_i, max_i + 1):
                add_blank_candidate(blank_keys, real_by_key, i, j)

    if fill_by_col_spans:
        filled_keys = set(real_by_key.keys()) | blank_keys
        col_spans: Dict[int, List[int]] = {}

        for key in filled_keys:
            i, j = parse_cell_key(key)
            if i not in col_spans:
                col_spans[i] = [j, j]
            else:
                col_spans[i][0] = min(col_spans[i][0], j)
                col_spans[i][1] = max(col_spans[i][1], j)

        for i, (min_j, max_j) in col_spans.items():
            for j in range(min_j, max_j + 1):
                add_blank_candidate(blank_keys, real_by_key, i, j)

    blank_ij: Dict[str, Tuple[int, int]] = {
        key: parse_cell_key(key)
        for key in sorted(blank_keys, key=lambda k: (parse_cell_key(k)[1], parse_cell_key(k)[0]))
    }

    ok(f"Real cells : {len(real_by_key)}")
    ok(f"Blank cells: {len(blank_ij)}")

    section("Loading real vertical time series")
    real_series_by_key: Dict[str, np.ndarray] = {}
    malformed_series = 0

    for key, rum_id in real_by_key.items():
        entry = real_series_raw.get(rum_id)
        if not entry:
            malformed_series += 1
            continue

        vals = parse_packed_array(entry.get("v", ""))
        if len(vals) != n_epochs:
            malformed_series += 1
            continue

        real_series_by_key[key] = vals

    ok(f"Loaded real cell series: {len(real_series_by_key)}")
    if malformed_series:
        warn(f"Missing/malformed real series: {malformed_series}")

    section("Interpolating blank vertical time series")
    t0 = time.time()
    blank_series: Dict[str, np.ndarray] = {}

    for pass_idx in range(max_fill_passes):
        updates: Dict[str, np.ndarray] = {}

        for key, (i, j) in blank_ij.items():
            if key in blank_series:
                continue

            arrays: List[np.ndarray] = []

            for nkey in neighbor_keys(i, j):
                if nkey in real_series_by_key:
                    arrays.append(real_series_by_key[nkey])
                elif nkey in blank_series:
                    arrays.append(blank_series[nkey])

            if not arrays:
                continue

            updates[key] = robust_median_stack(arrays, outlier_rejection_mm)

        if not updates:
            break

        blank_series.update(updates)
        print(
            f"  Pass {pass_idx + 1:02d}: filled {len(updates)} blanks "
            f"({len(blank_series)}/{len(blank_ij)})"
        )

        if len(blank_series) == len(blank_ij):
            break

    not_filled = len(blank_ij) - len(blank_series)

    if not_filled and fallback_to_zero:
        zero = np.zeros(n_epochs, dtype=np.float32)
        for key in blank_ij:
            if key not in blank_series:
                blank_series[key] = zero.copy()
        warn(f"{not_filled} blank cells used zero fallback")
    elif not_filled:
        warn(f"{not_filled} blank cells remain unfilled")

    if enable_block_flattening:
        maybe_apply_block_flattening(
            blank_series=blank_series,
            blank_ij=blank_ij,
            real_by_key=real_by_key,
            block_size=block_flatten_size,
            min_blanks=block_flatten_min_blanks,
        )
        ok("Optional block flattening applied")

    ok(f"Interpolated {len(blank_series)} blank series in {time.time() - t0:.2f}s")

    section("Computing blank footprints")
    blank_cells: Dict[str, Any] = {}
    all_blank_vals: List[np.ndarray] = []

    for key, (i, j) in blank_ij.items():
        blank_id = f"BLANK_{i}_{j}"
        corners = compute_corners_from_grid(i, j, grid_model, frame)

        center_lon = sum(c[0] for c in corners) / 4.0
        center_lat = sum(c[1] for c in corners) / 4.0

        series = blank_series[key]
        all_blank_vals.append(series)

        blank_cells[blank_id] = {
            "kind": "blank",
            "grid_i": i,
            "grid_j": j,
            "center": [center_lon, center_lat],
            "corners": corners,
            "vertical_mm": [round(float(v), 3) for v in series],
            "source": "robust_median_neighbor_interpolation",
        }

    if all_blank_vals:
        stack = np.vstack(all_blank_vals)
        print(f"  Blank vertical range: {float(stack.min()):.3f} → {float(stack.max()):.3f} mm")

    ok(f"Computed blank cells: {len(blank_cells)}")

    section("Writing output")
    output = {
        "metadata": {
            "schema_version": "blank_cells_v1",
            "blank_count": len(blank_cells),
            "real_count": len(real_by_key),
            "epoch_count": n_epochs,
            "method": "row_span_plus_col_span_fill_with_robust_neighbor_median",
            "fill_by_row_spans": fill_by_row_spans,
            "fill_by_col_spans": fill_by_col_spans,
            "max_fill_passes": max_fill_passes,
            "outlier_rejection_mm": outlier_rejection_mm,
            "fallback_to_zero": fallback_to_zero,
            "grid_model": grid_model,
            "source_footprints": str(footprints_path),
            "source_packed_series": str(packed_path),
            "created_by": "10_build_blank_cells.py",
            "created_unix": int(time.time()),
        },
        "epochs": epochs,
        "epoch_decimal_year": epoch_decimal_year,
        "epoch_unix": epoch_unix,
        "blank_cells": blank_cells,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, separators=(",", ":"))

    ok(f"Written: {output_path} ({output_path.stat().st_size / 1024 / 1024:.2f} MB)")

    section("SUMMARY")
    ok("Step 10 complete — blank cells generated")
    ok("Next template step: 11_build_height_texture.py")
    print(f"  Elapsed: {time.time() - t_start:.2f}s")
    print()


if __name__ == "__main__":
    main()
