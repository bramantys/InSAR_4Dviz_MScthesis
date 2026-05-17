#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
11_build_height_texture.py

Generic RUM-based InSAR template step.

Purpose
-------
Pack vertical displacement and vertical sigma into one browser-safe RGB PNG.

Inputs:
  config.generated_outputs.packed_series
  Data/tiles/tile_assignments.json
  config.generated_outputs.blank_cells

Outputs:
  config.generated_outputs.height_texture
  config.generated_outputs.height_meta

Texture channels:
  R = displacement uint16 high byte
  G = displacement uint16 low byte
  B = sigma_mm normalized to 0–255

Rows:
  0 .. n_real_rums-1      = real RUMs, ordered by tile_assignments.json
  n_real_rums .. n_rows-1 = blank cells, ordered by grid_j then grid_i

Notes:
  - Real RUM displacement comes from packed series field "v".
  - Real RUM sigma comes from packed series field "s".
  - Blank displacement comes from blank_cells.vertical_mm.
  - Blank sigma is deliberately encoded as 0/unused.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import png  # type: ignore
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pypng", "--quiet"])
    import png  # type: ignore


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
# HELPERS
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


def parse_float_array(value: Any) -> np.ndarray:
    if isinstance(value, str):
        if not value.strip():
            return np.array([], dtype=np.float32)
        return np.array(value.split(","), dtype=np.float32)
    if isinstance(value, list):
        return np.array(value, dtype=np.float32)
    return np.array([], dtype=np.float32)


def blank_sort_key(item: Tuple[str, Dict[str, Any]]) -> Tuple[int, int, str]:
    blank_id, entry = item
    return (int(entry.get("grid_j", 0)), int(entry.get("grid_i", 0)), blank_id)


def safe_percentile(arr: np.ndarray, p: float) -> float:
    arr = np.asarray(arr, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, p))


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    t_start = time.time()

    cfg = load_config()

    generated = cfg.get("generated_outputs", {})
    paths_cfg = cfg.get("paths", {})
    height_cfg = cfg.get("height_texture", {})

    packed_path = resolve_project_path(
        generated.get("packed_series", "Data/packed_series.json")
    )
    tiles_dir = resolve_project_path(
        paths_cfg.get("tiles_dir", "Data/tiles")
    )
    assignments_path = tiles_dir / "tile_assignments.json"
    blank_cells_path = resolve_project_path(
        generated.get("blank_cells", "Data/blank_cells.json")
    )
    output_png = resolve_project_path(
        generated.get("height_texture", "Data/tiles/height_texture.png")
    )
    output_meta = resolve_project_path(
        generated.get("height_meta", "Data/tiles/height_meta.json")
    )

    v_min = float(height_cfg.get("v_min_mm", -1500.0))
    v_max = float(height_cfg.get("v_max_mm", 1500.0))
    include_blank_cells = bool(height_cfg.get("include_blank_cells", True))

    snr_thin_threshold = float(height_cfg.get("snr_thin_threshold", 3.0))
    snr_thick_threshold = float(height_cfg.get("snr_thick_threshold", 1.0))

    if v_max <= v_min:
        raise ValueError(f"Invalid height texture range: v_min={v_min}, v_max={v_max}")

    section("Configuration")
    print(f"  Project root        : {PROJECT_DIR}")
    print(f"  Packed series       : {packed_path}")
    print(f"  Tile assignments    : {assignments_path}")
    print(f"  Blank cells         : {blank_cells_path}")
    print(f"  Output PNG          : {output_png}")
    print(f"  Output meta         : {output_meta}")
    print(f"  Displacement range  : {v_min} → {v_max} mm")
    print(f"  Include blank cells : {include_blank_cells}")

    section("Loading packed real RUM series")
    if not packed_path.exists():
        raise FileNotFoundError(f"Missing packed series: {packed_path}")
    if not assignments_path.exists():
        raise FileNotFoundError(f"Missing tile assignments: {assignments_path}")

    packed = load_json(packed_path)
    epochs = packed.get("epochs", [])
    rum_series = packed.get("series", {})

    if not epochs:
        raise ValueError("Packed series has no epochs")
    if not rum_series:
        raise ValueError("Packed series has no series")

    n_epochs = len(epochs)
    n_real = len(rum_series)

    ok(f"Loaded real RUMs: {n_real} × {n_epochs} epochs")

    section("Building stable real RUM row order")
    assignments = load_json(assignments_path)
    tiles = assignments.get("tiles", {})

    if not tiles:
        raise ValueError("tile_assignments.json has no tiles")

    rum_order: List[str] = []
    for tile_key in sorted(tiles.keys(), key=lambda k: tuple(int(x) for x in k.split("_"))):
        rum_order.extend(str(rid) for rid in tiles[tile_key].get("rum_ids", []))

    if len(rum_order) != n_real:
        raise ValueError(f"RUM count mismatch: assignments={len(rum_order)}, packed={n_real}")

    missing_in_series = [rid for rid in rum_order if rid not in rum_series]
    if missing_in_series:
        raise ValueError(f"{len(missing_in_series)} assigned RUMs missing in packed series; sample={missing_in_series[:5]}")

    rum_index = {rum_id: idx for idx, rum_id in enumerate(rum_order)}

    ok(f"Real RUM rows: {len(rum_order)}")
    print(f"  First real RUM: {rum_order[0]} row=0")
    print(f"  Last real RUM : {rum_order[-1]} row={len(rum_order)-1}")

    section("Loading blank cells")
    blank_cells: Dict[str, Any] = {}
    blank_order: List[str] = []

    if include_blank_cells and blank_cells_path.exists():
        blank_data = load_json(blank_cells_path)
        blank_cells = blank_data.get("blank_cells", {})
        blank_order = [
            blank_id
            for blank_id, _ in sorted(blank_cells.items(), key=blank_sort_key)
        ]
        ok(f"Blank rows: {len(blank_order)}")
        if blank_order:
            print(f"  First blank: {blank_order[0]}")
            print(f"  Last blank : {blank_order[-1]}")
    elif include_blank_cells:
        warn("Blank cells requested but file missing; continuing with real RUMs only")
    else:
        ok("Blank cells disabled by config")

    n_blank = len(blank_order)
    n_rows = n_real + n_blank

    blank_index = {
        blank_id: n_real + idx
        for idx, blank_id in enumerate(blank_order)
    }

    cell_order = rum_order + blank_order
    cell_index: Dict[str, int] = {}
    cell_kind: Dict[str, str] = {}

    for rum_id, idx in rum_index.items():
        cell_index[rum_id] = idx
        cell_kind[rum_id] = "real"

    for blank_id, idx in blank_index.items():
        cell_index[blank_id] = idx
        cell_kind[blank_id] = "blank"

    ok(f"Total texture rows: {n_rows} ({n_real} real + {n_blank} blank)")

    section(f"Building displacement + sigma matrices ({n_rows} × {n_epochs})")
    t0 = time.time()

    height_matrix = np.zeros((n_rows, n_epochs), dtype=np.float32)
    sigma_matrix = np.zeros((n_rows, n_epochs), dtype=np.float32)

    missing = 0
    missing_sigma = 0

    for rum_id, row_idx in rum_index.items():
        entry = rum_series.get(rum_id)
        if entry is None:
            missing += 1
            continue

        values = parse_float_array(entry.get("v", ""))
        if len(values) != n_epochs:
            warn(f"{rum_id}: expected {n_epochs} vertical values, got {len(values)}")
            missing += 1
            continue

        height_matrix[row_idx, :] = values - values[0]

        sigma_values = parse_float_array(entry.get("s", ""))
        if len(sigma_values) != n_epochs:
            missing_sigma += 1
            sigma_matrix[row_idx, :] = 0.0
        else:
            sigma_matrix[row_idx, :] = sigma_values

    for blank_id, row_idx in blank_index.items():
        entry = blank_cells.get(blank_id)
        if entry is None:
            missing += 1
            continue

        values = parse_float_array(entry.get("vertical_mm", []))
        if len(values) != n_epochs:
            warn(f"{blank_id}: expected {n_epochs} vertical values, got {len(values)}")
            missing += 1
            continue

        height_matrix[row_idx, :] = values - values[0]
        sigma_matrix[row_idx, :] = 0.0

    ok(f"Matrices built in {time.time() - t0:.2f}s")
    if missing:
        warn(f"{missing} rows had missing/malformed displacement data and remain zero")
    if missing_sigma:
        warn(f"{missing_sigma} real RUM rows had missing/malformed sigma and are encoded as 0")

    actual_min = float(height_matrix.min())
    actual_max = float(height_matrix.max())

    print(f"  Actual displacement range: {actual_min:.3f} → {actual_max:.3f} mm")
    print(f"  Normalization range      : {v_min:.3f} → {v_max:.3f} mm")

    if actual_min < v_min:
        warn(f"Some values below v_min ({v_min} mm) will be clamped")
    if actual_max > v_max:
        warn(f"Some values above v_max ({v_max} mm) will be clamped")

    epoch0_min = float(height_matrix[:, 0].min())
    epoch0_max = float(height_matrix[:, 0].max())
    if abs(epoch0_min) < 1e-6 and abs(epoch0_max) < 1e-6:
        ok("Epoch 0 is exactly 0 mm for all texture rows")
    else:
        warn(f"Epoch 0 range is {epoch0_min:.6f} → {epoch0_max:.6f} mm")

    real_sigma = sigma_matrix[:n_real, :].reshape(-1)
    real_sigma = real_sigma[np.isfinite(real_sigma)]
    real_sigma = real_sigma[real_sigma > 0.0]

    if real_sigma.size == 0:
        sigma_min = 0.0
        sigma_max = 1.0
        warn("No positive real sigma values found; using sigma range 0–1 mm")
    else:
        sigma_min = float(real_sigma.min())
        sigma_max = float(real_sigma.max())

    print(f"  Real sigma range: {sigma_min:.4f} → {sigma_max:.4f} mm")
    print(
        f"  Real sigma P50/P90/P95: "
        f"{safe_percentile(real_sigma, 50):.4f} / "
        f"{safe_percentile(real_sigma, 90):.4f} / "
        f"{safe_percentile(real_sigma, 95):.4f} mm"
    )

    real_disp = height_matrix[:n_real, :].reshape(-1)
    real_sig = sigma_matrix[:n_real, :].reshape(-1)
    valid = real_sig > 0.0

    if np.any(valid):
        snr = np.abs(real_disp[valid]) / (real_sig[valid] + 1e-6)
        pct_high = float(np.mean(snr >= snr_thin_threshold) * 100.0)
        pct_mid = float(np.mean((snr >= snr_thick_threshold) & (snr < snr_thin_threshold)) * 100.0)
        pct_low = float(np.mean(snr < snr_thick_threshold) * 100.0)

        print("  SNR class distribution over real RUM observations:")
        print(f"    SNR ≥ {snr_thin_threshold:g}                 no hatch    : {pct_high:5.1f}%")
        print(f"    {snr_thick_threshold:g} ≤ SNR < {snr_thin_threshold:g}           thin hatch  : {pct_mid:5.1f}%")
        print(f"    SNR < {snr_thick_threshold:g}                 thick hatch : {pct_low:5.1f}%")

    section("Normalizing and RGB-packing")

    v_range = v_max - v_min
    normalized_h = np.clip((height_matrix - v_min) / v_range, 0.0, 1.0).astype(np.float32)

    encoded_16bit = np.rint(normalized_h * 65535.0).astype(np.uint16)
    high_byte = (encoded_16bit >> 8).astype(np.uint8)
    low_byte = (encoded_16bit & 255).astype(np.uint8)

    if sigma_max <= sigma_min:
        normalized_s = np.zeros_like(sigma_matrix, dtype=np.float32)
    else:
        normalized_s = np.clip((sigma_matrix - sigma_min) / (sigma_max - sigma_min), 0.0, 1.0).astype(np.float32)

    sigma_byte = np.rint(normalized_s * 255.0).astype(np.uint8)

    if n_blank:
        sigma_byte[n_real:, :] = 0

    encoded_rgb = np.stack([high_byte, low_byte, sigma_byte], axis=2)

    precision_mm = v_range / 65535.0
    sigma_precision_mm = (sigma_max - sigma_min) / 255.0 if sigma_max > sigma_min else 0.0

    ok(f"RG-packed displacement precision: {precision_mm:.6f} mm/step")
    ok(f"B-packed sigma precision        : {sigma_precision_mm:.6f} mm/step")
    ok(f"Matrix shape                    : {encoded_rgb.shape}")

    section("Writing RGB PNG")
    output_png.parent.mkdir(parents=True, exist_ok=True)

    rows = [encoded_rgb[i, :, :].reshape(-1).tolist() for i in range(n_rows)]

    with output_png.open("wb") as f:
        writer = png.Writer(
            width=n_epochs,
            height=n_rows,
            bitdepth=8,
            greyscale=False, # type: ignore
            alpha=False,
        )
        writer.write(f, rows)

    ok(f"Written PNG: {output_png} ({output_png.stat().st_size / 1024:.1f} KB)")
    ok(f"Dimensions : {n_epochs}px wide × {n_rows}px tall × RGB")

    section("Writing metadata")

    meta = {
        # Backward-compatible fields expected by current viewer/scripts.
        "n_rums": n_rows,
        "n_epochs": n_epochs,

        # Clearer explicit fields.
        "schema_version": "height_texture_meta_v1",
        "n_rows": n_rows,
        "n_real_rums": n_real,
        "n_blank_cells": n_blank,

        "v_min": v_min,
        "v_max": v_max,
        "v_range": v_range,
        "precision_mm": precision_mm,

        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "sigma_range": sigma_max - sigma_min,
        "sigma_precision_mm": sigma_precision_mm,

        "snr_hatch_thresholds": {
            "no_hatch_min_snr": snr_thin_threshold,
            "thin_hatch_min_snr": snr_thick_threshold,
            "thick_hatch_below_snr": snr_thick_threshold,
        },

        "encoding": "8-bit RGB PNG: displacement uint16 packed in R/G, sigma_mm normalized in B",
        "texture_packing": "uint16_rg8_plus_sigma_b8",
        "channels": {
            "R": "displacement high byte",
            "G": "displacement low byte",
            "B": "sigma_mm normalized to sigma_min/sigma_max",
        },
        "displacement_denormalization": "disp_mm = ((R*256 + G) / 65535.0) * v_range + v_min",
        "sigma_denormalization": "sigma_mm = (B / 255.0) * sigma_range + sigma_min",
        "uncertainty_visualization": "viewer computes SNR = abs(disp_mm) / sigma_mm; hatch only on real RUM caps",

        "cell_order": cell_order,
        "cell_index": cell_index,
        "cell_kind": cell_kind,

        "rum_order": rum_order,
        "rum_index": rum_index,

        "blank_order": blank_order,
        "blank_index": blank_index,

        "epochs": epochs,

        "source_packed_series": str(packed_path),
        "source_tile_assignments": str(assignments_path),
        "source_blank_cells": str(blank_cells_path) if blank_cells_path.exists() else None,
        "height_texture": str(output_png),
        "created_by": "11_build_height_texture.py",
        "created_unix": int(time.time()),
    }

    output_meta.parent.mkdir(parents=True, exist_ok=True)
    with output_meta.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, separators=(",", ":"))

    ok(f"Written metadata: {output_meta} ({output_meta.stat().st_size / 1024:.1f} KB)")

    section("Round-trip verification")

    sample_rows: List[Tuple[str, int, bool]] = []
    for rid in rum_order[:2]:
        sample_rows.append((rid, rum_index[rid], True))
    for bid in blank_order[:2]:
        sample_rows.append((bid, blank_index[bid], False))

    for cell_id, row_idx, has_sigma in sample_rows:
        col_idx = n_epochs // 2

        original = float(height_matrix[row_idx, col_idx])
        hi = int(high_byte[row_idx, col_idx])
        lo = int(low_byte[row_idx, col_idx])
        encoded = hi * 256 + lo
        decoded = (encoded / 65535.0) * v_range + v_min
        err = abs(decoded - original)

        if has_sigma:
            sigma_original = float(sigma_matrix[row_idx, col_idx])
            b = int(sigma_byte[row_idx, col_idx])
            sigma_decoded = (b / 255.0) * (sigma_max - sigma_min) + sigma_min
            sigma_msg = f"sigma={sigma_original:.4f}→{sigma_decoded:.4f} mm B={b}"
        else:
            sigma_msg = "sigma=blank/unused"

        status = "OK" if err < precision_mm * 2.0 else "WARN"
        print(
            f"  [{status}] {cell_id:<24s} row={row_idx:5d} "
            f"disp={original:9.4f}→{decoded:9.4f} mm err={err:.5f} "
            f"R={hi} G={lo}; {sigma_msg}"
        )

    section("SUMMARY")
    ok(f"Step 11 complete — height texture generated: {n_epochs} × {n_rows} pixels")
    ok(f"Rows: {n_real} real RUMs + {n_blank} blank cells")
    ok("Next template step: 12_build_real_caps_b3dm.py")
    print(f"  Elapsed: {time.time() - t_start:.2f}s")
    print()


if __name__ == "__main__":
    main()
