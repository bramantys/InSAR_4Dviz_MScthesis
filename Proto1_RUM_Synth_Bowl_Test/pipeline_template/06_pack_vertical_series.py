#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_pack_vertical_series.py

Generic RUM-based InSAR template step.

Purpose
-------
Pack prepared vertical displacement and sigma time series into compact strings
used by later 3D tile / texture generation.

Input:
  config.prepared_inputs.vertical_epoch_json

Output:
  config.generated_outputs.packed_series

Expected input contract:
  epochs
  epoch_decimal_year
  epoch_unix
  series[rum_id].vertical_mm
  series[rum_id].sigma_mm
  series[rum_id].source_up_mm_yr

Output contract:
  epochs
  epoch_decimal_year
  epoch_unix
  series[rum_id].v   comma-separated vertical_mm
  series[rum_id].s   comma-separated sigma_mm
  series[rum_id].up  source_up_mm_yr
"""

from __future__ import annotations

import datetime as dt
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


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


def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


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


def safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        if value is None:
            return fallback
        out = float(value)
        if not math.isfinite(out):
            return fallback
        return out
    except (TypeError, ValueError):
        return fallback


def parse_float_array(value: Any) -> List[float]:
    if isinstance(value, str):
        if not value.strip():
            return []
        return [float(x) for x in value.split(",")]
    if isinstance(value, list):
        return [float(x) for x in value]
    return []


def pack_array(values: List[float], decimals: int) -> str:
    fmt = f"{{:.{decimals}f}}"
    return ",".join(fmt.format(float(v)) for v in values)


def decimal_year_from_date_string(date_string: str) -> float:
    d = dt.date.fromisoformat(date_string[:10])
    year_start = dt.date(d.year, 1, 1)
    next_year_start = dt.date(d.year + 1, 1, 1)
    return d.year + (d - year_start).days / (next_year_start - year_start).days


def unix_from_date_string(date_string: str) -> float:
    d = dt.date.fromisoformat(date_string[:10])
    return dt.datetime(d.year, d.month, d.day, tzinfo=dt.timezone.utc).timestamp()


def decimal_year_to_unix(dy: float) -> float:
    year = int(dy)
    days_in_year = 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365
    frac = dy - year
    return dt.datetime(year, 1, 1, tzinfo=dt.timezone.utc).timestamp() + frac * days_in_year * 86400.0


def get_epoch_series(epoch_data: Dict[str, Any]) -> Dict[str, Any]:
    series = epoch_data.get("series", {})
    if isinstance(series, dict):
        return series

    ignore = {"metadata", "epochs", "epoch_decimal_year", "epoch_unix"}
    return {
        k: v
        for k, v in epoch_data.items()
        if k not in ignore and isinstance(v, dict)
    }


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    t_start = time.time()

    cfg = load_config()
    prepared = cfg.get("prepared_inputs", {})
    generated = cfg.get("generated_outputs", {})
    packing_cfg = cfg.get("series_packing", {})

    epoch_path = resolve_project_path(
        prepared.get("vertical_epoch_json", "Data/vertical_epochs.json")
    )
    output_path = resolve_project_path(
        generated.get("packed_series", "Data/packed_series.json")
    )

    vertical_decimals = int(packing_cfg.get("vertical_decimals", 2))
    sigma_decimals = int(packing_cfg.get("sigma_decimals", 3))
    epoch_decimal_decimals = int(packing_cfg.get("epoch_decimal_decimals", 8))

    section("Configuration")
    print(f"  Project root       : {PROJECT_DIR}")
    print(f"  Epoch JSON         : {epoch_path}")
    print(f"  Packed output      : {output_path}")
    print(f"  Vertical decimals  : {vertical_decimals}")
    print(f"  Sigma decimals     : {sigma_decimals}")

    section("Loading epoch JSON")
    if not epoch_path.exists():
        raise FileNotFoundError(f"Missing epoch JSON: {epoch_path}")

    with epoch_path.open("r", encoding="utf-8") as f:
        epoch_data = json.load(f)

    epochs = epoch_data.get("epochs", [])
    epoch_decimal_year = epoch_data.get("epoch_decimal_year", [])
    epoch_unix = epoch_data.get("epoch_unix", [])
    rum_series = get_epoch_series(epoch_data)

    if not epochs:
        raise ValueError("Epoch JSON has no epochs")
    if not rum_series:
        raise ValueError("Epoch JSON has no RUM series")

    n_epochs = len(epochs)
    n_rums = len(rum_series)

    ok(f"Loaded {n_rums} RUMs × {n_epochs} epochs")

    section("Checking / building epoch time axis")
    if not epoch_decimal_year or len(epoch_decimal_year) != n_epochs:
        if all(isinstance(e, str) for e in epochs):
            epoch_decimal_year = [
                round(decimal_year_from_date_string(str(e)), epoch_decimal_decimals)
                for e in epochs
            ]
            warn("epoch_decimal_year missing/wrong length; rebuilt from epoch strings")
        else:
            raise ValueError("epoch_decimal_year missing and epochs are not ISO date strings")
    else:
        epoch_decimal_year = [round(float(v), epoch_decimal_decimals) for v in epoch_decimal_year]
        ok("Using epoch_decimal_year from source")

    if not epoch_unix or len(epoch_unix) != n_epochs:
        if all(isinstance(e, str) for e in epochs):
            epoch_unix = [unix_from_date_string(str(e)) for e in epochs]
            warn("epoch_unix missing/wrong length; rebuilt from epoch strings")
        else:
            epoch_unix = [decimal_year_to_unix(float(dy)) for dy in epoch_decimal_year]
            warn("epoch_unix missing/wrong length; rebuilt from decimal years")
    else:
        epoch_unix = [float(v) for v in epoch_unix]
        ok("Using epoch_unix from source")

    print(f"  First epoch: {epochs[0]}  dec={epoch_decimal_year[0]:.8f}  unix={epoch_unix[0]:.0f}")
    print(f"  Last epoch : {epochs[-1]}  dec={epoch_decimal_year[-1]:.8f}  unix={epoch_unix[-1]:.0f}")

    if len(epoch_unix) > 1:
        first_spacings = [epoch_unix[i + 1] - epoch_unix[i] for i in range(min(5, n_epochs - 1))]
        avg_spacing_days = sum(first_spacings) / len(first_spacings) / 86400.0
        print(f"  First spacings average: {avg_spacing_days:.3f} days")

    section("Packing vertical and sigma time series")
    t0 = time.time()

    packed: Dict[str, Any] = {}
    bad_length = 0
    bad_numeric = 0

    for rum_id, entry in rum_series.items():
        v = parse_float_array(entry.get("vertical_mm", []))
        s = parse_float_array(entry.get("sigma_mm", []))

        if len(v) != n_epochs or len(s) != n_epochs:
            bad_length += 1
            if bad_length <= 5:
                warn(f"{rum_id}: expected {n_epochs}, got vertical={len(v)}, sigma={len(s)}")
            continue

        if not all(math.isfinite(float(x)) for x in v) or not all(math.isfinite(float(x)) for x in s):
            bad_numeric += 1
            if bad_numeric <= 5:
                warn(f"{rum_id}: non-finite vertical/sigma value")
            continue

        packed[str(rum_id)] = {
            "v": pack_array(v, vertical_decimals),
            "s": pack_array(s, sigma_decimals),
            "up": round(safe_float(entry.get("source_up_mm_yr"), 0.0), 4),
        }

        # Preserve optional provenance if present and cheap.
        if "source_sigma_up_mm_yr" in entry:
            packed[str(rum_id)]["source_sigma_up_mm_yr"] = round(
                safe_float(entry.get("source_sigma_up_mm_yr"), 0.0), 4
            )

    ok(f"Packed {len(packed)} RUMs in {time.time() - t0:.2f}s")
    if bad_length:
        warn(f"Skipped due to wrong array length: {bad_length}")
    if bad_numeric:
        warn(f"Skipped due to non-finite values: {bad_numeric}")

    if not packed:
        fail("No valid RUM series packed")
        sys.exit(1)

    section("Round-trip verification")
    sample_ids = sorted(packed.keys())[:5]

    max_rounding_error = 0.0
    for rid in sample_ids:
        original_v = parse_float_array(rum_series[rid].get("vertical_mm", []))
        unpacked_v = parse_float_array(packed[rid]["v"])

        if len(unpacked_v) != n_epochs:
            fail(f"{rid}: unpacked length {len(unpacked_v)} != {n_epochs}")
            sys.exit(1)

        err = max(abs(a - b) for a, b in zip(original_v, unpacked_v))
        max_rounding_error = max(max_rounding_error, err)

        print(f"  {rid:<24s} max vertical rounding error = {err:.6f} mm")

    tolerance = 10 ** (-vertical_decimals) * 0.6
    if max_rounding_error <= tolerance:
        ok(f"Round-trip error within tolerance ({tolerance:.6f} mm)")
    else:
        warn(f"Round-trip error {max_rounding_error:.6f} exceeds tolerance {tolerance:.6f}")

    section("Size estimate")
    sample_id = sorted(packed.keys())[0]
    v_chars = len(packed[sample_id]["v"])
    s_chars = len(packed[sample_id]["s"])
    est_mb = (v_chars + s_chars) * len(packed) / 1e6

    print(f"  Sample vertical string length: {v_chars} chars")
    print(f"  Sample sigma string length   : {s_chars} chars")
    print(f"  Estimated packed text size   : {est_mb:.2f} MB")

    section("Writing output")
    output = {
        "metadata": {
            "schema_version": "packed_series_v1",
            "rum_count": len(packed),
            "epoch_count": n_epochs,
            "vertical_decimals": vertical_decimals,
            "sigma_decimals": sigma_decimals,
            "pack_format": "comma-separated floats, one value per epoch",
            "fields": {
                "v": "vertical_mm",
                "s": "sigma_mm",
                "up": "source_up_mm_yr",
            },
            "source_epoch_json": str(epoch_path),
            "created_by": "06_pack_vertical_series.py",
            "created_unix": int(time.time()),
        },
        "epochs": epochs,
        "epoch_decimal_year": epoch_decimal_year,
        "epoch_unix": epoch_unix,
        "series": packed,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, separators=(",", ":"))

    ok(f"Written: {output_path} ({output_path.stat().st_size / 1024 / 1024:.2f} MB)")

    section("SUMMARY")
    ok("Step 06 complete — vertical/sigma series packed")
    ok("Next template step: 08_build_tile_index.py")
    print(f"  Elapsed: {time.time() - t_start:.2f}s")
    print()


if __name__ == "__main__":
    main()
