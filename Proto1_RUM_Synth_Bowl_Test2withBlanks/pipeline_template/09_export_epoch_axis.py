#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
09_export_epoch_axis.py

Generic RUM-based InSAR template step.

Purpose
-------
Extract the epoch/time axis from the packed vertical series and write a small
browser-readable JSON file for the viewer.

Input:
  config.generated_outputs.packed_series

Output:
  config.generated_outputs.epoch_axis

Expected output:
  Data/tiles/epoch_axis.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict


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


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    t_start = time.time()

    cfg = load_config()
    generated = cfg.get("generated_outputs", {})

    packed_path = resolve_project_path(
        generated.get("packed_series", "Data/packed_series.json")
    )
    output_path = resolve_project_path(
        generated.get("epoch_axis", "Data/tiles/epoch_axis.json")
    )

    section("Configuration")
    print(f"  Project root  : {PROJECT_DIR}")
    print(f"  Packed series : {packed_path}")
    print(f"  Output axis   : {output_path}")

    section("Loading packed series")
    if not packed_path.exists():
        raise FileNotFoundError(f"Missing packed series: {packed_path}")

    with packed_path.open("r", encoding="utf-8") as f:
        packed = json.load(f)

    epochs = packed.get("epochs", [])
    epoch_decimal_year = packed.get("epoch_decimal_year", [])
    epoch_unix = packed.get("epoch_unix", [])

    if not epochs:
        raise ValueError("Packed series has no 'epochs'")
    if len(epoch_decimal_year) != len(epochs):
        raise ValueError(
            f"epoch_decimal_year length {len(epoch_decimal_year)} != epochs length {len(epochs)}"
        )
    if len(epoch_unix) != len(epochs):
        raise ValueError(
            f"epoch_unix length {len(epoch_unix)} != epochs length {len(epochs)}"
        )

    ok(f"Loaded epoch axis: {len(epochs)} epochs")
    print(f"  First epoch: {epochs[0]}")
    print(f"  Last epoch : {epochs[-1]}")

    section("Writing epoch axis")
    output = {
        "schema_version": "epoch_axis_v1",
        "epochs": epochs,
        "epoch_decimal_year": epoch_decimal_year,
        "epoch_unix": epoch_unix,
        "source_packed_series": str(packed_path),
        "created_by": "09_export_epoch_axis.py",
        "created_unix": int(time.time()),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, separators=(",", ":"))

    ok(f"Written: {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")

    section("SUMMARY")
    ok("Step 09 complete — viewer epoch axis exported")
    ok("Next template step: 10_build_blank_cells.py")
    print(f"  Elapsed: {time.time() - t_start:.2f}s")
    print()


if __name__ == "__main__":
    main()
