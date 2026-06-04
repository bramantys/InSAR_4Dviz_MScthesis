#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08_export_epoch_axis.py

InSAR4D RUM Viewer pipeline step 08.

Purpose
-------
Export the time axis used by the viewer slider/animation.

Input
-----
  generated_outputs.packed_series
    _internal/data_pipeline/packed_series.json

Output
------
  generated_outputs.epoch_axis
    _internal/data_pipeline/tiles/epoch_axis.json

Why this step exists
--------------------
The viewer needs the epoch list immediately for the slider, labels, and
animation controls. It should not need to parse the full packed_series.json
just to initialize time controls.
"""

from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pipeline_config import load_resolved_config, resolve_path


# =============================================================================
# ADVANCED SETTINGS
# =============================================================================

DEFAULT_EPOCH_INDEX_MODE = "last"  # first | last | middle
ROUND_DECIMAL_YEAR_DIGITS = 8


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
# HELPERS
# =============================================================================

def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))


def parse_date(value: str) -> dt.date:
    return dt.date.fromisoformat(str(value)[:10])


def decimal_year(d: dt.date) -> float:
    year_start = dt.date(d.year, 1, 1)
    next_year_start = dt.date(d.year + 1, 1, 1)
    return d.year + (d - year_start).days / (next_year_start - year_start).days


def unix_time(d: dt.date) -> float:
    return dt.datetime(d.year, d.month, d.day, tzinfo=dt.timezone.utc).timestamp()


def default_epoch_index(n: int, mode: str) -> int:
    if n <= 0:
        raise ValueError("epoch count must be positive")

    mode = str(mode).lower()
    if mode == "first":
        return 0
    if mode == "middle":
        return n // 2
    if mode == "last":
        return n - 1

    warn(f"Unknown DEFAULT_EPOCH_INDEX_MODE={mode!r}; using last")
    return n - 1


def validate_epochs(epochs: List[str]) -> List[dt.date]:
    if not isinstance(epochs, list) or not epochs:
        raise ValueError("packed_series.json has no epochs")

    dates = [parse_date(e) for e in epochs]

    for prev, cur in zip(dates, dates[1:]):
        if cur <= prev:
            raise ValueError(f"Epoch axis is not strictly increasing: {prev} then {cur}")

    return dates


def infer_interval_days(dates: List[dt.date]) -> Dict[str, Any]:
    if len(dates) <= 1:
        return {
            "interval_days_mode": None,
            "interval_days_min": None,
            "interval_days_max": None,
            "is_regular": True,
        }

    intervals = [(b - a).days for a, b in zip(dates, dates[1:])]
    mn = min(intervals)
    mx = max(intervals)

    # Last interval may be shorter/longer because the exact end date was appended.
    # So report both strict and main-mode regularity.
    counts: Dict[int, int] = {}
    for value in intervals:
        counts[value] = counts.get(value, 0) + 1

    mode_interval = max(counts, key=counts.get)
    irregular_count = sum(1 for value in intervals if value != mode_interval)

    return {
        "interval_days_mode": mode_interval,
        "interval_days_min": mn,
        "interval_days_max": mx,
        "is_regular": mn == mx,
        "intervals_different_from_mode": irregular_count,
    }


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    t_start = time.time()

    cfg = load_resolved_config(__file__)
    project_root = Path(cfg["_resolved"]["project_root"])

    generated = cfg["generated_outputs"]

    packed_path = resolve_path(project_root, generated["packed_series"])
    output_path = resolve_path(project_root, generated["epoch_axis"])

    section("Configuration")
    print(f"  Project root       : {project_root}")
    print(f"  Packed input       : {packed_path}")
    print(f"  Epoch axis output  : {output_path}")

    section("Loading packed series")
    packed = load_json(packed_path)

    epochs = packed.get("epochs")
    epoch_decimal_year = packed.get("epoch_decimal_year")
    epoch_unix = packed.get("epoch_unix")
    metadata = packed.get("metadata") or {}

    dates = validate_epochs(epochs)

    if not epoch_decimal_year:
        epoch_decimal_year = [round(decimal_year(d), ROUND_DECIMAL_YEAR_DIGITS) for d in dates]
        warn("epoch_decimal_year missing in packed series; recalculated")
    elif len(epoch_decimal_year) != len(epochs):
        raise ValueError("epoch_decimal_year length does not match epochs")

    if not epoch_unix:
        epoch_unix = [unix_time(d) for d in dates]
        warn("epoch_unix missing in packed series; recalculated")
    elif len(epoch_unix) != len(epochs):
        raise ValueError("epoch_unix length does not match epochs")

    epoch_count = len(epochs)
    idx_default = default_epoch_index(epoch_count, DEFAULT_EPOCH_INDEX_MODE)
    interval_summary = infer_interval_days(dates)

    ok(f"Loaded epoch axis with {epoch_count} epochs")
    print(f"  First epoch        : {epochs[0]}")
    print(f"  Last epoch         : {epochs[-1]}")
    print(f"  Default index      : {idx_default}")
    print(f"  Interval mode      : {interval_summary['interval_days_mode']} days")

    section("Writing epoch axis")
    payload = {
        "metadata": {
            "schema": "epoch_axis_v1",
            "source_packed_series": generated["packed_series"],
            "epoch_count": epoch_count,
            "first_epoch": epochs[0],
            "last_epoch": epochs[-1],
            "default_epoch_index": idx_default,
            "default_epoch": epochs[idx_default],
            "default_epoch_index_mode": DEFAULT_EPOCH_INDEX_MODE,
            "interval_summary": interval_summary,
            "rum_count": metadata.get("rum_count"),
            "vertical_unit": metadata.get("vertical_unit", "mm"),
            "sigma_unit": metadata.get("sigma_unit", "mm"),
        },
        "epochs": epochs,
        "epoch_decimal_year": epoch_decimal_year,
        "epoch_unix": epoch_unix,
        "labels": {
            "short": [e[:7] for e in epochs],
            "date": epochs,
        },
    }

    write_json(output_path, payload)

    elapsed = time.time() - t_start

    ok(f"Wrote epoch axis: {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")

    section("Summary")
    ok(f"Step 08 complete in {elapsed:.2f} s")
    print(f"  Epoch count            : {epoch_count}")
    print(f"  First / last            : {epochs[0]} / {epochs[-1]}")
    print(f"  Default epoch           : {epochs[idx_default]}")
    print(f"  Interval min/mode/max   : {interval_summary['interval_days_min']} / {interval_summary['interval_days_mode']} / {interval_summary['interval_days_max']} days")


if __name__ == "__main__":
    main()
