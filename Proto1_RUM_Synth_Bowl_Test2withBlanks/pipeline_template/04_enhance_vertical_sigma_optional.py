#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_enhance_vertical_sigma_optional.py

Generic RUM-based InSAR template step.

Purpose
-------
Optionally replace/enhance sigma_mm in a prepared vertical epoch JSON.

Input:
  - Base vertical epoch JSON:
      config.prepared_inputs.vertical_epoch_json_without_enhanced_sigma
  - RUM footprints:
      config.generated_outputs.rum_footprints

Output:
  - Enhanced vertical epoch JSON:
      config.prepared_inputs.vertical_epoch_json

Important
---------
This script does NOT modify vertical_mm.
It only replaces sigma_mm with a visually useful synthetic uncertainty field.

Why this exists
---------------
Some velocity-only products have sigma values that are too uniform or too
small to communicate uncertainty clearly in a thesis/demo viewer. This step
creates spatially and temporally structured sigma for visualization testing.

For scientific reporting, keep the output metadata note: sigma is synthetic.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def safe_float(value: Any, fallback: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return fallback
        out = float(value)
        if not math.isfinite(out):
            return fallback
        return out
    except (TypeError, ValueError):
        return fallback


def decimal_year_from_epoch(epoch_value: Any, fallback_index: int, n_epochs: int) -> float:
    """
    Robust fallback for seasonal modulation if epoch_decimal_year is missing.
    """
    if isinstance(epoch_value, (int, float)):
        return float(epoch_value)

    if isinstance(epoch_value, str):
        try:
            d = dt.date.fromisoformat(epoch_value[:10])
            y0 = dt.date(d.year, 1, 1)
            y1 = dt.date(d.year + 1, 1, 1)
            return d.year + (d - y0).days / (y1 - y0).days
        except Exception:
            pass

    return 2000.0 + fallback_index / max(n_epochs - 1, 1)


# =============================================================================
# SIGMA SYNTHESIS
# =============================================================================

def cell_key(i: int, j: int) -> str:
    return f"{i},{j}"


def classify_rums(footprints: Dict[str, Any],
                  edge_neighbour_threshold: int,
                  isolated_neighbour_threshold: int) -> Dict[str, Dict[str, Any]]:
    """
    Classify each RUM by 8-neighbour grid topology.
    """
    occupied = {}

    for rum_id, fp in footprints.items():
        gi = fp.get("grid_i")
        gj = fp.get("grid_j")
        if gi is not None and gj is not None:
            occupied[(int(gi), int(gj))] = rum_id

    classifications: Dict[str, Dict[str, Any]] = {}

    for rum_id, fp in footprints.items():
        gi = fp.get("grid_i")
        gj = fp.get("grid_j")

        if gi is None or gj is None:
            classifications[rum_id] = {"class": "interior", "n_neighbours": 8}
            continue

        gi = int(gi)
        gj = int(gj)

        n_present = 0
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                if (gi + di, gj + dj) in occupied:
                    n_present += 1

        if n_present <= isolated_neighbour_threshold:
            cls = "isolated"
        elif n_present <= edge_neighbour_threshold:
            cls = "edge"
        else:
            cls = "interior"

        classifications[rum_id] = {
            "class": cls,
            "n_neighbours": n_present,
        }

    return classifications


def base_sigma_for_class(cls: str,
                         sigma_interior: float,
                         sigma_edge: float,
                         sigma_isolated: float) -> float:
    if cls == "isolated":
        return sigma_isolated
    if cls == "edge":
        return sigma_edge
    return sigma_interior


def time_factor(epoch_idx: int, n_epochs: int, sigma_time_growth: float) -> float:
    t = epoch_idx / max(n_epochs - 1, 1)
    return 1.0 + (sigma_time_growth - 1.0) * math.sqrt(max(0.0, t))


def seasonal_factor(decimal_year: float,
                    base_sigma: float,
                    seasonal_amplitude: float,
                    peak_month: int) -> float:
    """
    Generic seasonal modulation.

    peak_month = month where uncertainty is highest.
    Set seasonal_amplitude = 0 to disable this behavior.
    """
    peak_phase = (peak_month - 1) / 12.0
    frac = decimal_year % 1.0
    seasonal = math.cos(2.0 * math.pi * (frac - peak_phase))
    return seasonal_amplitude * base_sigma * seasonal


def spike_contribution(epoch_idx: int,
                       spike_epochs: List[int],
                       spike_magnitude: float,
                       spike_decay_epochs: int) -> float:
    total = 0.0
    if spike_magnitude <= 0 or spike_decay_epochs <= 0:
        return total

    for sp in spike_epochs:
        dist = abs(epoch_idx - sp)
        if dist <= spike_decay_epochs * 2:
            total += spike_magnitude * math.exp(
                -0.5 * (dist / max(spike_decay_epochs / 2.0, 1e-9)) ** 2
            )
    return total


def choose_spike_epochs(n_epochs: int, n_spikes: int, rng: random.Random) -> List[int]:
    if n_spikes <= 0 or n_epochs < 10:
        return []

    if n_spikes == 1:
        return [rng.randint(1, n_epochs - 2)]

    spike_epochs = []
    for k in range(n_spikes):
        start = int(k * n_epochs / n_spikes)
        end = int((k + 1) * n_epochs / n_spikes) - 1
        start = max(1, start)
        end = min(n_epochs - 2, end)
        if start <= end:
            spike_epochs.append(rng.randint(start, end))

    return sorted(set(spike_epochs))


def percentile(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), p))


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    t_start = time.time()

    cfg = load_config()

    prepared = cfg.get("prepared_inputs", {})
    generated = cfg.get("generated_outputs", {})
    project_cfg = cfg.get("project", {})
    sigma_cfg = cfg.get("sigma_enhancement", {})

    input_epoch_path = resolve_project_path(
        prepared.get("vertical_epoch_json_without_enhanced_sigma", "Data/vertical_epochs_linear_14d.json")
    )
    footprints_path = resolve_project_path(
        generated.get("rum_footprints", "Data/rum_footprints.json")
    )
    output_epoch_path = resolve_project_path(
        prepared.get("vertical_epoch_json", "Data/vertical_epochs_linear_14d_synthsigma.json")
    )

    # Defaults are intentionally close to the previous Jakarta demo behavior.
    random_seed = int(sigma_cfg.get("random_seed", 42))

    sigma_interior = float(sigma_cfg.get("sigma_interior_mm", 0.25))
    sigma_edge = float(sigma_cfg.get("sigma_edge_mm", 3.5))
    sigma_isolated = float(sigma_cfg.get("sigma_isolated_mm", 7.0))

    sigma_time_growth = float(sigma_cfg.get("sigma_time_growth", 2.0))
    seasonal_amplitude = float(sigma_cfg.get("seasonal_amplitude", 0.4))
    seasonal_peak_month = int(sigma_cfg.get("seasonal_peak_month", 1))

    n_spike_epochs = int(sigma_cfg.get("n_spike_epochs", 3))
    spike_magnitude = float(sigma_cfg.get("spike_magnitude_mm", 4.0))
    spike_decay_epochs = int(sigma_cfg.get("spike_decay_epochs", 4))

    edge_neighbour_threshold = int(sigma_cfg.get("edge_neighbour_threshold", 5))
    isolated_neighbour_threshold = int(sigma_cfg.get("isolated_neighbour_threshold", 2))

    per_rum_jitter_min = float(sigma_cfg.get("per_rum_jitter_min", 0.85))
    per_rum_jitter_max = float(sigma_cfg.get("per_rum_jitter_max", 1.15))
    epoch_noise_fraction = float(sigma_cfg.get("epoch_noise_fraction_of_base", 0.04))
    sigma_floor_mm = float(sigma_cfg.get("sigma_floor_mm", 0.01))
    round_digits = int(sigma_cfg.get("round_digits", 4))

    rng = random.Random(random_seed)
    np.random.seed(random_seed)

    section("Configuration")
    print(f"  Project root         : {PROJECT_DIR}")
    print(f"  Input epoch JSON     : {input_epoch_path}")
    print(f"  Footprints           : {footprints_path}")
    print(f"  Output epoch JSON    : {output_epoch_path}")
    print(f"  Dataset              : {project_cfg.get('dataset_id', 'rum_project')}")
    print(f"  Base sigma interior  : {sigma_interior} mm")
    print(f"  Base sigma edge      : {sigma_edge} mm")
    print(f"  Base sigma isolated  : {sigma_isolated} mm")
    print(f"  Time growth          : {sigma_time_growth}")
    print(f"  Seasonal amplitude   : {seasonal_amplitude}")
    print(f"  Seasonal peak month  : {seasonal_peak_month}")
    print(f"  Spike epochs count   : {n_spike_epochs}")

    section("Loading inputs")
    if not input_epoch_path.exists():
        raise FileNotFoundError(f"Missing base epoch JSON: {input_epoch_path}")
    if not footprints_path.exists():
        raise FileNotFoundError(f"Missing footprints JSON: {footprints_path}")

    with input_epoch_path.open("r", encoding="utf-8") as f:
        epoch_data = json.load(f)
    with footprints_path.open("r", encoding="utf-8") as f:
        fp_data = json.load(f)

    rum_series = epoch_data.get("series", {})
    epochs = epoch_data.get("epochs", [])
    dec_years = epoch_data.get("epoch_decimal_year", [])

    if not isinstance(rum_series, dict) or not rum_series:
        raise ValueError("Input epoch JSON has no non-empty dict 'series'")
    if not epochs:
        raise ValueError("Input epoch JSON has no epochs")

    footprints = fp_data.get("footprints", {})
    if not isinstance(footprints, dict) or not footprints:
        raise ValueError("Footprints JSON has no non-empty dict 'footprints'")

    n_epochs = len(epochs)
    ok(f"Loaded epoch data: {len(rum_series)} RUMs × {n_epochs} epochs")
    ok(f"Loaded footprints: {len(footprints)} RUMs/cells")

    if not dec_years or len(dec_years) != n_epochs:
        dec_years = [
            decimal_year_from_epoch(epochs[i], i, n_epochs)
            for i in range(n_epochs)
        ]
        warn("epoch_decimal_year missing or wrong length; reconstructed from epochs/fallback")

    section("Classifying RUMs by grid topology")
    classifications = classify_rums(
        footprints=footprints,
        edge_neighbour_threshold=edge_neighbour_threshold,
        isolated_neighbour_threshold=isolated_neighbour_threshold,
    )

    counts = {"interior": 0, "edge": 0, "isolated": 0}
    for rum_id in rum_series:
        cls = classifications.get(rum_id, {"class": "interior"})["class"]
        counts[cls] = counts.get(cls, 0) + 1

    total = max(len(rum_series), 1)
    ok(f"Interior : {counts.get('interior', 0):5d} RUMs ({100 * counts.get('interior', 0) / total:5.1f}%)")
    ok(f"Edge     : {counts.get('edge', 0):5d} RUMs ({100 * counts.get('edge', 0) / total:5.1f}%)")
    ok(f"Isolated : {counts.get('isolated', 0):5d} RUMs ({100 * counts.get('isolated', 0) / total:5.1f}%)")

    section("Selecting global uncertainty spike epochs")
    spike_epochs = choose_spike_epochs(n_epochs, n_spike_epochs, rng)
    if spike_epochs:
        for sp in spike_epochs:
            print(f"  Spike at epoch {sp:3d}: {epochs[sp]}")
    else:
        ok("No spike epochs requested")

    section("Synthesizing sigma_mm")
    t0 = time.time()

    new_series: Dict[str, Any] = {}
    sigma_samples: List[float] = []
    snr_bins = {"<0.5": 0, "0.5-1": 0, "1-3": 0, ">3": 0}

    skipped_bad_series = 0

    for rum_id, entry in rum_series.items():
        vertical_mm = entry.get("vertical_mm", [])
        if not isinstance(vertical_mm, list) or len(vertical_mm) != n_epochs:
            skipped_bad_series += 1
            continue

        cls = classifications.get(rum_id, {"class": "interior"})["class"]
        base = base_sigma_for_class(cls, sigma_interior, sigma_edge, sigma_isolated)
        jitter = rng.uniform(per_rum_jitter_min, per_rum_jitter_max)

        new_sigma: List[float] = []

        for ep_idx in range(n_epochs):
            dy = float(dec_years[ep_idx])

            sigma = base * time_factor(ep_idx, n_epochs, sigma_time_growth) * jitter
            sigma += seasonal_factor(dy, base, seasonal_amplitude, seasonal_peak_month)
            sigma += spike_contribution(ep_idx, spike_epochs, spike_magnitude, spike_decay_epochs)
            sigma += abs(rng.gauss(0.0, base * epoch_noise_fraction))
            sigma = max(sigma_floor_mm, round(sigma, round_digits))

            new_sigma.append(sigma)
            sigma_samples.append(sigma)

            disp = abs(safe_float(vertical_mm[ep_idx], 0.0) or 0.0)
            snr = disp / (sigma + 1e-6)
            if snr < 0.5:
                snr_bins["<0.5"] += 1
            elif snr < 1.0:
                snr_bins["0.5-1"] += 1
            elif snr < 3.0:
                snr_bins["1-3"] += 1
            else:
                snr_bins[">3"] += 1

        new_entry = dict(entry)
        new_entry["vertical_mm"] = vertical_mm  # explicit: unchanged
        new_entry["sigma_mm"] = new_sigma
        new_entry["sigma_source"] = "synthetic_topology_time_season_spike"
        new_series[rum_id] = new_entry

    ok(f"Synthesized sigma for {len(new_series)} RUMs in {time.time() - t0:.2f}s")
    if skipped_bad_series:
        warn(f"Skipped malformed series: {skipped_bad_series}")

    section("Sigma distribution")
    if sigma_samples:
        print(f"  Min    : {min(sigma_samples):8.4f} mm")
        print(f"  P10    : {percentile(sigma_samples, 10):8.4f} mm")
        print(f"  Median : {percentile(sigma_samples, 50):8.4f} mm")
        print(f"  P90    : {percentile(sigma_samples, 90):8.4f} mm")
        print(f"  Max    : {max(sigma_samples):8.4f} mm")

    section("SNR distribution")
    total_snr = max(sum(snr_bins.values()), 1)
    for label, count in snr_bins.items():
        bar = "#" * int(35 * count / total_snr)
        print(f"  SNR {label:6s}: {100 * count / total_snr:5.1f}%  {bar}")

    clean_pct = 100 * snr_bins[">3"] / total_snr
    if clean_pct > 70:
        ok(f"{clean_pct:.1f}% high-SNR observations; most of the map remains visually reliable")
    else:
        warn(f"{clean_pct:.1f}% high-SNR observations; uncertainty visualization may dominate")

    section("Writing output")
    output = dict(epoch_data)

    metadata = dict(output.get("metadata", {}))
    metadata.update({
        "sigma_enhanced": True,
        "sigma_enhancement_method": "synthetic_topology_time_season_spike",
        "sigma_note": (
            "sigma_mm is SYNTHESIZED for visualization/testing. "
            "vertical_mm values are unchanged from the input epoch file. "
            "Spatial pattern is based on RUM grid topology; temporal pattern includes "
            "slow growth, optional seasonal modulation, and optional global spike events."
        ),
        "sigma_enhancement_config": {
            "random_seed": random_seed,
            "sigma_interior_mm": sigma_interior,
            "sigma_edge_mm": sigma_edge,
            "sigma_isolated_mm": sigma_isolated,
            "sigma_time_growth": sigma_time_growth,
            "seasonal_amplitude": seasonal_amplitude,
            "seasonal_peak_month": seasonal_peak_month,
            "n_spike_epochs": n_spike_epochs,
            "spike_magnitude_mm": spike_magnitude,
            "spike_decay_epochs": spike_decay_epochs,
            "edge_neighbour_threshold": edge_neighbour_threshold,
            "isolated_neighbour_threshold": isolated_neighbour_threshold,
            "sigma_floor_mm": sigma_floor_mm,
        },
        "spike_epochs": [epochs[sp] for sp in spike_epochs],
        "source_epoch_json": str(input_epoch_path),
        "source_footprints_json": str(footprints_path),
        "created_by": "04_enhance_vertical_sigma_optional.py",
        "created_unix": int(time.time()),
    })

    output["metadata"] = metadata
    output["series"] = new_series

    output_epoch_path.parent.mkdir(parents=True, exist_ok=True)
    with output_epoch_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, separators=(",", ":"))

    ok(f"Written: {output_epoch_path} ({output_epoch_path.stat().st_size / 1024 / 1024:.2f} MB)")

    section("SUMMARY")
    ok("Step 04 complete — enhanced sigma epoch JSON created")
    ok("vertical_mm unchanged; sigma_mm replaced/enhanced")
    ok("Next template step: 05_validate_prepared_inputs.py")
    print(f"  Elapsed: {time.time() - t_start:.2f}s")
    print()


if __name__ == "__main__":
    main()
