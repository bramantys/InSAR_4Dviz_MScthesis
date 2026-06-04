#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_enhance_vertical_sigma.py

InSAR4D RUM Viewer pipeline step 04.

Purpose
-------
Enhance/rewrite the vertical uncertainty series (`sigma_mm`) for the synthetic
epoch product.

Important rule
--------------
This step MUST NOT change vertical MEASUREMENT or MODEL values.

It reads:
  - vertical_epochs_base.json
  - rum_footprints.json

It writes:
  - vertical_epochs.json

The vertical MEASUREMENT and MODEL series remain identical to the base product.
Only sigma_mm is replaced/enhanced using the user-selected quality preset:

  user_inputs.synthetic_epochs.uncertainty_quality = high | medium | low

Why synthetic?
--------------
Prototype1 RUM input is velocity-only. We synthesize epoch displacement to
make a 4D visualization possible. The uncertainty/hatch field is therefore
also a visual/communication model, not a real epoch-by-epoch InSAR time-series
uncertainty product.

Sigma model components
----------------------
1. Topology component:
   interior cells get lower sigma, edge/isolated cells get higher sigma.

2. Temporal growth:
   uncertainty grows through time.

3. Seasonal component:
   mild periodic increase/decrease in visual uncertainty.

4. Spike component:
   a few synthetic event/noise epochs increase sigma.

5. Per-RUM jitter:
   avoids unnaturally uniform sigma surfaces.
"""

from __future__ import annotations

import calendar
import datetime as dt
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pipeline_config import load_resolved_config, resolve_path


# =============================================================================
# ADVANCED SETTINGS
# =============================================================================
# Normal users should edit config/project_config.json only.

USE_EIGHT_NEIGHBOURS = True
PRESERVE_VERTICAL_ROLES = True

# Spike footprint: "global" means all RUMs receive a spike at the same epoch(s),
# but scaled by RUM jitter. This is visually legible and deterministic.
SPIKE_MODE = "global"

# Safety clamp to avoid impossible/ugly sigma explosions.
MAX_SIGMA_MM = 9999.0

# Output rounding.
ROUND_SIGMA_DIGITS = 4
ROUND_SUMMARY_DIGITS = 4


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


def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


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
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))


def safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        if value is None:
            return fallback
        out = float(value)
        if not math.isfinite(out):
            return fallback
        return out
    except Exception:
        return fallback


def safe_int(value: Any, fallback: int = 0) -> int:
    try:
        if value is None:
            return fallback
        return int(value)
    except Exception:
        return fallback


def parse_date(value: str) -> dt.date:
    return dt.date.fromisoformat(str(value)[:10])


def month_distance_cyclic(month: int, peak_month: int) -> int:
    """
    Smallest cyclic distance between two months, range 0..6.
    """
    d = abs(month - peak_month)
    return min(d, 12 - d)


def percentile(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]

    xs = sorted(values)
    k = (len(xs) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)

    if f == c:
        return xs[int(k)]

    return xs[f] * (c - k) + xs[c] * (k - f)


# =============================================================================
# FOOTPRINT / TOPOLOGY HELPERS
# =============================================================================

def load_grid_from_footprints(footprints: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    features = footprints.get("features", [])
    if not features:
        raise ValueError("rum_footprints.json contains no features")

    out: Dict[str, Dict[str, Any]] = {}

    for idx, feature in enumerate(features):
        props = feature.get("properties") or {}
        rum_id = str(props.get("rum_id", f"RUM_{idx + 1:06d}"))

        if "grid_i" not in props or "grid_j" not in props:
            raise ValueError(f"Footprint for {rum_id} is missing grid_i/grid_j")

        out[rum_id] = {
            "grid_i": int(props["grid_i"]),
            "grid_j": int(props["grid_j"]),
        }

    return out


def neighbour_offsets() -> List[Tuple[int, int]]:
    if USE_EIGHT_NEIGHBOURS:
        return [
            (-1, -1), (0, -1), (1, -1),
            (-1,  0),          (1,  0),
            (-1,  1), (0,  1), (1,  1),
        ]

    return [(0, -1), (-1, 0), (1, 0), (0, 1)]


def classify_topology(
    grid: Dict[str, Dict[str, Any]],
    edge_threshold: int,
    isolated_threshold: int,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
    occupied = {
        (item["grid_i"], item["grid_j"])
        for item in grid.values()
    }

    offsets = neighbour_offsets()
    classified: Dict[str, Dict[str, Any]] = {}
    counts = {"interior": 0, "edge": 0, "isolated": 0}

    for rum_id, item in grid.items():
        gi = item["grid_i"]
        gj = item["grid_j"]

        n = sum((gi + di, gj + dj) in occupied for di, dj in offsets)

        if n <= isolated_threshold:
            cls = "isolated"
        elif n < edge_threshold:
            cls = "edge"
        else:
            cls = "interior"

        counts[cls] += 1

        classified[rum_id] = {
            "grid_i": gi,
            "grid_j": gj,
            "neighbour_count": n,
            "topology_class": cls,
        }

    return classified, counts


# =============================================================================
# SIGMA MODEL
# =============================================================================

def build_temporal_multiplier(n_epochs: int, growth_factor: float) -> List[float]:
    if n_epochs <= 1:
        return [1.0]

    growth_factor = max(0.0, float(growth_factor))
    return [
        1.0 + (growth_factor - 1.0) * (i / (n_epochs - 1))
        for i in range(n_epochs)
    ]


def build_seasonal_multiplier(epochs: List[str], amplitude: float, peak_month: int) -> List[float]:
    """
    Seasonal factor ranges approximately:
      1 - amplitude ... 1 + amplitude

    Peak occurs around peak_month. Values are clamped to avoid negative sigma.
    """
    amp = max(0.0, float(amplitude))
    peak_month = min(12, max(1, int(peak_month)))

    out: List[float] = []
    for e in epochs:
        d = parse_date(e)
        # cos = 1 at peak month, -1 opposite peak.
        angle = 2.0 * math.pi * month_distance_cyclic(d.month, peak_month) / 6.0
        factor = 1.0 + amp * math.cos(angle)
        out.append(max(0.05, factor))

    return out


def choose_spike_epochs(n_epochs: int, n_spikes: int, rng: random.Random) -> List[int]:
    if n_epochs <= 0 or n_spikes <= 0:
        return []

    # Avoid epoch 0 if possible because vertical series usually start at zero and
    # uncertainty spikes there look visually odd.
    population = list(range(1, n_epochs)) if n_epochs > 1 else [0]
    n = min(int(n_spikes), len(population))
    return sorted(rng.sample(population, n))


def build_spike_addition(
    n_epochs: int,
    spike_epochs: List[int],
    magnitude_mm: float,
    decay_epochs: float,
) -> List[float]:
    if not spike_epochs or magnitude_mm <= 0:
        return [0.0] * n_epochs

    decay = max(0.1, float(decay_epochs))
    out = [0.0] * n_epochs

    for i in range(n_epochs):
        additions = []
        for s in spike_epochs:
            dist = abs(i - s)
            additions.append(magnitude_mm * math.exp(-dist / decay))
        out[i] = max(additions) if additions else 0.0

    return out


def topology_base_sigma(topology_class: str, sigma_cfg: Dict[str, Any]) -> float:
    if topology_class == "isolated":
        return float(sigma_cfg["sigma_isolated_mm"])
    if topology_class == "edge":
        return float(sigma_cfg["sigma_edge_mm"])
    return float(sigma_cfg["sigma_interior_mm"])


def enhance_sigma_for_rum(
    topology_class: str,
    temporal_multiplier: List[float],
    seasonal_multiplier: List[float],
    spike_addition: List[float],
    sigma_cfg: Dict[str, Any],
    rng: random.Random,
) -> List[float]:
    n_epochs = len(temporal_multiplier)

    base = topology_base_sigma(topology_class, sigma_cfg)

    jitter = rng.uniform(
        float(sigma_cfg.get("per_rum_jitter_min", 1.0)),
        float(sigma_cfg.get("per_rum_jitter_max", 1.0)),
    )

    noise_fraction = max(0.0, float(sigma_cfg.get("epoch_noise_fraction_of_base", 0.0)))
    sigma_floor = max(0.0, float(sigma_cfg.get("sigma_floor_mm", 0.0)))

    out: List[float] = []

    for i in range(n_epochs):
        sigma = base * jitter
        sigma *= temporal_multiplier[i]
        sigma *= seasonal_multiplier[i]
        sigma += spike_addition[i] * jitter

        if noise_fraction > 0:
            sigma += rng.gauss(0.0, base * noise_fraction)

        sigma = max(sigma_floor, sigma)
        sigma = min(MAX_SIGMA_MM, sigma)

        out.append(round(sigma, ROUND_SIGMA_DIGITS))

    return out


def assert_vertical_roles_preserved(
    base_series: Dict[str, Any],
    enhanced_series: Dict[str, Any],
) -> None:
    """
    Step 04 is allowed to change only sigma_mm.

    It must preserve:
      - measurement_mm: trendline / popup / labelling series
      - model_mm: RUM height / choropleth / walls / blankies series
    """
    required_roles = ["measurement_mm", "model_mm"]

    for rum_id, base_item in base_series.items():
        if rum_id not in enhanced_series:
            raise ValueError(f"RUM missing after sigma enhancement: {rum_id}")

        enhanced_item = enhanced_series[rum_id]

        for role in required_roles:
            if role not in base_item:
                raise ValueError(f"Base epoch product missing {role} for {rum_id}")
            if role not in enhanced_item:
                raise ValueError(f"Enhanced epoch product missing {role} for {rum_id}")

            if base_item.get(role) != enhanced_item.get(role):
                raise ValueError(f"{role} changed for {rum_id}; Step 04 may only change sigma_mm")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    t_start = time.time()

    cfg = load_resolved_config(__file__)
    project_root = Path(cfg["_resolved"]["project_root"])

    prepared = cfg["prepared_inputs"]
    generated = cfg["generated_outputs"]
    sigma_cfg = cfg["sigma_enhancement"]

    input_base_path = resolve_path(project_root, prepared["vertical_epoch_json_without_enhanced_sigma"])
    footprints_path = resolve_path(project_root, generated["rum_footprints"])
    output_path = resolve_path(project_root, prepared["vertical_epoch_json"])

    quality = cfg["epoch_generation"].get("uncertainty_quality", "unknown")

    section("Configuration")
    print(f"  Project root          : {project_root}")
    print(f"  Base epoch input      : {input_base_path}")
    print(f"  Footprints input      : {footprints_path}")
    print(f"  Enhanced epoch output : {output_path}")
    print(f"  Quality preset        : {quality}")

    section("Loading inputs")
    base_product = load_json(input_base_path)
    footprints = load_json(footprints_path)

    epochs = base_product.get("epochs", [])
    base_series = base_product.get("series", {})

    if not epochs:
        raise ValueError("vertical epoch product has no epochs")
    if not base_series:
        raise ValueError("vertical epoch product has no series")

    ok(f"Loaded base epoch product: {len(base_series)} RUMs × {len(epochs)} epochs")

    grid = load_grid_from_footprints(footprints)
    ok(f"Loaded footprint grid for {len(grid)} RUMs")

    missing_grid = sorted(set(base_series) - set(grid))
    if missing_grid:
        raise ValueError(f"RUMs exist in vertical series but not footprints; sample={missing_grid[:10]}")

    section("Classifying topology")
    classified, topology_counts = classify_topology(
        grid=grid,
        edge_threshold=int(sigma_cfg.get("edge_neighbour_threshold", 5)),
        isolated_threshold=int(sigma_cfg.get("isolated_neighbour_threshold", 2)),
    )

    ok(
        "Topology classes: "
        f"interior={topology_counts['interior']}, "
        f"edge={topology_counts['edge']}, "
        f"isolated={topology_counts['isolated']}"
    )

    section("Building temporal sigma components")
    rng = random.Random(int(sigma_cfg.get("random_seed", 42)))

    temporal_multiplier = build_temporal_multiplier(
        n_epochs=len(epochs),
        growth_factor=float(sigma_cfg.get("sigma_time_growth", 1.0)),
    )
    seasonal_multiplier = build_seasonal_multiplier(
        epochs=epochs,
        amplitude=float(sigma_cfg.get("seasonal_amplitude", 0.0)),
        peak_month=int(sigma_cfg.get("seasonal_peak_month", 1)),
    )
    spike_epochs = choose_spike_epochs(
        n_epochs=len(epochs),
        n_spikes=int(sigma_cfg.get("n_spike_epochs", 0)),
        rng=rng,
    )
    spike_addition = build_spike_addition(
        n_epochs=len(epochs),
        spike_epochs=spike_epochs,
        magnitude_mm=float(sigma_cfg.get("spike_magnitude_mm", 0.0)),
        decay_epochs=float(sigma_cfg.get("spike_decay_epochs", 1.0)),
    )

    ok(f"Temporal growth factor end/start: {temporal_multiplier[-1]:.3f}")
    ok(f"Seasonal multiplier range: {min(seasonal_multiplier):.3f} to {max(seasonal_multiplier):.3f}")

    if spike_epochs:
        ok(f"Synthetic spike epochs: {spike_epochs}")
    else:
        ok("Synthetic spike epochs: none")

    section("Enhancing sigma")
    enhanced_product = json.loads(json.dumps(base_product))  # deep copy via JSON-safe structure
    enhanced_series = enhanced_product["series"]

    all_sigma_values: List[float] = []
    all_base_sigma_values: List[float] = []

    for rum_id, item in enhanced_series.items():
        topology = classified[rum_id]["topology_class"]
        old_sigma = item.get("sigma_mm", [])
        all_base_sigma_values.extend([safe_float(v, 0.0) for v in old_sigma])

        new_sigma = enhance_sigma_for_rum(
            topology_class=topology,
            temporal_multiplier=temporal_multiplier,
            seasonal_multiplier=seasonal_multiplier,
            spike_addition=spike_addition,
            sigma_cfg=sigma_cfg,
            rng=rng,
        )

        item["sigma_mm"] = new_sigma
        item["sigma_model"] = {
            "type": "synthetic_quality_preset",
            "quality_preset": quality,
            "topology_class": topology,
            "neighbour_count": classified[rum_id]["neighbour_count"],
        }

        all_sigma_values.extend(new_sigma)

    if PRESERVE_VERTICAL_ROLES:
        assert_vertical_roles_preserved(base_series, enhanced_series)

    ok("Vertical MEASUREMENT and MODEL roles preserved")
    ok(f"Enhanced sigma for {len(enhanced_series)} RUMs")

    section("Writing enhanced epoch product")
    metadata = enhanced_product.setdefault("metadata", {})
    metadata["sigma_enhancement"] = {
        "schema": "synthetic_sigma_quality_preset_v2_measurement_model",
        "quality_preset": quality,
        "preserves_vertical_roles": True,
        "topology_counts": topology_counts,
        "sigma_interior_mm": float(sigma_cfg["sigma_interior_mm"]),
        "sigma_edge_mm": float(sigma_cfg["sigma_edge_mm"]),
        "sigma_isolated_mm": float(sigma_cfg["sigma_isolated_mm"]),
        "sigma_time_growth": float(sigma_cfg["sigma_time_growth"]),
        "seasonal_amplitude": float(sigma_cfg["seasonal_amplitude"]),
        "spike_epochs": spike_epochs,
        "spike_magnitude_mm": float(sigma_cfg["spike_magnitude_mm"]),
        "spike_decay_epochs": float(sigma_cfg["spike_decay_epochs"]),
    }

    metadata.setdefault("summary", {})
    metadata["summary"].update({
        "base_sigma_min_mm": round(min(all_base_sigma_values), ROUND_SUMMARY_DIGITS) if all_base_sigma_values else None,
        "base_sigma_max_mm": round(max(all_base_sigma_values), ROUND_SUMMARY_DIGITS) if all_base_sigma_values else None,
        "enhanced_sigma_min_mm": round(min(all_sigma_values), ROUND_SUMMARY_DIGITS),
        "enhanced_sigma_p02_mm": round(percentile(all_sigma_values, 2), ROUND_SUMMARY_DIGITS),
        "enhanced_sigma_p50_mm": round(percentile(all_sigma_values, 50), ROUND_SUMMARY_DIGITS),
        "enhanced_sigma_p98_mm": round(percentile(all_sigma_values, 98), ROUND_SUMMARY_DIGITS),
        "enhanced_sigma_max_mm": round(max(all_sigma_values), ROUND_SUMMARY_DIGITS),
    })

    write_json(output_path, enhanced_product)

    elapsed = time.time() - t_start

    ok(f"Wrote enhanced vertical epochs: {output_path} ({output_path.stat().st_size / 1024 / 1024:.2f} MB)")

    section("Summary")
    ok(f"Step 04 complete in {elapsed:.2f} s")
    print(f"  Quality preset         : {quality}")
    print(f"  RUM count              : {len(enhanced_series)}")
    print(f"  Epoch count            : {len(epochs)}")
    print(f"  Topology interior/edge/isolated: {topology_counts['interior']}/{topology_counts['edge']}/{topology_counts['isolated']}")
    print(f"  Enhanced sigma min     : {min(all_sigma_values):.4f} mm")
    print(f"  Enhanced sigma median  : {percentile(all_sigma_values, 50):.4f} mm")
    print(f"  Enhanced sigma max     : {max(all_sigma_values):.4f} mm")


if __name__ == "__main__":
    main()
