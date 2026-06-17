#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_enhance_vertical_sigma.py

Build a deterministic, synthetic epoch-dependent vertical-uncertainty field.

Prototype 1 starts from a velocity-only RUM product. MODEL and MEASUREMENT
are therefore synthetic epoch series, and the changing epoch uncertainty is
also explicitly a visualization-test field rather than reconstructed InSAR
measurement uncertainty.

The generated sigma field preserves useful structure:
  * source vertical-velocity sigma controls the persistent spatial baseline;
  * edge / isolated RUMs receive a modest topology multiplier;
  * nearby RUMs vary smoothly through spatially varying periodic phases;
  * three deterministic regional episodes rise and recover smoothly;
  * a small smooth quasi-periodic term prevents a perfectly mechanical cycle.

Only ``sigma_mm`` is changed. ``measurement_mm`` and ``model_mm`` are copied
verbatim and verified after generation.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pipeline_config import load_resolved_config, resolve_path

USE_EIGHT_NEIGHBOURS = True
PRESERVE_VERTICAL_ROLES = True
MAX_SIGMA_MM = 9999.0
ROUND_SIGMA_DIGITS = 4
ROUND_SUMMARY_DIGITS = 4


def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def ok(msg: str) -> None:
    print(f"  [OK]   {msg}")


def warn(msg: str) -> None:
    print(f"  [WARN] {msg}")


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
        out = float(value)
        return out if math.isfinite(out) else fallback
    except Exception:
        return fallback


def percentile(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    xs = sorted(values)
    k = (len(xs) - 1) * p / 100.0
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return xs[int(k)]
    return xs[lo] * (hi - k) + xs[hi] * (k - lo)


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
            (-1, 0),            (1, 0),
            (-1, 1),  (0, 1),  (1, 1),
        ]
    return [(0, -1), (-1, 0), (1, 0), (0, 1)]


def classify_topology(
    grid: Dict[str, Dict[str, Any]],
    edge_threshold: int,
    isolated_threshold: int,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
    occupied = {(v["grid_i"], v["grid_j"]) for v in grid.values()}
    offsets = neighbour_offsets()
    counts = {"interior": 0, "edge": 0, "isolated": 0}
    out: Dict[str, Dict[str, Any]] = {}

    gi_values = [v["grid_i"] for v in grid.values()]
    gj_values = [v["grid_j"] for v in grid.values()]
    gi_min, gi_max = min(gi_values), max(gi_values)
    gj_min, gj_max = min(gj_values), max(gj_values)
    gi_span = max(1, gi_max - gi_min)
    gj_span = max(1, gj_max - gj_min)

    for rum_id, item in grid.items():
        gi, gj = item["grid_i"], item["grid_j"]
        neighbours = sum((gi + di, gj + dj) in occupied for di, dj in offsets)
        if neighbours <= isolated_threshold:
            topology = "isolated"
        elif neighbours < edge_threshold:
            topology = "edge"
        else:
            topology = "interior"
        counts[topology] += 1
        out[rum_id] = {
            "grid_i": gi,
            "grid_j": gj,
            "x_norm": (gi - gi_min) / gi_span,
            "y_norm": (gj - gj_min) / gj_span,
            "neighbour_count": neighbours,
            "topology_class": topology,
        }
    return out, counts


def gaussian(value: float, center: float, sigma: float) -> float:
    sigma = max(float(sigma), 1e-6)
    z = (float(value) - float(center)) / sigma
    return math.exp(-0.5 * z * z)


def topology_factor(topology: str, cfg: Dict[str, Any]) -> float:
    if topology == "isolated":
        return float(cfg.get("topology_isolated_factor", 1.30))
    if topology == "edge":
        return float(cfg.get("topology_edge_factor", 1.15))
    return 1.0


def default_episodes(cfg: Dict[str, Any]) -> List[Dict[str, float]]:
    episodes = cfg.get("regional_episodes")
    if isinstance(episodes, list) and episodes:
        return episodes
    return [
        {
            "name": "north_broad",
            "center_x": 0.55, "center_y": 0.15,
            "sigma_x": 0.38, "sigma_y": 0.20,
            "epoch_fraction": 0.25, "time_sigma_fraction": 0.055,
            "amplitude_mm": float(cfg.get("episode_1_amplitude_mm", 1.00)),
        },
        {
            "name": "west_central",
            "center_x": 0.30, "center_y": 0.52,
            "sigma_x": 0.24, "sigma_y": 0.26,
            "epoch_fraction": 0.58, "time_sigma_fraction": 0.065,
            "amplitude_mm": float(cfg.get("episode_2_amplitude_mm", 1.30)),
        },
        {
            "name": "citywide_soft",
            "center_x": 0.55, "center_y": 0.45,
            "sigma_x": 0.62, "sigma_y": 0.58,
            "epoch_fraction": 0.82, "time_sigma_fraction": 0.075,
            "amplitude_mm": float(cfg.get("episode_3_amplitude_mm", 0.90)),
        },
        {
            "name": "compact_quality_drop",
            "center_x": 0.72, "center_y": 0.34,
            "sigma_x": 0.065, "sigma_y": 0.065,
            "epoch_fraction": 0.43, "time_sigma_fraction": 0.026,
            "amplitude_mm": float(cfg.get("episode_4_amplitude_mm", 3.60)),
        },
    ]


def build_sigma_series(
    *,
    item: Dict[str, Any],
    spatial: Dict[str, Any],
    n_epochs: int,
    source_sigma_p98: float,
    cfg: Dict[str, Any],
    episodes: List[Dict[str, float]],
) -> Tuple[List[float], Dict[str, float]]:
    source_sigma = max(0.0, safe_float(item.get("source_sigma_up_mm_yr"), 0.0))
    cap_ratio = max(1.0, float(cfg.get("source_sigma_cap_ratio", 1.50)))
    source_ratio = min(cap_ratio, source_sigma / max(source_sigma_p98, 1e-9))

    floor_mm = max(0.0, float(cfg.get("source_sigma_floor_mm", 0.12)))
    scale_mm = max(0.0, float(cfg.get("source_sigma_scale_mm", 1.55)))
    baseline = (floor_mm + scale_mm * source_ratio) * topology_factor(
        spatial["topology_class"], cfg
    )

    x = float(spatial["x_norm"])
    y = float(spatial["y_norm"])

    # Slowly varying spatial texture; nearby RUMs receive similar values.
    spatial_jitter_fraction = max(0.0, float(cfg.get("spatial_jitter_fraction", 0.08)))
    smooth_spatial = (
        0.55 * math.sin(2.0 * math.pi * (0.72 * x + 0.31 * y))
        + 0.45 * math.cos(2.0 * math.pi * (0.18 * x - 0.63 * y))
    )
    baseline *= max(0.35, 1.0 + spatial_jitter_fraction * smooth_spatial)

    amp1 = max(0.0, float(cfg.get("seasonal_primary_fraction", 0.18)))
    amp2 = max(0.0, float(cfg.get("seasonal_secondary_fraction", 0.08)))
    noise_amp = max(0.0, float(cfg.get("smooth_noise_fraction", 0.06)))
    sigma_floor = max(0.0, float(cfg.get("sigma_floor_mm", 0.05)))

    phase1 = 2.0 * math.pi * (0.39 * x + 0.21 * y)
    phase2 = 2.0 * math.pi * (-0.17 * x + 0.47 * y)
    phase3 = 2.0 * math.pi * (0.61 * x - 0.23 * y)

    out: List[float] = []
    event_peak = 0.0
    for epoch_index in range(n_epochs):
        t = epoch_index / max(1, n_epochs - 1)

        seasonal = (
            amp1 * math.sin(2.0 * math.pi * 4.25 * t + phase1)
            + amp2 * math.sin(2.0 * math.pi * 8.50 * t + phase2)
        )
        smooth_noise = noise_amp * (
            0.60 * math.sin(2.0 * math.pi * 2.15 * t + phase3)
            + 0.40 * math.sin(2.0 * math.pi * 6.35 * t + phase1 * 0.7)
        )

        event_add = 0.0
        for event in episodes:
            spatial_weight = gaussian(x, event["center_x"], event["sigma_x"]) * gaussian(
                y, event["center_y"], event["sigma_y"]
            )
            temporal_weight = gaussian(
                t, event["epoch_fraction"], event["time_sigma_fraction"]
            )
            event_add += (
                float(event["amplitude_mm"])
                * spatial_weight
                * temporal_weight
                * (0.65 + 0.35 * min(1.0, source_ratio))
            )

        event_peak = max(event_peak, event_add)
        sigma = baseline * max(0.25, 1.0 + seasonal + smooth_noise) + event_add
        sigma = min(MAX_SIGMA_MM, max(sigma_floor, sigma))
        out.append(round(sigma, ROUND_SIGMA_DIGITS))

    return out, {
        "source_sigma_up_mm_yr": source_sigma,
        "source_sigma_ratio_to_p98": source_ratio,
        "persistent_baseline_mm": baseline,
        "regional_episode_peak_addition_mm": event_peak,
    }


def assert_vertical_roles_preserved(
    base_series: Dict[str, Any], enhanced_series: Dict[str, Any]
) -> None:
    for rum_id, base_item in base_series.items():
        if rum_id not in enhanced_series:
            raise ValueError(f"RUM missing after sigma enhancement: {rum_id}")
        for role in ("measurement_mm", "model_mm"):
            if base_item.get(role) != enhanced_series[rum_id].get(role):
                raise ValueError(f"{role} changed for {rum_id}; Step 04 may only change sigma_mm")


def main() -> None:
    t_start = time.time()
    cfg = load_resolved_config(__file__)
    project_root = Path(cfg["_resolved"]["project_root"])
    prepared = cfg["prepared_inputs"]
    generated = cfg["generated_outputs"]
    sigma_cfg = cfg["sigma_enhancement"]

    input_path = resolve_path(project_root, prepared["vertical_epoch_json_without_enhanced_sigma"])
    footprints_path = resolve_path(project_root, generated["rum_footprints"])
    output_path = resolve_path(project_root, prepared["vertical_epoch_json"])
    quality = cfg["epoch_generation"].get("uncertainty_quality", "unknown")

    section("Configuration")
    print(f"  Project root          : {project_root}")
    print(f"  Base epoch input      : {input_path}")
    print(f"  Footprints input      : {footprints_path}")
    print(f"  Enhanced epoch output : {output_path}")
    print(f"  Quality preset        : {quality}")
    print("  Sigma model           : deterministic spatial-temporal visualization mockup")

    base_product = load_json(input_path)
    footprints = load_json(footprints_path)
    epochs = base_product.get("epochs", [])
    base_series = base_product.get("series", {})
    if not epochs or not base_series:
        raise ValueError("Base vertical epoch product is empty")

    grid = load_grid_from_footprints(footprints)
    missing_grid = sorted(set(base_series) - set(grid))
    if missing_grid:
        raise ValueError(f"RUMs missing from footprints; sample={missing_grid[:10]}")

    classified, topology_counts = classify_topology(
        grid,
        edge_threshold=int(sigma_cfg.get("edge_neighbour_threshold", 5)),
        isolated_threshold=int(sigma_cfg.get("isolated_neighbour_threshold", 2)),
    )
    ok(
        f"Topology: interior={topology_counts['interior']}, "
        f"edge={topology_counts['edge']}, isolated={topology_counts['isolated']}"
    )

    source_sigmas = [
        max(0.0, safe_float(item.get("source_sigma_up_mm_yr"), 0.0))
        for item in base_series.values()
    ]
    source_sigma_p98 = max(percentile(source_sigmas, 98), 1e-9)
    episodes = default_episodes(sigma_cfg)
    ok(f"Source vertical-velocity sigma p98: {source_sigma_p98:.4f} mm/yr")
    ok("Regional episodes: " + ", ".join(str(e.get("name", "episode")) for e in episodes))

    enhanced_product = json.loads(json.dumps(base_product))
    enhanced_series = enhanced_product["series"]
    all_sigma: List[float] = []
    epoch_values: List[List[float]] = [[] for _ in epochs]

    for rum_id, item in enhanced_series.items():
        sigma_values, diagnostics = build_sigma_series(
            item=item,
            spatial=classified[rum_id],
            n_epochs=len(epochs),
            source_sigma_p98=source_sigma_p98,
            cfg=sigma_cfg,
            episodes=episodes,
        )
        item["sigma_mm"] = sigma_values
        item["sigma_model"] = {
            "type": "synthetic_spatiotemporal_visualization_mock",
            "quality_preset": quality,
            "topology_class": classified[rum_id]["topology_class"],
            "neighbour_count": classified[rum_id]["neighbour_count"],
            **{k: round(v, ROUND_SUMMARY_DIGITS) for k, v in diagnostics.items()},
        }
        all_sigma.extend(sigma_values)
        for i, value in enumerate(sigma_values):
            epoch_values[i].append(value)

    if PRESERVE_VERTICAL_ROLES:
        assert_vertical_roles_preserved(base_series, enhanced_series)
    ok("Vertical MEASUREMENT and MODEL roles preserved")
    ok(f"Generated sigma for {len(enhanced_series)} RUMs × {len(epochs)} epochs")

    epoch_summary = []
    for label, values in zip(epochs, epoch_values):
        epoch_summary.append({
            "epoch": label,
            "sigma_p02_mm": round(percentile(values, 2), ROUND_SUMMARY_DIGITS),
            "sigma_p50_mm": round(percentile(values, 50), ROUND_SUMMARY_DIGITS),
            "sigma_p98_mm": round(percentile(values, 98), ROUND_SUMMARY_DIGITS),
            "sigma_max_mm": round(max(values), ROUND_SUMMARY_DIGITS),
        })

    metadata = enhanced_product.setdefault("metadata", {})
    metadata["sigma_enhancement"] = {
        "schema": "synthetic_spatiotemporal_vertical_sigma_v3",
        "status": "visualization_test_mockup_not_observed_epoch_uncertainty",
        "quality_preset": quality,
        "preserves_vertical_roles": True,
        "deterministic": True,
        "random_seed": int(sigma_cfg.get("random_seed", 42)),
        "source_baseline": "source_sigma_up_mm_yr",
        "source_sigma_p98_mm_yr": round(source_sigma_p98, ROUND_SUMMARY_DIGITS),
        "topology_counts": topology_counts,
        "regional_episodes": episodes,
        "global_summary": {
            "sigma_min_mm": round(min(all_sigma), ROUND_SUMMARY_DIGITS),
            "sigma_p02_mm": round(percentile(all_sigma, 2), ROUND_SUMMARY_DIGITS),
            "sigma_p50_mm": round(percentile(all_sigma, 50), ROUND_SUMMARY_DIGITS),
            "sigma_p98_mm": round(percentile(all_sigma, 98), ROUND_SUMMARY_DIGITS),
            "sigma_p99_mm": round(percentile(all_sigma, 99), ROUND_SUMMARY_DIGITS),
            "sigma_max_mm": round(max(all_sigma), ROUND_SUMMARY_DIGITS),
        },
        "epoch_summary": epoch_summary,
        "notes": [
            "Synthetic epoch uncertainty is intended to exercise the vertical-uncertainty visualization.",
            "It must not be interpreted as reconstructed Jakarta epoch measurement uncertainty.",
            "The viewer applies the fixed global p98 display-height ceiling; raw values remain stored.",
        ],
    }

    summary = metadata.setdefault("summary", {})
    summary["sigma_min_mm"] = round(min(all_sigma), ROUND_SUMMARY_DIGITS)
    summary["sigma_max_mm"] = round(max(all_sigma), ROUND_SUMMARY_DIGITS)

    write_json(output_path, enhanced_product)
    elapsed = time.time() - t_start

    section("Summary")
    ok(f"Wrote enhanced epoch product: {output_path}")
    ok(f"Step 04 complete in {elapsed:.2f} s")
    print(f"  Sigma min / median : {min(all_sigma):.4f} / {percentile(all_sigma, 50):.4f} mm")
    print(f"  Sigma p98 / max    : {percentile(all_sigma, 98):.4f} / {max(all_sigma):.4f} mm")


if __name__ == "__main__":
    main()
