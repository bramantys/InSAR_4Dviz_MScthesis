#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

from _pass3_common import (
    Pass3Error,
    atomic_copy,
    clean_stage_area,
    file_record,
    print_pass,
    project_root_from,
    read_json,
    require,
    sha256,
    stage_root,
    write_json,
)
from _proto2_config import load_project_config, output_data_dir
import _color_scale_support as color_support
import _glb_piston_support as piston_support


def _validate_lod_stage(lod_root: Path) -> dict[str, Any]:
    manifest_path = require(lod_root / "uncertainty_lod_manifest.json", "uncertainty LOD manifest")
    manifest = read_json(manifest_path)
    families = manifest.get("lod_families")
    if not isinstance(families, dict):
        raise Pass3Error("uncertainty LOD manifest is missing lod_families")
    for family_key in ("detail", "overview"):
        family = families.get(family_key)
        tiles = family.get("tiles") if isinstance(family, dict) else None
        if not isinstance(tiles, list) or not tiles:
            raise Pass3Error(f"uncertainty LOD family {family_key!r} has no tiles")
        for tile in tiles:
            url = tile.get("url") if isinstance(tile, dict) else None
            if not isinstance(url, str) or not url:
                raise Pass3Error(f"uncertainty LOD {family_key} tile has no URL")
            file_name = Path(url).name
            stage_subdir = Path(url).parent.name
            if not stage_subdir or not (lod_root / stage_subdir / file_name).is_file():
                raise Pass3Error(f"uncertainty LOD tile missing from stage: {stage_subdir or family_key}/{file_name}")
    return {
        "manifest": file_record(manifest_path, lod_root.parents[4]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build staged Proto2 runtime manifests and viewer products.")
    parser.parse_args()

    project_root = project_root_from(__file__)
    config = load_project_config(project_root)
    output_data = output_data_dir(project_root, config)
    stage = stage_root(project_root)
    animation_dir = clean_stage_area(project_root, "animation")
    style_dir = clean_stage_area(project_root, "style")

    geometry_summary = read_json(require(stage / "geometry_build_summary.json", "geometry summary"))
    lookup_summary = read_json(require(stage / "lookup_build_summary.json", "lookup summary"))
    lod_stage = stage / "geometry" / "uncertainty_lod"
    lod_report = _validate_lod_stage(lod_stage)

    print("\n=== PROTO2 STAGE 10: BUILD VIEWER RUNTIME MANIFEST ===")

    try:
        color_support.main()
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise Pass3Error(f"Colour-scale builder failed with exit code {exc.code}")

    animation_sources = {
        "reversible": require(output_data / "parcel_displacement_reversible_f32.bin", "reversible array"),
        "irreversible": require(output_data / "parcel_displacement_irreversible_f32.bin", "irreversible array"),
        "total": require(output_data / "parcel_displacement_total_f32.bin", "MC total mean array"),
        "deterministic_total": require(output_data / "parcel_displacement_deterministic_total_f32.bin", "deterministic decomposition-total array"),
        "sigma_h": require(output_data / "parcel_displacement_sigma_h_f32.bin", "MC total sigma array"),
        "vi": require(output_data / "parcel_vi_f32.bin", "irreversible velocity array"),
    }
    animation_outputs = {key: animation_dir / f"{key}_f32.bin" for key in animation_sources}
    for key, source in animation_sources.items():
        atomic_copy(source, animation_outputs[key])

    color_source = require(output_data / "parcel_color_scales.json", "parcel color scales")
    color_out = style_dir / "parcel_color_scales.json"
    atomic_copy(color_source, color_out)

    tuning: Dict[str, Any] = piston_support.derive_display_tuning(piston_support.read_matrices_for_tuning())
    tuning_out = style_dir / "viewer_tuning.json"
    write_json(tuning_out, tuning)

    cap = geometry_summary["cap"]
    counts = lookup_summary["counts"]
    animation_manifest = read_json(require(output_data / "parcel_animation_manifest.json", "animation manifest"))
    runtime_product_paths = {
        "reversible": "_internal/data_pipeline/runtime/animation/reversible_f32.bin",
        "irreversible": "_internal/data_pipeline/runtime/animation/irreversible_f32.bin",
        "deterministic_total": "_internal/data_pipeline/runtime/animation/deterministic_total_f32.bin",
        "total": "_internal/data_pipeline/runtime/animation/total_f32.bin",
        "sigma_h": "_internal/data_pipeline/runtime/animation/sigma_h_f32.bin",
    }
    products = animation_manifest.get("products", {})
    if not isinstance(products, dict):
        products = {}
    viewer_products = {}
    for key, payload in products.items():
        item = dict(payload) if isinstance(payload, dict) else {"meaning": str(payload)}
        if key in runtime_product_paths:
            item["binary"] = runtime_product_paths[key]
        viewer_products[key] = item
    viewer_metadata = {
        "schema": "proto2_viewer_metadata_v5_3",
        "product_type": "parcel_deformation",
        "generated_from": "numbered_pipeline",
        "vertices": int(cap["vertices"]),
        "triangles": int(cap["triangles"]),
        "indices": int(cap["indices"]),
        "moving_vertices": int(cap["moving_vertices"]),
        "blank_vertices": int(cap["blank_vertices"]),
        "total_parcels": int(counts["search_parcels"]),
        "moving_parcels": int(counts["moving_parcels"]),
        "blank_parcels": int(counts["blank_parcels"]),
        "pick_features": int(counts["pick_features"]),
        "epochs": int(cap["epochs"]),
        "epoch_start": cap["epoch_start"],
        "epoch_end": cap["epoch_end"],
        "epoch_labels": cap["epoch_labels"],
        "stats": cap["stats"],
        "max_total_diff": cap["max_total_diff"],
        "glb_size_mb": (stage / "geometry" / "parcel_caps.glb").stat().st_size / (1024 * 1024),
        "center_lon": cap["center_lon"],
        "center_lat": cap["center_lat"],
        "center_height_m": float(tuning["display_datum_height_m"]),
        "camera_height_m": cap["camera_height_m"],
        "bounds_wgs84": cap["bounds_wgs84"],
        "local_span_m": cap["local_span_m"],
        "time_reference": animation_manifest.get("time_reference", {}),
        "products": viewer_products,
        "total_product": "monte_carlo_mean_t",
        "uncertainty": {
            "available": True,
            "applies_to": "total",
            "kind": "per_epoch_standard_deviation",
            "runtime_asset": "_internal/data_pipeline/runtime/animation/sigma_h_f32.bin",
            "visualization": "full_field_checkerboard_spikes",
            "lod": "exclusive_camera_height_20m_detail_40m_overview",
            "rule": "sigma_t is trimmed with the configured viewer period unchanged and is not rebased with viewer t0.",
        },
    }
    metadata_out = stage / "viewer_metadata.json"
    write_json(metadata_out, viewer_metadata)

    three_path = require(project_root / "_internal" / "three" / "three.min.js", "local Three.js")
    runtime_manifest = {
        "schema": "proto2_runtime_manifest_v5_3",
        "product_type": "parcel_deformation",
        "geometry": {
            "caps": "_internal/data_pipeline/runtime/geometry/parcel_caps.glb",
            "pistons": "_internal/data_pipeline/runtime/geometry/parcel_pistons.glb",
            "walls": "_internal/data_pipeline/runtime/geometry/parcel_walls.glb",
            "opaque_datum_caps": "_internal/data_pipeline/runtime/geometry/parcel_datum_caps_opaque.glb",
            "uncertainty_lod_manifest": "_internal/data_pipeline/runtime/geometry/uncertainty_lod/uncertainty_lod_manifest.json",
        },
        "animation": {
            "reversible": "_internal/data_pipeline/runtime/animation/reversible_f32.bin",
            "irreversible": "_internal/data_pipeline/runtime/animation/irreversible_f32.bin",
            "total": "_internal/data_pipeline/runtime/animation/total_f32.bin",
            "deterministic_total": "_internal/data_pipeline/runtime/animation/deterministic_total_f32.bin",
            "sigma_h": "_internal/data_pipeline/runtime/animation/sigma_h_f32.bin",
            "vi": "_internal/data_pipeline/runtime/animation/vi_f32.bin",
        },
        "lookup": {
            "pick": "_internal/data_pipeline/runtime/lookup/parcel_pick_index.json",
            "search": "_internal/data_pipeline/runtime/lookup/parcel_search_index.json",
            "trendline": "_internal/data_pipeline/runtime/lookup/parcel_trendline_manifest.json",
        },
        "style": {
            "color_scales": "_internal/data_pipeline/runtime/style/parcel_color_scales.json",
            "viewer_tuning": "_internal/data_pipeline/runtime/style/viewer_tuning.json",
        },
        "runtime": {
            "three": {
                "mode": "local",
                "path": "_internal/three/three.min.js",
                "size_bytes": three_path.stat().st_size,
                "sha256": sha256(three_path),
            },
            "uncertainty_lod": lod_report,
        },
    }
    manifest_out = stage / "runtime_manifest.json"
    write_json(manifest_out, runtime_manifest)

    summary = {
        "schema": "proto2_viewer_products_v5_3",
        "outputs": {
            "animation": {key: file_record(path, project_root) for key, path in animation_outputs.items()},
            "color_scales": file_record(color_out, project_root),
            "viewer_tuning": file_record(tuning_out, project_root),
            "viewer_metadata": file_record(metadata_out, project_root),
            "runtime_manifest": file_record(manifest_out, project_root),
            "uncertainty_lod": lod_report,
        },
    }
    report = stage / "viewer_products_summary.json"
    write_json(report, summary)
    print_pass("STAGE 10 RESULT", report)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Pass3Error as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        raise SystemExit(1)
