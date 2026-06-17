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
    load_legacy_module,
    print_pass,
    project_root_from,
    read_json,
    require,
    sha256,
    stage_root,
    write_json,
)


from _proto2_config import load_project_config, output_data_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Build staged Proto2 animation, style, tuning, metadata and manifest products.")
    parser.parse_args()

    project_root = project_root_from(__file__)
    pipeline_dir = Path(__file__).resolve().parent
    config = load_project_config(project_root)
    output_data = output_data_dir(project_root, config)
    stage = stage_root(project_root)
    animation_dir = clean_stage_area(project_root, "animation")
    style_dir = clean_stage_area(project_root, "style")

    geometry_summary = read_json(require(stage / "geometry_build_summary.json", "geometry summary"))
    lookup_summary = read_json(require(stage / "lookup_build_summary.json", "lookup summary"))

    print("\n=== PROTO2 PASS 3 / STAGE 08: BUILD VIEWER PRODUCTS ===")

    legacy14 = load_legacy_module(pipeline_dir, "14_build_parcel_color_scales.py", "proto2_legacy_phase14a")
    try:
        legacy14.main()
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise Pass3Error(f"Legacy color-scale builder failed with exit code {exc.code}")

    animation_sources = {
        "reversible": require(output_data / "parcel_displacement_reversible_f32.bin", "reversible array"),
        "irreversible": require(output_data / "parcel_displacement_irreversible_f32.bin", "irreversible array"),
        "total": require(output_data / "parcel_displacement_total_f32.bin", "total array"),
        "vi": require(output_data / "parcel_vi_f32.bin", "irreversible velocity array"),
    }
    animation_outputs = {
        "reversible": animation_dir / "reversible_f32.bin",
        "irreversible": animation_dir / "irreversible_f32.bin",
        "total": animation_dir / "total_f32.bin",
        "vi": animation_dir / "vi_f32.bin",
    }
    for key, source in animation_sources.items():
        atomic_copy(source, animation_outputs[key])

    color_source = require(output_data / "parcel_color_scales.json", "parcel color scales")
    color_out = style_dir / "parcel_color_scales.json"
    atomic_copy(color_source, color_out)

    legacy15 = load_legacy_module(pipeline_dir, "15_build_irreversible_piston_assets.py", "proto2_legacy_phase15_tuning")
    tuning: Dict[str, Any] = legacy15.derive_display_tuning(legacy15.read_matrices_for_tuning())
    tuning_out = style_dir / "viewer_tuning.json"
    write_json(tuning_out, tuning)

    cap = geometry_summary["cap"]
    counts = lookup_summary["counts"]
    viewer_metadata = {
        "schema": "proto2_viewer_metadata_v1",
        "product_type": "parcel_deformation",
        "generated_from": "pass3_generic_builder_bridge",
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
    }
    metadata_out = stage / "viewer_metadata.json"
    write_json(metadata_out, viewer_metadata)

    three_path = require(project_root / "_internal" / "three" / "three.min.js", "local Three.js")
    runtime_manifest = {
        "schema": "proto2_runtime_manifest_v2",
        "product_type": "parcel_deformation",
        "geometry": {
            "caps": "_internal/data_pipeline/geometry/parcel_caps.glb",
            "pistons": "_internal/data_pipeline/geometry/parcel_pistons.glb",
            "walls": "_internal/data_pipeline/geometry/parcel_walls.glb",
            "opaque_datum_caps": "_internal/data_pipeline/geometry/parcel_datum_caps_opaque.glb",
        },
        "animation": {
            "reversible": "_internal/data_pipeline/animation/reversible_f32.bin",
            "irreversible": "_internal/data_pipeline/animation/irreversible_f32.bin",
            "total": "_internal/data_pipeline/animation/total_f32.bin",
            "vi": "_internal/data_pipeline/animation/vi_f32.bin",
        },
        "lookup": {
            "pick": "_internal/data_pipeline/lookup/parcel_pick_index.json",
            "search": "_internal/data_pipeline/lookup/parcel_search_index.json",
            "trendline": "_internal/data_pipeline/lookup/parcel_trendline_manifest.json",
        },
        "style": {
            "color_scales": "_internal/data_pipeline/style/parcel_color_scales.json",
            "viewer_tuning": "_internal/data_pipeline/style/viewer_tuning.json",
        },
        "runtime": {
            "three": {
                "mode": "local",
                "path": "_internal/three/three.min.js",
                "size_bytes": three_path.stat().st_size,
                "sha256": sha256(three_path),
            }
        },
    }
    manifest_out = stage / "runtime_manifest.json"
    write_json(manifest_out, runtime_manifest)

    summary = {
        "schema": "proto2_pass3_viewer_products_v1",
        "algorithm_source": [
            "05_package_animation_arrays.py outputs",
            "14_build_parcel_color_scales.py",
            "15_build_irreversible_piston_assets.py tuning logic",
        ],
        "outputs": {
            "animation": {key: file_record(path, project_root) for key, path in animation_outputs.items()},
            "color_scales": file_record(color_out, project_root),
            "viewer_tuning": file_record(tuning_out, project_root),
            "viewer_metadata": file_record(metadata_out, project_root),
            "runtime_manifest": file_record(manifest_out, project_root),
        },
    }
    report = stage / "viewer_products_summary.json"
    write_json(report, summary)
    print_pass("STAGE 08 RESULT", report)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Pass3Error as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        raise SystemExit(1)
