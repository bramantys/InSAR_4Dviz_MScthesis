#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import geopandas as gpd
import pandas as pd

from _pass3_common import (
    Pass3Error,
    clean_stage_area,
    file_record,
    load_legacy_module,
    print_pass,
    project_root_from,
    read_json,
    require,
    stage_root,
    write_json,
)


from _proto2_config import load_project_config, output_data_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Build staged Proto2 parcel lookup assets.")
    parser.parse_args()

    project_root = project_root_from(__file__)
    pipeline_dir = Path(__file__).resolve().parent
    config = load_project_config(project_root)
    output_data = output_data_dir(project_root, config)
    lookup_dir = clean_stage_area(project_root, "lookup")
    geometry_summary_path = require(stage_root(project_root) / "geometry_build_summary.json", "Pass3 geometry summary")
    geometry_summary = read_json(geometry_summary_path)
    cap_summary = geometry_summary["cap"]

    pick_out = lookup_dir / "parcel_pick_index.json"
    trend_out = lookup_dir / "parcel_trendline_manifest.json"
    search_out = lookup_dir / "parcel_search_index.json"

    print("\n=== PROTO2 PASS 3 / STAGE 07: BUILD LOOKUP ASSETS ===")

    legacy12 = load_legacy_module(pipeline_dir, "12_export_cesium_pickable_viewer.py", "proto2_legacy_phase12")
    parts = gpd.read_parquet(require(output_data / "parcel_footprints_parts.parquet", "footprint parts"))
    render_index = pd.read_parquet(require(output_data / "parcel_render_index.parquet", "parcel render index"))
    pick_index = legacy12.build_pick_index(parts, render_index, cap_summary)
    write_json(pick_out, pick_index, compact=True)

    legacy17 = load_legacy_module(pipeline_dir, "17_export_proto2_trendline_viewer.py", "proto2_legacy_phase17")
    trend_manifest: Dict[str, Any] = legacy17.build_trend_manifest()
    trend_manifest["binary_assets"] = {
        "reversible": "_internal/data_pipeline/animation/reversible_f32.bin",
        "irreversible": "_internal/data_pipeline/animation/irreversible_f32.bin",
        "total": "_internal/data_pipeline/animation/total_f32.bin",
    }
    trend_manifest["notes"] = [
        "Trendline reads the semantic runtime float32 component binaries.",
        "Combined chart uses one shared mm axis: irreversible + total, with fill between them as reversible gap.",
    ]
    write_json(trend_out, trend_manifest)

    legacy18 = load_legacy_module(pipeline_dir, "18_export_proto2_parcel_search_viewer.py", "proto2_legacy_phase18")
    search_index: Dict[str, Any] = legacy18.build_search_index(pick_out)
    search_index["source_pick_index"] = "_internal/data_pipeline/lookup/parcel_pick_index.json"
    write_json(search_out, search_index)

    summary = {
        "schema": "proto2_pass3_lookup_build_v1",
        "algorithm_source": [
            "12_export_cesium_pickable_viewer.py",
            "17_export_proto2_trendline_viewer.py",
            "18_export_proto2_parcel_search_viewer.py",
        ],
        "counts": {
            "pick_features": int(pick_index["metadata"]["feature_count"]),
            "search_parcels": int(search_index["parcel_count"]),
            "moving_parcels": int(search_index["moving_count"]),
            "blank_parcels": int(search_index["blank_count"]),
            "trendline_rows": len(trend_manifest.get("parcel_ids_in_row_order", [])),
            "epochs": int(trend_manifest.get("epoch_count", 0)),
        },
        "outputs": {
            "pick": file_record(pick_out, project_root),
            "search": file_record(search_out, project_root),
            "trendline": file_record(trend_out, project_root),
        },
    }
    report = stage_root(project_root) / "lookup_build_summary.json"
    write_json(report, summary)
    print_pass("STAGE 07 RESULT", report)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Pass3Error as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        raise SystemExit(1)
