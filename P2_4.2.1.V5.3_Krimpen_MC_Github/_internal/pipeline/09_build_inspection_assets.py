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
    print_pass,
    project_root_from,
    read_json,
    require,
    stage_root,
    write_json,
)


from _proto2_config import load_project_config, output_data_dir
import _pick_index_support as pick_support
import _trendline_support as trend_support
import _search_index_support as search_support


def main() -> int:
    parser = argparse.ArgumentParser(description="Build staged Proto2 parcel lookup assets.")
    parser.parse_args()

    project_root = project_root_from(__file__)
    config = load_project_config(project_root)
    output_data = output_data_dir(project_root, config)
    lookup_dir = clean_stage_area(project_root, "lookup")
    geometry_summary_path = require(stage_root(project_root) / "geometry_build_summary.json", "Pass3 geometry summary")
    geometry_summary = read_json(geometry_summary_path)
    cap_summary = geometry_summary["cap"]

    pick_out = lookup_dir / "parcel_pick_index.json"
    trend_out = lookup_dir / "parcel_trendline_manifest.json"
    search_out = lookup_dir / "parcel_search_index.json"

    print("\n=== PROTO2 STAGE 09: BUILD INSPECTION ASSETS ===")

    parts = gpd.read_parquet(require(output_data / "parcel_footprints_parts.parquet", "footprint parts"))
    render_index = pd.read_parquet(require(output_data / "parcel_render_index.parquet", "parcel render index"))
    pick_index = pick_support.build_pick_index(parts, render_index, cap_summary)
    write_json(pick_out, pick_index, compact=True)

    trend_manifest: Dict[str, Any] = trend_support.build_trend_manifest()
    trend_manifest["binary_assets"] = {
        "reversible": "_internal/data_pipeline/runtime/animation/reversible_f32.bin",
        "irreversible": "_internal/data_pipeline/runtime/animation/irreversible_f32.bin",
        "total": "_internal/data_pipeline/runtime/animation/total_f32.bin",
        "deterministic_total": "_internal/data_pipeline/runtime/animation/deterministic_total_f32.bin",
        "sigma_h": "_internal/data_pipeline/runtime/animation/sigma_h_f32.bin",
    }
    trend_manifest["notes"] = [
        "Total reads supplied\'s Monte Carlo mean_t runtime binary.",
        "sigma_h is supplied\'s matching MC per-epoch standard deviation; it is supplied for Total only.",
        "The deterministic reversible and irreversible components are calculated directly across the configured viewer period.",
    ]
    write_json(trend_out, trend_manifest)

    search_index: Dict[str, Any] = search_support.build_search_index(pick_out)
    search_index["source_pick_index"] = "_internal/data_pipeline/runtime/lookup/parcel_pick_index.json"
    write_json(search_out, search_index)

    summary = {
        "schema": "proto2_pass3_lookup_build_v1",
        "algorithm_source": [
            "_pick_index_support.py",
            "_trendline_support.py",
            "_search_index_support.py",
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
    print_pass("STAGE 09 RESULT", report)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Pass3Error as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        raise SystemExit(1)
