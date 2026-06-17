#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from _pass3_common import Pass3Error, atomic_copy, file_record, project_root_from, read_json, semantic_root, stage_root, write_json
from _proto2_config import load_project_config, stage_records_dir


EXPECTED_STAGE_FILES = {
    "geometry": {
        "parcel_caps.glb": "geometry/caps",
        "parcel_pistons.glb": "geometry/pistons",
        "parcel_walls.glb": "geometry/walls",
        "parcel_datum_caps_opaque.glb": "geometry/opaque_datum_caps",
    },
    "animation": {
        "reversible_f32.bin": "animation/reversible",
        "irreversible_f32.bin": "animation/irreversible",
        "total_f32.bin": "animation/total",
        "vi_f32.bin": "animation/vi",
    },
    "lookup": {
        "parcel_pick_index.json": "lookup/pick",
        "parcel_search_index.json": "lookup/search",
        "parcel_trendline_manifest.json": "lookup/trendline",
    },
    "style": {
        "parcel_color_scales.json": "style/color_scales",
        "viewer_tuning.json": "style/viewer_tuning",
    },
}


def require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise Pass3Error(f"Missing {label}: {path}")
    if path.stat().st_size <= 0:
        raise Pass3Error(f"Empty {label}: {path}")
    return path


def validate_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = read_json(path)
    except Exception as exc:
        raise Pass3Error(f"Invalid JSON for {label}: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise Pass3Error(f"JSON object expected for {label}: {path}")
    return payload


def validate_metadata(metadata: dict[str, Any]) -> None:
    required = [
        "schema",
        "product_type",
        "vertices",
        "triangles",
        "moving_vertices",
        "blank_vertices",
        "total_parcels",
        "moving_parcels",
        "blank_parcels",
        "pick_features",
        "epochs",
        "epoch_labels",
        "center_lon",
        "center_lat",
        "center_height_m",
        "camera_height_m",
        "bounds_wgs84",
        "local_span_m",
        "stats",
    ]
    missing = [key for key in required if key not in metadata]
    if missing:
        raise Pass3Error(f"viewer_metadata missing required keys: {missing}")
    if int(metadata["moving_parcels"]) + int(metadata["blank_parcels"]) != int(metadata["total_parcels"]):
        raise Pass3Error("viewer_metadata parcel counts do not balance")
    labels = metadata["epoch_labels"]
    if not isinstance(labels, list) or len(labels) != int(metadata["epochs"]):
        raise Pass3Error("viewer_metadata epoch_labels length does not match epochs")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate staged Proto2 runtime products and publish them without accepted-fixture parity."
    )
    parser.add_argument("--publish", action="store_true", help="Atomically copy validated staged files into semantic runtime locations.")
    args = parser.parse_args()

    root = project_root_from(__file__)
    stage = stage_root(root)
    semantic = semantic_root(root)

    print("\n=== PROTO2 GENERIC RUNTIME PUBLISH ===")
    print(f"Project root: {root}")
    print(f"Stage root  : {stage}")

    if not stage.exists():
        raise Pass3Error(f"Missing stage root: {stage}")

    checked: dict[str, dict[str, Any]] = {}
    copy_plan: list[tuple[Path, Path]] = []

    for folder, files in EXPECTED_STAGE_FILES.items():
        for filename, manifest_key in files.items():
            src = require_file(stage / folder / filename, manifest_key)
            dst = semantic / folder / filename
            checked[manifest_key] = file_record(src, root)
            copy_plan.append((src, dst))
            if src.suffix == ".json":
                validate_json(src, manifest_key)

    runtime_manifest = validate_json(require_file(stage / "runtime_manifest.json", "runtime_manifest"), "runtime_manifest")
    viewer_metadata = validate_json(require_file(stage / "viewer_metadata.json", "viewer_metadata"), "viewer_metadata")
    validate_metadata(viewer_metadata)

    checked["runtime_manifest"] = file_record(stage / "runtime_manifest.json", root)
    checked["viewer_metadata"] = file_record(stage / "viewer_metadata.json", root)
    copy_plan.extend([
        (stage / "runtime_manifest.json", semantic / "runtime_manifest.json"),
        (stage / "viewer_metadata.json", semantic / "viewer_metadata.json"),
    ])

    published_files: dict[str, Any] = {}
    if args.publish:
        for src, dst in copy_plan:
            atomic_copy(src, dst)
            published_files[dst.resolve().relative_to(root.resolve()).as_posix()] = file_record(dst, root)
        published = True
        print("[OK] staged assets published atomically")
    else:
        published = False
        print("[OK] staged assets validated; not published because --publish was not given")

    record = {
        "schema": "proto2_generic_runtime_publish_report_v1",
        "status": "PASS",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "published": published,
        "checked_stage_files": checked,
        "published_files": published_files,
        "note": (
            "Schema publish validates runtime completeness and internal manifest shape. "
            "It does not compare against the accepted Krimpenerwaard fixture."
        ),
    }
    config = load_project_config(root)
    out = stage_records_dir(root, config) / "runtime_publish_report.json"
    write_json(out, record)

    print("\n=== PROTO2 GENERIC RUNTIME PUBLISH RESULT ===")
    print("Status : PASS")
    print(f"Output : {out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Pass3Error as exc:
        print(f"[FAIL] {exc}")
        raise SystemExit(1)
