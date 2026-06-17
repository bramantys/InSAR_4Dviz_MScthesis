#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from _pass3_common import Pass3Error, file_record, project_root_from, read_json, sha256, write_json
from _proto2_config import load_project_config, stage_records_dir, viewer_tuning_path


def rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def as_project_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def check(condition: bool, detail: str = "") -> dict[str, Any]:
    return {"status": "PASS" if condition else "FAIL", "detail": detail}


def validate(project_root: Path) -> dict[str, Any]:
    config = load_project_config(project_root)
    paths = config.get("paths", {})
    config_path = project_root / "config" / "project_config.json"
    runtime_path = as_project_path(project_root, paths.get("runtime_manifest", "_internal/data_pipeline/runtime_manifest.json"))
    metadata_path = as_project_path(project_root, paths.get("viewer_metadata", "_internal/data_pipeline/viewer_metadata.json"))
    tuning_path = viewer_tuning_path(project_root, config)
    template_path = as_project_path(project_root, paths.get("template", "_internal/templates/viz2_template.html"))
    viewer_path = as_project_path(project_root, paths.get("output_viewer", "viz2_dev_v11.html"))
    records = stage_records_dir(project_root, config)
    runtime_publish_report_path = records / "runtime_publish_report.json"

    checks: dict[str, dict[str, Any]] = {}
    required_files = {
        "config.project_config": config_path,
        "runtime_manifest": runtime_path,
        "viewer_metadata": metadata_path,
        "viewer_tuning": tuning_path,
        "viewer_template": template_path,
        "viewer_html": viewer_path,
        "local_cesium": project_root / "_internal" / "cesium" / "Cesium.js",
        "local_cesium_css": project_root / "_internal" / "cesium" / "Widgets" / "widgets.css",
        "local_three": project_root / "_internal" / "three" / "three.min.js",
        "runtime_publish_report": runtime_publish_report_path,
    }
    for label, path in required_files.items():
        checks[label] = check(path.is_file(), rel(path, project_root))

    if any(item["status"] == "FAIL" for item in checks.values()):
        return {
            "schema": "proto2_release_validation_v2",
            "status": "FAIL",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "checks": checks,
        }

    runtime = read_json(runtime_path)
    metadata = read_json(metadata_path)
    publish_report = read_json(runtime_publish_report_path)

    required_runtime = {
        "geometry": ["caps", "pistons", "walls", "opaque_datum_caps"],
        "animation": ["reversible", "irreversible", "total", "vi"],
        "lookup": ["pick", "search", "trendline"],
        "style": ["color_scales", "viewer_tuning"],
    }
    for section, keys in required_runtime.items():
        block = runtime.get(section, {})
        checks[f"runtime.{section}.object"] = check(isinstance(block, dict))
        for key in keys:
            value = block.get(key)
            path = as_project_path(project_root, value) if isinstance(value, str) else project_root / "__missing__"
            checks[f"runtime.{section}.{key}"] = check(isinstance(value, str) and path.is_file(), rel(path, project_root))

    required_metadata = [
        "schema", "product_type", "vertices", "triangles", "moving_vertices", "blank_vertices",
        "total_parcels", "moving_parcels", "blank_parcels", "pick_features", "epochs", "epoch_labels",
        "center_lon", "center_lat", "center_height_m", "camera_height_m", "bounds_wgs84", "local_span_m", "stats",
    ]
    missing_metadata = [key for key in required_metadata if key not in metadata]
    checks["metadata.required_keys"] = check(not missing_metadata, str(missing_metadata))
    labels = metadata.get("epoch_labels")
    checks["metadata.epoch_labels"] = check(
        isinstance(labels, list) and len(labels) == int(metadata.get("epochs", -1)),
        f"epochs={metadata.get('epochs')} labels={len(labels) if isinstance(labels, list) else 'not-list'}",
    )
    checks["metadata.parcel_counts"] = check(
        int(metadata.get("moving_parcels", -1)) + int(metadata.get("blank_parcels", -1))
        == int(metadata.get("total_parcels", -2))
    )

    viewer_text = viewer_path.read_text(encoding="utf-8", errors="replace")
    checks["viewer.no_template_markers"] = check(
        "__PROTO2_BOOTSTRAP_JSON__" not in viewer_text and "{{PAGE_TITLE}}" not in viewer_text
    )
    forbidden = ["phase12_assets/", "phase14_color_assets/", "phase15_piston_assets/", "D:/Kuliah/", "C:/Users/"]
    found_forbidden = [value for value in forbidden if value in viewer_text]
    checks["viewer.no_historical_paths"] = check(not found_forbidden, str(found_forbidden))
    checks["viewer.local_cesium_reference"] = check("_internal/cesium/Cesium.js" in viewer_text)
    checks["viewer.local_three_reference"] = check("_internal/three/three.min.js" in viewer_text)

    checks["runtime_publish_report.status"] = check(publish_report.get("status") == "PASS", str(publish_report.get("status")))
    checks["runtime_publish_report.published"] = check(bool(publish_report.get("published")))
    for path_str, record in publish_report.get("published_files", {}).items():
        path = project_root / path_str
        actual_ok = (
            path.is_file()
            and path.stat().st_size == int(record.get("size_bytes", -1))
            and sha256(path) == record.get("sha256")
        )
        checks[f"published.{path_str}"] = check(actual_ok, rel(path, project_root))

    failures = [name for name, item in checks.items() if item["status"] == "FAIL"]
    return {
        "schema": "proto2_release_validation_v2",
        "status": "FAIL" if failures else "PASS",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "failures": failures,
        "checks": checks,
        "viewer": file_record(viewer_path, project_root),
    }


def main() -> int:
    project_root = project_root_from(__file__)
    config = load_project_config(project_root)
    record = validate(project_root)
    out = stage_records_dir(project_root, config) / "release_validation_report.json"
    write_json(out, record)

    print("\n=== PROTO2 RELEASE VALIDATION ===")
    print(f"Status : {record['status']}")
    print(f"Output : {out}")
    if record.get("failures"):
        for item in record["failures"]:
            print(f"  - {item}: {record['checks'][item]['detail']}")
    return 0 if record["status"] == "PASS" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Pass3Error as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        raise SystemExit(1)
