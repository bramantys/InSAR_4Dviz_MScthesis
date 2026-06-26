#!/usr/bin/env python3
from __future__ import annotations

import json
import re
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
    runtime_path = as_project_path(project_root, paths.get("runtime_manifest", "_internal/data_pipeline/runtime/runtime_manifest.json"))
    metadata_path = as_project_path(project_root, paths.get("viewer_metadata", "_internal/data_pipeline/runtime/viewer_metadata.json"))
    tuning_path = viewer_tuning_path(project_root, config)
    template_path = as_project_path(project_root, paths.get("template", "_internal/templates/viz2_template.html"))
    viewer_path = as_project_path(project_root, paths.get("output_viewer", "viz2_parcel_viewer.html"))
    records = stage_records_dir(project_root, config)
    publish_path = records / "runtime_publish_report.json"

    checks: dict[str, dict[str, Any]] = {}
    required = {
        "config": project_root / "config" / "project_config.json",
        "runtime_manifest": runtime_path,
        "viewer_metadata": metadata_path,
        "viewer_tuning": tuning_path,
        "viewer_template": template_path,
        "viewer_html": viewer_path,
        "local_cesium": project_root / "_internal" / "cesium" / "Cesium.js",
        "local_cesium_css": project_root / "_internal" / "cesium" / "Widgets" / "widgets.css",
        "local_three": project_root / "_internal" / "three" / "three.min.js",
        "runtime_publish_report": publish_path,
    }
    for label, path in required.items():
        checks[label] = check(path.is_file(), rel(path, project_root))
    if any(value["status"] == "FAIL" for value in checks.values()):
        return {"schema": "proto2_release_validation_v5_3", "status": "FAIL", "generated_utc": datetime.now(timezone.utc).isoformat(), "checks": checks, "failures": [k for k,v in checks.items() if v['status']=='FAIL']}

    runtime = read_json(runtime_path)
    metadata = read_json(metadata_path)
    publish = read_json(publish_path)
    required_runtime = {
        "geometry": ["caps", "pistons", "walls", "opaque_datum_caps", "uncertainty_lod_manifest"],
        "animation": ["reversible", "irreversible", "total", "deterministic_total", "sigma_h", "vi"],
        "lookup": ["pick", "search", "trendline"],
        "style": ["color_scales", "viewer_tuning"],
    }
    for section, keys in required_runtime.items():
        block = runtime.get(section)
        checks[f"runtime.{section}.object"] = check(isinstance(block, dict))
        if isinstance(block, dict):
            for key in keys:
                value = block.get(key)
                path = as_project_path(project_root, value) if isinstance(value, str) else project_root / '__missing__'
                checks[f"runtime.{section}.{key}"] = check(isinstance(value, str) and path.is_file(), rel(path, project_root))

    lod_manifest_path = as_project_path(project_root, runtime["geometry"]["uncertainty_lod_manifest"])
    if lod_manifest_path.is_file():
        lod = read_json(lod_manifest_path)
        for family_key in ("detail", "overview"):
            family = lod.get("lod_families", {}).get(family_key, {})
            tiles = family.get("tiles") if isinstance(family, dict) else None
            checks[f"lod.{family_key}.tiles"] = check(isinstance(tiles, list) and bool(tiles), f"count={len(tiles) if isinstance(tiles, list) else 0}")
            if isinstance(tiles, list):
                for tile in tiles:
                    url = tile.get("url") if isinstance(tile, dict) else None
                    path = as_project_path(project_root, url) if isinstance(url, str) else project_root / '__missing__'
                    checks[f"lod.{family_key}.{Path(url).name if isinstance(url,str) else 'missing'}"] = check(isinstance(url, str) and path.is_file(), rel(path, project_root))

    required_metadata = ["schema", "product_type", "total_parcels", "moving_parcels", "blank_parcels", "epochs", "epoch_labels", "center_lon", "center_lat"]
    missing = [key for key in required_metadata if key not in metadata]
    checks["metadata.required_keys"] = check(not missing, str(missing))
    checks["metadata.parcel_counts"] = check(int(metadata.get("moving_parcels", -1)) + int(metadata.get("blank_parcels", -1)) == int(metadata.get("total_parcels", -2)))
    labels = metadata.get("epoch_labels")
    checks["metadata.epoch_labels"] = check(isinstance(labels, list) and len(labels) == int(metadata.get("epochs", -1)))

    viewer_text = viewer_path.read_text(encoding="utf-8", errors="replace")
    checks["viewer.no_template_markers"] = check("__PROTO2_BOOTSTRAP_JSON__" not in viewer_text and "{{PAGE_TITLE}}" not in viewer_text)
    checks["viewer.runtime_root"] = check("_internal/data_pipeline/runtime/" in viewer_text)
    bootstrap_match = re.search(r"const BOOTSTRAP = (.*?);\s*const PROJECT", viewer_text, flags=re.DOTALL)
    embedded_lod: dict[str, Any] = {}
    if bootstrap_match:
        try:
            parsed_bootstrap = json.loads(bootstrap_match.group(1))
            candidate = parsed_bootstrap.get("uncertainty_lod") if isinstance(parsed_bootstrap, dict) else None
            embedded_lod = candidate if isinstance(candidate, dict) else {}
        except json.JSONDecodeError:
            embedded_lod = {}
    embedded_families = embedded_lod.get("families") if isinstance(embedded_lod.get("families"), dict) else {}
    embedded_ok = bool(embedded_lod.get("enabled")) and all(
        isinstance(embedded_families.get(family), dict)
        and isinstance(embedded_families[family].get("tiles"), list)
        and bool(embedded_families[family]["tiles"])
        for family in ("detail", "overview")
    )
    checks["viewer.embedded_uncertainty_lod"] = check(embedded_ok, "enabled with populated detail + overview tile families")
    checks["viewer.no_external_lod_config_sidechannel"] = check(
        re.search(r"<script\b[^>]*uncertainty_lod_runtime\.js", viewer_text, flags=re.IGNORECASE) is None
    )
    checks["viewer.lod_runtime"] = check("BOOTSTRAP.uncertainty_lod" in viewer_text and "loadGate3UncertaintyTiles" in viewer_text)
    forbidden = [value for value in ("_internal/data_pipeline/work/", "_internal/build/", "D:/Kuliah/", "C:/Users/") if value in viewer_text]
    checks["viewer.no_historical_paths"] = check(not forbidden, str(forbidden))

    checks["publish.status"] = check(publish.get("status") == "PASS", str(publish.get("status")))
    checks["publish.published"] = check(bool(publish.get("published")))
    for path_str, record in publish.get("published_files", {}).items():
        path = project_root / path_str
        ok = path.is_file() and path.stat().st_size == int(record.get("size_bytes", -1)) and sha256(path) == record.get("sha256")
        checks[f"published.{path_str}"] = check(ok, rel(path, project_root))

    failures = [name for name, item in checks.items() if item["status"] == "FAIL"]
    return {
        "schema": "proto2_release_validation_v5_3",
        "status": "FAIL" if failures else "PASS",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "failures": failures,
        "checks": checks,
        "viewer": file_record(viewer_path, project_root),
    }


def main() -> int:
    root = project_root_from(__file__)
    config = load_project_config(root)
    record = validate(root)
    out = stage_records_dir(root, config) / "release_validation_report.json"
    write_json(out, record)
    print("\n=== PROTO2 STAGE 13: RELEASE VALIDATION ===")
    print(f"Status : {record['status']}")
    print(f"Output : {out}")
    for item in record.get("failures", []):
        print(f"  - {item}: {record['checks'][item]['detail']}")
    return 0 if record["status"] == "PASS" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Pass3Error as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        raise SystemExit(1)
