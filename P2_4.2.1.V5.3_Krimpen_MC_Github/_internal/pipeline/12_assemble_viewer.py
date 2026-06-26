#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, NoReturn

from _proto2_config import load_project_config, stage_records_dir

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "project_config.json"
BOOTSTRAP_MARKER = "__PROTO2_BOOTSTRAP_JSON__"
PLACEHOLDERS = {
    "{{PAGE_TITLE}}": ("project", "page_title"),
    "{{CESIUM_BASE_URL}}": ("runtime_libraries", "cesium_base_url"),
    "{{CESIUM_JS_URL}}": ("runtime_libraries", "cesium_js"),
    "{{CESIUM_WIDGETS_CSS_URL}}": ("runtime_libraries", "cesium_widgets_css"),
    "{{THREE_JS_URL}}": ("runtime_libraries", "three_js"),
}


def fail(message: str) -> NoReturn:
    raise SystemExit(f"[FAIL] {message}")


def read_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        fail(f"Required JSON file not found: {path}")
    except json.JSONDecodeError as exc:
        fail(f"Invalid JSON in {path}: {exc}")
    if not isinstance(value, dict):
        fail(f"Expected a JSON object in {path}")
    return value


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def project_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def get_nested(mapping: Dict[str, Any], keys: Iterable[str]) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            fail(f"Missing configuration key: {'.'.join(keys)}")
        value = value[key]
    return value


def validate_runtime_contract(runtime: Dict[str, Any]) -> None:
    required = {
        "geometry": ("caps", "pistons", "walls", "opaque_datum_caps", "uncertainty_lod_manifest"),
        "animation": ("reversible", "irreversible", "total", "deterministic_total", "sigma_h", "vi"),
        "lookup": ("pick", "search", "trendline"),
        "style": ("color_scales", "viewer_tuning"),
    }
    for section, keys in required.items():
        block = runtime.get(section)
        if not isinstance(block, dict):
            fail(f"runtime_manifest.json is missing object: {section}")
        for key in keys:
            if not isinstance(block.get(key), str) or not block[key]:
                fail(f"runtime_manifest.json is missing path: {section}.{key}")


def build_uncertainty_lod_bootstrap(root: Path, runtime: Dict[str, Any]) -> Dict[str, Any]:
    geometry = runtime.get("geometry", {})
    manifest_value = geometry.get("uncertainty_lod_manifest") if isinstance(geometry, dict) else None
    if not isinstance(manifest_value, str) or not manifest_value:
        fail("runtime_manifest.json is missing path: geometry.uncertainty_lod_manifest")
    manifest_path = project_path(root, manifest_value)
    manifest = read_json(manifest_path)
    source_families = manifest.get("lod_families")
    if not isinstance(source_families, dict):
        fail("uncertainty LOD manifest is missing object: lod_families")
    viewer_config = manifest.get("viewer_lod_config")
    if not isinstance(viewer_config, dict):
        viewer_config = {}
    thresholds = viewer_config.get("thresholds")
    if not isinstance(thresholds, dict):
        thresholds = {
            "detail_enter_height_m": 10000.0,
            "overview_enter_height_m": 13000.0,
            "default_lod": "overview",
        }
    families: Dict[str, Any] = {}
    for family_key in ("detail", "overview"):
        source = source_families.get(family_key)
        if not isinstance(source, dict):
            fail(f"uncertainty LOD manifest is missing family: {family_key}")
        tiles = source.get("tiles")
        if not isinstance(tiles, list) or not tiles:
            fail(f"uncertainty LOD family has no tiles: {family_key}")
        clean_tiles = []
        for tile in tiles:
            if not isinstance(tile, dict) or not isinstance(tile.get("url"), str) or not tile["url"]:
                fail(f"uncertainty LOD tile has no URL: {family_key}")
            clean_tiles.append({
                "id": str(tile.get("tile_id", tile.get("id", "tile"))),
                "url": tile["url"],
                "featureCount": int(tile.get("feature_count", 0)),
                "visibleTriangles": int(tile.get("visible_triangles", 0)),
            })
        totals = source.get("totals") if isinstance(source.get("totals"), dict) else {}
        families[family_key] = {
            "label": str(source.get("label", family_key.title())),
            "spacingM": float(source.get("spacing_m", 0.0)),
            "featureCount": int(totals.get("features", 0)),
            "visibleTriangles": int(totals.get("visible_triangles", 0)),
            "glbBytes": int(totals.get("bytes", 0)),
            "tiles": clean_tiles,
        }
    return {
        "enabled": bool(viewer_config.get("enabled", True)),
        "manifestUrl": manifest_value,
        "thresholds": thresholds,
        "families": families,
        "source": "embedded_validated_uncertainty_lod_manifest",
    }


def validate_dataset_contract(dataset: Dict[str, Any]) -> None:
    required = (
        "epochs", "epoch_labels", "moving_parcels", "center_lon", "center_lat",
        "camera_height_m", "bounds_wgs84", "local_span_m", "stats",
    )
    for key in required:
        if key not in dataset:
            fail(f"viewer_metadata.json is missing: {key}")
    labels = dataset.get("epoch_labels")
    if not isinstance(labels, list) or len(labels) != int(dataset["epochs"]):
        fail("viewer_metadata epoch_labels length does not equal epochs")


def validate_files(root: Path, config: Dict[str, Any], runtime: Dict[str, Any]) -> None:
    file_paths: list[tuple[str, str]] = []
    libs = config.get("runtime_libraries", {})
    for key in ("cesium_js", "cesium_widgets_css", "three_js"):
        value = libs.get(key)
        if isinstance(value, str):
            file_paths.append((f"runtime_libraries.{key}", value))
    for section in ("geometry", "animation", "lookup", "style"):
        for key, value in runtime.get(section, {}).items():
            if isinstance(value, str):
                file_paths.append((f"runtime.{section}.{key}", value))
    missing = [f"{label}: {project_path(root, value)}" for label, value in file_paths if not project_path(root, value).is_file()]
    if missing:
        fail("Runtime file validation failed:\n  - " + "\n  - ".join(missing))


def optional_safety_patch(rendered: str) -> str:
    anchor = "    viewer.scene.globe.enableLighting = false;"
    insert = """
    // Browser/GPU safety: the parcel viewer does not need atmospheric scattering.
    if (viewer.scene.fog) viewer.scene.fog.enabled = false;
    if (viewer.scene.skyAtmosphere) viewer.scene.skyAtmosphere.show = false;
    if (viewer.scene.globe && \"showGroundAtmosphere\" in viewer.scene.globe) viewer.scene.globe.showGroundAtmosphere = false;
"""
    if anchor in rendered and "Browser/GPU safety" not in rendered:
        rendered = rendered.replace(anchor, anchor + insert, 1)
    return rendered


def assemble(config_path: Path, output_override: str | None, check_files: bool, dry_run: bool) -> Dict[str, Any]:
    root = PROJECT_ROOT
    config = read_json(config_path)
    paths = config.get("paths", {})
    template_path = project_path(root, paths.get("template", "_internal/templates/viz2_template.html"))
    runtime_path = project_path(root, paths.get("runtime_manifest", "_internal/data_pipeline/runtime/runtime_manifest.json"))
    metadata_path = project_path(root, paths.get("viewer_metadata", "_internal/data_pipeline/runtime/viewer_metadata.json"))
    tuning_path = project_path(root, paths.get("viewer_tuning", "_internal/data_pipeline/runtime/style/viewer_tuning.json"))
    output_rel = output_override or paths.get("output_viewer") or config.get("project", {}).get("output_viewer") or "viz2_parcel_viewer.html"
    output_path = project_path(root, output_rel)

    try:
        template = template_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        fail(f"Viewer template not found: {template_path}")
    runtime = read_json(runtime_path)
    dataset = read_json(metadata_path)
    tuning = read_json(tuning_path)
    validate_runtime_contract(runtime)
    uncertainty_lod = build_uncertainty_lod_bootstrap(root, runtime)
    validate_dataset_contract(dataset)
    if check_files:
        validate_files(root, config, runtime)

    bootstrap = {
        "schema": "proto2_viewer_bootstrap_v5_3",
        "project": config.get("project", {}),
        "viewer": config.get("viewer", {}),
        "dataset": dataset,
        "viewer_tuning": tuning,
        "time_settings": config.get("time_settings", {}),
        "monte_carlo_total": config.get("monte_carlo_total", {}),
        "runtime": {key: runtime[key] for key in ("geometry", "animation", "lookup", "style")},
        "uncertainty_lod": uncertainty_lod,
    }
    rendered = template
    if rendered.count(BOOTSTRAP_MARKER) != 1:
        fail(f"Template must contain exactly one {BOOTSTRAP_MARKER} marker")
    encoded = json.dumps(bootstrap, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    rendered = rendered.replace(BOOTSTRAP_MARKER, encoded, 1)
    for marker, key_path in PLACEHOLDERS.items():
        value = str(get_nested(config, key_path))
        if rendered.count(marker) != 1:
            fail(f"Template must contain exactly one {marker} marker")
        rendered = rendered.replace(marker, value, 1)
    unresolved = [marker for marker in [BOOTSTRAP_MARKER, *PLACEHOLDERS] if marker in rendered]
    if unresolved:
        fail("Unresolved template markers: " + ", ".join(unresolved))
    rendered = optional_safety_patch(rendered)
    forbidden = ("_internal/data_pipeline/work/", "_internal/build/", "phase12_assets/", "phase14_color_assets/", "phase15_piston_assets/", "D:/Kuliah/", "C:/Users/")
    found = [value for value in forbidden if value in rendered]
    if found:
        fail("Generated viewer contains forbidden historical paths: " + ", ".join(found))

    data = rendered.encode("utf-8")
    record = {
        "schema": "proto2_viewer_assembly_record_v5_3",
        "generated_utc": utc_now(),
        "template": template_path.relative_to(root).as_posix(),
        "runtime_manifest": runtime_path.relative_to(root).as_posix(),
        "viewer_metadata": metadata_path.relative_to(root).as_posix(),
        "viewer_tuning": tuning_path.relative_to(root).as_posix(),
        "output": output_path.relative_to(root).as_posix(),
        "size_bytes": len(data),
        "sha256": sha256_bytes(data),
        "runtime_file_validation": bool(check_files),
        "status": "DRY_RUN" if dry_run else "PASS",
    }
    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(data)
        records = stage_records_dir(root, config)
        records.mkdir(parents=True, exist_ok=True)
        (records / "viewer_assembly_manifest.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble the one final Proto2 parcel viewer.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", default=None)
    parser.add_argument("--validate-files", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    record = assemble(cfg.resolve(), args.output, args.validate_files, args.dry_run)
    print("\n=== PROTO2 STAGE 12: ASSEMBLE FINAL VIEWER ===")
    print(f"Status : {record['status']}")
    print(f"Output : {record['output']}")
    print(f"Size   : {record['size_bytes']} bytes")


if __name__ == "__main__":
    main()
