#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Assemble the dataset-neutral Proto2 parcel viewer from controlled manifests.

Pass 4 production name for the assembler formerly introduced as 90_assemble_viewer.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

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


def fail(message: str) -> "NoReturn":
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
        "geometry": ("caps", "pistons", "walls", "opaque_datum_caps"),
        "animation": ("reversible", "irreversible", "total", "vi"),
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


def patch_datum_reference_behavior(rendered: str) -> str:
    """Apply the V3 datum-reference fix to the assembled viewer.

    The reference model intentionally remains loaded in Reversible/Combined modes
    because it also carries blank/no-data parcels. Moving reference parcels are
    hidden in the shader, while blank parcels remain visible for context.
    """
    datum_url_old = "const DATUM_CAP_GLB_URL = RUNTIME.geometry.opaque_datum_caps;"
    datum_url_new = (
        "// Datum/reference caps must use the BLEND-cap asset so shader alpha and the toggle work.\n"
        "const DATUM_CAP_GLB_URL = RUNTIME.geometry.caps;"
    )
    if rendered.count(datum_url_old) != 1:
        fail("Could not find the datum-cap runtime URL exactly once in the viewer template")
    rendered = rendered.replace(datum_url_old, datum_url_new, 1)

    toggle_old = (
        "            datumReferenceEnabled = datumReferenceToggle.checked;\n"
        "            setModelVisibilityForMode();"
    )
    toggle_new = (
        "            datumReferenceEnabled = datumReferenceToggle.checked;\n"
        "            // Refresh the shader uniform immediately; visibility alone does not update it.\n"
        "            setShaderTimeAndScale();\n"
        "            setModelVisibilityForMode();"
    )
    if rendered.count(toggle_old) != 1:
        fail("Could not find the datum-reference toggle handler exactly once")
    rendered = rendered.replace(toggle_old, toggle_new, 1)

    final_combined_old = """            if (u_capRole > 1.5) {
              // Combined irreversible cap: always visible, hard-opaque, displacement-coloured.
              float irr = sampleTimeComponentFrag(u_irreversibleTex, rowIndex);
              material.diffuse = irreversibleDisplacementLocalColorValue(irr);
              material.alpha = 1.0;
              return;
            }

            if (u_datumReferenceEnabled < 0.5) {
              material.diffuse = vec3(0.96, 0.98, 1.0);
              material.alpha = 0.0;
              return;
            }"""
    final_combined_new = """            // Moving datum/reference parcels obey the toggle in both Reversible and Combined modes.
            // Blank/no-data parcels returned above remain visible for spatial context.
            if (u_datumReferenceEnabled < 0.5) {
              material.diffuse = vec3(0.96, 0.98, 1.0);
              material.alpha = 0.0;
              return;
            }

            if (u_capRole > 1.5) {
              // Combined irreversible datum cap: displacement-coloured and slightly translucent.
              float irr = sampleTimeComponentFrag(u_irreversibleTex, rowIndex);
              material.diffuse = irreversibleDisplacementLocalColorValue(irr);
              material.alpha = 0.85;
              return;
            }"""
    if rendered.count(final_combined_old) != 1:
        fail("Could not find the final Combined datum-cap shader branch exactly once")
    rendered = rendered.replace(final_combined_old, final_combined_new, 1)

    # The template contains both an inherited and a final reference-cap shader.
    # Keeping both values aligned avoids the bug returning during later cleanup.
    reference_alpha_count = rendered.count("material.alpha = 0.50;")
    if reference_alpha_count < 1:
        fail("Could not find the datum-reference alpha in the viewer template")
    rendered = rendered.replace("material.alpha = 0.50;", "material.alpha = 0.85;")

    inherited_combined_old = """              // Combined irreversible cap: opaque and displacement-coloured.
              float irr = sampleTimeComponentFrag(u_irreversibleTex, rowIndex);
              material.diffuse = rampTotal(irr);
              material.alpha = 1.0;"""
    inherited_combined_new = """              // Combined irreversible datum cap: slightly translucent.
              float irr = sampleTimeComponentFrag(u_irreversibleTex, rowIndex);
              material.diffuse = rampTotal(irr);
              material.alpha = 0.85;"""
    if inherited_combined_old in rendered:
        rendered = rendered.replace(inherited_combined_old, inherited_combined_new, 1)

    return rendered


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
    file_paths = []
    libs = config.get("runtime_libraries", {})
    for key in ("cesium_js", "cesium_widgets_css", "three_js"):
        value = libs.get(key)
        if isinstance(value, str):
            file_paths.append((f"runtime_libraries.{key}", value))

    for section in ("geometry", "animation", "lookup", "style"):
        for key, value in runtime.get(section, {}).items():
            if isinstance(value, str):
                file_paths.append((f"runtime.{section}.{key}", value))

    missing = []
    for label, value in file_paths:
        path = project_path(root, value)
        if not path.is_file():
            missing.append(f"{label}: {path}")
    if missing:
        fail("Runtime file validation failed:\n  - " + "\n  - ".join(missing))


def assemble(config_path: Path, output_override: str | None, check_files: bool, dry_run: bool) -> Dict[str, Any]:
    root = PROJECT_ROOT
    config = read_json(config_path)

    paths = config.get("paths", {})
    template_path = project_path(root, paths.get("template", "_internal/templates/viz2_template.html"))
    runtime_path = project_path(root, paths.get("runtime_manifest", "_internal/data_pipeline/runtime_manifest.json"))
    dataset_path = project_path(root, paths.get("viewer_metadata", "_internal/data_pipeline/viewer_metadata.json"))
    tuning_path = project_path(root, paths.get("viewer_tuning", "_internal/data_pipeline/style/viewer_tuning.json"))
    output_rel = output_override or paths.get("output_viewer") or config.get("project", {}).get("output_viewer") or "viz2_dev_v11.html"
    output_path = project_path(root, output_rel)

    try:
        template = template_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        fail(f"Viewer template not found: {template_path}")

    runtime = read_json(runtime_path)
    dataset = read_json(dataset_path)
    tuning = read_json(tuning_path)
    validate_runtime_contract(runtime)
    validate_dataset_contract(dataset)
    if check_files:
        validate_files(root, config, runtime)

    bootstrap = {
        "schema": "proto2_viewer_bootstrap_v1",
        "project": config.get("project", {}),
        "viewer": config.get("viewer", {}),
        "dataset": dataset,
        "viewer_tuning": tuning,
        "runtime": {
            "geometry": runtime["geometry"],
            "animation": runtime["animation"],
            "lookup": runtime["lookup"],
            "style": runtime["style"],
        },
    }

    rendered = template
    if rendered.count(BOOTSTRAP_MARKER) != 1:
        fail(f"Template must contain exactly one {BOOTSTRAP_MARKER} marker")
    bootstrap_json = json.dumps(bootstrap, ensure_ascii=False, separators=(",", ":"))
    bootstrap_json = bootstrap_json.replace("</", "<\\/")
    rendered = rendered.replace(BOOTSTRAP_MARKER, bootstrap_json, 1)

    for marker, key_path in PLACEHOLDERS.items():
        value = str(get_nested(config, key_path))
        if rendered.count(marker) != 1:
            fail(f"Template must contain exactly one {marker} marker")
        rendered = rendered.replace(marker, value, 1)

    unresolved = [marker for marker in [BOOTSTRAP_MARKER, *PLACEHOLDERS] if marker in rendered]
    if unresolved:
        fail("Unresolved template markers: " + ", ".join(unresolved))

    # V3 datum-reference fix: preserve blank parcels while allowing the moving
    # datum surface to hide/show immediately with 0.85 visible opacity.
    rendered = patch_datum_reference_behavior(rendered)


    # Browser/GPU safety shim:
    #
    # Some Windows/browser/GPU combinations fail Cesium's ground-atmosphere
    # fragment shader with a vague "compile log: null".  The parcel viewer does
    # not need atmospheric scattering, so disable it deterministically in the
    # generated HTML.  This does not change deformation geometry, colors,
    # animation arrays, picking, trendlines, or any scientific runtime asset.
    webgl_safety_insert = """
    // Pass6 browser/GPU safety: this parcel viewer does not need atmospheric scattering.
    // Disabling it avoids Cesium ground-atmosphere fragment shader failures on some drivers.
    if (viewer.scene.fog) viewer.scene.fog.enabled = false;
    if (viewer.scene.skyAtmosphere) viewer.scene.skyAtmosphere.show = false;
    if (viewer.scene.globe && "showGroundAtmosphere" in viewer.scene.globe) viewer.scene.globe.showGroundAtmosphere = false;
    """
    webgl_safety_anchor = "    viewer.scene.globe.enableLighting = false;"
    if webgl_safety_anchor in rendered and "Pass6 browser/GPU safety" not in rendered:
        rendered = rendered.replace(webgl_safety_anchor, webgl_safety_anchor + webgl_safety_insert, 1)

    forbidden = ("phase12_assets/", "phase14_color_assets/", "phase15_piston_assets/", "D:/Kuliah/", "C:/Users/")
    found_forbidden = [value for value in forbidden if value in rendered]
    if found_forbidden:
        fail("Generated viewer contains forbidden historical paths: " + ", ".join(found_forbidden))

    output_bytes = rendered.encode("utf-8")
    record = {
        "schema": "proto2_viewer_assembly_record_v1",
        "generated_utc": utc_now(),
        "template": str(template_path.relative_to(root)).replace("\\", "/"),
        "runtime_manifest": str(runtime_path.relative_to(root)).replace("\\", "/"),
        "viewer_metadata": str(dataset_path.relative_to(root)).replace("\\", "/"),
        "viewer_tuning": str(tuning_path.relative_to(root)).replace("\\", "/"),
        "output": str(output_path.relative_to(root)).replace("\\", "/"),
        "size_bytes": len(output_bytes),
        "sha256": sha256_bytes(output_bytes),
        "runtime_file_validation": bool(check_files),
        "status": "DRY_RUN" if dry_run else "PASS",
    }

    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(output_bytes)
        record_path = stage_records_dir(root, config) / "viewer_assembly_manifest.json"
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble viz2_dev_v11.html from the Proto2 template and manifests.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", default=None, help="Override output path relative to project root")
    parser.add_argument("--validate-files", action="store_true", help="Check that all runtime assets exist")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    record = assemble(config_path.resolve(), args.output, args.validate_files, args.dry_run)
    print("\n=== PROTO2 VIEWER ASSEMBLY ===")
    print(f"Status : {record['status']}")
    print(f"Output : {record['output']}")
    print(f"Size   : {record['size_bytes']} bytes")
    print(f"SHA256 : {record['sha256']}")


if __name__ == "__main__":
    main()
