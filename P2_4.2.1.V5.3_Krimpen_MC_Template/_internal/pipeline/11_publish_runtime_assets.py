#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
        "deterministic_total_f32.bin": "animation/deterministic_total",
        "sigma_h_f32.bin": "animation/sigma_h",
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
        "schema", "product_type", "vertices", "triangles", "moving_vertices", "blank_vertices",
        "total_parcels", "moving_parcels", "blank_parcels", "pick_features", "epochs", "epoch_labels",
        "center_lon", "center_lat", "center_height_m", "camera_height_m", "bounds_wgs84", "local_span_m", "stats",
    ]
    missing = [key for key in required if key not in metadata]
    if missing:
        raise Pass3Error(f"viewer_metadata missing required keys: {missing}")
    if int(metadata["moving_parcels"]) + int(metadata["blank_parcels"]) != int(metadata["total_parcels"]):
        raise Pass3Error("viewer_metadata parcel counts do not balance")
    if not isinstance(metadata["epoch_labels"], list) or len(metadata["epoch_labels"]) != int(metadata["epochs"]):
        raise Pass3Error("viewer_metadata epoch_labels length does not match epochs")


def collect_lod_files(stage_lod: Path) -> list[Path]:
    manifest_path = require_file(stage_lod / "uncertainty_lod_manifest.json", "uncertainty LOD manifest")
    manifest = validate_json(manifest_path, "uncertainty LOD manifest")
    families = manifest.get("lod_families")
    if not isinstance(families, dict):
        raise Pass3Error("uncertainty LOD manifest is missing lod_families")
    expected: list[Path] = [manifest_path]
    for family_key in ("detail", "overview"):
        family = families.get(family_key)
        tiles = family.get("tiles") if isinstance(family, dict) else None
        if not isinstance(tiles, list) or not tiles:
            raise Pass3Error(f"uncertainty LOD family {family_key!r} has no tiles")
        for tile in tiles:
            url = tile.get("url") if isinstance(tile, dict) else None
            if not isinstance(url, str) or not url:
                raise Pass3Error(f"uncertainty LOD {family_key} tile has no URL")
            stage_subdir = Path(url).parent.name
            path = require_file(stage_lod / stage_subdir / Path(url).name, f"uncertainty LOD tile {family_key}")
            expected.append(path)
    return expected


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate staged Proto2 runtime products and publish them atomically.")
    parser.add_argument("--publish", action="store_true", help="Atomically copy validated staged files into runtime locations.")
    args = parser.parse_args()

    root = project_root_from(__file__)
    config = load_project_config(root)
    stage = stage_root(root)
    runtime = semantic_root(root)
    print("\n=== PROTO2 STAGE 11: PUBLISH RUNTIME ASSETS ===")
    print(f"Project root : {root}")
    print(f"Stage root   : {stage}")
    print(f"Runtime root : {runtime}")

    checked: dict[str, dict[str, Any]] = {}
    copy_plan: list[tuple[Path, Path]] = []
    for folder, files in EXPECTED_STAGE_FILES.items():
        for filename, key in files.items():
            src = require_file(stage / folder / filename, key)
            dst = runtime / folder / filename
            checked[key] = file_record(src, root)
            copy_plan.append((src, dst))
            if src.suffix == ".json":
                validate_json(src, key)

    stage_lod = stage / "geometry" / "uncertainty_lod"
    lod_files = collect_lod_files(stage_lod)
    for src in lod_files:
        rel = src.relative_to(stage_lod)
        checked[f"geometry/uncertainty_lod/{rel.as_posix()}"] = file_record(src, root)

    runtime_manifest = validate_json(require_file(stage / "runtime_manifest.json", "runtime manifest"), "runtime manifest")
    viewer_metadata = validate_json(require_file(stage / "viewer_metadata.json", "viewer metadata"), "viewer metadata")
    validate_metadata(viewer_metadata)
    copy_plan += [(stage / "runtime_manifest.json", runtime / "runtime_manifest.json"), (stage / "viewer_metadata.json", runtime / "viewer_metadata.json")]
    checked["runtime_manifest"] = file_record(stage / "runtime_manifest.json", root)
    checked["viewer_metadata"] = file_record(stage / "viewer_metadata.json", root)

    published_files: dict[str, Any] = {}
    if args.publish:
        # Normal files are replaced atomically one by one.
        for src, dst in copy_plan:
            atomic_copy(src, dst)
            published_files[dst.relative_to(root).as_posix()] = file_record(dst, root)
        # LOD is a directory family. Remove stale families only after stage validation is complete.
        dst_lod = runtime / "geometry" / "uncertainty_lod"
        if dst_lod.exists():
            shutil.rmtree(dst_lod)
        shutil.copytree(stage_lod, dst_lod)
        for dst in sorted(dst_lod.rglob('*')):
            if dst.is_file():
                published_files[dst.relative_to(root).as_posix()] = file_record(dst, root)
        published = True
        print("[OK] staged runtime products published")
    else:
        published = False
        print("[OK] staged runtime products validated; --publish not supplied")

    record = {
        "schema": "proto2_runtime_publish_report_v5_3",
        "status": "PASS",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "published": published,
        "checked_stage_files": checked,
        "published_files": published_files,
        "lod_contract": "20m detail / 40m overview; camera-based exclusive switching",
    }
    out = stage_records_dir(root, config) / "runtime_publish_report.json"
    write_json(out, record)
    print("\n=== STAGE 11 RESULT ===")
    print("Status : PASS")
    print(f"Output : {out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Pass3Error as exc:
        print(f"[FAIL] {exc}")
        raise SystemExit(1)
