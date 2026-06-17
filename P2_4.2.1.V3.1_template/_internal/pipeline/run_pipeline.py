#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from _pass3_common import Pass3Error, project_root_from
from _proto2_config import load_project_config, output_data_dir, run_records_dir, runtime_root, stage_records_dir


PIPELINE_STEPS = [
    ("00", "00_phase0_sanity_check.py", ()),
    ("01", "01_adapt_parcel_displacement.py", ()),
    ("02", "02_ingest_parcels.py", ()),
    ("03", "03_prepare_parcel_footprints.py", ()),
    ("04", "04_triangulate_parcel_caps.py", ()),
    ("05", "05_package_animation_arrays.py", ()),
    ("06", "06_build_runtime_geometry.py", ()),
    ("07", "07_build_lookup_assets.py", ()),
    ("08", "08_build_viewer_products.py", ()),
    ("91", "91_publish_runtime_products.py", ("--publish",)),
    ("09", "09_assemble_viewer.py", ("--validate-files",)),
    ("99", "99_validate_release.py", ()),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def local_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def run_step(script: Path, step_args: tuple[str, ...], console_lines: list[str]) -> None:
    command = [sys.executable, str(script), *step_args]
    banner = "\n>>> " + " ".join(command)
    print(banner)
    console_lines.append(banner)

    process = subprocess.Popen(
        command,
        cwd=str(script.parents[2]),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        console_lines.append(line.rstrip("\n"))
    return_code = process.wait()
    if return_code != 0:
        raise Pass3Error(f"Step failed ({return_code}): {script.name}")


def clean_generated_work(project_root: Path, config: dict[str, Any]) -> None:
    root = runtime_root(project_root, config)
    for path in [root / "_work", root / "_stage"]:
        if path.exists():
            shutil.rmtree(path)
    output_data_dir(project_root, config).mkdir(parents=True, exist_ok=True)
    stage_records_dir(project_root, config).mkdir(parents=True, exist_ok=True)


def write_run_pair(
    project_root: Path,
    config: dict[str, Any],
    *,
    run_id: str,
    status: str,
    started_utc: str,
    finished_utc: str,
    stages_run: list[str],
    failed_stage: str | None,
    error_message: str | None,
    console_lines: list[str],
) -> tuple[Path, Path]:
    records_dir = run_records_dir(project_root, config)
    records_dir.mkdir(parents=True, exist_ok=True)

    stage_records = stage_records_dir(project_root, config)
    phase0 = read_json_if_exists(stage_records / "phase0_sanity_report.json") or {}
    metadata_path = runtime_root(project_root, config) / "viewer_metadata.json"
    metadata = read_json_if_exists(metadata_path) or {}

    input_block = config.get("user_inputs", {}) if isinstance(config.get("user_inputs"), dict) else {}
    summary = phase0.get("summary") if isinstance(phase0.get("summary"), dict) else {}
    if metadata:
        summary = {
            **summary,
            "total_parcels": metadata.get("total_parcels", summary.get("total_parcels")),
            "moving_parcels": metadata.get("moving_parcels", summary.get("moving_parcels")),
            "blank_parcels": metadata.get("blank_parcels", summary.get("blank_parcels")),
            "epochs": metadata.get("epochs", summary.get("epochs")),
            "epoch_start": metadata.get("epoch_start", summary.get("epoch_start")),
            "epoch_end": metadata.get("epoch_end", summary.get("epoch_end")),
        }

    viewer_rel = config.get("paths", {}).get("output_viewer", "viz2_dev_v11.html")
    viewer_path = project_root / viewer_rel
    record = {
        "schema": "proto2_user_run_record_v1",
        "run_id": run_id,
        "status": status,
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "failed_stage": failed_stage,
        "error": error_message,
        "active_deformation_source": config.get("pipeline_source", {}).get("deformation_source"),
        "user_inputs": input_block,
        "summary": summary,
        "viewer": {
            "path": viewer_rel,
            "exists": viewer_path.is_file(),
            "size_bytes": viewer_path.stat().st_size if viewer_path.is_file() else None,
        },
        "stages_run": stages_run,
    }

    json_path = records_dir / f"run_{run_id}.json"
    txt_path = records_dir / f"run_{run_id}.txt"
    json_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_lines = [
        "PROTO2 PIPELINE RUN RECORD",
        "",
        f"run id: {run_id}",
        f"status: {status}",
        f"started UTC: {started_utc}",
        f"finished UTC: {finished_utc}",
        f"failed stage: {failed_stage or '-'}",
        f"error: {error_message or '-'}",
        f"viewer: {viewer_rel}",
        "",
        "DATA SUMMARY",
        f"total parcels: {summary.get('total_parcels', '-')}",
        f"moving parcels: {summary.get('moving_parcels', '-')}",
        f"blank parcels: {summary.get('blank_parcels', '-')}",
        f"epochs: {summary.get('epochs', '-')}",
        f"epoch range: {summary.get('epoch_start', '-')} to {summary.get('epoch_end', '-')}",
        "",
        "FULL PIPELINE LOG",
        "=" * 80,
        *console_lines,
        "",
    ]
    txt_path.write_text("\n".join(str(value) for value in summary_lines), encoding="utf-8")
    return json_path, txt_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Proto2 bonestock user pipeline.")
    parser.add_argument(
        "--resume-from",
        choices=[code for code, _, _ in PIPELINE_STEPS],
        default="00",
        help="Developer recovery option. Normal users should run without arguments.",
    )
    args = parser.parse_args()

    project_root = project_root_from(__file__)
    config = load_project_config(project_root)
    pipeline_dir = Path(__file__).resolve().parent
    run_id = local_run_id()
    started_utc = utc_now()
    console_lines: list[str] = []
    stages_run: list[str] = []
    failed_stage: str | None = None
    error_message: str | None = None
    status = "FAIL"

    header = [
        "",
        "=== PROTO2 USER PIPELINE ===",
        f"Project root : {project_root}",
        f"Run ID       : {run_id}",
        f"Resume from  : {args.resume_from}",
        "Source mode  : displacement_csv",
    ]
    for line in header:
        print(line)
        console_lines.append(line)

    selected_index = next(index for index, (code, _, _) in enumerate(PIPELINE_STEPS) if code == args.resume_from)
    selected_steps = PIPELINE_STEPS[selected_index:]

    try:
        if args.resume_from == "00":
            clean_generated_work(project_root, config)

        for code, filename, step_args in selected_steps:
            failed_stage = code
            run_step(pipeline_dir / filename, step_args, console_lines)
            stages_run.append(filename + ((" " + " ".join(step_args)) if step_args else ""))
        status = "PASS"
        failed_stage = None
    except Exception as exc:
        error_message = str(exc)
        line = f"\n[FAIL] {error_message}"
        print(line, file=sys.stderr)
        console_lines.append(line)

    finished_utc = utc_now()
    json_path, txt_path = write_run_pair(
        project_root,
        config,
        run_id=run_id,
        status=status,
        started_utc=started_utc,
        finished_utc=finished_utc,
        stages_run=stages_run,
        failed_stage=failed_stage,
        error_message=error_message,
        console_lines=console_lines,
    )

    print("\n=== PROTO2 PIPELINE RESULT ===")
    print(f"Status : {status}")
    print(f"JSON   : {json_path}")
    print(f"TXT    : {txt_path}")
    if status == "PASS":
        print(f"Viewer : {project_root / config.get('paths', {}).get('output_viewer', 'viz2_dev_v11.html')}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
