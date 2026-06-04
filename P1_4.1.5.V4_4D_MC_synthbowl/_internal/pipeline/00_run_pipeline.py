#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
00_run_pipeline.py

Pipeline runner for the InSAR4D RUM Viewer package.

This is the executable orchestration script called by:
  config/run_baby_run.bat

Purpose
-------
- Load the simple user config through pipeline_config.py
- Prepare/clean generated output folders
- Run pipeline steps 01..18 in order
- Capture stdout/stderr from each step
- Classify each step as OK / WARN / FAIL / MISSING / SKIPPED
- Write:
    run_records/latest_run.log
    run_records/latest_manifest.json
    run_records/archive/<run_id>_run.log
    run_records/archive/<run_id>_manifest.json

Notes
-----
This runner intentionally captures full console output because during
development some steps only need an "OK", while others produce useful
diagnostic details that should be preserved in the log.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pipeline_config import (
    build_run_metadata,
    clean_pipeline_output,
    collect_existing_outputs,
    ensure_standard_dirs,
    load_resolved_config,
    resolve_path,
    write_json,
)


# =============================================================================
# ADVANCED SETTINGS
# =============================================================================
# Normal users should not need to edit this.
# Edit only when the numbered pipeline script names change.

PIPELINE_STEPS: List[Dict[str, Any]] = [
    {
        "step": "01",
        "script": "01_convert_source_to_geojson.py",
        "title": "Convert source table to WGS84 GeoJSON",
        "importance": "core",
    },
    {
        "step": "02",
        "script": "02_prepare_vertical_epochs_from_velocity.py",
        "title": "Prepare synthetic vertical measurement/model epochs from velocity",
        "importance": "core",
    },
    {
        "step": "03",
        "script": "03_build_footprints.py",
        "title": "Build corrected RUM footprints",
        "importance": "core",
    },
    {
        "step": "04",
        "script": "04_enhance_vertical_sigma.py",
        "fallback_scripts": ["04_enhance_vertical_sigma_optional.py"],
        "title": "Enhance vertical sigma / uncertainty field",
        "importance": "core",
    },
    {
        "step": "05",
        "script": "05_validate_prepared_inputs.py",
        "title": "Validate prepared inputs",
        "importance": "validation",
    },
    {
        "step": "06",
        "script": "06_pack_vertical_series.py",
        "title": "Pack vertical measurement/model and sigma series",
        "importance": "core",
    },
    {
        "step": "07",
        "script": "07_build_tile_index.py",
        "fallback_scripts": ["08_build_tile_index.py"],
        "title": "Build spatial tile index",
        "importance": "core",
    },
    {
        "step": "08",
        "script": "08_export_epoch_axis.py",
        "fallback_scripts": ["09_export_epoch_axis.py"],
        "title": "Export epoch axis",
        "importance": "core",
    },
    {
        "step": "09",
        "script": "09_build_blank_cells.py",
        "fallback_scripts": ["10_build_blank_cells.py"],
        "title": "Build blank/no-data cells",
        "importance": "core",
    },
    {
        "step": "10",
        "script": "10_build_height_texture.py",
        "fallback_scripts": ["11_build_height_texture.py"],
        "title": "Build height/sigma texture",
        "importance": "core",
    },
    {
        "step": "11",
        "script": "11_build_real_caps_b3dm.py",
        "fallback_scripts": ["12_build_real_caps_b3dm.py"],
        "title": "Build real cap B3DM tiles",
        "importance": "core",
    },
    {
        "step": "12",
        "script": "12_build_blank_caps_b3dm.py",
        "fallback_scripts": ["13_build_blank_caps_b3dm.py"],
        "title": "Build blank cap B3DM tiles",
        "importance": "core",
    },
    {
        "step": "13",
        "script": "13_build_walls_b3dm.py",
        "fallback_scripts": ["14_build_walls_b3dm.py"],
        "title": "Build wall B3DM tiles",
        "importance": "core",
    },
    {
        "step": "14",
        "script": "14_build_horizontal_field.py",
        "fallback_scripts": ["16_build_horizontal_field.py"],
        "title": "Build horizontal velocity field",
        "importance": "core",
    },
    {
        "step": "15",
        "script": "15_check_horizontal_field.py",
        "fallback_scripts": ["17_check_horizontal_field.py"],
        "title": "Check horizontal field",
        "importance": "dev_validation",
    },
    {
        "step": "16",
        "script": "16_check_horizontal_uncertainty.py",
        "fallback_scripts": ["18_check_horizontal_uncertainty.py"],
        "title": "Check horizontal uncertainty",
        "importance": "dev_validation",
    },
    {
        "step": "17",
        "script": "17_build_horizontal_arrow_ellipse_b3dm.py",
        "fallback_scripts": ["19_build_horizontal_arrow_ellipse_layers.py"],
        "title": "Build horizontal arrows and confidence ellipses",
        "importance": "dev_product",
    },
    {
        "step": "18",
        "script": "18_build_viewer_tuning.py",
        "title": "Build data-derived viewer tuning",
        "importance": "viewer_product",
    },
]

# Lines containing these patterns are harvested into manifest["warnings"].
WARNING_PATTERNS = [
    r"\[WARN\]",
    r"\bWARN\b",
    r"\bWARNING\b",
    r"UserWarning",
]

# Lines containing these patterns are harvested into manifest["errors"].
ERROR_PATTERNS = [
    r"\[FAIL\]",
    r"\bFAIL\b",
    r"\bERROR\b",
    r"Traceback \(most recent call last\)",
    r"Exception:",
    r"FileNotFoundError",
    r"ValueError",
    r"RuntimeError",
]


# =============================================================================
# SMALL HELPERS
# =============================================================================

def now_local() -> dt.datetime:
    return dt.datetime.now().astimezone()


def make_run_id(ts: Optional[dt.datetime] = None) -> str:
    t = ts or now_local()
    return t.strftime("%Y%m%d_%H%M%S")


def iso(ts: dt.datetime) -> str:
    return ts.isoformat(timespec="seconds")


def seconds(value: float) -> float:
    return round(float(value), 3)


def read_text_safe(path: Path, max_chars: int = 4000) -> str:
    try:
        if path.exists() and path.is_file():
            text = path.read_text(encoding="utf-8", errors="replace")
            if len(text) > max_chars:
                return text[:max_chars] + "\n...[truncated]..."
            return text
    except Exception:
        pass
    return ""


def line_has_any(line: str, patterns: List[str]) -> bool:
    return any(re.search(pattern, line, flags=re.IGNORECASE) for pattern in patterns)


def extract_warning_error_lines(text: str) -> Tuple[List[str], List[str]]:
    warnings: List[str] = []
    errors: List[str] = []

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line_has_any(line, WARNING_PATTERNS):
            warnings.append(line)
        if line_has_any(line, ERROR_PATTERNS):
            errors.append(line)

    return warnings, errors


def strip_ansi(text: str) -> str:
    # Remove ANSI escape codes if any are emitted by dependencies.
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def find_step_script(pipeline_dir: Path, step: Dict[str, Any]) -> Tuple[Optional[Path], Optional[str]]:
    """
    Return the first existing script path and a note if fallback was used.
    """
    primary = pipeline_dir / step["script"]
    if primary.exists():
        return primary, None

    for fallback_name in step.get("fallback_scripts", []):
        fallback = pipeline_dir / fallback_name
        if fallback.exists():
            return fallback, f"primary missing; used fallback {fallback_name}"

    return None, None


def should_run_step(step_no: str, from_step: Optional[str], to_step: Optional[str]) -> bool:
    n = int(step_no)
    if from_step is not None and n < int(from_step):
        return False
    if to_step is not None and n > int(to_step):
        return False
    return True


def format_duration(duration_s: Optional[float]) -> str:
    if duration_s is None:
        return "-"
    if duration_s < 60:
        return f"{duration_s:.2f}s"
    return f"{duration_s/60.0:.2f}min"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# =============================================================================
# STEP EXECUTION
# =============================================================================

def run_one_step(
    step: Dict[str, Any],
    script_path: Path,
    project_root: Path,
    config_path: Path,
    extra_args: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run a single pipeline script as a subprocess.

    The config path is passed as --config if the script accepts it. Older scripts
    that do not define --config may fail with "unrecognized arguments", so we do
    not force --config by default. Scripts should find config/project_config.json
    via pipeline_config after patching.
    """
    started = now_local()
    t0 = time.time()

    command = [sys.executable, str(script_path)]
    if extra_args:
        command.extend(extra_args)

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    completed = subprocess.run(
        command,
        cwd=str(project_root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )

    ended = now_local()
    duration_s = seconds(time.time() - t0)

    stdout = strip_ansi(completed.stdout or "")
    stderr = strip_ansi(completed.stderr or "")
    combined = "\n".join(part for part in [stdout, stderr] if part.strip())

    warnings, errors = extract_warning_error_lines(combined)

    if completed.returncode == 0:
        status = "WARN" if warnings else "OK"
    else:
        status = "FAIL"

    return {
        "step": step["step"],
        "script": script_path.name,
        "title": step["title"],
        "importance": step.get("importance", ""),
        "status": status,
        "returncode": completed.returncode,
        "started_at": iso(started),
        "ended_at": iso(ended),
        "duration_seconds": duration_s,
        "command": command,
        "warnings": warnings,
        "errors": errors,
        "stdout": stdout,
        "stderr": stderr,
    }


# =============================================================================
# LOG / MANIFEST BUILDING
# =============================================================================

def build_step_table(step_results: List[Dict[str, Any]]) -> str:
    lines = []
    lines.append("Step summary")
    lines.append("-" * 100)
    lines.append(f"{'Step':<6} {'Status':<8} {'Duration':<10} {'Script':<42} Title")
    lines.append("-" * 100)

    for r in step_results:
        lines.append(
            f"{r['step']:<6} {r['status']:<8} "
            f"{format_duration(r.get('duration_seconds')):<10} "
            f"{r.get('script', '-'):<42} {r.get('title', '')}"
        )

    lines.append("-" * 100)
    return "\n".join(lines)


def build_human_log(manifest: Dict[str, Any]) -> str:
    meta = manifest["run_metadata"]
    lines: List[str] = []

    lines.append("=" * 100)
    lines.append("InSAR4D RUM Viewer Pipeline Run")
    lines.append("=" * 100)
    lines.append(f"Run ID              : {manifest['run_id']}")
    lines.append(f"Status              : {manifest['status']}")
    lines.append(f"Started             : {manifest['started_at']}")
    lines.append(f"Ended               : {manifest['ended_at']}")
    lines.append(f"Duration            : {format_duration(manifest['duration_seconds'])}")
    lines.append("")
    lines.append("Configuration")
    lines.append("-" * 100)
    lines.append(f"Project key         : {meta.get('project_key')}")
    lines.append(f"Dataset title       : {meta.get('dataset_title')}")
    lines.append(f"Source file         : {meta.get('source_file')}")
    lines.append(f"Source CRS          : {meta.get('source_crs')}")
    lines.append(f"RUM size            : {meta.get('rum_size_m')} m")
    lines.append(f"Expected RUM count  : {meta.get('expected_rum_count')}")
    se = meta.get("synthetic_epochs", {})
    lines.append(f"Measurement behavior: {se.get('vertical_measurement_behavior')}")
    lines.append(f"Measurement noise   : {se.get('vertical_measurement_noise')} "
                 f"({se.get('vertical_measurement_noise_sigma_mm')} mm)")
    lines.append(f"Vertical model      : {se.get('vertical_model')}")
    lines.append(f"Uncertainty quality : {se.get('uncertainty_quality')}")
    lines.append(f"Epoch start         : {se.get('start_date')}")
    lines.append(f"Epoch end           : {se.get('end_date')}")
    lines.append(f"Epoch interval      : {se.get('interval_days')} days")
    lines.append(f"Expected epochs     : {se.get('expected_epoch_count')}")
    lines.append("")
    lines.append(build_step_table(manifest["steps"]))
    lines.append("")

    if manifest["warnings"]:
        lines.append("Warnings")
        lines.append("-" * 100)
        for w in manifest["warnings"]:
            lines.append(f"- [{w.get('step', 'runner')}] {w.get('message')}")
        lines.append("")

    if manifest["errors"]:
        lines.append("Errors")
        lines.append("-" * 100)
        for e in manifest["errors"]:
            lines.append(f"- [{e.get('step', 'runner')}] {e.get('message')}")
        lines.append("")

    lines.append("Generated output summary")
    lines.append("-" * 100)
    for key, item in manifest.get("outputs", {}).items():
        exists = "yes" if item.get("exists") else "no"
        size = item.get("size_bytes")
        size_txt = f"{size} bytes" if size is not None else "-"
        lines.append(f"{key:<42} exists={exists:<3} size={size_txt:<14} path={item.get('path')}")
    lines.append("")

    lines.append("Full step output")
    lines.append("=" * 100)

    for r in manifest["steps"]:
        lines.append("")
        lines.append(f"[{r['step']}] {r.get('script', '-')} — {r.get('title', '')}")
        lines.append("-" * 100)
        lines.append(f"Status   : {r.get('status')}")
        lines.append(f"Duration : {format_duration(r.get('duration_seconds'))}")
        if r.get("note"):
            lines.append(f"Note     : {r.get('note')}")
        if r.get("command"):
            lines.append("Command  : " + " ".join(str(x) for x in r["command"]))

        stdout = r.get("stdout") or ""
        stderr = r.get("stderr") or ""

        if stdout.strip():
            lines.append("")
            lines.append("STDOUT")
            lines.append(stdout.rstrip())
        else:
            lines.append("")
            lines.append("STDOUT")
            lines.append("(empty)")

        if stderr.strip():
            lines.append("")
            lines.append("STDERR")
            lines.append(stderr.rstrip())

    lines.append("")
    lines.append("=" * 100)
    lines.append("End of log")
    lines.append("=" * 100)
    lines.append("")

    return "\n".join(lines)


def write_run_records(project_root: Path, cfg: Dict[str, Any], manifest: Dict[str, Any]) -> Dict[str, str]:
    run_records_dir = resolve_path(project_root, cfg["paths"]["run_records_dir"])
    archive_dir = run_records_dir / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)

    latest_log = run_records_dir / "latest_run.log"
    latest_manifest = run_records_dir / "latest_manifest.json"

    archived_log = archive_dir / f"{manifest['run_id']}_run.log"
    archived_manifest = archive_dir / f"{manifest['run_id']}_manifest.json"

    human_log = build_human_log(manifest)

    write_text(latest_log, human_log)
    write_json(latest_manifest, manifest, pretty=True)

    if bool(cfg.get("pipeline_behavior", {}).get("copy_latest_run_records", True)):
        write_text(archived_log, human_log)
        write_json(archived_manifest, manifest, pretty=True)

    return {
        "latest_log": latest_log.as_posix(),
        "latest_manifest": latest_manifest.as_posix(),
        "archived_log": archived_log.as_posix(),
        "archived_manifest": archived_manifest.as_posix(),
    }


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_pipeline(args: argparse.Namespace) -> int:
    started = now_local()
    run_id = make_run_id(started)

    step_results: List[Dict[str, Any]] = []
    runner_warnings: List[Dict[str, str]] = []
    runner_errors: List[Dict[str, str]] = []

    cfg: Dict[str, Any] = {}
    project_root: Optional[Path] = None

    try:
        cfg = load_resolved_config(__file__)
        project_root = Path(cfg["_resolved"]["project_root"])
        config_path = Path(cfg["_resolved"]["config_path"])
        pipeline_dir = Path(cfg["paths"]["pipeline_dir"])
        if not pipeline_dir.is_absolute():
            pipeline_dir = project_root / pipeline_dir

        if args.dry_run:
            ensure_standard_dirs(cfg)
        else:
            if bool(cfg.get("pipeline_behavior", {}).get("clean_pipeline_output_before_run", True)):
                clean_pipeline_output(cfg)
            else:
                ensure_standard_dirs(cfg)

        behavior = cfg.get("pipeline_behavior", {})
        stop_on_first_error = bool(behavior.get("stop_on_first_error", True))
        if args.continue_on_error:
            stop_on_first_error = False

        selected_steps = [
            step for step in PIPELINE_STEPS
            if should_run_step(step["step"], args.from_step, args.to_step)
        ]

        if args.list_steps or args.dry_run:
            print("=" * 88)
            print("Planned pipeline steps")
            print("=" * 88)
            for step in selected_steps:
                script_path, note = find_step_script(pipeline_dir, step)
                script_name = script_path.name if script_path else step["script"]
                exists = "yes" if script_path else "no"
                note_txt = f" ({note})" if note else ""
                print(f"{step['step']:>2}  exists={exists:<3}  {script_name:<42} {step['title']}{note_txt}")
            print("=" * 88)
            if args.list_steps and not args.dry_run:
                return 0

        if args.dry_run:
            print("Dry run only. No processing steps executed.")
            return 0

        print("=" * 88)
        print("InSAR4D RUM Viewer Pipeline")
        print("=" * 88)
        print(f"Run ID        : {run_id}")
        print(f"Project root  : {project_root}")
        print(f"Config        : {config_path}")
        print(f"Output dir    : {cfg['_resolved']['pipeline_output_dir']}")
        print(f"Run records   : {cfg['_resolved']['run_records_dir']}")
        print("=" * 88)

        for step in selected_steps:
            script_path, note = find_step_script(pipeline_dir, step)

            if script_path is None:
                result = {
                    "step": step["step"],
                    "script": step["script"],
                    "title": step["title"],
                    "importance": step.get("importance", ""),
                    "status": "MISSING",
                    "returncode": None,
                    "started_at": iso(now_local()),
                    "ended_at": iso(now_local()),
                    "duration_seconds": None,
                    "command": None,
                    "warnings": [],
                    "errors": [f"Missing script: {step['script']}"],
                    "stdout": "",
                    "stderr": "",
                    "note": "Script not found. This is expected only before renaming/creating this step.",
                }
                step_results.append(result)
                runner_errors.append({
                    "step": step["step"],
                    "message": f"Missing script: {step['script']}",
                })
                print(f"[{step['step']}] MISSING  {step['script']}")

                if stop_on_first_error:
                    break
                continue

            print("")
            print(f"[{step['step']}] START  {script_path.name} — {step['title']}")
            if note:
                print(f"     NOTE   {note}")

            result = run_one_step(
                step=step,
                script_path=script_path,
                project_root=project_root,
                config_path=config_path,
                extra_args=None,
            )

            if note:
                result["note"] = note

            step_results.append(result)

            for w in result.get("warnings", []):
                runner_warnings.append({"step": result["step"], "message": w})
            for e in result.get("errors", []):
                runner_errors.append({"step": result["step"], "message": e})

            print(
                f"[{step['step']}] {result['status']:<5}  "
                f"{script_path.name}  ({format_duration(result['duration_seconds'])})"
            )

            if result["status"] == "FAIL" and stop_on_first_error:
                print(f"Stopping after failed step {step['step']} because stop_on_first_error=true.")
                break

    except Exception as exc:
        tb = traceback.format_exc()
        runner_errors.append({"step": "runner", "message": str(exc)})
        step_results.append({
            "step": "runner",
            "script": "00_run_pipeline.py",
            "title": "Pipeline runner",
            "importance": "runner",
            "status": "FAIL",
            "returncode": 1,
            "started_at": iso(started),
            "ended_at": iso(now_local()),
            "duration_seconds": None,
            "command": None,
            "warnings": [],
            "errors": [str(exc)],
            "stdout": "",
            "stderr": tb,
        })
        print("Pipeline runner failed before/while executing steps.")
        print(tb)

    ended = now_local()
    duration_s = seconds((ended - started).total_seconds())

    any_fail = any(r.get("status") == "FAIL" for r in step_results)
    any_missing = any(r.get("status") == "MISSING" for r in step_results)
    any_warn = any(r.get("status") == "WARN" for r in step_results) or bool(runner_warnings)

    if any_fail or any_missing or runner_errors:
        final_status = "FAILED"
    elif any_warn:
        final_status = "SUCCESS_WITH_WARNINGS"
    else:
        final_status = "SUCCESS"

    if cfg:
        run_metadata = build_run_metadata(cfg)
        outputs = collect_existing_outputs(cfg)
    else:
        run_metadata = {}
        outputs = {}

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "status": final_status,
        "started_at": iso(started),
        "ended_at": iso(ended),
        "duration_seconds": duration_s,
        "run_metadata": run_metadata,
        "steps": step_results,
        "warnings": runner_warnings,
        "errors": runner_errors,
        "outputs": outputs,
    }

    record_paths: Dict[str, str] = {}
    if cfg and project_root is not None:
        record_paths = write_run_records(project_root, cfg, manifest)
        manifest["run_record_paths"] = record_paths
        # Re-write manifest with its own record paths included.
        write_json(Path(record_paths["latest_manifest"]), manifest, pretty=True)
        if bool(cfg.get("pipeline_behavior", {}).get("copy_latest_run_records", True)):
            write_json(Path(record_paths["archived_manifest"]), manifest, pretty=True)

    print("")
    print("=" * 88)
    print(f"Pipeline finished: {final_status}")
    print(f"Duration         : {format_duration(duration_s)}")
    if record_paths:
        print(f"Log              : {record_paths['latest_log']}")
        print(f"Manifest         : {record_paths['latest_manifest']}")
    print("=" * 88)

    return 0 if final_status in {"SUCCESS", "SUCCESS_WITH_WARNINGS"} else 1


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the InSAR4D RUM Viewer pipeline and write run logs/manifests."
    )
    parser.add_argument(
        "--list-steps",
        action="store_true",
        help="List planned pipeline steps and whether scripts exist, then exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List planned steps and prepare folders, but do not execute processing scripts.",
    )
    parser.add_argument(
        "--from-step",
        default=None,
        help="Start from this step number, e.g. 06.",
    )
    parser.add_argument(
        "--to-step",
        default=None,
        help="Stop at this step number, e.g. 13.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running later steps even if one step fails.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    raise SystemExit(run_pipeline(args))


if __name__ == "__main__":
    main()
