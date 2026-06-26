#!/usr/bin/env python3
from __future__ import annotations

import json
import queue
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from _pass3_common import Pass3Error, project_root_from
from _proto2_config import load_project_config, run_records_dir, runtime_root, stage_records_dir


PIPELINE_STEPS: list[tuple[str, str, tuple[str, ...], str]] = [
    ("00", "00_validate_inputs_and_model_contract.py", (), "Validate the configured input data and model contract"),
    ("01", "01_compute_spams_components.py", (), "Compute reversible and irreversible SPAMS components"),
    ("02", "02_prepare_parcel_inventory.py", (), "Build moving-versus-blank parcel inventory"),
    ("03", "03_prepare_parcel_footprints.py", (), "Normalize parcel footprints and multipart geometry"),
    ("04", "04_triangulate_parcel_geometry.py", (), "Triangulate parcel caps and wall geometry"),
    ("05", "05_pack_animation_arrays.py", (), "Pack compact Float32 animation arrays"),
    ("06", "06_build_main_runtime_geometry.py", (), "Build caps, pistons, walls and datum geometry"),
    ("07", "07_prepare_uncertainty_layouts.py", (), "Prepare uncertainty carrier layouts"),
    ("08", "08_build_uncertainty_runtime_geometry.py", (), "Build uncertainty GLB LOD geometry"),
    ("09", "09_build_inspection_assets.py", (), "Build picking, trendline and parcel-search indices"),
    ("10", "10_build_viewer_runtime_manifest.py", (), "Build colour scales, tuning and runtime manifest"),
    ("11", "11_publish_runtime_assets.py", ("--publish",), "Publish the validated runtime asset bundle"),
    ("12", "12_assemble_viewer.py", ("--validate-files",), "Assemble the final viewer HTML"),
    ("13", "13_validate_release.py", (), "Validate the viewer and every required runtime asset"),
]
HEARTBEAT_SECONDS = 15.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def run_step(root: Path, code: str, script: Path, args: tuple[str, ...], label: str, lines: list[str]) -> None:
    header = f"\n>>> [{code}] {label}"
    command = [sys.executable, str(script), *args]
    print(header, flush=True)
    print("    " + " ".join(command), flush=True)
    lines.extend([header, "    " + " ".join(command)])
    process = subprocess.Popen(
        command, cwd=str(root), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    assert process.stdout is not None
    stream: queue.Queue[str | None] = queue.Queue()

    def reader() -> None:
        try:
            for output in process.stdout:
                stream.put(output)
        finally:
            stream.put(None)

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    last_output = time.monotonic()
    done = False
    while not done:
        try:
            line = stream.get(timeout=1.0)
        except queue.Empty:
            if time.monotonic() - last_output >= HEARTBEAT_SECONDS:
                msg = f"[working] [{code}] still running: {label}"
                print(msg, flush=True)
                lines.append(msg)
                last_output = time.monotonic()
            if process.poll() is not None and not thread.is_alive():
                done = True
            continue
        if line is None:
            done = True
            continue
        print(line, end="", flush=True)
        lines.append(line.rstrip("\n"))
        last_output = time.monotonic()
    thread.join(timeout=2.0)
    exit_code = process.wait()
    if exit_code:
        raise Pass3Error(f"Stage {code} failed ({exit_code}): {script.name}")


def data_pipeline_root(root: Path, config: dict[str, Any]) -> Path:
    value = str(config.get("paths", {}).get("data_pipeline_root", "_internal/data_pipeline"))
    candidate = Path(value)
    return candidate if candidate.is_absolute() else root / candidate


def reset_generated_outputs(root: Path, config: dict[str, Any]) -> None:
    generated = data_pipeline_root(root, config)
    if generated.exists():
        shutil.rmtree(generated)
    generated.mkdir(parents=True, exist_ok=True)
    stage_records_dir(root, config).mkdir(parents=True, exist_ok=True)
    viewer_name = str(config.get("paths", {}).get("output_viewer", "viz2_parcel_viewer.html"))
    viewer = root / viewer_name
    if viewer.exists():
        viewer.unlink()


def cleanup_success_transients(root: Path, config: dict[str, Any]) -> None:
    keep = bool(config.get("pipeline_behavior", {}).get("keep_build_work", False))
    if keep:
        print("[info] pipeline_behavior.keep_build_work=true; retaining temporary build work")
        return
    work = data_pipeline_root(root, config) / "work"
    if work.exists():
        shutil.rmtree(work)
        print("[OK] Removed temporary build work; retained only _internal/data_pipeline/runtime")


def trim_history(history: Path, keep: int) -> None:
    records = sorted(history.glob("run_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for record in records[keep:]:
        text = record.with_suffix(".txt")
        record.unlink(missing_ok=True)
        text.unlink(missing_ok=True)


def write_receipt(root: Path, config: dict[str, Any], *, ident: str, status: str, started: str, finished: str, completed: list[str], failed: str | None, error: str | None, lines: list[str]) -> None:
    records = run_records_dir(root, config)
    history = records / "history"
    history.mkdir(parents=True, exist_ok=True)
    runtime = runtime_root(root, config)
    metadata = read_json_if_exists(runtime / "viewer_metadata.json")
    viewer_name = str(config.get("paths", {}).get("output_viewer", "viz2_parcel_viewer.html"))
    viewer = root / viewer_name
    payload = {
        "schema": "proto2_pipeline_receipt_v5_3",
        "run_id": ident,
        "status": status,
        "started_utc": started,
        "finished_utc": finished,
        "failed_stage": failed,
        "error": error,
        "project_id": config.get("project", {}).get("project_id"),
        "completed_stages": completed,
        "summary": {key: metadata.get(key) for key in ("total_parcels", "moving_parcels", "blank_parcels", "epochs", "epoch_start", "epoch_end") if key in metadata},
        "viewer": {"path": viewer_name, "exists": viewer.is_file(), "size_bytes": viewer.stat().st_size if viewer.is_file() else None},
        "runtime_root": str(runtime.relative_to(root)) if runtime.is_relative_to(root) else str(runtime),
    }
    content = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    human = [
        "PROTO2 PIPELINE RECEIPT", "", f"Run ID: {ident}", f"Status: {status}",
        f"Started UTC: {started}", f"Finished UTC: {finished}",
        f"Failed stage: {failed or 'None'}", f"Viewer: {viewer_name if viewer.is_file() else 'not produced'}", "",
        "Completed stages:", *(f"  - {item}" for item in completed), "", "Console log:", *lines,
    ]
    for path in (history / f"run_{ident}.json", records / "latest_run.json"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    for path in (history / f"run_{ident}.txt", records / "latest_run.txt"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(human) + "\n", encoding="utf-8")
    keep = int(config.get("pipeline_behavior", {}).get("keep_last_run_history", 10))
    trim_history(history, max(1, keep))


def main() -> int:
    root = project_root_from(__file__)
    config = load_project_config(root)
    started = utc_now()
    ident = run_id()
    lines: list[str] = []
    completed: list[str] = []
    failed: str | None = None
    try:
        reset_generated_outputs(root, config)
        print("\n================================================================================")
        print(" Proto2 parcel viewer: clean full build")
        print(" Runtime output: _internal/data_pipeline/runtime")
        print(" Temporary work: _internal/data_pipeline/work (removed after a PASS release)")
        print("================================================================================")
        for code, filename, args, label in PIPELINE_STEPS:
            failed = code
            script = root / "_internal" / "pipeline" / filename
            if not script.is_file():
                raise Pass3Error(f"Missing pipeline stage: {script}")
            run_step(root, code, script, args, label, lines)
            completed.append(code)
            failed = None
        finished = utc_now()
        write_receipt(root, config, ident=ident, status="PASS", started=started, finished=finished, completed=completed, failed=None, error=None, lines=lines)
        cleanup_success_transients(root, config)
        print("\n================================================================================")
        print(" PIPELINE FINISHED SUCCESSFULLY")
        print(" Viewer : viz2_parcel_viewer.html")
        print(" Runtime: _internal/data_pipeline/runtime")
        print(" Receipt: run_records/latest_run.txt")
        print("================================================================================")
        return 0
    except Exception as exc:
        finished = utc_now()
        message = str(exc)
        print(f"\n[FAIL] {message}", file=sys.stderr)
        write_receipt(root, config, ident=ident, status="FAIL", started=started, finished=finished, completed=completed, failed=failed, error=message, lines=lines)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
