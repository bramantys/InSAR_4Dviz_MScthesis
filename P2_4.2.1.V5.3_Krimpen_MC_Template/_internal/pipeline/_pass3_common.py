#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable


class Pass3Error(RuntimeError):
    pass


def project_root_from(script_file: str | Path) -> Path:
    return Path(script_file).resolve().parents[2]


def require(path: Path, label: str | None = None) -> Path:
    if not path.exists():
        raise Pass3Error(f"Missing {label or 'required path'}: {path}")
    return path


def ensure_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any, *, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    else:
        text = json.dumps(payload, indent=2, ensure_ascii=False)
    atomic_write_bytes(path, text.encode("utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_record(path: Path, root: Path) -> Dict[str, Any]:
    try:
        rel = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        rel = str(path.resolve())
    return {
        "path": rel,
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp", dir=str(destination.parent))
    os.close(fd)
    try:
        shutil.copy2(source, temp_name)
        os.replace(temp_name, destination)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise



def stage_root(project_root: Path) -> Path:
    return project_root / "_internal" / "data_pipeline" / "work" / "stage"


def semantic_root(project_root: Path) -> Path:
    return project_root / "_internal" / "data_pipeline" / "runtime"


def relative_to_project(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def clean_stage_area(project_root: Path, area: str) -> Path:
    path = stage_root(project_root) / area
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def print_pass(title: str, output: Path | None = None) -> None:
    print(f"\n=== {title} ===")
    print("Status : PASS")
    if output is not None:
        print(f"Output : {output}")
