#!/usr/bin/env python3
"""Stage 07: prepare locked 20 m detail and 40 m overview uncertainty layouts."""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path
from _pass3_common import Pass3Error, project_root_from, print_pass, stage_root

FAMILIES = (
    ("detail", 20.0, "uncertainty_layout_detail_20m", "PROTO2_UNCERTAINTY_DETAIL_20M_V5.3"),
    ("overview", 40.0, "uncertainty_layout_overview_40m", "PROTO2_UNCERTAINTY_OVERVIEW_40M_V5.3"),
)

def main() -> int:
    root = project_root_from(__file__)
    engine = Path(__file__).resolve().parent / "_uncertainty_layout.py"
    if not engine.is_file():
        raise Pass3Error(f"Missing shared layout engine: {engine}")
    print("\n=== PROTO2 STAGE 07: BUILD UNCERTAINTY LAYOUTS ===", flush=True)
    print("Locked LODs : detail 20 m / overview 40 m", flush=True)
    for role, spacing, stage_name, delivery_id in FAMILIES:
        print(f"\n--- {role.title()} layout · {spacing:.0f} m ---", flush=True)
        cmd = [sys.executable, str(engine), "--spacing", str(spacing), "--stage-name", stage_name, "--lod-role", role, "--delivery-id", delivery_id, "--script-id", "PROTO2_ADAPTIVE_UNCERTAINTY_LAYOUT_V5.3"]
        completed = subprocess.run(cmd, cwd=root)
        if completed.returncode != 0:
            raise Pass3Error(f"Uncertainty layout failed for {role} ({spacing:.0f} m)")
    out = stage_root(root) / "uncertainty_layout_detail_20m" / "uncertainty_layout_summary.json"
    print_pass("STAGE 07 RESULT", out)
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Pass3Error as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        raise SystemExit(1)
