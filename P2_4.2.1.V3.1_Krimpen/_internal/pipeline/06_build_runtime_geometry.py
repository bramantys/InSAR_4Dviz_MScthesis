#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from _pass3_common import (
    Pass3Error,
    atomic_write_bytes,
    clean_stage_area,
    file_record,
    load_legacy_module,
    print_pass,
    project_root_from,
    read_json,
    require,
    stage_root,
    write_json,
)


from _proto2_config import load_project_config, output_data_dir

def component_stats(arr: np.ndarray) -> Dict[str, Any]:
    """
    Match the accepted Phase 09 metadata semantics exactly.

    The runtime arrays are float32.  The accepted summary calculated min,
    max and mean on those float32 values directly.  Promoting to float64
    before the aggregate changes the reported mean slightly even though the
    binary runtime asset is byte-identical.
    """
    finite_mask = np.isfinite(arr)
    values = arr[finite_mask]
    if values.size == 0:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "finite_count": 0,
            "nan_count": int(arr.size),
        }
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "finite_count": int(finite_mask.sum()),
        "nan_count": int(np.isnan(arr).sum()),
    }


def read_matrix(path: Path, rows: int, cols: int) -> np.ndarray:
    arr = np.fromfile(path, dtype="<f4")
    expected = rows * cols
    if arr.size != expected:
        raise Pass3Error(f"Unexpected float32 count in {path}: {arr.size:,} != {expected:,}")
    return arr.reshape(rows, cols)


def build_cap_glb(project_root: Path, pipeline_dir: Path, out_path: Path) -> Dict[str, Any]:
    legacy = load_legacy_module(pipeline_dir, "09_export_cesium_animated_glb_preview.py", "proto2_legacy_phase09")
    config = load_project_config(project_root)
    output_data = output_data_dir(project_root, config)
    vertices_candidate = output_data / "parcel_cap_mesh_vertices_indexed.parquet"
    if not vertices_candidate.exists():
        raise Pass3Error(
            "Missing indexed cap vertices: "
            f"{vertices_candidate}\n"
            "The bonestock template does not retain generated intermediates. "
            "Run the full pipeline so stages 00-05 rebuild them first."
        )
    vertices_path = vertices_candidate
    triangles_path = require(output_data / "parcel_cap_mesh_triangles_indexed.parquet", "indexed cap triangles")
    animation_manifest_path = require(output_data / "parcel_animation_manifest.json", "animation manifest")
    rev_path = require(output_data / "parcel_displacement_reversible_f32.bin", "reversible float32 array")
    irr_path = require(output_data / "parcel_displacement_irreversible_f32.bin", "irreversible float32 array")
    total_path = require(output_data / "parcel_displacement_total_f32.bin", "total float32 array")

    animation_manifest = read_json(animation_manifest_path)
    shape = animation_manifest.get("shape") or animation_manifest.get("matrix_shape") or {}
    rows = int(shape.get("moving_parcels") or shape.get("rows") or shape.get("n_rows") or 0)
    epochs = int(shape.get("epochs") or shape.get("columns") or shape.get("n_epochs") or 0)
    if rows <= 0 or epochs <= 0:
        raise Pass3Error(f"Could not resolve animation shape from {animation_manifest_path}")

    reversible = read_matrix(rev_path, rows, epochs)
    irreversible = read_matrix(irr_path, rows, epochs)
    total = read_matrix(total_path, rows, epochs)
    max_total_diff = float(np.nanmax(np.abs(total - (reversible + irreversible))))
    if max_total_diff > 1e-4:
        raise Pass3Error(f"total != reversible + irreversible; max diff={max_total_diff:.9g}")

    vertices = pd.read_parquet(vertices_path)
    triangles = pd.read_parquet(triangles_path)
    required_vertex_cols = ["global_vertex_index", "lon", "lat", "has_displacement", "displacement_row_index"]
    required_triangle_cols = ["global_triangle_index", "v0", "v1", "v2"]
    missing_v = [c for c in required_vertex_cols if c not in vertices.columns]
    missing_t = [c for c in required_triangle_cols if c not in triangles.columns]
    if missing_v or missing_t:
        raise Pass3Error(f"Missing mesh columns; vertices={missing_v}, triangles={missing_t}")

    vertices = vertices.sort_values("global_vertex_index").reset_index(drop=True)
    triangles = triangles.sort_values("global_triangle_index").reset_index(drop=True)
    if not np.array_equal(vertices["global_vertex_index"].to_numpy(np.int64), np.arange(len(vertices), dtype=np.int64)):
        raise Pass3Error("global_vertex_index is not contiguous")
    if not np.array_equal(triangles["global_triangle_index"].to_numpy(np.int64), np.arange(len(triangles), dtype=np.int64)):
        raise Pass3Error("global_triangle_index is not contiguous")

    lon = vertices["lon"].to_numpy(np.float64)
    lat = vertices["lat"].to_numpy(np.float64)
    west, east = float(lon.min()), float(lon.max())
    south, north = float(lat.min()), float(lat.max())
    center_lon = 0.5 * (west + east)
    center_lat = 0.5 * (south + north)
    static_height = float(getattr(legacy, "STATIC_HEIGHT_OFFSET_M", 4.0))

    x_ecef, y_ecef, z_ecef = legacy.wgs84_to_ecef(lon, lat, np.full_like(lon, static_height))
    local_x, local_y, local_z = legacy.ecef_to_local_enu(
        x_ecef, y_ecef, z_ecef, center_lon, center_lat, static_height
    )
    positions = np.column_stack([local_y, -local_x, local_z]).astype("<f4")

    disp_row = vertices["displacement_row_index"].to_numpy(np.float32)
    has_disp = vertices["has_displacement"].astype(bool).to_numpy()
    if np.nanmin(disp_row) < -1 or np.nanmax(disp_row) >= rows:
        raise Pass3Error("displacement_row_index is outside animation matrix")

    texcoord = np.empty((len(vertices), 2), dtype="<f4")
    texcoord[:, 0] = disp_row.astype("<f4")
    texcoord[:, 1] = has_disp.astype(np.float32)
    colors = np.empty((len(vertices), 4), dtype=np.uint8)
    colors[has_disp] = np.array([47, 128, 237, 215], dtype=np.uint8)
    colors[~has_disp] = np.array([184, 184, 184, 90], dtype=np.uint8)
    indices = triangles[["v0", "v1", "v2"]].to_numpy(dtype="<u4").reshape(-1)
    if int(indices.min()) < 0 or int(indices.max()) >= len(vertices):
        raise Pass3Error("Cap indices reference vertices out of range")

    glb = legacy.build_glb(positions, colors, texcoord, indices)
    atomic_write_bytes(out_path, glb)

    # Reproduce the accepted Phase 09 metadata literally.
    #
    # Historical note:
    # POSITION is [north, -east, up] after the 90-degree orientation fix.
    # Phase 09 nevertheless stored POSITION[:, 0] under the key "east_west"
    # and POSITION[:, 1] under "north_south".  Those labels are therefore
    # historical render-axis labels, not clean geographic semantics.
    #
    # Pass 3 is a strict parity pass, so we preserve that accepted contract
    # exactly.  A future schema migration may rename these fields deliberately.
    # The span calculation also remains on float32 POSITION values, matching
    # the accepted camera-height precision.
    span_east_west = float(positions[:, 0].max() - positions[:, 0].min())
    span_north_south = float(positions[:, 1].max() - positions[:, 1].min())
    epoch = animation_manifest.get("epoch") or {}
    start = str(epoch.get("start") or animation_manifest.get("epoch_start") or "")
    end = str(epoch.get("end") or animation_manifest.get("epoch_end") or "")
    try:
        labels = pd.date_range(start=start, end=end, periods=epochs).strftime("%Y-%m-%d").tolist()
    except Exception:
        labels = [f"epoch {i}" for i in range(epochs)]

    return {
        "vertices": int(len(vertices)),
        "triangles": int(len(triangles)),
        "indices": int(indices.size),
        "moving_vertices": int(has_disp.sum()),
        "blank_vertices": int((~has_disp).sum()),
        "moving_parcels": rows,
        "epochs": epochs,
        "epoch_start": start,
        "epoch_end": end,
        "epoch_labels": labels,
        "stats": {
            "reversible": component_stats(reversible),
            "irreversible": component_stats(irreversible),
            "total": component_stats(total),
        },
        "max_total_diff": max_total_diff,
        "center_lon": center_lon,
        "center_lat": center_lat,
        "center_height_m": static_height,
        "camera_height_m": max(span_east_west, span_north_south) * 2.2,
        "bounds_wgs84": {"west": west, "south": south, "east": east, "north": north},
        "local_span_m": {
            "east_west": span_east_west,
            "north_south": span_north_south,
        },
    }


def build_piston_and_aux(project_root: Path, pipeline_dir: Path, cap_summary: Dict[str, Any], paths: Dict[str, Path]) -> Dict[str, Any]:
    legacy15 = load_legacy_module(pipeline_dir, "15_build_irreversible_piston_assets.py", "proto2_legacy_phase15")
    legacy16 = load_legacy_module(pipeline_dir, "16c_export_multimode_deformation_viewer.py", "proto2_legacy_phase16c")
    legacy16e = load_legacy_module(pipeline_dir, "16e_export_multimode_deformation_viewer.py", "proto2_legacy_phase16e")

    center_lon = float(cap_summary["center_lon"])
    center_lat = float(cap_summary["center_lat"])
    cap_pos, cap_col, cap_tex0, cap_tex1, cap_idx, cap_stats = legacy15.read_cap_mesh(center_lon, center_lat)
    wall_pos, wall_col, wall_tex0, wall_tex1, wall_idx, wall_stats = legacy15.read_wall_mesh(center_lon, center_lat)

    vertex_offset = np.uint32(cap_pos.shape[0])
    positions = np.vstack([cap_pos, wall_pos]).astype("<f4")
    colors = np.vstack([cap_col, wall_col]).astype(np.uint8)
    tex0 = np.vstack([cap_tex0, wall_tex0]).astype("<f4")
    tex1 = np.vstack([cap_tex1, wall_tex1]).astype("<f4")
    indices = np.concatenate([cap_idx, (wall_idx + vertex_offset).astype("<u4")]).astype("<u4")
    if int(indices.min()) < 0 or int(indices.max()) >= len(positions):
        raise Pass3Error("Combined piston indices reference vertices out of range")

    atomic_write_bytes(paths["pistons"], legacy15.build_glb(positions, colors, tex0, tex1, indices))
    wall_summary = legacy16.build_wall_only_glb_from_piston(paths["pistons"], paths["walls"])
    opaque_summary = legacy16e.build_opaque_datum_cap_glb(paths["caps"], paths["opaque_datum_caps"])

    return {
        "cap_stats": cap_stats,
        "wall_stats": wall_stats,
        "combined": {
            "vertices": int(len(positions)),
            "triangles": int(len(indices) // 3),
            "indices": int(len(indices)),
        },
        "wall_split": wall_summary,
        "opaque_datum": opaque_summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build staged Proto2 runtime geometry using accepted algorithms.")
    parser.parse_args()

    project_root = project_root_from(__file__)
    pipeline_dir = Path(__file__).resolve().parent
    geometry_dir = clean_stage_area(project_root, "geometry")
    paths = {
        "caps": geometry_dir / "parcel_caps.glb",
        "pistons": geometry_dir / "parcel_pistons.glb",
        "walls": geometry_dir / "parcel_walls.glb",
        "opaque_datum_caps": geometry_dir / "parcel_datum_caps_opaque.glb",
    }

    print("\n=== PROTO2 PASS 3 / STAGE 06: BUILD RUNTIME GEOMETRY ===")
    cap_summary = build_cap_glb(project_root, pipeline_dir, paths["caps"])
    aux_summary = build_piston_and_aux(project_root, pipeline_dir, cap_summary, paths)
    summary = {
        "schema": "proto2_pass3_geometry_build_v1",
        "algorithm_source": [
            "09_export_cesium_animated_glb_preview.py",
            "15_build_irreversible_piston_assets.py",
            "16c_export_multimode_deformation_viewer.py",
            "16e_export_multimode_deformation_viewer.py",
        ],
        "cap": cap_summary,
        "piston": aux_summary,
        "outputs": {key: file_record(path, project_root) for key, path in paths.items()},
    }
    report = stage_root(project_root) / "geometry_build_summary.json"
    write_json(report, summary)
    print_pass("STAGE 06 RESULT", report)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Pass3Error as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        raise SystemExit(1)
