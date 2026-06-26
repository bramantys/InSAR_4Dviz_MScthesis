#!/usr/bin/env python3
"""Package direct-SPAMS + supplied-MC [parcel, epoch] matrices into viewer assets."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from _proto2_config import expected_int, input_path, load_project_config, output_data_dir, project_root_from, stage_records_dir

PROJECT_ROOT = project_root_from(__file__)
CONFIG = load_project_config(PROJECT_ROOT)
OUTPUT_DATA = output_data_dir(PROJECT_ROOT, CONFIG)
RUN_RECORDS = stage_records_dir(PROJECT_ROOT, CONFIG)
CANONICAL_INPUT = OUTPUT_DATA / "spams_viewer_input_slice_float32.npz"
PARCEL_INVENTORY = OUTPUT_DATA / "parcel_inventory.parquet"
MESH_VERTICES = OUTPUT_DATA / "parcel_cap_mesh_vertices.parquet"
MESH_TRIANGLES = OUTPUT_DATA / "parcel_cap_mesh_triangles.parquet"
MODEL_PARAMS_PARQUET = input_path(PROJECT_ROOT, CONFIG, "model_parameters", "parquet_path", "data/model_params/nl_krimpenerwaard_spams10.parquet")

MOVING_PARCEL_INDEX_OUT = OUTPUT_DATA / "moving_parcel_index.parquet"
PARCEL_RENDER_INDEX_OUT = OUTPUT_DATA / "parcel_render_index.parquet"
MESH_VERTICES_INDEXED_OUT = OUTPUT_DATA / "parcel_cap_mesh_vertices_indexed.parquet"
MESH_TRIANGLES_INDEXED_OUT = OUTPUT_DATA / "parcel_cap_mesh_triangles_indexed.parquet"
ANIMATION_NPZ_OUT = OUTPUT_DATA / "parcel_displacement_matrices_float32.npz"
REVERSIBLE_BIN_OUT = OUTPUT_DATA / "parcel_displacement_reversible_f32.bin"
IRREVERSIBLE_BIN_OUT = OUTPUT_DATA / "parcel_displacement_irreversible_f32.bin"
TOTAL_BIN_OUT = OUTPUT_DATA / "parcel_displacement_total_f32.bin"
DETERMINISTIC_TOTAL_BIN_OUT = OUTPUT_DATA / "parcel_displacement_deterministic_total_f32.bin"
SIGMA_H_BIN_OUT = OUTPUT_DATA / "parcel_displacement_sigma_h_f32.bin"
ANIMATION_MANIFEST_OUT = OUTPUT_DATA / "parcel_animation_manifest.json"
ANIMATION_SUMMARY_OUT = OUTPUT_DATA / "parcel_animation_summary.json"
REPORT_TXT_OUT = RUN_RECORDS / "phase05_animation_arrays_report.txt"
REPORT_JSON_OUT = RUN_RECORDS / "phase05_animation_arrays_report.json"

EXPECTED_TOTAL_PARCELS = expected_int(CONFIG, "total_parcels")
EXPECTED_MOVING_PARCELS = expected_int(CONFIG, "moving_parcels")
EXPECTED_BLANK_PARCELS = expected_int(CONFIG, "blank_parcels")
EXPECTED_MESH_VERTICES = expected_int(CONFIG, "mesh_vertices")
EXPECTED_MESH_TRIANGLES = expected_int(CONFIG, "mesh_triangles")


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    raise SystemExit(1)


def ok(message: str) -> None:
    print(f"[OK] {message}")


def require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        fail("Missing required files:\n  - " + "\n  - ".join(missing))


def component_stats(arr: np.ndarray) -> dict[str, float | int | None]:
    finite = np.isfinite(arr)
    if not finite.any():
        return {"finite_count": 0, "nan_count": int(np.isnan(arr).sum()), "min": None, "max": None, "mean": None}
    values = arr[finite]
    return {
        "finite_count": int(finite.sum()),
        "nan_count": int(np.isnan(arr).sum()),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


def write_float32_binary(path: Path, arr: np.ndarray) -> None:
    np.ascontiguousarray(arr.astype("<f4", copy=False)).tofile(path)


def load_canonical_input(path: Path):
    try:
        with np.load(path, allow_pickle=False) as bundle:
            required = ["reversible", "irreversible", "deterministic_total", "total_mean", "total_sigma", "moving_parcel_id", "epoch"]
            missing = [key for key in required if key not in bundle.files]
            if missing:
                fail(f"canonical SPAMS input missing arrays: {missing}")
            return (
                np.ascontiguousarray(bundle["reversible"].astype(np.float32, copy=False)),
                np.ascontiguousarray(bundle["irreversible"].astype(np.float32, copy=False)),
                np.ascontiguousarray(bundle["deterministic_total"].astype(np.float32, copy=False)),
                np.ascontiguousarray(bundle["total_mean"].astype(np.float32, copy=False)),
                np.ascontiguousarray(bundle["total_sigma"].astype(np.float32, copy=False)),
                bundle["moving_parcel_id"].astype(np.int64, copy=False),
                bundle["epoch"].astype("U10", copy=False),
            )
    except SystemExit:
        raise
    except Exception as exc:
        fail(f"Could not read canonical SPAMS input: {path}: {exc}")


def main() -> None:
    OUTPUT_DATA.mkdir(parents=True, exist_ok=True)
    RUN_RECORDS.mkdir(parents=True, exist_ok=True)
    print("\n=== PROTO2 PHASE 05: PACKAGE DIRECT SPAMS ANIMATION ARRAYS ===")

    require_files([CANONICAL_INPUT, PARCEL_INVENTORY, MESH_VERTICES, MESH_TRIANGLES, MODEL_PARAMS_PARQUET])
    reversible, irreversible, deterministic_total, mc_total, sigma_h, moving_ids, epoch_labels = load_canonical_input(CANONICAL_INPUT)
    n_moving, n_epochs = reversible.shape
    matrices = {
        "reversible": reversible,
        "irreversible": irreversible,
        "deterministic_total": deterministic_total,
        "total_mean": mc_total,
        "total_sigma": sigma_h,
    }
    for label, matrix in matrices.items():
        if matrix.ndim != 2 or matrix.shape != (n_moving, n_epochs):
            fail(f"{label} matrix shape mismatch: {matrix.shape}; expected {(n_moving, n_epochs)}")
        if not np.isfinite(matrix).all():
            fail(f"{label} matrix contains non-finite values")
    if np.any(sigma_h < 0.0):
        fail("total_sigma contains negative values")
    if len(moving_ids) != n_moving or len(epoch_labels) != n_epochs or len(np.unique(moving_ids)) != n_moving:
        fail("canonical input parcel ID / epoch arrays do not match matrix shape")
    if EXPECTED_MOVING_PARCELS is not None and n_moving != EXPECTED_MOVING_PARCELS:
        fail(f"moving parcel count {n_moving:,} != expected {EXPECTED_MOVING_PARCELS:,}")

    component_diff = np.abs(deterministic_total - (reversible + irreversible))
    max_component_total_diff = float(component_diff.max())
    if max_component_total_diff > 1e-4:
        fail(f"direct deterministic total != reversible + irreversible; max diff={max_component_total_diff:.9g}")
    ok(f"canonical direct SPAMS + Monte Carlo input loaded: {n_moving:,} parcels × {n_epochs:,} epochs")
    ok(f"viewer period: {epoch_labels[0]} to {epoch_labels[-1]}")
    ok(f"direct deterministic total check passed; max diff={max_component_total_diff:.9g}")

    meta_cols = ["pnt_id", "pnt_gid", "pnt_lat", "pnt_lon", "vI", "var_vI"]
    params = pd.read_parquet(MODEL_PARAMS_PARQUET, columns=meta_cols)
    params["pnt_id"] = pd.to_numeric(params["pnt_id"], errors="raise").astype("int64")
    params["pnt_gid"] = pd.to_numeric(params["pnt_gid"], errors="raise").astype("int64")
    for col in ["pnt_lat", "pnt_lon", "vI", "var_vI"]:
        params[col] = pd.to_numeric(params[col], errors="raise").astype(float)
    params["std_vI"] = np.sqrt(np.clip(params["var_vI"].to_numpy(dtype=float), 0.0, None))
    try:
        moving_meta = params.set_index("pnt_id").loc[moving_ids].reset_index()
    except KeyError as exc:
        fail(f"Could not align parameter metadata to canonical direct-SPAMS row order: {exc}")
    moving_meta["displacement_row_index"] = np.arange(n_moving, dtype=np.int32)

    inventory = pd.read_parquet(PARCEL_INVENTORY, columns=["parcel_id", "parcel_status", "has_displacement"])
    inventory["parcel_id"] = pd.to_numeric(inventory["parcel_id"], errors="raise").astype("int64")
    inventory["has_displacement"] = inventory["has_displacement"].astype(bool)
    if EXPECTED_TOTAL_PARCELS is not None and len(inventory) != EXPECTED_TOTAL_PARCELS:
        fail(f"parcel inventory count {len(inventory):,} != expected {EXPECTED_TOTAL_PARCELS:,}")
    inv_moving = int(inventory["has_displacement"].sum())
    inv_blank = int((~inventory["has_displacement"]).sum())
    if inv_moving != n_moving:
        fail(f"inventory moving parcel count {inv_moving:,} != direct SPAMS row count {n_moving:,}")
    if EXPECTED_BLANK_PARCELS is not None and inv_blank != EXPECTED_BLANK_PARCELS:
        fail(f"inventory blank parcel count {inv_blank:,} != expected {EXPECTED_BLANK_PARCELS:,}")

    parcel_render_index = inventory.sort_values("parcel_id").reset_index(drop=True).copy()
    parcel_render_index["parcel_row_index"] = np.arange(len(parcel_render_index), dtype=np.int32)
    parcel_render_index = parcel_render_index.merge(
        moving_meta.rename(columns={"pnt_id": "parcel_id"}), how="left", on="parcel_id", validate="one_to_one"
    )
    parcel_render_index["displacement_row_index"] = parcel_render_index["displacement_row_index"].fillna(-1).astype("int32")
    missing_on_moving = int(parcel_render_index.loc[parcel_render_index["has_displacement"], "displacement_row_index"].eq(-1).sum())
    present_on_blank = int(parcel_render_index.loc[~parcel_render_index["has_displacement"], "displacement_row_index"].ne(-1).sum())
    if missing_on_moving or present_on_blank:
        fail(f"render index mismatch: missing_on_moving={missing_on_moving}, present_on_blank={present_on_blank}")

    mesh_vertices = pd.read_parquet(MESH_VERTICES)
    mesh_triangles = pd.read_parquet(MESH_TRIANGLES)
    if EXPECTED_MESH_VERTICES is not None and len(mesh_vertices) != EXPECTED_MESH_VERTICES:
        fail(f"mesh vertex count {len(mesh_vertices):,} != expected {EXPECTED_MESH_VERTICES:,}")
    if EXPECTED_MESH_TRIANGLES is not None and len(mesh_triangles) != EXPECTED_MESH_TRIANGLES:
        fail(f"mesh triangle count {len(mesh_triangles):,} != expected {EXPECTED_MESH_TRIANGLES:,}")
    index_cols = parcel_render_index[["parcel_id", "parcel_row_index", "displacement_row_index"]]
    mesh_vertices_indexed = mesh_vertices.merge(index_cols, how="left", on="parcel_id", validate="many_to_one")
    mesh_triangles_indexed = mesh_triangles.merge(index_cols, how="left", on="parcel_id", validate="many_to_one")
    for label, frame in [("mesh vertices", mesh_vertices_indexed), ("mesh triangles", mesh_triangles_indexed)]:
        if frame[["parcel_row_index", "displacement_row_index"]].isna().any().any():
            fail(f"{label} have missing runtime indices after parcel join")
        frame["parcel_row_index"] = frame["parcel_row_index"].astype("int32")
        frame["displacement_row_index"] = frame["displacement_row_index"].astype("int32")

    print("\nWriting runtime array products...")
    moving_meta.to_parquet(MOVING_PARCEL_INDEX_OUT, index=False)
    parcel_render_index.to_parquet(PARCEL_RENDER_INDEX_OUT, index=False)
    mesh_vertices_indexed.to_parquet(MESH_VERTICES_INDEXED_OUT, index=False)
    mesh_triangles_indexed.to_parquet(MESH_TRIANGLES_INDEXED_OUT, index=False)
    np.savez_compressed(
        ANIMATION_NPZ_OUT,
        reversible=reversible,
        irreversible=irreversible,
        total=mc_total,
        deterministic_total=deterministic_total,
        sigma_h=sigma_h,
        moving_parcel_id=moving_ids,
        epoch=epoch_labels,
    )
    write_float32_binary(REVERSIBLE_BIN_OUT, reversible)
    write_float32_binary(IRREVERSIBLE_BIN_OUT, irreversible)
    write_float32_binary(TOTAL_BIN_OUT, mc_total)
    write_float32_binary(DETERMINISTIC_TOTAL_BIN_OUT, deterministic_total)
    write_float32_binary(SIGMA_H_BIN_OUT, sigma_h)
    for path in [MOVING_PARCEL_INDEX_OUT, PARCEL_RENDER_INDEX_OUT, MESH_VERTICES_INDEXED_OUT, MESH_TRIANGLES_INDEXED_OUT, ANIMATION_NPZ_OUT, REVERSIBLE_BIN_OUT, IRREVERSIBLE_BIN_OUT, TOTAL_BIN_OUT, DETERMINISTIC_TOTAL_BIN_OUT, SIGMA_H_BIN_OUT]:
        ok(f"wrote {path}")

    stats = {
        "reversible": component_stats(reversible),
        "irreversible": component_stats(irreversible),
        "deterministic_total": component_stats(deterministic_total),
        "total_mc_mean": component_stats(mc_total),
        "sigma_h": component_stats(sigma_h),
    }
    manifest = {
        "product": "parcel_animation_arrays",
        "version": 5,
        "epoch_count": int(n_epochs),
        "matrix_shape": [int(n_moving), int(n_epochs)],
        "shape": {"moving_parcels": int(n_moving), "epochs": int(n_epochs), "matrix_order": "row-major C order", "matrix_indexing": "matrix[displacement_row_index, epoch_index]"},
        "epoch": {"start": str(epoch_labels[0]), "end": str(epoch_labels[-1]), "count": int(n_epochs), "labels": epoch_labels.tolist()},
        "row_order": "Rows preserve Monte Carlo / model-parameter Parquet order; pnt_id values are recorded in moving_parcel_index.parquet.",
        "products": {
            "reversible": {"binary": str(REVERSIBLE_BIN_OUT), "availability": {"start": str(epoch_labels[0]), "end": str(epoch_labels[-1]), "epochs": int(n_epochs)}, "meaning": "direct deterministic reversible SPAMS component from Parquet + KNMI + utils.py"},
            "irreversible": {"binary": str(IRREVERSIBLE_BIN_OUT), "availability": {"start": str(epoch_labels[0]), "end": str(epoch_labels[-1]), "epochs": int(n_epochs)}, "meaning": "direct deterministic irreversible SPAMS component from Parquet + KNMI + utils.py"},
            "deterministic_total": {"binary": str(DETERMINISTIC_TOTAL_BIN_OUT), "meaning": "direct deterministic reversible + irreversible"},
            "total": {"binary": str(TOTAL_BIN_OUT), "meaning": "supplied Monte Carlo mean_t of Total SPAMS displacement; authoritative Total-mode height/time-series product"},
            "sigma_h": {"binary": str(SIGMA_H_BIN_OUT), "meaning": "supplied matching Monte Carlo sigma_t per epoch; Total-only uncertainty product"},
        },
        "time_reference": {
            "source_reconstruction_period": CONFIG["monte_carlo_total"]["source_reconstruction_period"],
            "viewer_period": CONFIG["time_settings"]["viewer_period"],
            "viewer_reference_rule": "Viewer may re-reference displayed mean total to a loaded epoch. sigma_h remains a supplied per-epoch MC standard deviation and is not rebased by this pipeline.",
        },
        "checks": {"direct_deterministic_total_max_abs_component_diff": max_component_total_diff, "mc_sigma_negative_count": int((sigma_h < 0.0).sum())},
    }
    summary = {
        "source_canonical_direct_spams_input": str(CANONICAL_INPUT),
        "total_parcels": int(len(inventory)),
        "moving_parcels": int(n_moving),
        "blank_parcels": int(inv_blank),
        "epochs": int(n_epochs),
        "matrix_shape": [int(n_moving), int(n_epochs)],
        "epoch_start": str(epoch_labels[0]),
        "epoch_end": str(epoch_labels[-1]),
        "default_reference_date": CONFIG["time_settings"]["viewer_period"]["default_reference_date"],
        "deterministic_component_availability": {"start": str(epoch_labels[0]), "end": str(epoch_labels[-1]), "epochs": int(n_epochs)},
        "component_stats": stats,
        "outputs": {"manifest": str(ANIMATION_MANIFEST_OUT), "summary": str(ANIMATION_SUMMARY_OUT)},
    }
    ANIMATION_MANIFEST_OUT.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    ANIMATION_SUMMARY_OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    REPORT_JSON_OUT.write_text(json.dumps({"summary": summary, "manifest": manifest}, indent=2), encoding="utf-8")
    REPORT_TXT_OUT.write_text(
        "\n".join([
            "PROTO2 PHASE 05 DIRECT SPAMS ANIMATION ARRAYS REPORT", "",
            f"viewer period: {epoch_labels[0]} to {epoch_labels[-1]} ({n_epochs:,} epochs)",
            f"direct deterministic component availability: full viewer period ({n_epochs:,} epochs)",
            f"matrix shape: {n_moving:,} x {n_epochs:,}",
            f"MC total finite cells: {stats['total_mc_mean']['finite_count']:,}",
            f"MC sigma finite cells: {stats['sigma_h']['finite_count']:,}",
            f"direct deterministic component total max diff: {max_component_total_diff:.9g}",
            "", "RESULT: PASS",
        ]), encoding="utf-8")
    ok(f"wrote {ANIMATION_MANIFEST_OUT}")
    ok(f"wrote {ANIMATION_SUMMARY_OUT}")
    ok(f"wrote {REPORT_JSON_OUT}")
    ok(f"wrote {REPORT_TXT_OUT}")
    print("\nPHASE 05 RESULT: PASS. Direct SPAMS components and Monte Carlo Total share one runtime epoch axis.")


if __name__ == "__main__":
    main()
