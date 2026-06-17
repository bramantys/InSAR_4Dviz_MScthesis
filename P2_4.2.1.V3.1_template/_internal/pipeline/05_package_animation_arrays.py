from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

from _proto2_config import expected_int, load_project_config, output_data_dir, stage_records_dir


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = load_project_config(PROJECT_ROOT)

OUTPUT_DATA = output_data_dir(PROJECT_ROOT, CONFIG)
RUN_RECORDS = stage_records_dir(PROJECT_ROOT, CONFIG)

DISPLACEMENT_LONG = OUTPUT_DATA / "parcel_displacement_long.parquet"
PARCEL_INVENTORY = OUTPUT_DATA / "parcel_inventory.parquet"
MESH_VERTICES = OUTPUT_DATA / "parcel_cap_mesh_vertices.parquet"
MESH_TRIANGLES = OUTPUT_DATA / "parcel_cap_mesh_triangles.parquet"

MOVING_PARCEL_INDEX_OUT = OUTPUT_DATA / "moving_parcel_index.parquet"
PARCEL_RENDER_INDEX_OUT = OUTPUT_DATA / "parcel_render_index.parquet"

MESH_VERTICES_INDEXED_OUT = OUTPUT_DATA / "parcel_cap_mesh_vertices_indexed.parquet"
MESH_TRIANGLES_INDEXED_OUT = OUTPUT_DATA / "parcel_cap_mesh_triangles_indexed.parquet"

ANIMATION_NPZ_OUT = OUTPUT_DATA / "parcel_displacement_matrices_float32.npz"
REVERSIBLE_BIN_OUT = OUTPUT_DATA / "parcel_displacement_reversible_f32.bin"
IRREVERSIBLE_BIN_OUT = OUTPUT_DATA / "parcel_displacement_irreversible_f32.bin"
TOTAL_BIN_OUT = OUTPUT_DATA / "parcel_displacement_total_f32.bin"
SIGMA_H_BIN_OUT = OUTPUT_DATA / "parcel_displacement_sigma_h_f32.bin"

ANIMATION_MANIFEST_OUT = OUTPUT_DATA / "parcel_animation_manifest.json"
ANIMATION_SUMMARY_OUT = OUTPUT_DATA / "parcel_animation_summary.json"

REPORT_TXT_OUT = RUN_RECORDS / "phase05_animation_arrays_report.txt"
REPORT_JSON_OUT = RUN_RECORDS / "phase05_animation_arrays_report.json"

EXPECTED_TOTAL_PARCELS = expected_int(CONFIG, "total_parcels")
EXPECTED_MOVING_PARCELS = expected_int(CONFIG, "moving_parcels")
EXPECTED_BLANK_PARCELS = expected_int(CONFIG, "blank_parcels")
EXPECTED_EPOCHS = expected_int(CONFIG, "epochs")
EXPECTED_ROWS = None if EXPECTED_MOVING_PARCELS is None or EXPECTED_EPOCHS is None else EXPECTED_MOVING_PARCELS * EXPECTED_EPOCHS
EXPECTED_MESH_VERTICES = expected_int(CONFIG, "mesh_vertices")
EXPECTED_MESH_TRIANGLES = expected_int(CONFIG, "mesh_triangles")


def fail(message):
    print(f"[FAIL] {message}")
    sys.exit(1)


def ok(message):
    print(f"[OK] {message}")


def warn(message):
    print(f"[WARN] {message}")


def require_files(paths):
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        fail(f"Missing required files: {missing}")


def component_stats(arr):
    finite = np.isfinite(arr)
    if not finite.any():
        return {
            "finite_count": 0,
            "nan_count": int(np.isnan(arr).sum()),
            "min": None,
            "max": None,
            "mean": None,
        }

    vals = arr[finite]
    return {
        "finite_count": int(finite.sum()),
        "nan_count": int(np.isnan(arr).sum()),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "mean": float(vals.mean()),
    }


def write_float32_binary(path, arr):
    arr_le = np.ascontiguousarray(arr.astype("<f4", copy=False))
    arr_le.tofile(path)


def main():
    OUTPUT_DATA.mkdir(parents=True, exist_ok=True)
    RUN_RECORDS.mkdir(parents=True, exist_ok=True)

    print("\n=== PROTO2 PHASE 05: PACKAGE ANIMATION ARRAYS ===")
    print(f"Project root: {PROJECT_ROOT}")

    require_files([
        DISPLACEMENT_LONG,
        PARCEL_INVENTORY,
        MESH_VERTICES,
        MESH_TRIANGLES,
    ])

    print(f"\nReading canonical displacement product:\n  {DISPLACEMENT_LONG}")
    long = pd.read_parquet(DISPLACEMENT_LONG)

    required_disp_cols = [
        "pnt_id",
        "pnt_gid",
        "epoch",
        "epoch_index",
        "reversible",
        "irreversible",
        "total",
        "sigma_h",
        "pnt_lat",
        "pnt_lon",
        "vI",
        "std_vI",
        "var_vI",
    ]

    missing_disp_cols = [c for c in required_disp_cols if c not in long.columns]
    if missing_disp_cols:
        fail(f"displacement product missing required columns: {missing_disp_cols}")

    if EXPECTED_ROWS is not None and len(long) != EXPECTED_ROWS:
        fail(f"displacement row count {len(long):,} != expected {EXPECTED_ROWS:,}")

    long["pnt_id"] = pd.to_numeric(long["pnt_id"], errors="raise").astype("int64")
    long["pnt_gid"] = pd.to_numeric(long["pnt_gid"], errors="raise").astype("int64")
    long["epoch_index"] = pd.to_numeric(long["epoch_index"], errors="raise").astype("int64")

    for col in ["reversible", "irreversible", "total", "sigma_h", "pnt_lat", "pnt_lon", "vI", "std_vI", "var_vI"]:
        long[col] = pd.to_numeric(long[col], errors="coerce")

    moving_ids = np.sort(long["pnt_id"].unique())
    n_moving = len(moving_ids)

    if EXPECTED_MOVING_PARCELS is not None and n_moving != EXPECTED_MOVING_PARCELS:
        fail(f"moving parcel count {n_moving:,} != expected {EXPECTED_MOVING_PARCELS:,}")

    epoch_table = (
        long[["epoch_index", "epoch"]]
        .drop_duplicates()
        .sort_values("epoch_index")
        .reset_index(drop=True)
    )

    n_epochs = len(epoch_table)
    if EXPECTED_EPOCHS is not None and n_epochs != EXPECTED_EPOCHS:
        fail(f"epoch count {n_epochs:,} != expected {EXPECTED_EPOCHS:,}")

    expected_epoch_indices = np.arange(n_epochs)
    found_epoch_indices = epoch_table["epoch_index"].to_numpy(dtype=np.int64)

    if not np.array_equal(found_epoch_indices, expected_epoch_indices):
        fail("epoch_index is not contiguous from 0 to n_epochs-1")

    epoch_strings = epoch_table["epoch"].astype(str).to_numpy(dtype="U10")

    duplicate_count = int(long.duplicated(subset=["pnt_id", "epoch_index"]).sum())
    if duplicate_count:
        fail(f"duplicate pnt_id/epoch_index rows found: {duplicate_count:,}")

    ok(f"displacement rows loaded: {len(long):,}")
    ok(f"moving parcels: {n_moving:,}")
    ok(f"epochs: {n_epochs:,}")
    ok(f"epoch range: {epoch_strings[0]} to {epoch_strings[-1]}")

    print(f"\nReading parcel inventory:\n  {PARCEL_INVENTORY}")
    inventory = pd.read_parquet(
        PARCEL_INVENTORY,
        columns=["parcel_id", "parcel_status", "has_displacement"],
    )

    inventory["parcel_id"] = pd.to_numeric(inventory["parcel_id"], errors="raise").astype("int64")
    inventory["has_displacement"] = inventory["has_displacement"].astype(bool)

    if EXPECTED_TOTAL_PARCELS is not None and len(inventory) != EXPECTED_TOTAL_PARCELS:
        fail(f"parcel inventory count {len(inventory):,} != expected {EXPECTED_TOTAL_PARCELS:,}")

    inv_moving = int(inventory["has_displacement"].sum())
    inv_blank = int((~inventory["has_displacement"]).sum())

    if (EXPECTED_MOVING_PARCELS is not None and inv_moving != EXPECTED_MOVING_PARCELS) or (EXPECTED_BLANK_PARCELS is not None and inv_blank != EXPECTED_BLANK_PARCELS):
        fail(
            f"inventory moving/blank mismatch: moving={inv_moving:,}, blank={inv_blank:,}; "
            f"expected moving={EXPECTED_MOVING_PARCELS}, blank={EXPECTED_BLANK_PARCELS}"
        )

    ok(f"parcel inventory loaded: {len(inventory):,}")
    ok(f"inventory moving/blank confirmed: moving={inv_moving:,}, blank={inv_blank:,}")

    # Build moving parcel index: this is the row order used by animation matrices.
    moving_meta = (
        long.sort_values(["pnt_id", "epoch_index"])
        .drop_duplicates(subset=["pnt_id"])
        [[
            "pnt_id",
            "pnt_gid",
            "pnt_lat",
            "pnt_lon",
            "vI",
            "std_vI",
            "var_vI",
        ]]
        .sort_values("pnt_id")
        .reset_index(drop=True)
    )

    moving_meta["displacement_row_index"] = np.arange(len(moving_meta), dtype=np.int32)

    if not np.array_equal(moving_meta["pnt_id"].to_numpy(dtype=np.int64), moving_ids):
        fail("moving parcel index order does not match sorted moving_ids")

    parcel_to_disp_row = {
        int(pid): int(idx)
        for pid, idx in zip(
            moving_meta["pnt_id"].to_numpy(dtype=np.int64),
            moving_meta["displacement_row_index"].to_numpy(dtype=np.int32),
        )
    }

    # Build all-parcel render index.
    parcel_render_index = (
        inventory.sort_values("parcel_id")
        .reset_index(drop=True)
        .copy()
    )

    parcel_render_index["parcel_row_index"] = np.arange(len(parcel_render_index), dtype=np.int32)

    parcel_render_index = parcel_render_index.merge(
        moving_meta.rename(columns={"pnt_id": "parcel_id"}),
        how="left",
        on="parcel_id",
        validate="one_to_one",
    )

    parcel_render_index["displacement_row_index"] = (
        parcel_render_index["displacement_row_index"]
        .fillna(-1)
        .astype("int32")
    )

    missing_disp_on_moving = int(
        parcel_render_index.loc[
            parcel_render_index["has_displacement"],
            "displacement_row_index",
        ].eq(-1).sum()
    )

    disp_on_blank = int(
        parcel_render_index.loc[
            ~parcel_render_index["has_displacement"],
            "displacement_row_index",
        ].ne(-1).sum()
    )

    if missing_disp_on_moving or disp_on_blank:
        fail(
            f"render index mismatch: missing_disp_on_moving={missing_disp_on_moving:,}, "
            f"disp_on_blank={disp_on_blank:,}"
        )

    ok("parcel render index built: moving parcels have displacement rows; blanks have -1")

    # Build dense float32 animation matrices.
    print("\nBuilding parcel × epoch float32 matrices...")

    row_index_series = long["pnt_id"].map(parcel_to_disp_row)
    if row_index_series.isna().any():
        fail("some displacement rows could not be mapped to displacement_row_index")

    row_idx = row_index_series.to_numpy(dtype=np.int64)
    col_idx = long["epoch_index"].to_numpy(dtype=np.int64)

    if row_idx.min() < 0 or row_idx.max() >= n_moving:
        fail("row_idx out of range")

    if col_idx.min() < 0 or col_idx.max() >= n_epochs:
        fail("epoch index out of range")

    reversible = np.full((n_moving, n_epochs), np.nan, dtype=np.float32)
    irreversible = np.full((n_moving, n_epochs), np.nan, dtype=np.float32)
    total = np.full((n_moving, n_epochs), np.nan, dtype=np.float32)
    sigma_h = np.full((n_moving, n_epochs), np.nan, dtype=np.float32)

    reversible[row_idx, col_idx] = long["reversible"].to_numpy(dtype=np.float32)
    irreversible[row_idx, col_idx] = long["irreversible"].to_numpy(dtype=np.float32)
    total[row_idx, col_idx] = long["total"].to_numpy(dtype=np.float32)
    sigma_h[row_idx, col_idx] = long["sigma_h"].to_numpy(dtype=np.float32)

    expected_cells = n_moving * n_epochs

    for name, arr in [
        ("reversible", reversible),
        ("irreversible", irreversible),
        ("total", total),
    ]:
        finite_count = int(np.isfinite(arr).sum())
        if finite_count != expected_cells:
            fail(f"{name} matrix has {finite_count:,} finite cells; expected {expected_cells:,}")
        ok(f"{name} matrix complete: {finite_count:,} cells")

    sigma_finite = int(np.isfinite(sigma_h).sum())
    if sigma_finite == 0:
        ok("sigma_h matrix contains no finite values; uncertainty unavailable as intended")
    else:
        warn(f"sigma_h matrix has {sigma_finite:,} finite values; check whether Phase 04 uncertainty has been added")

    max_total_diff = float(np.nanmax(np.abs(total - (reversible + irreversible))))
    if max_total_diff > 1e-4:
        fail(f"total != reversible + irreversible after matrix build; max diff={max_total_diff:.9g}")

    ok(f"total matrix check passed; max abs diff={max_total_diff:.9g}")

    # Index mesh tables with parcel_row_index and displacement_row_index.
    print("\nIndexing mesh tables for renderer use...")

    print(f"Reading mesh vertices:\n  {MESH_VERTICES}")
    mesh_vertices = pd.read_parquet(MESH_VERTICES)

    print(f"Reading mesh triangles:\n  {MESH_TRIANGLES}")
    mesh_triangles = pd.read_parquet(MESH_TRIANGLES)

    if EXPECTED_MESH_VERTICES is not None and len(mesh_vertices) != EXPECTED_MESH_VERTICES:
        fail(f"mesh vertex count {len(mesh_vertices):,} != expected {EXPECTED_MESH_VERTICES:,}")

    if EXPECTED_MESH_TRIANGLES is not None and len(mesh_triangles) != EXPECTED_MESH_TRIANGLES:
        fail(f"mesh triangle count {len(mesh_triangles):,} != expected {EXPECTED_MESH_TRIANGLES:,}")

    index_cols = parcel_render_index[[
        "parcel_id",
        "parcel_row_index",
        "displacement_row_index",
    ]].copy()

    mesh_vertices_indexed = mesh_vertices.merge(
        index_cols,
        how="left",
        on="parcel_id",
        validate="many_to_one",
    )

    mesh_triangles_indexed = mesh_triangles.merge(
        index_cols,
        how="left",
        on="parcel_id",
        validate="many_to_one",
    )

    for name, df in [
        ("mesh vertices", mesh_vertices_indexed),
        ("mesh triangles", mesh_triangles_indexed),
    ]:
        if df["parcel_row_index"].isna().any():
            fail(f"{name} have missing parcel_row_index after join")

        if df["displacement_row_index"].isna().any():
            fail(f"{name} have missing displacement_row_index after join")

        df["parcel_row_index"] = df["parcel_row_index"].astype("int32")
        df["displacement_row_index"] = df["displacement_row_index"].astype("int32")

    vertex_moving_count = int(mesh_vertices_indexed["displacement_row_index"].ge(0).sum())
    vertex_blank_count = int(mesh_vertices_indexed["displacement_row_index"].eq(-1).sum())
    triangle_moving_count = int(mesh_triangles_indexed["displacement_row_index"].ge(0).sum())
    triangle_blank_count = int(mesh_triangles_indexed["displacement_row_index"].eq(-1).sum())

    ok(f"mesh vertices indexed: moving={vertex_moving_count:,}, blank={vertex_blank_count:,}")
    ok(f"mesh triangles indexed: moving={triangle_moving_count:,}, blank={triangle_blank_count:,}")

    # Write products.
    print("\nWriting animation/index outputs...")

    moving_meta.to_parquet(MOVING_PARCEL_INDEX_OUT, index=False)
    ok(f"wrote {MOVING_PARCEL_INDEX_OUT}")

    parcel_render_index.to_parquet(PARCEL_RENDER_INDEX_OUT, index=False)
    ok(f"wrote {PARCEL_RENDER_INDEX_OUT}")

    mesh_vertices_indexed.to_parquet(MESH_VERTICES_INDEXED_OUT, index=False)
    ok(f"wrote {MESH_VERTICES_INDEXED_OUT}")

    mesh_triangles_indexed.to_parquet(MESH_TRIANGLES_INDEXED_OUT, index=False)
    ok(f"wrote {MESH_TRIANGLES_INDEXED_OUT}")

    np.savez_compressed(
        ANIMATION_NPZ_OUT,
        reversible=reversible,
        irreversible=irreversible,
        total=total,
        sigma_h=sigma_h,
        moving_parcel_id=moving_ids.astype(np.int64),
        epoch=epoch_strings,
    )
    ok(f"wrote {ANIMATION_NPZ_OUT}")

    write_float32_binary(REVERSIBLE_BIN_OUT, reversible)
    ok(f"wrote {REVERSIBLE_BIN_OUT}")

    write_float32_binary(IRREVERSIBLE_BIN_OUT, irreversible)
    ok(f"wrote {IRREVERSIBLE_BIN_OUT}")

    write_float32_binary(TOTAL_BIN_OUT, total)
    ok(f"wrote {TOTAL_BIN_OUT}")

    write_float32_binary(SIGMA_H_BIN_OUT, sigma_h)
    ok(f"wrote {SIGMA_H_BIN_OUT}")

    stats = {
        "reversible": component_stats(reversible),
        "irreversible": component_stats(irreversible),
        "total": component_stats(total),
        "sigma_h": component_stats(sigma_h),
    }

    manifest = {
        "product": "parcel_animation_arrays",
        "version": 1,
        "source_displacement": str(DISPLACEMENT_LONG),
        "parcel_index_rule": {
            "moving_parcel_index": "row order for displacement matrices",
            "parcel_render_index": "all-parcel index; blank parcels have displacement_row_index = -1",
        },
        "shape": {
            "moving_parcels": int(n_moving),
            "epochs": int(n_epochs),
            "matrix_order": "row-major C order",
            "matrix_indexing": "matrix[displacement_row_index, epoch_index]",
        },
        "dtype": {
            "arrays": "float32 little-endian",
            "parcel_ids": "int64",
            "indices": "int32",
        },
        "epoch": {
            "start": str(epoch_strings[0]),
            "end": str(epoch_strings[-1]),
            "count": int(n_epochs),
        },
        "components": {
            "reversible": {
                "binary": str(REVERSIBLE_BIN_OUT),
                "meaning": "seasonal reversible deformation component in source displacement units",
            },
            "irreversible": {
                "binary": str(IRREVERSIBLE_BIN_OUT),
                "meaning": "irreversible cumulative deformation component in source displacement units",
            },
            "total": {
                "binary": str(TOTAL_BIN_OUT),
                "meaning": "reversible + irreversible in source displacement units",
            },
            "sigma_h": {
                "binary": str(SIGMA_H_BIN_OUT),
                "meaning": "per-epoch uncertainty of total displacement; NaN means unavailable",
            },
        },
        "combined_python_npz": str(ANIMATION_NPZ_OUT),
        "indices": {
            "moving_parcel_index": str(MOVING_PARCEL_INDEX_OUT),
            "parcel_render_index": str(PARCEL_RENDER_INDEX_OUT),
            "mesh_vertices_indexed": str(MESH_VERTICES_INDEXED_OUT),
            "mesh_triangles_indexed": str(MESH_TRIANGLES_INDEXED_OUT),
        },
        "blank_parcel_rule": {
            "has_displacement": False,
            "displacement_row_index": -1,
            "viewer_interpretation": "show as blank/no-data geometry; do not sample animation arrays",
        },
        "units_note": {
            "displacement_values": "source displacement units; no unit conversion applied in this phase",
            "height_scaling": "viewer/pipeline stage later should decide unit conversion and vertical exaggeration",
        },
    }

    summary = {
        "source_displacement": str(DISPLACEMENT_LONG),
        "total_parcels": int(len(inventory)),
        "moving_parcels": int(n_moving),
        "blank_parcels": int(inv_blank),
        "epochs": int(n_epochs),
        "matrix_shape": [int(n_moving), int(n_epochs)],
        "matrix_cells_per_component": int(expected_cells),
        "epoch_start": str(epoch_strings[0]),
        "epoch_end": str(epoch_strings[-1]),
        "max_total_diff_after_matrix_build": max_total_diff,
        "component_stats": stats,
        "mesh_vertices_indexed": int(len(mesh_vertices_indexed)),
        "mesh_triangles_indexed": int(len(mesh_triangles_indexed)),
        "mesh_vertex_moving_count": vertex_moving_count,
        "mesh_vertex_blank_count": vertex_blank_count,
        "mesh_triangle_moving_count": triangle_moving_count,
        "mesh_triangle_blank_count": triangle_blank_count,
        "outputs": {
            "moving_parcel_index": str(MOVING_PARCEL_INDEX_OUT),
            "parcel_render_index": str(PARCEL_RENDER_INDEX_OUT),
            "mesh_vertices_indexed": str(MESH_VERTICES_INDEXED_OUT),
            "mesh_triangles_indexed": str(MESH_TRIANGLES_INDEXED_OUT),
            "animation_npz": str(ANIMATION_NPZ_OUT),
            "reversible_bin": str(REVERSIBLE_BIN_OUT),
            "irreversible_bin": str(IRREVERSIBLE_BIN_OUT),
            "total_bin": str(TOTAL_BIN_OUT),
            "sigma_h_bin": str(SIGMA_H_BIN_OUT),
            "manifest": str(ANIMATION_MANIFEST_OUT),
            "summary": str(ANIMATION_SUMMARY_OUT),
        },
    }

    ANIMATION_MANIFEST_OUT.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    ok(f"wrote {ANIMATION_MANIFEST_OUT}")

    ANIMATION_SUMMARY_OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    ok(f"wrote {ANIMATION_SUMMARY_OUT}")

    report = {
        "project_root": str(PROJECT_ROOT),
        "summary": summary,
        "manifest": manifest,
    }

    REPORT_JSON_OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    txt_lines = [
        "PROTO2 PHASE 05 ANIMATION ARRAYS REPORT",
        "",
        f"source displacement: {DISPLACEMENT_LONG}",
        "",
        f"moving parcels: {n_moving:,}",
        f"blank parcels: {inv_blank:,}",
        f"epochs: {n_epochs:,}",
        f"matrix shape: {n_moving:,} x {n_epochs:,}",
        f"cells per component: {expected_cells:,}",
        f"epoch range: {epoch_strings[0]} to {epoch_strings[-1]}",
        "",
        "component ranges:",
        f"reversible: {stats['reversible']['min']} to {stats['reversible']['max']}",
        f"irreversible: {stats['irreversible']['min']} to {stats['irreversible']['max']}",
        f"total: {stats['total']['min']} to {stats['total']['max']}",
        f"sigma_h finite count: {stats['sigma_h']['finite_count']}",
        "",
        f"max total diff after matrix build: {max_total_diff:.9g}",
        "",
        "indexed mesh:",
        f"vertices: {len(mesh_vertices_indexed):,}",
        f"triangles: {len(mesh_triangles_indexed):,}",
        "",
        "outputs:",
        f"- {MOVING_PARCEL_INDEX_OUT}",
        f"- {PARCEL_RENDER_INDEX_OUT}",
        f"- {MESH_VERTICES_INDEXED_OUT}",
        f"- {MESH_TRIANGLES_INDEXED_OUT}",
        f"- {ANIMATION_NPZ_OUT}",
        f"- {REVERSIBLE_BIN_OUT}",
        f"- {IRREVERSIBLE_BIN_OUT}",
        f"- {TOTAL_BIN_OUT}",
        f"- {SIGMA_H_BIN_OUT}",
        f"- {ANIMATION_MANIFEST_OUT}",
    ]

    REPORT_TXT_OUT.write_text("\n".join(txt_lines), encoding="utf-8")
    ok(f"wrote {REPORT_JSON_OUT}")
    ok(f"wrote {REPORT_TXT_OUT}")

    print("\n=== SUMMARY ===")
    print(f"Animation matrix shape: {n_moving:,} parcels x {n_epochs:,} epochs")
    print(f"Cells per component: {expected_cells:,}")
    print(f"Reversible range: {stats['reversible']['min']:.6f} to {stats['reversible']['max']:.6f}")
    print(f"Irreversible range: {stats['irreversible']['min']:.6f} to {stats['irreversible']['max']:.6f}")
    print(f"Total range: {stats['total']['min']:.6f} to {stats['total']['max']:.6f}")
    print(f"Sigma_h finite count: {stats['sigma_h']['finite_count']:,}")
    print("\nPHASE 05 RESULT: PASS. Animation arrays and render indices packaged.")


if __name__ == "__main__":
    main()