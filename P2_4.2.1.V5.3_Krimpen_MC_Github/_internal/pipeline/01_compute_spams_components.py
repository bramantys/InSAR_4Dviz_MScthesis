#!/usr/bin/env python3
"""Build the direct-SPAMS deformation input bundle for Prototype 2 Batch 1.

Inputs
------
- supplied SPAMS parameter Parquet
- supplied ``utils.py`` (`read_knmi`, `spams_model`)
- local KNMI station 344 / 348 daily files
- supplied Monte Carlo NPZ (`mean_t`, `sigma_t`)

Output
------
A single canonical [parcel, epoch] NPZ used by the remaining pipeline. It
contains deterministic reversible / irreversible / deterministic_total and
the Monte Carlo total_mean / total_sigma. This NPZ is an internal
pipeline intermediate, not a browser asset.
"""
from __future__ import annotations

import importlib.util
import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from _mc_total import MonteCarloContractError, validate_and_load_monte_carlo_total
from _proto2_config import input_path, load_project_config, output_data_dir, project_root_from, stage_records_dir

PROJECT_ROOT = project_root_from(__file__)
CONFIG = load_project_config(PROJECT_ROOT)
OUTPUT_DATA = output_data_dir(PROJECT_ROOT, CONFIG)
RUN_RECORDS = stage_records_dir(PROJECT_ROOT, CONFIG)
MODEL_PARAMS_PARQUET = input_path(PROJECT_ROOT, CONFIG, "model_parameters", "parquet_path", "data/model_params/nl_krimpenerwaard_spams10.parquet")
PYSPAMS_DIR = input_path(PROJECT_ROOT, CONFIG, "pyspams", "directory", "data/pyspams")
CANONICAL_NPZ_OUT = OUTPUT_DATA / "spams_viewer_input_slice_float32.npz"
CANONICAL_MANIFEST_OUT = OUTPUT_DATA / "spams_viewer_input_slice_manifest.json"
EPOCH_AXIS_OUT = OUTPUT_DATA / "epoch_axis.json"
SUMMARY_OUT = OUTPUT_DATA / "spams_direct_summary.json"
REPORT_JSON_OUT = RUN_RECORDS / "phase01_direct_spams_report.json"
REPORT_TXT_OUT = RUN_RECORDS / "phase01_direct_spams_report.txt"


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    raise SystemExit(1)


def ok(message: str) -> None:
    print(f"[OK] {message}")


def load_utils(path: Path):
    spec = importlib.util.spec_from_file_location("proto2_pyspams_utils", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load PySPAMS utils.py: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stats(values: np.ndarray) -> dict[str, float | int | None]:
    finite = np.isfinite(values)
    if not finite.any():
        return {"finite_count": 0, "min": None, "max": None, "mean": None}
    selected = values[finite]
    return {
        "finite_count": int(finite.sum()),
        "min": float(selected.min()),
        "max": float(selected.max()),
        "mean": float(selected.mean()),
    }


def knmi_paths() -> dict[int, Path]:
    raw = CONFIG.get("user_inputs", {}).get("knmi_daily_files")
    if not isinstance(raw, dict) or not raw:
        fail("user_inputs.knmi_daily_files must be a station-to-file mapping")
    result: dict[int, Path] = {}
    for key, value in raw.items():
        try:
            station = int(key)
        except Exception:
            fail(f"KNMI station key is not an integer: {key!r}")
        if not isinstance(value, str) or not value.strip():
            fail(f"KNMI station {station} path must be a non-empty string")
        result[station] = PROJECT_ROOT / value
    return result


def weather_slice(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, tau: int) -> pd.DataFrame:
    first = start - timedelta(days=tau - 1)
    target = df[(df["datum"] >= first) & (df["datum"] <= end)].copy()
    expected = pd.date_range(first, end, freq="D")
    actual = pd.DatetimeIndex(target["datum"].dt.normalize())
    if len(target) != len(expected) or not actual.equals(expected):
        missing = len(expected.difference(actual.unique()))
        duplicated = int(actual.duplicated().sum())
        fail(
            f"KNMI weather slice is not a continuous daily sequence for tau={tau}: "
            f"{first.date()} to {end.date()}, rows={len(target)}, expected={len(expected)}, "
            f"missing={missing}, duplicated={duplicated}"
        )
    if target[["precip", "evapo"]].isna().any().any():
        fail(f"KNMI weather slice contains NaN precipitation/evaporation for tau={tau}")
    return target


def main() -> None:
    OUTPUT_DATA.mkdir(parents=True, exist_ok=True)
    RUN_RECORDS.mkdir(parents=True, exist_ok=True)
    print("\n=== PROTO2 PHASE 01: DIRECT SPAMS BATCH + MONTE CARLO TOTAL ===")

    if not MODEL_PARAMS_PARQUET.is_file():
        fail(f"Missing SPAMS parameter Parquet: {MODEL_PARAMS_PARQUET}")
    utils_path = PYSPAMS_DIR / "utils.py"
    if not utils_path.is_file():
        fail(f"Missing PySPAMS utils.py: {utils_path}")
    utils = load_utils(utils_path)
    if not callable(getattr(utils, "spams_model", None)) or not callable(getattr(utils, "read_knmi", None)):
        fail("utils.py must expose read_knmi() and spams_model()")

    try:
        mc = validate_and_load_monte_carlo_total(PROJECT_ROOT, CONFIG, load_arrays=True)
    except MonteCarloContractError as exc:
        fail(f"supplied Monte Carlo total contract failed: {exc}")
    if mc is None:
        fail("Batch 1 requires the Monte Carlo total NPZ")

    required_cols = ["pnt_id", "pnt_gid", "pnt_lat", "pnt_lon", "xP", "xE", "xI", "tau", "meteo_id", "vI", "var_vI"]
    try:
        params = pd.read_parquet(MODEL_PARAMS_PARQUET, columns=required_cols)
    except Exception as exc:
        fail(f"Could not read direct SPAMS parameters: {exc}")
    for col in ["pnt_id", "pnt_gid", "tau", "meteo_id"]:
        params[col] = pd.to_numeric(params[col], errors="raise").astype("int64")
    for col in ["pnt_lat", "pnt_lon", "xP", "xE", "xI", "vI", "var_vI"]:
        params[col] = pd.to_numeric(params[col], errors="raise").astype(float)
    if not params["pnt_id"].is_unique:
        fail("SPAMS parameter Parquet contains duplicate pnt_id values")

    moving_ids = mc.parcel_ids.astype(np.int64, copy=False)
    try:
        ordered = params.set_index("pnt_id").loc[moving_ids].reset_index()
    except KeyError as exc:
        fail(f"Monte Carlo NPZ row IDs cannot be aligned to parameter Parquet: {exc}")
    if not np.array_equal(ordered["pnt_id"].to_numpy(dtype=np.int64), moving_ids):
        fail("Parameter Parquet order could not be aligned to Monte Carlo row order")

    epoch_labels = mc.epoch_labels.astype("U10", copy=False)
    target_start = pd.Timestamp(str(epoch_labels[0]))
    target_end = pd.Timestamp(str(epoch_labels[-1]))
    n_moving, n_epochs = len(ordered), len(epoch_labels)
    print(f"Target viewer period: {target_start.date()} to {target_end.date()} ({n_epochs:,} daily epochs)")
    print(f"SPAMS moving parcels: {n_moving:,}")

    paths = knmi_paths()
    stations = sorted(ordered["meteo_id"].unique().tolist())
    meteo_by_station: dict[int, pd.DataFrame] = {}
    for station in stations:
        path = paths.get(int(station))
        if path is None or not path.is_file():
            fail(f"Missing configured KNMI file for required station {station}")
        try:
            frame = utils.read_knmi(path)
        except Exception as exc:
            fail(f"KNMI station {station} could not be parsed by utils.read_knmi(): {exc}")
        frame = frame.sort_values("datum", kind="mergesort").reset_index(drop=True)
        station_values = set(frame["meteo_id"].astype(int).unique().tolist())
        if station_values != {int(station)}:
            fail(f"Configured KNMI file {path} does not contain station {station}: got {station_values}")
        meteo_by_station[int(station)] = frame
        ok(f"loaded KNMI station {station}: {frame['datum'].min().date()} to {frame['datum'].max().date()} ({len(frame):,} days)")

    reversible = np.empty((n_moving, n_epochs), dtype=np.float32)
    irreversible = np.empty((n_moving, n_epochs), dtype=np.float32)
    deterministic_total = np.empty((n_moving, n_epochs), dtype=np.float32)

    print("\nRunning utils.spams_model() for every parcel...")
    group_counts: dict[str, int] = {}
    for row_index, row in ordered.iterrows():
        tau = int(row["tau"])
        station = int(row["meteo_id"])
        meteorology = weather_slice(meteo_by_station[station], target_start, target_end, tau)
        rev, irr, total = utils.spams_model(float(row["xP"]), float(row["xE"]), float(row["xI"]), tau, meteorology)
        if len(rev) != n_epochs or len(irr) != n_epochs or len(total) != n_epochs:
            fail(
                f"SPAMS output length mismatch for pnt_id={int(row['pnt_id'])}: "
                f"rev={len(rev)}, irr={len(irr)}, total={len(total)}, expected={n_epochs}"
            )
        rev_arr = np.asarray(rev, dtype=np.float32)
        irr_arr = np.asarray(irr, dtype=np.float32)
        total_arr = np.asarray(total, dtype=np.float32)
        if not np.isfinite(rev_arr).all() or not np.isfinite(irr_arr).all() or not np.isfinite(total_arr).all():
            fail(f"SPAMS output contains non-finite values for pnt_id={int(row['pnt_id'])}")
        max_diff = float(np.max(np.abs(total_arr - (rev_arr + irr_arr))))
        if max_diff > 1e-4:
            fail(f"SPAMS total != reversible + irreversible for pnt_id={int(row['pnt_id'])}; max diff={max_diff:.9g}")
        reversible[row_index] = rev_arr
        irreversible[row_index] = irr_arr
        deterministic_total[row_index] = total_arr
        key = f"station_{station}_tau_{tau}"
        group_counts[key] = group_counts.get(key, 0) + 1
        if (row_index + 1) % 500 == 0 or row_index + 1 == n_moving:
            print(f"  {row_index + 1:,}/{n_moving:,} parcels")

    total_mean = np.ascontiguousarray(mc.mean_total.astype(np.float32, copy=False))
    total_sigma = np.ascontiguousarray(mc.sigma_total.astype(np.float32, copy=False))
    if total_mean.shape != reversible.shape or total_sigma.shape != reversible.shape:
        fail(
            f"Monte Carlo and direct SPAMS shapes differ: direct={reversible.shape}, "
            f"mean={total_mean.shape}, sigma={total_sigma.shape}"
        )
    if not np.isfinite(total_mean).all() or not np.isfinite(total_sigma).all() or np.any(total_sigma < 0.0):
        fail("Monte Carlo total/sigma contains non-finite values or negative sigma")

    np.savez_compressed(
        CANONICAL_NPZ_OUT,
        reversible=np.ascontiguousarray(reversible),
        irreversible=np.ascontiguousarray(irreversible),
        deterministic_total=np.ascontiguousarray(deterministic_total),
        total_mean=total_mean,
        total_sigma=total_sigma,
        moving_parcel_id=moving_ids,
        epoch=epoch_labels,
    )
    ok(f"wrote {CANONICAL_NPZ_OUT}")

    epoch_axis = {
        "schema": "proto2_epoch_axis_v4_2_direct_spams",
        "epoch_count": int(n_epochs),
        "start": str(target_start.date()),
        "end": str(target_end.date()),
        "epochs": [{"epoch_index": int(i), "epoch": str(label)} for i, label in enumerate(epoch_labels.tolist())],
        "time_reference": {
            "source_reconstruction_start": mc.source_start_date,
            "source_reconstruction_end": mc.source_end_date,
            "source_reference_date": mc.source_reference_date,
            "default_viewer_reference_date": mc.default_reference_date,
        },
        "note": "Direct deterministic SPAMS components and Monte Carlo Total share this exact viewer axis.",
    }
    EPOCH_AXIS_OUT.write_text(json.dumps(epoch_axis, indent=2), encoding="utf-8")
    ok(f"wrote {EPOCH_AXIS_OUT}")

    max_component_diff = float(np.max(np.abs(deterministic_total - (reversible + irreversible))))
    manifest = {
        "schema": "proto2_spams_viewer_input_v1",
        "input_bundle": str(CANONICAL_NPZ_OUT),
        "shape": [int(n_moving), int(n_epochs)],
        "matrix_indexing": "matrix[displacement_row_index, epoch_index]",
        "parcel_id_array": "moving_parcel_id",
        "epoch_array": "epoch",
        "products": {
            "reversible": "direct deterministic SPAMS component from Parquet + KNMI + utils.spams_model",
            "irreversible": "direct deterministic SPAMS component from Parquet + KNMI + utils.spams_model",
            "deterministic_total": "direct deterministic SPAMS reversible + irreversible",
            "total_mean": "supplied Monte Carlo mean_t of Total displacement",
            "total_sigma": "supplied Monte Carlo sigma_t for Total displacement only",
        },
        "time": {"start": str(target_start.date()), "end": str(target_end.date()), "epochs": int(n_epochs)},
        "checks": {
            "direct_component_max_abs_total_diff": max_component_diff,
            "total_sigma_negative_count": int((total_sigma < 0.0).sum()),
        },
        "notes": [
            "Total MC mean and deterministic_total are intentionally distinct products: the former is the Monte Carlo central total curve, while the latter is the direct deterministic component sum.",
            "total_sigma is valid for Total only. It must not be attached to reversible, irreversible, or combined component views.",
        ],
    }
    CANONICAL_MANIFEST_OUT.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    ok(f"wrote {CANONICAL_MANIFEST_OUT}")

    summary = {
        "schema": "proto2_direct_spams_summary_v1",
        "source": {
            "parameters_parquet": str(MODEL_PARAMS_PARQUET),
            "utils_py": str(utils_path),
            "knmi_files": {str(station): str(paths[station]) for station in stations},
            "kaan_monte_carlo": mc.audit,
        },
        "output": {"canonical_npz": str(CANONICAL_NPZ_OUT), "shape": [int(n_moving), int(n_epochs)]},
        "component_stats": {
            "reversible": stats(reversible),
            "irreversible": stats(irreversible),
            "deterministic_total": stats(deterministic_total),
            "total_mean": stats(total_mean),
            "total_sigma": stats(total_sigma),
        },
        "station_tau_groups": group_counts,
        "notes": manifest["notes"],
    }
    SUMMARY_OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    REPORT_JSON_OUT.write_text(json.dumps({"summary": summary, "manifest": manifest}, indent=2), encoding="utf-8")
    REPORT_TXT_OUT.write_text(
        "\n".join([
            "PROTO2 PHASE 01 DIRECT SPAMS REPORT", "",
            f"viewer period: {target_start.date()} to {target_end.date()} ({n_epochs:,} epochs)",
            f"moving parcels: {n_moving:,}",
            f"canonical direct+MC input: {CANONICAL_NPZ_OUT}",
            f"direct total max component difference: {max_component_diff:.9g}",
            "", "RESULT: PASS",
        ]),
        encoding="utf-8",
    )
    ok(f"wrote {SUMMARY_OUT}")
    ok(f"wrote {REPORT_JSON_OUT}")
    ok(f"wrote {REPORT_TXT_OUT}")
    print("\nPHASE 01 RESULT: PASS. Direct deterministic SPAMS and Monte Carlo Total share one runtime axis.")


if __name__ == "__main__":
    main()
