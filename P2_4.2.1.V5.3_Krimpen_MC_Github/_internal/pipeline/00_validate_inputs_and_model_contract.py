#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from _mc_total import MonteCarloContractError, monte_carlo_enabled, validate_and_load_monte_carlo_total
from _proto2_config import (
    active_deformation_source,
    declared_crs,
    geometry_id_candidates,
    input_path,
    load_project_config,
    project_root_from,
    pyspams_automated,
    stage_records_dir,
)

PROJECT_ROOT = project_root_from(__file__)
CONFIG = load_project_config(PROJECT_ROOT)
STAGE_RECORDS = stage_records_dir(PROJECT_ROOT, CONFIG)
SHP = input_path(PROJECT_ROOT, CONFIG, "parcel_geometry", "path", "data/shapefile/krimpenerwaard_attributes_wgs84.shp")
PARAMS_PARQUET = input_path(PROJECT_ROOT, CONFIG, "model_parameters", "parquet_path", "data/model_params/nl_krimpenerwaard_spams10.parquet")
PARAMS_JSON = input_path(PROJECT_ROOT, CONFIG, "model_parameters", "metadata_path", "data/model_params/nl_krimpenerwaard_spams10.json")
PYSPAMS_DIR = input_path(PROJECT_ROOT, CONFIG, "pyspams", "directory", "data/pyspams")


def ok(message: str) -> None:
    print(f"[OK ] {message}")


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    raise SystemExit(1)


def check(condition: bool, message: str) -> None:
    if condition:
        ok(message)
    else:
        fail(message)


def normalize_epsg(value: str) -> str:
    return value.strip().upper().replace(" ", "")


def load_utils(path: Path):
    spec = importlib.util.spec_from_file_location("proto2_pyspams_utils", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load PySPAMS utils.py: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config_dates() -> tuple[pd.Timestamp, pd.Timestamp]:
    viewer = CONFIG.get("time_settings", {}).get("viewer_period", {})
    start = pd.Timestamp(viewer.get("start_date"))
    end = pd.Timestamp(viewer.get("end_date"))
    if pd.isna(start) or pd.isna(end) or end < start:
        fail("time_settings.viewer_period start_date/end_date are invalid")
    return start.normalize(), end.normalize()


def knmi_paths() -> dict[int, Path]:
    raw = CONFIG.get("user_inputs", {}).get("knmi_daily_files")
    if not isinstance(raw, dict) or not raw:
        fail("user_inputs.knmi_daily_files must be an object mapping station IDs to file paths")
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


def daily_coverage_check(df: pd.DataFrame, *, station: int, start: pd.Timestamp, end: pd.Timestamp) -> tuple[int, int, int]:
    subset = df[(df["datum"] >= start) & (df["datum"] <= end)].copy()
    expected = pd.date_range(start, end, freq="D")
    dates = pd.DatetimeIndex(subset["datum"].dt.normalize())
    duplicate_count = int(dates.duplicated().sum())
    missing_count = int(len(expected.difference(dates.unique())))
    nan_count = int(subset[["precip", "evapo"]].isna().sum().sum())
    if len(subset) != len(expected) or duplicate_count or missing_count or nan_count:
        fail(
            f"KNMI station {station} does not provide a clean daily weather sequence "
            f"for {start.date()} to {end.date()}: rows={len(subset)}, expected={len(expected)}, "
            f"duplicates={duplicate_count}, missing={missing_count}, NaN={nan_count}"
        )
    return len(subset), duplicate_count, missing_count


def main() -> None:
    STAGE_RECORDS.mkdir(parents=True, exist_ok=True)
    print("\n=== PROTO2 PHASE 0: DIRECT SPAMS SOURCE CONTRACT CHECK ===")
    print(f"Project root: {PROJECT_ROOT}")

    check(active_deformation_source(CONFIG) == "spams_parquet_knmi", "deterministic component source: spams_parquet_knmi")
    check(pyspams_automated(CONFIG), "direct PySPAMS calculation is enabled")

    required_sidecars = CONFIG.get("input_contract", {}).get("required_shapefile_sidecars", [".shp", ".dbf", ".shx", ".prj"])
    required_files = [PARAMS_PARQUET, PARAMS_JSON, *[SHP.with_suffix(str(ext)) for ext in required_sidecars]]
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        fail("required user input files are missing:\n  - " + "\n  - ".join(missing))
    ok("configured Parquet, SPAMS metadata JSON, shapefile, and sidecars are present")

    if not PYSPAMS_DIR.is_dir():
        fail(f"configured PySPAMS directory is missing: {PYSPAMS_DIR}")
    required_pyspams = CONFIG.get("input_contract", {}).get("required_pyspams_files", ["utils.py"])
    missing_pyspams = [str(PYSPAMS_DIR / name) for name in required_pyspams if not (PYSPAMS_DIR / name).is_file()]
    if missing_pyspams:
        fail("required PySPAMS files are missing:\n  - " + "\n  - ".join(missing_pyspams))
    utils = load_utils(PYSPAMS_DIR / "utils.py")
    check(callable(getattr(utils, "spams_model", None)), "utils.py exposes spams_model()")
    check(callable(getattr(utils, "read_knmi", None)), "utils.py exposes read_knmi()")

    try:
        metadata_payload = json.loads(PARAMS_JSON.read_text(encoding="utf-8"))
    except Exception as exc:
        fail(f"SPAMS metadata JSON cannot be read: {PARAMS_JSON}: {exc}")
    check(isinstance(metadata_payload, dict), "SPAMS metadata JSON is a JSON object")

    check(normalize_epsg(declared_crs(CONFIG, "parcel_crs")) == "EPSG:4326", "configured parcel CRS is EPSG:4326")
    gdf = gpd.read_file(SHP)
    ok(f"parcel shapefile opens: {len(gdf):,} features")
    check(gdf.crs is not None and gdf.crs.to_epsg() == 4326, f"shapefile CRS matches EPSG:4326: {gdf.crs}")
    join_col = next((name for name in geometry_id_candidates(CONFIG) if name in gdf.columns), None)
    if join_col is None:
        fail(f"parcel ID field missing; tried {geometry_id_candidates(CONFIG)}")
    geom_ids = pd.to_numeric(gdf[join_col], errors="coerce")
    check(not geom_ids.isna().any(), f"parcel ID field is numeric: {join_col}")
    check(int(geom_ids.nunique()) == len(gdf), "parcel IDs are unique")
    invalid = int((~gdf.geometry.is_valid).sum())
    empty = int(gdf.geometry.is_empty.sum())
    null = int(gdf.geometry.isna().sum())
    check(invalid == 0 and empty == 0 and null == 0, f"parcel geometry valid: invalid={invalid}, empty={empty}, null={null}")

    required_cols = CONFIG.get("input_contract", {}).get("required_spams_parameter_columns", [])
    if not isinstance(required_cols, list) or not required_cols:
        fail("input_contract.required_spams_parameter_columns must be a non-empty list")
    try:
        params = pd.read_parquet(PARAMS_PARQUET, columns=required_cols)
    except Exception as exc:
        fail(f"SPAMS parameter Parquet does not satisfy the required column contract: {exc}")
    check(len(params) > 0, "SPAMS parameter Parquet has rows")
    params["pnt_id"] = pd.to_numeric(params["pnt_id"], errors="raise").astype("int64")
    check(params["pnt_id"].is_unique, f"SPAMS parameter Parquet has {len(params):,} unique pnt_id values")
    numeric_cols = [c for c in ["xP", "xE", "xI", "tau", "meteo_id", "vI", "var_vI"] if c in params.columns]
    for column in numeric_cols:
        params[column] = pd.to_numeric(params[column], errors="coerce")
    check(not params[numeric_cols].isna().any().any(), "SPAMS parameter values required for Batch 1 are finite")
    check((params["tau"] >= 1).all() and np.allclose(params["tau"], np.rint(params["tau"])), "SPAMS tau values are positive integer days")
    param_ids = set(params["pnt_id"].tolist())
    check(not (param_ids - set(geom_ids.astype("int64").tolist())), "every SPAMS parcel has geometry")

    metadata_stations = metadata_payload.get("meteo_id", [])
    metadata_stations = {int(value) for value in metadata_stations} if isinstance(metadata_stations, list) else set()
    parquet_stations = {int(value) for value in params["meteo_id"].astype(int).unique().tolist()}
    check(parquet_stations == metadata_stations, f"SPAMS metadata stations match Parquet: {sorted(parquet_stations)}")

    knmi_file_map = knmi_paths()
    required_stations = set(CONFIG.get("input_contract", {}).get("required_knmi_station_ids", sorted(parquet_stations)))
    required_stations = {int(value) for value in required_stations}
    check(required_stations == parquet_stations, f"configured KNMI stations match Parquet: {sorted(required_stations)}")
    missing_stations = sorted(required_stations - set(knmi_file_map))
    if missing_stations:
        fail(f"KNMI station files are not configured for stations: {missing_stations}")

    viewer_start, viewer_end = config_dates()
    station_max_tau = params.groupby(params["meteo_id"].astype(int))["tau"].max().astype(int).to_dict()
    weather_audit: dict[str, object] = {}
    for station in sorted(required_stations):
        weather_path = knmi_file_map[station]
        if not weather_path.is_file():
            fail(f"KNMI station {station} file is missing: {weather_path}")
        try:
            weather = utils.read_knmi(weather_path)
        except Exception as exc:
            fail(f"KNMI station {station} could not be read by utils.read_knmi(): {exc}")
        station_values = set(pd.to_numeric(weather["meteo_id"], errors="coerce").dropna().astype(int).unique().tolist())
        check(station_values == {station}, f"KNMI file station identity matches {station}")
        max_tau = int(station_max_tau[station])
        required_start = viewer_start - timedelta(days=max_tau - 1)
        rows, duplicates, missing_dates = daily_coverage_check(weather, station=station, start=required_start, end=viewer_end)
        weather_audit[str(station)] = {
            "path": str(weather_path),
            "max_tau_days": max_tau,
            "required_weather_start": str(required_start.date()),
            "viewer_end": str(viewer_end.date()),
            "validated_days": rows,
            "duplicate_dates": duplicates,
            "missing_dates": missing_dates,
        }
        ok(f"KNMI {station} covers direct SPAMS weather need: {required_start.date()} to {viewer_end.date()} ({rows:,} days)")

    if not monte_carlo_enabled(CONFIG):
        fail("Batch 1 requires monte_carlo_total.enabled = true")
    try:
        mc = validate_and_load_monte_carlo_total(PROJECT_ROOT, CONFIG, load_arrays=False)
    except MonteCarloContractError as exc:
        fail(f"supplied Monte Carlo total contract failed: {exc}")
    assert mc is not None
    mc_ids = set(mc.parcel_ids.astype("int64").tolist())
    check(mc_ids == param_ids, "supplied Monte Carlo row IDs match SPAMS parameter Parquet IDs")
    ok(
        "supplied Monte Carlo source valid: "
        f"{mc.source_start_date} to {mc.source_end_date}; viewer slice "
        f"{mc.viewer_start_date} to {mc.viewer_end_date} ({len(mc.epoch_labels):,} epochs)"
    )
    ok(f"supplied Monte Carlo default displayed reference date: {mc.default_reference_date}")
    ok("supplied sigma_t policy: selected epoch columns are copied unchanged; sigma_t is never applied to components")

    summary = {
        "schema": "proto2_phase0_direct_spams_v1",
        "status": "PASS",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "deformation_source": active_deformation_source(CONFIG),
        "viewer_period": {"start_date": str(viewer_start.date()), "end_date": str(viewer_end.date())},
        "parcel_geometry": {"features": int(len(gdf)), "join_field": join_col},
        "spams_parameters": {"rows": int(len(params)), "stations": sorted(parquet_stations), "tau_min": int(params["tau"].min()), "tau_max": int(params["tau"].max())},
        "weather": weather_audit,
        "monte_carlo_total": mc.audit,
        "notes": [
            "Deterministic reversible and irreversible arrays will be calculated directly from Parquet + local KNMI + utils.spams_model.",
            "supplied NPZ supplies Monte Carlo total mean_t and matching sigma_t only.",
            "No CSV component source is part of the Batch 1 contract.",
        ],
    }
    report_path = STAGE_RECORDS / "phase0_sanity_report.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== PHASE 0 RESULT: PASS ===")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
