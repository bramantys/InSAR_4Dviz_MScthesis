#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import pandas as pd

from _proto2_config import (
    active_deformation_source,
    declared_crs,
    geometry_id_candidates,
    input_path,
    load_project_config,
    project_root_from,
    pyspams_automated,
    source_columns,
    stage_records_dir,
)


PROJECT_ROOT = project_root_from(__file__)
CONFIG = load_project_config(PROJECT_ROOT)
STAGE_RECORDS = stage_records_dir(PROJECT_ROOT, CONFIG)

SHP = input_path(PROJECT_ROOT, CONFIG, "parcel_geometry", "path", "data/shapefile/krimpenerwaard_attributes_wgs84.shp")
DISP_CSV = input_path(PROJECT_ROOT, CONFIG, "displacement", "path", "data/displacement/example_spams_model_2025.csv")
PARAMS_PARQUET = input_path(PROJECT_ROOT, CONFIG, "model_parameters", "parquet_path", "data/model_params/nl_krimpenerwaard_spams10.parquet")
PARAMS_JSON = input_path(PROJECT_ROOT, CONFIG, "model_parameters", "metadata_path", "data/model_params/nl_krimpenerwaard_spams10.json")
PYSPAMS_DIR = input_path(PROJECT_ROOT, CONFIG, "pyspams", "directory", "data/pyspams")


def ok(message: str) -> None:
    print(f"[OK ] {message}")


def warn(message: str) -> None:
    print(f"[WARN] {message}")


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    raise SystemExit(1)


def check(condition: bool, message: str) -> None:
    if condition:
        ok(message)
    else:
        fail(message)


def parse_epoch_series(epoch_series: pd.Series) -> tuple[pd.Series, str, int]:
    raw = epoch_series.astype(str).str.strip()
    candidate_formats = [
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%d-%m-%y",
        "%Y/%m/%d",
        "%d/%m/%Y",
        "%d/%m/%y",
        "%m/%d/%Y",
        "%m/%d/%y",
    ]
    best_parsed = None
    best_label = None
    best_fail_count = len(raw) + 1
    for fmt in candidate_formats:
        parsed = pd.to_datetime(raw, format=fmt, errors="coerce")
        fail_count = int(parsed.isna().sum())
        if fail_count < best_fail_count:
            best_parsed = parsed
            best_label = fmt
            best_fail_count = fail_count
        if fail_count == 0:
            break
    if best_fail_count != 0:
        parsed = pd.to_datetime(raw, errors="coerce", dayfirst=True)
        fail_count = int(parsed.isna().sum())
        if fail_count < best_fail_count:
            best_parsed = parsed
            best_label = "pandas flexible dayfirst=True"
            best_fail_count = fail_count
    return best_parsed, str(best_label), int(best_fail_count)


def normalize_epsg(value: str) -> str:
    return value.strip().upper().replace(" ", "")


def main() -> None:
    STAGE_RECORDS.mkdir(parents=True, exist_ok=True)

    print("\n=== PROTO2 PHASE 0: SOURCE CONTRACT CHECK ===")
    print(f"Project root: {PROJECT_ROOT}")

    source_mode = active_deformation_source(CONFIG)
    check(source_mode == "displacement_csv", f"active deformation source: {source_mode}")
    check(not pyspams_automated(CONFIG), "PySPAMS automation is disabled; precomputed displacement CSV will be used")

    required_sidecars = CONFIG.get("input_contract", {}).get(
        "required_shapefile_sidecars", [".shp", ".dbf", ".shx", ".prj"]
    )
    sidecars = [SHP.with_suffix(str(ext)) for ext in required_sidecars]
    required_files = [DISP_CSV, PARAMS_PARQUET, PARAMS_JSON, *sidecars]
    missing_files = [str(path) for path in required_files if not path.is_file()]
    if missing_files:
        fail("required user input files are missing:\n  - " + "\n  - ".join(missing_files))
    ok("all configured user input files are present")

    if not PYSPAMS_DIR.is_dir():
        fail(f"configured PySPAMS directory is missing: {PYSPAMS_DIR}")
    required_pyspams = CONFIG.get("input_contract", {}).get("required_pyspams_files", ["utils.py", "spams_main.py"])
    missing_pyspams = [str(PYSPAMS_DIR / name) for name in required_pyspams if not (PYSPAMS_DIR / name).is_file()]
    if missing_pyspams:
        fail("required PySPAMS files are missing:\n  - " + "\n  - ".join(missing_pyspams))
    ok("configured PySPAMS directory and required files are present")

    try:
        metadata_payload = json.loads(PARAMS_JSON.read_text(encoding="utf-8"))
    except Exception as exc:
        fail(f"model metadata JSON cannot be read: {PARAMS_JSON}: {exc}")
    check(isinstance(metadata_payload, dict), "model metadata JSON is a JSON object")

    parcel_crs = declared_crs(CONFIG, "parcel_crs")
    displacement_crs = declared_crs(CONFIG, "displacement_crs")
    check(normalize_epsg(parcel_crs) == "EPSG:4326", f"configured parcel CRS is supported: {parcel_crs}")
    check(normalize_epsg(displacement_crs) == "EPSG:4326", f"configured displacement CRS is supported: {displacement_crs}")

    gdf = gpd.read_file(SHP)
    ok(f"parcel shapefile opens: {len(gdf):,} features")
    epsg = gdf.crs.to_epsg() if gdf.crs is not None else None
    check(epsg == 4326, f"shapefile CRS matches configured EPSG:4326: {gdf.crs}")

    join_col = next((name for name in geometry_id_candidates(CONFIG) if name in gdf.columns), None)
    if join_col is None:
        fail(f"parcel ID field missing; tried {geometry_id_candidates(CONFIG)}")
    ok(f"parcel ID field found: {join_col}")

    geom_ids = pd.to_numeric(gdf[join_col], errors="coerce")
    check(not geom_ids.isna().any(), "parcel IDs parse as numeric")
    check(int(geom_ids.nunique()) == len(gdf), "parcel IDs are unique")
    invalid = int((~gdf.geometry.is_valid).sum())
    empty = int(gdf.geometry.is_empty.sum())
    null = int(gdf.geometry.isna().sum())
    check(invalid == 0 and empty == 0 and null == 0, f"parcel geometry valid: invalid={invalid}, empty={empty}, null={null}")

    params = pd.read_parquet(PARAMS_PARQUET, columns=["pnt_id"])
    params_ids = set(pd.to_numeric(params["pnt_id"], errors="raise").astype("int64").tolist())
    ok(f"model parameter parquet opens: {len(params_ids):,} parcel IDs")

    required_cols = source_columns(CONFIG)
    try:
        disp = pd.read_csv(DISP_CSV, usecols=required_cols, low_memory=False)
    except ValueError as exc:
        fail(f"displacement CSV does not satisfy the required column contract: {exc}")
    missing_cols = [column for column in required_cols if column not in disp.columns]
    if missing_cols:
        fail(f"displacement CSV missing required columns: {missing_cols}")
    ok(f"displacement CSV columns valid: {len(required_cols)} required columns")

    epoch_dt, parser_label, parse_fail_count = parse_epoch_series(disp["epoch"])
    check(parse_fail_count == 0, f"epoch values parse successfully using {parser_label}")

    disp_ids = set(pd.to_numeric(disp["pnt_id"], errors="raise").astype("int64").unique().tolist())
    geom_id_set = set(geom_ids.astype("int64").tolist())
    check(not (disp_ids - geom_id_set), "every displacement parcel has matching geometry")
    check(not (disp_ids - params_ids), "every displacement parcel exists in model parameter parquet")

    dates = sorted(pd.Series(epoch_dt).dt.normalize().unique())
    if not dates:
        fail("displacement CSV contains no epochs")
    start = pd.Timestamp(dates[0])
    end = pd.Timestamp(dates[-1])
    expected_dates = pd.date_range(start=start, end=end, freq="D")
    missing_dates = sorted(set(expected_dates) - set(pd.Timestamp(value) for value in dates))
    check(not missing_dates, f"daily epoch sequence is continuous: {start.date()} to {end.date()}")

    duplicate_count = int(
        disp.assign(_epoch=pd.Series(epoch_dt).dt.strftime("%Y-%m-%d"))
        .duplicated(subset=["pnt_id", "_epoch"])
        .sum()
    )
    check(duplicate_count == 0, "no duplicate parcel-date rows")

    total_diff = pd.to_numeric(disp["h_spams_final"], errors="raise") - (
        pd.to_numeric(disp["reversible"], errors="raise")
        + pd.to_numeric(disp["irreversible"], errors="raise")
    )
    bad_rows = int((total_diff.abs() > 1e-6).sum())
    check(bad_rows == 0, "h_spams_final equals reversible + irreversible")

    summary = {
        "total_parcels": int(len(gdf)),
        "moving_parcels": int(len(disp_ids)),
        "blank_parcels": int(len(geom_id_set - disp_ids)),
        "epochs": int(len(dates)),
        "rows": int(len(disp)),
        "epoch_start": str(start.date()),
        "epoch_end": str(end.date()),
        "geometry_join_key": join_col,
        "epoch_parser": parser_label,
    }
    report = {
        "schema": "proto2_phase0_sanity_report_v3",
        "status": "PASS",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "config_schema": CONFIG.get("schema"),
        "active_deformation_source": source_mode,
        "inputs": {
            "displacement_csv": str(DISP_CSV.relative_to(PROJECT_ROOT)),
            "parcel_shapefile": str(SHP.relative_to(PROJECT_ROOT)),
            "model_parameters_parquet": str(PARAMS_PARQUET.relative_to(PROJECT_ROOT)),
            "model_metadata_json": str(PARAMS_JSON.relative_to(PROJECT_ROOT)),
            "pyspams_directory": str(PYSPAMS_DIR.relative_to(PROJECT_ROOT)),
        },
        "summary": summary,
    }
    json_path = STAGE_RECORDS / "phase0_sanity_report.json"
    txt_path = STAGE_RECORDS / "phase0_sanity_report.txt"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    txt_path.write_text(
        "\n".join(
            [
                "PROTO2 PHASE 0 SOURCE CONTRACT REPORT",
                "",
                "status: PASS",
                f"total parcels: {summary['total_parcels']:,}",
                f"moving parcels: {summary['moving_parcels']:,}",
                f"blank parcels: {summary['blank_parcels']:,}",
                f"epochs: {summary['epochs']:,}",
                f"rows: {summary['rows']:,}",
                f"epoch range: {summary['epoch_start']} to {summary['epoch_end']}",
                f"geometry join key: {join_col}",
            ]
        ),
        encoding="utf-8",
    )

    print("\n=== PHASE 0 RESULT ===")
    print("Status : PASS")
    print(f"Parcels: {summary['total_parcels']:,} total / {summary['moving_parcels']:,} moving / {summary['blank_parcels']:,} blank")
    print(f"Epochs : {summary['epochs']:,} ({summary['epoch_start']} to {summary['epoch_end']})")


if __name__ == "__main__":
    main()
