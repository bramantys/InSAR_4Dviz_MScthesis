from pathlib import Path
import gc
import json
import os
import sys

import numpy as np
import pandas as pd

from _proto2_config import (
    expected_int,
    input_path,
    load_project_config,
    output_data_dir,
    should_require_model_parameters,
    source_columns,
    stage_records_dir,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = load_project_config(PROJECT_ROOT)

DATA = PROJECT_ROOT / "data"
OUTPUT_DATA = output_data_dir(PROJECT_ROOT, CONFIG)
RUN_RECORDS = stage_records_dir(PROJECT_ROOT, CONFIG)

SOURCE_DISPLACEMENT_CSV = input_path(PROJECT_ROOT, CONFIG, "displacement", "path", "data/displacement/example_spams_model_2025.csv")
MODEL_PARAMS_PARQUET = input_path(PROJECT_ROOT, CONFIG, "model_parameters", "parquet_path", "data/model_params/nl_krimpenerwaard_spams10.parquet")

LONG_CSV_OUT = OUTPUT_DATA / "parcel_displacement_long.csv"
LONG_PARQUET_OUT = OUTPUT_DATA / "parcel_displacement_long.parquet"
EPOCH_AXIS_OUT = OUTPUT_DATA / "epoch_axis.json"
SUMMARY_OUT = OUTPUT_DATA / "parcel_displacement_summary.json"

REPORT_TXT_OUT = RUN_RECORDS / "phase01_adapter_report.txt"
REPORT_JSON_OUT = RUN_RECORDS / "phase01_adapter_report.json"

EXPECTED_MOVING_PARCELS = expected_int(CONFIG, "moving_parcels")
EXPECTED_EPOCHS = expected_int(CONFIG, "epochs")
EXPECTED_ROWS = None if EXPECTED_MOVING_PARCELS is None or EXPECTED_EPOCHS is None else EXPECTED_MOVING_PARCELS * EXPECTED_EPOCHS

# The production pipeline consumes the Parquet product. The old 1.4-million-row
# duplicate CSV is skipped by default because converting the full mixed-type
# dataframe to text creates a large temporary Unicode allocation.
#
# Set PROTO2_WRITE_LEGACY_LONG_CSV=1 only when that compatibility CSV is
# explicitly needed.
WRITE_LEGACY_LONG_CSV = (
    os.environ.get("PROTO2_WRITE_LEGACY_LONG_CSV", "0").strip().lower()
    in {"1", "true", "yes", "on"}
)


REQUIRED_SOURCE_COLUMNS = source_columns(CONFIG)


def parse_epoch_series(epoch_series):
    """
    Robust parser for source parcel displacement epoch columns.
    Current confirmed raw format: YYYY-MM-DD, e.g. 2025-01-01.
    Kept robust so the template survives future parcel datasets.
    """
    raw = epoch_series.astype(str).str.strip()

    candidate_formats = [
        "%Y-%m-%d",      # 2025-01-01
        "%d-%m-%Y",      # 01-01-2025
        "%d-%m-%y",      # 01-01-25
        "%Y/%m/%d",      # 2025/01/01
        "%d/%m/%Y",      # 01/01/2025
        "%d/%m/%y",      # 01/01/25
        "%m/%d/%Y",      # 1/1/2025
        "%m/%d/%y",      # 1/1/25
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
            return parsed, fmt, fail_count

    parsed_dayfirst = pd.to_datetime(raw, errors="coerce", dayfirst=True)
    fail_dayfirst = int(parsed_dayfirst.isna().sum())
    if fail_dayfirst < best_fail_count:
        best_parsed = parsed_dayfirst
        best_label = "pandas flexible dayfirst=True"
        best_fail_count = fail_dayfirst

    parsed_monthfirst = pd.to_datetime(raw, errors="coerce", dayfirst=False)
    fail_monthfirst = int(parsed_monthfirst.isna().sum())
    if fail_monthfirst < best_fail_count:
        best_parsed = parsed_monthfirst
        best_label = "pandas flexible dayfirst=False"
        best_fail_count = fail_monthfirst

    return best_parsed, best_label, best_fail_count


def fail(message):
    print(f"[FAIL] {message}")
    sys.exit(1)


def ok(message):
    print(f"[OK] {message}")


def main():
    OUTPUT_DATA.mkdir(parents=True, exist_ok=True)
    RUN_RECORDS.mkdir(parents=True, exist_ok=True)

    print("\n=== PROTO2 PHASE 01: ADAPT PARCEL DISPLACEMENT ===")
    print(f"Project root: {PROJECT_ROOT}")

    if not SOURCE_DISPLACEMENT_CSV.exists():
        fail(f"Missing source displacement CSV: {SOURCE_DISPLACEMENT_CSV}")

    model_params_required = should_require_model_parameters(CONFIG)
    if model_params_required and not MODEL_PARAMS_PARQUET.exists():
        fail(f"Missing model parameter parquet: {MODEL_PARAMS_PARQUET}")

    print(f"\nReading source displacement:\n  {SOURCE_DISPLACEMENT_CSV}")
    # Read only the columns needed by the canonical runtime contract.
    # This avoids retaining both the complete source dataframe and a second
    # 1.4-million-row copy in memory.
    try:
        df = pd.read_csv(
            SOURCE_DISPLACEMENT_CSV,
            usecols=REQUIRED_SOURCE_COLUMNS,
            low_memory=False,
        )
    except ValueError as exc:
        fail(f"Source displacement missing required columns: {exc}")

    missing_cols = [c for c in REQUIRED_SOURCE_COLUMNS if c not in df.columns]
    if missing_cols:
        fail(f"Source displacement missing required columns: {missing_cols}")

    ok(f"source columns present: {len(REQUIRED_SOURCE_COLUMNS)} required columns")

    # Normalize IDs.
    df["pnt_id"] = pd.to_numeric(df["pnt_id"], errors="raise").astype("int64")
    df["pnt_gid"] = pd.to_numeric(df["pnt_gid"], errors="raise").astype("int64")

    # Normalize dates.
    epoch_dt, parser_label, parse_fail_count = parse_epoch_series(df["epoch"])

    if parse_fail_count != 0:
        fail(f"epoch parse failed for {parse_fail_count:,} rows using best parser: {parser_label}")

    ok(f"epoch parser selected: {parser_label}")

    df["epoch"] = epoch_dt.dt.strftime("%Y-%m-%d")
    del epoch_dt
    gc.collect()

    # Normalize numeric columns.
    numeric_cols = [
        "reversible",
        "irreversible",
        "h_spams_final",
        "pnt_lat",
        "pnt_lon",
        "vI",
        "std_vI",
        "var_vI",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="raise")

    # Rename total displacement to a generic runtime name.
    # Source name is preserved in reports, runtime contract uses "total".
    df = df.rename(columns={"h_spams_final": "total"})

    # Verify total = reversible + irreversible.
    total_diff = df["total"] - (df["reversible"] + df["irreversible"])
    bad_total_rows = int((total_diff.abs() > 1e-6).sum())
    max_abs_total_diff = float(total_diff.abs().max())

    if bad_total_rows:
        fail(
            f"total != reversible + irreversible for {bad_total_rows:,} rows; "
            f"max abs diff={max_abs_total_diff:.12g}"
        )

    ok(f"total = reversible + irreversible for all rows; max abs diff={max_abs_total_diff:.12g}")
    del total_diff
    gc.collect()

    # Add reserved total-displacement uncertainty column.
    # Important: NaN means unavailable/unknown, NOT zero.
    df["sigma_h"] = np.nan

    # Sort and assign epoch index.
    unique_epochs = sorted(df["epoch"].unique().tolist())
    epoch_to_index = {epoch: i for i, epoch in enumerate(unique_epochs)}
    df["epoch_index"] = df["epoch"].map(epoch_to_index).astype("int32")

    df.sort_values(["pnt_id", "epoch_index"], inplace=True, kind="mergesort")
    df.reset_index(drop=True, inplace=True)
    gc.collect()

    # Basic checks.
    row_count = len(df)
    parcel_count = int(df["pnt_id"].nunique())
    epoch_count = len(unique_epochs)

    if EXPECTED_ROWS is not None and row_count != EXPECTED_ROWS:
        fail(f"row count {row_count:,} does not match expected {EXPECTED_ROWS:,}")

    if EXPECTED_MOVING_PARCELS is not None and parcel_count != EXPECTED_MOVING_PARCELS:
        fail(f"parcel count {parcel_count:,} does not match expected {EXPECTED_MOVING_PARCELS:,}")

    if EXPECTED_EPOCHS is not None and epoch_count != EXPECTED_EPOCHS:
        fail(f"epoch count {epoch_count:,} does not match expected {EXPECTED_EPOCHS:,}")

    ok(f"row count: {row_count:,}")
    ok(f"moving parcel count: {parcel_count:,}")
    ok(f"epoch count: {epoch_count:,}")

    duplicate_pairs = int(df.duplicated(subset=["pnt_id", "epoch"]).sum())
    if duplicate_pairs:
        fail(f"found {duplicate_pairs:,} duplicate pnt_id/epoch rows")

    ok("no duplicate pnt_id/epoch rows")

    # Check ID set against model parameters when available.
    disp_ids = set(df["pnt_id"].unique().tolist())
    source_not_in_params = []
    params_not_in_source = []
    if MODEL_PARAMS_PARQUET.exists():
        params = pd.read_parquet(MODEL_PARAMS_PARQUET, columns=["pnt_id"])
        params_ids = set(pd.to_numeric(params["pnt_id"], errors="raise").astype("int64").tolist())
        source_not_in_params = sorted(disp_ids - params_ids)
        params_not_in_source = sorted(params_ids - disp_ids)

        if source_not_in_params or params_not_in_source:
            fail(
                "source displacement pnt_id set does not match model parameter pnt_id set: "
                f"source-not-in-params={len(source_not_in_params):,}, "
                f"params-not-in-source={len(params_not_in_source):,}"
            )
        ok("source displacement pnt_id set matches model parameter pnt_id set")
    else:
        ok("model parameter parquet absent; skipped source-vs-model pnt_id set check")

    # Final canonical column order.
    canonical_cols = [
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

    df = df[canonical_cols]

    # Epoch axis.
    epoch_axis = {
        "epoch_count": epoch_count,
        "start": unique_epochs[0],
        "end": unique_epochs[-1],
        "epochs": [
            {
                "epoch_index": i,
                "epoch": epoch,
            }
            for i, epoch in enumerate(unique_epochs)
        ],
    }

    # Summary.
    summary = {
        "source_displacement_csv": str(SOURCE_DISPLACEMENT_CSV),
        "runtime_contract": "parcel_displacement_long",
        "rows": row_count,
        "moving_parcels": parcel_count,
        "epoch_count": epoch_count,
        "epoch_start": unique_epochs[0],
        "epoch_end": unique_epochs[-1],
        "epoch_parser": parser_label,
        "columns": canonical_cols,
        "reserved_columns": {
            "sigma_h": {
                "meaning": "per-epoch uncertainty of total displacement",
                "current_status": "unavailable",
                "null_rule": "null means unknown/unavailable, not zero uncertainty",
            }
        },
        "ranges": {
            "reversible": {
                "min": float(df["reversible"].min()),
                "max": float(df["reversible"].max()),
            },
            "irreversible": {
                "min": float(df["irreversible"].min()),
                "max": float(df["irreversible"].max()),
            },
            "total": {
                "min": float(df["total"].min()),
                "max": float(df["total"].max()),
            },
            "vI": {
                "min": float(df["vI"].min()),
                "max": float(df["vI"].max()),
            },
            "std_vI": {
                "min": float(df["std_vI"].min()),
                "max": float(df["std_vI"].max()),
            },
            "var_vI": {
                "min": float(df["var_vI"].min()),
                "max": float(df["var_vI"].max()),
            },
        },
        "checks": {
            "bad_total_rows": bad_total_rows,
            "max_abs_total_component_diff": max_abs_total_diff,
            "duplicate_pnt_id_epoch_rows": duplicate_pairs,
            "source_not_in_params": len(source_not_in_params),
            "params_not_in_source": len(params_not_in_source),
        },
    }

    # Write outputs.
    print("\nWriting canonical outputs...")

    # Parquet is the canonical downstream product used by phases 02 and 05.
    # Write it first and treat failure as fatal.
    try:
        df.to_parquet(
            LONG_PARQUET_OUT,
            index=False,
            engine="pyarrow",
            compression="snappy",
            row_group_size=100_000,
        )
        ok(f"wrote {LONG_PARQUET_OUT}")
        parquet_status = "written"
    except Exception as e:
        fail(f"Could not write required canonical parquet: {e}")

    # The duplicate long CSV is legacy-only and is not consumed downstream.
    if WRITE_LEGACY_LONG_CSV:
        try:
            df.to_csv(
                LONG_CSV_OUT,
                index=False,
                chunksize=25_000,
            )
            ok(f"wrote optional legacy CSV {LONG_CSV_OUT}")
            csv_status = "written"
        except Exception as e:
            fail(f"Could not write requested legacy CSV: {e}")
    else:
        if LONG_CSV_OUT.exists():
            LONG_CSV_OUT.unlink()
        csv_status = "skipped (Parquet is canonical; set PROTO2_WRITE_LEGACY_LONG_CSV=1 to request it)"
        ok("skipped duplicate 1.4M-row legacy CSV; downstream phases use Parquet")

    EPOCH_AXIS_OUT.write_text(json.dumps(epoch_axis, indent=2), encoding="utf-8")
    ok(f"wrote {EPOCH_AXIS_OUT}")

    SUMMARY_OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    ok(f"wrote {SUMMARY_OUT}")

    # Reports.
    report = {
        "project_root": str(PROJECT_ROOT),
        "outputs": {
            "long_csv": str(LONG_CSV_OUT),
            "long_csv_status": csv_status,
            "long_parquet": str(LONG_PARQUET_OUT),
            "epoch_axis": str(EPOCH_AXIS_OUT),
            "summary": str(SUMMARY_OUT),
            "parquet_status": parquet_status,
        },
        "summary": summary,
    }

    REPORT_JSON_OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    txt_lines = [
        "PROTO2 PHASE 01 ADAPTER REPORT",
        "",
        f"source displacement: {SOURCE_DISPLACEMENT_CSV}",
        f"epoch parser: {parser_label}",
        f"rows: {row_count:,}",
        f"moving parcels: {parcel_count:,}",
        f"epochs: {epoch_count:,}",
        f"epoch range: {unique_epochs[0]} to {unique_epochs[-1]}",
        "",
        "outputs:",
        f"- {LONG_CSV_OUT} ({csv_status})",
        f"- {LONG_PARQUET_OUT} ({parquet_status}; canonical)",
        f"- {EPOCH_AXIS_OUT}",
        f"- {SUMMARY_OUT}",
        "",
        "ranges:",
        f"- reversible:   {summary['ranges']['reversible']['min']:.6f} to {summary['ranges']['reversible']['max']:.6f}",
        f"- irreversible: {summary['ranges']['irreversible']['min']:.6f} to {summary['ranges']['irreversible']['max']:.6f}",
        f"- total:        {summary['ranges']['total']['min']:.6f} to {summary['ranges']['total']['max']:.6f}",
        "",
        "sigma_h rule:",
        "null sigma_h means uncertainty unavailable, not zero uncertainty.",
    ]

    REPORT_TXT_OUT.write_text("\n".join(txt_lines), encoding="utf-8")
    ok(f"wrote {REPORT_JSON_OUT}")
    ok(f"wrote {REPORT_TXT_OUT}")

    print("\n=== SUMMARY ===")
    print(f"Canonical rows: {row_count:,}")
    print(f"Moving parcels: {parcel_count:,}")
    print(f"Epochs: {epoch_count:,}")
    print(f"Epoch range: {unique_epochs[0]} to {unique_epochs[-1]}")
    print(f"Total displacement range: {df['total'].min():.6f} to {df['total'].max():.6f}")
    print("\nPHASE 01 RESULT: PASS. Canonical parcel displacement product written.")


if __name__ == "__main__":
    main()