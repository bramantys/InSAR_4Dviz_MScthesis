#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01 — Convert RUM source table to WGS84 GeoJSON

Generic RUM-based InSAR source-preparation step.

Purpose
-------
Read a RUM source table from Data/1_OG, convert its projected x/y coordinates
into WGS84 lon/lat, create stable rum_id values, and write viewer-ready GeoJSON.

Default config
--------------
Reads:
  config/project_config.json

Uses config fields:
  source_inputs.primary_source_csv/json/pkl
  source_inputs.source_crs
  source_inputs.source_coordinate_fields.x/y
  prepared_inputs.plain_points_geojson
  prepared_inputs.points_geojson

Typical run
-----------
From the project root:
  python pipeline_template/01_convert_source_to_geojson.py

Notes
-----
- Jakarta currently uses EPSG:23830 source coordinates.
- Other RUM-based InSAR projects can use a different source CRS by changing
  config/project_config.json, not this script.
- This script preserves source velocity/covariance fields such as east, north,
  up, var_east, var_north, var_up, covar_en if they exist in the source table.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd

try:
    import geopandas as gpd
except ImportError as exc:
    raise SystemExit(
        "geopandas is required for CRS conversion. Install it in your environment first."
    ) from exc


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def ok(msg: str) -> None:
    print(f"  [OK]   {msg}")


def warn(msg: str) -> None:
    print(f"  [WARN] {msg}")


def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def safe_float(value: Any, fallback: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return fallback
        out = float(value)
        if not math.isfinite(out):
            return fallback
        return out
    except Exception:
        return fallback


def resolve_project_root(script_path: Path) -> Path:
    """Script lives in project_root/pipeline_template/ by default."""
    return script_path.resolve().parent.parent


def load_config(project_root: Path, config_arg: Optional[str]) -> Dict[str, Any]:
    config_path = Path(config_arg) if config_arg else project_root / "config" / "project_config.json"
    if not config_path.is_absolute():
        config_path = project_root / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    return cfg


def rel_to_root(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def normalize_records(obj: Any) -> pd.DataFrame:
    """Convert common JSON/PKL structures to a DataFrame."""
    if isinstance(obj, pd.DataFrame):
        return obj.copy()

    # GeoPandas GeoDataFrame should also arrive here as a DataFrame subclass.
    if hasattr(obj, "to_dict") and hasattr(obj, "columns"):
        return pd.DataFrame(obj)

    if isinstance(obj, list):
        return pd.DataFrame(obj)

    if isinstance(obj, dict):
        # GeoJSON FeatureCollection
        if obj.get("type") == "FeatureCollection":
            rows = []
            for ft in obj.get("features", []):
                props = dict(ft.get("properties") or {})
                geom = ft.get("geometry") or {}
                coords = geom.get("coordinates") or []
                if geom.get("type") == "Point" and len(coords) >= 2:
                    props.setdefault("lon", coords[0])
                    props.setdefault("lat", coords[1])
                rows.append(props)
            return pd.DataFrame(rows)

        # Dict of records keyed by ID
        if obj and all(isinstance(v, dict) for v in obj.values()):
            rows = []
            for key, value in obj.items():
                row = dict(value)
                row.setdefault("_source_key", key)
                rows.append(row)
            return pd.DataFrame(rows)

        # Column-oriented dict
        list_keys = [k for k, v in obj.items() if isinstance(v, list)]
        if list_keys:
            lengths = {len(obj[k]) for k in list_keys}
            if len(lengths) == 1:
                return pd.DataFrame({k: obj[k] for k in list_keys})

    raise ValueError(f"Unsupported source data structure: {type(obj)}")


def load_source_dataframe(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()

    if ext == ".csv":
        return pd.read_csv(path)

    if ext in [".json", ".geojson"]:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return normalize_records(obj)

    if ext in [".pkl", ".pickle"]:
        with path.open("rb") as f:
            obj = pickle.load(f)
        return normalize_records(obj)

    raise ValueError(f"Unsupported source extension: {ext}")


def make_base_rum_id(row: pd.Series, x_field: str, y_field: str) -> str:
    # Preferred: projected RUM coordinates, stable across reprojection.
    x = safe_float(row.get(x_field))
    y = safe_float(row.get(y_field))
    if x is not None and y is not None:
        return f"RUM_{int(round(x))}_{int(round(y))}"

    # Fallback: WGS84 lon/lat if source coordinates are unavailable.
    lon = safe_float(row.get("lon"), 0.0) or 0.0
    lat = safe_float(row.get("lat"), 0.0) or 0.0
    return f"RUM_lon{lon:.7f}_lat{lat:.7f}".replace("-", "m").replace(".", "p")


def add_stable_rum_ids(gdf: "gpd.GeoDataFrame", x_field: str, y_field: str, id_field: str) -> "gpd.GeoDataFrame":
    out = gdf.copy()
    seen: Dict[str, int] = {}
    rum_ids = []

    for _, row in out.iterrows():
        existing = row.get(id_field)
        if existing is not None and str(existing).strip() and str(existing).lower() != "nan":
            base_id = str(existing)
        else:
            base_id = make_base_rum_id(row, x_field, y_field)

        duplicate_index = seen.get(base_id, 0)
        seen[base_id] = duplicate_index + 1
        rum_id = base_id if duplicate_index == 0 else f"{base_id}_dup{duplicate_index + 1}"
        rum_ids.append(rum_id)

    out[id_field] = rum_ids
    return out


def print_numeric_range(df: pd.DataFrame, column: str) -> None:
    if column not in df.columns:
        warn(f"Column missing: {column}")
        return
    vals = pd.to_numeric(df[column], errors="coerce")
    vals = vals[vals.notna()]
    if vals.empty:
        warn(f"Column has no numeric values: {column}")
        return
    print(f"  {column:<16s}: {vals.min():.6f} → {vals.max():.6f}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Convert RUM source data to WGS84 GeoJSON.")
    parser.add_argument("--config", default=None, help="Path to project_config.json. Default: config/project_config.json")
    parser.add_argument("--source", default=None, help="Optional source file override: CSV, JSON, GeoJSON, or PKL")
    parser.add_argument("--source-crs", default=None, help="Optional CRS override, e.g. EPSG:23830")
    parser.add_argument("--x-field", default=None, help="Optional source x-coordinate field override")
    parser.add_argument("--y-field", default=None, help="Optional source y-coordinate field override")
    parser.add_argument("--id-field", default="rum_id", help="RUM ID field name. Default: rum_id")
    args = parser.parse_args()

    project_root = resolve_project_root(Path(__file__))
    cfg = load_config(project_root, args.config)

    source_inputs = cfg.get("source_inputs", {})
    prepared_inputs = cfg.get("prepared_inputs", {})
    coord_fields = source_inputs.get("source_coordinate_fields", {})

    x_field = args.x_field or coord_fields.get("x", "x_rum")
    y_field = args.y_field or coord_fields.get("y", "y_rum")
    source_crs = args.source_crs or source_inputs.get("source_crs", "EPSG:4326")

    if args.source:
        source_path = rel_to_root(project_root, args.source)
    else:
        candidates = [
            rel_to_root(project_root, source_inputs.get("primary_source_csv", "Data/1_OG/source.csv")),
            rel_to_root(project_root, source_inputs.get("primary_source_json", "Data/1_OG/source.json")),
            rel_to_root(project_root, source_inputs.get("primary_source_pkl", "Data/1_OG/source.pkl")),
        ]
        source_path = first_existing(candidates)
        if source_path is None:
            raise FileNotFoundError(
                "No source file found. Checked config source_inputs primary_source_csv/json/pkl."
            )

    plain_output = rel_to_root(
        project_root,
        prepared_inputs.get("plain_points_geojson", "Data/points_wgs84.geojson"),
    )
    rumid_output = rel_to_root(
        project_root,
        prepared_inputs.get("points_geojson", "Data/points_wgs84_with_rumid.geojson"),
    )

    section("Configuration")
    print(f"  Project root : {project_root}")
    print(f"  Source file  : {source_path}")
    print(f"  Source CRS   : {source_crs}")
    print(f"  X/Y fields   : {x_field}, {y_field}")
    print(f"  Plain output : {plain_output}")
    print(f"  RUMID output : {rumid_output}")

    section("Loading source table")
    df = load_source_dataframe(source_path)
    ok(f"Loaded rows: {len(df)}")
    ok(f"Loaded columns: {len(df.columns)}")
    print("  Columns:")
    print("  " + ", ".join(map(str, df.columns)))

    if x_field not in df.columns or y_field not in df.columns:
        raise KeyError(
            f"Source coordinate fields not found. Need '{x_field}' and '{y_field}'. "
            f"Available columns: {list(df.columns)}"
        )

    # Force numeric x/y and remove rows that cannot be projected.
    df = df.copy()
    df[x_field] = pd.to_numeric(df[x_field], errors="coerce")
    df[y_field] = pd.to_numeric(df[y_field], errors="coerce")

    before = len(df)
    df = df[df[x_field].notna() & df[y_field].notna()].copy()
    dropped = before - len(df)
    if dropped:
        warn(f"Dropped {dropped} rows with missing/non-numeric {x_field}/{y_field}")
    ok(f"Rows with valid source coordinates: {len(df)}")

    section("Source coordinate range")
    print_numeric_range(df, x_field)
    print_numeric_range(df, y_field)

    section("Reprojecting to WGS84")
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[x_field], df[y_field]),
        crs=source_crs,
    )

    gdf_wgs84 = gdf.to_crs(epsg=4326)
    gdf_wgs84["lon"] = gdf_wgs84.geometry.x
    gdf_wgs84["lat"] = gdf_wgs84.geometry.y

    ok("Converted source CRS to EPSG:4326")
    print_numeric_range(gdf_wgs84, "lon")
    print_numeric_range(gdf_wgs84, "lat")

    section("Creating stable RUM IDs")
    gdf_with_ids = add_stable_rum_ids(gdf_wgs84, x_field, y_field, args.id_field)
    duplicate_count = gdf_with_ids[args.id_field].duplicated().sum()
    if duplicate_count:
        warn(f"Duplicate rum_id values remain: {duplicate_count}")
    else:
        ok("All rum_id values are unique")

    print("  Sample rum_id values:")
    for rid in gdf_with_ids[args.id_field].head(5).tolist():
        print(f"    {rid}")

    section("Writing GeoJSON outputs")
    plain_output.parent.mkdir(parents=True, exist_ok=True)
    rumid_output.parent.mkdir(parents=True, exist_ok=True)

    # Plain output: WGS84 point layer. It may already contain rum_id if source had it,
    # but the guaranteed-ID output is written separately below.
    gdf_wgs84.to_file(plain_output, driver="GeoJSON")
    ok(f"Plain WGS84 GeoJSON: {plain_output} ({plain_output.stat().st_size / 1024:.1f} KB)")

    gdf_with_ids.to_file(rumid_output, driver="GeoJSON")
    ok(f"WGS84 GeoJSON with rum_id: {rumid_output} ({rumid_output.stat().st_size / 1024:.1f} KB)")

    section("Field preservation check")
    likely_fields = [
        "east", "north", "up",
        "var_east", "var_north", "var_up",
        "covar_en", "covar_eu", "covar_nu",
    ]
    for field in likely_fields:
        if field in gdf_with_ids.columns:
            ok(f"Preserved field: {field}")
        else:
            warn(f"Field not present in source/output: {field}")

    section("SUMMARY")
    ok("Step 01 complete — source table converted to WGS84 GeoJSON")
    ok("Next template step: 02_prepare_vertical_epochs_from_velocity.py")
    print()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        fail(str(exc))
        raise
