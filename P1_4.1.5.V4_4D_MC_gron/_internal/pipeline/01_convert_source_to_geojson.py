#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_convert_source_to_geojson.py

InSAR4D RUM Viewer pipeline step 01.

Purpose
-------
Read the user-provided RUM ENU estimates table and convert RUM center
coordinates from the configured source CRS to WGS84 lon/lat GeoJSON.

Input
-----
Defined in config/project_config.json through pipeline_config.py:
  user_inputs.source_file
  user_inputs.source_crs
  source_schema.coordinate_fields
  source_schema.velocity_fields
  source_schema.uncertainty_fields

Supported source table formats:
  - .csv
  - .json  (list of records, dict of records, or GeoJSON FeatureCollection)
  - .pkl / .pickle

Outputs
-------
  prepared_inputs.plain_points_geojson
  prepared_inputs.points_geojson

The second output is the important downstream product. It guarantees:
  - FeatureCollection
  - Point geometry in EPSG:4326 lon/lat
  - stable rum_id for each feature
  - ENU velocity and uncertainty fields preserved in properties
  - x_source / y_source preserved in properties
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

try:
    from pyproj import CRS, Transformer
except ImportError as exc:
    raise ImportError(
        "pyproj is required for coordinate transformation. Install it with: pip install pyproj"
    ) from exc

from pipeline_config import load_resolved_config, resolve_path


# =============================================================================
# ADVANCED SETTINGS
# =============================================================================
# Normal users should edit config/project_config.json, not this script.

RUM_ID_COLUMN_CANDIDATES = ["rum_id", "RUM_ID", "id", "ID"]
GENERATED_RUM_ID_PREFIX = "RUM"
GENERATED_RUM_ID_DIGITS = 6

ROUND_LON_LAT_DIGITS = 8
ROUND_SOURCE_XY_DIGITS = 4
ROUND_NUMERIC_PROPERTY_DIGITS = 8

WRITE_PLAIN_POINTS = True
FAIL_ON_DUPLICATE_RUM_IDS = True
DROP_ROWS_WITH_INVALID_COORDINATES = True


# =============================================================================
# PRINT HELPERS
# =============================================================================

def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def ok(msg: str) -> None:
    print(f"  [OK]   {msg}")


def warn(msg: str) -> None:
    print(f"  [WARN] {msg}")


# =============================================================================
# SMALL HELPERS
# =============================================================================

def clean_json_value(value: Any) -> Any:
    """Convert pandas/numpy-ish values into JSON-safe values."""
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, bool):
        return bool(value)

    if isinstance(value, int):
        return int(value)

    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return round(float(value), ROUND_NUMERIC_PROPERTY_DIGITS)

    # numpy scalar fallback
    if hasattr(value, "item"):
        try:
            return clean_json_value(value.item())
        except Exception:
            pass

    return str(value)


def ensure_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> None:
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required source column(s): {missing}")


def choose_rum_id_column(df: pd.DataFrame) -> Optional[str]:
    for col in RUM_ID_COLUMN_CANDIDATES:
        if col in df.columns:
            return col
    return None


def derive_rum_ids(df: pd.DataFrame) -> List[str]:
    existing_col = choose_rum_id_column(df)

    if existing_col:
        ids = [str(v).strip() for v in df[existing_col].tolist()]
        if any(not v or v.lower() == "nan" for v in ids):
            warn(f"RUM ID column '{existing_col}' has empty values; generated IDs will be used instead")
        else:
            ok(f"Using existing RUM ID column: {existing_col}")
            return ids

    ok("No stable RUM ID column found; generated sequential RUM IDs")
    return [
        f"{GENERATED_RUM_ID_PREFIX}_{i + 1:0{GENERATED_RUM_ID_DIGITS}d}"
        for i in range(len(df))
    ]


def check_duplicate_ids(rum_ids: List[str]) -> None:
    seen = set()
    duplicates = set()

    for rid in rum_ids:
        if rid in seen:
            duplicates.add(rid)
        seen.add(rid)

    if duplicates:
        sample = sorted(duplicates)[:10]
        msg = f"Duplicate rum_id values found: {len(duplicates)} duplicate IDs; sample={sample}"
        if FAIL_ON_DUPLICATE_RUM_IDS:
            raise ValueError(msg)
        warn(msg)


# =============================================================================
# SOURCE LOADERS
# =============================================================================

def load_geojson_as_dataframe(path: Path) -> Optional[pd.DataFrame]:
    """If a JSON is a GeoJSON FeatureCollection, read properties + geometry coords."""
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict) or payload.get("type") != "FeatureCollection":
        return None

    rows: List[Dict[str, Any]] = []

    for feature in payload.get("features", []):
        props = dict(feature.get("properties") or {})
        geom = feature.get("geometry") or {}
        coords = geom.get("coordinates") or []

        if geom.get("type") == "Point" and len(coords) >= 2:
            props.setdefault("lon", coords[0])
            props.setdefault("lat", coords[1])

        rows.append(props)

    return pd.DataFrame(rows)


def load_source_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix == ".json":
        maybe_geojson = load_geojson_as_dataframe(path)
        if maybe_geojson is not None:
            return maybe_geojson
        return pd.read_json(path)

    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)

    raise ValueError(
        f"Unsupported source file extension: {suffix}. Supported: .csv, .json, .pkl, .pickle"
    )


# =============================================================================
# CRS / TRANSFORMATION
# =============================================================================

def build_transformer(source_crs_value: str) -> Transformer:
    source_crs = CRS.from_user_input(source_crs_value)
    target_crs = CRS.from_epsg(4326)

    if source_crs == target_crs:
        ok("Source CRS is already EPSG:4326")
    else:
        ok(f"Coordinate transform: {source_crs.to_string()} -> EPSG:4326")

    return Transformer.from_crs(source_crs, target_crs, always_xy=True)


def transform_xy_to_lonlat(
    transformer: Transformer,
    x_values: Iterable[float],
    y_values: Iterable[float],
) -> Tuple[List[float], List[float]]:
    lons: List[float] = []
    lats: List[float] = []

    for x, y in zip(x_values, y_values):
        lon, lat = transformer.transform(float(x), float(y))
        lons.append(float(lon))
        lats.append(float(lat))

    return lons, lats


# =============================================================================
# GEOJSON BUILDING
# =============================================================================

def dataframe_to_geojson(
    df: pd.DataFrame,
    rum_ids: List[str],
    lon_values: List[float],
    lat_values: List[float],
    x_col: str,
    y_col: str,
    include_rum_id: bool,
) -> Dict[str, Any]:
    features: List[Dict[str, Any]] = []

    for row_idx, (_, row) in enumerate(df.iterrows()):
        lon = round(float(lon_values[row_idx]), ROUND_LON_LAT_DIGITS)
        lat = round(float(lat_values[row_idx]), ROUND_LON_LAT_DIGITS)

        props: Dict[str, Any] = {}

        if include_rum_id:
            props["rum_id"] = rum_ids[row_idx]

        props["source_row"] = int(row_idx)
        props["x_source"] = round(float(row[x_col]), ROUND_SOURCE_XY_DIGITS)
        props["y_source"] = round(float(row[y_col]), ROUND_SOURCE_XY_DIGITS)

        for col in df.columns:
            value = clean_json_value(row[col])
            if value is not None:
                props[str(col)] = value

        # Ensure generated rum_id wins over source column if present.
        if include_rum_id:
            props["rum_id"] = rum_ids[row_idx]

        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat],
            },
            "properties": props,
        })

    return {
        "type": "FeatureCollection",
        "features": features,
    }


def write_geojson(path: Path, geojson: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False, separators=(",", ":"))


def compute_bbox(lons: List[float], lats: List[float]) -> Dict[str, float]:
    return {
        "west": round(min(lons), ROUND_LON_LAT_DIGITS),
        "south": round(min(lats), ROUND_LON_LAT_DIGITS),
        "east": round(max(lons), ROUND_LON_LAT_DIGITS),
        "north": round(max(lats), ROUND_LON_LAT_DIGITS),
    }


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    t_start = time.time()

    cfg = load_resolved_config(__file__)
    project_root = Path(cfg["_resolved"]["project_root"])

    source_inputs = cfg["source_inputs"]
    prepared = cfg["prepared_inputs"]
    source_schema = cfg.get("source_schema", {})

    selected_source_file = Path(cfg["_resolved"]["source_file"])
    source_file_rel = cfg["_resolved"]["source_file_rel"]
    source_crs = source_inputs["source_crs"]

    coord_fields = source_schema.get("coordinate_fields", {})
    vel_fields = source_schema.get("velocity_fields", {})
    unc_fields = source_schema.get("uncertainty_fields", {})

    # Fallback for old-style resolved keys.
    x_col = coord_fields.get("x") or source_inputs["source_coordinate_fields"]["x"]
    y_col = coord_fields.get("y") or source_inputs["source_coordinate_fields"]["y"]

    plain_points_path = resolve_path(project_root, prepared["plain_points_geojson"])
    points_path = resolve_path(project_root, prepared["points_geojson"])

    section("Configuration")
    print(f"  Project root       : {project_root}")
    print(f"  Source file        : {source_file_rel}")
    print(f"  Source CRS         : {source_crs}")
    print(f"  X/Y fields         : {x_col}, {y_col}")
    print(f"  Plain GeoJSON      : {plain_points_path}")
    print(f"  RUM ID GeoJSON     : {points_path}")

    section("Loading source table")
    if not selected_source_file.exists():
        raise FileNotFoundError(f"Missing source file: {selected_source_file}")

    df = load_source_table(selected_source_file)
    if df.empty:
        raise ValueError(f"Source table is empty: {selected_source_file}")

    ok(f"Loaded {len(df)} rows × {len(df.columns)} columns")
    print(f"  Columns: {', '.join(str(c) for c in df.columns)}")

    section("Checking required columns")
    required_columns = [x_col, y_col]

    for field in ["east", "north", "up"]:
        col = vel_fields.get(field)
        if col:
            required_columns.append(col)

    ensure_columns(df, required_columns)
    ok("Required coordinate and velocity columns found")

    optional_uncertainty_missing = []
    for _key, col in unc_fields.items():
        if col not in df.columns:
            optional_uncertainty_missing.append(col)

    if optional_uncertainty_missing:
        warn(f"Missing optional uncertainty columns: {optional_uncertainty_missing}")
    else:
        ok("All configured uncertainty columns found")

    section("Cleaning coordinates")
    x_numeric = pd.to_numeric(df[x_col], errors="coerce")
    y_numeric = pd.to_numeric(df[y_col], errors="coerce")

    valid_mask = x_numeric.notna() & y_numeric.notna()
    invalid_count = int((~valid_mask).sum())
    original_row_count = len(df)

    if invalid_count:
        msg = f"Rows with invalid coordinates: {invalid_count}"
        if DROP_ROWS_WITH_INVALID_COORDINATES:
            warn(msg + " — dropped")
            df = df.loc[valid_mask].copy()
            df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
            df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
        else:
            raise ValueError(msg)
    else:
        ok("All source coordinates are finite")

    if df.empty:
        raise ValueError("No rows remain after coordinate cleaning")

    section("Building RUM IDs")
    rum_ids = derive_rum_ids(df)
    check_duplicate_ids(rum_ids)
    ok(f"Stable RUM IDs ready: {len(rum_ids)}")

    section("Transforming coordinates")
    transformer = build_transformer(source_crs)
    lons, lats = transform_xy_to_lonlat(transformer, df[x_col].tolist(), df[y_col].tolist())

    bad_lonlat = [
        i for i, (lon, lat) in enumerate(zip(lons, lats))
        if not (math.isfinite(lon) and math.isfinite(lat) and -180 <= lon <= 180 and -90 <= lat <= 90)
    ]
    if bad_lonlat:
        raise ValueError(f"Invalid transformed lon/lat for {len(bad_lonlat)} rows; sample indices={bad_lonlat[:10]}")

    bbox = compute_bbox(lons, lats)
    ok(f"Transformed {len(lons)} coordinates")
    print(f"  BBox WGS84: west={bbox['west']}, south={bbox['south']}, east={bbox['east']}, north={bbox['north']}")

    section("Writing GeoJSON outputs")
    geojson_with_ids = dataframe_to_geojson(
        df=df,
        rum_ids=rum_ids,
        lon_values=lons,
        lat_values=lats,
        x_col=x_col,
        y_col=y_col,
        include_rum_id=True,
    )
    write_geojson(points_path, geojson_with_ids)
    ok(f"Wrote RUM ID GeoJSON: {points_path} ({points_path.stat().st_size / 1024 / 1024:.2f} MB)")

    if WRITE_PLAIN_POINTS:
        geojson_plain = dataframe_to_geojson(
            df=df,
            rum_ids=rum_ids,
            lon_values=lons,
            lat_values=lats,
            x_col=x_col,
            y_col=y_col,
            include_rum_id=False,
        )
        write_geojson(plain_points_path, geojson_plain)
        ok(f"Wrote plain GeoJSON: {plain_points_path} ({plain_points_path.stat().st_size / 1024 / 1024:.2f} MB)")

    elapsed = time.time() - t_start

    section("Summary")
    ok(f"Step 01 complete in {elapsed:.2f} s")
    print(f"  Source rows input          : {original_row_count}")
    print(f"  Rows used                  : {len(rum_ids)}")
    print(f"  Rows dropped               : {invalid_count}")
    print(f"  Output RUM count           : {len(rum_ids)}")
    print(f"  BBox west/south/east/north : {bbox['west']}, {bbox['south']}, {bbox['east']}, {bbox['north']}")


if __name__ == "__main__":
    main()
