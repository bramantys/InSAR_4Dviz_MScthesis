from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

from _proto2_config import (
    expected_int,
    geometry_id_candidates,
    input_path,
    load_project_config,
    output_data_dir,
    should_require_model_parameters,
    stage_records_dir,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = load_project_config(PROJECT_ROOT)

DATA = PROJECT_ROOT / "data"
OUTPUT_DATA = output_data_dir(PROJECT_ROOT, CONFIG)
RUN_RECORDS = stage_records_dir(PROJECT_ROOT, CONFIG)

SHP = input_path(PROJECT_ROOT, CONFIG, "parcel_geometry", "path", "data/shapefile/krimpenerwaard_attributes_wgs84.shp")
SHAPE_BASE = SHP.with_suffix("")

MODEL_PARAMS_PARQUET = input_path(PROJECT_ROOT, CONFIG, "model_parameters", "parquet_path", "data/model_params/nl_krimpenerwaard_spams10.parquet")
DEFORMATION_INPUT_NPZ = OUTPUT_DATA / "spams_viewer_input_slice_float32.npz"

INVENTORY_GEOPARQUET_OUT = OUTPUT_DATA / "parcel_inventory.parquet"
INVENTORY_GEOJSON_OUT = OUTPUT_DATA / "parcel_inventory.geojson"
INVENTORY_METADATA_CSV_OUT = OUTPUT_DATA / "parcel_inventory_metadata.csv"
SUMMARY_OUT = OUTPUT_DATA / "parcel_inventory_summary.json"

REPORT_TXT_OUT = RUN_RECORDS / "phase02_parcel_ingest_report.txt"
REPORT_JSON_OUT = RUN_RECORDS / "phase02_parcel_ingest_report.json"

EXPECTED_TOTAL_PARCELS = expected_int(CONFIG, "total_parcels")
EXPECTED_MOVING_PARCELS = expected_int(CONFIG, "moving_parcels")
EXPECTED_BLANK_PARCELS = expected_int(CONFIG, "blank_parcels")


PREFERRED_POPUP_COLUMNS = [
    "knmi_id",
    "knmi_name",
    "soilcode",
    "soil_descr",
    "ghg_mean",
    "glg_mean",
    "region",
    "peilgebied",
    "cropcode",
    "area",
    "perimeter",
]


MODEL_PARAM_COLUMNS_TO_KEEP = [
    "pnt_id",
    "pnt_gid",
    "pnt_lat",
    "pnt_lon",
    "meteo_id",
    "soilcode",
    "vI",
    "var_vI",
    "T",
    "dof",
    "sigmasq_test",
    "xP",
    "xE",
    "xI",
    "tau",
    "var_xP",
    "var_xE",
    "var_xI",
    "cov_xPxE",
    "cov_xPxI",
    "cov_xExI",
]


def fail(message):
    print(f"[FAIL] {message}")
    sys.exit(1)


def ok(message):
    print(f"[OK] {message}")


def warn(message):
    print(f"[WARN] {message}")


def as_int_series(s):
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def geometry_part_count(geom):
    if geom is None:
        return 0
    if geom.geom_type == "MultiPolygon":
        return len(geom.geoms)
    if geom.geom_type == "Polygon":
        return 1
    return 0


def exterior_vertex_count(geom):
    if geom is None:
        return 0

    if geom.geom_type == "Polygon":
        return len(geom.exterior.coords) if geom.exterior is not None else 0

    if geom.geom_type == "MultiPolygon":
        return sum(
            len(part.exterior.coords) if part.exterior is not None else 0
            for part in geom.geoms
        )

    return 0


def interior_ring_count(geom):
    if geom is None:
        return 0

    if geom.geom_type == "Polygon":
        return len(geom.interiors)

    if geom.geom_type == "MultiPolygon":
        return sum(len(part.interiors) for part in geom.geoms)

    return 0


def main():
    OUTPUT_DATA.mkdir(parents=True, exist_ok=True)
    RUN_RECORDS.mkdir(parents=True, exist_ok=True)

    print("\n=== PROTO2 PHASE 02: INGEST PARCELS ===")
    print(f"Project root: {PROJECT_ROOT}")

    required = [
        SHP,
        SHAPE_BASE.with_suffix(".dbf"),
        SHAPE_BASE.with_suffix(".shx"),
        SHAPE_BASE.with_suffix(".prj"),
        DEFORMATION_INPUT_NPZ,
    ]

    missing = [str(p) for p in required if not p.exists()]
    if missing:
        fail(f"Missing required files: {missing}")

    try:
        import geopandas as gpd
    except ImportError:
        fail("geopandas not installed. Install with: conda install -c conda-forge geopandas pyarrow")

    print(f"\nReading parcel geometry:\n  {SHP}")
    gdf = gpd.read_file(SHP)

    if EXPECTED_TOTAL_PARCELS is not None and len(gdf) != EXPECTED_TOTAL_PARCELS:
        fail(f"parcel feature count {len(gdf):,} != expected {EXPECTED_TOTAL_PARCELS:,}")

    ok(f"parcel geometry loaded: {len(gdf):,} features")

    epsg = gdf.crs.to_epsg() if gdf.crs is not None else None
    if epsg != 4326:
        fail(f"expected EPSG:4326, found {gdf.crs}")

    ok("CRS is EPSG:4326")

    join_col = None
    for candidate in geometry_id_candidates(CONFIG):
        if candidate in gdf.columns:
            join_col = candidate
            break

    if join_col is None:
        fail(f"No generic parcel join key found. Tried {geometry_id_candidates(CONFIG)}. Columns: {list(gdf.columns)}")

    ok(f"source geometry join key: {join_col}")

    # Generic template-facing ID.
    gdf["parcel_id"] = as_int_series(gdf[join_col])

    if gdf["parcel_id"].isna().any():
        fail("Some parcel_id values could not be parsed as integers")

    if int(gdf["parcel_id"].duplicated().sum()) != 0:
        fail("parcel_id values are not unique")

    gdf["parcel_id"] = gdf["parcel_id"].astype("int64")

    # Read canonical direct-SPAMS+MC input bundle, only the moving parcel ID set.
    print(f"\nReading canonical SPAMS deformation input:\n  {DEFORMATION_INPUT_NPZ}")
    try:
        with np.load(DEFORMATION_INPUT_NPZ, allow_pickle=False) as deformation_input:
            moving_ids_array = deformation_input["moving_parcel_id"].astype("int64", copy=False)
    except Exception as exc:
        fail(f"Could not read canonical SPAMS deformation input: {exc}")
    moving_ids = set(moving_ids_array.tolist())

    if EXPECTED_MOVING_PARCELS is not None and len(moving_ids) != EXPECTED_MOVING_PARCELS:
        fail(f"moving parcel count {len(moving_ids):,} != expected {EXPECTED_MOVING_PARCELS:,}")

    ok(f"moving parcel IDs loaded: {len(moving_ids):,}")

    # Read model params for popup/model metadata.
    print(f"\nReading model parameter metadata:\n  {MODEL_PARAMS_PARQUET}")
    if not MODEL_PARAMS_PARQUET.exists():
        if should_require_model_parameters(CONFIG):
            fail(f"Missing model parameter parquet: {MODEL_PARAMS_PARQUET}")
        params_all_cols = pd.DataFrame({"pnt_id": sorted(moving_ids)})
        print("[OK] model parameter parquet absent; using displacement IDs only")
    else:
        params_all_cols = pd.read_parquet(MODEL_PARAMS_PARQUET)
    keep_cols = [c for c in MODEL_PARAM_COLUMNS_TO_KEEP if c in params_all_cols.columns]
    params = params_all_cols[keep_cols].copy()

    if "pnt_id" not in params.columns:
        fail("model params missing pnt_id")

    params["pnt_id"] = pd.to_numeric(params["pnt_id"], errors="raise").astype("int64")

    # Avoid duplicate column names after join.
    # geometry has soilcode too, so suffix model-param soilcode.
    rename_map = {}
    for col in params.columns:
        if col == "pnt_id":
            continue
        if col in gdf.columns:
            rename_map[col] = f"model_{col}"
    params = params.rename(columns=rename_map)

    # Attach real/blank status.
    gdf["has_displacement"] = gdf["parcel_id"].isin(moving_ids)
    gdf["parcel_status"] = gdf["has_displacement"].map({
        True: "moving",
        False: "blank_no_displacement",
    })

    real_count = int(gdf["has_displacement"].sum())
    blank_count = int((~gdf["has_displacement"]).sum())

    if (
        (EXPECTED_MOVING_PARCELS is not None and real_count != EXPECTED_MOVING_PARCELS)
        or (EXPECTED_BLANK_PARCELS is not None and blank_count != EXPECTED_BLANK_PARCELS)
    ):
        fail(
            f"real/blank count mismatch: real={real_count:,}, blank={blank_count:,}; "
            f"expected real={EXPECTED_MOVING_PARCELS}, blank={EXPECTED_BLANK_PARCELS}"
        )

    ok(f"real/blank parcel counts: real={real_count:,}, blank={blank_count:,}")

    # Join model parameters to moving parcels.
    gdf = gdf.merge(
        params,
        how="left",
        left_on="parcel_id",
        right_on="pnt_id",
        validate="one_to_one",
    )

    # Confirm only blank parcels have no pnt_id.
    missing_model_on_moving = int(gdf.loc[gdf["has_displacement"], "pnt_id"].isna().sum())
    model_on_blank = int(gdf.loc[~gdf["has_displacement"], "pnt_id"].notna().sum())

    if missing_model_on_moving or model_on_blank:
        fail(
            f"model join mismatch: missing_model_on_moving={missing_model_on_moving:,}, "
            f"model_on_blank={model_on_blank:,}"
        )

    ok("model parameters join correctly to moving parcels only")

    # Geometry diagnostics.
    gdf["geometry_type"] = gdf.geometry.geom_type
    gdf["is_multipart"] = gdf["geometry_type"].eq("MultiPolygon")
    gdf["part_count"] = gdf.geometry.apply(geometry_part_count).astype("int32")
    gdf["exterior_vertex_count"] = gdf.geometry.apply(exterior_vertex_count).astype("int32")
    gdf["interior_ring_count"] = gdf.geometry.apply(interior_ring_count).astype("int32")

    # Compute centroids in a metric CRS, then transform them back to WGS84.
    # This avoids geographic-CRS centroid warnings and gives stable popup/search locations.
    centroids_wgs84 = gdf.to_crs(epsg=28992).geometry.centroid.to_crs(epsg=4326)
    gdf["centroid_lon"] = centroids_wgs84.x
    gdf["centroid_lat"] = centroids_wgs84.y

    invalid_geom = int((~gdf.geometry.is_valid).sum())
    empty_geom = int(gdf.geometry.is_empty.sum())
    null_geom = int(gdf.geometry.isna().sum())

    if invalid_geom or empty_geom or null_geom:
        fail(f"geometry issue after ingest: invalid={invalid_geom}, empty={empty_geom}, null={null_geom}")

    ok("geometry valid after ingest")

    geom_type_counts = gdf["geometry_type"].value_counts().to_dict()
    multipart_count = int(gdf["is_multipart"].sum())
    total_exterior_vertices = int(gdf["exterior_vertex_count"].sum())
    total_interior_rings = int(gdf["interior_ring_count"].sum())

    bounds = gdf.total_bounds.tolist()  # minx, miny, maxx, maxy

    # Template metadata columns for non-geometry table.
    possible_metadata_cols = [
        "parcel_id",
        "parcel_status",
        "has_displacement",
        "pnt_id",
        "pnt_gid",
        "pnt_lat",
        "pnt_lon",
        "centroid_lat",
        "centroid_lon",
        "geometry_type",
        "is_multipart",
        "part_count",
        "exterior_vertex_count",
        "interior_ring_count",
    ]

    for col in PREFERRED_POPUP_COLUMNS:
        if col in gdf.columns and col not in possible_metadata_cols:
            possible_metadata_cols.append(col)

    for col in [
        "meteo_id",
        "model_soilcode",
        "vI",
        "var_vI",
        "T",
        "dof",
        "sigmasq_test",
        "xP",
        "xE",
        "xI",
        "tau",
    ]:
        if col in gdf.columns and col not in possible_metadata_cols:
            possible_metadata_cols.append(col)

    metadata_cols = [c for c in possible_metadata_cols if c in gdf.columns]

    # Summary object.
    summary = {
        "source_shapefile": str(SHP),
        "source_geometry_join_key": join_col,
        "template_join_key": "parcel_id",
        "crs": str(gdf.crs),
        "epsg": epsg,
        "total_parcels": int(len(gdf)),
        "moving_parcels": real_count,
        "blank_parcels": blank_count,
        "geometry_type_counts": {k: int(v) for k, v in geom_type_counts.items()},
        "multipart_count": multipart_count,
        "total_exterior_vertices": total_exterior_vertices,
        "total_interior_rings": total_interior_rings,
        "bounds_wgs84": {
            "west": float(bounds[0]),
            "south": float(bounds[1]),
            "east": float(bounds[2]),
            "north": float(bounds[3]),
        },
        "popup_candidate_columns": [c for c in PREFERRED_POPUP_COLUMNS if c in gdf.columns],
        "metadata_columns_written": metadata_cols,
        "outputs": {
            "parcel_inventory_parquet": str(INVENTORY_GEOPARQUET_OUT),
            "parcel_inventory_geojson": str(INVENTORY_GEOJSON_OUT),
            "parcel_inventory_metadata_csv": str(INVENTORY_METADATA_CSV_OUT),
            "parcel_inventory_summary": str(SUMMARY_OUT),
        },
        "status_rule": {
            "moving": "parcel has source displacement time series",
            "blank_no_displacement": "parcel geometry exists but no displacement time series is available",
        },
    }

    print("\nWriting parcel inventory outputs...")

    # GeoParquet is the canonical geometry inventory for downstream scripts.
    gdf.to_parquet(INVENTORY_GEOPARQUET_OUT, index=False)
    ok(f"wrote {INVENTORY_GEOPARQUET_OUT}")

    # GeoJSON is useful for quick inspection/debugging. It may be larger/slower.
    gdf.to_file(INVENTORY_GEOJSON_OUT, driver="GeoJSON")
    ok(f"wrote {INVENTORY_GEOJSON_OUT}")

    # Metadata-only CSV for quick spreadsheet inspection.
    metadata_df = pd.DataFrame(gdf.drop(columns="geometry"))[metadata_cols]
    metadata_df.to_csv(INVENTORY_METADATA_CSV_OUT, index=False)
    ok(f"wrote {INVENTORY_METADATA_CSV_OUT}")

    SUMMARY_OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    ok(f"wrote {SUMMARY_OUT}")

    report = {
        "project_root": str(PROJECT_ROOT),
        "summary": summary,
    }

    REPORT_JSON_OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    txt_lines = [
        "PROTO2 PHASE 02 PARCEL INGEST REPORT",
        "",
        f"source shapefile: {SHP}",
        f"source geometry join key: {join_col}",
        "template join key: parcel_id",
        f"crs: {gdf.crs}",
        "",
        f"total parcels: {len(gdf):,}",
        f"moving parcels: {real_count:,}",
        f"blank parcels: {blank_count:,}",
        "",
        f"geometry types: {geom_type_counts}",
        f"multipart parcels: {multipart_count:,}",
        f"total exterior vertices: {total_exterior_vertices:,}",
        f"total interior rings: {total_interior_rings:,}",
        "",
        "bounds WGS84:",
        f"west={bounds[0]:.8f}, south={bounds[1]:.8f}, east={bounds[2]:.8f}, north={bounds[3]:.8f}",
        "",
        "outputs:",
        f"- {INVENTORY_GEOPARQUET_OUT}",
        f"- {INVENTORY_GEOJSON_OUT}",
        f"- {INVENTORY_METADATA_CSV_OUT}",
        f"- {SUMMARY_OUT}",
    ]

    REPORT_TXT_OUT.write_text("\n".join(txt_lines), encoding="utf-8")
    ok(f"wrote {REPORT_JSON_OUT}")
    ok(f"wrote {REPORT_TXT_OUT}")

    print("\n=== SUMMARY ===")
    print(f"Total parcels: {len(gdf):,}")
    print(f"Moving parcels: {real_count:,}")
    print(f"Blank parcels: {blank_count:,}")
    print(f"Geometry types: {geom_type_counts}")
    print(f"Multipart parcels: {multipart_count:,}")
    print(f"Exterior vertices: {total_exterior_vertices:,}")
    print("\nPHASE 02 RESULT: PASS. Canonical parcel inventory written.")


if __name__ == "__main__":
    main()