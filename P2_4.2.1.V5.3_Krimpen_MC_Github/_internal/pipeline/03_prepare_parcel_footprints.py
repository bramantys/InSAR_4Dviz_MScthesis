from pathlib import Path
import json
import sys

import pandas as pd

from _proto2_config import expected_int, load_project_config, output_data_dir, stage_records_dir


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = load_project_config(PROJECT_ROOT)

OUTPUT_DATA = output_data_dir(PROJECT_ROOT, CONFIG)
RUN_RECORDS = stage_records_dir(PROJECT_ROOT, CONFIG)

PARCEL_INVENTORY = OUTPUT_DATA / "parcel_inventory.parquet"

FOOTPRINT_PARTS_PARQUET_OUT = OUTPUT_DATA / "parcel_footprints_parts.parquet"
FOOTPRINT_PARTS_GEOJSON_OUT = OUTPUT_DATA / "parcel_footprints_parts.geojson"
FOOTPRINT_VERTICES_CSV_OUT = OUTPUT_DATA / "parcel_footprint_vertices.csv"
FOOTPRINT_SUMMARY_OUT = OUTPUT_DATA / "parcel_footprint_summary.json"

REPORT_TXT_OUT = RUN_RECORDS / "phase03_footprint_report.txt"
REPORT_JSON_OUT = RUN_RECORDS / "phase03_footprint_report.json"

EXPECTED_TOTAL_PARCELS = expected_int(CONFIG, "total_parcels")
EXPECTED_MOVING_PARCELS = expected_int(CONFIG, "moving_parcels")
EXPECTED_BLANK_PARCELS = expected_int(CONFIG, "blank_parcels")
EXPECTED_MULTIPART_PARCELS = expected_int(CONFIG, "multipart_parcels")

# For Krimpenerwaard / Netherlands metric diagnostics.
# Template note: for another parcel case, make this configurable later.
METRIC_CRS_EPSG = 28992  # RD New / Amersfoort


def fail(message):
    print(f"[FAIL] {message}")
    sys.exit(1)


def ok(message):
    print(f"[OK] {message}")


def warn(message):
    print(f"[WARN] {message}")


def iter_polygon_parts(geom):
    if geom is None:
        return

    if geom.geom_type == "Polygon":
        yield 0, geom
    elif geom.geom_type == "MultiPolygon":
        for i, part in enumerate(geom.geoms):
            yield i, part
    else:
        return


def ring_rows_for_polygon(row, polygon_wgs84, polygon_metric, footprint_id):
    """
    Build vertex rows for one polygon part.

    Exterior ring and interior rings are both kept.
    Future triangulation/extrusion can use this to preserve holes.
    """
    rows = []

    # Exterior ring
    ext_wgs = list(polygon_wgs84.exterior.coords)
    ext_met = list(polygon_metric.exterior.coords)

    for i, ((lon, lat), (x_m, y_m)) in enumerate(zip(ext_wgs, ext_met)):
        rows.append({
            "parcel_id": row["parcel_id"],
            "footprint_id": footprint_id,
            "part_index": row["part_index"],
            "ring_type": "exterior",
            "ring_index": 0,
            "vertex_index": i,
            "lon": lon,
            "lat": lat,
            "x_m": x_m,
            "y_m": y_m,
        })

    # Interior rings / holes
    for ring_index, (ring_wgs, ring_met) in enumerate(
        zip(polygon_wgs84.interiors, polygon_metric.interiors),
        start=1,
    ):
        coords_wgs = list(ring_wgs.coords)
        coords_met = list(ring_met.coords)

        for i, ((lon, lat), (x_m, y_m)) in enumerate(zip(coords_wgs, coords_met)):
            rows.append({
                "parcel_id": row["parcel_id"],
                "footprint_id": footprint_id,
                "part_index": row["part_index"],
                "ring_type": "interior",
                "ring_index": ring_index,
                "vertex_index": i,
                "lon": lon,
                "lat": lat,
                "x_m": x_m,
                "y_m": y_m,
            })

    return rows


def main():
    OUTPUT_DATA.mkdir(parents=True, exist_ok=True)
    RUN_RECORDS.mkdir(parents=True, exist_ok=True)

    print("\n=== PROTO2 PHASE 03: PREPARE PARCEL FOOTPRINTS ===")
    print(f"Project root: {PROJECT_ROOT}")

    if not PARCEL_INVENTORY.exists():
        fail(f"Missing parcel inventory: {PARCEL_INVENTORY}")

    try:
        import geopandas as gpd
    except ImportError:
        fail("geopandas not installed. Install with: conda install -c conda-forge geopandas pyarrow")

    print(f"\nReading parcel inventory:\n  {PARCEL_INVENTORY}")
    inventory = gpd.read_parquet(PARCEL_INVENTORY)

    if EXPECTED_TOTAL_PARCELS is not None and len(inventory) != EXPECTED_TOTAL_PARCELS:
        fail(f"inventory parcel count {len(inventory):,} != expected {EXPECTED_TOTAL_PARCELS:,}")

    if inventory.crs is None:
        fail("parcel inventory has no CRS")

    epsg = inventory.crs.to_epsg() # type: ignore
    if epsg != 4326:
        fail(f"expected inventory CRS EPSG:4326, found {inventory.crs}")

    ok(f"parcel inventory loaded: {len(inventory):,} parcels")
    ok("inventory CRS is EPSG:4326")

    required_cols = ["parcel_id", "parcel_status", "has_displacement", "geometry"]
    missing_cols = [c for c in required_cols if c not in inventory.columns]
    if missing_cols:
        fail(f"parcel inventory missing required columns: {missing_cols}")

    moving_count = int(inventory["has_displacement"].sum())
    blank_count = int((~inventory["has_displacement"]).sum())
    multipart_count = int(inventory.geometry.geom_type.eq("MultiPolygon").sum())

    if (EXPECTED_MOVING_PARCELS is not None and moving_count != EXPECTED_MOVING_PARCELS) or (EXPECTED_BLANK_PARCELS is not None and blank_count != EXPECTED_BLANK_PARCELS):
        fail(
            f"moving/blank mismatch: moving={moving_count:,}, blank={blank_count:,}; "
            f"expected moving={EXPECTED_MOVING_PARCELS}, blank={EXPECTED_BLANK_PARCELS}"
        )

    ok(f"moving/blank counts confirmed: moving={moving_count:,}, blank={blank_count:,}")

    if EXPECTED_MULTIPART_PARCELS is not None and multipart_count != EXPECTED_MULTIPART_PARCELS:
        warn(
            f"multipart count {multipart_count:,} differs from expected {EXPECTED_MULTIPART_PARCELS:,}; "
            "continuing because this is not fatal"
        )
    else:
        ok(f"multipart parcel count confirmed: {multipart_count:,}")

    invalid = int((~inventory.geometry.is_valid).sum())
    empty = int(inventory.geometry.is_empty.sum())
    null = int(inventory.geometry.isna().sum())

    if invalid or empty or null:
        fail(f"invalid/empty/null geometry found: invalid={invalid}, empty={empty}, null={null}")

    ok("all inventory geometries valid/non-empty")

    # Project once for metric diagnostics and triangulation-friendly vertex coordinates.
    print(f"\nProjecting geometry to metric CRS EPSG:{METRIC_CRS_EPSG} for diagnostics/vertices...")
    inventory_metric = inventory.to_crs(epsg=METRIC_CRS_EPSG)
    ok(f"projected to EPSG:{METRIC_CRS_EPSG}")

    # Build one-row-per-polygon-part table.
    part_records = []
    vertex_records = []

    print("\nExploding multipart parcels into polygon footprint parts...")

    for row_idx, row in inventory.iterrows():
        geom_wgs = row.geometry
        geom_metric = inventory_metric.loc[row_idx].geometry # type: ignore

        if geom_wgs.geom_type == "Polygon":
            wgs_parts = [geom_wgs]
            metric_parts = [geom_metric]
        elif geom_wgs.geom_type == "MultiPolygon":
            wgs_parts = list(geom_wgs.geoms)
            metric_parts = list(geom_metric.geoms)
        else:
            fail(f"Unsupported geometry type for parcel_id={row['parcel_id']}: {geom_wgs.geom_type}")

        for part_index, (part_wgs, part_metric) in enumerate(zip(wgs_parts, metric_parts)):
            footprint_id = f"{int(row['parcel_id'])}_{part_index}"

            minx, miny, maxx, maxy = part_wgs.bounds
            minx_m, miny_m, maxx_m, maxy_m = part_metric.bounds

            exterior_vertices = len(part_wgs.exterior.coords)
            interior_rings = len(part_wgs.interiors)
            interior_vertices = sum(len(r.coords) for r in part_wgs.interiors)

            part_records.append({
                "parcel_id": int(row["parcel_id"]),
                "footprint_id": footprint_id,
                "part_index": int(part_index),
                "parcel_status": row["parcel_status"],
                "has_displacement": bool(row["has_displacement"]),
                "source_geometry_type": geom_wgs.geom_type,
                "footprint_geometry_type": part_wgs.geom_type,
                "exterior_vertex_count": int(exterior_vertices),
                "interior_ring_count": int(interior_rings),
                "interior_vertex_count": int(interior_vertices),
                "area_m2_calc": float(part_metric.area),
                "perimeter_m_calc": float(part_metric.length),
                "west": float(minx),
                "south": float(miny),
                "east": float(maxx),
                "north": float(maxy),
                "minx_m": float(minx_m),
                "miny_m": float(miny_m),
                "maxx_m": float(maxx_m),
                "maxy_m": float(maxy_m),
                "geometry": part_wgs,
            })

            vertex_rows = ring_rows_for_polygon(
                {
                    "parcel_id": int(row["parcel_id"]),
                    "part_index": int(part_index),
                },
                part_wgs,
                part_metric,
                footprint_id,
            )
            vertex_records.extend(vertex_rows)

    parts = gpd.GeoDataFrame(part_records, geometry="geometry", crs=inventory.crs)
    vertices = pd.DataFrame(vertex_records)

    part_count = len(parts)
    moving_part_count = int(parts["has_displacement"].sum())
    blank_part_count = int((~parts["has_displacement"]).sum())
    vertex_count = len(vertices)
    exterior_vertex_count = int((vertices["ring_type"] == "exterior").sum())
    interior_vertex_count = int((vertices["ring_type"] == "interior").sum())
    parcels_with_holes = int(parts["interior_ring_count"].gt(0).sum())

    ok(f"footprint parts created: {part_count:,}")
    ok(f"moving footprint parts: {moving_part_count:,}")
    ok(f"blank footprint parts: {blank_part_count:,}")
    ok(f"vertex rows created: {vertex_count:,}")

    if part_count < len(inventory):
        fail(f"part count {part_count:,} is less than parcel count {len(inventory):,}; impossible if polygons exploded correctly")

    # Check each parcel still represented.
    represented_parcels = int(parts["parcel_id"].nunique())
    if represented_parcels != len(inventory):
        fail(f"represented parcel count {represented_parcels:,} != expected inventory count {len(inventory):,}")

    ok("all parcels represented by at least one footprint part")

    # Summary.
    bounds = parts.total_bounds.tolist()

    largest_parts = (
        parts.sort_values("area_m2_calc", ascending=False)
        [["parcel_id", "footprint_id", "area_m2_calc", "exterior_vertex_count", "interior_ring_count", "has_displacement"]]
        .head(10)
        .to_dict(orient="records")
    )

    most_vertices = (
        parts.sort_values("exterior_vertex_count", ascending=False)
        [["parcel_id", "footprint_id", "area_m2_calc", "exterior_vertex_count", "interior_ring_count", "has_displacement"]]
        .head(10)
        .to_dict(orient="records")
    )

    summary = {
        "source_inventory": str(PARCEL_INVENTORY),
        "input_crs": str(inventory.crs),
        "input_epsg": epsg,
        "metric_crs_epsg": METRIC_CRS_EPSG,
        "total_parcels": int(len(inventory)),
        "moving_parcels": moving_count,
        "blank_parcels": blank_count,
        "multipart_parcels": multipart_count,
        "footprint_parts": part_count,
        "moving_footprint_parts": moving_part_count,
        "blank_footprint_parts": blank_part_count,
        "vertex_rows": vertex_count,
        "exterior_vertex_rows": exterior_vertex_count,
        "interior_vertex_rows": interior_vertex_count,
        "footprint_parts_with_holes": parcels_with_holes,
        "area_m2_calc": {
            "min": float(parts["area_m2_calc"].min()),
            "median": float(parts["area_m2_calc"].median()),
            "mean": float(parts["area_m2_calc"].mean()),
            "max": float(parts["area_m2_calc"].max()),
            "sum": float(parts["area_m2_calc"].sum()),
        },
        "exterior_vertex_count": {
            "min": int(parts["exterior_vertex_count"].min()),
            "median": float(parts["exterior_vertex_count"].median()),
            "mean": float(parts["exterior_vertex_count"].mean()),
            "max": int(parts["exterior_vertex_count"].max()),
            "sum": int(parts["exterior_vertex_count"].sum()),
        },
        "bounds_wgs84": {
            "west": float(bounds[0]),
            "south": float(bounds[1]),
            "east": float(bounds[2]),
            "north": float(bounds[3]),
        },
        "largest_parts": largest_parts,
        "most_vertex_heavy_parts": most_vertices,
        "outputs": {
            "footprint_parts_parquet": str(FOOTPRINT_PARTS_PARQUET_OUT),
            "footprint_parts_geojson": str(FOOTPRINT_PARTS_GEOJSON_OUT),
            "footprint_vertices_csv": str(FOOTPRINT_VERTICES_CSV_OUT),
            "footprint_summary": str(FOOTPRINT_SUMMARY_OUT),
        },
        "notes": {
            "multipart_meaning": "one parcel_id may contain multiple separate polygon pieces",
            "interior_ring_meaning": "a hole inside a polygon footprint",
            "template_status": "generic parcel footprint product; not tied to source CSV naming",
        },
    }

    print("\nWriting footprint outputs...")

    parts.to_parquet(FOOTPRINT_PARTS_PARQUET_OUT, index=False)
    ok(f"wrote {FOOTPRINT_PARTS_PARQUET_OUT}")

    parts.to_file(FOOTPRINT_PARTS_GEOJSON_OUT, driver="GeoJSON")
    ok(f"wrote {FOOTPRINT_PARTS_GEOJSON_OUT}")

    vertices.to_csv(FOOTPRINT_VERTICES_CSV_OUT, index=False)
    ok(f"wrote {FOOTPRINT_VERTICES_CSV_OUT}")

    FOOTPRINT_SUMMARY_OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    ok(f"wrote {FOOTPRINT_SUMMARY_OUT}")

    report = {
        "project_root": str(PROJECT_ROOT),
        "summary": summary,
    }

    REPORT_JSON_OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    txt_lines = [
        "PROTO2 PHASE 03 FOOTPRINT REPORT",
        "",
        f"source inventory: {PARCEL_INVENTORY}",
        f"metric CRS EPSG: {METRIC_CRS_EPSG}",
        "",
        f"total parcels: {len(inventory):,}",
        f"moving parcels: {moving_count:,}",
        f"blank parcels: {blank_count:,}",
        f"multipart parcels: {multipart_count:,}",
        "",
        f"footprint parts: {part_count:,}",
        f"moving footprint parts: {moving_part_count:,}",
        f"blank footprint parts: {blank_part_count:,}",
        f"vertex rows: {vertex_count:,}",
        f"exterior vertex rows: {exterior_vertex_count:,}",
        f"interior vertex rows: {interior_vertex_count:,}",
        f"footprint parts with holes: {parcels_with_holes:,}",
        "",
        "area_m2_calc:",
        f"min={summary['area_m2_calc']['min']:.3f}",
        f"median={summary['area_m2_calc']['median']:.3f}",
        f"mean={summary['area_m2_calc']['mean']:.3f}",
        f"max={summary['area_m2_calc']['max']:.3f}",
        f"sum={summary['area_m2_calc']['sum']:.3f}",
        "",
        "bounds WGS84:",
        f"west={bounds[0]:.8f}, south={bounds[1]:.8f}, east={bounds[2]:.8f}, north={bounds[3]:.8f}",
        "",
        "outputs:",
        f"- {FOOTPRINT_PARTS_PARQUET_OUT}",
        f"- {FOOTPRINT_PARTS_GEOJSON_OUT}",
        f"- {FOOTPRINT_VERTICES_CSV_OUT}",
        f"- {FOOTPRINT_SUMMARY_OUT}",
    ]

    REPORT_TXT_OUT.write_text("\n".join(txt_lines), encoding="utf-8")
    ok(f"wrote {REPORT_JSON_OUT}")
    ok(f"wrote {REPORT_TXT_OUT}")

    print("\n=== SUMMARY ===")
    print(f"Parcels: {len(inventory):,}")
    print(f"Footprint parts: {part_count:,}")
    print(f"Multipart parcels: {multipart_count:,}")
    print(f"Vertex rows: {vertex_count:,}")
    print(f"Area total: {summary['area_m2_calc']['sum']:.3f} m²")
    print("\nPHASE 03 RESULT: PASS. Parcel footprints prepared for triangulation/extrusion.")


if __name__ == "__main__":
    main()