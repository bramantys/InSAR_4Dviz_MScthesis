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

FOOTPRINT_PARTS = OUTPUT_DATA / "parcel_footprints_parts.parquet"

MESH_VERTICES_OUT = OUTPUT_DATA / "parcel_cap_mesh_vertices.parquet"
MESH_TRIANGLES_OUT = OUTPUT_DATA / "parcel_cap_mesh_triangles.parquet"
MESH_SUMMARY_OUT = OUTPUT_DATA / "parcel_cap_mesh_summary.json"

REPORT_TXT_OUT = RUN_RECORDS / "phase04_triangulation_report.txt"
REPORT_JSON_OUT = RUN_RECORDS / "phase04_triangulation_report.json"

METRIC_CRS_EPSG = 28992

EXPECTED_PARTS = expected_int(CONFIG, "footprint_parts")
EXPECTED_MOVING_PARTS = expected_int(CONFIG, "moving_footprint_parts")
EXPECTED_BLANK_PARTS = expected_int(CONFIG, "blank_footprint_parts")


def fail(message):
    print(f"[FAIL] {message}")
    sys.exit(1)


def ok(message):
    print(f"[OK] {message}")


def warn(message):
    print(f"[WARN] {message}")


def ring_without_closure(ring):
    coords = list(ring.coords)
    if len(coords) >= 2 and coords[0] == coords[-1]:
        coords = coords[:-1]
    return coords


def polygon_coords_for_earcut(poly_metric, poly_wgs84):
    """
    Returns:
      xy_metric: Nx2 float64 array for earcut
      lonlat: Nx2 float64 array matching the same local vertex order
      ring_ends: uint32 array of cumulative ring end indices
      ring_info: list of ring metadata
    """
    metric_coords = []
    lonlat_coords = []
    ring_ends = []
    ring_info = []

    # Exterior ring
    ext_metric = ring_without_closure(poly_metric.exterior)
    ext_wgs84 = ring_without_closure(poly_wgs84.exterior)

    if len(ext_metric) != len(ext_wgs84):
        raise ValueError("metric/WGS84 exterior coordinate length mismatch")

    metric_coords.extend(ext_metric)
    lonlat_coords.extend(ext_wgs84)
    ring_ends.append(len(metric_coords))
    ring_info.append({
        "ring_type": "exterior",
        "ring_index": 0,
        "vertex_count": len(ext_metric),
    })

    # Interior rings / holes
    for hole_index, (hole_metric, hole_wgs84) in enumerate(
        zip(poly_metric.interiors, poly_wgs84.interiors),
        start=1,
    ):
        h_metric = ring_without_closure(hole_metric)
        h_wgs84 = ring_without_closure(hole_wgs84)

        if len(h_metric) != len(h_wgs84):
            raise ValueError("metric/WGS84 interior coordinate length mismatch")

        metric_coords.extend(h_metric)
        lonlat_coords.extend(h_wgs84)
        ring_ends.append(len(metric_coords))
        ring_info.append({
            "ring_type": "interior",
            "ring_index": hole_index,
            "vertex_count": len(h_metric),
        })

    xy_metric = np.asarray(metric_coords, dtype=np.float64)
    lonlat = np.asarray(lonlat_coords, dtype=np.float64)
    ring_ends = np.asarray(ring_ends, dtype=np.uint32)

    return xy_metric, lonlat, ring_ends, ring_info


def earcut_triangulate(xy_metric, ring_ends):
    """
    Wrapper for mapbox_earcut.
    Newer Python package exposes triangulate_float64(vertices, ring_end_indices).
    """
    try:
        import mapbox_earcut as earcut
    except ImportError:
        fail(
            "mapbox-earcut is not installed. Install it with:\n"
            "  pip install mapbox-earcut"
        )

    if xy_metric.ndim != 2 or xy_metric.shape[1] != 2:
        raise ValueError(f"expected xy_metric shape Nx2, got {xy_metric.shape}")

    if len(xy_metric) < 3:
        return np.asarray([], dtype=np.uint32)

    try:
        tri = earcut.triangulate_float64(xy_metric, ring_ends)
    except TypeError:
        # Fallback for possible older variants.
        tri = earcut.triangulate_float64(xy_metric.flatten(), ring_ends)

    tri = np.asarray(tri, dtype=np.uint32)

    if len(tri) % 3 != 0:
        raise ValueError("earcut returned index array length not divisible by 3")

    return tri.reshape((-1, 3))


def triangle_signed_area_xy(a, b, c):
    return 0.5 * (
        (b[0] - a[0]) * (c[1] - a[1])
        - (c[0] - a[0]) * (b[1] - a[1])
    )


def main():
    OUTPUT_DATA.mkdir(parents=True, exist_ok=True)
    RUN_RECORDS.mkdir(parents=True, exist_ok=True)

    print("\n=== PROTO2 PHASE 04: TRIANGULATE PARCEL CAPS ===")
    print(f"Project root: {PROJECT_ROOT}")

    if not FOOTPRINT_PARTS.exists():
        fail(f"Missing footprint parts file: {FOOTPRINT_PARTS}")

    try:
        import geopandas as gpd
    except ImportError:
        fail("geopandas not installed. Install with: conda install -c conda-forge geopandas pyarrow")

    print(f"\nReading footprint parts:\n  {FOOTPRINT_PARTS}")
    parts = gpd.read_parquet(FOOTPRINT_PARTS)

    if EXPECTED_PARTS is not None and len(parts) != EXPECTED_PARTS:
        fail(f"footprint part count {len(parts):,} != expected {EXPECTED_PARTS:,}")

    moving_parts = int(parts["has_displacement"].sum())
    blank_parts = int((~parts["has_displacement"]).sum())

    if (EXPECTED_MOVING_PARTS is not None and moving_parts != EXPECTED_MOVING_PARTS) or (EXPECTED_BLANK_PARTS is not None and blank_parts != EXPECTED_BLANK_PARTS):
        fail(
            f"moving/blank footprint part mismatch: moving={moving_parts:,}, blank={blank_parts:,}; "
            f"expected moving={EXPECTED_MOVING_PARTS}, blank={EXPECTED_BLANK_PARTS}"
        )

    ok(f"footprint parts loaded: {len(parts):,}")
    ok(f"moving/blank footprint parts confirmed: moving={moving_parts:,}, blank={blank_parts:,}")

    if parts.crs is None:
        fail("footprint parts have no CRS")

    if parts.crs.to_epsg() != 4326: # type: ignore
        fail(f"expected EPSG:4326, got {parts.crs}")

    print(f"\nProjecting to metric CRS EPSG:{METRIC_CRS_EPSG} for triangulation...")
    parts_metric = parts.to_crs(epsg=METRIC_CRS_EPSG)
    ok(f"projected to EPSG:{METRIC_CRS_EPSG}")

    vertex_records = []
    triangle_records = []
    failed_parts = []

    global_vertex_index = 0
    global_triangle_index = 0

    print("\nTriangulating footprint parts...")

    for row_idx, row in parts.iterrows():
        geom_wgs84 = row.geometry
        geom_metric = parts_metric.loc[row_idx].geometry # type: ignore

        parcel_id = int(row["parcel_id"])
        footprint_id = str(row["footprint_id"])
        part_index = int(row["part_index"])
        has_displacement = bool(row["has_displacement"])
        parcel_status = str(row["parcel_status"])

        if geom_wgs84.geom_type != "Polygon":
            failed_parts.append({
                "parcel_id": parcel_id,
                "footprint_id": footprint_id,
                "reason": f"unsupported geometry type {geom_wgs84.geom_type}",
            })
            continue

        try:
            xy_metric, lonlat, ring_ends, ring_info = polygon_coords_for_earcut(
                geom_metric,
                geom_wgs84,
            )

            triangles_local = earcut_triangulate(xy_metric, ring_ends)

            if len(triangles_local) == 0:
                failed_parts.append({
                    "parcel_id": parcel_id,
                    "footprint_id": footprint_id,
                    "reason": "earcut returned zero triangles",
                })
                continue

            local_to_global = {}

            for local_i, ((x_m, y_m), (lon, lat)) in enumerate(zip(xy_metric, lonlat)):
                gvi = global_vertex_index
                local_to_global[local_i] = gvi

                vertex_records.append({
                    "global_vertex_index": gvi,
                    "parcel_id": parcel_id,
                    "footprint_id": footprint_id,
                    "part_index": part_index,
                    "local_vertex_index": int(local_i),
                    "has_displacement": has_displacement,
                    "parcel_status": parcel_status,
                    "x_m": float(x_m),
                    "y_m": float(y_m),
                    "lon": float(lon),
                    "lat": float(lat),
                })

                global_vertex_index += 1

            for local_tri in triangles_local:
                i0, i1, i2 = [int(v) for v in local_tri]

                p0 = xy_metric[i0]
                p1 = xy_metric[i1]
                p2 = xy_metric[i2]

                signed_area = triangle_signed_area_xy(p0, p1, p2)
                area_m2 = abs(signed_area)

                triangle_records.append({
                    "global_triangle_index": global_triangle_index,
                    "parcel_id": parcel_id,
                    "footprint_id": footprint_id,
                    "part_index": part_index,
                    "has_displacement": has_displacement,
                    "parcel_status": parcel_status,
                    "v0": int(local_to_global[i0]),
                    "v1": int(local_to_global[i1]),
                    "v2": int(local_to_global[i2]),
                    "local_v0": i0,
                    "local_v1": i1,
                    "local_v2": i2,
                    "triangle_area_m2": float(area_m2),
                    "triangle_signed_area_m2": float(signed_area),
                })

                global_triangle_index += 1

        except Exception as e:
            failed_parts.append({
                "parcel_id": parcel_id,
                "footprint_id": footprint_id,
                "reason": repr(e),
            })

    if failed_parts:
        fail(
            f"triangulation failed for {len(failed_parts):,} footprint parts. "
            f"First failures: {failed_parts[:5]}"
        )

    vertices = pd.DataFrame(vertex_records)
    triangles = pd.DataFrame(triangle_records)

    if vertices.empty or triangles.empty:
        fail("triangulation produced empty mesh tables")

    ok(f"mesh vertices created: {len(vertices):,}")
    ok(f"mesh triangles created: {len(triangles):,}")

    # Integrity checks.
    min_ref = int(triangles[["v0", "v1", "v2"]].min().min())
    max_ref = int(triangles[["v0", "v1", "v2"]].max().max())
    max_vertex = int(vertices["global_vertex_index"].max())

    if min_ref < 0 or max_ref > max_vertex:
        fail(f"triangle index reference out of range: min_ref={min_ref}, max_ref={max_ref}, max_vertex={max_vertex}")

    ok("triangle vertex references are in range")


    ZERO_AREA_EPS_M2 = 1e-12

    zero_area_mask = triangles["triangle_area_m2"] <= ZERO_AREA_EPS_M2
    zero_area_triangles = int(zero_area_mask.sum())

    if zero_area_triangles:
        zero_area_examples = (
            triangles.loc[
                zero_area_mask,
                ["parcel_id", "footprint_id", "part_index", "triangle_area_m2"]
            ]
            .head(10)
            .to_dict(orient="records")
        )

        warn(
            f"found {zero_area_triangles:,} near-zero-area triangles; "
            "dropping them before writing mesh outputs"
        )
        warn(f"near-zero examples: {zero_area_examples}")

        triangles = triangles.loc[~zero_area_mask].copy().reset_index(drop=True)
        triangles["global_triangle_index"] = np.arange(len(triangles), dtype=np.int64)

        ok(f"remaining mesh triangles after cleanup: {len(triangles):,}")
    else:
        ok("no near-zero-area triangles")

    represented_parts = int(triangles["footprint_id"].nunique())
    if represented_parts != len(parts):
        fail(f"triangulated footprint part count {represented_parts:,} != expected input parts {len(parts):,}")

    ok("all footprint parts represented in triangle table")

    moving_triangles = int(triangles["has_displacement"].sum())
    blank_triangles = int((~triangles["has_displacement"]).sum())
    moving_vertices = int(vertices["has_displacement"].sum())
    blank_vertices = int((~vertices["has_displacement"]).sum())

    # Area sanity: compare summed triangle areas to footprint polygon area.
    tri_area_by_part = triangles.groupby("footprint_id")["triangle_area_m2"].sum()
    footprint_area_by_part = parts.set_index("footprint_id")["area_m2_calc"]

    area_compare = pd.DataFrame({
        "tri_area_m2": tri_area_by_part,
        "footprint_area_m2": footprint_area_by_part,
    })
    area_compare["abs_diff_m2"] = (area_compare["tri_area_m2"] - area_compare["footprint_area_m2"]).abs()
    area_compare["rel_diff"] = area_compare["abs_diff_m2"] / area_compare["footprint_area_m2"].replace(0, np.nan)

    max_abs_area_diff = float(area_compare["abs_diff_m2"].max())
    max_rel_area_diff = float(area_compare["rel_diff"].max())

    # Earcut area should match closely, but allow tiny numerical tolerance.
    bad_area_parts = int((area_compare["rel_diff"] > 1e-6).sum())
    if bad_area_parts:
        fail(
            f"triangle area does not match footprint area for {bad_area_parts:,} parts; "
            f"max_abs_diff={max_abs_area_diff:.9f} m2, max_rel_diff={max_rel_area_diff:.9g}"
        )

    ok(
        "triangle areas match footprint polygon areas "
        f"(max_abs_diff={max_abs_area_diff:.9f} m2, max_rel_diff={max_rel_area_diff:.9g})"
    )

    # Write outputs.
    print("\nWriting mesh outputs...")

    vertices.to_parquet(MESH_VERTICES_OUT, index=False)
    ok(f"wrote {MESH_VERTICES_OUT}")

    triangles.to_parquet(MESH_TRIANGLES_OUT, index=False)
    ok(f"wrote {MESH_TRIANGLES_OUT}")

    summary = {
        "source_footprint_parts": str(FOOTPRINT_PARTS),
        "metric_crs_epsg": METRIC_CRS_EPSG,
        "footprint_parts": int(len(parts)),
        "moving_footprint_parts": moving_parts,
        "blank_footprint_parts": blank_parts,
        "mesh_vertices": int(len(vertices)),
        "mesh_triangles": int(len(triangles)),
        "moving_vertices": moving_vertices,
        "blank_vertices": blank_vertices,
        "moving_triangles": moving_triangles,
        "blank_triangles": blank_triangles,
        "zero_area_triangles": zero_area_triangles,
        "area_check": {
            "bad_area_parts_rel_gt_1e_minus_6": bad_area_parts,
            "max_abs_area_diff_m2": max_abs_area_diff,
            "max_rel_area_diff": max_rel_area_diff,
        },
        "outputs": {
            "vertices": str(MESH_VERTICES_OUT),
            "triangles": str(MESH_TRIANGLES_OUT),
            "summary": str(MESH_SUMMARY_OUT),
        },
        "notes": {
            "mesh_stage": "2D cap triangulation only; no height/extrusion yet",
            "z_rule_later": "z/height will be assigned downstream from irreversible/reversible/total displacement arrays",
            "template_status": "generic parcel cap mesh product, not source-specific",
        },
    }

    MESH_SUMMARY_OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    ok(f"wrote {MESH_SUMMARY_OUT}")

    report = {
        "project_root": str(PROJECT_ROOT),
        "summary": summary,
    }

    REPORT_JSON_OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    txt_lines = [
        "PROTO2 PHASE 04 TRIANGULATION REPORT",
        "",
        f"source footprint parts: {FOOTPRINT_PARTS}",
        f"metric CRS EPSG: {METRIC_CRS_EPSG}",
        "",
        f"footprint parts: {len(parts):,}",
        f"moving footprint parts: {moving_parts:,}",
        f"blank footprint parts: {blank_parts:,}",
        "",
        f"mesh vertices: {len(vertices):,}",
        f"mesh triangles: {len(triangles):,}",
        f"moving triangles: {moving_triangles:,}",
        f"blank triangles: {blank_triangles:,}",
        "",
        "area check:",
        f"bad_area_parts_rel_gt_1e-6: {bad_area_parts:,}",
        f"max_abs_area_diff_m2: {max_abs_area_diff:.9f}",
        f"max_rel_area_diff: {max_rel_area_diff:.9g}",
        "",
        "outputs:",
        f"- {MESH_VERTICES_OUT}",
        f"- {MESH_TRIANGLES_OUT}",
        f"- {MESH_SUMMARY_OUT}",
    ]

    REPORT_TXT_OUT.write_text("\n".join(txt_lines), encoding="utf-8")
    ok(f"wrote {REPORT_JSON_OUT}")
    ok(f"wrote {REPORT_TXT_OUT}")

    print("\n=== SUMMARY ===")
    print(f"Mesh vertices: {len(vertices):,}")
    print(f"Mesh triangles: {len(triangles):,}")
    print(f"Moving triangles: {moving_triangles:,}")
    print(f"Blank triangles: {blank_triangles:,}")
    print(f"Max area relative diff: {max_rel_area_diff:.9g}")
    print("\nPHASE 04 RESULT: PASS. Parcel cap mesh triangulated.")


if __name__ == "__main__":
    main()