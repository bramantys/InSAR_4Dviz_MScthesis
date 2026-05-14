"""
PHASE 3 — glTF Geometry Builder
Converts RUM footprints (WGS84 corners) to ECEF Cartesian3 positions,
builds a flat quad cap per RUM, and writes a sample GLB file for
visual validation in https://gltf-viewer.donmccurdy.com

Save to: 4.1.3.V3_4D/pipeline/phase3_geometry.py
Run as:  python pipeline/phase3_geometry.py

Output:  Data/sample_tile.glb   (first 200 RUMs — open in gltf-viewer)
"""

import json
import math
import os
import struct
import time

import numpy as np

# ============================================================
# PATHS  (Codey-approved absolute path pattern)
# ============================================================
_BASE             = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FOOTPRINTS_PATH   = os.path.join(_BASE, "Data", "jakarta_rum_footprints.json")
OUTPUT_GLB        = os.path.join(_BASE, "Data", "sample_tile.glb")
OUTPUT_META       = os.path.join(_BASE, "Data", "sample_tile_meta.json")

# ============================================================
# CONFIG
# ============================================================
SAMPLE_COUNT    = 200      # RUMs to include in the sample tile
DATUM_HEIGHT_M  = 0.0      # height above WGS84 ellipsoid (ground level for now)
                            # visual height exaggeration added in Phase 7-8

# ============================================================
# HELPERS
# ============================================================

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

def ok(msg):   print(f"  [OK]   {msg}")
def warn(msg): print(f"  [WARN] {msg}")
def fail(msg): print(f"  [FAIL] {msg}")


def wgs84_to_ecef(lon_deg, lat_deg, h_m=0.0):
    """
    Convert WGS84 geodetic coordinates to ECEF Cartesian3.
    Returns (X, Y, Z) in metres.
    """
    a   = 6_378_137.0           # semi-major axis (m)
    f   = 1.0 / 298.257223563   # flattening
    e2  = 2*f - f*f             # first eccentricity squared

    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)

    N = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)

    x = (N + h_m) * cos_lat * math.cos(lon)
    y = (N + h_m) * cos_lat * math.sin(lon)
    z = (N * (1.0 - e2) + h_m) * sin_lat
    return (x, y, z)


def build_glb(positions_f32, batchids_f32, indices_u32, rtc_center):
    """
    Pack geometry arrays into a valid GLB (binary glTF 2.0) file.

    positions_f32 : np.ndarray shape (N, 3) float32  — relative to rtc_center
    batchids_f32  : np.ndarray shape (N,)   float32  — per-vertex batch id
    indices_u32   : np.ndarray shape (M,)   uint32   — triangle indices
    rtc_center    : [X, Y, Z] ECEF in metres (for b3dm feature table later)

    Returns bytes of the GLB file.
    """
    n_verts   = len(positions_f32)
    n_indices = len(indices_u32)

    # --- Binary buffer layout ---
    pos_bytes   = positions_f32.tobytes()       # VEC3 float32
    bid_bytes   = batchids_f32.tobytes()        # SCALAR float32
    idx_bytes   = indices_u32.tobytes()         # SCALAR uint32

    # Pad each section to 4-byte boundary
    def pad4(b):
        r = len(b) % 4
        return b + b'\x00' * ((4 - r) % 4)

    pos_bytes = pad4(pos_bytes)
    bid_bytes = pad4(bid_bytes)
    idx_bytes = pad4(idx_bytes)

    bin_buffer = pos_bytes + bid_bytes + idx_bytes
    total_bin  = len(bin_buffer)

    pos_offset = 0
    bid_offset = len(pos_bytes)
    idx_offset = bid_offset + len(bid_bytes)

    # --- Accessor min/max for POSITION (required by spec) ---
    pos_min = positions_f32.min(axis=0).tolist()
    pos_max = positions_f32.max(axis=0).tolist()

    # --- glTF JSON ---
    gltf = {
        "asset": {
            "version": "2.0",
            "generator": "Jakarta 3D Tiles Pipeline — Phase 3"
        },
        "extensionsUsed": ["CESIUM_RTC"],
        "extensions": {
            "CESIUM_RTC": {
                "center": rtc_center
            }
        },
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes":  [{"mesh": 0}],
        "meshes": [{
            "primitives": [{
                "attributes": {
                    "POSITION":  0,
                    "_BATCHID":  1
                },
                "indices": 2,
                "mode":    4    # TRIANGLES
            }]
        }],
        "accessors": [
            {
                # POSITION — VEC3 FLOAT
                "bufferView":     0,
                "byteOffset":     0,
                "componentType":  5126,    # FLOAT
                "count":          n_verts,
                "type":           "VEC3",
                "min":            [round(v, 4) for v in pos_min],
                "max":            [round(v, 4) for v in pos_max]
            },
            {
                # _BATCHID — SCALAR FLOAT
                "bufferView":     1,
                "byteOffset":     0,
                "componentType":  5126,    # FLOAT
                "count":          n_verts,
                "type":           "SCALAR"
            },
            {
                # Indices — SCALAR UNSIGNED_INT
                "bufferView":     2,
                "byteOffset":     0,
                "componentType":  5125,    # UNSIGNED_INT
                "count":          n_indices,
                "type":           "SCALAR"
            }
        ],
        "bufferViews": [
            {   # positions
                "buffer":     0,
                "byteOffset": pos_offset,
                "byteLength": len(pos_bytes),
                "target":     34962   # ARRAY_BUFFER
            },
            {   # batchids
                "buffer":     0,
                "byteOffset": bid_offset,
                "byteLength": len(bid_bytes),
                "target":     34962
            },
            {   # indices
                "buffer":     0,
                "byteOffset": idx_offset,
                "byteLength": len(idx_bytes),
                "target":     34963   # ELEMENT_ARRAY_BUFFER
            }
        ],
        "buffers": [{"byteLength": total_bin}]
    }

    json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")

    # Pad JSON to 4-byte boundary with spaces (spec requirement)
    r = len(json_bytes) % 4
    if r:
        json_bytes += b' ' * (4 - r)

    # --- GLB header + chunks ---
    MAGIC   = b'glTF'
    VERSION = 2
    JSON_CHUNK_TYPE = 0x4E4F534A   # JSON
    BIN_CHUNK_TYPE  = 0x004E4942   # BIN\0

    json_chunk = struct.pack("<II", len(json_bytes), JSON_CHUNK_TYPE) + json_bytes
    bin_chunk  = struct.pack("<II", total_bin, BIN_CHUNK_TYPE) + bin_buffer

    total_length = 12 + len(json_chunk) + len(bin_chunk)
    header = MAGIC + struct.pack("<II", VERSION, total_length)

    return header + json_chunk + bin_chunk


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Load footprints
    # --------------------------------------------------------
    section("Loading footprints")

    print(f"  {FOOTPRINTS_PATH}")
    with open(FOOTPRINTS_PATH, encoding="utf-8") as f:
        fp_data = json.load(f)

    all_footprints = fp_data["footprints"]
    rum_ids = sorted(all_footprints.keys())[:SAMPLE_COUNT]

    ok(f"Using first {len(rum_ids)} RUMs of {len(all_footprints)} total")

    # --------------------------------------------------------
    # Convert corners to ECEF
    # --------------------------------------------------------
    section(f"Converting corners to ECEF  (height = {DATUM_HEIGHT_M} m)")

    t0 = time.time()

    all_ecef = []   # list of 4 ECEF tuples per RUM (NE, NW, SW, SE)

    for rum_id in rum_ids:
        fp      = all_footprints[rum_id]
        corners = fp["corners"]   # [[lon,lat], [lon,lat], [lon,lat], [lon,lat]]
        ecef_corners = [
            wgs84_to_ecef(c[0], c[1], DATUM_HEIGHT_M)
            for c in corners
        ]
        all_ecef.append(ecef_corners)

    elapsed = time.time() - t0
    ok(f"Converted {len(all_ecef) * 4} ECEF vertices in {elapsed:.3f}s")

    # --------------------------------------------------------
    # Compute RTC_CENTER (centroid of all vertices)
    # --------------------------------------------------------
    section("Computing RTC_CENTER")

    flat_ecef = [pt for rum in all_ecef for pt in rum]
    cx = sum(p[0] for p in flat_ecef) / len(flat_ecef)
    cy = sum(p[1] for p in flat_ecef) / len(flat_ecef)
    cz = sum(p[2] for p in flat_ecef) / len(flat_ecef)
    rtc_center = [cx, cy, cz]

    print(f"  RTC_CENTER ECEF: X={cx:.2f}, Y={cy:.2f}, Z={cz:.2f}")

    # Convert center back to lon/lat for sanity check
    # (just X-Y plane angle gives approximate lon)
    approx_lon = math.degrees(math.atan2(cy, cx))
    approx_lat = math.degrees(math.asin(cz / math.sqrt(cx*cx + cy*cy + cz*cz)))
    print(f"  ≈ lon={approx_lon:.4f}, lat={approx_lat:.4f}  (should be inside Jakarta)")

    # --------------------------------------------------------
    # Build vertex arrays
    # --------------------------------------------------------
    section("Building vertex arrays")

    positions_list = []
    batchids_list  = []
    indices_list   = []

    for batch_id, ecef_corners in enumerate(all_ecef):
        base_idx = batch_id * 4   # 4 vertices per RUM

        # Positions relative to RTC_CENTER
        for (ex, ey, ez) in ecef_corners:
            positions_list.append([ex - cx, ey - cy, ez - cz])

        # Batch ID for all 4 vertices of this RUM
        batchids_list.extend([float(batch_id)] * 4)

        # 2 triangles (CCW winding):
        #   NE(0), NW(1), SW(2) and NE(0), SW(2), SE(3)
        i = base_idx
        indices_list.extend([i+0, i+1, i+2,   i+0, i+2, i+3])

    positions_f32 = np.array(positions_list, dtype=np.float32)
    batchids_f32  = np.array(batchids_list,  dtype=np.float32)
    indices_u32   = np.array(indices_list,   dtype=np.uint32)

    ok(f"Vertices : {len(positions_f32)}")
    ok(f"Triangles: {len(indices_u32) // 3}")
    ok(f"Position buffer: {positions_f32.nbytes / 1024:.1f} KB")

    # --------------------------------------------------------
    # Sanity: check relative position magnitudes
    # --------------------------------------------------------
    section("Position magnitude check")

    dists = np.linalg.norm(positions_f32, axis=1)
    print(f"  Max offset from RTC_CENTER: {dists.max():.2f} m")
    print(f"  Mean offset               : {dists.mean():.2f} m")

    # For a ~40km × 30km area, half-diagonal ≈ 25km → max offset should be <30000m
    if dists.max() < 30_000:
        ok("Offsets within expected range (<30 km)")
    else:
        warn(f"Large offsets detected — check coordinate conversion")

    # --------------------------------------------------------
    # Build and write GLB
    # --------------------------------------------------------
    section("Building GLB")

    t0 = time.time()
    glb_bytes = build_glb(positions_f32, batchids_f32, indices_u32, rtc_center)
    elapsed   = time.time() - t0
    ok(f"GLB built in {elapsed:.3f}s  ({len(glb_bytes)/1024:.1f} KB)")

    with open(OUTPUT_GLB, "wb") as f:
        f.write(glb_bytes)
    ok(f"Written: {OUTPUT_GLB}")

    # --------------------------------------------------------
    # Write metadata (RTC_CENTER + rum_ids in this tile)
    # for use in Phase 5 when building the b3dm)
    # --------------------------------------------------------
    meta = {
        "rum_ids":    rum_ids,
        "rtc_center": rtc_center,
        "rum_count":  len(rum_ids),
        "note":       "Sample tile only — full tiling done in Phase 4+5"
    }
    with open(OUTPUT_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    ok(f"Metadata: {OUTPUT_META}")

    # --------------------------------------------------------
    # Summary + next step instructions
    # --------------------------------------------------------
    section("SUMMARY")
    ok(f"GLB file:  {OUTPUT_GLB}")
    print()
    print("  *** VALIDATION STEP ***")
    print("  1. Open https://gltf-viewer.donmccurdy.com in your browser")
    print("  2. Drag-and-drop  Data/sample_tile.glb  onto the viewer")
    print("  3. You should see ~200 flat rectangles arranged in a grid")
    print("     (they will look tiny — zoom in with scroll wheel)")
    print("  4. If you see flat quads with no gaps between them = PASS")
    print("  5. If nothing appears, or viewer errors = FAIL, paste error here")
    print()
    ok("Ready for Phase 4 (spatial tiler) after visual validation.")
    print()


if __name__ == "__main__":
    main()
