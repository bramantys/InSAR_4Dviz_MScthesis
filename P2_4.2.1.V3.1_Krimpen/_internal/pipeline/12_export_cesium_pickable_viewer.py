from pathlib import Path
import json
import math
import shutil
import sys

import geopandas as gpd
import numpy as np
import pandas as pd


from _proto2_config import load_project_config, output_cesium_dir, output_data_dir, stage_records_dir


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = load_project_config(PROJECT_ROOT)

OUTPUT_DATA = output_data_dir(PROJECT_ROOT, CONFIG)
OUTPUT_CESIUM = output_cesium_dir(PROJECT_ROOT, CONFIG)
RUN_RECORDS = stage_records_dir(PROJECT_ROOT, CONFIG)

FOOTPRINT_PARTS = OUTPUT_DATA / "parcel_footprints_parts.parquet"
PARCEL_RENDER_INDEX = OUTPUT_DATA / "parcel_render_index.parquet"

SOURCE_GLB = OUTPUT_CESIUM / "proto2_animated_parcel_mesh.glb"
SOURCE_PHASE09_SUMMARY = OUTPUT_CESIUM / "proto2_cesium_animated_glb_summary.json"

SOURCE_REVERSIBLE = OUTPUT_DATA / "parcel_displacement_reversible_f32.bin"
SOURCE_IRREVERSIBLE = OUTPUT_DATA / "parcel_displacement_irreversible_f32.bin"
SOURCE_TOTAL = OUTPUT_DATA / "parcel_displacement_total_f32.bin"

ASSET_DIR = OUTPUT_CESIUM / "phase12_assets"

GLB_OUT = ASSET_DIR / "proto2_animated_parcel_mesh.glb"
REVERSIBLE_OUT = ASSET_DIR / "parcel_displacement_reversible_f32.bin"
IRREVERSIBLE_OUT = ASSET_DIR / "parcel_displacement_irreversible_f32.bin"
TOTAL_OUT = ASSET_DIR / "parcel_displacement_total_f32.bin"
PICK_INDEX_OUT = ASSET_DIR / "parcel_pick_index.json"
MANIFEST_OUT = ASSET_DIR / "phase12_pickable_assets_manifest.json"

HTML_OUT = OUTPUT_CESIUM / "proto2_cesium_phase12_pickable_viewer.html"
SUMMARY_OUT = OUTPUT_CESIUM / "proto2_cesium_phase12_pickable_summary.json"

REPORT_TXT_OUT = RUN_RECORDS / "phase12_cesium_pickable_report.txt"
REPORT_JSON_OUT = RUN_RECORDS / "phase12_cesium_pickable_report.json"

CESIUM_JS_URL = "https://cesium.com/downloads/cesiumjs/releases/1.123/Build/Cesium/Cesium.js"
CESIUM_CSS_URL = "https://cesium.com/downloads/cesiumjs/releases/1.123/Build/Cesium/Widgets/widgets.css"


def fail(message):
    print(f"[FAIL] {message}")
    sys.exit(1)


def ok(message):
    print(f"[OK] {message}")


def warn(message):
    print(f"[WARN] {message}")


def require_files(paths):
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        fail(f"Missing required files: {missing}")


def file_size_mb(path):
    return path.stat().st_size / (1024 * 1024)


def copy_asset(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    ok(f"copied {src.name} -> {dst}")


def wgs84_to_ecef(lon_deg, lat_deg, h_m):
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = f * (2.0 - f)

    lon = np.deg2rad(lon_deg)
    lat = np.deg2rad(lat_deg)

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    cos_lon = np.cos(lon)
    sin_lon = np.sin(lon)

    n = a / np.sqrt(1.0 - e2 * sin_lat * sin_lat)

    x = (n + h_m) * cos_lat * cos_lon
    y = (n + h_m) * cos_lat * sin_lon
    z = (n * (1.0 - e2) + h_m) * sin_lat

    return x, y, z


def ecef_to_local_enu(x, y, z, center_lon_deg, center_lat_deg, center_h_m):
    cx, cy, cz = wgs84_to_ecef(
        np.array([center_lon_deg], dtype=np.float64),
        np.array([center_lat_deg], dtype=np.float64),
        np.array([center_h_m], dtype=np.float64),
    )

    cx = float(cx[0])
    cy = float(cy[0])
    cz = float(cz[0])

    lon0 = math.radians(center_lon_deg)
    lat0 = math.radians(center_lat_deg)

    sin_lon = math.sin(lon0)
    cos_lon = math.cos(lon0)
    sin_lat = math.sin(lat0)
    cos_lat = math.cos(lat0)

    east = np.array([-sin_lon, cos_lon, 0.0], dtype=np.float64)
    north = np.array([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat], dtype=np.float64)

    dx = x - cx
    dy = y - cy
    dz = z - cz

    local_x = dx * east[0] + dy * east[1] + dz * east[2]
    local_y = dx * north[0] + dy * north[1] + dz * north[2]
    local_z = np.zeros_like(local_x)

    return local_x, local_y, local_z


def ring_lonlat_to_pick_local(ring_coords, center_lon, center_lat, center_h):
    coords = np.asarray(ring_coords, dtype=np.float64)

    lon = coords[:, 0]
    lat = coords[:, 1]

    x_ecef, y_ecef, z_ecef = wgs84_to_ecef(
        lon,
        lat,
        np.full_like(lon, center_h),
    )

    local_x, local_y, _ = ecef_to_local_enu(
        x_ecef,
        y_ecef,
        z_ecef,
        center_lon,
        center_lat,
        center_h,
    )

    # IMPORTANT:
    # Do NOT apply the Phase 09/10 GLB horizontal correction here.
    #
    # The GLB vertices need that correction because they pass through the
    # glTF node matrix / Cesium model-axis pipeline.
    #
    # This CPU pick index and the yellow Cesium polyline outline do NOT pass
    # through the GLB internal node transform. They live directly in the
    # Cesium root local ENU frame from modelMatrix / inverseModelMatrix.
    #
    # Therefore the pick index must use plain local ENU XY.
    return np.column_stack([local_x, local_y])



def round_ring_xy(ring_xy, decimals=3):
    return [
        [round(float(x), decimals), round(float(y), decimals)]
        for x, y in ring_xy
    ]


def build_pick_index(parts, render_index, phase09_summary):
    center_lon = float(phase09_summary["center_lon"])
    center_lat = float(phase09_summary["center_lat"])
    center_h = float(phase09_summary["center_height_m"])

    if "parcel_id" not in parts.columns:
        fail("footprint parts missing parcel_id")

    if "geometry" not in parts.columns:
        fail("footprint parts missing geometry")

    if "parcel_id" not in render_index.columns:
        fail("parcel_render_index missing parcel_id")

    if "displacement_row_index" not in render_index.columns:
        fail("parcel_render_index missing displacement_row_index")

    row_map = dict(
        zip(
            render_index["parcel_id"].astype(str),
            render_index["displacement_row_index"].astype(int),
        )
    )

    features = []

    for _, row in parts.iterrows():
        geom = row.geometry

        if geom is None or geom.is_empty:
            continue

        if geom.geom_type != "Polygon":
            warn(f"Skipping non-Polygon footprint part geometry: {geom.geom_type}")
            continue

        parcel_id_raw = row["parcel_id"]
        parcel_id_key = str(parcel_id_raw)

        displacement_row_index = int(row_map.get(parcel_id_key, -1))
        has_displacement = displacement_row_index >= 0

        exterior = ring_lonlat_to_pick_local(
            geom.exterior.coords,
            center_lon,
            center_lat,
            center_h,
        )

        rings = [round_ring_xy(exterior)]

        for interior in geom.interiors:
            hole = ring_lonlat_to_pick_local(
                interior.coords,
                center_lon,
                center_lat,
                center_h,
            )
            rings.append(round_ring_xy(hole))

        all_xy = np.vstack([np.asarray(r, dtype=np.float64) for r in rings if len(r) > 0])
        bbox = [
            round(float(all_xy[:, 0].min()), 3),
            round(float(all_xy[:, 1].min()), 3),
            round(float(all_xy[:, 0].max()), 3),
            round(float(all_xy[:, 1].max()), 3),
        ]

        feature = {
            "parcel_id": int(parcel_id_raw) if str(parcel_id_raw).isdigit() else str(parcel_id_raw),
            "footprint_id": str(row["footprint_id"]) if "footprint_id" in row else parcel_id_key,
            "part_index": int(row["part_index"]) if "part_index" in row and pd.notna(row["part_index"]) else 0,
            "parcel_status": str(row["parcel_status"]) if "parcel_status" in row and pd.notna(row["parcel_status"]) else ("moving" if has_displacement else "blank"),
            "has_displacement": bool(has_displacement),
            "displacement_row_index": displacement_row_index,
            "area_m2": float(row["area_m2"]) if "area_m2" in row and pd.notna(row["area_m2"]) else None,
            "bbox": bbox,
            "rings": rings,
        }

        features.append(feature)

    metadata = {
        "product": "proto2_phase12_parcel_pick_index",
        "coordinate_frame": "Cesium root local ENU XY after inverse modelMatrix; no GLB horizontal correction",
        "ring_rule": "rings[0] exterior; rings[1:] holes",
        "feature_count": len(features),
        "center_lon": center_lon,
        "center_lat": center_lat,
        "center_height_m": center_h,
        "notes": {
            "not_b3dm": True,
            "picking_method": "CPU bbox + point-in-polygon after Cesium screen click is transformed to local model XY",
            "selection_highlight": "one temporary polyline outline only",
        },
    }

    return {
        "metadata": metadata,
        "features": features,
    }


def make_html(summary):
    meta_json = json.dumps(summary)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Proto2 Phase 12 Pickable Cesium Viewer</title>
  <script src="{CESIUM_JS_URL}"></script>
  <link href="{CESIUM_CSS_URL}" rel="stylesheet"/>
  <style>
    html, body, #cesiumContainer {{
      width: 100%;
      height: 100%;
      margin: 0;
      padding: 0;
      overflow: hidden;
      background: #111318;
      font-family: Arial, sans-serif;
    }}
    #panel {{
      position: absolute;
      left: 14px;
      top: 14px;
      z-index: 10;
      background: rgba(0,0,0,0.82);
      color: #f1f1f1;
      border: 1px solid rgba(255,255,255,0.18);
      border-radius: 12px;
      padding: 12px 14px;
      min-width: 470px;
      line-height: 1.45;
      font-size: 13px;
      backdrop-filter: blur(8px);
    }}
    #pickBox {{
      position: absolute;
      right: 14px;
      top: 14px;
      z-index: 10;
      background: rgba(0,0,0,0.82);
      color: #f1f1f1;
      border: 1px solid rgba(255,255,255,0.18);
      border-radius: 12px;
      padding: 12px 14px;
      min-width: 310px;
      max-width: 390px;
      line-height: 1.45;
      font-size: 13px;
      backdrop-filter: blur(8px);
    }}
    #panel h1, #pickBox h2 {{
      margin: 0 0 6px 0;
      font-size: 17px;
    }}
    .row {{
      display: flex;
      justify-content: space-between;
      gap: 24px;
    }}
    .label {{
      color: #bbbbbb;
    }}
    .value {{
      color: #ffffff;
      font-variant-numeric: tabular-nums;
    }}
    .controls {{
      display: grid;
      grid-template-columns: 125px 1fr;
      gap: 8px 10px;
      margin-top: 10px;
      align-items: center;
    }}
    .buttonrow {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-top: 10px;
    }}
    button, select, input {{
      background: #222936;
      color: #ffffff;
      border: 1px solid #4b5568;
      border-radius: 8px;
      padding: 6px 8px;
    }}
    input[type="range"] {{
      padding: 0;
    }}
    button:hover {{
      background: #30394a;
      cursor: pointer;
    }}
    #status {{
      margin-top: 8px;
      color: #ffcc66;
    }}
    #legend {{
      margin-top: 8px;
      color: #d0d0d0;
    }}
    .red {{ color: #ff6b6b; }}
    .green {{ color: #6bff95; }}
    .white {{ color: #ffffff; }}
    .muted {{ color: #aaaaaa; }}
  </style>
</head>
<body>
  <div id="cesiumContainer"></div>

  <div id="panel">
    <h1>Proto2 Phase 12 pickable Cesium viewer</h1>

    <div class="row"><span class="label">Vertices</span><span class="value">{summary["vertices"]:,}</span></div>
    <div class="row"><span class="label">Triangles</span><span class="value">{summary["triangles"]:,}</span></div>
    <div class="row"><span class="label">Moving parcels</span><span class="value">{summary["moving_parcels"]:,}</span></div>
    <div class="row"><span class="label">Pick features</span><span class="value">{summary["phase12"]["pick_features"]:,}</span></div>
    <div class="row"><span class="label">Epoch</span><span class="value" id="epochText">--</span></div>
    <div class="row"><span class="label">FPS-ish</span><span class="value" id="fpsText">--</span></div>
    <div class="row"><span class="label">Loaded models</span><span class="value" id="modelCountText">1</span></div>

    <div class="controls">
      <button id="playBtn">Pause</button>
      <button id="homeBtn">Oblique home</button>

      <label>Mode</label>
      <select id="modeSelect">
        <option value="total" selected>total only</option>
        <option value="reversible">reversible only</option>
        <option value="irreversible">irreversible only</option>
        <option value="stacked">stacked: total + irreversible floor</option>
      </select>

      <label>Epoch</label>
      <input id="epochSlider" type="range" min="0" max="{summary["epochs"] - 1}" value="0"/>

      <label>Height scale</label>
      <input id="heightScale" type="range" min="0" max="0.8" step="0.01" value="0.16"/>

      <label>Step ms</label>
      <input id="stepMs" type="range" min="30" max="300" step="10" value="80"/>
    </div>

    <div class="buttonrow">
      <button id="northBtn">Top-down north-up</button>
      <button id="axisBtn">Toggle ENU axes</button>
    </div>

    <div id="legend">
      Click a parcel to inspect it. Axes:
      <span class="red">red = EAST</span>,
      <span class="green">green = NORTH</span>,
      <span class="white">white = UP</span>.
    </div>

    <div id="status">Loading external assets...</div>
  </div>

  <div id="pickBox">
    <h2>Parcel pick</h2>
    <div id="pickContent" class="muted">Click a parcel.</div>
  </div>

  <script>
    const META = {meta_json};

    const ASSET_BASE = "phase12_assets/";
    const GLB_URL = ASSET_BASE + "proto2_animated_parcel_mesh.glb";
    const REVERSIBLE_URL = ASSET_BASE + "parcel_displacement_reversible_f32.bin";
    const IRREVERSIBLE_URL = ASSET_BASE + "parcel_displacement_irreversible_f32.bin";
    const TOTAL_URL = ASSET_BASE + "parcel_displacement_total_f32.bin";
    const PICK_INDEX_URL = ASSET_BASE + "parcel_pick_index.json";

    let reversibleArr = null;
    let irreversibleArr = null;
    let totalArr = null;
    let pickIndex = null;

    let selectedFeature = null;
    let selectedOutline = null;

    function htmlEscape(value) {{
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
    }}

    async function fetchFloat32(url, expectedCount) {{
      const res = await fetch(url);
      if (!res.ok) {{
        throw new Error("Failed to fetch " + url + ": " + res.status + " " + res.statusText);
      }}

      const buffer = await res.arrayBuffer();

      if (buffer.byteLength !== expectedCount * 4) {{
        throw new Error(
          url + " byte length mismatch: got " + buffer.byteLength +
          ", expected " + (expectedCount * 4)
        );
      }}

      return new Float32Array(buffer);
    }}

    async function fetchJson(url) {{
      const res = await fetch(url);
      if (!res.ok) {{
        throw new Error("Failed to fetch " + url + ": " + res.status + " " + res.statusText);
      }}
      return await res.json();
    }}

    function makeTextureUniform(arr) {{
      try {{
        return new Cesium.TextureUniform({{
          typedArray: arr,
          width: META.epochs,
          height: META.moving_parcels,
          pixelFormat: Cesium.PixelFormat.RED,
          pixelDatatype: Cesium.PixelDatatype.FLOAT
        }});
      }} catch (e1) {{
        console.warn("TextureUniform typedArray path failed, trying source.arrayBufferView path:", e1);

        return new Cesium.TextureUniform({{
          source: {{
            arrayBufferView: arr,
            width: META.epochs,
            height: META.moving_parcels
          }},
          pixelFormat: Cesium.PixelFormat.RED,
          pixelDatatype: Cesium.PixelDatatype.FLOAT
        }});
      }}
    }}

    function makeCustomShader(componentValue, textures) {{
      return new Cesium.CustomShader({{
        mode: Cesium.CustomShaderMode.MODIFY_MATERIAL,
        uniforms: {{
          u_epoch: {{
            type: Cesium.UniformType.FLOAT,
            value: 0.0
          }},
          u_epochs: {{
            type: Cesium.UniformType.FLOAT,
            value: META.epochs
          }},
          u_rows: {{
            type: Cesium.UniformType.FLOAT,
            value: META.moving_parcels
          }},
          u_heightScale: {{
            type: Cesium.UniformType.FLOAT,
            value: 0.16
          }},
          u_component: {{
            type: Cesium.UniformType.FLOAT,
            value: componentValue
          }},
          u_reversibleTex: {{
            type: Cesium.UniformType.SAMPLER_2D,
            value: textures.reversible
          }},
          u_irreversibleTex: {{
            type: Cesium.UniformType.SAMPLER_2D,
            value: textures.irreversible
          }},
          u_totalTex: {{
            type: Cesium.UniformType.SAMPLER_2D,
            value: textures.total
          }}
        }},
        vertexShaderText: `
          float sampleComponent(sampler2D tex, float rowIndex) {{
            float u = (u_epoch + 0.5) / u_epochs;
            float v = (rowIndex + 0.5) / u_rows;
            return texture(tex, vec2(u, v)).r;
          }}

          void vertexMain(VertexInput vsInput, inout czm_modelVertexOutput vsOutput) {{
            float rowIndex = vsInput.attributes.texCoord_0.x;
            float hasDisp = vsInput.attributes.texCoord_0.y;

            if (hasDisp > 0.5 && rowIndex >= 0.0) {{
              float disp = 0.0;

              if (u_component < 0.5) {{
                disp = sampleComponent(u_totalTex, rowIndex);
              }} else if (u_component < 1.5) {{
                disp = sampleComponent(u_reversibleTex, rowIndex);
              }} else {{
                disp = sampleComponent(u_irreversibleTex, rowIndex);
              }}

              vsOutput.positionMC.z += disp * u_heightScale;
            }}
          }}
        `
      }});
    }}

    async function loadModel(viewer, modelMatrix, customShader) {{
      if (Cesium.Model.fromGltfAsync) {{
        const model = await Cesium.Model.fromGltfAsync({{
          url: GLB_URL,
          modelMatrix: modelMatrix,
          customShader: customShader,
          allowPicking: true,
          asynchronous: true
        }});
        viewer.scene.primitives.add(model);
        return model;
      }}

      const model = Cesium.Model.fromGltf({{
        url: GLB_URL,
        modelMatrix: modelMatrix,
        customShader: customShader,
        allowPicking: true,
        asynchronous: true
      }});

      viewer.scene.primitives.add(model);
      return model;
    }}

    function localPoint(modelMatrix, x, y, z) {{
      return Cesium.Matrix4.multiplyByPoint(
        modelMatrix,
        new Cesium.Cartesian3(x, y, z),
        new Cesium.Cartesian3()
      );
    }}

    function addAxes(viewer, modelMatrix) {{
      const axisLength = Math.max(
        META.local_span_m.east_west,
        META.local_span_m.north_south
      ) * 0.20;

      const zLift = 120.0;

      const origin = localPoint(modelMatrix, 0, 0, zLift);
      const east = localPoint(modelMatrix, axisLength, 0, zLift);
      const north = localPoint(modelMatrix, 0, axisLength, zLift);
      const up = localPoint(modelMatrix, 0, 0, zLift + axisLength * 0.25);

      const lines = viewer.scene.primitives.add(new Cesium.PolylineCollection());

      lines.add({{
        positions: [origin, east],
        width: 7,
        material: Cesium.Material.fromType("Color", {{
          color: Cesium.Color.RED.withAlpha(0.95)
        }})
      }});

      lines.add({{
        positions: [origin, north],
        width: 7,
        material: Cesium.Material.fromType("Color", {{
          color: Cesium.Color.LIME.withAlpha(0.95)
        }})
      }});

      lines.add({{
        positions: [origin, up],
        width: 7,
        material: Cesium.Material.fromType("Color", {{
          color: Cesium.Color.WHITE.withAlpha(0.95)
        }})
      }});

      const labels = viewer.scene.primitives.add(new Cesium.LabelCollection());

      labels.add({{
        position: east,
        text: "EAST +X",
        font: "16px sans-serif",
        fillColor: Cesium.Color.RED,
        outlineColor: Cesium.Color.BLACK,
        outlineWidth: 3,
        style: Cesium.LabelStyle.FILL_AND_OUTLINE,
        pixelOffset: new Cesium.Cartesian2(10, 0)
      }});

      labels.add({{
        position: north,
        text: "NORTH +Y",
        font: "16px sans-serif",
        fillColor: Cesium.Color.LIME,
        outlineColor: Cesium.Color.BLACK,
        outlineWidth: 3,
        style: Cesium.LabelStyle.FILL_AND_OUTLINE,
        pixelOffset: new Cesium.Cartesian2(10, 0)
      }});

      labels.add({{
        position: up,
        text: "UP +Z",
        font: "16px sans-serif",
        fillColor: Cesium.Color.WHITE,
        outlineColor: Cesium.Color.BLACK,
        outlineWidth: 3,
        style: Cesium.LabelStyle.FILL_AND_OUTLINE,
        pixelOffset: new Cesium.Cartesian2(10, 0)
      }});

      return {{ lines, labels }};
    }}

    function pointInRing(x, y, ring) {{
      let inside = false;

      for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {{
        const xi = ring[i][0];
        const yi = ring[i][1];
        const xj = ring[j][0];
        const yj = ring[j][1];

        const intersect = ((yi > y) !== (yj > y)) &&
          (x < (xj - xi) * (y - yi) / ((yj - yi) || 1e-30) + xi);

        if (intersect) inside = !inside;
      }}

      return inside;
    }}

    function pointInFeature(x, y, feature) {{
      const bbox = feature.bbox;

      if (x < bbox[0] || y < bbox[1] || x > bbox[2] || y > bbox[3]) {{
        return false;
      }}

      const rings = feature.rings;

      if (!rings || rings.length === 0) {{
        return false;
      }}

      if (!pointInRing(x, y, rings[0])) {{
        return false;
      }}

      for (let i = 1; i < rings.length; i++) {{
        if (pointInRing(x, y, rings[i])) {{
          return false;
        }}
      }}

      return true;
    }}

    function findFeatureAtLocalXY(x, y) {{
      const features = pickIndex.features;

      for (let i = 0; i < features.length; i++) {{
        const f = features[i];
        if (pointInFeature(x, y, f)) {{
          return f;
        }}
      }}

      return null;
    }}

    function displacementAt(feature, epoch, arr) {{
      const row = feature.displacement_row_index;

      if (row < 0) {{
        return null;
      }}

      return arr[row * META.epochs + epoch];
    }}

    function formatMm(value) {{
      if (value === null || value === undefined || Number.isNaN(value)) {{
        return "unavailable";
      }}
      return value.toFixed(3);
    }}

    function renderPickInfo(epoch) {{
      const el = document.getElementById("pickContent");

      if (!selectedFeature) {{
        el.innerHTML = '<span class="muted">Click a parcel.</span>';
        return;
      }}

      const f = selectedFeature;

      const rv = displacementAt(f, epoch, reversibleArr);
      const ir = displacementAt(f, epoch, irreversibleArr);
      const tt = displacementAt(f, epoch, totalArr);

      el.innerHTML = `
        <div class="row"><span class="label">parcel_id</span><span class="value">${{htmlEscape(f.parcel_id)}}</span></div>
        <div class="row"><span class="label">status</span><span class="value">${{htmlEscape(f.parcel_status)}}</span></div>
        <div class="row"><span class="label">footprint</span><span class="value">${{htmlEscape(f.footprint_id)}}</span></div>
        <div class="row"><span class="label">row index</span><span class="value">${{f.displacement_row_index}}</span></div>
        <div class="row"><span class="label">epoch</span><span class="value">${{META.epoch_labels[epoch]}}</span></div>
        <hr style="border-color: rgba(255,255,255,0.16);">
        <div class="row"><span class="label">reversible</span><span class="value">${{formatMm(rv)}} mm</span></div>
        <div class="row"><span class="label">irreversible</span><span class="value">${{formatMm(ir)}} mm</span></div>
        <div class="row"><span class="label">total</span><span class="value">${{formatMm(tt)}} mm</span></div>
      `;
    }}

    function clearSelectedOutline(viewer) {{
      if (selectedOutline) {{
        viewer.scene.primitives.remove(selectedOutline);
        selectedOutline = null;
      }}
    }}

    function drawSelectedOutline(viewer, modelMatrix, feature) {{
      clearSelectedOutline(viewer);

      selectedOutline = viewer.scene.primitives.add(new Cesium.PolylineCollection());

      const z = 2.0;

      for (const ring of feature.rings) {{
        const positions = ring.map((xy) => localPoint(modelMatrix, xy[0], xy[1], z));

        selectedOutline.add({{
          positions: positions,
          width: 4,
          material: Cesium.Material.fromType("Color", {{
            color: Cesium.Color.YELLOW.withAlpha(0.98)
          }})
        }});
      }}
    }}

    async function main() {{
      const status = document.getElementById("status");
      const expectedCount = META.moving_parcels * META.epochs;

      status.textContent = "Fetching displacement arrays and pick index...";

      [reversibleArr, irreversibleArr, totalArr, pickIndex] = await Promise.all([
        fetchFloat32(REVERSIBLE_URL, expectedCount),
        fetchFloat32(IRREVERSIBLE_URL, expectedCount),
        fetchFloat32(TOTAL_URL, expectedCount),
        fetchJson(PICK_INDEX_URL)
      ]);

      status.textContent = "Creating Cesium textures...";

      const textures = {{
        reversible: makeTextureUniform(reversibleArr),
        irreversible: makeTextureUniform(irreversibleArr),
        total: makeTextureUniform(totalArr)
      }};

      const viewer = new Cesium.Viewer("cesiumContainer", {{
        animation: false,
        timeline: false,
        geocoder: false,
        sceneModePicker: true,
        navigationHelpButton: false,
        baseLayerPicker: false,
        terrainProvider: new Cesium.EllipsoidTerrainProvider()
      }});

      viewer.imageryLayers.removeAll();
      viewer.scene.globe.baseColor = Cesium.Color.fromCssColorString("#101820");
      viewer.scene.globe.depthTestAgainstTerrain = false;
      viewer.scene.highDynamicRange = false;
      viewer.scene.requestRenderMode = false;
      viewer.scene.fxaa = true;

      const center = Cesium.Cartesian3.fromDegrees(
        META.center_lon,
        META.center_lat,
        META.center_height_m
      );

      const modelMatrix = Cesium.Transforms.eastNorthUpToFixedFrame(center);
      const inverseModelMatrix = Cesium.Matrix4.inverse(modelMatrix, new Cesium.Matrix4());

      const topShader = makeCustomShader(0.0, textures);
      const bottomShader = makeCustomShader(2.0, textures);

      status.textContent = "Loading GLB model...";

      await loadModel(viewer, modelMatrix, topShader);

      let bottomModel = null;
      let bottomLoaded = false;

      async function ensureBottomModel() {{
        if (bottomLoaded) {{
          return bottomModel;
        }}

        status.textContent = "Loading stacked floor model...";
        bottomModel = await loadModel(viewer, modelMatrix, bottomShader);
        bottomModel.show = false;
        bottomLoaded = true;
        document.getElementById("modelCountText").textContent = "2";
        status.textContent = "Stacked floor model loaded.";
        return bottomModel;
      }}

      const axes = addAxes(viewer, modelMatrix);
      let axesVisible = true;

      let epoch = 0;
      let playing = true;
      let lastStep = performance.now();

      const playBtn = document.getElementById("playBtn");
      const modeSelect = document.getElementById("modeSelect");
      const epochSlider = document.getElementById("epochSlider");
      const heightScale = document.getElementById("heightScale");
      const stepMsSlider = document.getElementById("stepMs");

      function setShaderTimeAndScale() {{
        const h = Number(heightScale.value);

        topShader.setUniform("u_epoch", epoch);
        topShader.setUniform("u_heightScale", h);

        if (bottomLoaded) {{
          bottomShader.setUniform("u_epoch", epoch);
          bottomShader.setUniform("u_heightScale", h);
        }}

        document.getElementById("epochText").textContent = META.epoch_labels[epoch];
        renderPickInfo(epoch);
      }}

      async function applyMode() {{
        const modeName = modeSelect.value;

        if (modeName === "total") {{
          topShader.setUniform("u_component", 0.0);
          if (bottomLoaded) bottomModel.show = false;
        }}

        if (modeName === "reversible") {{
          topShader.setUniform("u_component", 1.0);
          if (bottomLoaded) bottomModel.show = false;
        }}

        if (modeName === "irreversible") {{
          topShader.setUniform("u_component", 2.0);
          if (bottomLoaded) bottomModel.show = false;
        }}

        if (modeName === "stacked") {{
          topShader.setUniform("u_component", 0.0);
          bottomShader.setUniform("u_component", 2.0);
          await ensureBottomModel();
          bottomModel.show = true;
        }}

        setShaderTimeAndScale();
      }}

      playBtn.addEventListener("click", () => {{
        playing = !playing;
        playBtn.textContent = playing ? "Pause" : "Play";
      }});

      epochSlider.addEventListener("input", () => {{
        epoch = Number(epochSlider.value);
        setShaderTimeAndScale();
      }});

      heightScale.addEventListener("input", setShaderTimeAndScale);

      modeSelect.addEventListener("change", () => {{
        applyMode().catch((err) => {{
          console.error(err);
          status.textContent = "Mode error: " + err.message;
        }});
      }});

      function flyObliqueHome() {{
        viewer.camera.flyTo({{
          destination: Cesium.Cartesian3.fromDegrees(
            META.center_lon,
            META.center_lat,
            META.camera_height_m
          ),
          orientation: {{
            heading: Cesium.Math.toRadians(0.0),
            pitch: Cesium.Math.toRadians(-62.0),
            roll: 0.0
          }},
          duration: 1.0
        }});
      }}

      function flyNorthUp() {{
        viewer.camera.flyTo({{
          destination: Cesium.Cartesian3.fromDegrees(
            META.center_lon,
            META.center_lat,
            META.camera_height_m * 0.85
          ),
          orientation: {{
            heading: Cesium.Math.toRadians(0.0),
            pitch: Cesium.Math.toRadians(-90.0),
            roll: 0.0
          }},
          duration: 1.0
        }});
      }}

      document.getElementById("homeBtn").addEventListener("click", flyObliqueHome);
      document.getElementById("northBtn").addEventListener("click", flyNorthUp);

      document.getElementById("axisBtn").addEventListener("click", () => {{
        axesVisible = !axesVisible;
        axes.lines.show = axesVisible;
        axes.labels.show = axesVisible;
      }});

      const handler = new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas);

      handler.setInputAction((movement) => {{
        let cartesian = null;

        if (viewer.scene.pickPositionSupported) {{
          cartesian = viewer.scene.pickPosition(movement.position);
        }}

        if (!Cesium.defined(cartesian)) {{
          const ray = viewer.camera.getPickRay(movement.position);
          cartesian = viewer.scene.globe.pick(ray, viewer.scene);
        }}

        if (!Cesium.defined(cartesian)) {{
          selectedFeature = null;
          clearSelectedOutline(viewer);
          renderPickInfo(epoch);
          return;
        }}

        const local = Cesium.Matrix4.multiplyByPoint(
          inverseModelMatrix,
          cartesian,
          new Cesium.Cartesian3()
        );

        const hit = findFeatureAtLocalXY(local.x, local.y);

        selectedFeature = hit;

        if (hit) {{
          drawSelectedOutline(viewer, modelMatrix, hit);
        }} else {{
          clearSelectedOutline(viewer);
        }}

        renderPickInfo(epoch);
      }}, Cesium.ScreenSpaceEventType.LEFT_CLICK);

      await applyMode();
      flyObliqueHome();

      status.textContent = "Loaded. Click parcels to inspect.";

      let frameCount = 0;
      let fpsTime = performance.now();

      viewer.scene.preRender.addEventListener(function(scene, time) {{
        const now = performance.now();
        const stepMs = Number(stepMsSlider.value);

        if (playing && now - lastStep >= stepMs) {{
          epoch = (epoch + 1) % META.epochs;
          epochSlider.value = epoch;
          setShaderTimeAndScale();
          lastStep = now;
        }}

        frameCount++;
        if (now - fpsTime > 500) {{
          const fps = frameCount * 1000.0 / (now - fpsTime);
          document.getElementById("fpsText").textContent = fps.toFixed(0);
          frameCount = 0;
          fpsTime = now;
        }}
      }});

      console.log("Proto2 Phase 12 metadata:", META);
      console.log("Pick index:", pickIndex.metadata);
    }}

    main().catch((err) => {{
      console.error(err);
      document.getElementById("status").textContent = "ERROR: " + err.message;
    }});
  </script>
</body>
</html>
"""


def main():
    OUTPUT_CESIUM.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    RUN_RECORDS.mkdir(parents=True, exist_ok=True)

    print("\n=== PROTO2 PHASE 12: CESIUM PICKABLE VIEWER ===")
    print(f"Project root: {PROJECT_ROOT}")

    require_files([
        FOOTPRINT_PARTS,
        PARCEL_RENDER_INDEX,
        SOURCE_GLB,
        SOURCE_PHASE09_SUMMARY,
        SOURCE_REVERSIBLE,
        SOURCE_IRREVERSIBLE,
        SOURCE_TOTAL,
    ])

    phase09_summary = json.loads(SOURCE_PHASE09_SUMMARY.read_text(encoding="utf-8"))

    print("\nReading footprint parts...")
    parts = gpd.read_parquet(FOOTPRINT_PARTS)
    ok(f"footprint parts loaded: {len(parts):,}")

    print("Reading parcel render index...")
    render_index = pd.read_parquet(PARCEL_RENDER_INDEX)
    ok(f"parcel render index loaded: {len(render_index):,}")

    print("\nBuilding CPU parcel pick index...")
    pick_index = build_pick_index(parts, render_index, phase09_summary)

    PICK_INDEX_OUT.write_text(
        json.dumps(pick_index, separators=(",", ":")),
        encoding="utf-8",
    )
    ok(f"wrote {PICK_INDEX_OUT}")

    copy_asset(SOURCE_GLB, GLB_OUT)
    copy_asset(SOURCE_REVERSIBLE, REVERSIBLE_OUT)
    copy_asset(SOURCE_IRREVERSIBLE, IRREVERSIBLE_OUT)
    copy_asset(SOURCE_TOTAL, TOTAL_OUT)

    manifest = {
        "product": "proto2_phase12_pickable_cesium_assets",
        "version": 1,
        "asset_base": "phase12_assets/",
        "files": {
            "glb": GLB_OUT.name,
            "reversible": REVERSIBLE_OUT.name,
            "irreversible": IRREVERSIBLE_OUT.name,
            "total": TOTAL_OUT.name,
            "pick_index": PICK_INDEX_OUT.name,
        },
        "picking": {
            "method": "CPU bbox + point-in-polygon",
            "feature_count": pick_index["metadata"]["feature_count"],
            "coordinate_frame": pick_index["metadata"]["coordinate_frame"],
            "not_b3dm": True,
        },
        "viewer": {
            "html": HTML_OUT.name,
            "requires_http_server": True,
        },
    }

    MANIFEST_OUT.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    ok(f"wrote {MANIFEST_OUT}")

    summary = dict(phase09_summary)
    summary["phase12"] = {
        "product": "proto2_cesium_phase12_pickable_viewer",
        "asset_mode": "external fetch",
        "asset_base": "phase12_assets/",
        "manifest": str(MANIFEST_OUT),
        "pick_index": str(PICK_INDEX_OUT),
        "pick_features": pick_index["metadata"]["feature_count"],
        "html": str(HTML_OUT),
        "requires_http_server": True,
    }

    HTML_OUT.write_text(make_html(summary), encoding="utf-8")
    ok(f"wrote {HTML_OUT}")

    summary_out = {
        "product": "proto2_cesium_phase12_pickable_viewer",
        "html": str(HTML_OUT),
        "manifest": str(MANIFEST_OUT),
        "asset_dir": str(ASSET_DIR),
        "pick_index": {
            "path": str(PICK_INDEX_OUT),
            "size_mb": file_size_mb(PICK_INDEX_OUT),
            "feature_count": pick_index["metadata"]["feature_count"],
        },
        "copied_assets": {
            "glb": {"path": str(GLB_OUT), "size_mb": file_size_mb(GLB_OUT)},
            "reversible": {"path": str(REVERSIBLE_OUT), "size_mb": file_size_mb(REVERSIBLE_OUT)},
            "irreversible": {"path": str(IRREVERSIBLE_OUT), "size_mb": file_size_mb(IRREVERSIBLE_OUT)},
            "total": {"path": str(TOTAL_OUT), "size_mb": file_size_mb(TOTAL_OUT)},
        },
        "inherited": {
            "vertices": phase09_summary["vertices"],
            "triangles": phase09_summary["triangles"],
            "moving_parcels": phase09_summary["moving_parcels"],
            "epochs": phase09_summary["epochs"],
            "center_lon": phase09_summary["center_lon"],
            "center_lat": phase09_summary["center_lat"],
        },
        "notes": [
            "Open through a local HTTP server, not file://.",
            "This phase does not change GLB geometry or displacement arrays.",
            "This phase adds parcel lookup by CPU point-in-polygon, not b3dm batch metadata.",
        ],
    }

    SUMMARY_OUT.write_text(json.dumps(summary_out, indent=2), encoding="utf-8")
    ok(f"wrote {SUMMARY_OUT}")

    report = {
        "project_root": str(PROJECT_ROOT),
        "summary": summary_out,
        "manifest": manifest,
    }

    REPORT_JSON_OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    txt_lines = [
        "PROTO2 PHASE 12 CESIUM PICKABLE VIEWER REPORT",
        "",
        f"HTML: {HTML_OUT}",
        f"Asset directory: {ASSET_DIR}",
        f"Manifest: {MANIFEST_OUT}",
        f"Pick index: {PICK_INDEX_OUT}",
        "",
        "Assets:",
        f"- GLB: {GLB_OUT.name} ({file_size_mb(GLB_OUT):.2f} MB)",
        f"- reversible: {REVERSIBLE_OUT.name} ({file_size_mb(REVERSIBLE_OUT):.2f} MB)",
        f"- irreversible: {IRREVERSIBLE_OUT.name} ({file_size_mb(IRREVERSIBLE_OUT):.2f} MB)",
        f"- total: {TOTAL_OUT.name} ({file_size_mb(TOTAL_OUT):.2f} MB)",
        f"- pick index: {PICK_INDEX_OUT.name} ({file_size_mb(PICK_INDEX_OUT):.2f} MB)",
        "",
        "Picking:",
        f"- features: {pick_index['metadata']['feature_count']:,}",
        "- method: CPU bbox + point-in-polygon",
        "- no b3dm required yet",
    ]

    REPORT_TXT_OUT.write_text("\n".join(txt_lines), encoding="utf-8")
    ok(f"wrote {REPORT_JSON_OUT}")
    ok(f"wrote {REPORT_TXT_OUT}")

    total_payload_mb = sum(
        file_size_mb(p)
        for p in [GLB_OUT, REVERSIBLE_OUT, IRREVERSIBLE_OUT, TOTAL_OUT, PICK_INDEX_OUT]
    )

    print("\n=== SUMMARY ===")
    print(f"HTML: {HTML_OUT}")
    print(f"Asset dir: {ASSET_DIR}")
    print(f"Pick features: {pick_index['metadata']['feature_count']:,}")
    print(f"Pick index size: {file_size_mb(PICK_INDEX_OUT):.2f} MB")
    print(f"Total asset payload: {total_payload_mb:.2f} MB")
    print("\nPHASE 12 RESULT: PASS. Pickable Cesium viewer exported.")


if __name__ == "__main__":
    main()
