from pathlib import Path
import base64
import json
import math
import struct
import sys

import numpy as np
import pandas as pd


from _proto2_config import load_project_config, output_cesium_dir, output_data_dir, stage_records_dir


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = load_project_config(PROJECT_ROOT)

OUTPUT_DATA = output_data_dir(PROJECT_ROOT, CONFIG)
OUTPUT_CESIUM = output_cesium_dir(PROJECT_ROOT, CONFIG)
RUN_RECORDS = stage_records_dir(PROJECT_ROOT, CONFIG)

MESH_VERTICES_INDEXED = OUTPUT_DATA / "parcel_cap_mesh_vertices_indexed.parquet"
MESH_TRIANGLES_INDEXED = OUTPUT_DATA / "parcel_cap_mesh_triangles_indexed.parquet"
ANIMATION_MANIFEST = OUTPUT_DATA / "parcel_animation_manifest.json"

REVERSIBLE_BIN = OUTPUT_DATA / "parcel_displacement_reversible_f32.bin"
IRREVERSIBLE_BIN = OUTPUT_DATA / "parcel_displacement_irreversible_f32.bin"
TOTAL_BIN = OUTPUT_DATA / "parcel_displacement_total_f32.bin"

GLB_OUT = OUTPUT_CESIUM / "proto2_animated_parcel_mesh.glb"
HTML_OUT = OUTPUT_CESIUM / "proto2_cesium_animated_glb_preview.html"

SUMMARY_OUT = OUTPUT_CESIUM / "proto2_cesium_animated_glb_summary.json"
MANIFEST_OUT = OUTPUT_CESIUM / "proto2_cesium_animated_glb_manifest.json"

REPORT_TXT_OUT = RUN_RECORDS / "phase09_cesium_animated_glb_report.txt"
REPORT_JSON_OUT = RUN_RECORDS / "phase09_cesium_animated_glb_report.json"

EXPECTED_VERTICES = 249212
EXPECTED_TRIANGLES = 237299
EXPECTED_MOVING_PARCELS = 3923
EXPECTED_EPOCHS = 365

STATIC_HEIGHT_OFFSET_M = 4.0

CESIUM_JS_URL = "https://cesium.com/downloads/cesiumjs/releases/1.123/Build/Cesium/Cesium.js"
CESIUM_CSS_URL = "https://cesium.com/downloads/cesiumjs/releases/1.123/Build/Cesium/Widgets/widgets.css"


def fail(message):
    print(f"[FAIL] {message}")
    sys.exit(1)


def ok(message):
    print(f"[OK] {message}")


def require_files(paths):
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        fail(f"Missing required files: {missing}")


def pad4_bytes(data: bytes, pad_byte: bytes = b"\x00") -> bytes:
    rem = len(data) % 4
    if rem == 0:
        return data
    return data + pad_byte * (4 - rem)


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


def read_f32_matrix(path, rows, cols):
    arr = np.fromfile(path, dtype="<f4")
    expected = rows * cols
    if arr.size != expected:
        fail(f"{path.name} has {arr.size:,} float32 values; expected {expected:,}")
    return arr.reshape((rows, cols))


def component_stats(arr):
    finite = np.isfinite(arr)
    vals = arr[finite]
    return {
        "min": float(vals.min()),
        "max": float(vals.max()),
        "mean": float(vals.mean()),
        "finite_count": int(finite.sum()),
        "nan_count": int(np.isnan(arr).sum()),
    }


def build_glb(positions_f32, colors_u8, texcoord_f32, indices_u32):
    """
    Minimal GLB 2.0:
      POSITION  float32 vec3, local ENU coordinates
      COLOR_0   uint8 normalized vec4
      TEXCOORD_0 float32 vec2: x = displacement_row_index, y = has_displacement flag
      indices   uint32
    """
    if positions_f32.dtype != np.dtype("<f4"):
        fail("positions_f32 must be little-endian float32")
    if colors_u8.dtype != np.dtype("uint8"):
        fail("colors_u8 must be uint8")
    if texcoord_f32.dtype != np.dtype("<f4"):
        fail("texcoord_f32 must be little-endian float32")
    if indices_u32.dtype != np.dtype("<u4"):
        fail("indices_u32 must be little-endian uint32")

    vertex_count = positions_f32.shape[0]
    index_count = indices_u32.size

    chunks = []
    buffer_views = []
    accessors = []
    byte_offset = 0

    def add_buffer_view(data_bytes, target):
        nonlocal byte_offset

        aligned_offset = (byte_offset + 3) // 4 * 4
        padding_needed = aligned_offset - byte_offset

        if padding_needed:
            chunks.append(b"\x00" * padding_needed)
            byte_offset = aligned_offset

        view_index = len(buffer_views)
        buffer_views.append({
            "buffer": 0,
            "byteOffset": byte_offset,
            "byteLength": len(data_bytes),
            "target": target,
        })

        chunks.append(data_bytes)
        byte_offset += len(data_bytes)

        return view_index

    pos_view = add_buffer_view(positions_f32.tobytes(order="C"), 34962)
    color_view = add_buffer_view(colors_u8.tobytes(order="C"), 34962)
    tex_view = add_buffer_view(texcoord_f32.tobytes(order="C"), 34962)
    index_view = add_buffer_view(indices_u32.tobytes(order="C"), 34963)

    bin_chunk = pad4_bytes(b"".join(chunks), b"\x00")

    pos_min = positions_f32.min(axis=0).astype(float).tolist()
    pos_max = positions_f32.max(axis=0).astype(float).tolist()

    pos_accessor = len(accessors)
    accessors.append({
        "bufferView": pos_view,
        "byteOffset": 0,
        "componentType": 5126,
        "count": vertex_count,
        "type": "VEC3",
        "min": pos_min,
        "max": pos_max,
    })

    color_accessor = len(accessors)
    accessors.append({
        "bufferView": color_view,
        "byteOffset": 0,
        "componentType": 5121,
        "count": vertex_count,
        "type": "VEC4",
        "normalized": True,
    })

    tex_accessor = len(accessors)
    accessors.append({
        "bufferView": tex_view,
        "byteOffset": 0,
        "componentType": 5126,
        "count": vertex_count,
        "type": "VEC2",
    })

    index_accessor = len(accessors)
    accessors.append({
        "bufferView": index_view,
        "byteOffset": 0,
        "componentType": 5125,
        "count": index_count,
        "type": "SCALAR",
        "min": [int(indices_u32.min())],
        "max": [int(indices_u32.max())],
    })

    gltf = {
        "asset": {
            "version": "2.0",
            "generator": "Proto2 Phase 09 animated parcel mesh exporter",
        },
        "scene": 0,
        "scenes": [{"nodes": [0]}],

        "nodes": [
            {
                "mesh": 0,
                "name": "proto2_animated_parcel_mesh",
                # z-up → y-up correction matrix, column-major glTF convention.
                # This mirrors the Proto1 standing-card fix:
                # it cancels Cesium's automatic y-up → z-up transform at runtime.
                "matrix": [
                    1, 0, 0, 0,
                    0, 0, -1, 0,
                    0, 1, 0, 0,
                    0, 0, 0, 1,
                ],
            }
        ],

        "meshes": [
            {
                "name": "proto2_animated_parcel_mesh",
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": pos_accessor,
                            "COLOR_0": color_accessor,
                            "TEXCOORD_0": tex_accessor,
                        },
                        "indices": index_accessor,
                        "material": 0,
                        "mode": 4,
                    }
                ],
            }
        ],
        "materials": [
            {
                "name": "vertex_color_translucent_material",
                "doubleSided": True,
                "alphaMode": "BLEND",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 1.0,
                },
            }
        ],
        "buffers": [{"byteLength": len(bin_chunk)}],
        "bufferViews": buffer_views,
        "accessors": accessors,
    }

    json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    json_chunk = pad4_bytes(json_bytes, b" ")

    total_length = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)

    header = struct.pack("<4sII", b"glTF", 2, total_length)
    json_header = struct.pack("<I4s", len(json_chunk), b"JSON")
    bin_header = struct.pack("<I4s", len(bin_chunk), b"BIN\x00")

    return header + json_header + json_chunk + bin_header + bin_chunk


def make_html(summary, glb_b64, reversible_b64, irreversible_b64, total_b64):
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Proto2 Cesium Animated GLB Preview</title>
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
      background: rgba(0,0,0,0.80);
      color: #f1f1f1;
      border: 1px solid rgba(255,255,255,0.18);
      border-radius: 12px;
      padding: 12px 14px;
      min-width: 420px;
      line-height: 1.45;
      font-size: 13px;
      backdrop-filter: blur(8px);
    }}
    #panel h1 {{
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
      grid-template-columns: 90px 1fr;
      gap: 8px 10px;
      margin-top: 10px;
      align-items: center;
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
  </style>
</head>
<body>
  <div id="cesiumContainer"></div>

  <div id="panel">
    <h1>Proto2 Cesium animated GLB preview</h1>
    <div class="row"><span class="label">Vertices</span><span class="value">{summary["vertices"]:,}</span></div>
    <div class="row"><span class="label">Triangles</span><span class="value">{summary["triangles"]:,}</span></div>
    <div class="row"><span class="label">Moving parcels</span><span class="value">{summary["moving_parcels"]:,}</span></div>
    <div class="row"><span class="label">Epoch</span><span class="value" id="epochText">--</span></div>
    <div class="row"><span class="label">FPS-ish</span><span class="value" id="fpsText">--</span></div>

    <div class="controls">
      <button id="playBtn">Pause</button>
      <button id="homeBtn">Fly to parcels</button>

      <label>Mode</label>
      <select id="modeSelect">
        <option value="stacked" selected>stacked</option>
        <option value="total">total</option>
        <option value="reversible">reversible</option>
        <option value="irreversible">irreversible</option>
      </select>

      <label>Epoch</label>
      <input id="epochSlider" type="range" min="0" max="{summary["epochs"] - 1}" value="0"/>

      <label>Height scale</label>
      <input id="heightScale" type="range" min="0" max="0.8" step="0.01" value="0.16"/>
    </div>

    <div id="status">Loading...</div>
  </div>

  <script>
    const META = {json.dumps(summary)};

    const GLB_DATA_URI = "data:model/gltf-binary;base64,{glb_b64}";

    const REVERSIBLE_B64 = "{reversible_b64}";
    const IRREVERSIBLE_B64 = "{irreversible_b64}";
    const TOTAL_B64 = "{total_b64}";

    function b64ToArrayBuffer(b64) {{
      const binary = atob(b64);
      const len = binary.length;
      const bytes = new Uint8Array(len);
      for (let i = 0; i < len; i++) {{
        bytes[i] = binary.charCodeAt(i);
      }}
      return bytes.buffer;
    }}

    const reversible = new Float32Array(b64ToArrayBuffer(REVERSIBLE_B64));
    const irreversible = new Float32Array(b64ToArrayBuffer(IRREVERSIBLE_B64));
    const total = new Float32Array(b64ToArrayBuffer(TOTAL_B64));

    function makeTextureUniform(arr) {{
      // Cesium TextureUniform API has varied slightly across versions.
      // This first path is expected for current CesiumJS releases.
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

    function makeCustomShader(layerValue) {{
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
          u_mode: {{
            type: Cesium.UniformType.FLOAT,
            value: 3.0
          }},
          u_layer: {{
            type: Cesium.UniformType.FLOAT,
            value: layerValue
          }},
          u_reversibleTex: {{
            type: Cesium.UniformType.SAMPLER_2D,
            value: makeTextureUniform(reversible)
          }},
          u_irreversibleTex: {{
            type: Cesium.UniformType.SAMPLER_2D,
            value: makeTextureUniform(irreversible)
          }},
          u_totalTex: {{
            type: Cesium.UniformType.SAMPLER_2D,
            value: makeTextureUniform(total)
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
              float rv = sampleComponent(u_reversibleTex, rowIndex);
              float ir = sampleComponent(u_irreversibleTex, rowIndex);
              float tt = sampleComponent(u_totalTex, rowIndex);

              float disp = tt;

              // mode:
              // 0 = total
              // 1 = reversible
              // 2 = irreversible
              // 3 = stacked: top model = total, bottom model = irreversible
              if (u_mode < 0.5) {{
                disp = tt;
              }} else if (u_mode < 1.5) {{
                disp = rv;
              }} else if (u_mode < 2.5) {{
                disp = ir;
              }} else {{
                if (u_layer < 0.5) {{
                  disp = tt;
                }} else {{
                  disp = ir;
                }}
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
          url: GLB_DATA_URI,
          modelMatrix: modelMatrix,
          customShader: customShader,
          allowPicking: true,
          asynchronous: true
        }});
        viewer.scene.primitives.add(model);
        return model;
      }}

      const model = Cesium.Model.fromGltf({{
        url: GLB_DATA_URI,
        modelMatrix: modelMatrix,
        customShader: customShader,
        allowPicking: true,
        asynchronous: true
      }});
      viewer.scene.primitives.add(model);
      return model;
    }}

    async function main() {{
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

      const topShader = makeCustomShader(0.0);
      const bottomShader = makeCustomShader(1.0);

      const topModel = await loadModel(viewer, modelMatrix, topShader);
      const bottomModel = await loadModel(viewer, modelMatrix, bottomShader);

      bottomModel.show = true;

      let epoch = 0;
      let playing = true;
      let lastStep = performance.now();
      let stepMs = 80;

      const playBtn = document.getElementById("playBtn");
      const modeSelect = document.getElementById("modeSelect");
      const epochSlider = document.getElementById("epochSlider");
      const heightScale = document.getElementById("heightScale");

      function setModeUniform(modeName) {{
        let modeValue = 3.0;

        if (modeName === "total") modeValue = 0.0;
        if (modeName === "reversible") modeValue = 1.0;
        if (modeName === "irreversible") modeValue = 2.0;
        if (modeName === "stacked") modeValue = 3.0;

        topShader.setUniform("u_mode", modeValue);
        bottomShader.setUniform("u_mode", modeValue);

        bottomModel.show = modeName === "stacked";
      }}

      function updateUniforms() {{
        const h = Number(heightScale.value);
        topShader.setUniform("u_epoch", epoch);
        bottomShader.setUniform("u_epoch", epoch);
        topShader.setUniform("u_heightScale", h);
        bottomShader.setUniform("u_heightScale", h);
        setModeUniform(modeSelect.value);
        document.getElementById("epochText").textContent = META.epoch_labels[epoch];
      }}

      playBtn.addEventListener("click", () => {{
        playing = !playing;
        playBtn.textContent = playing ? "Pause" : "Play";
      }});

      epochSlider.addEventListener("input", () => {{
        epoch = Number(epochSlider.value);
        updateUniforms();
      }});

      modeSelect.addEventListener("change", updateUniforms);
      heightScale.addEventListener("input", updateUniforms);

      function flyHome() {{
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

      document.getElementById("homeBtn").addEventListener("click", flyHome);

      updateUniforms();
      flyHome();

      document.getElementById("status").textContent = "Animated model loaded.";

      let frameCount = 0;
      let fpsTime = performance.now();

      viewer.scene.preRender.addEventListener(function(scene, time) {{
        const now = performance.now();

        if (playing && now - lastStep >= stepMs) {{
          epoch = (epoch + 1) % META.epochs;
          epochSlider.value = epoch;
          updateUniforms();
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

      console.log("Proto2 Cesium animated GLB metadata:", META);
      console.log("top model:", topModel);
      console.log("bottom model:", bottomModel);
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
    RUN_RECORDS.mkdir(parents=True, exist_ok=True)

    print("\n=== PROTO2 PHASE 09: EXPORT CESIUM ANIMATED GLB PREVIEW ===")
    print(f"Project root: {PROJECT_ROOT}")

    require_files([
        MESH_VERTICES_INDEXED,
        MESH_TRIANGLES_INDEXED,
        ANIMATION_MANIFEST,
        REVERSIBLE_BIN,
        IRREVERSIBLE_BIN,
        TOTAL_BIN,
    ])

    print(f"\nReading animation manifest:\n  {ANIMATION_MANIFEST}")
    animation_manifest = json.loads(ANIMATION_MANIFEST.read_text(encoding="utf-8"))

    n_moving = int(animation_manifest["shape"]["moving_parcels"])
    n_epochs = int(animation_manifest["shape"]["epochs"])

    if n_moving != EXPECTED_MOVING_PARCELS:
        fail(f"moving parcels {n_moving:,} != expected {EXPECTED_MOVING_PARCELS:,}")

    if n_epochs != EXPECTED_EPOCHS:
        fail(f"epochs {n_epochs:,} != expected {EXPECTED_EPOCHS:,}")

    print("\nReading displacement arrays...")
    reversible = read_f32_matrix(REVERSIBLE_BIN, n_moving, n_epochs)
    irreversible = read_f32_matrix(IRREVERSIBLE_BIN, n_moving, n_epochs)
    total = read_f32_matrix(TOTAL_BIN, n_moving, n_epochs)

    max_total_diff = float(np.nanmax(np.abs(total - (reversible + irreversible))))
    if max_total_diff > 1e-4:
        fail(f"total != reversible + irreversible; max diff={max_total_diff:.9g}")

    ok(f"displacement arrays loaded: {n_moving:,} x {n_epochs:,}")
    ok(f"component check passed; max total diff={max_total_diff:.9g}")

    print(f"\nReading indexed mesh vertices:\n  {MESH_VERTICES_INDEXED}")
    vertices = pd.read_parquet(MESH_VERTICES_INDEXED)

    print(f"Reading indexed mesh triangles:\n  {MESH_TRIANGLES_INDEXED}")
    triangles = pd.read_parquet(MESH_TRIANGLES_INDEXED)

    if len(vertices) != EXPECTED_VERTICES:
        fail(f"vertex count {len(vertices):,} != expected {EXPECTED_VERTICES:,}")

    if len(triangles) != EXPECTED_TRIANGLES:
        fail(f"triangle count {len(triangles):,} != expected {EXPECTED_TRIANGLES:,}")

    required_vertex_cols = [
        "global_vertex_index",
        "lon",
        "lat",
        "has_displacement",
        "displacement_row_index",
    ]
    required_triangle_cols = [
        "global_triangle_index",
        "v0",
        "v1",
        "v2",
    ]

    missing_v = [c for c in required_vertex_cols if c not in vertices.columns]
    missing_t = [c for c in required_triangle_cols if c not in triangles.columns]

    if missing_v:
        fail(f"vertices missing required columns: {missing_v}")

    if missing_t:
        fail(f"triangles missing required columns: {missing_t}")

    vertices = vertices.sort_values("global_vertex_index").reset_index(drop=True)
    triangles = triangles.sort_values("global_triangle_index").reset_index(drop=True)

    if not np.array_equal(
        vertices["global_vertex_index"].to_numpy(dtype=np.int64),
        np.arange(len(vertices), dtype=np.int64),
    ):
        fail("global_vertex_index is not contiguous/sorted")

    if not np.array_equal(
        triangles["global_triangle_index"].to_numpy(dtype=np.int64),
        np.arange(len(triangles), dtype=np.int64),
    ):
        fail("global_triangle_index is not contiguous/sorted")

    ok(f"mesh loaded: {len(vertices):,} vertices, {len(triangles):,} triangles")

    lon = vertices["lon"].to_numpy(dtype=np.float64)
    lat = vertices["lat"].to_numpy(dtype=np.float64)

    west = float(lon.min())
    east = float(lon.max())
    south = float(lat.min())
    north = float(lat.max())

    center_lon = 0.5 * (west + east)
    center_lat = 0.5 * (south + north)
    center_h = STATIC_HEIGHT_OFFSET_M

    print("\nConverting lon/lat to local ENU positions...")
    x_ecef, y_ecef, z_ecef = wgs84_to_ecef(
        lon,
        lat,
        np.full_like(lon, STATIC_HEIGHT_OFFSET_M),
    )

    local_x, local_y, local_z = ecef_to_local_enu(
        x_ecef,
        y_ecef,
        z_ecef,
        center_lon,
        center_lat,
        STATIC_HEIGHT_OFFSET_M,
    )

    # Horizontal orientation correction after standing-card fix.
    # Phase 10 diagnostic showed the mesh is 90° counterclockwise relative to
    # the SVG/static preview truth. Rotate local ENU positions 90° clockwise:
    #
    #   x' =  y
    #   y' = -x
    #
    # Keep z unchanged so the existing standing-card fix and vertical breathing
    # remain valid.
    rot_x = local_y
    rot_y = -local_x
    rot_z = local_z

    positions = np.column_stack([rot_x, rot_y, rot_z]).astype("<f4")

    disp_row = vertices["displacement_row_index"].to_numpy(dtype=np.float32)
    has_disp = vertices["has_displacement"].astype(bool).to_numpy()

    if np.nanmin(disp_row) < -1:
        fail("displacement_row_index below -1 found")

    if np.nanmax(disp_row) >= n_moving:
        fail("displacement_row_index references outside animation matrix")

    texcoord = np.empty((len(vertices), 2), dtype="<f4")
    texcoord[:, 0] = disp_row.astype("<f4")
    texcoord[:, 1] = has_disp.astype(np.float32)

    colors = np.empty((len(vertices), 4), dtype=np.uint8)
    colors[has_disp] = np.array([47, 128, 237, 215], dtype=np.uint8)
    colors[~has_disp] = np.array([184, 184, 184, 90], dtype=np.uint8)

    indices = triangles[["v0", "v1", "v2"]].to_numpy(dtype="<u4").reshape(-1)

    if int(indices.min()) < 0 or int(indices.max()) >= len(vertices):
        fail("indices reference vertices out of range")

    ok("positions, texcoords, colors, and indices built")

    print("\nBuilding animated GLB carrier...")
    glb = build_glb(positions, colors, texcoord, indices)
    GLB_OUT.write_bytes(glb)
    ok(f"wrote {GLB_OUT}")

    span_east_m = float(positions[:, 0].max() - positions[:, 0].min())
    span_north_m = float(positions[:, 1].max() - positions[:, 1].min())
    camera_height_m = max(span_east_m, span_north_m) * 2.2

    try:
        epoch_labels = pd.date_range(
            start=animation_manifest["epoch"]["start"],
            end=animation_manifest["epoch"]["end"],
            periods=n_epochs,
        ).strftime("%Y-%m-%d").tolist()
    except Exception:
        epoch_labels = [f"epoch {i}" for i in range(n_epochs)]

    stats = {
        "reversible": component_stats(reversible),
        "irreversible": component_stats(irreversible),
        "total": component_stats(total),
    }

    summary = {
        "product": "proto2_cesium_animated_glb_preview",
        "purpose": "first Cesium shader-driven animation bridge",
        "vertices": int(len(vertices)),
        "triangles": int(len(triangles)),
        "indices": int(indices.size),
        "moving_vertices": int(has_disp.sum()),
        "blank_vertices": int((~has_disp).sum()),
        "moving_parcels": int(n_moving),
        "epochs": int(n_epochs),
        "epoch_start": animation_manifest["epoch"]["start"],
        "epoch_end": animation_manifest["epoch"]["end"],
        "epoch_labels": epoch_labels,
        "stats": stats,
        "max_total_diff": max_total_diff,
        "glb_size_mb": float(GLB_OUT.stat().st_size / (1024 * 1024)),
        "center_lon": center_lon,
        "center_lat": center_lat,
        "center_height_m": center_h,
        "camera_height_m": float(camera_height_m),
        "bounds_wgs84": {
            "west": west,
            "south": south,
            "east": east,
            "north": north,
        },
        "local_span_m": {
            "east_west": span_east_m,
            "north_south": span_north_m,
        },
        "outputs": {
            "glb": str(GLB_OUT),
            "html": str(HTML_OUT),
            "summary": str(SUMMARY_OUT),
            "manifest": str(MANIFEST_OUT),
        },
        "notes": {
            "not_b3dm_yet": True,
            "architecture": "one GLB carrier loaded twice for stacked mode; deformation sampled in Cesium CustomShader from float arrays",
            "blank_rule": "TEXCOORD_0.x = -1 and TEXCOORD_0.y = 0 for blank vertices",
            "moving_rule": "TEXCOORD_0.x = displacement_row_index and TEXCOORD_0.y = 1",
            "if_hiccup": "likely Cesium TextureUniform or CustomShader syntax, not data failure",
        },
    }

    manifest = {
        "product": "proto2_cesium_animated_glb_assets",
        "version": 1,
        "glb": str(GLB_OUT),
        "html_preview": str(HTML_OUT),
        "animation_arrays": {
            "reversible": str(REVERSIBLE_BIN),
            "irreversible": str(IRREVERSIBLE_BIN),
            "total": str(TOTAL_BIN),
            "shape": [n_moving, n_epochs],
        },
        "attribute_contract": {
            "POSITION": "local ENU XYZ",
            "COLOR_0": "moving/blank visual color",
            "TEXCOORD_0.x": "displacement_row_index; -1 for blank",
            "TEXCOORD_0.y": "has_displacement flag; 1 moving, 0 blank",
        },
        "coordinate_system": {
            "glb_local_frame": "ENU",
            "model_matrix": "Cesium.Transforms.eastNorthUpToFixedFrame(center)",
            "center_lon": center_lon,
            "center_lat": center_lat,
            "center_height_m": center_h,
        },
    }

    SUMMARY_OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    ok(f"wrote {SUMMARY_OUT}")

    MANIFEST_OUT.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    ok(f"wrote {MANIFEST_OUT}")

    print("\nEmbedding GLB and displacement arrays into Cesium HTML...")
    html_text = make_html(
        summary=summary,
        glb_b64=base64.b64encode(glb).decode("ascii"),
        reversible_b64=base64.b64encode(reversible.astype("<f4", copy=False).tobytes()).decode("ascii"),
        irreversible_b64=base64.b64encode(irreversible.astype("<f4", copy=False).tobytes()).decode("ascii"),
        total_b64=base64.b64encode(total.astype("<f4", copy=False).tobytes()).decode("ascii"),
    )

    HTML_OUT.write_text(html_text, encoding="utf-8")
    ok(f"wrote {HTML_OUT}")

    report = {
        "project_root": str(PROJECT_ROOT),
        "summary": summary,
        "manifest": manifest,
    }

    REPORT_JSON_OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    txt_lines = [
        "PROTO2 PHASE 09 CESIUM ANIMATED GLB REPORT",
        "",
        f"vertices: {len(vertices):,}",
        f"triangles: {len(triangles):,}",
        f"moving vertices: {int(has_disp.sum()):,}",
        f"blank vertices: {int((~has_disp).sum()):,}",
        f"moving parcels: {n_moving:,}",
        f"epochs: {n_epochs:,}",
        f"GLB size MB: {summary['glb_size_mb']:.3f}",
        "",
        f"center lon: {center_lon:.8f}",
        f"center lat: {center_lat:.8f}",
        f"camera height m: {camera_height_m:.3f}",
        "",
        "component ranges:",
        f"reversible: {stats['reversible']['min']:.6f} to {stats['reversible']['max']:.6f}",
        f"irreversible: {stats['irreversible']['min']:.6f} to {stats['irreversible']['max']:.6f}",
        f"total: {stats['total']['min']:.6f} to {stats['total']['max']:.6f}",
        "",
        "outputs:",
        f"- {GLB_OUT}",
        f"- {HTML_OUT}",
        f"- {SUMMARY_OUT}",
        f"- {MANIFEST_OUT}",
    ]

    REPORT_TXT_OUT.write_text("\n".join(txt_lines), encoding="utf-8")
    ok(f"wrote {REPORT_JSON_OUT}")
    ok(f"wrote {REPORT_TXT_OUT}")

    html_size_mb = HTML_OUT.stat().st_size / (1024 * 1024)

    print("\n=== SUMMARY ===")
    print(f"Vertices: {len(vertices):,}")
    print(f"Triangles: {len(triangles):,}")
    print(f"Moving vertices: {int(has_disp.sum()):,}")
    print(f"Blank vertices: {int((~has_disp).sum()):,}")
    print(f"Epochs: {n_epochs:,}")
    print(f"GLB size: {summary['glb_size_mb']:.2f} MB")
    print(f"HTML size: {html_size_mb:.2f} MB")
    print(f"Preview HTML: {HTML_OUT}")
    print("\nPHASE 09 RESULT: PASS. Cesium animated GLB preview exported.")


if __name__ == "__main__":
    main()
