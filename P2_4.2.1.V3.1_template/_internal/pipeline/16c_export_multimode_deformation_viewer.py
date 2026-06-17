from __future__ import annotations

from pathlib import Path
import json
import re
import shutil
import sys
from typing import Any, Dict, Tuple
import struct
import numpy as np


from _proto2_config import load_project_config, output_cesium_dir, output_data_dir, stage_records_dir


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = load_project_config(PROJECT_ROOT)
OUTPUT_DATA = output_data_dir(PROJECT_ROOT, CONFIG)
OUTPUT_CESIUM = output_cesium_dir(PROJECT_ROOT, CONFIG)
RUN_RECORDS = stage_records_dir(PROJECT_ROOT, CONFIG)

SOURCE_HTML = OUTPUT_CESIUM / "proto2_m1_irreversible_piston_viewer.html"
DISPLAY_TUNING_JSON = OUTPUT_DATA / "parcel_display_tuning.json"
PISTON_SUMMARY_JSON = OUTPUT_DATA / "parcel_piston_mesh_summary.json"

ASSET_DIR = OUTPUT_CESIUM / "phase15_piston_assets"
PISTON_GLB = ASSET_DIR / "proto2_irreversible_piston_mesh.glb"
PHASE14_CAP_GLB = OUTPUT_CESIUM / "phase14_color_assets" / "proto2_animated_parcel_mesh.glb"
CAP_GLB_COPY = ASSET_DIR / "proto2_animated_parcel_mesh.glb"
WALL_ONLY_GLB = ASSET_DIR / "proto2_parcel_wall_mesh_blend.glb"

HTML_OUT = OUTPUT_CESIUM / "proto2_m1_multimode_deformation_viewer_16c.html"
SUMMARY_OUT = OUTPUT_CESIUM / "proto2_m1_multimode_deformation_viewer_16c_summary.json"
REPORT_TXT_OUT = RUN_RECORDS / "phase16c_multimode_deformation_viewer_report.txt"
REPORT_JSON_OUT = RUN_RECORDS / "phase16c_multimode_deformation_viewer_report.json"


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    sys.exit(1)


def ok(message: str) -> None:
    print(f"[OK] {message}")


def require(path: Path, label: str) -> None:
    if not path.exists():
        fail(f"Missing {label}: {path}")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def replace_one(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        fail(f"Could not find replacement target for {label}")
    return text.replace(old, new, 1)


def replace_regex(text: str, pattern: str, repl: str, label: str, flags: int = re.S) -> str:
    new, n = re.subn(pattern, lambda _m: repl, text, count=1, flags=flags)
    if n != 1:
        fail(f"Expected 1 regex replacement for {label}, got {n}")
    return new


def replace_regex_func(text: str, pattern: str, func, label: str, flags: int = re.S) -> str:
    new, n = re.subn(pattern, func, text, count=1, flags=flags)
    if n != 1:
        fail(f"Expected 1 regex replacement for {label}, got {n}")
    return new


def extract_meta_from_html(html: str) -> Dict[str, Any]:
    match = re.search(r"const META = (.*?);\n", html, flags=re.S)
    if not match:
        fail("Could not find const META object in source HTML")
    try:
        meta = json.loads(match.group(1)) # type: ignore
    except Exception as exc:
        fail(f"Could not parse source META JSON: {exc}")
    if not isinstance(meta, dict):
        fail("Source META is not a JSON object")
    return meta



def pad4_bytes(data: bytes, pad: bytes) -> bytes:
    extra = (-len(data)) % 4
    return data + pad * extra


def parse_glb(path: Path) -> Tuple[Dict[str, Any], bytes]:
    data = path.read_bytes()
    if len(data) < 20:
        fail(f"GLB too short: {path}")
    magic, version, total_len = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF" or version != 2:
        fail(f"Not a GLB v2 file: {path}")
    if total_len != len(data):
        fail(f"GLB length mismatch for {path}: header={total_len}, actual={len(data)}")
    offset = 12
    json_obj = None
    bin_chunk = None
    while offset < len(data):
        chunk_len, chunk_type = struct.unpack_from("<I4s", data, offset)
        offset += 8
        chunk = data[offset:offset + chunk_len]
        offset += chunk_len
        if chunk_type == b"JSON":
            json_obj = json.loads(chunk.rstrip(b" \x00").decode("utf-8"))
        elif chunk_type == b"BIN\x00":
            bin_chunk = chunk
    if json_obj is None or bin_chunk is None:
        fail(f"Could not read JSON/BIN chunks from {path}")
    return json_obj, bin_chunk # type: ignore


def accessor_array(gltf: Dict[str, Any], bin_chunk: bytes, accessor_index: int) -> np.ndarray:
    accessor = gltf["accessors"][accessor_index]
    view = gltf["bufferViews"][accessor["bufferView"]]
    count = int(accessor["count"])
    acc_offset = int(accessor.get("byteOffset", 0))
    view_offset = int(view.get("byteOffset", 0))
    byte_offset = view_offset + acc_offset
    comp = int(accessor["componentType"])
    typ = accessor["type"]
    ncomp = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[typ]
    dtype = {
        5126: np.dtype("<f4"),
        5125: np.dtype("<u4"),
        5123: np.dtype("<u2"),
        5121: np.dtype("u1"),
    }.get(comp)
    if dtype is None:
        fail(f"Unsupported GLB accessor component type {comp}")
    arr = np.frombuffer(bin_chunk, dtype=dtype, count=count * ncomp, offset=byte_offset)
    if ncomp > 1:
        arr = arr.reshape(count, ncomp)
    return np.array(arr, copy=True)


def write_simple_glb(path: Path, positions_f32: np.ndarray, colors_u8: np.ndarray, tex0_f32: np.ndarray, tex1_f32: np.ndarray, indices_u32: np.ndarray, *, alpha_mode: str = "BLEND") -> None:
    positions_f32 = np.asarray(positions_f32, dtype="<f4")
    colors_u8 = np.asarray(colors_u8, dtype=np.uint8)
    tex0_f32 = np.asarray(tex0_f32, dtype="<f4")
    tex1_f32 = np.asarray(tex1_f32, dtype="<f4")
    indices_u32 = np.asarray(indices_u32, dtype="<u4")

    chunks = []
    buffer_views = []
    accessors = []
    byte_offset = 0

    def add_view(raw: bytes, target: int) -> int:
        nonlocal byte_offset
        aligned = (byte_offset + 3) // 4 * 4
        if aligned > byte_offset:
            chunks.append(b"\x00" * (aligned - byte_offset))
            byte_offset = aligned
        idx = len(buffer_views)
        buffer_views.append({"buffer": 0, "byteOffset": byte_offset, "byteLength": len(raw), "target": target})
        chunks.append(raw)
        byte_offset += len(raw)
        return idx

    pos_view = add_view(positions_f32.tobytes(order="C"), 34962)
    color_view = add_view(colors_u8.tobytes(order="C"), 34962)
    tex0_view = add_view(tex0_f32.tobytes(order="C"), 34962)
    tex1_view = add_view(tex1_f32.tobytes(order="C"), 34962)
    index_view = add_view(indices_u32.tobytes(order="C"), 34963)
    bin_chunk = pad4_bytes(b"".join(chunks), b"\x00")

    vertex_count = int(len(positions_f32))
    index_count = int(indices_u32.size)
    pos_min = positions_f32.min(axis=0).astype(float).tolist()
    pos_max = positions_f32.max(axis=0).astype(float).tolist()

    pos_accessor = len(accessors); accessors.append({"bufferView": pos_view, "componentType": 5126, "count": vertex_count, "type": "VEC3", "min": pos_min, "max": pos_max})
    color_accessor = len(accessors); accessors.append({"bufferView": color_view, "componentType": 5121, "count": vertex_count, "type": "VEC4", "normalized": True})
    tex0_accessor = len(accessors); accessors.append({"bufferView": tex0_view, "componentType": 5126, "count": vertex_count, "type": "VEC2"})
    tex1_accessor = len(accessors); accessors.append({"bufferView": tex1_view, "componentType": 5126, "count": vertex_count, "type": "VEC2"})
    index_accessor = len(accessors); accessors.append({"bufferView": index_view, "componentType": 5125, "count": index_count, "type": "SCALAR", "min": [int(indices_u32.min())], "max": [int(indices_u32.max())]})

    gltf = {
        "asset": {"version": "2.0", "generator": "Proto2 Phase16C wall-only splitter"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{
            "mesh": 0,
            "name": "proto2_parcel_wall_mesh_blend",
            "matrix": [
                1, 0, 0, 0,
                0, 0, -1, 0,
                0, 1, 0, 0,
                0, 0, 0, 1,
            ],
        }],
        "meshes": [{"name": "proto2_parcel_wall_mesh_blend", "primitives": [{
            "attributes": {"POSITION": pos_accessor, "COLOR_0": color_accessor, "TEXCOORD_0": tex0_accessor, "TEXCOORD_1": tex1_accessor},
            "indices": index_accessor,
            "material": 0,
            "mode": 4,
        }]}],
        "materials": [{
            "name": "wall_blend_material",
            "doubleSided": True,
            "alphaMode": alpha_mode,
            "pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1], "metallicFactor": 0.0, "roughnessFactor": 1.0},
        }],
        "buffers": [{"byteLength": len(bin_chunk)}],
        "bufferViews": buffer_views,
        "accessors": accessors,
    }
    json_chunk = pad4_bytes(json.dumps(gltf, separators=(",", ":")).encode("utf-8"), b" ")
    total_len = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
    out = struct.pack("<4sII", b"glTF", 2, total_len)
    out += struct.pack("<I4s", len(json_chunk), b"JSON") + json_chunk
    out += struct.pack("<I4s", len(bin_chunk), b"BIN\x00") + bin_chunk
    path.write_bytes(out)


def build_wall_only_glb_from_piston(src: Path, dst: Path) -> Dict[str, Any]:
    gltf, bin_chunk = parse_glb(src)
    prim = gltf["meshes"][0]["primitives"][0]
    attrs = prim["attributes"]
    positions = accessor_array(gltf, bin_chunk, attrs["POSITION"]).astype("<f4")
    colors = accessor_array(gltf, bin_chunk, attrs["COLOR_0"]).astype(np.uint8)
    tex0 = accessor_array(gltf, bin_chunk, attrs["TEXCOORD_0"]).astype("<f4")
    tex1 = accessor_array(gltf, bin_chunk, attrs["TEXCOORD_1"]).astype("<f4")
    indices = accessor_array(gltf, bin_chunk, prim["indices"]).astype("<u4").reshape(-1)

    tris = indices.reshape(-1, 3)
    wall_mask = np.all(tex1[tris, 1] > 0.5, axis=1)
    wall_tris = tris[wall_mask]
    if wall_tris.size == 0:
        fail("No wall triangles found in piston GLB")

    used = np.unique(wall_tris.reshape(-1))
    remap = np.full(len(positions), -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    new_indices = remap[wall_tris.reshape(-1)].astype("<u4")

    write_simple_glb(dst, positions[used], colors[used], tex0[used], tex1[used], new_indices, alpha_mode="BLEND")
    return {
        "source": str(src),
        "output": str(dst),
        "source_vertices": int(len(positions)),
        "source_triangles": int(len(indices) // 3),
        "wall_vertices": int(len(used)),
        "wall_triangles": int(len(new_indices) // 3),
        "wall_glb_size_mb": float(dst.stat().st_size / (1024 * 1024)),
    }

def make_piston_shader_js() -> str:
    return r'''function makePistonShader(componentValue, texturesObj, pistonVisualMode = 0.0) {
    const rampShaderText = buildColorRampShaderText();

    return new Cesium.CustomShader({
        mode: Cesium.CustomShaderMode.MODIFY_MATERIAL,
        uniforms: {
            u_epoch: { type: Cesium.UniformType.FLOAT, value: 0.0 },
            u_epochs: { type: Cesium.UniformType.FLOAT, value: META.epochs },
            u_rows: { type: Cesium.UniformType.FLOAT, value: META.moving_parcels },
            u_heightScale: { type: Cesium.UniformType.FLOAT, value: heightScaleValue },
            u_component: { type: Cesium.UniformType.FLOAT, value: componentValue },
            u_pistonVisualMode: { type: Cesium.UniformType.FLOAT, value: pistonVisualMode },
            u_irreversibleColorMode: { type: Cesium.UniformType.FLOAT, value: 0.0 },
            u_modelOriginHeight: { type: Cesium.UniformType.FLOAT, value: META.center_height_m },
            u_activeDisplayDatumHeight: { type: Cesium.UniformType.FLOAT, value: activeDatumHeight },
            u_groundHeight: { type: Cesium.UniformType.FLOAT, value: META.phase15.display_tuning.ground_height_m },
            u_irreversibleMotionBoost: { type: Cesium.UniformType.FLOAT, value: META.phase15.display_tuning.irreversible_motion_boost || 1.0 },
            u_reversibleTex: { type: Cesium.UniformType.SAMPLER_2D, value: texturesObj.reversible },
            u_irreversibleTex: { type: Cesium.UniformType.SAMPLER_2D, value: texturesObj.irreversible },
            u_totalTex: { type: Cesium.UniformType.SAMPLER_2D, value: texturesObj.total },
            u_viTex: { type: Cesium.UniformType.SAMPLER_2D, value: texturesObj.vi }
        },
        vertexShaderText: `
          float sampleComponent(sampler2D tex, float rowIndex) {
            float u = (u_epoch + 0.5) / u_epochs;
            float v = (rowIndex + 0.5) / u_rows;
            return texture(tex, vec2(u, v)).r;
          }

          float sampleActiveDisplacement(float rowIndex) {
            if (u_component < 0.5) return sampleComponent(u_totalTex, rowIndex);
            if (u_component < 1.5) return sampleComponent(u_reversibleTex, rowIndex);
            return sampleComponent(u_irreversibleTex, rowIndex);
          }

          void vertexMain(VertexInput vsInput, inout czm_modelVertexOutput vsOutput) {
            float rowIndex = vsInput.attributes.texCoord_0.x;
            float hasDisp = vsInput.attributes.texCoord_0.y;
            float pistonT = clamp(vsInput.attributes.texCoord_1.x, 0.0, 1.0);

            float bottomZ = u_groundHeight - u_modelOriginHeight;
            float topZ = u_activeDisplayDatumHeight - u_modelOriginHeight;

            if (hasDisp > 0.5 && rowIndex >= 0.0) {
              // Phase16C: piston height is always displacement-based.
              // The irreversible colour toggle changes colour only, not geometry.
              topZ += sampleActiveDisplacement(rowIndex) * u_heightScale;
            }

            vsOutput.positionMC.z = mix(bottomZ, topZ, pistonT);
          }
        `,
        fragmentShaderText: `
          float sampleTimeComponentFrag(sampler2D tex, float rowIndex) {
            float u = (u_epoch + 0.5) / u_epochs;
            float v = (rowIndex + 0.5) / u_rows;
            return texture(tex, vec2(u, v)).r;
          }

          float sampleParcelVectorFrag(sampler2D tex, float rowIndex) {
            float v = (rowIndex + 0.5) / u_rows;
            return texture(tex, vec2(0.5, v)).r;
          }

          ${rampShaderText}

          vec3 velocityColor(float rowIndex) {
            float vi = sampleParcelVectorFrag(u_viTex, rowIndex);
            return rampVelocity(vi);
          }

          vec3 reversibleColor(float rowIndex) {
            float v = sampleTimeComponentFrag(u_reversibleTex, rowIndex);
            return rampReversible(v);
          }

          vec3 irreversibleDisplacementColor(float rowIndex) {
            float v = sampleTimeComponentFrag(u_irreversibleTex, rowIndex);
            return rampTotal(v);
          }

          void fragmentMain(FragmentInput fsInput, inout czm_modelMaterial material) {
            float rowIndex = fsInput.attributes.texCoord_0.x;
            float hasDisp = fsInput.attributes.texCoord_0.y;
            float wallFlag = fsInput.attributes.texCoord_1.y;

            if (hasDisp <= 0.5 || rowIndex < 0.0) {
              vec3 blankColor = vec3(0.28, 0.28, 0.28);
              if (wallFlag > 0.5) blankColor = mix(blankColor, vec3(0.05, 0.055, 0.06), 0.42);
              material.diffuse = blankColor;
              material.alpha = 1.0;
              return;
            }

            vec3 color;
            if (wallFlag > 0.5) {
              if (u_pistonVisualMode > 0.5 && u_pistonVisualMode < 1.5) {
                // Total mode wall = long-term velocity.
                color = mix(velocityColor(rowIndex), vec3(0.045, 0.050, 0.060), 0.52);
              } else if (u_irreversibleColorMode > 0.5 || u_pistonVisualMode > 1.5) {
                color = mix(irreversibleDisplacementColor(rowIndex), vec3(0.045, 0.050, 0.060), 0.42);
              } else {
                color = mix(velocityColor(rowIndex), vec3(0.045, 0.050, 0.060), 0.52);
              }
              material.alpha = 1.0;
            } else {
              if (u_pistonVisualMode > 0.5 && u_pistonVisualMode < 1.5) {
                // Total mode cap = reversible component.
                color = reversibleColor(rowIndex);
              } else if (u_irreversibleColorMode > 0.5 || u_pistonVisualMode > 1.5) {
                color = irreversibleDisplacementColor(rowIndex);
              } else {
                color = velocityColor(rowIndex);
              }
              material.alpha = 1.0;
            }
            material.diffuse = color;
          }
        `
    });
}

function makeCapShader(componentValue, texturesObj, capRole = 0.0) {
    const rampShaderText = buildColorRampShaderText();

    return new Cesium.CustomShader({
        mode: Cesium.CustomShaderMode.MODIFY_MATERIAL,
        uniforms: {
            u_epoch: { type: Cesium.UniformType.FLOAT, value: 0.0 },
            u_epochs: { type: Cesium.UniformType.FLOAT, value: META.epochs },
            u_rows: { type: Cesium.UniformType.FLOAT, value: META.moving_parcels },
            u_heightScale: { type: Cesium.UniformType.FLOAT, value: heightScaleValue },
            u_component: { type: Cesium.UniformType.FLOAT, value: componentValue },
            u_capRole: { type: Cesium.UniformType.FLOAT, value: capRole },
            u_datumReferenceEnabled: { type: Cesium.UniformType.FLOAT, value: 1.0 },
            u_modelOriginHeight: { type: Cesium.UniformType.FLOAT, value: META.center_height_m },
            u_activeDisplayDatumHeight: { type: Cesium.UniformType.FLOAT, value: activeDatumHeight },
            u_reversibleTex: { type: Cesium.UniformType.SAMPLER_2D, value: texturesObj.reversible },
            u_irreversibleTex: { type: Cesium.UniformType.SAMPLER_2D, value: texturesObj.irreversible },
            u_totalTex: { type: Cesium.UniformType.SAMPLER_2D, value: texturesObj.total }
        },
        vertexShaderText: `
          float sampleComponent(sampler2D tex, float rowIndex) {
            float u = (u_epoch + 0.5) / u_epochs;
            float v = (rowIndex + 0.5) / u_rows;
            return texture(tex, vec2(u, v)).r;
          }

          float sampleByComponent(float rowIndex) {
            if (u_component < 0.5) return sampleComponent(u_totalTex, rowIndex);
            if (u_component < 1.5) return sampleComponent(u_reversibleTex, rowIndex);
            return sampleComponent(u_irreversibleTex, rowIndex);
          }

          void vertexMain(VertexInput vsInput, inout czm_modelVertexOutput vsOutput) {
            float rowIndex = vsInput.attributes.texCoord_0.x;
            float hasDisp = vsInput.attributes.texCoord_0.y;
            float z = u_activeDisplayDatumHeight - u_modelOriginHeight;

            if (hasDisp > 0.5 && rowIndex >= 0.0) {
              if (u_capRole < 0.5) {
                // Moving cap: reversible in reversible mode; total in combined mode.
                z += sampleByComponent(rowIndex) * u_heightScale;
              } else if (u_capRole > 1.5) {
                // Combined datum cap = irreversible piston top, with honest displacement scale.
                z += sampleComponent(u_irreversibleTex, rowIndex) * u_heightScale;
              }
            }

            vsOutput.positionMC.z = z;
          }
        `,
        fragmentShaderText: `
          float sampleTimeComponentFrag(sampler2D tex, float rowIndex) {
            float u = (u_epoch + 0.5) / u_epochs;
            float v = (rowIndex + 0.5) / u_rows;
            return texture(tex, vec2(u, v)).r;
          }

          ${rampShaderText}

          void fragmentMain(FragmentInput fsInput, inout czm_modelMaterial material) {
            float rowIndex = fsInput.attributes.texCoord_0.x;
            float hasDisp = fsInput.attributes.texCoord_0.y;

            if (u_capRole < 0.5) {
              if (hasDisp <= 0.5 || rowIndex < 0.0) {
                material.diffuse = vec3(0.28, 0.28, 0.28);
                material.alpha = 0.0;
                return;
              }
              float rev = sampleTimeComponentFrag(u_reversibleTex, rowIndex);
              material.diffuse = rampReversible(rev);
              material.alpha = 0.96;
              return;
            }

            // Reference cap: blanks stay visible even when datum reference is off.
            if (hasDisp <= 0.5 || rowIndex < 0.0) {
              material.diffuse = vec3(0.30, 0.31, 0.33);
              material.alpha = 0.44;
              return;
            }

            if (u_datumReferenceEnabled < 0.5) {
              material.diffuse = vec3(0.96, 0.98, 1.0);
              material.alpha = 0.0;
              return;
            }

            if (u_capRole > 1.5) {
              // Combined irreversible cap: opaque and displacement-coloured.
              float irr = sampleTimeComponentFrag(u_irreversibleTex, rowIndex);
              material.diffuse = rampTotal(irr);
              material.alpha = 1.0;
            } else {
              material.diffuse = vec3(0.96, 0.98, 1.0);
              material.alpha = 0.50;
            }
          }
        `
    });
}

function makeWallShader(wallMode, texturesObj) {
    // wallMode: 0 = combined irreversible body wall, 1 = reversible datum-to-cap wall, 2 = combined irreversible-to-total breathing wall
    const rampShaderText = buildColorRampShaderText();

    return new Cesium.CustomShader({
        mode: Cesium.CustomShaderMode.MODIFY_MATERIAL,
        uniforms: {
            u_epoch: { type: Cesium.UniformType.FLOAT, value: 0.0 },
            u_epochs: { type: Cesium.UniformType.FLOAT, value: META.epochs },
            u_rows: { type: Cesium.UniformType.FLOAT, value: META.moving_parcels },
            u_heightScale: { type: Cesium.UniformType.FLOAT, value: heightScaleValue },
            u_wallMode: { type: Cesium.UniformType.FLOAT, value: wallMode },
            u_modelOriginHeight: { type: Cesium.UniformType.FLOAT, value: META.center_height_m },
            u_activeDisplayDatumHeight: { type: Cesium.UniformType.FLOAT, value: activeDatumHeight },
            u_groundHeight: { type: Cesium.UniformType.FLOAT, value: META.phase15.display_tuning.ground_height_m },
            u_reversibleTex: { type: Cesium.UniformType.SAMPLER_2D, value: texturesObj.reversible },
            u_irreversibleTex: { type: Cesium.UniformType.SAMPLER_2D, value: texturesObj.irreversible },
            u_totalTex: { type: Cesium.UniformType.SAMPLER_2D, value: texturesObj.total },
            u_viTex: { type: Cesium.UniformType.SAMPLER_2D, value: texturesObj.vi }
        },
        vertexShaderText: `
          float sampleComponent(sampler2D tex, float rowIndex) {
            float u = (u_epoch + 0.5) / u_epochs;
            float v = (rowIndex + 0.5) / u_rows;
            return texture(tex, vec2(u, v)).r;
          }

          void vertexMain(VertexInput vsInput, inout czm_modelVertexOutput vsOutput) {
            float rowIndex = vsInput.attributes.texCoord_0.x;
            float hasDisp = vsInput.attributes.texCoord_0.y;
            float pistonT = clamp(vsInput.attributes.texCoord_1.x, 0.0, 1.0);
            float z0 = u_activeDisplayDatumHeight - u_modelOriginHeight;
            float z1 = z0;

            if (hasDisp > 0.5 && rowIndex >= 0.0) {
              float irr = sampleComponent(u_irreversibleTex, rowIndex);
              float rev = sampleComponent(u_reversibleTex, rowIndex);
              float total = sampleComponent(u_totalTex, rowIndex);
              if (u_wallMode < 0.5) {
                z0 = u_groundHeight - u_modelOriginHeight;
                z1 = u_activeDisplayDatumHeight - u_modelOriginHeight + irr * u_heightScale;
              } else if (u_wallMode < 1.5) {
                z0 = u_activeDisplayDatumHeight - u_modelOriginHeight;
                z1 = z0 + rev * u_heightScale;
              } else {
                z0 = u_activeDisplayDatumHeight - u_modelOriginHeight + irr * u_heightScale;
                z1 = u_activeDisplayDatumHeight - u_modelOriginHeight + total * u_heightScale;
              }
            }

            vsOutput.positionMC.z = mix(z0, z1, pistonT);
          }
        `,
        fragmentShaderText: `
          float sampleTimeComponentFrag(sampler2D tex, float rowIndex) {
            float u = (u_epoch + 0.5) / u_epochs;
            float v = (rowIndex + 0.5) / u_rows;
            return texture(tex, vec2(u, v)).r;
          }
          float sampleParcelVectorFrag(sampler2D tex, float rowIndex) {
            float v = (rowIndex + 0.5) / u_rows;
            return texture(tex, vec2(0.5, v)).r;
          }
          ${rampShaderText}
          void fragmentMain(FragmentInput fsInput, inout czm_modelMaterial material) {
            float rowIndex = fsInput.attributes.texCoord_0.x;
            float hasDisp = fsInput.attributes.texCoord_0.y;
            if (hasDisp <= 0.5 || rowIndex < 0.0) {
              material.diffuse = vec3(0.0);
              material.alpha = 0.0;
              return;
            }
            if (u_wallMode < 0.5) {
              float irr = sampleTimeComponentFrag(u_irreversibleTex, rowIndex);
              material.diffuse = mix(rampTotal(irr), vec3(0.04, 0.045, 0.055), 0.36);
              material.alpha = 0.42;
            } else {
              float rev = sampleTimeComponentFrag(u_reversibleTex, rowIndex);
              material.diffuse = mix(rampReversible(rev), vec3(0.05, 0.055, 0.065), 0.20);
              material.alpha = 0.52;
            }
          }
        `
    });
}
'''
def make_guides_js() -> str:
    return r'''function modeUsesDatumReference(mode) {
    return mode === "reversible" || mode === "combined";
}

function modeUsesGuides(mode) {
    return mode === "reversible" || mode === "combined";
}

function featureCenterXY(feature) {
    if (Array.isArray(feature?.bbox) && feature.bbox.length >= 4) {
        return [0.5 * (Number(feature.bbox[0]) + Number(feature.bbox[2])), 0.5 * (Number(feature.bbox[1]) + Number(feature.bbox[3]))];
    }
    const ring = feature?.rings?.[0] || [];
    if (!ring.length) return [0.0, 0.0];
    let sx = 0.0, sy = 0.0;
    for (const p of ring) { sx += Number(p[0]); sy += Number(p[1]); }
    return [sx / ring.length, sy / ring.length];
}

function guideEndpointsForFeature(feature) {
    const row = feature.displacement_row_index;
    const origin = META.center_height_m || 0.0;
    const [x, y] = featureCenterXY(feature);
    if (row < 0) return null;

    if (currentMode === "combined") {
        const irr = displacementAt(feature, irreversibleArr) ?? 0.0;
        const total = displacementAt(feature, totalArr) ?? 0.0;
        const boost = META.phase15.display_tuning.irreversible_motion_boost || 1.0;
        const z0 = activeDatumHeight - origin + irr * heightScaleValue * boost;
        const z1 = activeDatumHeight - origin + total * heightScaleValue;
        return { x, y, z0, z1 };
    }

    if (currentMode === "reversible") {
        const rev = displacementAt(feature, reversibleArr) ?? 0.0;
        const z0 = activeDatumHeight - origin;
        const z1 = z0 + rev * heightScaleValue;
        return { x, y, z0, z1 };
    }

    return null;
}

function ensureGuideCollection() {
    if (guideCollection) return guideCollection;
    guideCollection = viewer.scene.primitives.add(new Cesium.PolylineCollection());
    guidePrimitives = [];

    const features = (pickIndex && Array.isArray(pickIndex.features)) ? pickIndex.features : [];
    const material = Cesium.Material.fromType("PolylineDash", {
        color: Cesium.Color.WHITE.withAlpha(0.58),
        dashLength: 12.0
    });

    for (const feature of features) {
        if ((feature.displacement_row_index ?? -1) < 0) continue;
        const end = guideEndpointsForFeature(feature);
        if (!end) continue;
        const line = guideCollection.add({
            positions: [
                localPoint(end.x, end.y, end.z0),
                localPoint(end.x, end.y, end.z1)
            ],
            width: 1.0,
            material
        });
        guidePrimitives.push({ feature, line });
    }
    return guideCollection;
}

function updateGuides() {
    if (!viewer || !pickIndex) return;
    const shouldShow = guideLinesEnabled && modeUsesGuides(currentMode);
    if (!shouldShow) {
        if (guideCollection) guideCollection.show = false;
        return;
    }

    ensureGuideCollection();
    guideCollection.show = true;

    for (const item of guidePrimitives) {
        const end = guideEndpointsForFeature(item.feature);
        if (!end) {
            item.line.show = false;
            continue;
        }
        item.line.show = true;
        item.line.positions = [
            localPoint(end.x, end.y, end.z0),
            localPoint(end.x, end.y, end.z1)
        ];
    }
}'''


def make_outline_js() -> str:
    return r'''function currentSelectedTopZ(feature) {
    if (!feature) return activeDatumHeight - (META.center_height_m || 0.0);
    if (currentMode === "reversible") {
        const rev = displacementAt(feature, reversibleArr) ?? 0.0;
        return activeDatumHeight - (META.center_height_m || 0.0) + rev * heightScaleValue;
    }
    if (currentMode === "total") {
        const total = displacementAt(feature, totalArr) ?? 0.0;
        return activeDatumHeight - (META.center_height_m || 0.0) + total * heightScaleValue;
    }
    if (currentMode === "combined") {
        const total = displacementAt(feature, totalArr) ?? 0.0;
        return activeDatumHeight - (META.center_height_m || 0.0) + total * heightScaleValue;
    }
    const irr = displacementAt(feature, irreversibleArr) ?? 0.0;
    const boost = META.phase15.display_tuning.irreversible_motion_boost || 1.0;
    return activeDatumHeight - (META.center_height_m || 0.0) + irr * heightScaleValue * boost;
}

function drawSelectedOutline(feature) {
    clearSelectedOutline();
    selectedOutline = viewer.scene.primitives.add(new Cesium.PolylineCollection());

    const z = currentSelectedTopZ(feature) + 0.35;
    for (const ring of feature.rings) {
        const positions = ring.map((xy) => localPoint(xy[0], xy[1], z));
        selectedOutline.add({
            positions: positions,
            width: 4,
            material: Cesium.Material.fromType("Color", { color: Cesium.Color.YELLOW.withAlpha(0.98) })
        });
    }
}'''


def make_datum_helpers_js() -> str:
    return r"""function phase16ScalePerExagUnit() {
    return Number(META.phase15?.display_tuning?.height_scale_per_exag_unit_m_per_mm ?? 0.1);
}

function verticalExagToHeightScale(v) {
    return Number(v) * phase16ScalePerExagUnit();
}

function ceilToDatumStep(value, step) {
    const raw = Number(value);
    const s = Number(step);
    if (!Number.isFinite(raw) || raw <= 0 || !Number.isFinite(s) || s <= 0) return 0.0;
    return Math.ceil(raw / s) * s;
}

function modeDatumComponent(mode) {
    if (mode === "reversible") return "reversible";
    if (mode === "total") return "total";
    if (mode === "combined") return "total";
    return "irreversible";
}

function activeDatumForMode(mode, v) {
    const tuning = META.phase15?.display_tuning || {};
    const component = modeDatumComponent(mode);
    const minDatum = Number(tuning.min_display_datum_height_m ?? 10.0);
    const step = Number(tuning.datum_round_step_m ?? 5.0);
    const safety = Number(tuning.safety_clearance_m ?? 5.0);
    const boostMap = tuning.component_motion_boost || {};
    const boost = Number(boostMap[component] ?? (component === "irreversible" ? (tuning.irreversible_motion_boost ?? 1.0) : 1.0));
    const downward = Number(tuning.component_downward_mm?.[component] ?? Math.max(0.0, -Number(tuning.component_min_mm?.[component] ?? 0.0)));
    const scale = verticalExagToHeightScale(v);
    const rawDatum = downward * boost * scale + safety;
    return Math.max(minDatum, ceilToDatumStep(rawDatum, step));
}

function activeDatumForVerticalExag(v) {
    return activeDatumForMode(currentMode || "irreversible", v);
}"""


def make_mode_logic_js() -> str:
    return r'''function activeScaleKeyForMode(mode) {
    if (mode === "irreversible") return "irreversible_velocity";
    return "reversible_displacement";
}

function currentModeLabel(mode) {
    if (mode === "reversible") return "reversible";
    if (mode === "total") return "total";
    if (mode === "combined") return "combined";
    return "irreversible";
}

function updateColorLegendForMode(mode) {
    if (!colorScales || !proto2ColorbarScale || !proto2ColorbarLabels || !proto2ColorbarTitle) return;
    const scaleKey = activeScaleKeyForMode(mode);
    const scale = colorScales.scales[scaleKey];
    if (!scale) return;

    // Phase16 note:
    // total/combined intentionally use cap colour = reversible displacement
    // and wall/body colour = vI. For now the active legend follows the dominant cap.
    proto2ColorbarTitle.textContent = scale.title || scaleKey;
    proto2ColorbarScale.style.background = gradientFromScale(scale);
    proto2ColorbarLabels.innerHTML = "";

    const labels = (scale.legend && Array.isArray(scale.legend.labels)) ? scale.legend.labels : [];
    for (const item of labels) {
        const span = document.createElement("span");
        span.textContent = item.label || String(item.value);
        span.style.left = `${Math.max(0, Math.min(100, Number(item.position_pct || 50))).toFixed(3)}%`;
        proto2ColorbarLabels.appendChild(span);
    }
}

''' + make_guides_js() + r'''

function setModelVisibilityForMode() {
    if (pistonModel) pistonModel.show = (currentMode === "irreversible" || currentMode === "total" || currentMode === "combined");
    if (movingCapModel) movingCapModel.show = (currentMode === "reversible" || currentMode === "combined");
    if (datumCapModel) datumCapModel.show = datumReferenceEnabled && modeUsesDatumReference(currentMode);
    updateGuides();
}

async function applyMode(modeName) {
    currentMode = modeName;
    activeDatumHeight = activeDatumForVerticalExag(verticalExagValue);

    if (currentMode === "irreversible") {
        pistonShader.setUniform("u_component", 2.0);
        pistonShader.setUniform("u_pistonVisualMode", 0.0);
        updateColorLegendForMode("irreversible");
    } else if (currentMode === "reversible") {
        movingCapShader.setUniform("u_component", 1.0);
        datumCapShader.setUniform("u_capRole", 1.0);
        updateColorLegendForMode("reversible");
    } else if (currentMode === "total") {
        pistonShader.setUniform("u_component", 0.0);
        pistonShader.setUniform("u_pistonVisualMode", 1.0);
        updateColorLegendForMode("total");
    } else if (currentMode === "combined") {
        pistonShader.setUniform("u_component", 2.0);
        pistonShader.setUniform("u_pistonVisualMode", 2.0);
        movingCapShader.setUniform("u_component", 0.0);
        datumCapShader.setUniform("u_capRole", 2.0);
        updateColorLegendForMode("combined");
    }

    if (parcelModeSelect) parcelModeSelect.value = currentMode;
    if (parcelModeValue) parcelModeValue.textContent = currentModeLabel(currentMode);
    if (parcelModeHint) {
        if (currentMode === "irreversible") parcelModeHint.textContent = "Piston height = irreversible; cap/wall colour = vI.";
        else if (currentMode === "reversible") parcelModeHint.textContent = "Moving cap = reversible; reference cap and dotted guides show the datum.";
        else if (currentMode === "total") parcelModeHint.textContent = "Piston height = total; cap colour = reversible; wall colour = vI.";
        else parcelModeHint.textContent = "Combined: irreversible body plus breathing cap riding on the drowning floor.";
    }

    setModelVisibilityForMode();
    setShaderTimeAndScale();
}'''



def make_phase16_runtime_safety_shim_js() -> str:
    return r"""// ---- Phase16C runtime safety shim ----
function phase16ScalePerExagUnit() {
    return Number(META.phase15?.display_tuning?.height_scale_per_exag_unit_m_per_mm ?? 0.1);
}
function verticalExagToHeightScale(v) { return Number(v) * phase16ScalePerExagUnit(); }
function ceilToDatumStep(value, step) {
    const raw = Number(value), s = Number(step);
    if (!Number.isFinite(raw) || raw <= 0 || !Number.isFinite(s) || s <= 0) return 0.0;
    return Math.ceil(raw / s) * s;
}
function modeDatumComponent(mode) {
    if (mode === "reversible") return "reversible";
    if (mode === "total") return "total";
    if (mode === "combined") return "total";
    return "irreversible";
}
function activeDatumForMode(mode, v) {
    const tuning = META.phase15?.display_tuning || {};
    const component = modeDatumComponent(mode);
    const minDatum = Number(tuning.min_display_datum_height_m ?? 10.0);
    const step = Number(tuning.datum_round_step_m ?? 5.0);
    const safety = Number(tuning.safety_clearance_m ?? 5.0);
    // Phase16C: datum follows actual displacement geometry, not the old standalone visibility boost.
    const downward = Number(tuning.component_downward_mm?.[component] ?? Math.max(0.0, -Number(tuning.component_min_mm?.[component] ?? 0.0)));
    return Math.max(minDatum, ceilToDatumStep(downward * verticalExagToHeightScale(v) + safety, step));
}
function activeDatumForVerticalExag(v) { return activeDatumForMode(currentMode || "irreversible", v); }

function phase16CIrreversibleColorMode() {
    const el = document.getElementById("irreversibleColorSelect");
    return el ? el.value : "velocity";
}
function activeScaleKeyForMode(mode) {
    if (mode === "irreversible") return phase16CIrreversibleColorMode() === "displacement" ? "total_displacement" : "irreversible_velocity";
    return "reversible_displacement";
}
function currentModeLabel(mode) {
    if (mode === "reversible") return "reversible";
    if (mode === "total") return "total";
    if (mode === "combined") return "combined";
    return "irreversible";
}
function updateColorLegendForMode(mode) {
    if (!colorScales || !proto2ColorbarScale || !proto2ColorbarLabels || !proto2ColorbarTitle) return;
    const scaleKey = activeScaleKeyForMode(mode);
    const scale = colorScales.scales?.[scaleKey];
    if (!scale) return;
    if (mode === "irreversible" && phase16CIrreversibleColorMode() === "displacement") {
        proto2ColorbarTitle.textContent = "Irreversible displacement [mm]";
    } else if (mode === "combined") {
        proto2ColorbarTitle.textContent = "Combined: irreversible cap + reversible layer [mm]";
    } else {
        proto2ColorbarTitle.textContent = scale.title || scaleKey;
    }
    proto2ColorbarScale.style.background = gradientFromScale(scale);
    proto2ColorbarLabels.innerHTML = "";
    const labels = (scale.legend && Array.isArray(scale.legend.labels)) ? scale.legend.labels : [];
    for (const item of labels) {
        const span = document.createElement("span");
        span.textContent = item.label || String(item.value);
        span.style.left = `${Math.max(0, Math.min(100, Number(item.position_pct || 50))).toFixed(3)}%`;
        proto2ColorbarLabels.appendChild(span);
    }
}
function modeUsesDatumReference(mode) { return mode === "reversible" || mode === "combined"; }
function modeUsesBreathingWalls(mode) { return mode === "reversible" || mode === "combined"; }
function updateGuides() {
    // Phase16C: the old dotted guide system is replaced by a GPU wall-only mesh.
    if (breathingWallModel) breathingWallModel.show = !!(guideLinesEnabled && modeUsesBreathingWalls(currentMode));
}
function setModelVisibilityForMode() {
    if (pistonModel) pistonModel.show = (currentMode === "irreversible" || currentMode === "total");
    if (bodyWallModel) bodyWallModel.show = (currentMode === "combined");
    if (breathingWallModel) breathingWallModel.show = !!(guideLinesEnabled && modeUsesBreathingWalls(currentMode));
    if (movingCapModel) movingCapModel.show = (currentMode === "reversible" || currentMode === "combined");
    // Keep this model visible in reversible/combined because blank parcels live here even when datum reference is off.
    if (datumCapModel) datumCapModel.show = modeUsesDatumReference(currentMode);
}
function setShaderTimeAndScale() {
    heightScaleValue = verticalExagToHeightScale(verticalExagValue);
    activeDatumHeight = activeDatumForVerticalExag(verticalExagValue);
    const shaderList = [pistonShader, movingCapShader, datumCapShader, bodyWallShader, breathingWallShader, topShader, bottomShader].filter(Boolean);
    for (const shader of shaderList) {
        try {
            shader.setUniform("u_epoch", epoch);
            shader.setUniform("u_heightScale", heightScaleValue);
            shader.setUniform("u_activeDisplayDatumHeight", activeDatumHeight);
            shader.setUniform("u_datumReferenceEnabled", datumReferenceEnabled ? 1.0 : 0.0);
            shader.setUniform("u_irreversibleColorMode", phase16CIrreversibleColorMode() === "displacement" ? 1.0 : 0.0);
        } catch (e) {}
    }
    updateGuides();
    if (selectedFeature && typeof drawSelectedOutline === "function") drawSelectedOutline(selectedFeature);
    if (typeof updateEpochUi === "function") updateEpochUi();
    viewer?.scene?.requestRender();
}
async function applyMode(modeName) {
    currentMode = modeName;
    activeDatumHeight = activeDatumForVerticalExag(verticalExagValue);
    if (currentMode === "irreversible") {
        pistonShader?.setUniform("u_component", 2.0);
        pistonShader?.setUniform("u_pistonVisualMode", 0.0);
        pistonShader?.setUniform("u_irreversibleColorMode", phase16CIrreversibleColorMode() === "displacement" ? 1.0 : 0.0);
        updateColorLegendForMode("irreversible");
    } else if (currentMode === "reversible") {
        movingCapShader?.setUniform("u_component", 1.0);
        datumCapShader?.setUniform("u_capRole", 1.0);
        breathingWallShader?.setUniform("u_wallMode", 1.0);
        updateColorLegendForMode("reversible");
    } else if (currentMode === "total") {
        pistonShader?.setUniform("u_component", 0.0);
        pistonShader?.setUniform("u_pistonVisualMode", 1.0);
        updateColorLegendForMode("total");
    } else if (currentMode === "combined") {
        bodyWallShader?.setUniform("u_wallMode", 0.0);
        breathingWallShader?.setUniform("u_wallMode", 2.0);
        movingCapShader?.setUniform("u_component", 0.0);
        datumCapShader?.setUniform("u_capRole", 2.0);
        updateColorLegendForMode("combined");
    }
    if (parcelModeSelect) parcelModeSelect.value = currentMode;
    if (parcelModeValue) parcelModeValue.textContent = currentModeLabel(currentMode);
    if (parcelModeHint) {
        if (currentMode === "irreversible") parcelModeHint.textContent = "Piston height = irreversible displacement; colour = velocity or irreversible displacement.";
        else if (currentMode === "reversible") parcelModeHint.textContent = "Moving cap = reversible; reference/blank cap marks datum; breathing walls show above/below datum.";
        else if (currentMode === "total") parcelModeHint.textContent = "Piston height = total; cap colour = reversible; wall colour = vI.";
        else parcelModeHint.textContent = "Combined: irreversible displacement body/cap plus reversible layer; moving cap = total.";
    }
    setModelVisibilityForMode();
    setShaderTimeAndScale();
}
function phase16CWireControls() {
    const colorSel = document.getElementById("irreversibleColorSelect");
    if (colorSel && !colorSel.dataset.phase16cBound) {
        colorSel.dataset.phase16cBound = "1";
        colorSel.addEventListener("change", () => {
            if (currentMode === "irreversible") applyMode("irreversible");
            else updateColorLegendForMode(currentMode);
            setShaderTimeAndScale();
        });
    }
    const wallToggle = document.getElementById("verticalGuidesToggle");
    if (wallToggle) {
        const label = wallToggle.closest("label")?.querySelector("span");
        if (label) label.textContent = "Breathing walls";
    }
}
setTimeout(phase16CWireControls, 0);
// ---- /Phase16C runtime safety shim ----"""

def main() -> None:
    OUTPUT_CESIUM.mkdir(parents=True, exist_ok=True)
    RUN_RECORDS.mkdir(parents=True, exist_ok=True)

    print("\n=== PROTO2 PHASE 16C: MULTIMODE DEFORMATION VIEWER CLEANUP ===")
    print(f"Project root: {PROJECT_ROOT}")

    require(SOURCE_HTML, "Phase15 irreversible piston viewer HTML")
    require(DISPLAY_TUNING_JSON, "display tuning")
    require(PISTON_SUMMARY_JSON, "piston summary")
    require(PISTON_GLB, "piston GLB")
    require(PHASE14_CAP_GLB, "Phase14 cap GLB")
    require(ASSET_DIR, "phase15 asset directory")

    html = SOURCE_HTML.read_text(encoding="utf-8")
    meta = extract_meta_from_html(html)
    display_tuning = json.loads(DISPLAY_TUNING_JSON.read_text(encoding="utf-8"))
    piston_summary = json.loads(PISTON_SUMMARY_JSON.read_text(encoding="utf-8"))

    shutil.copy2(PHASE14_CAP_GLB, CAP_GLB_COPY)
    wall_split_summary = build_wall_only_glb_from_piston(PISTON_GLB, WALL_ONLY_GLB)

    meta["product"] = "proto2_m1_multimode_deformation_viewer_16c"
    meta["phase16"] = {
        "product": "proto2_m1_multimode_deformation_viewer_16c",
        "asset_base": "phase15_piston_assets/",
        "source": "Phase15 piston viewer + Phase14 cap GLB + Phase16C wall-only GLB",
        "modes": {
            "irreversible": "piston height = irreversible displacement; color = vI or irreversible displacement",
            "reversible": "moving cap height/color = reversible; transparent datum cap and optional dotted guides",
            "total": "piston height = total; cap color = reversible; wall color = vI",
            "combined": "irreversible displacement body/cap plus reversible layer; moving cap = total"
        },
        "copied_cap_glb": str(CAP_GLB_COPY),
        "wall_only_glb": str(WALL_ONLY_GLB),
    }
    meta_json = json.dumps(meta, separators=(",", ":"))
    html = replace_regex(html, r"const META = .*?;\n", f"const META = {meta_json};\n", "META JSON")

    html = html.replace("Proto2 M1 Irreversible Piston Viewer", "Proto2 M1 Multimode Deformation Viewer")
    html = html.replace("Proto2 irreversible piston viewer loaded", "Proto2 multimode deformation viewer loaded")
    html = html.replace("<title>Proto2 M1 Irreversible Piston Viewer</title>", "<title>Proto2 M1 Multimode Deformation Viewer</title>")

    html = replace_one(
        html,
        'const GLB_URL = ASSET_BASE + "proto2_irreversible_piston_mesh.glb";',
        'const PISTON_GLB_URL = ASSET_BASE + "proto2_irreversible_piston_mesh.glb";\nconst CAP_GLB_URL = ASSET_BASE + "proto2_animated_parcel_mesh.glb";\nconst WALL_GLB_URL = ASSET_BASE + "proto2_parcel_wall_mesh_blend.glb";',
        "GLB constants",
    )

    css_patch = r'''
        /* Phase16 multimode UI additions */
        .modeToggleCard {
            display: grid;
            grid-template-columns: 1fr;
            gap: 6px;
        }
        .drawerCheckRow {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 8px;
            color: var(--ui-text-muted);
            font-size: 10.4px;
            line-height: 1.1;
        }
        .drawerCheckRow input {
            accent-color: var(--ui-accent-2);
        }
        .drawerCheckRow select {
            width: 112px;
            border: 1px solid rgba(255,255,255,0.16);
            border-radius: 999px;
            background: rgba(2, 6, 23, 0.72);
            color: var(--ui-text);
            font-size: 10px;
            padding: 3px 7px;
            outline: none;
        }
'''
    html = replace_one(html, "</style>", css_patch + "\n    </style>", "phase16 CSS")

    html = replace_regex(
        html,
        r'<select id="parcelModeSelect">.*?</select>',
        '<select id="parcelModeSelect">\n'
        '                            <option value="irreversible" selected>irreversible</option>\n'
        '                            <option value="reversible">reversible</option>\n'
        '                            <option value="total">total</option>\n'
        '                            <option value="combined">combined</option>\n'
        '                        </select>',
        "mode select",
    )
    html = replace_regex(html, r'<span id="parcelModeValue">.*?</span>', '<span id="parcelModeValue">irreversible</span>', "mode value", flags=0)
    html = replace_regex(
        html,
        r'<div class="drawerSubhint" id="parcelModeHint">.*?</div>',
        '<div class="drawerSubhint" id="parcelModeHint">Mode-specific geometry: piston, breathing cap, total piston, or combined decomposition.</div>',
        "mode hint",
    )

    html = replace_regex_func(
        html,
        r'(<div class="drawerSubhint" id="parcelModeHint">.*?</div>\s*</div>)',
        lambda m: m.group(1) + '\n'
        '                    <div class="drawerControlCard modeToggleCard">\n'
        '                        <label class="drawerCheckRow"><span>Irreversible color</span><select id="irreversibleColorSelect"><option value="velocity" selected>velocity</option><option value="displacement">displacement</option></select></label>\n'
        '                        <label class="drawerCheckRow"><span>Datum reference</span><input id="datumReferenceToggle" type="checkbox" checked /></label>\n'
        '                        <label class="drawerCheckRow"><span>Breathing walls</span><input id="verticalGuidesToggle" type="checkbox" checked /></label>\n'
        '                        <div class="drawerSubhint">Reference cap and breathing walls are used by reversible/combined modes.</div>\n'
        '                    </div>',
        "datum/guide toggles",
    )

    html = replace_one(
        html,
        "let topModel = null;\nlet selectedOutline = null;",
        "let topModel = null;\nlet pistonModel = null;\nlet movingCapModel = null;\nlet datumCapModel = null;\nlet pistonShader = null;\nlet movingCapShader = null;\nlet datumCapShader = null;\nlet guideCollection = null;\nlet guidePrimitives = [];\nlet bodyWallModel = null;\nlet breathingWallModel = null;\nlet bodyWallShader = null;\nlet breathingWallShader = null;\nlet datumReferenceEnabled = true;\nlet guideLinesEnabled = true;\nlet selectedOutline = null;",
        "model globals",
    )
    html = replace_one(
        html,
        'const parcelModeHint = document.getElementById("parcelModeHint");',
        'const parcelModeHint = document.getElementById("parcelModeHint");\nconst datumReferenceToggle = document.getElementById("datumReferenceToggle");\nconst verticalGuidesToggle = document.getElementById("verticalGuidesToggle");',
        "toggle DOM refs",
    )

    html = replace_regex(
        html,
        r'function makeCustomShader\(componentValue, texturesObj, floorMode = false\) \{.*?\n\}\n\nasync function loadModel',
        make_piston_shader_js() + "\n\nasync function loadModel",
        "piston/cap shader functions",
    )

    html = replace_regex(
        html,
        r'async function loadModel\(customShader\) \{.*?\n\}\n\nfunction localPoint',
        r'''async function loadModel(customShader, url = PISTON_GLB_URL) {
    if (Cesium.Model && Cesium.Model.fromGltfAsync) {
        const model = await Cesium.Model.fromGltfAsync({
            url,
            modelMatrix,
            customShader,
            allowPicking: true,
            asynchronous: true
        });
        viewer.scene.primitives.add(model);
        return model;
    }
    const model = Cesium.Model.fromGltf({
        url,
        modelMatrix,
        customShader,
        allowPicking: true,
        asynchronous: true
    });
    viewer.scene.primitives.add(model);
    return model;
}

function localPoint''',
        "url-aware loadModel",
    )

    html = replace_regex(
        html,
        r'function verticalExagToHeightScale\(v\) \{.*?function activeDatumForVerticalExag\(v\) \{.*?\n\}',
        r'''function verticalExagToHeightScale(v) {
    return Number(v) * SCALE_PER_EXAG_UNIT;
}

function ceilToDatumStep(value, step) {
    const raw = Number(value);
    const s = Number(step);
    if (!Number.isFinite(raw) || raw <= 0 || !Number.isFinite(s) || s <= 0) return 0.0;
    return Math.ceil(raw / s) * s;
}

function modeDatumComponent(mode) {
    if (mode === "reversible") return "reversible";
    if (mode === "total") return "total";
    if (mode === "combined") return "total";
    return "irreversible";
}

function activeDatumForMode(mode, v) {
    const tuning = META.phase15?.display_tuning || {};
    const component = modeDatumComponent(mode);
    const minDatum = Number(tuning.min_display_datum_height_m ?? 10.0);
    const step = Number(tuning.datum_round_step_m ?? 5.0);
    const safety = Number(tuning.safety_clearance_m ?? 5.0);
    const boostMap = tuning.component_motion_boost || {};
    const boost = Number(boostMap[component] ?? (component === "irreversible" ? (tuning.irreversible_motion_boost ?? 1.0) : 1.0));
    const downward = Number(tuning.component_downward_mm?.[component] ?? Math.max(0.0, -Number(tuning.component_min_mm?.[component] ?? 0.0)));
    const scale = verticalExagToHeightScale(v);
    const rawDatum = downward * boost * scale + safety;
    return Math.max(minDatum, ceilToDatumStep(rawDatum, step));
}

function activeDatumForVerticalExag(v) {
    return activeDatumForMode(currentMode || "irreversible", v);
}''',
        "mode-aware active datum",
    )

    html = replace_regex(
        html,
        r'let heightScaleValue = .*?;\s*// .*?\nlet activeDatumHeight = .*?;\s*// .*?\n',
        'const SCALE_PER_EXAG_UNIT = META.phase15.display_tuning.height_scale_per_exag_unit_m_per_mm || 0.1;\n'
        # Do not call activeDatumForVerticalExag here: currentMode is declared just after this block in the source HTML.
        # applyMode()/setShaderTimeAndScale() will compute the real values after all globals exist.
        'let heightScaleValue = 0.0;\n'
        'let activeDatumHeight = 0.0;\n',
        "height scale globals",
        flags=re.S,
    )

    html = replace_regex(
        html,
        r'function setShaderTimeAndScale\(\) \{.*?viewer\?\.scene\?\.requestRender\(\);\s*\}',
        r'''function setShaderTimeAndScale() {
    heightScaleValue = verticalExagToHeightScale(verticalExagValue);
    activeDatumHeight = activeDatumForVerticalExag(verticalExagValue);

    const shaderList = [pistonShader, movingCapShader, datumCapShader, topShader, bottomShader].filter(Boolean);
    for (const shader of shaderList) {
        try {
            shader.setUniform("u_epoch", epoch);
            shader.setUniform("u_heightScale", heightScaleValue);
            shader.setUniform("u_activeDisplayDatumHeight", activeDatumHeight);
        } catch (e) {}
    }

    updateGuides();
    if (selectedFeature) drawSelectedOutline(selectedFeature);
    updateEpochUi();
    viewer?.scene?.requestRender();
}''',
        "multi-shader setShaderTimeAndScale",
    )

    html = replace_regex(
        html,
        r'function activeScaleKeyForMode\(mode\) \{.*?function setEpochIndex',
        make_datum_helpers_js() + "\n\n" + make_mode_logic_js() + "\n\nfunction setEpochIndex",
        "replace old mode logic with multimode logic and datum helpers",
    )

    html = replace_regex(
        html,
        r'function drawSelectedOutline\(feature\) \{.*?\n\}\n\nfunction rumInfoMetric',
        make_outline_js() + "\n\nfunction rumInfoMetric",
        "mode-aware outline",
    )

    multimode_loading_block = r'''pistonShader = makePistonShader(2.0, textures, 0.0);
    movingCapShader = makeCapShader(1.0, textures, 0.0);
    datumCapShader = makeCapShader(1.0, textures, 1.0);
    bodyWallShader = makeWallShader(0.0, textures);
    breathingWallShader = makeWallShader(1.0, textures);
    topShader = pistonShader;
    bottomShader = null;

    setStatus("loading piston, cap, and breathing wall meshes...");
    pistonModel = await loadModel(pistonShader, PISTON_GLB_URL);
    topModel = pistonModel;
    movingCapModel = await loadModel(movingCapShader, CAP_GLB_URL);
    datumCapModel = await loadModel(datumCapShader, CAP_GLB_URL);
    bodyWallModel = await loadModel(bodyWallShader, WALL_GLB_URL);
    breathingWallModel = await loadModel(breathingWallShader, WALL_GLB_URL);
    movingCapModel.show = false;
    datumCapModel.show = false;
    bodyWallModel.show = false;
    breathingWallModel.show = false;'''

    # Source can be either the plain Phase15 HTML (makeCustomShader startup block)
    # or a previously patched/retried Phase16-like block. Try both, once.
    startup_patterns = [
        r'topShader = makeCustomShader\(0\.0, textures\);\s*bottomShader = makeCustomShader\(2\.0, textures, true\);\s*setStatus\("loading parcel mesh\.\.\."\);\s*topModel = await loadModel\(topShader\);',
        r'topShader = makePistonShader\(.*?\);\s*bottomShader = null;.*?topModel = pistonModel;.*?movingCapModel\.show = false;\s*datumCapModel\.show = false;',
    ]
    replaced_startup = False
    for pat in startup_patterns:
        new_html, n = re.subn(pat, lambda _m: multimode_loading_block, html, count=1, flags=re.S)
        if n == 1:
            html = new_html
            replaced_startup = True
            break
    if not replaced_startup:
        fail("Could not replace startup model loading block for Phase16A")

    html = replace_regex_func(
        html,
        r'(parcelModeSelect\.addEventListener\("change".*?\n    \}\);\n)',
        lambda m: m.group(1) + r'''
    if (datumReferenceToggle) {
        datumReferenceToggle.addEventListener("change", () => {
            datumReferenceEnabled = datumReferenceToggle.checked;
            setModelVisibilityForMode();
        });
    }
    if (verticalGuidesToggle) {
        verticalGuidesToggle.addEventListener("change", () => {
            guideLinesEnabled = verticalGuidesToggle.checked;
            updateGuides();
        });
    }
''',
        "toggle event listeners",
    )

    html = html.replace('setStatus("Proto2 irreversible piston viewer loaded");', 'setStatus("Proto2 multimode deformation viewer loaded");')

    # Runtime safety shim: define/override all phase16 mode helpers in one known place.
    # This avoids browser-side ReferenceError surprises when source Phase15 HTML layout varies.
    html = replace_one(
        html,
        "async function main()",
        make_phase16_runtime_safety_shim_js() + "\n\nasync function main()",
        "phase16 runtime safety shim",
    )

    required_js_functions = [
        "activeDatumForVerticalExag",
        "updateColorLegendForMode",
        "setShaderTimeAndScale",
        "setModelVisibilityForMode",
        "updateGuides",
        "applyMode",
        "makeWallShader",
        "phase16CWireControls",
    ]
    missing_functions = [name for name in required_js_functions if f"function {name}" not in html]
    if missing_functions:
        fail("Generated HTML missing required JS functions after safety shim: " + ", ".join(missing_functions))

    HTML_OUT.write_text(html, encoding="utf-8")

    summary = dict(meta)
    summary["outputs"] = {
        "html": str(HTML_OUT),
        "summary": str(SUMMARY_OUT),
        "asset_dir": str(ASSET_DIR),
        "piston_glb": str(PISTON_GLB),
        "cap_glb": str(CAP_GLB_COPY),
        "wall_glb": str(WALL_ONLY_GLB),
    }
    summary["display_tuning"] = display_tuning
    summary["piston_summary"] = {
        "vertices": piston_summary.get("combined", {}).get("vertices"),
        "triangles": piston_summary.get("combined", {}).get("triangles"),
        "glb_size_mb": piston_summary.get("combined", {}).get("glb_size_mb"),
    }
    summary["wall_split_summary"] = wall_split_summary

    write_json(SUMMARY_OUT, summary)
    write_json(REPORT_JSON_OUT, summary)
    REPORT_TXT_OUT.write_text(
        "PROTO2 PHASE 16C: MULTIMODE DEFORMATION VIEWER CLEANUP\n"
        f"Project root: {PROJECT_ROOT}\n"
        f"HTML: {HTML_OUT}\n"
        f"Piston GLB: {PISTON_GLB}\n"
        f"Cap GLB copy: {CAP_GLB_COPY}\n"
        f"Wall-only GLB: {WALL_ONLY_GLB}\n"
        f"Modes: irreversible, reversible, total, combined (16C cleanup)\n",
        encoding="utf-8",
    )

    ok(f"copied cap GLB: {CAP_GLB_COPY}")
    ok(f"built wall-only GLB: {WALL_ONLY_GLB}")
    ok(f"wrote {HTML_OUT}")
    ok(f"wrote {SUMMARY_OUT}")
    print("\n=== PHASE 16C RESULT: PASS ===")


if __name__ == "__main__":
    main()
