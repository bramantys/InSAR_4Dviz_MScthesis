from __future__ import annotations

from pathlib import Path
import json
import re
import struct
import sys
from typing import Any, Dict, List, Tuple


from _proto2_config import load_project_config, output_cesium_dir, output_data_dir, stage_records_dir


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = load_project_config(PROJECT_ROOT)
OUTPUT_CESIUM = output_cesium_dir(PROJECT_ROOT, CONFIG)
RUN_RECORDS = stage_records_dir(PROJECT_ROOT, CONFIG)
ASSET_DIR = OUTPUT_CESIUM / "phase15_piston_assets"

SOURCE_HTML = OUTPUT_CESIUM / "proto2_m1_multimode_deformation_viewer_16d.html"
SOURCE_SUMMARY = OUTPUT_CESIUM / "proto2_m1_multimode_deformation_viewer_16d_summary.json"

CAP_GLB = ASSET_DIR / "proto2_animated_parcel_mesh.glb"
OPAQUE_DATUM_CAP_GLB = ASSET_DIR / "proto2_animated_parcel_mesh_opaque_datum.glb"

HTML_OUT = OUTPUT_CESIUM / "proto2_m1_multimode_deformation_viewer_16e.html"
SUMMARY_OUT = OUTPUT_CESIUM / "proto2_m1_multimode_deformation_viewer_16e_summary.json"
REPORT_TXT_OUT = RUN_RECORDS / "phase16e_combined_cap_opacity_report.txt"
REPORT_JSON_OUT = RUN_RECORDS / "phase16e_combined_cap_opacity_report.json"


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


def pad4(data: bytes, pad_byte: bytes = b" ") -> bytes:
    extra = (-len(data)) % 4
    return data + pad_byte * extra


def read_glb(path: Path) -> Tuple[Dict[str, Any], List[Tuple[int, bytes]]]:
    raw = path.read_bytes()
    if len(raw) < 12:
        fail(f"GLB too short: {path}")
    magic, version, total_len = struct.unpack_from("<III", raw, 0)
    if magic != 0x46546C67 or version != 2:
        fail(f"Not a glTF 2.0 GLB: {path}")
    if total_len != len(raw):
        fail(f"GLB length mismatch: header={total_len}, actual={len(raw)}")

    offset = 12
    chunks: List[Tuple[int, bytes]] = []
    json_obj = None
    while offset < len(raw):
        if offset + 8 > len(raw):
            fail(f"Malformed GLB chunk header at offset {offset}")
        chunk_len, chunk_type = struct.unpack_from("<II", raw, offset)
        offset += 8
        chunk_data = raw[offset:offset + chunk_len]
        offset += chunk_len
        chunks.append((chunk_type, chunk_data))
        if chunk_type == 0x4E4F534A:  # JSON
            json_obj = json.loads(chunk_data.decode("utf-8").rstrip(" \t\r\n\0"))
    if json_obj is None:
        fail(f"No JSON chunk found in {path}")
    return json_obj, chunks # type: ignore


def write_glb(path: Path, gltf: Dict[str, Any], chunks: List[Tuple[int, bytes]]) -> None:
    json_bytes = pad4(json.dumps(gltf, separators=(",", ":"), ensure_ascii=False).encode("utf-8"), b" ")
    out_chunks: List[Tuple[int, bytes]] = [(0x4E4F534A, json_bytes)]
    for chunk_type, chunk_data in chunks:
        if chunk_type == 0x4E4F534A:
            continue
        pad_byte = b"\0" if chunk_type == 0x004E4942 else b" "
        out_chunks.append((chunk_type, pad4(chunk_data, pad_byte)))

    total_len = 12 + sum(8 + len(data) for _typ, data in out_chunks)
    out = bytearray()
    out += struct.pack("<III", 0x46546C67, 2, total_len)
    for chunk_type, data in out_chunks:
        out += struct.pack("<II", len(data), chunk_type)
        out += data
    path.write_bytes(bytes(out))


def build_opaque_datum_cap_glb(src: Path, dst: Path) -> Dict[str, Any]:
    gltf, chunks = read_glb(src)

    material_count = 0
    for mat in gltf.get("materials", []):
        material_count += 1
        mat["alphaMode"] = "OPAQUE"
        mat.pop("alphaCutoff", None)
        pbr = mat.setdefault("pbrMetallicRoughness", {})
        base = pbr.get("baseColorFactor")
        if not isinstance(base, list) or len(base) < 4:
            base = [1.0, 1.0, 1.0, 1.0]
        else:
            base = list(base[:4])
            base[3] = 1.0
        pbr["baseColorFactor"] = base
        # Make sure no old material extensions force transparency.
        extensions = mat.get("extensions")
        if isinstance(extensions, dict):
            extensions.pop("KHR_materials_transmission", None)
            extensions.pop("KHR_materials_volume", None)
            if not extensions:
                mat.pop("extensions", None)

    write_glb(dst, gltf, chunks)
    return {
        "source": str(src),
        "output": str(dst),
        "materials_forced_opaque": material_count,
        "output_size_mb": dst.stat().st_size / (1024 * 1024),
    }


def scrub_meta_backslashes(html: str) -> str:
    start = html.find("const META = ")
    if start < 0:
        fail("Could not find const META")
    end = html.find(";\n", start)
    if end < 0:
        end = html.find(";</script>", start)
        if end < 0:
            fail("Could not find end of const META")
        end += 1
    else:
        end += 2

    block = html[start:end].replace("\\", "/")
    if "\\" in block:
        fail("META block still contains a backslash after scrub")
    return html[:start] + block + html[end:]


def meta_backslash_count(html: str) -> int:
    start = html.find("const META = ")
    if start < 0:
        return -1
    end = html.find(";\n", start)
    if end < 0:
        end = html.find(";</script>", start)
        if end < 0:
            return -1
        end += 1
    else:
        end += 2
    return html[start:end].count("\\")


def patch_html(html: str) -> str:
    html = scrub_meta_backslashes(html)

    # Add a dedicated opaque cap URL. Moving caps keep using the original cap GLB,
    # datum/combined irreversible caps use the opaque copy.
    if "const DATUM_CAP_GLB_URL" not in html:
        target = 'const CAP_GLB_URL = ASSET_BASE + "proto2_animated_parcel_mesh.glb";'
        if target not in html:
            fail("Could not find CAP_GLB_URL constant")
        html = html.replace(
            target,
            target + '\nconst DATUM_CAP_GLB_URL = ASSET_BASE + "proto2_animated_parcel_mesh_opaque_datum.glb";',
            1,
        )

    # Route datumCapModel only to the opaque cap GLB.
    old = "datumCapModel = await loadModel(datumCapShader, CAP_GLB_URL);"
    new = "datumCapModel = await loadModel(datumCapShader, DATUM_CAP_GLB_URL);"
    if old in html:
        html = html.replace(old, new, 1)
    elif new not in html:
        fail("Could not find datumCapModel load statement")

    # Strengthen the shader path too. This is redundant with the opaque GLB, but harmless.
    html = html.replace(
        "Combined irreversible cap: always visible, opaque, displacement-coloured.",
        "Combined irreversible cap: always visible, hard-opaque, displacement-coloured.",
    )
    html = re.sub(
        r"(if \(u_capRole > 1\.5\) \{.*?material\.alpha\s*=\s*)[0-9.]+(\s*;\s*return;\s*\})",
        r"\g<1>1.0\2",
        html,
        count=1,
        flags=re.S,
    )

    html = html.replace("proto2_m1_multimode_deformation_viewer_16d", "proto2_m1_multimode_deformation_viewer_16e")
    html = html.replace("Phase16D", "Phase16E")
    html = html.replace("PHASE 16D", "PHASE 16E")

    html = scrub_meta_backslashes(html)
    if meta_backslash_count(html) != 0:
        fail(f"META still has {meta_backslash_count(html)} backslashes")
    return html


def main() -> None:
    OUTPUT_CESIUM.mkdir(parents=True, exist_ok=True)
    RUN_RECORDS.mkdir(parents=True, exist_ok=True)

    print("\n=== PROTO2 PHASE 16E: COMBINED OPAQUE IRREVERSIBLE CAP ===")
    print(f"Project root: {PROJECT_ROOT}")

    require(SOURCE_HTML, "Phase16D HTML")
    require(CAP_GLB, "Phase16 cap GLB")

    opaque_summary = build_opaque_datum_cap_glb(CAP_GLB, OPAQUE_DATUM_CAP_GLB)

    html = SOURCE_HTML.read_text(encoding="utf-8")
    html = patch_html(html)
    HTML_OUT.write_text(html, encoding="utf-8")

    source_summary: Dict[str, Any] = {}
    if SOURCE_SUMMARY.exists():
        try:
            source_summary = json.loads(SOURCE_SUMMARY.read_text(encoding="utf-8"))
        except Exception:
            source_summary = {}

    summary = {
        "product": "proto2_m1_multimode_deformation_viewer_16e",
        "source_html": str(SOURCE_HTML),
        "output_html": str(HTML_OUT),
        "source_summary": str(SOURCE_SUMMARY) if SOURCE_SUMMARY.exists() else None,
        "inherited_product": source_summary.get("product"),
        "opaque_datum_cap_glb": opaque_summary,
        "changes": [
            "Dedicated opaque cap GLB for datum/combined irreversible cap.",
            "datumCapModel now loads DATUM_CAP_GLB_URL instead of CAP_GLB_URL.",
            "META block hard-scrubbed to contain zero backslashes, avoiding stale octal diagnostics.",
        ],
        "meta_backslashes_remaining": meta_backslash_count(html),
    }
    write_json(SUMMARY_OUT, summary)
    write_json(REPORT_JSON_OUT, summary)
    REPORT_TXT_OUT.write_text(
        "PROTO2 PHASE 16E: COMBINED OPAQUE IRREVERSIBLE CAP\n"
        f"Project root: {PROJECT_ROOT}\n"
        f"Source HTML: {SOURCE_HTML}\n"
        f"Output HTML: {HTML_OUT}\n"
        f"Opaque datum cap GLB: {OPAQUE_DATUM_CAP_GLB}\n"
        f"META backslashes remaining: {meta_backslash_count(html)}\n",
        encoding="utf-8",
    )

    ok(f"wrote {OPAQUE_DATUM_CAP_GLB}")
    ok(f"wrote {HTML_OUT}")
    ok(f"wrote {SUMMARY_OUT}")
    ok(f"META backslash check: {meta_backslash_count(html)} remaining")
    print("\n=== PHASE 16E RESULT: PASS ===")


if __name__ == "__main__":
    main()
