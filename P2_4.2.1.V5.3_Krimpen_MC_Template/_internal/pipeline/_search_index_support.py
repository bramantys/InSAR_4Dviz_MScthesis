#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parcel search support module

Runtime search support — parcel-ID search bar.

Input:
  _internal/data_pipeline/work/geometry_support/proto2_m1_multimode_deformation_viewer_17_fixed7.html

Output:
  _internal/data_pipeline/work/geometry_support/proto2_m1_multimode_deformation_viewer_18_search.html
  _internal/data_pipeline/work/geometry_support/phase18_search_assets/parcel_search_index.json
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import pandas as pd
except Exception:
    pd = None


from _proto2_config import load_project_config, output_cesium_dir, output_data_dir, stage_records_dir


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = load_project_config(PROJECT_ROOT)
OUTPUT_CESIUM = output_cesium_dir(PROJECT_ROOT, CONFIG)
OUTPUT_DATA = output_data_dir(PROJECT_ROOT, CONFIG)
RUN_RECORDS = stage_records_dir(PROJECT_ROOT, CONFIG)

SOURCE_HTML = OUTPUT_CESIUM / "proto2_m1_multimode_deformation_viewer_17_fixed7.html"
SOURCE_SUMMARY = OUTPUT_CESIUM / "proto2_m1_multimode_deformation_viewer_17_fixed7_summary.json"

SEARCH_ASSET_DIR = OUTPUT_CESIUM / "phase18_search_assets"
SEARCH_INDEX_OUT = SEARCH_ASSET_DIR / "parcel_search_index.json"

HTML_OUT = OUTPUT_CESIUM / "proto2_m1_multimode_deformation_viewer_18_search.html"
SUMMARY_OUT = OUTPUT_CESIUM / "proto2_m1_multimode_deformation_viewer_18_search_summary.json"
REPORT_JSON_OUT = RUN_RECORDS / "phase18_parcel_search_report.json"
REPORT_TXT_OUT = RUN_RECORDS / "phase18_parcel_search_report.txt"

MOVING_INDEX_PARQUET = OUTPUT_DATA / "moving_parcel_index.parquet"

START_MARKER = "<!-- PHASE18_PARCEL_SEARCH_START -->"
END_MARKER = "<!-- PHASE18_PARCEL_SEARCH_END -->"


def ok(msg: str) -> None:
    print(f"[OK] {msg}")


def warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    sys.exit(1)


def require(path: Path, label: str) -> None:
    if not path.exists():
        fail(f"Missing {label}: {path}")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def sanitize_meta_block(html: str) -> str:
    start = html.find("const META = ")
    if start < 0:
        return html
    end = html.find(";\n", start)
    if end < 0:
        end = html.find(";</script>", start)
        if end < 0:
            return html
        end += 1
    else:
        end += 2
    block = html[start:end].replace("\\", "/")
    return html[:start] + block + html[end:]


def strip_old_block(html: str) -> str:
    while True:
        s = html.find(START_MARKER)
        if s < 0:
            return html
        e = html.find(END_MARKER, s)
        if e < 0:
            return html
        html = html[:s] + html[e + len(END_MARKER):]


def extract_pick_index_path(html: str) -> Path: # type: ignore
    asset_base = ""
    m_base = re.search(r'const\s+ASSET_BASE\s*=\s*"([^"]+)"\s*;', html)
    if m_base:
        asset_base = m_base.group(1)

    m_pick = re.search(r'const\s+PICK_INDEX_URL\s*=\s*ASSET_BASE\s*\+\s*"([^"]+)"\s*;', html)
    if m_pick:
        candidate = OUTPUT_CESIUM / asset_base / m_pick.group(1)
        if candidate.exists():
            return candidate

    m_pick = re.search(r'const\s+PICK_INDEX_URL\s*=\s*"([^"]+)"\s*;', html)
    if m_pick:
        candidate = OUTPUT_CESIUM / m_pick.group(1)
        if candidate.exists():
            return candidate

    for rel in [
        "phase15_piston_assets/parcel_pick_index.json",
        "phase13_m1_assets/parcel_pick_index.json",
        "phase14_color_assets/parcel_pick_index.json",
    ]:
        candidate = OUTPUT_CESIUM / rel
        if candidate.exists():
            return candidate

    fail("Could not locate parcel_pick_index.json from source HTML or known asset folders")


def normalize_id(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        f = float(s)
        if math.isfinite(f) and abs(f - round(f)) < 1e-9:
            return str(int(round(f)))
    except Exception:
        pass
    return s


def bbox_union(bboxes: List[List[float]]) -> Optional[List[float]]:
    clean = []
    for b in bboxes:
        if isinstance(b, list) and len(b) == 4:
            vals = [float(x) for x in b]
            if all(math.isfinite(x) for x in vals):
                clean.append(vals)
    if not clean:
        return None
    return [
        min(b[0] for b in clean),
        min(b[1] for b in clean),
        max(b[2] for b in clean),
        max(b[3] for b in clean),
    ]


def load_moving_aliases() -> Dict[str, Dict[str, Any]]:
    if pd is None or not MOVING_INDEX_PARQUET.exists():
        return {}

    try:
        df = pd.read_parquet(MOVING_INDEX_PARQUET)
    except Exception as exc:
        warn(f"Could not read moving index parquet for aliases: {exc}")
        return {}

    cols = {str(c).lower(): str(c) for c in df.columns}
    pnt_col = cols.get("pnt_id") or cols.get("parcel_id") or cols.get("int_id")
    gid_col = cols.get("pnt_gid")
    row_col = (
        cols.get("row_index")
        or cols.get("displacement_row_index")
        or cols.get("moving_row_index")
        or cols.get("moving_index")
        or cols.get("row")
    )
    vi_col = cols.get("vi") or cols.get("v_i")

    aliases: Dict[str, Dict[str, Any]] = {}
    if not pnt_col:
        return aliases

    for _, row in df.iterrows():
        pnt_id = normalize_id(row.get(pnt_col))
        if not pnt_id:
            continue
        item: Dict[str, Any] = {"pnt_id": pnt_id}
        if gid_col:
            gid = normalize_id(row.get(gid_col))
            if gid:
                item["pnt_gid"] = gid
        if row_col:
            try:
                item["row_index"] = int(row.get(row_col)) # type: ignore
            except Exception:
                pass
        if vi_col:
            try:
                val = float(row.get(vi_col)) # type: ignore
                if math.isfinite(val):
                    item["vI"] = val
            except Exception:
                pass
        aliases[pnt_id] = item
    return aliases


def build_search_index(pick_index_path: Path) -> Dict[str, Any]:
    pick = read_json(pick_index_path)
    features = pick.get("features")
    if not isinstance(features, list):
        fail(f"Invalid pick index: no features list in {pick_index_path}")

    moving_aliases = load_moving_aliases()

    by_parcel: Dict[str, List[int]] = defaultdict(list)
    by_footprint: Dict[str, int] = {}

    for i, f in enumerate(features): # type: ignore
        parcel_id = normalize_id(f.get("parcel_id") or f.get("int_id") or f.get("pnt_id") or f.get("id"))
        if not parcel_id:
            continue
        by_parcel[parcel_id].append(i)

        footprint = normalize_id(f.get("footprint_id") or f.get("footprint") or f.get("display_id"))
        if footprint:
            by_footprint[footprint] = i

    records: Dict[str, Dict[str, Any]] = {}
    aliases: Dict[str, str] = {}

    for parcel_id, indices in by_parcel.items():
        feats = [features[i] for i in indices] # type: ignore
        b = bbox_union([f.get("bbox") for f in feats])
        rows = []
        footprint_ids = []
        for f in feats:
            try:
                rows.append(int(f.get("displacement_row_index", -1)))
            except Exception:
                rows.append(-1)
            fp = normalize_id(f.get("footprint_id") or f.get("footprint") or f.get("display_id"))
            if fp:
                footprint_ids.append(fp)

        moving_rows = [r for r in rows if r >= 0]
        is_moving = bool(moving_rows)
        row_index = min(moving_rows) if moving_rows else -1
        status = "moving" if is_moving else "blank"

        alias_data = moving_aliases.get(parcel_id, {})
        pnt_id = alias_data.get("pnt_id") if alias_data else (parcel_id if is_moving else None)
        pnt_gid = alias_data.get("pnt_gid") if alias_data else None

        record = {
            "parcel_id": parcel_id,
            "int_id": parcel_id,
            "pnt_id": pnt_id,
            "pnt_gid": pnt_gid,
            "status": status,
            "has_displacement": is_moving,
            "displacement_row_index": row_index,
            "feature_indices": indices,
            "footprint_ids": sorted(set(footprint_ids)),
            "bbox_local": b,
        }
        records[parcel_id] = record

        aliases[parcel_id] = parcel_id
        if pnt_id:
            aliases[str(pnt_id)] = parcel_id

    gid_counter = Counter()
    for record in records.values():
        gid = record.get("pnt_gid")
        if gid:
            gid_counter[str(gid)] += 1
    for record in records.values():
        gid = record.get("pnt_gid")
        if gid and gid_counter[str(gid)] == 1:
            aliases[str(gid)] = record["parcel_id"]

    counts = Counter(r["status"] for r in records.values())

    return {
        "product": "proto2_phase18_parcel_search_index",
        "schema": "phase18_parcel_search_v1",
        "source_pick_index": str(pick_index_path),
        "feature_count": len(features), # type: ignore
        "parcel_count": len(records),
        "moving_count": int(counts.get("moving", 0)),
        "blank_count": int(counts.get("blank", 0)),
        "records": records,
        "aliases": aliases,
        "footprint_to_feature_index": by_footprint,
        "search_contract": {
            "primary": "parcel_id / int_id",
            "aliases": ["pnt_id", "pnt_gid when unique", "footprint_id"],
            "never_advertise": ["displacement_row_index"],
        },
    }


PHASE18_STYLE = r"""
<style id="phase18ParcelSearchStyle">
#leftSearchBar.phase18ParcelSearchReady {
    position: relative;
}
#addressSearchInput.phase18ParcelSearchInput::placeholder {
    color: var(--ui-search-muted, #6b7280);
}
#leftSearchIcon.phase18SearchClickable {
    cursor: pointer;
    transition: color 0.15s ease, transform 0.15s ease;
}
#leftSearchIcon.phase18SearchClickable:hover {
    color: var(--ui-text-dark, #202124);
    transform: scale(1.06);
}
#parcelSearchFeedback {
    position: absolute;
    top: calc(100% + 7px);
    left: 10px;
    right: 10px;
    min-height: 22px;
    padding: 6px 9px;
    border-radius: 9px;
    font-family: var(--ui-font, Arial, sans-serif);
    font-size: 11px;
    font-weight: 700;
    line-height: 1.25;
    color: rgba(255,255,255,0.94);
    background: rgba(34,36,38,0.94);
    border: 1px solid rgba(255,255,255,0.14);
    box-shadow: 0 4px 16px rgba(0,0,0,0.25);
    opacity: 0;
    transform: translateY(-4px);
    pointer-events: none;
    transition: opacity 0.16s ease, transform 0.16s ease;
    z-index: 10095;
}
#parcelSearchFeedback.open { opacity: 1; transform: translateY(0); }
#parcelSearchFeedback.ok { border-color: rgba(126,245,255,0.38); color: var(--ui-accent, #7ef5ff); }
#parcelSearchFeedback.warn { border-color: rgba(255,213,79,0.44); color: var(--ui-warning, #ffd54f); }
#parcelSearchFeedback.err { border-color: rgba(255,82,82,0.46); color: #ff9a9a; }
</style>
"""


PHASE18_SCRIPT = r"""
<script id="phase18ParcelSearchScript">
(function(){
'use strict';

const PHASE18 = {
  indexUrl: 'phase18_search_assets/parcel_search_index.json',
  index: null,
  ready: false,
  feedbackTimer: null,
};

function phase18Q(id){ return document.getElementById(id); }

function phase18ShowFeedback(message, kind='ok', timeout=2400){
  const box = phase18Q('parcelSearchFeedback');
  if(!box) return;
  box.textContent = message;
  box.className = '';
  box.classList.add('open', kind);
  clearTimeout(PHASE18.feedbackTimer);
  PHASE18.feedbackTimer = setTimeout(()=>box.classList.remove('open'), timeout);
}

function phase18NormalizeInput(raw){
  const s = String(raw || '').trim();
  if(!s) return { kind:'empty' };
  const footprint = s.match(/^\s*(\d+[_-]\d+)\s*$/);
  if(footprint) return { kind:'footprint', value: footprint[1].replace('-', '_') };
  const parcelWord = s.match(/\bparcel\s+(\d{1,12})\b/i);
  if(parcelWord) return { kind:'id', value: String(Number(parcelWord[1])) };
  const onlyNumber = s.match(/^\d{1,12}$/);
  if(onlyNumber) return { kind:'id', value: String(Number(s)) };
  const anyNumber = s.match(/\b\d{2,12}\b/);
  if(anyNumber) return { kind:'id', value: String(Number(anyNumber[0])) };
  return { kind:'invalid' };
}

async function phase18LoadIndex(){
  const response = await fetch(PHASE18.indexUrl);
  if(!response.ok) throw new Error(`Failed to fetch ${PHASE18.indexUrl}: ${response.status}`);
  PHASE18.index = await response.json();
  PHASE18.ready = true;
}

function phase18FeatureFromRecord(record){
  if(!record || !pickIndex || !Array.isArray(pickIndex.features)) return null;
  const features = (record.feature_indices || []).map(i => pickIndex.features[Number(i)]).filter(Boolean);
  if(!features.length) return null;
  if(features.length === 1) return features[0];

  const rings = [];
  const bboxes = [];
  for(const f of features){
    if(Array.isArray(f.rings)) rings.push(...f.rings);
    if(Array.isArray(f.bbox) && f.bbox.length === 4) bboxes.push(f.bbox);
  }

  const bbox = bboxes.length ? [
    Math.min(...bboxes.map(b=>Number(b[0]))),
    Math.min(...bboxes.map(b=>Number(b[1]))),
    Math.max(...bboxes.map(b=>Number(b[2]))),
    Math.max(...bboxes.map(b=>Number(b[3]))),
  ] : features[0].bbox;

  return {
    ...features[0],
    parcel_id: record.parcel_id,
    int_id: record.int_id || record.parcel_id,
    pnt_id: record.pnt_id,
    pnt_gid: record.pnt_gid,
    parcel_status: record.status,
    displacement_row_index: Number(record.displacement_row_index ?? -1),
    footprint_id: (record.footprint_ids || []).length > 1 ? `${record.footprint_ids[0]} +${record.footprint_ids.length - 1}` : ((record.footprint_ids || [features[0].footprint_id || '—'])[0]),
    bbox,
    rings,
  };
}

function phase18RecordForQuery(raw){
  if(!PHASE18.index) return null;
  const parsed = phase18NormalizeInput(raw);
  if(parsed.kind === 'empty') return { error:'empty' };
  if(parsed.kind === 'invalid') return { error:'invalid' };

  if(parsed.kind === 'footprint'){
    const featureIdx = PHASE18.index.footprint_to_feature_index?.[parsed.value];
    if(featureIdx !== undefined && pickIndex?.features?.[Number(featureIdx)]){
      const f = pickIndex.features[Number(featureIdx)];
      const parcelId = String(f.parcel_id ?? f.int_id ?? f.pnt_id ?? '');
      const record = PHASE18.index.records?.[parcelId];
      if(record) return { record, footprintFeature: f };
    }
    return { error:'not_found', query:parsed.value };
  }

  const aliasParcel = PHASE18.index.aliases?.[parsed.value] || parsed.value;
  const record = PHASE18.index.records?.[String(aliasParcel)];
  if(!record) return { error:'not_found', query:parsed.value };
  return { record };
}

function phase18FlyToFeature(feature){
  try{
    if(!viewer || !feature || !Array.isArray(feature.bbox) || feature.bbox.length !== 4 || typeof localPoint !== 'function') return;
    const b = feature.bbox.map(Number);
    const cx = (b[0] + b[2]) * 0.5;
    const cy = (b[1] + b[3]) * 0.5;
    const dx = Math.max(1, b[2] - b[0]);
    const dy = Math.max(1, b[3] - b[1]);
    const center = localPoint(cx, cy, 30.0);
    const radius = Math.max(90.0, Math.sqrt(dx*dx + dy*dy) * 3.5);
    const pitch = (typeof currentViewMode !== 'undefined' && currentViewMode === '2D') ? Cesium.Math.toRadians(-89.0) : Cesium.Math.toRadians(-48.0);
    const heading = viewer.camera.heading || 0.0;
    const range = Math.max(240.0, radius * 3.0);
    viewer.camera.flyToBoundingSphere(new Cesium.BoundingSphere(center, radius), {
      duration: 0.85,
      offset: new Cesium.HeadingPitchRange(heading, pitch, range)
    });
  }catch(error){
    console.warn('[parcel search] fly-to failed', error);
  }
}

function phase18SelectFeature(feature){
  if(!feature) return false;
  try{
    selectedFeature = feature;
    if(typeof drawSelectedOutline === 'function') drawSelectedOutline(feature);
    if(typeof openParcelPopup === 'function') openParcelPopup();
    if(typeof renderParcelPopup === 'function') renderParcelPopup();
    if(typeof phase17CloseChart === 'function') phase17CloseChart();
    phase18FlyToFeature(feature);
    viewer?.scene?.requestRender?.();
    return true;
  }catch(error){
    console.error('[parcel search] selection failed', error);
    return false;
  }
}

function phase18Search(raw){
  const result = phase18RecordForQuery(raw);
  if(result?.error === 'empty' || result?.error === 'invalid'){
    phase18ShowFeedback('Enter a numeric parcel ID.', 'warn');
    return;
  }
  if(result?.error === 'not_found'){
    phase18ShowFeedback('Parcel ID not found. Check the number and try again.', 'err');
    return;
  }

  const feature = result.footprintFeature || phase18FeatureFromRecord(result.record);
  if(!feature){
    phase18ShowFeedback('Parcel found, but selection geometry is unavailable.', 'err');
    return;
  }
  if(!phase18SelectFeature(feature)){
    phase18ShowFeedback('Parcel found, but could not select it.', 'err');
    return;
  }
  if(result.record.has_displacement) phase18ShowFeedback(`Parcel ${result.record.parcel_id} selected.`, 'ok');
  else phase18ShowFeedback(`Parcel ${result.record.parcel_id} selected · no SPAMS displacement.`, 'warn', 3200);
}

function phase18InstallUi(){
  const input = phase18Q('addressSearchInput');
  const bar = phase18Q('leftSearchBar');
  const icon = phase18Q('leftSearchIcon');
  if(!input || !bar){
    console.warn('[parcel search] left search bar not found');
    return;
  }

  bar.classList.add('phase18ParcelSearchReady');
  input.classList.add('phase18ParcelSearchInput');
  input.placeholder = 'Search parcel ID';
  input.setAttribute('autocomplete', 'off');
  input.setAttribute('inputmode', 'numeric');

  if(icon){
    icon.classList.add('phase18SearchClickable');
    icon.title = 'Search parcel ID';
  }

  if(!phase18Q('parcelSearchFeedback')){
    const box = document.createElement('div');
    box.id = 'parcelSearchFeedback';
    box.setAttribute('aria-live', 'polite');
    bar.appendChild(box);
  }

  if(!input.dataset.phase18Bound){
    input.dataset.phase18Bound = '1';
    input.addEventListener('keydown', (evt) => {
      if(evt.key === 'Enter'){
        evt.preventDefault();
        phase18Search(input.value);
      }
    });
  }
  if(icon && !icon.dataset.phase18Bound){
    icon.dataset.phase18Bound = '1';
    icon.addEventListener('click', (evt) => {
      evt.preventDefault();
      evt.stopPropagation();
      phase18Search(input.value);
    });
  }
}

function phase18WaitForViewerReady(){
  const timer = setInterval(() => {
    try{
      if(PHASE18.ready && typeof pickIndex !== 'undefined' && pickIndex && typeof viewer !== 'undefined' && viewer){
        clearInterval(timer);
        phase18InstallUi();
        console.log('[parcel search] ready', PHASE18.index?.parcel_count, 'parcels');
      }
    }catch(e){}
  }, 150);
}

async function phase18Main(){
  try{
    await phase18LoadIndex();
    phase18WaitForViewerReady();
  }catch(error){
    console.error('[parcel search] failed', error);
  }
}

if(document.readyState === 'loading') document.addEventListener('DOMContentLoaded', phase18Main);
else phase18Main();

try { window.phase18SearchParcel = phase18Search; } catch(e) {}
})();
</script>
"""


def patch_html(html: str) -> str:
    html = strip_old_block(html)
    html = sanitize_meta_block(html)

    if 'id="phase18ParcelSearchStyle"' not in html:
        if "</head>" not in html:
            fail("Could not find </head>")
        html = html.replace("</head>", PHASE18_STYLE + "\n</head>", 1)

    block = START_MARKER + "\n" + PHASE18_SCRIPT + "\n" + END_MARKER + "\n"
    if "</body>" not in html:
        fail("Could not find </body>")
    html = html.replace("</body>", block + "</body>", 1)

    html = html.replace("proto2_m1_multimode_deformation_viewer_17_fixed7", "proto2_m1_multimode_deformation_viewer_18_search")
    html = html.replace("Phase17", "Phase18")
    html = html.replace("PHASE 17", "PHASE 18")
    return html


def main() -> None:
    print("\n=== PROTO2 PHASE 18: PARCEL SEARCH VIEWER ===")
    print(f"Project root: {PROJECT_ROOT}")

    require(SOURCE_HTML, "Phase17 fixed7 HTML")
    OUTPUT_CESIUM.mkdir(parents=True, exist_ok=True)
    SEARCH_ASSET_DIR.mkdir(parents=True, exist_ok=True)
    RUN_RECORDS.mkdir(parents=True, exist_ok=True)

    source_html = SOURCE_HTML.read_text(encoding="utf-8", errors="replace")
    pick_index_path = extract_pick_index_path(source_html)
    ok(f"using pick index: {pick_index_path}")

    search_index = build_search_index(pick_index_path)
    write_json(SEARCH_INDEX_OUT, search_index)
    ok(f"wrote {SEARCH_INDEX_OUT}")

    html = patch_html(source_html)
    HTML_OUT.write_text(html, encoding="utf-8")
    ok(f"wrote {HTML_OUT}")

    inherited = read_json(SOURCE_SUMMARY) if SOURCE_SUMMARY.exists() else {}
    summary = {
        "product": "proto2_m1_multimode_deformation_viewer_18_search",
        "source_html": str(SOURCE_HTML),
        "output_html": str(HTML_OUT),
        "source_summary": str(SOURCE_SUMMARY) if SOURCE_SUMMARY.exists() else None,
        "inherited_product": inherited.get("product"),
        "search_index": str(SEARCH_INDEX_OUT),
        "feature_count": search_index["feature_count"],
        "parcel_count": search_index["parcel_count"],
        "moving_count": search_index["moving_count"],
        "blank_count": search_index["blank_count"],
        "search_contract": search_index["search_contract"],
        "behavior": [
            "Search by parcel_id/int_id, pnt_id alias, unique pnt_gid alias, or hidden footprint_id.",
            "Moving parcels select normally with yellow outline, popup, and trendline support.",
            "Blank parcels are selectable with yellow outline and no-displacement popup behavior.",
            "Search selection flies camera to the selected parcel.",
        ],
    }
    write_json(SUMMARY_OUT, summary)
    write_json(REPORT_JSON_OUT, summary)
    REPORT_TXT_OUT.write_text(
        "PROTO2 PHASE 18: PARCEL SEARCH VIEWER\n"
        f"Project root: {PROJECT_ROOT}\n"
        f"Source HTML: {SOURCE_HTML}\n"
        f"Output HTML: {HTML_OUT}\n"
        f"Search index: {SEARCH_INDEX_OUT}\n"
        f"Features: {search_index['feature_count']}\n"
        f"Parcels: {search_index['parcel_count']}\n"
        f"Moving: {search_index['moving_count']}\n"
        f"Blank: {search_index['blank_count']}\n",
        encoding="utf-8",
    )
    ok(f"wrote {SUMMARY_OUT}")
    ok(f"wrote {REPORT_JSON_OUT}")
    print("\n=== PHASE 18 RESULT: PASS ===")


if __name__ == "__main__":
    main()
