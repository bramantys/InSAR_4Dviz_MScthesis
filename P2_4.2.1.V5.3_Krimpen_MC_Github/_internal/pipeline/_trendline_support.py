#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trendline support module

Runtime trendline support — add a Proto1-style parcel trendline/chart panel to the
accepted Phase16E multimode deformation viewer.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import pandas as pd
except Exception as exc:
    print(f"[FAIL] pandas import failed: {exc}")
    sys.exit(1)

from _proto2_config import load_project_config, output_cesium_dir, output_data_dir, stage_records_dir


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = load_project_config(PROJECT_ROOT)
OUTPUT_CESIUM = output_cesium_dir(PROJECT_ROOT, CONFIG)
OUTPUT_DATA = output_data_dir(PROJECT_ROOT, CONFIG)
RUN_RECORDS = stage_records_dir(PROJECT_ROOT, CONFIG)

SOURCE_HTML = OUTPUT_CESIUM / "proto2_m1_multimode_deformation_viewer_16e.html"
SOURCE_SUMMARY = OUTPUT_CESIUM / "proto2_m1_multimode_deformation_viewer_16e_summary.json"
TREND_ASSET_DIR = OUTPUT_CESIUM / "phase17_trendline_assets"
TREND_MANIFEST = TREND_ASSET_DIR / "parcel_trendline_manifest.json"
HTML_OUT = OUTPUT_CESIUM / "proto2_m1_multimode_deformation_viewer_17_fixed7.html"
SUMMARY_OUT = OUTPUT_CESIUM / "proto2_m1_multimode_deformation_viewer_17_fixed7_summary.json"
REPORT_JSON_OUT = RUN_RECORDS / "phase17_trendline_viewer_report.json"
REPORT_TXT_OUT = RUN_RECORDS / "phase17_trendline_viewer_report.txt"

MOVING_INDEX_PARQUET = OUTPUT_DATA / "moving_parcel_index.parquet"
ANIMATION_MANIFEST_JSON = OUTPUT_DATA / "parcel_animation_manifest.json"
EPOCH_AXIS_JSON = OUTPUT_DATA / "epoch_axis.json"
DISPLACEMENT_SUMMARY_JSON = OUTPUT_DATA / "parcel_displacement_summary.json"
DISPLACEMENT_LONG_PARQUET = OUTPUT_DATA / "parcel_displacement_long.parquet"

BIN_REV = "phase15_piston_assets/parcel_displacement_reversible_f32.bin"
BIN_IRR = "phase15_piston_assets/parcel_displacement_irreversible_f32.bin"
BIN_TOTAL = "phase15_piston_assets/parcel_displacement_total_f32.bin"
START_MARKER = "<!-- PHASE17_TRENDLINE_START -->"
END_MARKER = "<!-- PHASE17_TRENDLINE_END -->"


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


def choose_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    lower = {str(c).lower(): str(c) for c in columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def load_moving_index() -> Tuple[List[int], Dict[str, int], Dict[str, float], int]:
    require(MOVING_INDEX_PARQUET, "moving parcel index parquet")
    df = pd.read_parquet(MOVING_INDEX_PARQUET)
    if df.empty:
        fail("moving_parcel_index.parquet is empty")

    cols = [str(c) for c in df.columns]
    id_col = choose_column(cols, ["pnt_id", "parcel_id", "int_id", "id"])
    row_col = choose_column(cols, ["row_index", "displacement_row_index", "moving_row_index", "moving_index", "row", "render_row_index"])
    vi_col = choose_column(cols, ["vI", "vi", "VI"])

    if id_col is None:
        fail(f"Could not find parcel ID column in moving_parcel_index.parquet. Columns: {cols}")
    if row_col is None:
        warn("No explicit row-index column found; using dataframe order as moving row index")
        df = df.reset_index(drop=True).copy()
        row_col = "__row_index__"
        df[row_col] = df.index.astype(int)

    keep = [id_col, row_col] + ([vi_col] if vi_col else [])
    df = df[keep].copy()
    df[id_col] = pd.to_numeric(df[id_col], errors="coerce")
    df[row_col] = pd.to_numeric(df[row_col], errors="coerce")
    df = df[df[id_col].notna() & df[row_col].notna()].copy()
    df[id_col] = df[id_col].astype(int)
    df[row_col] = df[row_col].astype(int)
    df = df.sort_values(row_col).reset_index(drop=True)

    parcel_ids = df[id_col].astype(int).tolist()
    parcel_to_row = {str(pid): int(r) for pid, r in zip(df[id_col], df[row_col])}
    vi_by_parcel: Dict[str, float] = {}
    if vi_col:
        vi_vals = pd.to_numeric(df[vi_col], errors="coerce")
        for pid, val in zip(df[id_col], vi_vals):
            if pd.notna(val):
                vi_by_parcel[str(int(pid))] = float(val)

    epoch_count = None
    if ANIMATION_MANIFEST_JSON.exists():
        manifest = read_json(ANIMATION_MANIFEST_JSON)
        for key in ["epoch_count", "epochs", "num_epochs"]:
            if isinstance(manifest.get(key), int):
                epoch_count = int(manifest[key])
                break
        shape = manifest.get("matrix_shape")
        if epoch_count is None and isinstance(shape, list) and len(shape) == 2:
            epoch_count = int(shape[1])

    if epoch_count is None and EPOCH_AXIS_JSON.exists():
        ejson = read_json(EPOCH_AXIS_JSON)
        labels = ejson.get("epoch_labels") or ejson.get("labels") or ejson.get("epochs")
        if isinstance(labels, list):
            epoch_count = len(labels)
    if epoch_count is None:
        epoch_count = 365
    return parcel_ids, parcel_to_row, vi_by_parcel, epoch_count


def normalize_epoch_label(value: Any) -> str:
    """Return a clean YYYY-MM-DD label from strings, timestamps, or dict records.

    Some upstream epoch_axis manifests store records like
    {"epoch_index": 0, "epoch_label": "2025-01-01"}.
    The previous Phase17 exporter used str(record)[:10], which produced
    labels like "{'epoch_in" and broke the x-axis/date badge.
    """
    if value is None:
        return ""
    if isinstance(value, dict):
        preferred_keys = [
            "date", "epoch_date", "epoch_label", "label", "epoch",
            "time", "timestamp", "iso_date", "datetime"
        ]
        for key in preferred_keys:
            if key in value and value[key] is not None:
                return normalize_epoch_label(value[key])
        # Last resort: search values for an ISO date.
        for item in value.values():
            s = normalize_epoch_label(item)
            if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
                return s
        return ""
    if hasattr(value, "strftime"):
        try:
            return value.strftime("%Y-%m-%d")
        except Exception:
            pass
    s = str(value)
    m = re.search(r"\d{4}-\d{2}-\d{2}", s)
    if m:
        return m.group(0)
    try:
        parsed = pd.to_datetime(s, errors="coerce")
        if pd.notna(parsed):
            return parsed.strftime("%Y-%m-%d")
    except Exception:
        pass
    return s[:10]


def load_epoch_labels(epoch_count: int) -> List[str]:
    labels: Optional[List[str]] = None
    for path in [EPOCH_AXIS_JSON, ANIMATION_MANIFEST_JSON]:
        if not path.exists():
            continue
        data = read_json(path)
        for key in ["epoch_labels", "labels", "epochs"]:
            val = data.get(key)
            if isinstance(val, list) and val:
                labels = [normalize_epoch_label(v) for v in val]
                break
        if labels:
            break

    if labels is None and DISPLACEMENT_LONG_PARQUET.exists():
        df = pd.read_parquet(DISPLACEMENT_LONG_PARQUET, columns=["epoch"])
        labels = sorted({normalize_epoch_label(v) for v in df["epoch"].tolist()})
    if labels is None:
        base = pd.date_range("2025-01-01", periods=epoch_count, freq="D")
        labels = [d.strftime("%Y-%m-%d") for d in base]
    labels = [label for label in labels if label]
    valid_iso = [label for label in labels if re.match(r"^\d{4}-\d{2}-\d{2}$", label)]
    if len(valid_iso) == len(labels) and labels:
        pass
    elif len(valid_iso) >= max(2, int(0.9 * len(labels))):
        labels = valid_iso
    else:
        warn("Epoch labels could not be parsed cleanly; falling back to daily 2025 axis")
        base = pd.date_range("2025-01-01", periods=epoch_count, freq="D")
        labels = [d.strftime("%Y-%m-%d") for d in base]

    if len(labels) != epoch_count:
        warn(f"Epoch label count {len(labels)} differs from epoch_count {epoch_count}; using label count")
    return labels


def load_component_ranges() -> Dict[str, Dict[str, float]]:
    if DISPLACEMENT_SUMMARY_JSON.exists():
        data = read_json(DISPLACEMENT_SUMMARY_JSON)
        out: Dict[str, Dict[str, float]] = {}
        for comp in ["reversible", "irreversible", "total"]:
            obj = data.get(comp)
            if isinstance(obj, dict) and obj.get("min") is not None and obj.get("max") is not None:
                out[comp] = {"min": float(obj["min"]), "max": float(obj["max"])}
        if out:
            return out
    if DISPLACEMENT_LONG_PARQUET.exists():
        df = pd.read_parquet(DISPLACEMENT_LONG_PARQUET)
        total_col = "h_spams_final" if "h_spams_final" in df.columns else "total"
        return {
            "reversible": {"min": float(df["reversible"].min()), "max": float(df["reversible"].max())},
            "irreversible": {"min": float(df["irreversible"].min()), "max": float(df["irreversible"].max())},
            "total": {"min": float(df[total_col].min()), "max": float(df[total_col].max())},
        }
    return {
        "reversible": {"min": -72.228317, "max": 29.513287},
        "irreversible": {"min": -10.487019, "max": 0.0},
        "total": {"min": -75.873001, "max": 29.513287},
    }


def build_trend_manifest() -> Dict[str, Any]:
    parcel_ids, parcel_to_row, vi_by_parcel, epoch_count = load_moving_index()
    labels = load_epoch_labels(epoch_count)
    epoch_count = len(labels)
    return {
        "product": "proto2_phase17_parcel_trendline_manifest",
        "schema": "phase17_trendline_v1",
        "epoch_count": epoch_count,
        "epoch_labels": labels,
        "parcel_ids_in_row_order": parcel_ids,
        "parcel_to_row": parcel_to_row,
        "vi_by_parcel": vi_by_parcel,
        "component_ranges_mm": load_component_ranges(),
        "binary_assets": {"reversible": BIN_REV, "irreversible": BIN_IRR, "total": BIN_TOTAL},
        "notes": [
            "Trendline uses existing phase15 float32 component binaries.",
            "Combined chart uses one shared mm axis: irreversible + total, with fill between them as reversible gap.",
        ],
    }


TRENDLINE_STYLE = r'''
<style id="phase17TrendlineStyle">
/* Phase17 compact/expanded chart shell.
   Default = compact epoch slider only.
   Expanded = chart open after parcel popup chart button is clicked. */
#rumTrendlinePanel,
#rumTrendlineCanvas {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
}

#epochPanel.phase17-trendline-ready {
    position: absolute !important;
    left: 16px !important;
    bottom: 40px !important;
    width: var(--epoch-panel-width, min(688px, calc(40vw + 96px))) !important;
    min-width: var(--epoch-panel-min-width, 528px) !important;
    height: var(--epoch-panel-height, 92px) !important;
    min-height: var(--epoch-panel-height, 92px) !important;
    max-height: var(--epoch-panel-height, 92px) !important;
    padding: 0 var(--epoch-pad-x, 16px) !important;
    overflow: hidden !important;
    background: var(--ui-bg-panel, rgba(34,36,38,0.86)) !important;
    color: var(--ui-text, rgba(255,255,255,0.94)) !important;
    border: 1px solid var(--ui-border, rgba(255,255,255,0.14)) !important;
    border-radius: 18px !important;
    box-shadow: var(--ui-shadow-panel, 0 4px 16px rgba(0,0,0,0.28)) !important;
    z-index: 10045 !important;
    transform: none !important;
    opacity: 1 !important;
    pointer-events: auto !important;
    transition: left 0.26s ease, height 0.20s ease, opacity 0.20s ease, filter 0.20s ease, background 0.20s ease !important;
}

#leftControlRoot.drawerOpen ~ #epochPanel.phase17-trendline-ready {
    left: calc(var(--left-drawer-width, 372px) + 16px) !important;
}

#epochPanel.phase17-trendline-ready.phase17-chart-open {
    width: var(--epoch-panel-width, min(688px, calc(40vw + 96px))) !important;
    min-width: var(--epoch-panel-min-width, 528px) !important;
    height: calc(var(--epoch-panel-height, 92px) + var(--epoch-trendline-extra-height, 300px)) !important;
    min-height: calc(var(--epoch-panel-height, 92px) + var(--epoch-trendline-extra-height, 300px)) !important;
    max-height: calc(var(--epoch-panel-height, 92px) + var(--epoch-trendline-extra-height, 300px)) !important;
    background: var(--ui-bg-panel, rgba(34,36,38,0.86)) !important;
    box-shadow: var(--ui-shadow-panel-open, 0 8px 28px rgba(0,0,0,0.34)) !important;
}

/* Do not let old Proto1/placeholder class expand the panel. Phase17 owns expansion. */
#epochPanel.phase17-trendline-ready.trendlineOpen:not(.phase17-chart-open) {
    height: var(--epoch-panel-height, 92px) !important;
    min-height: var(--epoch-panel-height, 92px) !important;
    max-height: var(--epoch-panel-height, 92px) !important;
}

/* Compact epoch controls, matching Proto1 footprint. */
#epochPanel.phase17-trendline-ready #epochTopRow {
    position: absolute !important;
    left: var(--epoch-pad-x, 16px) !important;
    right: var(--epoch-pad-x, 16px) !important;
    bottom: 9px !important;
    height: 34px !important;
    min-height: 34px !important;
    margin: 0 !important;
    display: flex !important;
    align-items: center !important;
    gap: 12px !important;
    z-index: 10059 !important;
}

#epochPanel.phase17-trendline-ready #epochControls {
    height: 34px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;
    gap: 7px !important;
    margin: 0 !important;
    flex: 0 0 auto !important;
    z-index: 10059 !important;
}

#epochPanel.phase17-trendline-ready .epochBtn {
    width: 34px !important;
    height: 34px !important;
    min-width: 34px !important;
    min-height: 34px !important;
    border-radius: var(--ui-radius-pill, 999px) !important;
    border: 0 !important;
    background: var(--ui-bubble-bg, rgba(245,245,245,0.95)) !important;
    color: var(--ui-bubble-text, #202124) !important;
    box-shadow: var(--ui-shadow-bubble, 0 2px 10px rgba(0,0,0,0.24)) !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    padding: 0 !important;
    line-height: 1 !important;
}
#epochPanel.phase17-trendline-ready #epochPlayBtn {
    width: 38px !important;
    min-width: 38px !important;
}

#epochPanel.phase17-trendline-ready #epochTextGroup {
    display: flex !important;
    align-items: center !important;
    gap: 12px !important;
    height: 34px !important;
    min-width: 0 !important;
    overflow: hidden !important;
    white-space: nowrap !important;
}

#epochPanel.phase17-trendline-ready #epochLabel {
    margin: 0 !important;
    text-align: left !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    line-height: 1 !important;
    color: var(--ui-accent, #7ef5ff) !important;
    max-width: 310px !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    white-space: nowrap !important;
}

#epochPanel.phase17-trendline-ready #epochInfo {
    margin: 0 !important;
    color: var(--ui-text-muted, rgba(255,255,255,0.66)) !important;
    font-size: 10px !important;
    line-height: 1 !important;
    flex: 0 0 auto !important;
}

#epochPanel.phase17-trendline-ready #timeLockedNote {
    position: absolute !important;
    left: 230px !important;
    bottom: 33px !important;
    color: var(--ui-accent, #7ef5ff) !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    z-index: 10060 !important;
}

#epochPanel.phase17-trendline-ready #epochAxis,
#epochPanel.phase17-trendline-ready #epochAxis .epochAxisTick,
#epochPanel.phase17-trendline-ready #epochAxis .epochAxisTick span {
    display: none !important;
}

#epochPanel.phase17-trendline-ready #epochSlider {
    position: absolute !important;
    left: calc(var(--epoch-plot-left, 45px) - var(--epoch-slider-outset, 11px)) !important;
    right: calc(var(--epoch-plot-right, 45px) - var(--epoch-slider-outset, 11px)) !important;
    bottom: 52px !important;
    width: calc(100% - var(--epoch-plot-left, 45px) - var(--epoch-plot-right, 45px) + (2 * var(--epoch-slider-outset, 11px))) !important;
    margin: 0 !important;
    display: block !important;
    accent-color: #d7e7ff !important;
    z-index: 10059 !important;
}

#epochPanel.phase17-trendline-ready.timeLocked {
    opacity: 0.52 !important;
    filter: grayscale(1) !important;
    background: rgba(25,25,25,0.62) !important;
}

/* Chart overlay is hidden unless the panel is explicitly open. */
#phase17TrendlineChrome {
    position: absolute !important;
    inset: 0 !important;
    display: none !important;
    pointer-events: none !important;
    z-index: 10057 !important;
}
#epochPanel.phase17-chart-open #phase17TrendlineChrome {
    display: block !important;
}
#phase17TrendlineChrome * {
    box-sizing: border-box;
    font-family: var(--ui-font, Arial, sans-serif);
}

#phase17TrendlineHeader {
    position: absolute !important;
    top: 0 !important;
    left: 0 !important;
    right: 0 !important;
    height: var(--rum-trendline-header-height, 48px) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    padding: 4px 12px 4px 14px !important;
    box-sizing: border-box !important;
    background: transparent !important;
    border-bottom: 0 !important;
    pointer-events: none !important;
    z-index: 10058 !important;
}

#phase17TrendlineTitleBlock {
    min-width: 0 !important;
    flex: 1 1 auto !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
    gap: 3px !important;
    padding-right: 12px !important;
    overflow: hidden !important;
}

#phase17TrendlineTitle {
    color: rgba(255,255,255,0.96) !important;
    font-family: var(--ui-font, Arial, sans-serif) !important;
    font-size: 13px !important;
    font-weight: 750 !important;
    line-height: 1.08 !important;
    min-width: 0 !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    white-space: nowrap !important;
}

#phase17TrendlineSubtitle {
    color: var(--ui-text-muted, rgba(255,255,255,0.66)) !important;
    font-family: var(--ui-font, Arial, sans-serif) !important;
    font-size: 9.6px !important;
    font-weight: 600 !important;
    line-height: 1.05 !important;
    min-height: 10px !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    white-space: nowrap !important;
}

#phase17TrendlineControls {
    margin-left: auto !important;
    margin-right: 0 !important;
    height: 38px !important;
    display: flex !important;
    align-items: center !important;
    gap: 7px !important;
    pointer-events: auto !important;
    flex: 0 0 auto !important;
    z-index: 10061 !important;
}

.phase17Select {
    height: 26px !important;
    min-width: 72px !important;
    border-radius: 8px !important;
    border: 1px solid var(--ui-border, rgba(255,255,255,0.14)) !important;
    background: var(--ui-bg-panel-strong, rgba(34,36,38,0.94)) !important;
    color: var(--ui-text, rgba(255,255,255,0.94)) !important;
    font-family: var(--ui-font, Arial, sans-serif) !important;
    font-size: 10.4px !important;
    font-weight: 700 !important;
    padding: 2px 6px !important;
    outline: none !important;
}

.phase17AxisStack {
    height: 38px !important;
    width: 92px !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
    align-items: stretch !important;
    gap: 3px !important;
}

.phase17AxisRow {
    height: 17px !important;
    display: grid !important;
    grid-template-columns: 24px 1fr !important;
    align-items: center !important;
    column-gap: 5px !important;
    color: var(--ui-text-muted, rgba(255,255,255,0.66)) !important;
    font-size: 8.8px !important;
    line-height: 1 !important;
    white-space: nowrap !important;
    text-align: right !important;
}

.phase17AxisInput {
    width: 63px !important;
    height: 17px !important;
    box-sizing: border-box !important;
    border-radius: 5px !important;
    border: 1px solid var(--ui-border-soft, rgba(255,255,255,0.09)) !important;
    background: rgba(255,255,255,0.10) !important;
    color: var(--ui-text, rgba(255,255,255,0.94)) !important;
    font-family: var(--ui-font, Arial, sans-serif) !important;
    font-size: 9px !important;
    font-weight: 700 !important;
    padding: 0 5px !important;
    text-align: right !important;
    outline: none !important;
}

.phase17AxisInput[data-editable="1"] {
    background: var(--ui-search-bg, rgba(250,250,250,0.96)) !important;
    color: var(--ui-text-dark, #202124) !important;
    border-color: var(--ui-accent-2, #b9d7ff) !important;
}

#phase17PngBtn {
    height: 26px !important;
    min-width: 42px !important;
    border: 1px solid var(--ui-border, rgba(255,255,255,0.14)) !important;
    border-radius: 999px !important;
    background: var(--ui-bg-soft, rgba(255,255,255,0.08)) !important;
    color: var(--ui-text, rgba(255,255,255,0.94)) !important;
    font-family: var(--ui-font, Arial, sans-serif) !important;
    font-size: 9px !important;
    font-weight: 800 !important;
    cursor: pointer !important;
    padding: 0 8px !important;
}

#phase17CloseBtn {
    border: 0 !important;
    background: transparent !important;
    color: rgba(255,255,255,0.92) !important;
    font-size: 22px !important;
    line-height: 1 !important;
    cursor: pointer !important;
    padding: 2px 4px !important;
    pointer-events: auto !important;
    z-index: 10060 !important;
}

#phase17ChartWrap {
    position: absolute !important;
    left: 0 !important;
    top: var(--rum-trendline-header-height, 48px) !important;
    width: 100% !important;
    height: calc(100% - var(--rum-trendline-header-height, 48px)) !important;
    pointer-events: none !important;
    background: transparent !important;
    overflow: visible !important;
}

#phase17TrendSvg {
    position: absolute !important;
    left: 0 !important;
    top: 0 !important;
    width: 100% !important;
    height: 100% !important;
    display: block !important;
}

#phase17Placeholder {
    position: absolute;
    left: 42px;
    right: 42px;
    top: 28px;
    height: 190px;
    display: none;
    align-items: center;
    justify-content: center;
    text-align: center;
    color: rgba(255,255,255,0.70);
    font-size: 12px;
    font-weight: 700;
    background: rgba(0,0,0,0.16);
    border-radius: 10px;
}

#phase17CurrentDateBadge {
    position: absolute !important;
    bottom: 82px !important;
    min-width: 82px !important;
    text-align: center !important;
    left: 100px;
    transform: translateX(-50%);
    padding: 2px 8px;
    background: #2f7cf6;
    color: #fff;
    font-size: 11px;
    font-weight: 700;
    pointer-events: none;
    box-shadow: 0 1px 4px rgba(0,0,0,0.28);
    z-index: 10062 !important;
}

#phase17ChartToggleBtn {
    display: none !important;
}
</style>
'''


TRENDLINE_SCRIPT = r'''
<script id="phase17TrendlineScript">
(function(){
'use strict';
const PHASE17 = { manifestUrl:'phase17_trendline_assets/parcel_trendline_manifest.json', manifest:null, arrays:null, selectedParcelId:null, selectedRow:null, chartView:'auto', axisMode:'auto', manualMin:null, manualMax:null, visible:true, lastKey:'', cache:new Map() };
function q(id){return document.getElementById(id);} function clamp(v,a,b){return Math.max(a,Math.min(b,v));} function fmt(v,d=2){return Number.isFinite(v)?Number(v).toFixed(d):'—';} function fmtVi(v){return Number.isFinite(v)?`${Number(v).toFixed(2)} mm/yr`:'—';}
function currentEpoch(){const el=q('epochSlider'); return el?Math.max(0,Number(el.value||0)):0;} function currentModeName(){try{return typeof currentMode!=='undefined'?String(currentMode):(q('parcelModeSelect')?.value||'total');}catch(e){return'total';}}

function disableLegacyTrendlineShell(){
  const legacy = q('rumTrendlinePanel');
  if(legacy){
    legacy.classList.remove('open');
    legacy.setAttribute('aria-hidden','true');
    legacy.style.setProperty('display','none','important');
    legacy.style.setProperty('visibility','hidden','important');
    legacy.style.setProperty('opacity','0','important');
    legacy.style.setProperty('pointer-events','none','important');
  }
  const canvas = q('rumTrendlineCanvas');
  if(canvas){
    canvas.style.setProperty('display','none','important');
    canvas.style.setProperty('visibility','hidden','important');
    canvas.style.setProperty('opacity','0','important');
  }
  const panel = q('epochPanel');
  if(panel) panel.classList.remove('trendlineOpen');
  // Override the old Phase13/14 shell functions if they exist.
  try { window.openTrendlinePlaceholder = function(){ disableLegacyTrendlineShell(); renderNow(true); }; } catch(e) {}
  try { window.closeTrendlinePlaceholder = function(){ disableLegacyTrendlineShell(); }; } catch(e) {}
  try { window.drawTrendlinePlaceholder = function(){}; } catch(e) {}
}
function modeToChart(mode){if(PHASE17.chartView!=='auto')return PHASE17.chartView; if(mode==='irreversible')return'irreversible'; if(mode==='reversible')return'reversible'; if(mode==='combined')return'decomposition'; return'total';}
function rawEpochLabel(raw){
  if(raw === undefined || raw === null) return '';
  if(typeof raw === 'string' || typeof raw === 'number') return String(raw);
  if(typeof raw === 'object'){
    for(const key of ['date','epoch','label','epoch_label','epoch_date','iso_date','time','timestamp']){
      if(raw[key] !== undefined && raw[key] !== null) return String(raw[key]);
    }
    try{
      const s = JSON.stringify(raw);
      const m = s.match(/\d{4}-\d{2}-\d{2}/);
      if(m) return m[0];
      return s;
    }catch(e){ return String(raw); }
  }
  return String(raw);
}
function epochDateString(raw){
  const s = rawEpochLabel(raw);
  const m = s.match(/\d{4}-\d{2}-\d{2}/);
  if(m) return m[0];
  const d = new Date(s);
  if(Number.isFinite(d.getTime())) return d.toISOString().slice(0,10);
  return s.slice(0,10);
}
function parseDateLabel(s){
  const ds = epochDateString(s);
  const d = new Date(ds + 'T00:00:00Z');
  return Number.isFinite(d.getTime()) ? d : null;
} function monthShort(d){return['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][d.getUTCMonth()];}
async function fetchFloat32(url,expectedLength){const res=await fetch(url); if(!res.ok)throw new Error(`Failed to fetch ${url}: ${res.status}`); const arr=new Float32Array(await res.arrayBuffer()); if(Number.isFinite(expectedLength)&&arr.length!==expectedLength)throw new Error(`${url} length mismatch: got ${arr.length}, expected ${expectedLength}`); return arr;}
async function loadAssets(){const res=await fetch(PHASE17.manifestUrl); if(!res.ok)throw new Error(`Failed to fetch ${PHASE17.manifestUrl}: ${res.status}`); PHASE17.manifest=await res.json(); const rows=Array.isArray(PHASE17.manifest.parcel_ids_in_row_order)?PHASE17.manifest.parcel_ids_in_row_order.length:0; const epochs=Number(PHASE17.manifest.epoch_count||0); const expected=rows*epochs; const bins=PHASE17.manifest.binary_assets||{}; PHASE17.arrays={reversible:await fetchFloat32(String(bins.reversible),expected),irreversible:await fetchFloat32(String(bins.irreversible),expected),total:await fetchFloat32(String(bins.total),expected),rows,epochs};}
function getSeries(row,comp){const key=`${row}:${comp}`; if(PHASE17.cache.has(key))return PHASE17.cache.get(key); const e=PHASE17.arrays.epochs; const start=row*e; const out=Array.from(PHASE17.arrays[comp].slice(start,start+e)); PHASE17.cache.set(key,out); return out;}
function findFeatureProperty(feature,keys){if(!feature||typeof feature.getProperty!=='function')return null; for(const key of keys){try{const v=feature.getProperty(key); if(v!==undefined&&v!==null&&String(v).trim()!=='')return v;}catch(e){}} return null;}
function resolveSelectedParcelId(){try{if(typeof selectedFeature!=='undefined'&&selectedFeature){const raw=findFeatureProperty(selectedFeature,['pnt_id','parcel_id','int_id','id','ID','parcelId','name']); if(raw!==null){const txt=String(raw).trim(); const n=Number(txt); return Number.isFinite(n)?String(Math.trunc(n)):txt;}}}catch(e){} for(const id of ['parcelInfoTitle','rumInfoTitle','selectedParcelTitle']){const txt=q(id)?.textContent?.trim(); if(!txt)continue; const m=txt.match(/(\d{2,})/); if(m)return String(Number(m[1]));} return null;}
function resolveRowForParcel(parcelId){if(!parcelId||!PHASE17.manifest)return null; const map=PHASE17.manifest.parcel_to_row||{}; if(Object.prototype.hasOwnProperty.call(map,String(parcelId)))return Number(map[String(parcelId)]); const n=Number(parcelId); if(Number.isFinite(n)&&Object.prototype.hasOwnProperty.call(map,String(Math.trunc(n))))return Number(map[String(Math.trunc(n))]); return null;}
function computeYRange(chartMode,row){const ranges=(PHASE17.manifest&&PHASE17.manifest.component_ranges_mm)||{}; if(PHASE17.axisMode==='manual'&&Number.isFinite(PHASE17.manualMin)&&Number.isFinite(PHASE17.manualMax)&&PHASE17.manualMax>PHASE17.manualMin)return{min:PHASE17.manualMin,max:PHASE17.manualMax}; if(PHASE17.axisMode==='fixed'){if(chartMode==='irreversible')return ranges.irreversible||{min:-15,max:5}; if(chartMode==='reversible')return ranges.reversible||{min:-80,max:35}; if(chartMode==='total')return ranges.total||{min:-80,max:35}; return{min:Math.min(ranges.total?.min??-80,ranges.irreversible?.min??-15),max:Math.max(ranges.total?.max??35,ranges.irreversible?.max??5)};} let mn=Infinity,mx=-Infinity; const add=(arr)=>{for(const v of arr){if(Number.isFinite(v)){mn=Math.min(mn,v);mx=Math.max(mx,v);}}}; if(row!==null&&row!==undefined){if(chartMode==='irreversible')add(getSeries(row,'irreversible')); else if(chartMode==='reversible')add(getSeries(row,'reversible')); else if(chartMode==='total')add(getSeries(row,'total')); else{add(getSeries(row,'irreversible'));add(getSeries(row,'total'));}} if(!Number.isFinite(mn)||!Number.isFinite(mx)){mn=-10;mx=10;} if(chartMode==='reversible'){mn=Math.min(mn,0);mx=Math.max(mx,0);} const span=Math.max(1e-6,mx-mn); const pad=Math.max(1.0,span*0.12); return{min:mn-pad,max:mx+pad};}
function niceTickStep(span){if(!Number.isFinite(span)||span<=0)return 10; const rough=span/5,pow=Math.pow(10,Math.floor(Math.log10(rough))),r=rough/pow; return(r<=1?1:r<=2?2:r<=5?5:10)*pow;} function xPos(i,n,l,w){return l+w*(n<=1?0:i/(n-1));} function yPos(v,yr,t,h){return t+h*(1-((v-yr.min)/Math.max(1e-9,yr.max-yr.min)));}
function linePath(series,yr,l,t,w,h){let d=''; for(let i=0;i<series.length;i++){const v=series[i]; if(!Number.isFinite(v))continue; const x=xPos(i,series.length,l,w),y=yPos(v,yr,t,h); d+=(d?' L ':'M ')+x.toFixed(2)+' '+y.toFixed(2);} return d;}
function areaBetweenPath(a,b,yr,l,t,w,h){let top=[],bot=[]; for(let i=0;i<a.length;i++){if(!Number.isFinite(a[i])||!Number.isFinite(b[i]))continue; const x=xPos(i,a.length,l,w); top.push([x,yPos(a[i],yr,t,h)]); bot.push([x,yPos(b[i],yr,t,h)]);} if(!top.length)return''; let d=`M ${top[0][0].toFixed(2)} ${top[0][1].toFixed(2)}`; for(let i=1;i<top.length;i++)d+=` L ${top[i][0].toFixed(2)} ${top[i][1].toFixed(2)}`; for(let i=bot.length-1;i>=0;i--)d+=` L ${bot[i][0].toFixed(2)} ${bot[i][1].toFixed(2)}`; return d+' Z';}
function buildMonthTicks(labels){
  const out=[]; let last='';
  for(let i=0;i<labels.length;i++){
    const d=parseDateLabel(labels[i]); if(!d)continue;
    const key=`${d.getUTCFullYear()}-${d.getUTCMonth()}`;
    if(key!==last){
      const m=d.getUTCMonth();
      const isMajor=(m%3===0)||i===0;
      out.push({
        index:i,
        label:isMajor ? `${monthShort(d)}${m===0?' '+d.getUTCFullYear():''}` : '',
        major:isMajor
      });
      last=key;
    }
  }
  if(out.length && out[out.length-1].label===''){
    const d=parseDateLabel(labels[out[out.length-1].index]);
    if(d) out[out.length-1].label=monthShort(d);
  }
  return out;
}
function ensureUi(){const panel=q('epochPanel'); if(!panel)return null; panel.classList.add('phase17-trendline-ready'); let chrome=q('phase17TrendlineChrome'); if(!chrome){chrome=document.createElement('div'); chrome.id='phase17TrendlineChrome'; chrome.innerHTML=`<div id="phase17TrendlineHeader"><div id="phase17TrendlineTitleBlock"><div id="phase17TrendlineTitle">Parcel displacement trendline</div><div id="phase17TrendlineSubtitle">Select a parcel to inspect</div></div><div id="phase17TrendlineControls"><select id="phase17ChartView" class="phase17Select" title="Chart view"><option value="auto">Auto</option><option value="irreversible">Irreversible</option><option value="reversible">Reversible</option><option value="total">Total</option><option value="decomposition">Decomp</option></select><select id="phase17AxisMode" class="phase17Select" title="Y-axis mode"><option value="auto">Auto</option><option value="fixed">Fixed</option><option value="manual">Manual</option></select><div class="phase17AxisStack"><label class="phase17AxisRow"><span>Max</span><input id="phase17AxisMax" class="phase17AxisInput" type="text" value=""/></label><label class="phase17AxisRow"><span>Min</span><input id="phase17AxisMin" class="phase17AxisInput" type="text" value=""/></label></div><button id="phase17PngBtn" title="Export chart to PNG">PNG</button><button id="phase17CloseBtn" title="Hide chart">×</button></div></div><div id="phase17ChartWrap"><svg id="phase17TrendSvg" xmlns="http://www.w3.org/2000/svg"></svg><div id="phase17Placeholder">Select a moving parcel to inspect its time series.</div></div><div id="phase17CurrentDateBadge">—</div>`; panel.appendChild(chrome); const reopen=document.createElement('button'); reopen.id='phase17ChartToggleBtn'; reopen.textContent='Show chart'; reopen.addEventListener('click',()=>{PHASE17.visible=true;panel.classList.remove('phase17-chart-collapsed');renderNow(true);}); panel.appendChild(reopen); q('phase17ChartView').addEventListener('change',e=>{PHASE17.chartView=e.target.value;renderNow(true);}); q('phase17AxisMode').addEventListener('change',e=>{PHASE17.axisMode=e.target.value;applyAxisModeUi();renderNow(true);}); q('phase17AxisMin').addEventListener('change',()=>{PHASE17.manualMin=Number(q('phase17AxisMin').value);renderNow(true);}); q('phase17AxisMax').addEventListener('change',()=>{PHASE17.manualMax=Number(q('phase17AxisMax').value);renderNow(true);}); q('phase17PngBtn').addEventListener('click',exportPng); q('phase17CloseBtn').addEventListener('click',()=>{PHASE17.visible=false;panel.classList.add('phase17-chart-collapsed');});} applyAxisModeUi(); return panel;}
function applyAxisModeUi(){const editable=PHASE17.axisMode==='manual'; for(const id of ['phase17AxisMin','phase17AxisMax']){const el=q(id); if(!el)continue; el.readOnly=!editable; el.dataset.editable=editable?'1':'0';}}
function setTitle(pid,chartMode){const title=q('phase17TrendlineTitle'),sub=q('phase17TrendlineSubtitle'); if(!title||!sub)return; if(!pid||PHASE17.selectedRow===null||PHASE17.selectedRow===undefined){title.textContent='Parcel displacement trendline';sub.textContent='Select a moving parcel to inspect';return;} title.textContent=`Parcel ${pid} · displacement trendline`; if(chartMode==='irreversible')sub.textContent='Permanent subsidence component [mm]'; else if(chartMode==='reversible')sub.textContent='Seasonal reversible component around datum 0 [mm]'; else if(chartMode==='decomposition')sub.textContent='Irreversible baseline + total trajectory; fill = reversible gap'; else sub.textContent='Total displacement trajectory [mm]';}
function addSvg(tag,attrs,parent){const el=document.createElementNS('http://www.w3.org/2000/svg',tag); for(const[k,v]of Object.entries(attrs||{}))el.setAttribute(k,String(v)); parent.appendChild(el); return el;}
function renderNow(force=false){
  disableLegacyTrendlineShell();
  const panel=ensureUi();
  if(!panel||!PHASE17.manifest||!PHASE17.arrays)return;

  const pid=resolveSelectedParcelId(), row=resolveRowForParcel(pid);
  PHASE17.selectedParcelId=pid;
  PHASE17.selectedRow=Number.isFinite(row)?row:null;

  const chartMode=modeToChart(currentModeName());
  setTitle(pid,chartMode);

  const key=[pid,row,chartMode,currentEpoch(),PHASE17.axisMode,PHASE17.chartView,PHASE17.manualMin,PHASE17.manualMax,PHASE17.visible].join('|');
  if(!force&&key===PHASE17.lastKey)return;
  PHASE17.lastKey=key;

  const placeholder=q('phase17Placeholder'), svg=q('phase17TrendSvg');
  if(!svg)return;

  if(!pid||row===null||row===undefined||!Number.isFinite(row)){
    if(placeholder){
      placeholder.style.display='flex';
      placeholder.textContent=pid?'No displacement series for this parcel (blank / no-data parcel).':'Select a moving parcel to inspect its time series.';
    }
    svg.innerHTML='';
    const badge=q('phase17CurrentDateBadge'); if(badge) badge.textContent='—';
    return;
  }
  if(placeholder)placeholder.style.display='none';

  const labels=PHASE17.manifest.epoch_labels||[];
  const width=svg.clientWidth||620;
  const height=svg.clientHeight||344;
  svg.setAttribute('viewBox',`0 0 ${width} ${height}`);
  svg.innerHTML='';

  // Proto1-aligned chart geometry.
  const padL=45, padR=45;
  const axisFromBottom=108;
  const yearFromBottom=82;
  const cursorBottomFromBottom=58;
  const padT=8;
  const plotBottomY=height-axisFromBottom;
  const plotH=Math.max(120,plotBottomY-padT);
  const plotW=Math.max(50,width-padL-padR);
  const plot={x:padL,y:padT,w:plotW,h:plotH};
  const yr=computeYRange(chartMode,row);

  q('phase17AxisMin').value=Number.isFinite(yr.min)?yr.min.toFixed(0):'';
  q('phase17AxisMax').value=Number.isFinite(yr.max)?yr.max.toFixed(0):'';

  // White chart area
  addSvg('rect',{x:plot.x,y:plot.y,width:plot.w,height:plot.h,rx:8,ry:8,fill:'rgba(255,255,255,0.96)',stroke:'rgba(0,0,0,0.45)','stroke-width':1},svg);

  const step=niceTickStep(yr.max-yr.min),y0=Math.ceil(yr.min/step)*step;
  for(let v=y0;v<=yr.max+1e-9;v+=step){
    const y=yPos(v,yr,plot.y,plot.h);
    addSvg('line',{x1:plot.x,x2:plot.x+plot.w,y1:y,y2:y,stroke:'rgba(0,0,0,0.12)','stroke-width':1},svg);
  }

  if(yr.min<=0 && yr.max>=0){
    const zy=yPos(0,yr,plot.y,plot.h);
    addSvg('line',{x1:plot.x,x2:plot.x+plot.w,y1:zy,y2:zy,stroke:'rgba(0,0,0,0.34)','stroke-width':1.8},svg);
  }

  const series={
    irreversible:getSeries(row,'irreversible'),
    reversible:getSeries(row,'reversible'),
    total:getSeries(row,'total')
  };

  if(chartMode==='decomposition'){
    addSvg('path',{d:areaBetweenPath(series.total,series.irreversible,yr,plot.x,plot.y,plot.w,plot.h),fill:'rgba(124,245,255,0.24)',stroke:'none'},svg);
  }

  const drawSeries=(name,color,w)=>{
    addSvg('path',{d:linePath(series[name],yr,plot.x,plot.y,plot.w,plot.h),fill:'none',stroke:color,'stroke-width':w,'stroke-linejoin':'round','stroke-linecap':'round'},svg);
  };
  if(chartMode==='irreversible')drawSeries('irreversible','#ff375f',2.2);
  else if(chartMode==='reversible')drawSeries('reversible','#2d6fff',2.0);
  else if(chartMode==='total')drawSeries('total','#1f4aa8',2.4);
  else{drawSeries('irreversible','#ff375f',2.0);drawSeries('total','#2355d9',2.2);}

  // Chart frame on top
  addSvg('rect',{x:plot.x,y:plot.y,width:plot.w,height:plot.h,rx:8,ry:8,fill:'none',stroke:'rgba(0,0,0,0.45)','stroke-width':1},svg);

  // Y-axis labels outside white chart, white like Proto1.
  for(let v=y0;v<=yr.max+1e-9;v+=step){
    const y=yPos(v,yr,plot.y,plot.h);
    const txt=addSvg('text',{x:plot.x-6,y:y+3,'text-anchor':'end','font-size':10,fill:'rgba(255,255,255,0.88)','font-weight':700},svg);
    txt.textContent=String(Math.round(v));
  }
  const yl=addSvg('text',{transform:`translate(${Math.max(12,plot.x-38)} ${plot.y+plot.h/2}) rotate(-90)`,'text-anchor':'middle','font-size':11,'font-weight':700,fill:'rgba(255,255,255,0.88)'},svg);
  yl.textContent='Displacement [mm]';

  // Month ticks/date axis, below chart and above slider.
  for(const tick of buildMonthTicks(labels)){
    const x=xPos(tick.index,labels.length,plot.x,plot.w);
    addSvg('line',{x1:x,x2:x,y1:plotBottomY,y2:plotBottomY+(tick.major?8:5),stroke:tick.major?'rgba(255,255,255,0.78)':'rgba(255,255,255,0.46)','stroke-width':1},svg);
    if(tick.label){
      const t=addSvg('text',{x:x,y:height-yearFromBottom,'text-anchor':'middle','font-size':11,fill:'rgba(255,255,255,0.92)'},svg);
      t.textContent=tick.label;
    }
  }

  const epochIdx=clamp(currentEpoch(),0,labels.length-1);
  const cursorX=xPos(epochIdx,labels.length,plot.x,plot.w);
  const epochLabel=epochDateString(labels[epochIdx]||`Epoch ${epochIdx+1}`);

  addSvg('line',{x1:cursorX,x2:cursorX,y1:plot.y,y2:height-cursorBottomFromBottom,stroke:'rgba(0,95,255,0.92)','stroke-width':1.3},svg);

  const badge=q('phase17CurrentDateBadge');
  if(badge){
    badge.textContent=epochLabel;
    badge.style.left=`${cursorX}px`;
  }

  const addCallout=(text,anchorX,anchorY,fill,stroke,preferSide)=>{
    if(!Number.isFinite(anchorX)||!Number.isFinite(anchorY))return;

    // Measure text first.
    const temp=addSvg('text',{x:0,y:0,'font-size':11,'font-weight':800,fill:'#fff'},svg);
    temp.textContent=text;
    const bb=temp.getBBox();
    svg.removeChild(temp);

    const boxW=Math.min(bb.width+20,plot.w-14);
    const boxH=24;
    const gap=12;
    const side = preferSide || (anchorX > plot.x + plot.w*0.62 ? 'left' : 'right');

    let boxX = side==='left' ? anchorX - boxW - gap : anchorX + gap;
    let boxY = anchorY - boxH/2;
    boxX = clamp(boxX,plot.x+4,plot.x+plot.w-boxW-4);
    boxY = clamp(boxY,plot.y+6,plot.y+plot.h-boxH-6);

    const edgeX = side==='left' ? boxX+boxW : boxX;
    const edgeY = boxY + boxH/2;

    const g=addSvg('g',{},svg);
    addSvg('line',{x1:anchorX,y1:anchorY,x2:edgeX,y2:edgeY,stroke:stroke,'stroke-width':1.25},g);
    addSvg('circle',{cx:anchorX,cy:anchorY,r:3.2,fill:stroke,stroke:'rgba(255,255,255,0.75)','stroke-width':0.8},g);
    addSvg('rect',{x:boxX,y:boxY,rx:10,ry:10,width:boxW,height:boxH,fill:fill,stroke:stroke,'stroke-width':1.5},g);
    const tt=addSvg('text',{x:boxX+10,y:boxY+15,'font-size':11,'font-weight':800,fill:'#fff'},g);
    tt.textContent=text;
  };

  const v=(name)=>Number.isFinite(series[name][epochIdx])?series[name][epochIdx]:NaN;
  const vi=Number((PHASE17.manifest.vi_by_parcel||{})[String(pid)]);

  if(chartMode==='irreversible'){
    const val=v('irreversible');
    const yVal=yPos(val,yr,plot.y,plot.h);
    const trendY=yVal; // vI is persistent; no separate fitted series here.
    addCallout(`Trend: ${fmtVi(vi)}`,cursorX,trendY,'#202430','#ff375f',cursorX>plot.x+plot.w*0.55?'left':'right');
    addCallout(`Displacement ${fmt(val)} mm`,cursorX,yVal,'#24375e','#2f7cf6',cursorX>plot.x+plot.w*0.55?'left':'right');
  } else if(chartMode==='reversible'){
    const val=v('reversible');
    addCallout(`Displacement ${fmt(val)} mm`,cursorX,yPos(val,yr,plot.y,plot.h),'#24375e','#2f7cf6',cursorX>plot.x+plot.w*0.55?'left':'right');
  } else if(chartMode==='total'){
    const val=v('total');
    addCallout(`Displacement ${fmt(val)} mm`,cursorX,yPos(val,yr,plot.y,plot.h),'#24375e','#2f7cf6',cursorX>plot.x+plot.w*0.55?'left':'right');
  } else {
    const irr=v('irreversible');
    const tot=v('total');
    const side=cursorX>plot.x+plot.w*0.55?'left':'right';
    addCallout(`Irreversible ${fmt(irr)} mm`,cursorX,yPos(irr,yr,plot.y,plot.h),'#202430','#ff375f',side);
    addCallout(`Total ${fmt(tot)} mm`,cursorX,yPos(tot,yr,plot.y,plot.h),'#24375e','#2f7cf6',side);
  }
}
async function exportPng(){const svg=q('phase17TrendSvg'); if(!svg)return; const clone=svg.cloneNode(true),width=svg.clientWidth||620,height=svg.clientHeight||220; clone.setAttribute('width',width); clone.setAttribute('height',height); const blob=new Blob([new XMLSerializer().serializeToString(clone)],{type:'image/svg+xml;charset=utf-8'}); const url=URL.createObjectURL(blob),img=new Image(); img.onload=()=>{const canvas=document.createElement('canvas'); canvas.width=width; canvas.height=height; const ctx=canvas.getContext('2d'); ctx.fillStyle='#f4f4f4'; ctx.fillRect(0,0,width,height); ctx.drawImage(img,0,0); URL.revokeObjectURL(url); const link=document.createElement('a'); link.download=`parcel_${PHASE17.selectedParcelId||'parcel'}_trendline.png`; link.href=canvas.toDataURL('image/png'); link.click();}; img.src=url;}
function installHooks(){disableLegacyTrendlineShell();q('epochSlider')?.addEventListener('input',()=>renderNow(true)); q('parcelModeSelect')?.addEventListener('change',()=>renderNow(true)); if(typeof applyMode==='function'&&!applyMode.__phase17Wrapped){const orig=applyMode; window.applyMode=async function(...args){const out=await orig.apply(this,args); renderNow(true); return out;}; window.applyMode.__phase17Wrapped=true;} const poll=()=>{renderNow(false); requestAnimationFrame(poll);}; requestAnimationFrame(poll);}
async function mainPhase17(){try{ensureUi(); await loadAssets(); installHooks(); renderNow(true); console.log('Phase17 trendline ready');}catch(error){console.error('Phase17 trendline failed',error); const p=q('phase17Placeholder'); if(p){p.style.display='flex'; p.textContent='Trendline failed to load. See console.';}}}

// ---- Phase17C compact/expanded behavior override ----
// This block is deliberately inside the Phase17 IIFE, so it can access PHASE17,
// renderNow(), resolveSelectedParcelId(), etc.

PHASE17.chartOpen = false;

const phase17OldResolveSelectedParcelId = resolveSelectedParcelId;
resolveSelectedParcelId = function(){
  let pid = null;
  try { pid = phase17OldResolveSelectedParcelId(); } catch(e) {}
  if(pid) return pid;

  // Proto2 popup header currently renders "Parcel 13991"; do not depend on a fixed id/class.
  const roots = Array.from(document.querySelectorAll(
    '#parcelInfoPanel,#parcelInfoPopup,#parcelPopup,#infoPopup,#rumInfoPanel,' +
    '.parcel-info,.parcelInfo,.info-popup,.popup,.card,.panel'
  ));
  for(const root of roots){
    const txt = root && root.textContent ? root.textContent.trim() : '';
    if(!txt) continue;
    let m = txt.match(/\bParcel\s+(\d{2,})\b/i);
    if(m) return String(Number(m[1]));
    m = txt.match(/\bparcel_id\s+(\d{2,})\b/i);
    if(m) return String(Number(m[1]));
  }

  // Last fallback: visible text around the clicked popup, but avoid scanning huge body first.
  const bodyTxt = document.body?.innerText || '';
  const m = bodyTxt.match(/\bParcel\s+(\d{2,})\b/i);
  if(m) return String(Number(m[1]));
  return null;
};

function phase17OpenChart(){
  // Button only exists for a selected parcel, so open the shell regardless.
  // If row resolution fails, renderNow will show the no-data placeholder instead of silently doing nothing.
  PHASE17.chartOpen = true;
  const panel = q('epochPanel');
  if(panel){
    panel.classList.add('phase17-chart-open');
    panel.classList.remove('trendlineOpen');
    panel.classList.remove('phase17-chart-collapsed');
  }
  disableLegacyTrendlineShell();
  PHASE17.lastKey = '';
  renderNow(true);
}

function phase17CloseChart(){
  PHASE17.chartOpen = false;
  const panel = q('epochPanel');
  if(panel){
    panel.classList.remove('phase17-chart-open');
    panel.classList.remove('trendlineOpen');
  }
  disableLegacyTrendlineShell();
  PHASE17.lastKey = '';
  renderNow(true);
}

// Let any old inline popup handlers route into the Phase17 shell.
try { window.openTrendlinePlaceholder = phase17OpenChart; } catch(e) {}
try { window.drawTrendlinePlaceholder = function(){}; } catch(e) {}
try { window.closeTrendlinePlaceholder = phase17CloseChart; } catch(e) {}
try { window.openParcelTrendline = phase17OpenChart; } catch(e) {}
try { window.openTrendline = phase17OpenChart; } catch(e) {}
try { window.showTrendlinePanel = phase17OpenChart; } catch(e) {}

const phase17OldDisableLegacyTrendlineShell = disableLegacyTrendlineShell;
disableLegacyTrendlineShell = function(){
  phase17OldDisableLegacyTrendlineShell();
  const panel = q('epochPanel');
  if(panel && !PHASE17.chartOpen){
    panel.classList.remove('phase17-chart-open');
    panel.classList.remove('trendlineOpen');
  }
};

const phase17OldEnsureUi = ensureUi;
ensureUi = function(){
  const panel = phase17OldEnsureUi();
  if(panel){
    panel.classList.toggle('phase17-chart-open', !!PHASE17.chartOpen);
    if(!PHASE17.chartOpen) panel.classList.remove('trendlineOpen');
  }
  return panel;
};

const phase17OldRenderNow = renderNow;
renderNow = function(force=false){
  const panel = q('epochPanel');
  if(panel){
    panel.classList.toggle('phase17-chart-open', !!PHASE17.chartOpen);
    if(!PHASE17.chartOpen) panel.classList.remove('trendlineOpen');
  }

  if(!PHASE17.chartOpen){
    disableLegacyTrendlineShell();
    return;
  }
  return phase17OldRenderNow(force);
};

const phase17OldInstallHooks = installHooks;
installHooks = function(){
  phase17OldInstallHooks();

  const closeBtn = q('phase17CloseBtn');
  if(closeBtn && !closeBtn.dataset.phase17cBound){
    closeBtn.dataset.phase17cBound = '1';
    closeBtn.addEventListener('click', function(evt){
      evt.preventDefault();
      evt.stopPropagation();
      phase17CloseChart();
    }, true);
  }

  // Catch the actual yellow popup button by text, not by assumed class/id.
  if(!document.documentElement.dataset.phase17cTrendlineClickBound){
    document.documentElement.dataset.phase17cTrendlineClickBound = '1';
    document.addEventListener('click', function(evt){
      const target = evt.target;
      const clickable = target && target.closest
        ? target.closest('button,a,[role="button"],.popupAction,.actionButton,.yellowBtn,.btn')
        : null;
      if(!clickable) return;

      const txt = (clickable.textContent || clickable.getAttribute('title') || clickable.getAttribute('aria-label') || '').trim().toLowerCase();
      const idc = `${clickable.id || ''} ${clickable.className || ''} ${clickable.dataset?.action || ''}`.toLowerCase();

      if(txt.includes('open trendline') || txt === 'trendline' || idc.includes('trendline')){
        evt.preventDefault();
        evt.stopPropagation();
        phase17OpenChart();
      }
    }, true);
  }

  const popupCloseCandidates = ['parcelInfoClose','parcelInfoCloseBtn','rumInfoClose','rumInfoCloseBtn','infoPopupCloseBtn'];
  for(const id of popupCloseCandidates){
    const el = q(id);
    if(el && !el.dataset.phase17cBound){
      el.dataset.phase17cBound = '1';
      el.addEventListener('click', phase17CloseChart, true);
    }
  }

  const panel = q('epochPanel');
  if(panel){
    panel.classList.remove('phase17-chart-open');
    panel.classList.remove('trendlineOpen');
  }
};
// ---- /Phase17C compact/expanded behavior override ----


if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',mainPhase17); else mainPhase17();
})();


</script>
'''


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


def patch_html(html: str) -> str:
    html = strip_old_block(html)
    html = sanitize_meta_block(html)
    if 'id="phase17TrendlineStyle"' not in html:
        if "</head>" not in html:
            fail("Could not find </head>")
        html = html.replace("</head>", TRENDLINE_STYLE + "\n</head>", 1)
    if 'id="phase17TrendlineScript"' not in html:
        block = START_MARKER + "\n" + TRENDLINE_SCRIPT + "\n" + END_MARKER + "\n"
        if "</body>" not in html:
            fail("Could not find </body>")
        html = html.replace("</body>", block + "</body>", 1)
    html = html.replace("proto2_m1_multimode_deformation_viewer_16e", "proto2_m1_multimode_deformation_viewer_17_fixed7")
    html = html.replace("Phase16E", "Phase17")
    html = html.replace("PHASE 16E", "PHASE 17")
    return html


def main() -> None:
    print("\n=== PROTO2 PHASE 17: TRENDLINE VIEWER ===")
    print(f"Project root: {PROJECT_ROOT}")
    require(SOURCE_HTML, "Phase16E HTML")
    TREND_ASSET_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_CESIUM.mkdir(parents=True, exist_ok=True)
    RUN_RECORDS.mkdir(parents=True, exist_ok=True)
    manifest = build_trend_manifest()
    write_json(TREND_MANIFEST, manifest)
    ok(f"wrote {TREND_MANIFEST}")
    html = SOURCE_HTML.read_text(encoding="utf-8", errors="replace")
    html = patch_html(html)
    HTML_OUT.write_text(html, encoding="utf-8")
    ok(f"wrote {HTML_OUT}")
    inherited = read_json(SOURCE_SUMMARY) if SOURCE_SUMMARY.exists() else {}
    summary = {
        "product": "proto2_m1_multimode_deformation_viewer_17_fixed7",
        "source_html": str(SOURCE_HTML),
        "output_html": str(HTML_OUT),
        "trend_manifest": str(TREND_MANIFEST),
        "inherited_product": inherited.get("product"),
        "epoch_count": manifest["epoch_count"],
        "moving_parcels": len(manifest["parcel_ids_in_row_order"]),
        "chart_views": ["auto", "irreversible", "reversible", "total", "decomposition"],
        "axis_modes": ["auto", "fixed", "manual"],
    }
    write_json(SUMMARY_OUT, summary)
    write_json(REPORT_JSON_OUT, summary)
    REPORT_TXT_OUT.write_text(
        "PROTO2 PHASE 17: TRENDLINE VIEWER\n"
        f"Project root: {PROJECT_ROOT}\n"
        f"Source HTML: {SOURCE_HTML}\n"
        f"Output HTML: {HTML_OUT}\n"
        f"Trend manifest: {TREND_MANIFEST}\n"
        f"Moving parcels: {len(manifest['parcel_ids_in_row_order'])}\n"
        f"Epochs: {manifest['epoch_count']}\n",
        encoding="utf-8",
    )
    ok(f"wrote {SUMMARY_OUT}")
    ok(f"wrote {REPORT_JSON_OUT}")
    print("\n=== PHASE 17 RESULT: PASS ===")


if __name__ == "__main__":
    main()
