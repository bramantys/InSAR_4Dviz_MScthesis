from pathlib import Path
import json
import math
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from _proto2_config import load_project_config, output_cesium_dir, output_data_dir, stage_records_dir


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = load_project_config(PROJECT_ROOT)
OUTPUT_DATA = output_data_dir(PROJECT_ROOT, CONFIG)
RUN_RECORDS = stage_records_dir(PROJECT_ROOT, CONFIG)

MOVING_INDEX_PATH = OUTPUT_DATA / "moving_parcel_index.parquet"
DISP_LONG_PARQUET = OUTPUT_DATA / "parcel_displacement_long.parquet"
DISP_LONG_CSV = OUTPUT_DATA / "parcel_displacement_long.csv"
MATRIX_NPZ = OUTPUT_DATA / "parcel_displacement_matrices_float32.npz"
ANIMATION_MANIFEST = OUTPUT_DATA / "parcel_animation_manifest.json"

PARCEL_VI_BIN = OUTPUT_DATA / "parcel_vi_f32.bin"
PARCEL_COLOR_SCALES_JSON = OUTPUT_DATA / "parcel_color_scales.json"
PARCEL_COLOR_SUMMARY_JSON = OUTPUT_DATA / "parcel_color_summary.json"

REPORT_TXT_OUT = RUN_RECORDS / "parcel_color_scales_report.txt"
REPORT_JSON_OUT = RUN_RECORDS / "parcel_color_scales_report.json"

ROUND_NUMERIC_DIGITS = 6

# Same core visual/color constants as Proto1 Step 18.  See 18_build_viewer_tuning.py.
COLOR_SCALE_UNCERTAINTY_PERCENTILE = 75.0
COLOR_SCALE_PERCENTILE = 98.0
COLOR_SCALE_EXTREME_PERCENTILE = 99.5
COLOR_SCALE_NEAR_ZERO_STEP = 0.5
COLOR_SCALE_MIN_ACTIVE_FRACTION = 0.01
COLOR_SCALE_FALLBACK_LIMIT = 10.0
COLOR_SCALE_MIN_SPAN = 2.0
COLOR_SCALE_ZERO_POSITION_DAMPING = 0.60
COLOR_SCALE_ZERO_POSITION_MIN_PCT = 35.0
COLOR_SCALE_ZERO_POSITION_MAX_PCT = 72.0
COLOR_SCALE_STABLE_BAND_WIDTH_PCT = 16.0

VERTICAL_COLOR_PALETTE_NAME = "RdBu_11"
VERTICAL_COLOR_PALETTE_11 = [
    "#67001f",  # far-far negative / clipped subsidence or downward displacement
    "#b2182b",  # far negative / P98
    "#ef8a62",  # mid negative
    "#fddbc7",  # near negative shoulder
    "#f7f7f7",  # -tau / stable boundary low
    "#f7f7f7",  # zero reference
    "#f7f7f7",  # +tau / stable boundary high
    "#d1e5f0",  # near positive shoulder
    "#67a9cf",  # mid positive
    "#2166ac",  # far positive / P98
    "#053061",  # far-far positive / P99.5
]


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    sys.exit(1)


def ok(message: str) -> None:
    print(f"[OK] {message}")


def warn(message: str) -> None:
    print(f"[WARN] {message}")


def require(path: Path, label: str) -> None:
    if not path.exists():
        fail(f"Missing {label}: {path}")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def percentile(values: np.ndarray, p: float) -> Optional[float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(np.percentile(arr, p))


def ceil_to_step(value: float, step: float) -> float:
    if not math.isfinite(float(value)) or value <= 0 or step <= 0:
        return 0.0
    return float(math.ceil(float(value) / float(step)) * float(step))


def round_near_zero_threshold(tau_raw: float, step: float = COLOR_SCALE_NEAR_ZERO_STEP) -> float:
    return ceil_to_step(tau_raw, step)


def round_p98_limit(limit_raw: float) -> float:
    if not math.isfinite(float(limit_raw)) or limit_raw <= 0:
        return 0.0
    if limit_raw < 10.0:
        step = 1.0
    elif limit_raw < 20.0:
        step = 2.0
    elif limit_raw <= 100.0:
        step = 5.0
    else:
        step = 10.0
    return ceil_to_step(limit_raw, step)


def derive_extreme_limit(base_limit: float, extreme_raw: float, tau: float) -> float:
    base = max(float(base_limit), float(tau), 1e-9)
    if math.isfinite(float(extreme_raw)) and extreme_raw > 0.0:
        extreme = round_p98_limit(float(extreme_raw))
    else:
        extreme = 0.0
    if extreme <= base:
        extreme = round_p98_limit(base * 1.15)
    if extreme <= base:
        extreme = base + max(float(tau), 0.5)
    return float(extreme)


def clamp(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


def interp(a: float, b: float, f: float) -> float:
    return float(a) + (float(b) - float(a)) * float(f)


def fmt_limit(value: float) -> str:
    v = float(value)
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:g}"


def build_adaptive_stops_11(
    L_neg: float,
    L_pos: float,
    tau: float,
    L_neg_extreme: Optional[float] = None,
    L_pos_extreme: Optional[float] = None,
    unit_value_key: str = "value",
) -> List[Dict[str, Any]]:
    L_neg = max(float(L_neg), float(tau), 1e-9)
    L_pos = max(float(L_pos), float(tau), 1e-9)
    tau = max(float(tau), 0.0)

    if L_neg_extreme is None:
        L_neg_extreme = derive_extreme_limit(L_neg, 0.0, tau)
    if L_pos_extreme is None:
        L_pos_extreme = derive_extreme_limit(L_pos, 0.0, tau)

    L_neg_extreme = max(float(L_neg_extreme), L_neg)
    L_pos_extreme = max(float(L_pos_extreme), L_pos)

    c = VERTICAL_COLOR_PALETTE_11

    zero_raw = 100.0 * L_neg / max(L_neg + L_pos, 1e-9)
    zero_damped = 50.0 + COLOR_SCALE_ZERO_POSITION_DAMPING * (zero_raw - 50.0)
    zero_pct = clamp(zero_damped, COLOR_SCALE_ZERO_POSITION_MIN_PCT, COLOR_SCALE_ZERO_POSITION_MAX_PCT)
    stable_width = clamp(COLOR_SCALE_STABLE_BAND_WIDTH_PCT, 10.0, 22.0)
    stable_left = clamp(zero_pct - stable_width / 2.0, 2.0, 96.0)
    stable_right = clamp(zero_pct + stable_width / 2.0, 4.0, 98.0)
    if stable_right <= stable_left:
        stable_right = min(98.0, stable_left + stable_width)

    neg_mid_mag = interp(L_neg, tau, 0.50)
    pos_mid_value = interp(tau, L_pos, 0.50)
    near_neg_mag = min(L_neg, max(3.0 * tau, tau + 2.0 * COLOR_SCALE_NEAR_ZERO_STEP))
    near_pos_value = min(L_pos, max(2.0 * tau, tau + 2.0 * COLOR_SCALE_NEAR_ZERO_STEP))

    def neg_pct_for_mag(mag: float) -> float:
        denom = max(L_neg_extreme - tau, 1e-9)
        return stable_left * clamp((L_neg_extreme - float(mag)) / denom, 0.0, 1.0)

    def pos_pct_for_value(value: float) -> float:
        denom = max(L_pos_extreme - tau, 1e-9)
        return stable_right + (100.0 - stable_right) * clamp((float(value) - tau) / denom, 0.0, 1.0)

    raw_stops = [
        (-L_neg_extreme, c[0],  "clipped_extreme_negative", 0.0),
        (-L_neg,         c[1],  "far_negative",             neg_pct_for_mag(L_neg)),
        (-neg_mid_mag,   c[2],  "moderate_negative",        neg_pct_for_mag(neg_mid_mag)),
        (-near_neg_mag,  c[3],  "near_zero_negative",       neg_pct_for_mag(near_neg_mag)),
        (-tau,           c[4],  "stable_boundary_low",      stable_left),
        (0.0,            c[5],  "zero_reference",           zero_pct),
        (tau,            c[6],  "stable_boundary_high",     stable_right),
        (near_pos_value, c[7],  "near_zero_positive",       pos_pct_for_value(near_pos_value)),
        (pos_mid_value,  c[8],  "moderate_positive",        pos_pct_for_value(pos_mid_value)),
        (L_pos,          c[9],  "far_positive",             pos_pct_for_value(L_pos)),
        (L_pos_extreme,  c[10], "clipped_extreme_positive", 100.0),
    ]

    out: List[Dict[str, Any]] = []
    for value, color, role, pct in raw_stops:
        item = {
            "value": round(float(value), ROUND_NUMERIC_DIGITS),
            unit_value_key: round(float(value), ROUND_NUMERIC_DIGITS),
            "color": color,
            "role": role,
            "position_pct": round(float(pct), 3),
        }
        out.append(item)
    return sorted(out, key=lambda x: x["value"])


def build_adaptive_legend(
    L_neg: float,
    L_pos: float,
    tau: float,
    L_neg_extreme: float,
    L_pos_extreme: float,
    title: str,
    unit: str,
    unit_value_key: str,
) -> Dict[str, Any]:
    stops = build_adaptive_stops_11(L_neg, L_pos, tau, L_neg_extreme, L_pos_extreme, unit_value_key)
    by_role = {s["role"]: s for s in stops}
    zero_pct = by_role.get("zero_reference", {}).get("position_pct", 50.0)
    stable_left = by_role.get("stable_boundary_low", {}).get("position_pct", 42.0)
    stable_right = by_role.get("stable_boundary_high", {}).get("position_pct", 58.0)
    far_neg = by_role.get("far_negative", {})
    neg_mid = by_role.get("moderate_negative", {})
    pos_mid = by_role.get("moderate_positive", {})
    far_pos = by_role.get("far_positive", {})

    far_neg_value = float(far_neg.get("value", -float(L_neg)))
    far_pos_value = float(far_pos.get("value", float(L_pos)))
    labels = [
        (far_neg_value, f"≤−{fmt_limit(abs(far_neg_value))}", far_neg.get("position_pct", 6.0), "far_minus_label"),
        (float(neg_mid.get("value", -float(L_neg))), f"−{fmt_limit(abs(float(neg_mid.get('value', -float(L_neg)))))}", neg_mid.get("position_pct", stable_left * 0.5), "mid_minus_label"),
        (-float(tau), f"−{fmt_limit(tau)}", stable_left, "stable_min_label"),
        (0.0, "0", zero_pct, "zero_label"),
        (float(tau), f"+{fmt_limit(tau)}", stable_right, "stable_plus_label"),
        (float(pos_mid.get("value", float(L_pos))), f"+{fmt_limit(float(pos_mid.get('value', float(L_pos))))}", pos_mid.get("position_pct", stable_right + (100.0 - stable_right) * 0.5), "mid_plus_label"),
        (far_pos_value, f"≥+{fmt_limit(far_pos_value)}", far_pos.get("position_pct", 90.0), "far_plus_label"),
    ]

    return {
        "title": title,
        "unit": unit,
        "labels": [
            {
                "value": round(float(value), ROUND_NUMERIC_DIGITS),
                unit_value_key: round(float(value), ROUND_NUMERIC_DIGITS),
                "label": label,
                "position_pct": round(float(position), 3),
                "role": role,
            }
            for value, label, position, role in labels
        ],
        "readability_note": "Seven labels are shown for readability; P99.5 extremes are encoded as end colours but not labelled at 0%/100%.",
    }


def derive_generic_scale(
    values: np.ndarray,
    *,
    title: str,
    unit: str,
    value_field: str,
    mode: str,
    tau: float,
    unit_value_key: str,
    negative_label: str,
    positive_label: str,
    tau_source: str,
) -> Dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        fail(f"No finite values available for color scale: {title}")

    sub_values = arr[arr < -tau]
    pos_values = arr[arr > tau]
    n_valid = arr.size
    n_sub = sub_values.size
    n_pos = pos_values.size
    neg_active = (n_sub / max(n_valid, 1)) >= COLOR_SCALE_MIN_ACTIVE_FRACTION
    pos_active = (n_pos / max(n_valid, 1)) >= COLOR_SCALE_MIN_ACTIVE_FRACTION

    if neg_active:
        abs_sub = np.abs(sub_values)
        L_neg_raw = percentile(abs_sub, COLOR_SCALE_PERCENTILE) or 0.0
        L_neg_far_raw = percentile(abs_sub, COLOR_SCALE_EXTREME_PERCENTILE) or L_neg_raw
        L_neg = round_p98_limit(L_neg_raw)
    else:
        L_neg_raw = 0.0
        L_neg_far_raw = 0.0
        L_neg = 0.0

    if pos_active:
        L_pos_raw = percentile(pos_values, COLOR_SCALE_PERCENTILE) or 0.0
        L_pos_far_raw = percentile(pos_values, COLOR_SCALE_EXTREME_PERCENTILE) or L_pos_raw
        L_pos = round_p98_limit(L_pos_raw)
    else:
        L_pos_raw = 0.0
        L_pos_far_raw = 0.0
        L_pos = 0.0

    if neg_active and not pos_active:
        L_pos = max(5.0 * tau, tau, COLOR_SCALE_MIN_SPAN)
    elif pos_active and not neg_active:
        L_neg = max(5.0 * tau, tau, COLOR_SCALE_MIN_SPAN)
    elif not neg_active and not pos_active:
        L_neg = max(5.0 * tau, tau, COLOR_SCALE_FALLBACK_LIMIT)
        L_pos = max(5.0 * tau, tau, COLOR_SCALE_FALLBACK_LIMIT)

    L_neg = max(float(L_neg), float(tau), 1e-9)
    L_pos = max(float(L_pos), float(tau), 1e-9)
    L_neg_extreme = derive_extreme_limit(L_neg, L_neg_far_raw, tau)
    L_pos_extreme = derive_extreme_limit(L_pos, L_pos_far_raw, tau)

    stops = build_adaptive_stops_11(L_neg, L_pos, tau, L_neg_extreme, L_pos_extreme, unit_value_key)
    legend = build_adaptive_legend(L_neg, L_pos, tau, L_neg_extreme, L_pos_extreme, title, unit, unit_value_key)

    return {
        "mode": mode,
        "title": title,
        "unit": unit,
        "value_field": value_field,
        "recommended_min": round(float(-L_neg_extreme), ROUND_NUMERIC_DIGITS),
        "recommended_center": 0.0,
        "recommended_max": round(float(L_pos_extreme), ROUND_NUMERIC_DIGITS),
        "near_zero_threshold": round(float(tau), ROUND_NUMERIC_DIGITS),
        "tau": round(float(tau), ROUND_NUMERIC_DIGITS),
        "tau_source": tau_source,
        "L_negative": round(float(L_neg), ROUND_NUMERIC_DIGITS),
        "L_positive": round(float(L_pos), ROUND_NUMERIC_DIGITS),
        "L_negative_extreme": round(float(L_neg_extreme), ROUND_NUMERIC_DIGITS),
        "L_positive_extreme": round(float(L_pos_extreme), ROUND_NUMERIC_DIGITS),
        "negative_active": bool(neg_active),
        "positive_active": bool(pos_active),
        "n_valid": int(n_valid),
        "n_negative": int(n_sub),
        "n_positive": int(n_pos),
        "data_min": round(float(np.nanmin(arr)), ROUND_NUMERIC_DIGITS),
        "data_p02": round(float(np.nanpercentile(arr, 2)), ROUND_NUMERIC_DIGITS),
        "data_p50": round(float(np.nanpercentile(arr, 50)), ROUND_NUMERIC_DIGITS),
        "data_p98": round(float(np.nanpercentile(arr, 98)), ROUND_NUMERIC_DIGITS),
        "data_max": round(float(np.nanmax(arr)), ROUND_NUMERIC_DIGITS),
        "palette_name": VERTICAL_COLOR_PALETTE_NAME,
        "palette_colours": VERTICAL_COLOR_PALETTE_11,
        "color_stop_count": 11,
        "color_stops": stops,
        "legend": legend,
        "interpretation": {
            "negative": negative_label,
            "zero": "near datum / near reference value",
            "positive": positive_label,
            "red_side": negative_label,
            "blue_side": positive_label,
        },
        "clipping": {
            "low": f"values <= -{fmt_limit(L_neg_extreme)} {unit} use darkest red",
            "high": f"values >= +{fmt_limit(L_pos_extreme)} {unit} use darkest blue",
            "actual_values_remain_available": True,
        },
        "rule": {
            "colour_percentile": COLOR_SCALE_PERCENTILE,
            "extreme_colour_percentile": COLOR_SCALE_EXTREME_PERCENTILE,
            "min_active_fraction": COLOR_SCALE_MIN_ACTIVE_FRACTION,
            "fixed_scale_over_all_epochs": True,
            "note": "Scale limits are visualisation limits, not hazard thresholds.",
        },
    }


def read_matrices() -> Dict[str, np.ndarray]:
    require(MATRIX_NPZ, "displacement matrix NPZ")
    data = np.load(MATRIX_NPZ)
    keys = {k.lower(): k for k in data.files}

    def get_component(*names: str) -> np.ndarray: # type: ignore
        for name in names:
            if name.lower() in keys:
                return np.asarray(data[keys[name.lower()]], dtype=np.float32)
        # fallback: substring match
        for key in data.files:
            low = key.lower()
            if any(name.lower() in low for name in names):
                return np.asarray(data[key], dtype=np.float32)
        fail(f"Could not find matrix component in {MATRIX_NPZ}; tried {names}; available keys={data.files}")

    return {
        "reversible": get_component("reversible"),
        "irreversible": get_component("irreversible"),
        "total": get_component("total", "h_spams_final"),
    }


def load_displacement_long_id_stats(id_values: List[Any]) -> pd.DataFrame:
    if DISP_LONG_PARQUET.exists():
        df = pd.read_parquet(DISP_LONG_PARQUET, columns=None)
    elif DISP_LONG_CSV.exists():
        df = pd.read_csv(DISP_LONG_CSV)
    else:
        fail(f"Missing canonical displacement long table: {DISP_LONG_PARQUET} or {DISP_LONG_CSV}")

    needed = {"vI", "std_vI", "var_vI"}
    if not needed.issubset(set(df.columns)):
        fail(f"Displacement long table lacks required columns {needed}; available={list(df.columns)}")

    id_candidates = ["pnt_id", "parcel_id", "source_pnt_id", "int_id"]
    id_col = next((c for c in id_candidates if c in df.columns), None)
    if id_col is None:
        fail(f"Could not find ID column in displacement long table; tried {id_candidates}")

    stats = df.groupby(id_col, sort=False)[["vI", "std_vI", "var_vI"]].first().reset_index()
    stats = stats.rename(columns={id_col: "_join_id"})
    return stats


def load_vi_arrays(moving_index: pd.DataFrame) -> Dict[str, np.ndarray]:
    if "vI" in moving_index.columns and "std_vI" in moving_index.columns:
        vi = moving_index["vI"].to_numpy(dtype=np.float32)
        std = moving_index["std_vI"].to_numpy(dtype=np.float32)
        var = moving_index["var_vI"].to_numpy(dtype=np.float32) if "var_vI" in moving_index.columns else np.square(std).astype(np.float32)
        return {"vI": vi, "std_vI": std, "var_vI": var, "source": np.array(["moving_parcel_index"])}

    id_candidates = ["pnt_id", "parcel_id", "source_pnt_id", "int_id"]
    moving_id_col = next((c for c in id_candidates if c in moving_index.columns), None)
    if moving_id_col is None:
        fail(f"Could not find an ID column in {MOVING_INDEX_PATH}; available={list(moving_index.columns)}")

    stats = load_displacement_long_id_stats(moving_index[moving_id_col].tolist())
    merged = moving_index[[moving_id_col]].rename(columns={moving_id_col: "_join_id"}).merge(stats, on="_join_id", how="left")

    if merged["vI"].isna().any():
        n_missing = int(merged["vI"].isna().sum())
        fail(f"Missing vI for {n_missing} moving parcels after joining on {moving_id_col}")

    vi = merged["vI"].to_numpy(dtype=np.float32)
    std = merged["std_vI"].to_numpy(dtype=np.float32)
    var = merged["var_vI"].to_numpy(dtype=np.float32)
    return {"vI": vi, "std_vI": std, "var_vI": var, "source": np.array([f"displacement_long join on {moving_id_col}"])}


def main() -> None:
    OUTPUT_DATA.mkdir(parents=True, exist_ok=True)
    RUN_RECORDS.mkdir(parents=True, exist_ok=True)

    print("\n=== PROTO2: PARCEL COLOR SCALE SUPPORT ===")
    print(f"Project root: {PROJECT_ROOT}")

    require(MOVING_INDEX_PATH, "moving parcel index")
    moving_index = pd.read_parquet(MOVING_INDEX_PATH)
    n_moving = len(moving_index)
    ok(f"moving parcel index loaded: {n_moving:,} rows")

    matrices = read_matrices()
    rev = matrices["reversible"]
    irr = matrices["irreversible"]
    total = matrices["total"]
    ok(f"animation matrices loaded: reversible={rev.shape}, irreversible={irr.shape}, total={total.shape}")

    if rev.shape != total.shape or irr.shape != total.shape:
        fail(f"Matrix shape mismatch: reversible={rev.shape}, irreversible={irr.shape}, total={total.shape}")
    if rev.shape[0] != n_moving:
        fail(f"Matrix row count {rev.shape[0]:,} != moving parcel count {n_moving:,}")

    vi_data = load_vi_arrays(moving_index)
    vi = vi_data["vI"].astype(np.float32)
    std_vi = vi_data["std_vI"].astype(np.float32)
    var_vi = vi_data["var_vI"].astype(np.float32)
    ok(f"vI loaded from {vi_data['source'][0]}: range {float(np.nanmin(vi)):.6f} to {float(np.nanmax(vi)):.6f} mm/yr")

    if len(vi) != n_moving:
        fail(f"vI count {len(vi):,} != moving parcel count {n_moving:,}")

    vi.astype("<f4").tofile(PARCEL_VI_BIN)
    ok(f"wrote {PARCEL_VI_BIN}")

    # Velocity tau follows Proto1 when velocity uncertainty is available.
    finite_std = std_vi[np.isfinite(std_vi) & (std_vi >= 0.0)]
    tau_raw_velocity = percentile(2.0 * finite_std.astype(np.float64), COLOR_SCALE_UNCERTAINTY_PERCENTILE) if finite_std.size else None
    tau_velocity = round_near_zero_threshold(float(tau_raw_velocity or 0.0))
    if tau_velocity <= 0.0:
        tau_velocity = COLOR_SCALE_NEAR_ZERO_STEP

    # V4.1 carries MC sigma_h(t) for Total as a separate uncertainty product.
    # Keep the colour deadband independent of sigma until the viewer's uncertainty
    # encoding is explicitly selected; colour remains a deformation encoding.
    tau_displacement = 1.0

    scales = {
        "irreversible_velocity": derive_generic_scale(
            vi,
            title="Drowning velocity [mm/yr]",
            unit="mm/yr",
            value_field="vI",
            mode="adaptive_asymmetric_11step_velocity_vI",
            tau=tau_velocity,
            unit_value_key="value_mm_yr",
            negative_label="faster irreversible subsidence / drowning",
            positive_label="uplift / less drowning side",
            tau_source="P75(2*std_vI), rounded upward to nearest 0.5 mm/yr",
        ),
        "reversible_displacement": derive_generic_scale(
            rev.reshape(-1),
            title="Breathing displacement [mm]",
            unit="mm",
            value_field="reversible(t)",
            mode="adaptive_asymmetric_11step_displacement_reversible",
            tau=tau_displacement,
            unit_value_key="value_mm",
            negative_label="negative reversible displacement / exhaling-shrinking",
            positive_label="positive reversible displacement / inhaling-swelling",
            tau_source="fixed ±1.0 mm colour deadband; uncertainty is carried separately when available",
        ),
        "total_displacement": derive_generic_scale(
            total.reshape(-1),
            title="Total displacement [mm]",
            unit="mm",
            value_field="total(t)",
            mode="adaptive_asymmetric_11step_displacement_total",
            tau=tau_displacement,
            unit_value_key="value_mm",
            negative_label="net downward displacement",
            positive_label="net upward displacement",
            tau_source="fixed ±1.0 mm colour deadband; uncertainty is carried separately when available",
        ),
    }

    payload = {
        "schema": "proto2_parcel_color_scales_v1",
        "generated_by": "_color_scale_support.py",
        "project_root": str(PROJECT_ROOT),
        "row_order_contract": "Rows match the internal work-data moving_parcel_index.parquet and displacement_row_index in the GLB TEXCOORD_0.x attribute.",
        "palette_name": VERTICAL_COLOR_PALETTE_NAME,
        "palette_colours": VERTICAL_COLOR_PALETTE_11,
        "blank_parcels": {
            "color_rgb": [0.28, 0.28, 0.28],
            "alpha": 0.55,
            "interpretation": "No SPAMS displacement source; never colored by deformation scale.",
        },
        "scale_selection": {
            "total": "total_displacement",
            "reversible": "reversible_displacement",
            "irreversible": "irreversible_velocity",
            "stacked_top": "total_displacement",
            "stacked_bottom": "muted_irreversible_floor",
        },
        "scales": scales,
        "notes": [
            "Color scales are fixed over the configured viewer epoch range; they do not rescale per epoch.",
            "Irreversible mode colors by vI in mm/yr while height uses irreversible(t) in mm.",
            "Reversible and total modes color by current displacement from datum 0 in mm.",
            "MC per-epoch sigma_h(t) is available for Total only and is intentionally not converted into a colour threshold here.",
        ],
    }

    write_json(PARCEL_COLOR_SCALES_JSON, payload)
    ok(f"wrote {PARCEL_COLOR_SCALES_JSON}")

    summary = {
        "product": "proto2_parcel_color_scales",
        "moving_parcels": int(n_moving),
        "epochs": int(total.shape[1]),
        "parcel_vi_bin": str(PARCEL_VI_BIN),
        "parcel_color_scales_json": str(PARCEL_COLOR_SCALES_JSON),
        "ranges": {
            "vI_mm_yr": [float(np.nanmin(vi)), float(np.nanmax(vi))],
            "std_vI_mm_yr": [float(np.nanmin(std_vi)), float(np.nanmax(std_vi))],
            "reversible_mm": [float(np.nanmin(rev)), float(np.nanmax(rev))],
            "irreversible_mm": [float(np.nanmin(irr)), float(np.nanmax(irr))],
            "total_mm": [float(np.nanmin(total)), float(np.nanmax(total))],
        },
        "scale_limits": {
            key: {
                "recommended_min": val["recommended_min"],
                "recommended_max": val["recommended_max"],
                "tau": val["tau"],
                "unit": val["unit"],
            }
            for key, val in scales.items()
        },
    }
    write_json(PARCEL_COLOR_SUMMARY_JSON, summary)
    ok(f"wrote {PARCEL_COLOR_SUMMARY_JSON}")

    report = {"summary": summary, "color_scales": payload}
    write_json(REPORT_JSON_OUT, report)

    lines = [
        "PROTO2 PARCEL COLOR SCALE SUPPORT REPORT",
        "",
        f"Moving parcels: {n_moving:,}",
        f"Epochs: {total.shape[1]:,}",
        f"vI bin: {PARCEL_VI_BIN}",
        f"color scales: {PARCEL_COLOR_SCALES_JSON}",
        "",
        "Scale limits:",
    ]
    for key, val in summary["scale_limits"].items():
        lines.append(f"- {key}: {val['recommended_min']} to {val['recommended_max']} {val['unit']} (tau={val['tau']})")
    REPORT_TXT_OUT.write_text("\n".join(lines), encoding="utf-8")
    ok(f"wrote {REPORT_JSON_OUT}")
    ok(f"wrote {REPORT_TXT_OUT}")

    print("\n=== SUMMARY ===")
    for key, val in summary["scale_limits"].items():
        print(f"{key:26s}: {val['recommended_min']:>8} to {val['recommended_max']:<8} {val['unit']}  tau={val['tau']}")
    print("\nCOLOUR SCALE SUPPORT RESULT: PASS. Parcel color-scale products written.")


if __name__ == "__main__":
    main()
