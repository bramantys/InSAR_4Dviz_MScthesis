"""
CHECK — Horizontal ENU Source Discovery v2

This version explicitly checks the OG source folder:

  Data/1_OG/jakarta_enu_estimates.csv
  Data/1_OG/jakarta_enu_estimates.json
  Data/1_OG/jakarta_enu_estimates.pkl

It also checks the current processed files used by the vertical pipeline:

  Data/jakarta_rum_footprints.json
  Data/jakarta_points_wgs84_with_rumid.geojson

Goal:
  Identify the correct horizontal velocity columns and how to join them
  to the current RUM grid/footprints.

Run:
  python pipeline/check_horizontal_inputs_v2.py

This script only reads files. It does not modify anything.
"""

import csv
import json
import math
import os
import pickle
from collections import Counter, defaultdict

import numpy as np

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_BASE, "Data")
OG_DIR = os.path.join(DATA_DIR, "1_OG")

OG_CANDIDATES = [
    os.path.join(OG_DIR, "jakarta_enu_estimates.csv"),
    os.path.join(OG_DIR, "jakarta_enu_estimates.json"),
    os.path.join(OG_DIR, "jakarta_enu_estimates.pkl"),
]

PROCESSED_CANDIDATES = [
    os.path.join(DATA_DIR, "jakarta_rum_footprints.json"),
    os.path.join(DATA_DIR, "jakarta_points_wgs84_with_rumid.geojson"),
    os.path.join(DATA_DIR, "jakarta_points_wgs84.geojson"),
]

HORIZONTAL_KEYWORDS = [
    "east", "north", "e_", "n_", "enu",
    "ve", "vn", "vx", "vy", "v_e", "v_n",
    "vel", "velocity", "rate", "mm_yr", "mmyr", "yr",
    "horizontal", "horiz"
]

COORD_KEYWORDS = [
    "lon", "longitude", "lat", "latitude", "x", "y", "rum", "id",
    "x_rum", "y_rum", "grid_i", "grid_j"
]

UNCERTAINTY_KEYWORDS = [
    "sigma", "std", "stdev", "unc", "uncert", "cov",
    "ellipse", "semi", "major", "minor", "azimuth", "angle",
    "confidence", "conf", "quality"
]


# ---------------------------------------------------------------------
# print helpers
# ---------------------------------------------------------------------

def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print("="*70)


def ok(msg):
    print(f"  [OK]   {msg}")


def warn(msg):
    print(f"  [WARN] {msg}")


# ---------------------------------------------------------------------
# generic helpers
# ---------------------------------------------------------------------

def is_number(x):
    if isinstance(x, bool) or x is None:
        return False
    if isinstance(x, (int, float)):
        return math.isfinite(float(x))
    if isinstance(x, str):
        try:
            return math.isfinite(float(x))
        except Exception:
            return False
    return False


def to_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def key_score(key, keywords):
    lk = str(key).lower()
    return sum(1 for kw in keywords if kw in lk)


def summarize_numeric(values):
    arr = np.array([to_float(v) for v in values], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return None
    return {
        "count": int(len(arr)),
        "min": float(np.min(arr)),
        "p05": float(np.percentile(arr, 5)),
        "median": float(np.percentile(arr, 50)),
        "mean": float(np.mean(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def print_stat(key, stat, indent="    "):
    print(
        f"{indent}{key:<36s} n={stat['count']:6d} "
        f"min={stat['min']:11.4f} p05={stat['p05']:11.4f} "
        f"med={stat['median']:11.4f} mean={stat['mean']:11.4f} "
        f"p95={stat['p95']:11.4f} max={stat['max']:11.4f}"
    )


def normalize_record_list(obj):
    """
    Convert common JSON/PKL structures into list[dict].
    """
    # pandas DataFrame support without requiring pandas import here
    if hasattr(obj, "to_dict") and hasattr(obj, "columns"):
        return obj.to_dict(orient="records")

    if isinstance(obj, list):
        if len(obj) == 0:
            return []
        if isinstance(obj[0], dict):
            return obj
        return []

    if isinstance(obj, dict):
        # GeoJSON
        if obj.get("type") == "FeatureCollection":
            records = []
            for ft in obj.get("features", []):
                props = dict(ft.get("properties") or {})
                geom = ft.get("geometry") or {}
                if geom.get("type") == "Point":
                    coords = geom.get("coordinates") or []
                    if len(coords) >= 2:
                        props["_geom_lon"] = coords[0]
                        props["_geom_lat"] = coords[1]
                records.append(props)
            return records

        # Dict of records keyed by ID
        if all(isinstance(v, dict) for v in obj.values()):
            records = []
            for k, v in obj.items():
                r = dict(v)
                r.setdefault("_dict_key", k)
                records.append(r)
            return records

        # Column-oriented dict: each key has a list of same length
        list_keys = [k for k, v in obj.items() if isinstance(v, list)]
        if list_keys:
            n = len(obj[list_keys[0]])
            if all(len(obj[k]) == n for k in list_keys):
                records = []
                for i in range(n):
                    r = {}
                    for k in list_keys:
                        r[k] = obj[k][i]
                    records.append(r)
                return records

    return []


def load_records_from_file(path):
    ext = os.path.splitext(path)[1].lower()

    if ext == ".csv":
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            return list(reader)

    if ext in [".json", ".geojson"]:
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
        return normalize_record_list(obj)

    if ext == ".pkl":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return normalize_record_list(obj)

    return []


def inspect_records(label, path, records):
    section(f"Inspecting {label}")

    print(f"  File: {path}")
    ok(f"Records: {len(records)}")

    if not records:
        warn("No records found / unsupported structure")
        return {
            "records": records,
            "keys": [],
            "numeric_stats": {},
            "h_candidates": [],
            "coord_candidates": [],
            "u_candidates": [],
            "rum_keys": [],
        }

    key_counter = Counter()
    samples = defaultdict(list)

    for r in records:
        for k, v in r.items():
            key_counter[k] += 1
            if len(samples[k]) < 5:
                samples[k].append(v)

    print("\n  Keys:")
    for k, c in key_counter.most_common():
        print(f"    {str(k):<40s} present in {c}/{len(records)}  sample={samples[k]}")

    numeric_stats = {}
    for k in key_counter.keys():
        vals = [r.get(k) for r in records if k in r and is_number(r.get(k))]
        if vals:
            stat = summarize_numeric(vals)
            if stat:
                numeric_stats[k] = stat

    section(f"Numeric summaries: {label}")
    if numeric_stats:
        for k in sorted(numeric_stats.keys(), key=str):
            print_stat(str(k), numeric_stats[k])
    else:
        warn("No numeric fields found")

    def candidate_list(keywords):
        rows = []
        for k, stat in numeric_stats.items():
            score = key_score(k, keywords)
            if score > 0:
                rows.append((score, str(k), stat))
        return sorted(rows, key=lambda x: (-x[0], x[1]))

    h_candidates = candidate_list(HORIZONTAL_KEYWORDS)
    coord_candidates = candidate_list(COORD_KEYWORDS)
    u_candidates = candidate_list(UNCERTAINTY_KEYWORDS)

    section(f"Likely horizontal velocity columns: {label}")
    if h_candidates:
        for score, k, stat in h_candidates:
            print(f"  score={score}  ", end="")
            print_stat(k, stat, indent="")
    else:
        warn("No obvious horizontal velocity columns detected by name")

    section(f"Likely coordinate / RUM ID columns: {label}")
    if coord_candidates:
        for score, k, stat in coord_candidates:
            print(f"  score={score}  ", end="")
            print_stat(k, stat, indent="")
    else:
        warn("No obvious coordinate/RUM numeric columns detected by name")

    section(f"Likely uncertainty / ellipse columns: {label}")
    if u_candidates:
        for score, k, stat in u_candidates:
            print(f"  score={score}  ", end="")
            print_stat(k, stat, indent="")
    else:
        warn("No obvious uncertainty / ellipse numeric columns detected by name")

    rum_keys = []
    for k in key_counter.keys():
        lk = str(k).lower()
        if "rum" in lk or lk == "id" or lk.endswith("_id") or "rumid" in lk:
            rum_keys.append(k)

    section(f"RUM-like keys: {label}")
    if rum_keys:
        for k in rum_keys:
            print(f"  {k}: sample={samples[k]}")
    else:
        warn("No obvious RUM-like key")

    return {
        "records": records,
        "keys": list(key_counter.keys()),
        "numeric_stats": numeric_stats,
        "h_candidates": h_candidates,
        "coord_candidates": coord_candidates,
        "u_candidates": u_candidates,
        "rum_keys": rum_keys,
    }


def load_footprint_ids_and_grids():
    fp_path = os.path.join(DATA_DIR, "jakarta_rum_footprints.json")
    if not os.path.isfile(fp_path):
        return {}, {}

    with open(fp_path, encoding="utf-8") as f:
        data = json.load(f)

    footprints = data.get("footprints", {})
    id_set = set(footprints.keys())

    grid_by_id = {}
    for rid, fp in footprints.items():
        if "grid_i" in fp and "grid_j" in fp:
            grid_by_id[rid] = (fp["grid_i"], fp["grid_j"])
        elif "grid" in fp and isinstance(fp["grid"], list) and len(fp["grid"]) >= 2:
            grid_by_id[rid] = (fp["grid"][0], fp["grid"][1])

    return id_set, grid_by_id


def try_match_rum_ids(label, result, footprint_ids):
    records = result["records"]

    if not records or not footprint_ids:
        return

    section(f"RUM ID matching to footprints: {label}")

    # Direct key matching
    for key in result["rum_keys"]:
        vals = [str(r.get(key)) for r in records if r.get(key) is not None]
        unique = set(vals)
        matched = len(unique & footprint_ids)
        print(f"  Direct key {str(key):<30s}: unique={len(unique):6d}, matched={matched:6d}")

    # Derive RUM_x_y if x_rum/y_rum-like fields exist
    keys_lower = {str(k).lower(): k for k in result["keys"]}

    possible_x = []
    possible_y = []
    for lk, k in keys_lower.items():
        if lk in ["x_rum", "xrum", "rum_x", "x"]:
            possible_x.append(k)
        if lk in ["y_rum", "yrum", "rum_y", "y"]:
            possible_y.append(k)

    for xk in possible_x:
        for yk in possible_y:
            vals = []
            for r in records:
                if r.get(xk) is None or r.get(yk) is None:
                    continue
                try:
                    x = int(round(float(r.get(xk))))
                    y = int(round(float(r.get(yk))))
                    vals.append(f"RUM_{x}_{y}")
                except Exception:
                    pass
            unique = set(vals)
            matched = len(unique & footprint_ids)
            print(f"  Derived RUM_{{{xk}}}_{{{yk}}}: unique={len(unique):6d}, matched={matched:6d}")


def main():
    section("File existence")

    all_files = OG_CANDIDATES + PROCESSED_CANDIDATES
    existing = []
    for path in all_files:
        if os.path.isfile(path):
            existing.append(path)
            ok(f"{path} ({os.path.getsize(path)/1024/1024:.2f} MB)")
        else:
            warn(f"Missing: {path}")

    if not existing:
        warn("No candidate files found")
        return

    footprint_ids, grid_by_id = load_footprint_ids_and_grids()

    section("Footprint reference")
    if footprint_ids:
        ok(f"Footprint RUM IDs: {len(footprint_ids)}")
        ok(f"Footprints with grid coordinates: {len(grid_by_id)}")
        sample = list(sorted(footprint_ids))[:3]
        print(f"  Sample footprint IDs: {sample}")
    else:
        warn("No footprint reference loaded")

    results = {}

    for path in existing:
        label = os.path.relpath(path, DATA_DIR)
        try:
            records = load_records_from_file(path)
        except Exception as e:
            section(f"Inspecting {label}")
            warn(f"Could not read: {e}")
            continue

        result = inspect_records(label, path, records)
        results[label] = result

        try_match_rum_ids(label, result, footprint_ids)

    section("H0 decision checklist")
    print("  Send this terminal output.")
    print()
    print("  We need to identify:")
    print("    1. Which OG file has the horizontal velocity components.")
    print("    2. Exact east/north column names and units.")
    print("    3. How to join OG rows to current RUM IDs:")
    print("       - direct rum_id key, or")
    print("       - derived RUM_{x_rum}_{y_rum}, or")
    print("       - join via processed GeoJSON.")
    print("    4. Whether the confidence ellipse columns are available for later.")
    print()
    print("  Important note:")
    print("    The coordinate skew fix should come from jakarta_rum_footprints.json,")
    print("    not from raw OG point coordinates. The horizontal field should inherit")
    print("    the Phase 1 grid model/footprints that already fixed overlap/skew.")
    print()


if __name__ == "__main__":
    main()
