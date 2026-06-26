#!/usr/bin/env python3
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

from _mc_total import MonteCarloContractError, monte_carlo_enabled, validate_and_load_monte_carlo_total

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "project_config.json"
REQUIRED_PACKAGES = ["numpy", "pandas", "geopandas", "pyarrow", "shapely", "mapbox_earcut"]


def fail(message: str) -> None:
    print(f"[ERROR] {message}")
    raise SystemExit(1)


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    print("\n[CHECK] Python environment and direct-SPAMS project wiring")
    print(f"[CHECK] Python: {sys.executable}")

    missing_packages = []
    for package in REQUIRED_PACKAGES:
        try:
            importlib.import_module(package)
        except Exception:
            missing_packages.append(package)
    if missing_packages:
        fail(
            "Missing Python packages: " + ", ".join(missing_packages)
            + "\nInstall requirements.txt in the selected Python environment before running the pipeline."
        )
    print("[OK] Required Python packages are available")

    cesium_js = PROJECT_ROOT / "_internal" / "cesium" / "Cesium.js"
    cesium_css = PROJECT_ROOT / "_internal" / "cesium" / "Widgets" / "widgets.css"
    if not cesium_js.is_file() or not cesium_css.is_file():
        fail(
            "Local Cesium runtime is missing. Copy your existing _internal\\cesium folder into this template before running.\n"
            f"Expected: {cesium_js}\n"
            f"Expected: {cesium_css}"
        )
    print("[OK] Local Cesium runtime is present")

    three_js = PROJECT_ROOT / "_internal" / "three" / "three.min.js"
    if not three_js.is_file():
        fail(f"Local Three.js runtime is missing: {three_js}")
    print("[OK] Local Three.js runtime is present")

    if not CONFIG_PATH.is_file():
        fail(f"project_config.json not found: {CONFIG_PATH}")
    try:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        fail(f"project_config.json is not valid JSON: {exc}")

    inputs = config.get("user_inputs")
    if not isinstance(inputs, dict):
        fail("project_config.json must contain a user_inputs object")
    required_keys = [
        "parcel_shapefile",
        "parcel_crs",
        "model_parameters_parquet",
        "model_metadata_json",
        "pyspams_directory",
        "knmi_daily_files",
    ]
    if monte_carlo_enabled(config):
        required_keys.append("monte_carlo_npz")
    missing_keys = [key for key in required_keys if key not in inputs or inputs[key] in (None, "")]
    if missing_keys:
        fail("Missing or empty user_inputs fields: " + ", ".join(missing_keys))

    source_mode = config.get("pipeline_source", {}).get("deformation_source")
    if source_mode != "spams_parquet_knmi":
        fail(f"This package requires pipeline_source.deformation_source = spams_parquet_knmi, found: {source_mode}")

    paths = {
        "parcel shapefile": project_path(inputs["parcel_shapefile"]),
        "model parameters parquet": project_path(inputs["model_parameters_parquet"]),
        "model metadata JSON": project_path(inputs["model_metadata_json"]),
        "PySPAMS directory": project_path(inputs["pyspams_directory"]),
    }
    if monte_carlo_enabled(config):
        paths["supplied Monte Carlo total NPZ"] = project_path(inputs["monte_carlo_npz"])
    for label, path in paths.items():
        exists = path.is_dir() if label == "PySPAMS directory" else path.is_file()
        if not exists:
            fail(f"Configured {label} not found: {path}")
        print(f"[OK] {label}: {path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path}")

    knmi_files = inputs["knmi_daily_files"]
    if not isinstance(knmi_files, dict) or not knmi_files:
        fail("user_inputs.knmi_daily_files must map KNMI station IDs to file paths")
    for station, value in sorted(knmi_files.items(), key=lambda item: int(item[0])):
        if not isinstance(value, str) or not value.strip():
            fail(f"KNMI station {station} path must be a non-empty string")
        path = project_path(value)
        if not path.is_file():
            fail(f"Configured KNMI station {station} file not found: {path}")
        print(f"[OK] KNMI station {station}: {path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path}")

    shp = paths["parcel shapefile"]
    sidecars = config.get("input_contract", {}).get("required_shapefile_sidecars", [".shp", ".dbf", ".shx", ".prj"])
    missing_sidecars = [shp.with_suffix(str(ext)) for ext in sidecars if not shp.with_suffix(str(ext)).is_file()]
    if missing_sidecars:
        fail("Incomplete shapefile; missing:\n  - " + "\n  - ".join(str(path) for path in missing_sidecars))
    print("[OK] Shapefile sidecars are complete")

    utils_path = paths["PySPAMS directory"] / "utils.py"
    if not utils_path.is_file():
        fail(f"PySPAMS utils.py not found: {utils_path}")
    print("[OK] PySPAMS utils.py is present")

    if monte_carlo_enabled(config):
        try:
            mc = validate_and_load_monte_carlo_total(PROJECT_ROOT, config, load_arrays=False)
        except MonteCarloContractError as exc:
            fail(f"supplied Monte Carlo NPZ + Parquet contract failed: {exc}")
        assert mc is not None
        audit = mc.audit
        print("[OK] supplied Monte Carlo total contract: " f"{audit['source_shape'][0]:,} parcels × {audit['source_shape'][1]:,} source epochs")
        print("[OK] Viewer period selected from MC source: " f"{mc.viewer_start_date} to {mc.viewer_end_date} ({audit['viewer_shape'][1]:,} daily epochs)")
        print(f"[OK] Default displayed reference date: {mc.default_reference_date}")
        print("[OK] sigma_t policy: Total-only; trim unchanged; never apply to components")

    print("\nPREFLIGHT RESULT: PASS")


if __name__ == "__main__":
    main()
