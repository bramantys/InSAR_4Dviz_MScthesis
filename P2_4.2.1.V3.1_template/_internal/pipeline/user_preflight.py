#!/usr/bin/env python3
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path


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
    print("\n[CHECK] Python environment and project wiring")
    print(f"[CHECK] Python: {sys.executable}")

    missing_packages = []
    for package in REQUIRED_PACKAGES:
        try:
            importlib.import_module(package)
        except Exception:
            missing_packages.append(package)
    if missing_packages:
        fail(
            "Missing Python packages: "
            + ", ".join(missing_packages)
            + "\nInstall them in the selected Python environment before running the pipeline."
        )
    print("[OK] Required Python packages are available")

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
        "displacement_csv",
        "displacement_crs",
        "parcel_shapefile",
        "parcel_crs",
        "model_parameters_parquet",
        "model_metadata_json",
        "pyspams_directory",
    ]
    missing_keys = [key for key in required_keys if not isinstance(inputs.get(key), str) or not inputs[key].strip()]
    if missing_keys:
        fail("Missing or empty user_inputs fields: " + ", ".join(missing_keys))

    source_mode = config.get("pipeline_source", {}).get("deformation_source")
    if source_mode != "displacement_csv":
        fail(f"This template version requires pipeline_source.deformation_source = displacement_csv, found: {source_mode}")

    paths = {
        "displacement CSV": project_path(inputs["displacement_csv"]),
        "parcel shapefile": project_path(inputs["parcel_shapefile"]),
        "model parameters parquet": project_path(inputs["model_parameters_parquet"]),
        "model metadata JSON": project_path(inputs["model_metadata_json"]),
        "PySPAMS directory": project_path(inputs["pyspams_directory"]),
    }

    for label, path in paths.items():
        exists = path.is_dir() if label == "PySPAMS directory" else path.is_file()
        if not exists:
            fail(f"Configured {label} not found: {path}")
        print(f"[OK] {label}: {path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path}")

    shp = paths["parcel shapefile"]
    sidecars = config.get("input_contract", {}).get("required_shapefile_sidecars", [".shp", ".dbf", ".shx", ".prj"])
    missing_sidecars = [shp.with_suffix(str(ext)) for ext in sidecars if not shp.with_suffix(str(ext)).is_file()]
    if missing_sidecars:
        fail("Incomplete shapefile; missing:\n  - " + "\n  - ".join(str(path) for path in missing_sidecars))
    print("[OK] Shapefile sidecars are complete")

    pyspams_dir = paths["PySPAMS directory"]
    required_pyspams = config.get("input_contract", {}).get("required_pyspams_files", ["utils.py", "spams_main.py"])
    missing_pyspams = [pyspams_dir / name for name in required_pyspams if not (pyspams_dir / name).is_file()]
    if missing_pyspams:
        fail("PySPAMS directory is incomplete; missing:\n  - " + "\n  - ".join(str(path) for path in missing_pyspams))
    print("[OK] PySPAMS files are present")

    print(f"[OK] Declared displacement CRS: {inputs['displacement_crs']}")
    print(f"[OK] Declared parcel CRS: {inputs['parcel_crs']}")
    print("\nPREFLIGHT RESULT: PASS")


if __name__ == "__main__":
    main()
