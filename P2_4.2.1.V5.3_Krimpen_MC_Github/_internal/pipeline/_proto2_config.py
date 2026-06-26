#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_REQUIRED_SOURCE_COLUMNS = [
    "pnt_id",
    "pnt_gid",
    "epoch",
    "reversible",
    "irreversible",
    "h_spams_final",
    "pnt_lat",
    "pnt_lon",
    "vI",
    "std_vI",
    "var_vI",
]


USER_INPUT_DEFAULTS = {
    "monte_carlo_npz": "data/monte_carlo/area_mc_results_full_20150101_20251231_1000_42.npz",
    "displacement_crs": "EPSG:4326",
    "parcel_shapefile": "data/shapefile/krimpenerwaard_attributes_wgs84.shp",
    "parcel_crs": "EPSG:4326",
    "model_parameters_parquet": "data/model_params/nl_krimpenerwaard_spams10.parquet",
    "model_metadata_json": "data/model_params/nl_krimpenerwaard_spams10.json",
    "pyspams_directory": "data/pyspams",
    "knmi_daily_files": {"344": "data/KNMI/etmgeg_344.txt", "348": "data/KNMI/etmgeg_348.txt"},
}


def project_root_from(script_file: str | Path) -> Path:
    return Path(script_file).resolve().parents[2]


def load_project_config(project_root: Path) -> dict[str, Any]:
    path = project_root / "config" / "project_config.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing project config: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def rel_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def get_nested(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def user_input_value(config: dict[str, Any], key: str, default: str | None = None) -> str:
    fallback = USER_INPUT_DEFAULTS.get(key) if default is None else default
    value = get_nested(config, "user_inputs", key, default=fallback)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"project_config.json user_inputs.{key} must be a non-empty string")
    return value.strip()


def user_input_path(project_root: Path, config: dict[str, Any], key: str, default: str | None = None) -> Path:
    return rel_path(project_root, user_input_value(config, key, default))


def declared_crs(config: dict[str, Any], key: str, default: str = "EPSG:4326") -> str:
    value = get_nested(config, "user_inputs", key, default=default)
    return str(value).strip() if value is not None else default


def input_path(
    project_root: Path,
    config: dict[str, Any],
    section: str,
    key: str,
    default: str,
) -> Path:
    """Compatibility resolver for production scripts migrated from config v2."""
    mapping = {
        ("parcel_geometry", "path"): "parcel_shapefile",
        ("monte_carlo", "npz_path"): "monte_carlo_npz",
        ("model_parameters", "parquet_path"): "model_parameters_parquet",
        ("model_parameters", "metadata_path"): "model_metadata_json",
        ("pyspams", "directory"): "pyspams_directory",
    }
    user_key = mapping.get((section, key))
    if user_key is not None:
        return user_input_path(project_root, config, user_key, default)
    value = get_nested(config, "inputs", section, key, default=default)
    return rel_path(project_root, value)


def runtime_root(project_root: Path, config: dict[str, Any]) -> Path:
    value = get_nested(config, "paths", "runtime_root", default="_internal/data_pipeline/runtime")
    return rel_path(project_root, value)


def output_data_dir(project_root: Path, config: dict[str, Any]) -> Path:
    value = get_nested(config, "paths", "work_data", default="_internal/data_pipeline/work/data")
    return rel_path(project_root, value)


def output_cesium_dir(project_root: Path, config: dict[str, Any]) -> Path:
    value = get_nested(config, "paths", "work_cesium", default="_internal/data_pipeline/work/geometry_support")
    return rel_path(project_root, value)


def stage_records_dir(project_root: Path, config: dict[str, Any]) -> Path:
    value = get_nested(config, "paths", "stage_records", default="_internal/data_pipeline/work/reports")
    return rel_path(project_root, value)


def viewer_tuning_path(project_root: Path, config: dict[str, Any]) -> Path:
    value = get_nested(config, "paths", "viewer_tuning", default="_internal/data_pipeline/runtime/style/viewer_tuning.json")
    return rel_path(project_root, value)


def run_records_dir(project_root: Path, config: dict[str, Any]) -> Path:
    value = get_nested(config, "paths", "run_records", default="run_records")
    return rel_path(project_root, value)


def expected_int(config: dict[str, Any], key: str, default: int | None = None) -> int | None:
    value = get_nested(config, "validation", "expected", key, default=default)
    if value is None:
        return None
    return int(value)


def source_columns(config: dict[str, Any]) -> list[str]:
    value = get_nested(config, "input_contract", "required_displacement_columns")
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    return list(DEFAULT_REQUIRED_SOURCE_COLUMNS)


def geometry_id_candidates(config: dict[str, Any]) -> list[str]:
    contract = get_nested(config, "input_contract", default={}) or {}
    candidates: list[str] = []
    for key in ["parcel_id_field", "alternate_parcel_id_field"]:
        value = contract.get(key)
        if isinstance(value, str) and value and value not in candidates:
            candidates.append(value)
    for value in ["parcel_id", "int_id", "pnt_id"]:
        if value not in candidates:
            candidates.append(value)
    return candidates


def displacement_id_field(config: dict[str, Any]) -> str:
    value = get_nested(config, "input_contract", "displacement_parcel_id_field", default="pnt_id")
    return str(value)


def strict_input_counts(config: dict[str, Any]) -> bool:
    return False


def should_require_model_parameters(config: dict[str, Any]) -> bool:
    # All declared user inputs are part of the declared input contract, even
    # though displacement is currently read from the direct SPAMS + KNMI inputs.
    return bool(get_nested(config, "input_contract", "require_all_declared_inputs", default=True))


def active_deformation_source(config: dict[str, Any]) -> str:
    return str(get_nested(config, "pipeline_source", "deformation_source", default="spams_parquet_knmi"))


def pyspams_automated(config: dict[str, Any]) -> bool:
    return bool(get_nested(config, "pipeline_source", "pyspams_automated", default=True))
