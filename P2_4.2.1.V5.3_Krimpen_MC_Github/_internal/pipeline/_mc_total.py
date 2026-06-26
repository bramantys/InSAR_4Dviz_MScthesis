#!/usr/bin/env python3
"""Monte Carlo Total-product input contract for Prototype 2 V4.1.

This module is deliberately small and source-neutral:
- Parquet defines the authoritative moving-parcel row order and IDs.
- NPZ carries dense ``mean_t`` / ``sigma_t`` arrays using that row order.
- The project config selects a viewer period from the source reconstruction.

It never recalculates a Monte Carlo solution.  It validates and slices the
already-computed product before downstream pipeline stages package it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from _proto2_config import get_nested, rel_path


class MonteCarloContractError(ValueError):
    """Raised when the configured NPZ + Parquet Monte Carlo contract is invalid."""


@dataclass(frozen=True)
class MonteCarloTotalSlice:
    mean_total: np.ndarray
    sigma_total: np.ndarray
    parcel_ids: np.ndarray
    epoch_labels: np.ndarray
    source_epoch_labels: np.ndarray
    source_start_date: str
    source_end_date: str
    source_reference_date: str
    viewer_start_date: str
    viewer_end_date: str
    default_reference_date: str
    source_path: Path
    parcel_order_path: Path
    mean_key: str
    sigma_key: str
    audit: dict[str, Any]


def _require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MonteCarloContractError(f"project_config.json {label} must be an object")
    return value


def _require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MonteCarloContractError(f"project_config.json {label} must be a non-empty string")
    return value.strip()


def _as_date(value: Any, label: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(str(value)).normalize()
    except Exception as exc:  # pragma: no cover - defensive error path
        raise MonteCarloContractError(f"Invalid date for {label}: {value!r} ({exc})") from exc
    if pd.isna(timestamp):
        raise MonteCarloContractError(f"Invalid date for {label}: {value!r}")
    return timestamp


def _date_text(value: pd.Timestamp) -> str:
    return value.strftime("%Y-%m-%d")


def monte_carlo_enabled(config: dict[str, Any]) -> bool:
    block = get_nested(config, "monte_carlo_total", default={})
    return bool(block.get("enabled", False)) if isinstance(block, dict) else False


def _resolve_user_input_path(project_root: Path, config: dict[str, Any], reference: str, label: str) -> Path:
    user_inputs = _require_object(config.get("user_inputs"), "user_inputs")
    if reference in user_inputs:
        raw = _require_nonempty_string(user_inputs.get(reference), f"user_inputs.{reference}")
    else:
        # Permit a direct project-relative path for future packages.
        raw = reference
    return rel_path(project_root, raw)


def validate_and_load_monte_carlo_total(
    project_root: Path,
    config: dict[str, Any],
    *,
    load_arrays: bool = True,
) -> MonteCarloTotalSlice | None:
    """Validate the configured Monte Carlo product and return the selected viewer slice.

    ``mean_t`` and ``sigma_t`` are loaded as float32 arrays.  The requested
    viewer period is selected by *trimming columns only*.  No values are
    rebased here: the viewer owns visual reference-date handling for mean_t.
    """
    if not monte_carlo_enabled(config):
        return None

    mc = _require_object(config.get("monte_carlo_total"), "monte_carlo_total")
    npz_ref = _require_nonempty_string(mc.get("npz_input"), "monte_carlo_total.npz_input")
    parcel_ref = _require_nonempty_string(mc.get("parcel_order_source"), "monte_carlo_total.parcel_order_source")
    npz_path = _resolve_user_input_path(project_root, config, npz_ref, "Monte Carlo NPZ")
    parquet_path = _resolve_user_input_path(project_root, config, parcel_ref, "Monte Carlo parcel-order Parquet")

    if not npz_path.is_file():
        raise MonteCarloContractError(f"Monte Carlo NPZ not found: {npz_path}")
    if not parquet_path.is_file():
        raise MonteCarloContractError(f"Monte Carlo parcel-order Parquet not found: {parquet_path}")

    mean_key = _require_nonempty_string(mc.get("mean_key", "mean_t"), "monte_carlo_total.mean_key")
    sigma_key = _require_nonempty_string(mc.get("sigma_key", "sigma_t"), "monte_carlo_total.sigma_key")

    required_keys = get_nested(config, "input_contract", "required_monte_carlo_keys", default=[mean_key, sigma_key])
    if not isinstance(required_keys, list) or not all(isinstance(key, str) and key for key in required_keys):
        raise MonteCarloContractError("input_contract.required_monte_carlo_keys must be a list of non-empty strings")
    required_keys = list(dict.fromkeys([*required_keys, mean_key, sigma_key]))

    source = _require_object(mc.get("source_reconstruction_period"), "monte_carlo_total.source_reconstruction_period")
    source_start = _as_date(source.get("start_date"), "monte_carlo_total.source_reconstruction_period.start_date")
    source_end = _as_date(source.get("end_date"), "monte_carlo_total.source_reconstruction_period.end_date")
    source_t0 = _as_date(source.get("reference_date"), "monte_carlo_total.source_reconstruction_period.reference_date")
    if source_end < source_start:
        raise MonteCarloContractError("Monte Carlo source reconstruction end_date is before start_date")
    if source_t0 < source_start or source_t0 > source_end:
        raise MonteCarloContractError("Monte Carlo source reconstruction reference_date is outside its source period")

    time_settings = _require_object(config.get("time_settings"), "time_settings")
    viewer_period = _require_object(time_settings.get("viewer_period"), "time_settings.viewer_period")
    viewer_start = _as_date(viewer_period.get("start_date"), "time_settings.viewer_period.start_date")
    viewer_end = _as_date(viewer_period.get("end_date"), "time_settings.viewer_period.end_date")
    viewer_t0 = _as_date(viewer_period.get("default_reference_date"), "time_settings.viewer_period.default_reference_date")
    if viewer_end < viewer_start:
        raise MonteCarloContractError("Viewer period end_date is before start_date")
    if viewer_start < source_start or viewer_end > source_end:
        raise MonteCarloContractError(
            "Viewer period must fall inside the supplied Monte Carlo source reconstruction period"
        )
    if viewer_t0 < viewer_start or viewer_t0 > viewer_end:
        raise MonteCarloContractError("Viewer default_reference_date must fall inside the selected viewer period")

    source_epoch_labels = pd.date_range(source_start, source_end, freq="D").strftime("%Y-%m-%d").to_numpy(dtype="U10")
    viewer_epoch_labels = pd.date_range(viewer_start, viewer_end, freq="D").strftime("%Y-%m-%d").to_numpy(dtype="U10")
    start_idx = int((viewer_start - source_start).days)
    end_idx = int((viewer_end - source_start).days) + 1

    try:
        parcel_frame = pd.read_parquet(parquet_path, columns=["pnt_id"])
    except Exception as exc:
        raise MonteCarloContractError(f"Could not read Monte Carlo parcel-order Parquet: {parquet_path}: {exc}") from exc
    if "pnt_id" not in parcel_frame.columns:
        raise MonteCarloContractError("Monte Carlo parcel-order Parquet must contain pnt_id")
    parcel_ids = pd.to_numeric(parcel_frame["pnt_id"], errors="coerce")
    if parcel_ids.isna().any():
        raise MonteCarloContractError("Monte Carlo parcel-order Parquet contains non-numeric or null pnt_id values")
    parcel_ids_np = parcel_ids.astype("int64").to_numpy()
    if len(np.unique(parcel_ids_np)) != len(parcel_ids_np):
        raise MonteCarloContractError("Monte Carlo parcel-order Parquet contains duplicate pnt_id values")

    try:
        with np.load(npz_path, allow_pickle=False) as bundle:
            missing_keys = [key for key in required_keys if key not in bundle.files]
            if missing_keys:
                raise MonteCarloContractError(
                    f"Monte Carlo NPZ missing required arrays: {missing_keys}; available={bundle.files}"
                )
            mean = np.asarray(bundle[mean_key])
            sigma = np.asarray(bundle[sigma_key])
    except MonteCarloContractError:
        raise
    except Exception as exc:
        raise MonteCarloContractError(f"Could not read Monte Carlo NPZ: {npz_path}: {exc}") from exc

    if mean.ndim != 2 or sigma.ndim != 2:
        raise MonteCarloContractError(
            f"Monte Carlo arrays must both be 2D [parcel, epoch]; mean={mean.shape}, sigma={sigma.shape}"
        )
    if mean.shape != sigma.shape:
        raise MonteCarloContractError(f"Monte Carlo mean/sigma shape mismatch: mean={mean.shape}, sigma={sigma.shape}")
    if mean.shape[0] != len(parcel_ids_np):
        raise MonteCarloContractError(
            f"Monte Carlo NPZ parcel rows ({mean.shape[0]:,}) do not match Parquet row count ({len(parcel_ids_np):,})"
        )
    if mean.shape[1] != len(source_epoch_labels):
        raise MonteCarloContractError(
            "Monte Carlo NPZ epoch columns "
            f"({mean.shape[1]:,}) do not match configured inclusive source period "
            f"({_date_text(source_start)} to {_date_text(source_end)} = {len(source_epoch_labels):,} daily epochs)"
        )
    if not np.issubdtype(mean.dtype, np.number) or not np.issubdtype(sigma.dtype, np.number):
        raise MonteCarloContractError(f"Monte Carlo arrays must be numeric: mean={mean.dtype}, sigma={sigma.dtype}")

    mean32 = np.ascontiguousarray(mean.astype(np.float32, copy=False))
    sigma32 = np.ascontiguousarray(sigma.astype(np.float32, copy=False))
    if not np.isfinite(mean32).all():
        raise MonteCarloContractError("Monte Carlo mean array contains NaN or infinite values")
    if not np.isfinite(sigma32).all():
        raise MonteCarloContractError("Monte Carlo sigma array contains NaN or infinite values")
    if np.any(sigma32 < 0.0):
        raise MonteCarloContractError("Monte Carlo sigma array contains negative values")

    mean_view = np.ascontiguousarray(mean32[:, start_idx:end_idx])
    sigma_view = np.ascontiguousarray(sigma32[:, start_idx:end_idx])
    audit = {
        "enabled": True,
        "npz_path": str(npz_path),
        "parcel_order_path": str(parquet_path),
        "mean_key": mean_key,
        "sigma_key": sigma_key,
        "source_shape": [int(mean.shape[0]), int(mean.shape[1])],
        "viewer_shape": [int(mean_view.shape[0]), int(mean_view.shape[1])],
        "source_reconstruction_period": {
            "start_date": _date_text(source_start),
            "end_date": _date_text(source_end),
            "reference_date": _date_text(source_t0),
        },
        "viewer_period": {
            "start_date": _date_text(viewer_start),
            "end_date": _date_text(viewer_end),
            "default_reference_date": _date_text(viewer_t0),
            "source_start_index": int(start_idx),
            "source_end_index_inclusive": int(end_idx - 1),
        },
        "row_order": "NPZ row i maps to Parquet row i; pnt_id is retained in Parquet order",
        "mean": {
            "dtype": str(mean32.dtype),
            "min": float(mean32.min()),
            "max": float(mean32.max()),
        },
        "sigma": {
            "dtype": str(sigma32.dtype),
            "min": float(sigma32.min()),
            "max": float(sigma32.max()),
            "negative_count": int((sigma32 < 0.0).sum()),
        },
        "uncertainty_rule": "sigma_t is trimmed to the viewer period unchanged; it is not rebased with the displayed mean reference date.",
    }
    return MonteCarloTotalSlice(
        mean_total=mean_view if load_arrays else np.empty((0, 0), dtype=np.float32),
        sigma_total=sigma_view if load_arrays else np.empty((0, 0), dtype=np.float32),
        parcel_ids=parcel_ids_np,
        epoch_labels=viewer_epoch_labels,
        source_epoch_labels=source_epoch_labels,
        source_start_date=_date_text(source_start),
        source_end_date=_date_text(source_end),
        source_reference_date=_date_text(source_t0),
        viewer_start_date=_date_text(viewer_start),
        viewer_end_date=_date_text(viewer_end),
        default_reference_date=_date_text(viewer_t0),
        source_path=npz_path,
        parcel_order_path=parquet_path,
        mean_key=mean_key,
        sigma_key=sigma_key,
        audit=audit,
    )
