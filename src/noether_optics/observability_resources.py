"""Static storage admission for the frozen P2 observability protocol.

The estimator imports no JAX module, constructs no optical plant, opens no cohort and
records no elapsed time. It translates the frozen retention policy into explicit array
schemas and sizes the maximum released run without assuming compression.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import shutil
from typing import Any

import yaml

from .observability_protocol import load_protocol, protocol_plan, validate_protocol
from .resource_envelope import (
    ARCHIVE_MEMBER_ALLOWANCE_BYTES,
    ARCHIVE_PAYLOAD_FACTOR,
    BOOL_BYTES,
    FLOAT64_BYTES,
    INT64_BYTES,
    archive_category,
    finalize_archive_budget,
    positive_integer,
)

ROOT = Path(__file__).resolve().parents[2]
DTYPE_BYTES = {"float64": FLOAT64_BYTES, "int64": INT64_BYTES, "bool": BOOL_BYTES}


def load_resource_inputs(
    path: Path, repository: Path | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the protocol and exact pinned base bytes without importing JAX."""
    root = Path(repository or ROOT).resolve()
    protocol_path = Path(path)
    protocol_path = protocol_path if protocol_path.is_absolute() else root / protocol_path
    if not protocol_path.is_file() or not protocol_path.resolve().is_relative_to(root):
        raise ValueError("Protocol must be a regular file inside the source repository")
    for current in (protocol_path, *protocol_path.parents):
        if current == root:
            break
        if current.is_symlink():
            raise ValueError("Protocol path cannot contain a symlink")
    protocol = load_protocol(protocol_path, repository=root)
    base_path = root / protocol["base_config"]
    for current in (base_path, *base_path.parents):
        if current == root:
            break
        if current.is_symlink():
            raise ValueError("Base configuration path cannot contain a symlink")
    if not base_path.is_file() or not base_path.resolve().is_relative_to(root):
        raise ValueError("Base simulation configuration must stay inside the repository")
    base_bytes = base_path.read_bytes()
    if hashlib.sha256(base_bytes).hexdigest() != protocol["base_config_sha256"]:
        raise ValueError("Base simulation configuration hash changed during resource loading")
    base = yaml.safe_load(base_bytes)
    if not isinstance(base, dict):
        raise ValueError("Base simulation configuration must be a mapping")
    return protocol, base


def _array_spec(name: str, dtype: str, shape: tuple[int, ...]) -> dict[str, Any]:
    if dtype not in DTYPE_BYTES or not shape:
        raise ValueError("Unsupported retained-array schema")
    dimensions = [positive_integer(value, f"{name} dimension") for value in shape]
    elements = math.prod(dimensions)
    return {
        "name": name,
        "dtype": dtype,
        "shape": dimensions,
        "elements": elements,
        "bytes_per_trial": elements * DTYPE_BYTES[dtype],
    }


def _retained_schemas(
    protocol: dict[str, Any], base: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    frames = positive_integer(protocol["loop"]["frames"], "trial frames")
    burn_in = protocol["loop"]["burn_in"]
    window = positive_integer(protocol["detector"]["window_frames"], "detector window")
    persistence = positive_integer(
        protocol["detector"]["persistence_windows"], "detector persistence"
    )
    statistics = frames - burn_in - window + 1
    alarms = statistics - persistence + 1
    modes = positive_integer(base.get("modes"), "mode count")
    detector_n = positive_integer(base.get("detector_n"), "detector width")
    if base.get("sensor") != "pyramid":
        raise ValueError("Observability sizing requires the pinned pyramid sensor")

    frame_arrays = [
        _array_spec("score", "float64", (frames,)),
        _array_spec("statistic_frame_id", "int64", (statistics,)),
        _array_spec("lower_statistic", "float64", (statistics,)),
        _array_spec("raw_crossing", "bool", (statistics,)),
        _array_spec("alarm_frame_id", "int64", (alarms,)),
        _array_spec("p2_alarm", "bool", (alarms,)),
        _array_spec("variance", "float64", (frames,)),
        _array_spec("strehl", "float64", (frames,)),
        _array_spec("saturated", "bool", (frames,)),
        _array_spec("fallback", "bool", (frames,)),
        _array_spec("unavailable", "bool", (frames,)),
        _array_spec("command", "float64", (frames, modes)),
        _array_spec("applied", "float64", (frames, modes)),
    ]
    full_wfs = [
        _array_spec("measurement", "float64", (frames, detector_n, detector_n)),
        _array_spec("prediction", "float64", (frames, detector_n, detector_n)),
    ]
    retention = protocol["retention"]
    if [row["name"] for row in frame_arrays] != retention["all_trial_frame_arrays"]:
        raise ValueError("Retained frame-array schema differs from the frozen protocol")
    if [row["name"] for row in full_wfs] != retention["full_wfs_arrays"]:
        raise ValueError("Full-WFS schema differs from the frozen protocol")
    return frame_arrays, full_wfs


def estimate_observability_resources(
    protocol: dict[str, Any], base: dict[str, Any]
) -> dict[str, Any]:
    """Return the worst-case evidence envelope for a released complete P2 run."""
    validate_protocol(protocol)
    plan = protocol_plan(protocol)
    frame_arrays, full_wfs_arrays = _retained_schemas(protocol, base)
    trace_names = {
        "statistic_frame_id",
        "lower_statistic",
        "raw_crossing",
        "alarm_frame_id",
        "p2_alarm",
    }
    training_raw = [row for row in frame_arrays if row["name"] not in trace_names]
    training_trace = [row for row in frame_arrays if row["name"] in trace_names]
    core_payload = sum(row["bytes_per_trial"] for row in frame_arrays)
    training_raw_payload = sum(row["bytes_per_trial"] for row in training_raw)
    training_trace_payload = sum(row["bytes_per_trial"] for row in training_trace)
    full_wfs_payload = sum(row["bytes_per_trial"] for row in full_wfs_arrays)
    regimes = len(plan["regimes"])
    role_trials = {
        "training": plan["planned_training_trials"],
        "validation": plan["planned_validation_trials"],
        "evaluation": plan["maximum_planned_evaluation_trials"],
    }
    full_wfs_counts = {
        role: regimes * len(protocol["conditions"][role]) for role in role_trials
    }
    categories = []
    for role, trials in role_trials.items():
        full = full_wfs_counts[role]
        standard = trials - full
        payload = training_raw_payload if role == "training" else core_payload
        members = len(training_raw) if role == "training" else len(frame_arrays)
        categories.extend(
            [
                archive_category(
                    f"{role}_standard_trial_outputs",
                    standard,
                    payload,
                    members,
                ),
                archive_category(
                    f"{role}_full_wfs_trial_outputs",
                    full,
                    payload + full_wfs_payload,
                    members + len(full_wfs_arrays),
                ),
            ]
        )
        if role == "training":
            categories.append(
                archive_category(
                    "training_threshold_detector_outputs",
                    trials,
                    training_trace_payload,
                    len(training_trace),
                )
            )
    budget = finalize_archive_budget(categories)
    return {
        "schema_version": 1,
        "kind": "optical_gain_observability_resource_budget",
        "study_id": protocol["study_id"],
        "status": "derived_not_measured",
        "counts": {
            "training_trials": role_trials["training"],
            "validation_trials": role_trials["validation"],
            "maximum_evaluation_trials": role_trials["evaluation"],
            "full_wfs_trial_archives": sum(full_wfs_counts.values()),
            "numeric_archives": sum(row["archives"] for row in categories),
        },
        "budget": budget,
        "categories": categories,
        "schemas": {
            "all_trial_frame_arrays": frame_arrays,
            "full_wfs_arrays": full_wfs_arrays,
            "all_trial_metrics": protocol["retention"]["all_trial_metrics"],
        },
        "assumptions": {
            "numeric_float_dtype": "float64",
            "frame_identity_dtype": "int64",
            "numeric_boolean_bytes": BOOL_BYTES,
            "archive_payload_factor": ARCHIVE_PAYLOAD_FACTOR,
            "archive_member_allowance_bytes": ARCHIVE_MEMBER_ALLOWANCE_BYTES,
            "metadata_filesystem_fraction": "1/8",
            "full_wfs_policy": protocol["retention"]["full_wfs_policy"],
            "per_run_metrics_in_metadata_envelope": True,
            "maximum_evaluation_release_assumed": True,
            "failed_or_withheld_trials_reduce_storage": True,
            "elapsed_costs_included": False,
            "training_threshold_dependent_arrays_are_separate": True,
        },
        "outcomes_generated": False,
    }


def resource_preflight(
    protocol: dict[str, Any], base: dict[str, Any], target: Path
) -> dict[str, Any]:
    """Compare the static budget with a filesystem without creating the target."""
    target = Path(target).expanduser().resolve()
    if target.exists() and not target.is_dir():
        raise ValueError("Observability target must be a directory or an uncreated path")
    ancestor = target
    while not ancestor.exists():
        ancestor = ancestor.parent
    budget = estimate_observability_resources(protocol, base)
    free = shutil.disk_usage(ancestor).free
    required = budget["budget"]["required_free_bytes"]
    return {
        **budget,
        "kind": "optical_gain_observability_resource_preflight",
        "status": "sufficient" if free >= required else "insufficient",
        "target": str(target),
        "filesystem": {"checked_path": str(ancestor), "free_bytes": free},
        "admitted": free >= required,
    }
