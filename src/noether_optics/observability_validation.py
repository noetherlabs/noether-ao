"""Immutable validation decision and evaluation-release barrier for P2 v1.

This module opens no cohort and runs no optical model. It audits already-published
training and validation artifacts, reconstructs detector outputs, applies the frozen
per-regime gates and publishes one non-replaceable release decision.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .controller_selection import _bytes, _digest, _provenance, _publish
from .observability_detector import (
    _array_digest,
    _artifact_arrays,
    _build_threshold_bundle,
    _case_id,
    _file_digest,
    _load_protocol_binding,
    _score,
    detector_trace,
    training_cases,
)
from .observability_protocol import protocol_plan, validate_protocol
from .observability_resources import load_resource_inputs
from .observability_trials import (
    training_artifact_layout,
    verify_failed_evidence,
    verify_training_evidence,
    verify_training_trace,
    verify_validation_evidence,
)


def validation_cases(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Enumerate all 240 validation identities without opening any evidence."""
    validate_protocol(config)
    plan = protocol_plan(config)
    return [
        {
            "role": "validation",
            "condition": condition,
            "regime_id": regime["id"],
            "photons": regime["photons"],
            "wind_scale": regime["wind_scale"],
            "atmosphere_seed": stream["atmosphere"],
            "detector_seed": stream["detector"],
        }
        for regime in plan["regimes"]
        for condition in config["conditions"]["validation"]
        for stream in plan["seed_streams"]["validation"]
    ]


def _threshold_bundle(
    root: Path,
    config: dict[str, Any],
    base: dict[str, Any],
    protocol_sha256: str,
    provenance: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str], dict[str, float]]:
    """Rebuild the frozen thresholds from their retained training artifacts."""
    path = root / "thresholds.json"
    if path.is_symlink() or not path.is_file():
        raise ValueError("Validation requires a regular frozen thresholds.json")
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    try:
        bundle = json.loads(data)
    except (TypeError, ValueError) as error:
        raise ValueError("Frozen threshold artifact is not valid JSON") from error
    if not isinstance(bundle, dict) or not isinstance(bundle.get("training_records"), list):
        raise ValueError("Invalid frozen threshold artifact")
    if data != _bytes(bundle):
        raise ValueError("Frozen threshold artifact is not canonical JSON")

    schema = bundle.get("schema_version")
    if schema not in {1, 2}:
        raise ValueError("Unsupported frozen threshold schema")
    manifest_path = root / "manifest.json"
    if manifest_path.exists() and schema != 2:
        raise ValueError("Protocol-bound runs require complete training retention")
    base_keys = {"id", "case", "artifact", "score_sha256"}
    row_keys = (
        base_keys
        if schema == 1
        else base_keys | {"detector_artifact", "metadata"}
    )
    records = []
    saved_rows: dict[
        str, tuple[dict[str, Any], np.ndarray, dict[str, str] | None]
    ] = {}
    for row in bundle["training_records"]:
        if not isinstance(row, dict) or set(row) != row_keys:
            raise ValueError("Invalid frozen training-record binding")
        if not isinstance(row["case"], dict):
            raise ValueError("Invalid frozen training-record case")
        try:
            case_id = _case_id(row["case"])
        except (TypeError, ValueError) as error:
            raise ValueError("Invalid frozen training-record case") from error
        if row["id"] != case_id:
            raise ValueError("Frozen training-record identity changed")
        if schema == 2:
            artifact, arrays, metadata = verify_training_evidence(
                root,
                config,
                base,
                row["case"],
                row["artifact"],
                row["metadata"],
            )
        else:
            artifact, arrays = _artifact_arrays(root, row["artifact"], "training")
            metadata = None
        if "score" not in arrays:
            raise ValueError("Frozen training artifact is missing its score")
        score = _score(config, arrays["score"])
        if row["score_sha256"] != _array_digest(score):
            raise ValueError("Frozen training score changed")
        saved_rows[case_id] = (row, score, metadata)
        records.append(
            {
                "case": row["case"],
                "status": "completed",
                "artifact": artifact,
                "score": score,
                "error": None,
            }
        )
    rebuilt = _build_threshold_bundle(config, protocol_sha256, provenance, records)
    thresholds = {
        row["regime"]["id"]: float(row["threshold"])
        for row in rebuilt["thresholds"]
    }
    if schema == 2:
        rebuilt_rows = {row["id"]: row for row in rebuilt["training_records"]}
        for case in training_cases(config):
            case_id = _case_id(case)
            saved, score, metadata = saved_rows[case_id]
            if metadata is None:
                raise ValueError("Frozen training metadata is missing")
            detector_artifact = verify_training_trace(
                root,
                config,
                case,
                score,
                thresholds[case["regime_id"]],
                saved["detector_artifact"],
            )
            rebuilt_rows[case_id]["detector_artifact"] = detector_artifact
            rebuilt_rows[case_id]["metadata"] = metadata
        rebuilt["schema_version"] = 2
        rebuilt["training_artifact_layout"] = training_artifact_layout(config)
        rebuilt["training_manifest_sha256"] = _digest(
            _bytes(rebuilt["training_records"])
        )
    if _bytes(bundle) != _bytes(rebuilt):
        raise ValueError("Frozen thresholds do not reproduce from their bound evidence")
    if _file_digest(path) != digest:
        raise ValueError("Frozen threshold artifact changed while it was audited")
    return bundle, {"path": path.name, "sha256": digest}, thresholds


def _true_runs(flags: np.ndarray, frames: np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(flags)
    identities = np.asarray(frames)
    if (
        values.ndim != 1
        or values.dtype != np.dtype(bool)
        or identities.shape != values.shape
        or identities.dtype.kind not in "iu"
    ):
        raise ValueError("Run accounting requires aligned boolean flags and integer frames")
    if len(values) == 0:
        return []
    padded = np.pad(values.astype(np.int8), (1, 1))
    starts = np.flatnonzero(np.diff(padded) == 1)
    ends = np.flatnonzero(np.diff(padded) == -1) - 1
    return [
        (int(identities[start]), int(identities[end]))
        for start, end in zip(starts, ends)
    ]


def _target_metrics(
    config: dict[str, Any], trace: dict[str, np.ndarray]
) -> dict[str, Any]:
    onset = config["loop"]["fault_onset"]
    fault_end = config["loop"]["fault_end"]
    persistence = config["detector"]["persistence_windows"]
    runs = [
        (start, end)
        for start, end in _true_runs(
            trace["raw_crossing"], trace["statistic_frame_id"]
        )
        if end - start + 1 >= persistence
    ]
    pre_onset = any(start + persistence - 1 < onset for start, _ in runs)
    boundary = any(start < onset <= end for start, end in runs)
    completions = [
        start + persistence - 1
        for start, end in runs
        if start >= onset and start + persistence - 1 <= end
        and start + persistence - 1 < fault_end
    ]
    detected = not pre_onset and bool(completions)
    first = min(completions) if detected else None
    return {
        "pre_onset_alarm": pre_onset,
        "boundary_straddling_alarm": boundary,
        "detected": detected,
        "detection_delay_frames": None if first is None else first - onset,
    }


def _completed_record(
    root: Path,
    config: dict[str, Any],
    base: dict[str, Any],
    case: dict[str, Any],
    reference: Any,
    metadata_reference: Any,
    threshold: float,
) -> dict[str, Any]:
    artifact, arrays, metadata, saved_metadata = verify_validation_evidence(
        root,
        config,
        base,
        case,
        reference,
        metadata_reference,
        threshold,
    )
    score = _score(config, arrays["score"])
    trace = detector_trace(config, score, threshold)

    target = _target_metrics(config, trace)
    metrics = {
        "all_finite": True,
        "alarm_run": bool(trace["p2_alarm"].any()),
        "raw_crossings": int(trace["raw_crossing"].sum()),
        "raw_crossing_frames": len(trace["raw_crossing"]),
        "alarm_frames": int(trace["p2_alarm"].sum()),
        "alarm_eligible_frames": len(trace["p2_alarm"]),
        "fallback_frames": int(arrays["fallback"].sum()),
        "unavailable_frames": int(arrays["unavailable"].sum()),
        "saturated_frames": int(arrays["saturated"].sum()),
        "saturation_fraction": float(arrays["saturated"].mean()),
        "retained_trial_metrics": saved_metadata["metrics"],
        **target,
    }
    return {
        "id": _case_id(case),
        "case": case,
        "status": "completed",
        "artifact": artifact,
        "metadata": metadata,
        "metrics": metrics,
        "error": None,
    }


def _validation_records(
    root: Path,
    config: dict[str, Any],
    base: dict[str, Any],
    thresholds: dict[str, float],
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(records, list):
        raise ValueError("Validation records must be a list")
    expected_cases = validation_cases(config)
    expected = {_case_id(case): case for case in expected_cases}
    found: dict[str, dict[str, Any]] = {}
    artifact_paths: set[str] = set()
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("case"), dict):
            raise ValueError("Invalid validation record schema")
        try:
            case_id = _case_id(record["case"])
        except (TypeError, ValueError) as error:
            raise ValueError("Invalid validation case schema") from error
        if (
            case_id not in expected
            or record["case"] != expected[case_id]
            or case_id in found
        ):
            raise ValueError("Foreign or duplicate validation identity")
        case = expected[case_id]
        row = _normalize_validation_record(
            root, config, base, thresholds, record, case
        )
        if row["artifact"] is not None:
            if row["artifact"]["path"] in artifact_paths:
                raise ValueError("Validation records cannot share a raw artifact")
            artifact_paths.add(row["artifact"]["path"])
        found[case_id] = row
    if len(found) != len(expected):
        raise ValueError("Release requires the complete declared validation cohort")
    return [found[_case_id(case)] for case in expected_cases]


def _normalize_validation_record(
    root: Path,
    config: dict[str, Any],
    base: dict[str, Any],
    thresholds: dict[str, float],
    record: dict[str, Any],
    expected_case: dict[str, Any],
) -> dict[str, Any]:
    """Validate one terminal record and derive its immutable release metrics."""
    if not isinstance(record, dict) or set(record) != {
        "case",
        "status",
        "artifact",
        "metadata",
        "error",
    }:
        raise ValueError("Invalid validation record schema")
    if record["case"] != expected_case:
        raise ValueError("Validation record identity changed")
    case_id = _case_id(expected_case)
    if record["status"] == "completed":
        if record["error"] is not None:
            raise ValueError("Completed validation record cannot contain an error")
        return _completed_record(
            root,
            config,
            base,
            expected_case,
            record["artifact"],
            record["metadata"],
            thresholds[expected_case["regime_id"]],
        )
    if record["status"] == "failed":
        if not isinstance(record["error"], str) or not record["error"].strip():
            raise ValueError("Failed validation record requires an error")
        artifact, metadata = record["artifact"], record["metadata"]
        if (artifact is None) != (metadata is None):
            raise ValueError("Failed validation raw evidence and metadata must be paired")
        if artifact is not None:
            artifact, metadata = verify_failed_evidence(
                root,
                config,
                base,
                expected_case,
                artifact,
                metadata,
                "validation",
                record["error"],
            )
        return {
            "id": case_id,
            "case": expected_case,
            "status": "failed",
            "artifact": artifact,
            "metadata": metadata,
            "metrics": None,
            "error": record["error"],
        }
    raise ValueError("Unknown validation record status")


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _criterion(
    observed: Any, comparison: str, limit: Any, passed: bool
) -> dict[str, Any]:
    return {
        "observed": observed,
        "comparison": comparison,
        "limit": limit,
        "passed": bool(passed),
    }


def _regime_decision(
    config: dict[str, Any], regime: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    gate = config["validation"]
    planned = len(config["seeds"]["validation"])
    clean = [row for row in rows if row["case"]["condition"] == "clean"]
    target = [row for row in rows if row["case"]["condition"] == "gain_035"]
    clean_complete = [row for row in clean if row["status"] == "completed"]
    target_complete = [row for row in target if row["status"] == "completed"]
    complete = clean_complete + target_complete
    failed = [row for row in rows if row["status"] == "failed"]

    clean_alarm_runs = sum(row["metrics"]["alarm_run"] for row in clean_complete)
    raw_crossings = sum(row["metrics"]["raw_crossings"] for row in clean_complete)
    raw_frames = sum(row["metrics"]["raw_crossing_frames"] for row in clean_complete)
    alarm_frames = sum(row["metrics"]["alarm_frames"] for row in clean_complete)
    eligible_frames = sum(
        row["metrics"]["alarm_eligible_frames"] for row in clean_complete
    )
    raw_fraction = _ratio(raw_crossings, raw_frames)
    alarm_fraction = _ratio(alarm_frames, eligible_frames)

    pre_onset = sum(row["metrics"]["pre_onset_alarm"] for row in target_complete)
    boundary = sum(
        row["metrics"]["boundary_straddling_alarm"] for row in target_complete
    )
    detected = sum(row["metrics"]["detected"] for row in target_complete)
    missed = planned - detected
    delays = sorted(
        row["metrics"]["detection_delay_frames"]
        for row in target_complete
        if row["metrics"]["detected"]
    )
    maximum_delay = max(delays) if delays else None
    fallback_frames = sum(row["metrics"]["fallback_frames"] for row in complete)
    unavailable_frames = sum(row["metrics"]["unavailable_frames"] for row in complete)
    saturated_frames = sum(row["metrics"]["saturated_frames"] for row in complete)
    total_frames = len(complete) * config["loop"]["frames"]
    saturation_fraction = _ratio(saturated_frames, total_frames)
    complete_cells = len(clean_complete) == planned and len(target_complete) == planned
    all_finite = complete_cells and all(
        row["metrics"]["all_finite"] for row in complete
    )

    criteria = {
        "clean_alarm_runs": _criterion(
            clean_alarm_runs,
            "<=",
            gate["max_clean_alarm_runs"],
            complete_cells and clean_alarm_runs <= gate["max_clean_alarm_runs"],
        ),
        "clean_raw_crossing_fraction": _criterion(
            raw_fraction,
            "<=",
            gate["max_clean_raw_crossing_fraction"],
            complete_cells
            and raw_fraction is not None
            and raw_fraction <= gate["max_clean_raw_crossing_fraction"],
        ),
        "clean_alarm_frame_fraction": _criterion(
            alarm_fraction,
            "<=",
            gate["max_clean_alarm_frame_fraction"],
            complete_cells
            and alarm_fraction is not None
            and alarm_fraction <= gate["max_clean_alarm_frame_fraction"],
        ),
        "gain_pre_onset_alarm_runs": _criterion(
            pre_onset,
            "<=",
            gate["max_gain_pre_onset_alarm_runs"],
            complete_cells and pre_onset <= gate["max_gain_pre_onset_alarm_runs"],
        ),
        "gain_detected_runs": _criterion(
            detected,
            ">=",
            gate["min_gain_detected_runs"],
            complete_cells and detected >= gate["min_gain_detected_runs"],
        ),
        "gain_missed_runs": _criterion(
            missed,
            "<=",
            gate["max_gain_missed_runs"],
            complete_cells and missed <= gate["max_gain_missed_runs"],
        ),
        "maximum_detection_delay_frames": _criterion(
            maximum_delay,
            "<=",
            gate["max_detection_delay_frames"],
            complete_cells
            and maximum_delay is not None
            and maximum_delay <= gate["max_detection_delay_frames"],
        ),
        "failed_runs": _criterion(len(failed), "==", 0, len(failed) == 0),
        "all_finite_outputs": _criterion(
            all_finite,
            "==",
            True,
            not gate["require_all_finite"] or all_finite,
        ),
        "fallback_frames": _criterion(
            fallback_frames,
            "==",
            0,
            not gate["require_no_fallback"] or fallback_frames == 0,
        ),
        "unavailable_frames": _criterion(
            unavailable_frames,
            "==",
            0,
            not gate["require_no_unavailable"] or unavailable_frames == 0,
        ),
        "saturation_fraction": _criterion(
            saturation_fraction,
            "<=",
            gate["max_saturation_fraction"],
            complete_cells
            and saturation_fraction is not None
            and saturation_fraction <= gate["max_saturation_fraction"],
        ),
    }
    return {
        "regime": regime,
        "planned_runs_per_condition": planned,
        "completed_clean_runs": len(clean_complete),
        "completed_gain_runs": len(target_complete),
        "failed_runs": len(failed),
        "clean": {
            "alarm_runs": clean_alarm_runs,
            "raw_crossings": raw_crossings,
            "raw_crossing_frames": raw_frames,
            "raw_crossing_fraction": raw_fraction,
            "alarm_frames": alarm_frames,
            "alarm_eligible_frames": eligible_frames,
            "alarm_frame_fraction": alarm_fraction,
        },
        "gain_035": {
            "pre_onset_alarm_runs": pre_onset,
            "boundary_straddling_alarm_runs": boundary,
            "detected_runs": detected,
            "missed_runs": missed,
            "detected_only_delay_frames": delays,
        },
        "integrity": {
            "all_completed_outputs_finite": all_finite,
            "fallback_frames": fallback_frames,
            "unavailable_frames": unavailable_frames,
            "saturated_frames": saturated_frames,
            "saturation_fraction": saturation_fraction,
        },
        "criteria": criteria,
        "passed": all(row["passed"] for row in criteria.values()),
    }


def _build_validation_bundle(
    config: dict[str, Any],
    protocol_sha256: str,
    provenance: dict[str, Any],
    threshold_reference: dict[str, str],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Apply every frozen per-regime gate to normalized validation records."""
    validate_protocol(config)
    _provenance(provenance)
    plan = protocol_plan(config)
    decisions = [
        _regime_decision(
            config,
            regime,
            [row for row in records if row["case"]["regime_id"] == regime["id"]],
        )
        for regime in plan["regimes"]
    ]
    released = all(row["passed"] for row in decisions)
    detached_provenance = json.loads(_bytes(provenance))
    return {
        "schema_version": 1,
        "kind": "optical_gain_validation_release",
        "status": "evaluation_released" if released else "evaluation_withheld",
        "study_id": config["study_id"],
        "bindings": {
            "protocol_sha256": protocol_sha256,
            "base_config_sha256": config["base_config_sha256"],
            "threshold_sha256": threshold_reference["sha256"],
            **detached_provenance,
        },
        "threshold_reference": threshold_reference,
        "validation_records": records,
        "validation_manifest_sha256": _digest(_bytes(records)),
        "regimes": decisions,
        "failure_action": config["validation"]["failure_action"],
        "evaluation_released": released,
    }


def _publish_release(root: Path, bundle: dict[str, Any]) -> dict[str, str]:
    path = root / "validation_release.json"
    if path.is_symlink():
        raise ValueError("Validation release cannot replace a symlink")
    _publish(path, bundle)
    data = path.read_bytes()
    return {"path": path.name, "sha256": hashlib.sha256(data).hexdigest()}


def freeze_validation_release(
    root: Path,
    protocol_path: Path,
    provenance: dict[str, Any],
    records: list[dict[str, Any]],
    repository: Path | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Audit the complete validation cohort and publish its one-way release gate."""
    requested_root = Path(root)
    if not requested_root.is_dir() or requested_root.is_symlink():
        raise ValueError("Validation root must be an existing non-symlink directory")
    root = requested_root.resolve()
    config, protocol_sha256, protocol_file, resolved_repository = _load_protocol_binding(
        protocol_path, repository
    )
    checked_config, base = load_resource_inputs(protocol_file, resolved_repository)
    if _bytes(checked_config) != _bytes(config):
        raise ValueError("Protocol changed while the pinned base was loaded")
    threshold_bundle, threshold_reference, thresholds = _threshold_bundle(
        root, config, base, protocol_sha256, provenance
    )
    normalized = _validation_records(root, config, base, thresholds, records)
    bundle = _build_validation_bundle(
        config, protocol_sha256, provenance, threshold_reference, normalized
    )

    if (
        _file_digest(root / threshold_reference["path"])
        != threshold_reference["sha256"]
    ):
        raise ValueError("Frozen threshold artifact changed before release publication")
    for row in threshold_bundle["training_records"]:
        for key in ("artifact", "detector_artifact", "metadata"):
            if key in row:
                path = root / row[key]["path"]
                if _file_digest(path) != row[key]["sha256"]:
                    raise ValueError("Training artifact changed before release publication")
    for row in normalized:
        for key in ("artifact", "metadata"):
            if row.get(key) is not None:
                path = root / row[key]["path"]
                if _file_digest(path) != row[key]["sha256"]:
                    raise ValueError("Validation artifact changed before release publication")
    if _file_digest(protocol_file) != protocol_sha256:
        raise ValueError("Protocol changed before validation release publication")
    return bundle, _publish_release(root, bundle)
