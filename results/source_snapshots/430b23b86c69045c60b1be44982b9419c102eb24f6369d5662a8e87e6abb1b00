"""Release-gated P2 evaluation records and descriptive final statistics.

The report is immutable and outcome-descriptive. It never changes the detector,
reopens validation or applies a second acceptance gate. Terminal numerical failures
remain in the declared cohort and incomplete physical pairs are reported, not replaced.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
from scipy.stats import beta as beta_distribution
from scipy.stats import t as student_t

from .controller_selection import _bytes, _digest, _provenance, _publish
from .observability_detector import _case_id
from .observability_protocol import protocol_plan, validate_protocol
from .observability_trials import verify_complete_evidence, verify_failed_evidence
from .observability_validation import _target_metrics


def evaluation_cases(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Enumerate held-out identities without reading release state or evidence."""
    validate_protocol(config)
    plan = protocol_plan(config)
    return [
        {
            "role": "evaluation",
            "condition": condition,
            "regime_id": regime["id"],
            "photons": regime["photons"],
            "wind_scale": regime["wind_scale"],
            "atmosphere_seed": stream["atmosphere"],
            "detector_seed": stream["detector"],
        }
        for regime in plan["regimes"]
        for condition in config["conditions"]["evaluation"]
        for stream in plan["seed_streams"]["evaluation"]
    ]


def clopper_pearson(
    events: int, trials: int, confidence: float
) -> dict[str, Any]:
    """Return a two-sided exact binomial interval with explicit denominator."""
    if (
        type(events) is not int
        or type(trials) is not int
        or events < 0
        or trials < 0
        or events > trials
        or type(confidence) is not float
        or not 0.0 < confidence < 1.0
    ):
        raise ValueError("Invalid exact-binomial interval inputs")
    if trials == 0:
        return {
            "events": events,
            "trials": trials,
            "fraction": None,
            "confidence": confidence,
            "method": "clopper_pearson_exact_two_sided",
            "lower": None,
            "upper": None,
        }
    alpha = 1.0 - confidence
    lower = (
        0.0
        if events == 0
        else float(beta_distribution.ppf(alpha / 2.0, events, trials - events + 1))
    )
    upper = (
        1.0
        if events == trials
        else float(
            beta_distribution.ppf(
                1.0 - alpha / 2.0, events + 1, trials - events
            )
        )
    )
    return {
        "events": events,
        "trials": trials,
        "fraction": events / trials,
        "confidence": confidence,
        "method": "clopper_pearson_exact_two_sided",
        "lower": lower,
        "upper": upper,
    }


def paired_student_interval(
    values: list[float], familywise_alpha: float, family_size: int
) -> dict[str, Any]:
    """Return the frozen Bonferroni-adjusted paired Student-t interval."""
    if (
        not isinstance(values, list)
        or any(type(value) not in (int, float) for value in values)
        or any(not math.isfinite(float(value)) for value in values)
        or type(familywise_alpha) is not float
        or not 0.0 < familywise_alpha < 1.0
        or type(family_size) is not int
        or family_size < 1
    ):
        raise ValueError("Invalid paired interval inputs")
    confidence = 1.0 - familywise_alpha / family_size
    normalized = np.asarray(values, dtype=np.float64)
    result = {
        "method": "paired_student_t_bonferroni",
        "familywise_alpha": familywise_alpha,
        "family_size": family_size,
        "confidence": confidence,
        "pairs": len(values),
        "mean": None,
        "lower": None,
        "upper": None,
    }
    if len(values) == 0:
        return result
    mean = float(normalized.mean())
    result["mean"] = mean
    if len(values) == 1:
        return result
    critical = float(
        student_t.ppf(1.0 - (familywise_alpha / family_size) / 2.0, len(values) - 1)
    )
    half_width = float(critical * normalized.std(ddof=1) / np.sqrt(len(values)))
    result.update(lower=mean - half_width, upper=mean + half_width)
    return result


def _fault_window_metrics(
    config: dict[str, Any], base: dict[str, Any], arrays: dict[str, np.ndarray]
) -> dict[str, float]:
    start, stop = config["loop"]["fault_onset"], config["loop"]["fault_end"]
    variance = arrays["variance"][start:stop]
    strehl = arrays["strehl"][start:stop]
    saturated = arrays["saturated"][start:stop]
    if (
        len(variance) != stop - start
        or len(variance) == 0
        or not np.isfinite(variance).all()
        or not np.isfinite(strehl).all()
        or np.any(variance < 0)
    ):
        raise ValueError("Evaluation fault window is not finite and complete")
    return {
        "wfe_nm": float(
            np.sqrt(variance.mean()) * base["wavelength_m"] / (2 * np.pi) * 1e9
        ),
        "strehl": float(strehl.mean()),
        "saturation_fraction": float(saturated.mean()),
    }


def _completed_metrics(
    config: dict[str, Any],
    base: dict[str, Any],
    arrays: dict[str, np.ndarray],
    saved_metadata: dict[str, Any],
) -> dict[str, Any]:
    target = _target_metrics(config, arrays)
    return {
        "all_finite": True,
        "alarm_run": bool(arrays["p2_alarm"].any()),
        "raw_crossings": int(arrays["raw_crossing"].sum()),
        "raw_crossing_frames": len(arrays["raw_crossing"]),
        "alarm_frames": int(arrays["p2_alarm"].sum()),
        "alarm_eligible_frames": len(arrays["p2_alarm"]),
        "fallback_frames": int(arrays["fallback"].sum()),
        "unavailable_frames": int(arrays["unavailable"].sum()),
        "saturated_frames": int(arrays["saturated"].sum()),
        "saturation_fraction": float(arrays["saturated"].mean()),
        "retained_trial_metrics": saved_metadata["metrics"],
        "fault_window": _fault_window_metrics(config, base, arrays),
        **target,
    }


def normalize_evaluation_record(
    root: Path,
    config: dict[str, Any],
    base: dict[str, Any],
    thresholds: dict[str, float],
    record: dict[str, Any],
    expected_case: dict[str, Any],
) -> dict[str, Any]:
    """Validate one terminal evaluation record and derive report metrics."""
    if not isinstance(record, dict) or set(record) != {
        "case",
        "status",
        "artifact",
        "metadata",
        "error",
    }:
        raise ValueError("Invalid evaluation record schema")
    if record["case"] != expected_case:
        raise ValueError("Evaluation record identity changed")
    case_id = _case_id(expected_case)
    if record["status"] == "completed":
        if record["error"] is not None:
            raise ValueError("Completed evaluation record cannot contain an error")
        artifact, arrays, metadata, saved_metadata = verify_complete_evidence(
            root,
            config,
            base,
            expected_case,
            record["artifact"],
            record["metadata"],
            thresholds[expected_case["regime_id"]],
            "evaluation",
        )
        return {
            "id": case_id,
            "case": expected_case,
            "status": "completed",
            "artifact": artifact,
            "metadata": metadata,
            "metrics": _completed_metrics(config, base, arrays, saved_metadata),
            "error": None,
        }
    if record["status"] == "failed":
        if not isinstance(record["error"], str) or not record["error"].strip():
            raise ValueError("Failed evaluation record requires an error")
        artifact, metadata = record["artifact"], record["metadata"]
        if (artifact is None) != (metadata is None):
            raise ValueError("Failed evaluation raw evidence and metadata must be paired")
        if artifact is not None:
            artifact, metadata = verify_failed_evidence(
                root,
                config,
                base,
                expected_case,
                artifact,
                metadata,
                "evaluation",
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
    raise ValueError("Unknown evaluation record status")


def _evaluation_records(
    root: Path,
    config: dict[str, Any],
    base: dict[str, Any],
    thresholds: dict[str, float],
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(records, list):
        raise ValueError("Evaluation records must be a list")
    expected_cases = evaluation_cases(config)
    expected = {_case_id(case): case for case in expected_cases}
    found: dict[str, dict[str, Any]] = {}
    artifact_paths: set[str] = set()
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("case"), dict):
            raise ValueError("Invalid evaluation record schema")
        try:
            case_id = _case_id(record["case"])
        except (TypeError, ValueError) as error:
            raise ValueError("Invalid evaluation case schema") from error
        if case_id not in expected or record["case"] != expected[case_id] or case_id in found:
            raise ValueError("Foreign or duplicate evaluation identity")
        row = normalize_evaluation_record(
            root, config, base, thresholds, record, expected[case_id]
        )
        if row["artifact"] is not None:
            path = row["artifact"]["path"]
            if path in artifact_paths:
                raise ValueError("Evaluation records cannot share a raw artifact")
            artifact_paths.add(path)
        found[case_id] = row
    if len(found) != len(expected):
        raise ValueError("Report requires the complete declared evaluation cohort")
    return [found[_case_id(case)] for case in expected_cases]


def _fraction(events: int, trials: int, confidence: float) -> dict[str, Any]:
    return clopper_pearson(events, trials, confidence)


def _frame_fraction(events: int, frames: int) -> dict[str, Any]:
    return {
        "events": events,
        "frames": frames,
        "fraction": events / frames if frames else None,
        "interval": None,
        "reason": "temporally_correlated_descriptive_only",
    }


def _condition_summary(
    config: dict[str, Any], condition: str, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    confidence = config["evaluation"]["run_interval_confidence"]
    completed = [row for row in rows if row["status"] == "completed"]
    failed = [row for row in rows if row["status"] == "failed"]
    summary: dict[str, Any] = {
        "planned_runs": len(rows),
        "completed_runs": len(completed),
        "failed_runs": len(failed),
        "integrity": {
            "finite_runs": len(completed),
            "failed_runs": len(failed),
            "fallback_frames": sum(
                row["metrics"]["fallback_frames"] for row in completed
            ),
            "unavailable_frames": sum(
                row["metrics"]["unavailable_frames"] for row in completed
            ),
            "saturated_frames": sum(
                row["metrics"]["saturated_frames"] for row in completed
            ),
            "observed_frames": len(completed) * config["loop"]["frames"],
        },
    }
    observed_frames = summary["integrity"]["observed_frames"]
    summary["integrity"]["saturation_fraction"] = (
        summary["integrity"]["saturated_frames"] / observed_frames
        if observed_frames
        else None
    )
    if condition == "clean":
        alarms = sum(row["metrics"]["alarm_run"] for row in completed)
        raw = sum(row["metrics"]["raw_crossings"] for row in completed)
        raw_frames = sum(row["metrics"]["raw_crossing_frames"] for row in completed)
        alarm_frames = sum(row["metrics"]["alarm_frames"] for row in completed)
        eligible = sum(row["metrics"]["alarm_eligible_frames"] for row in completed)
        summary.update(
            alarm_runs=alarms,
            alarm_run_fraction=_fraction(alarms, len(completed), confidence),
            raw_crossing_fraction=_frame_fraction(raw, raw_frames),
            alarm_frames=alarm_frames,
            alarm_frame_fraction=_frame_fraction(alarm_frames, eligible),
            denominator_note="failed runs are unclassified, never counted as quiet",
        )
        return summary

    pre_onset = sum(row["metrics"]["pre_onset_alarm"] for row in completed)
    boundary = sum(
        row["metrics"]["boundary_straddling_alarm"] for row in completed
    )
    post_onset = sum(row["metrics"]["detected"] for row in completed)
    delays = sorted(
        row["metrics"]["detection_delay_frames"]
        for row in completed
        if row["metrics"]["detected"]
    )
    summary.update(
        pre_onset_alarm_runs=_fraction(pre_onset, len(completed), confidence),
        boundary_straddling_alarm_runs=_fraction(
            boundary, len(completed), confidence
        ),
    )
    if condition in config["conditions"]["target_conditions"]:
        summary.update(
            detected_runs_without_prior_alarm=_fraction(
                post_onset, len(rows), confidence
            ),
            missed_runs=_fraction(len(rows) - post_onset, len(rows), confidence),
            detected_only_delay_frames=delays,
            denominator_note="terminal failures count as misses; alarm classes use completed runs",
        )
    else:
        quiet = sum(not row["metrics"]["alarm_run"] for row in completed)
        summary.update(
            alarm_runs_without_prior_alarm=_fraction(
                post_onset, len(completed), confidence
            ),
            quiet_runs=_fraction(quiet, len(completed), confidence),
            alarm_delay_frames=delays,
            denominator_note="failed runs are unclassified, never counted as quiet",
        )
    return summary


def _paired_cell(
    config: dict[str, Any],
    regime_id: str,
    condition: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    clean = {
        (row["case"]["atmosphere_seed"], row["case"]["detector_seed"]): row
        for row in rows
        if row["case"]["condition"] == "clean"
    }
    fault = {
        (row["case"]["atmosphere_seed"], row["case"]["detector_seed"]): row
        for row in rows
        if row["case"]["condition"] == condition
    }
    expected = [
        (seed, seed + config["seeds"]["noise_offset"])
        for seed in config["seeds"]["evaluation"]
    ]
    wfe, strehl, saturation, incomplete = [], [], [], []
    for key in expected:
        left, right = clean[key], fault[key]
        if left["status"] != "completed" or right["status"] != "completed":
            incomplete.append(
                {
                    "atmosphere_seed": key[0],
                    "detector_seed": key[1],
                    "clean_status": left["status"],
                    "fault_status": right["status"],
                    "clean_error": left["error"],
                    "fault_error": right["error"],
                }
            )
            continue
        clean_window = left["metrics"]["fault_window"]
        fault_window = right["metrics"]["fault_window"]
        wfe.append(fault_window["wfe_nm"] - clean_window["wfe_nm"])
        strehl.append(fault_window["strehl"] - clean_window["strehl"])
        saturation.append(
            {
                "atmosphere_seed": key[0],
                "clean": clean_window["saturation_fraction"],
                "fault": fault_window["saturation_fraction"],
            }
        )
    family = protocol_plan(config)["predeclared_paired_error_family"]
    interval = paired_student_interval(
        wfe,
        config["evaluation"]["paired_error_familywise_alpha"],
        family,
    )
    return {
        "regime_id": regime_id,
        "condition": condition,
        "interval_frames": [
            config["loop"]["fault_onset"],
            config["loop"]["fault_end"],
        ],
        "planned_pairs": len(expected),
        "complete_pairs": len(wfe),
        "incomplete_pairs": incomplete,
        "paired_wfe_nm_delta": {**interval, "values": wfe},
        "paired_strehl_delta": {
            "method": "descriptive_only",
            "values": strehl,
            "mean": float(np.mean(strehl)) if strehl else None,
        },
        "saturation_fraction": {
            "method": "descriptive_only",
            "pairs": saturation,
        },
    }


def _regime_report(
    config: dict[str, Any], regime: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    by_condition = {
        condition: [row for row in rows if row["case"]["condition"] == condition]
        for condition in config["conditions"]["evaluation"]
    }
    summaries = {
        condition: _condition_summary(config, condition, values)
        for condition, values in by_condition.items()
    }
    nonclean = [
        condition
        for condition in config["conditions"]["evaluation"]
        if condition != "clean"
    ]
    return {
        "regime": regime,
        "conditions": summaries,
        "paired_controller_error": [
            _paired_cell(config, regime["id"], condition, rows)
            for condition in nonclean
        ],
    }


def _binding(reference: Any, expected_path: str, label: str) -> dict[str, str]:
    if (
        not isinstance(reference, dict)
        or set(reference) != {"path", "sha256"}
        or reference["path"] != expected_path
        or not isinstance(reference["sha256"], str)
        or not re.fullmatch(r"[0-9a-f]{64}", reference["sha256"])
    ):
        raise ValueError(f"Invalid {label} binding")
    return {"path": expected_path, "sha256": reference["sha256"]}


def build_evaluation_report(
    root: Path,
    config: dict[str, Any],
    base: dict[str, Any],
    manifest_sha256: str,
    protocol_sha256: str,
    provenance: dict[str, Any],
    threshold_reference: dict[str, str],
    validation_reference: dict[str, str],
    thresholds: dict[str, float],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Reconstruct the complete descriptive report from retained evidence."""
    validate_protocol(config)
    _provenance(provenance)
    for label, value in (
        ("manifest", manifest_sha256),
        ("protocol", protocol_sha256),
    ):
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError(f"Require the exact {label} SHA256")
    threshold = _binding(threshold_reference, "thresholds.json", "threshold")
    release = _binding(
        validation_reference, "validation_release.json", "validation release"
    )
    normalized = _evaluation_records(root, config, base, thresholds, records)
    plan = protocol_plan(config)
    regimes = [
        _regime_report(
            config,
            regime,
            [row for row in normalized if row["case"]["regime_id"] == regime["id"]],
        )
        for regime in plan["regimes"]
    ]
    failed = [
        {"id": row["id"], "case": row["case"], "error": row["error"]}
        for row in normalized
        if row["status"] == "failed"
    ]
    completed = [row for row in normalized if row["status"] == "completed"]
    return {
        "schema_version": 1,
        "kind": "optical_gain_observability_evaluation_report",
        "status": "complete_descriptive",
        "study_id": config["study_id"],
        "bindings": {
            "manifest_sha256": manifest_sha256,
            "protocol_sha256": protocol_sha256,
            "base_config_sha256": config["base_config_sha256"],
            "threshold_sha256": threshold["sha256"],
            "validation_release_sha256": release["sha256"],
            **json.loads(_bytes(provenance)),
        },
        "threshold_reference": threshold,
        "validation_release_reference": release,
        "evaluation_records": normalized,
        "evaluation_manifest_sha256": _digest(_bytes(normalized)),
        "record_counts": {
            "planned": len(normalized),
            "completed": len(completed),
            "failed": len(failed),
        },
        "failed_records": failed,
        "regimes": regimes,
        "inference": {
            "run_interval_method": config["evaluation"]["run_interval_method"],
            "run_interval_confidence": config["evaluation"]["run_interval_confidence"],
            "frame_interval_method": config["evaluation"]["frame_interval_method"],
            "paired_error_method": config["evaluation"]["paired_error_method"],
            "paired_error_familywise_alpha": config["evaluation"][
                "paired_error_familywise_alpha"
            ],
            "paired_error_family_size": plan["predeclared_paired_error_family"],
            "paired_error_confidence": plan["simultaneous_interval_confidence"],
            "failed_run_policy": {
                "target_detection": "terminal failures count as misses",
                "clean_and_nuisance": "terminal failures are unclassified and never quiet",
                "paired_error": "incomplete pairs are listed and not imputed",
            },
        },
        "evaluation_released": True,
        "second_pass_gate": False,
    }


def publish_evaluation_report(root: Path, report: dict[str, Any]) -> dict[str, str]:
    path = Path(root) / "evaluation_report.json"
    if path.is_symlink():
        raise ValueError("Evaluation report cannot replace a symlink")
    _publish(path, report)
    data = path.read_bytes()
    return {"path": path.name, "sha256": hashlib.sha256(data).hexdigest()}
