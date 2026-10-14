"""Causal P2 detector transforms and immutable training-threshold calibration."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any
import zipfile

import numpy as np

from .controller_selection import _bytes, _digest, _provenance, _publish
from .observability_protocol import load_protocol, protocol_plan, validate_protocol


ROOT = Path(__file__).resolve().parents[2]


def causal_mean(
    score: np.ndarray, burn_in: int, window: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return frame identities and full causal windows after burn-in."""
    values = np.asarray(score)
    if values.ndim != 1 or values.dtype.kind not in "fiu" or not np.isfinite(values).all():
        raise ValueError("Score must be a finite vector of real values")
    if (
        type(burn_in) is not int
        or type(window) is not int
        or burn_in < 0
        or window < 1
    ):
        raise ValueError("Burn-in and window must be nonnegative/positive integers")
    if burn_in + window > len(values):
        raise ValueError("Score is too short for a full post-burn-in window")
    with np.errstate(over="ignore", invalid="ignore"):
        values = np.array(values, dtype=np.float64, copy=True, order="C")
    if not np.isfinite(values).all():
        raise ValueError("Score cannot be represented as a finite float64 vector")
    statistic = np.convolve(values[burn_in:], np.ones(window) / window, mode="valid")
    frames = np.arange(burn_in + window - 1, len(values), dtype=np.int64)
    return frames, statistic


def persistent_alarm(crossings: np.ndarray, persistence: int) -> np.ndarray:
    """Mark each endpoint whose current crossing streak meets ``persistence``."""
    values = np.asarray(crossings)
    if values.ndim != 1 or values.dtype != np.dtype(bool):
        raise ValueError("Crossings must be a boolean vector")
    if type(persistence) is not int or persistence < 1:
        raise ValueError("Persistence must be a positive integer")
    if persistence > len(values):
        raise ValueError("Crossing vector is too short for the requested persistence")
    counts = np.convolve(
        values.astype(np.int64), np.ones(persistence, dtype=np.int64), mode="full"
    )
    alarms = counts[: len(values)] == persistence
    alarms[: persistence - 1] = False
    return alarms


def maximum_true_run(values: np.ndarray) -> int:
    """Return the longest consecutive run in a boolean vector."""
    flags = np.asarray(values)
    if flags.ndim != 1 or flags.dtype != np.dtype(bool):
        raise ValueError("Values must be a boolean vector")
    maximum = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        maximum = max(maximum, current)
    return maximum


def _score(config: dict[str, Any], score: Any) -> np.ndarray:
    values = np.asarray(score)
    frames = config["loop"]["frames"]
    if (
        values.shape != (frames,)
        or values.dtype.kind not in "fiu"
        or not np.isfinite(values).all()
        or np.any(values < 0)
    ):
        raise ValueError(f"Score must contain {frames} finite nonnegative real frames")
    with np.errstate(over="ignore", invalid="ignore"):
        normalized = np.array(values, dtype=np.float64, copy=True, order="C")
    if not np.isfinite(normalized).all():
        raise ValueError(f"Score must contain {frames} finite float64 frames")
    return normalized


def detector_trace(
    config: dict[str, Any], score: Any, threshold: float
) -> dict[str, np.ndarray]:
    """Apply the frozen transform and return arrays with explicit frame identities."""
    validate_protocol(config)
    if type(threshold) not in (int, float):
        raise ValueError("Threshold must be a finite nonnegative scalar")
    try:
        threshold_value = float(threshold)
    except OverflowError as error:
        raise ValueError("Threshold must be a finite nonnegative scalar") from error
    if not math.isfinite(threshold_value) or threshold_value < 0:
        raise ValueError("Threshold must be a finite nonnegative scalar")
    detector = config["detector"]
    frames, statistic = causal_mean(
        _score(config, score), config["loop"]["burn_in"], detector["window_frames"]
    )
    crossings = statistic < threshold_value
    alarms = persistent_alarm(crossings, detector["persistence_windows"])
    eligible = detector["persistence_windows"] - 1
    return {
        "statistic_frame_id": frames,
        "lower_statistic": statistic,
        "raw_crossing": crossings,
        "alarm_frame_id": frames[eligible:],
        "p2_alarm": alarms[eligible:],
    }


def training_cases(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Enumerate the 64 declared clean-calibration identities without opening a cohort."""
    validate_protocol(config)
    plan = protocol_plan(config)
    return [
        {
            "role": "training",
            "condition": "clean",
            "regime_id": regime["id"],
            "photons": regime["photons"],
            "wind_scale": regime["wind_scale"],
            "atmosphere_seed": stream["atmosphere"],
            "detector_seed": stream["detector"],
        }
        for regime in plan["regimes"]
        for stream in plan["seed_streams"]["training"]
    ]


def _case_id(case: dict[str, Any]) -> str:
    return _digest(_bytes(case))


def _artifact(reference: Any) -> dict[str, str]:
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise ValueError("Training record requires one raw artifact reference")
    path, digest = reference["path"], reference["sha256"]
    if (
        not isinstance(path, str)
        or not path.startswith("evidence/training/")
        or Path(path).is_absolute()
        or Path(path).as_posix() != path
        or ".." in Path(path).parts
        or Path(path).suffix != ".npz"
        or not isinstance(digest, str)
        or not re.fullmatch(r"[0-9a-f]{64}", digest)
    ):
        raise ValueError("Training artifact must be a canonical hashed NPZ reference")
    return {"path": path, "sha256": digest}


def _array_digest(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values, dtype="<f8")
    header = json.dumps(
        {"dtype": "float64", "shape": list(array.shape)}, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(header + b"\0" + array.tobytes()).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_score(root: Path, reference: Any) -> np.ndarray:
    """Verify one immutable raw artifact and read its score without pickle."""
    artifact = _artifact(reference)
    path = root / artifact["path"]
    for current in (path, *path.parents):
        if current == root:
            break
        if current.is_symlink():
            raise ValueError("Training artifact path cannot contain a symlink")
    if not path.is_file() or not path.resolve().is_relative_to(
        root / "evidence" / "training"
    ):
        raise ValueError("Training artifact is missing or outside evidence/training")
    if _file_digest(path) != artifact["sha256"]:
        raise ValueError("Training artifact hash changed")
    try:
        with np.load(path, allow_pickle=False) as archive:
            if len(archive.files) != len(set(archive.files)) or "score" not in archive.files:
                raise ValueError("Training artifact must contain one unambiguous score array")
            score = np.array(archive["score"], copy=True)
    except (OSError, TypeError, ValueError, EOFError, zipfile.BadZipFile) as error:
        raise ValueError("Training artifact is not a valid pickle-free NPZ") from error
    if _file_digest(path) != artifact["sha256"]:
        raise ValueError("Training artifact changed while it was read")
    return score


def _build_threshold_bundle(
    config: dict[str, Any],
    protocol_sha256: str,
    provenance: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Calibrate one threshold per regime from the exact complete training cohort."""
    validate_protocol(config)
    if not isinstance(protocol_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", protocol_sha256
    ):
        raise ValueError("Require the exact protocol SHA256")
    _provenance(provenance)
    if not isinstance(records, list):
        raise ValueError("Training records must be a list")

    plan = protocol_plan(config)
    expected_cases = training_cases(config)
    expected = {_case_id(case): case for case in expected_cases}
    found: dict[str, tuple[dict[str, Any], np.ndarray]] = {}
    artifact_paths: set[str] = set()
    failures = 0
    for record in records:
        if not isinstance(record, dict) or set(record) != {
            "case", "status", "artifact", "score", "error"
        }:
            raise ValueError("Invalid training record schema")
        if not isinstance(record["case"], dict):
            raise ValueError("Invalid training case schema")
        try:
            case_id = _case_id(record["case"])
        except (TypeError, ValueError) as error:
            raise ValueError("Invalid training case schema") from error
        if (
            case_id not in expected
            or record["case"] != expected[case_id]
            or case_id in found
        ):
            raise ValueError("Foreign or duplicate training identity")
        artifact = _artifact(record["artifact"])
        if artifact["path"] in artifact_paths:
            raise ValueError("Training records cannot share a raw artifact")
        artifact_paths.add(artifact["path"])
        if record["status"] == "failed":
            if (
                not isinstance(record["error"], str)
                or not record["error"]
                or record["score"] is not None
            ):
                raise ValueError("Failed training record requires an error and no score")
            failures += 1
            found[case_id] = (artifact, np.empty(0, dtype=np.float64))
        elif record["status"] == "completed":
            if record["error"] is not None:
                raise ValueError("Completed training record cannot contain an error")
            found[case_id] = (artifact, _score(config, record["score"]))
        else:
            raise ValueError("Unknown training record status")
    if len(found) != len(expected):
        raise ValueError("Threshold freeze requires the complete declared training cohort")
    if failures:
        raise ValueError("Threshold freeze requires every training record to complete")

    record_rows = []
    statistics: dict[str, list[np.ndarray]] = {row["id"]: [] for row in plan["regimes"]}
    record_ids: dict[str, list[str]] = {key: [] for key in statistics}
    detector = config["detector"]
    for case in expected_cases:
        case_id = _case_id(case)
        artifact, score = found[case_id]
        _, statistic = causal_mean(
            score, config["loop"]["burn_in"], detector["window_frames"]
        )
        statistics[case["regime_id"]].append(statistic)
        record_ids[case["regime_id"]].append(case_id)
        record_rows.append(
            {
                "id": case_id,
                "case": case,
                "artifact": artifact,
                "score_sha256": _array_digest(score),
            }
        )

    thresholds = []
    for regime in plan["regimes"]:
        values = np.concatenate(statistics[regime["id"]])
        threshold = float(
            np.quantile(
                values,
                detector["training_quantile"],
                method=detector["quantile_method"],
            )
        )
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError("Calibration produced an invalid threshold")
        thresholds.append(
            {
                "regime": regime,
                "threshold": threshold,
                "calibration_records": len(statistics[regime["id"]]),
                "calibration_windows": len(values),
                "training_record_ids": record_ids[regime["id"]],
                "statistic_sha256": _array_digest(values),
            }
        )

    detached_provenance = json.loads(_bytes(provenance))
    return {
        "schema_version": 1,
        "kind": "optical_gain_threshold_freeze",
        "status": "frozen_before_validation",
        "study_id": config["study_id"],
        "bindings": {
            "protocol_sha256": protocol_sha256,
            "base_config_sha256": config["base_config_sha256"],
            **detached_provenance,
        },
        "detector": json.loads(_bytes(detector)),
        "calculation_dtype": "float64",
        "training_records": record_rows,
        "training_manifest_sha256": _digest(_bytes(record_rows)),
        "thresholds": thresholds,
        "validation_released": False,
    }


def _publish_threshold_bundle(root: Path, bundle: dict[str, Any]) -> dict[str, str]:
    """Publish ``thresholds.json`` exclusively; identical retries are idempotent."""
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("Threshold root must be an existing non-symlink directory")
    path = root / "thresholds.json"
    if path.is_symlink():
        raise ValueError("Threshold publication cannot replace a symlink")
    _publish(path, bundle)
    data = path.read_bytes()
    return {"path": path.name, "sha256": hashlib.sha256(data).hexdigest()}


def freeze_thresholds(
    root: Path,
    protocol_path: Path,
    provenance: dict[str, Any],
    records: list[dict[str, Any]],
    repository: Path | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Verify raw training evidence, then publish the run's only threshold bundle."""
    requested_root = Path(root)
    if not requested_root.is_dir() or requested_root.is_symlink():
        raise ValueError("Threshold root must be an existing non-symlink directory")
    root = requested_root.resolve()
    repository = Path(repository or ROOT).resolve()
    path = Path(protocol_path)
    path = path if path.is_absolute() else repository / path
    if not path.is_file() or not path.resolve().is_relative_to(repository):
        raise ValueError("Protocol must be a regular file inside the source repository")
    for current in (path, *path.parents):
        if current == repository:
            break
        if current.is_symlink():
            raise ValueError("Protocol path cannot contain a symlink")
    protocol_sha256 = _file_digest(path)
    config = load_protocol(path, repository)
    if _file_digest(path) != protocol_sha256:
        raise ValueError("Protocol changed while it was loaded")
    if not isinstance(records, list):
        raise ValueError("Training records must be a list")
    verified = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Invalid training record schema")
        row = dict(record)
        if row.get("status") == "completed":
            artifact_score = _score(config, _artifact_score(root, row.get("artifact")))
            supplied_score = _score(config, row.get("score"))
            if not np.array_equal(artifact_score, supplied_score):
                raise ValueError("Training record score differs from its raw artifact")
            row["score"] = artifact_score
        verified.append(row)
    bundle = _build_threshold_bundle(config, protocol_sha256, provenance, verified)
    if _file_digest(path) != protocol_sha256:
        raise ValueError("Protocol changed before threshold publication")
    return bundle, _publish_threshold_bundle(root, bundle)
