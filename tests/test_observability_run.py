from __future__ import annotations

import fcntl
import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from noether_optics.controller_evidence import save_arrays, save_metadata
from noether_optics.observability_detector import _case_id, detector_trace
from noether_optics.observability_run import ObservabilityLedger
from noether_optics.observability_trials import retain_full_wfs, trial_metadata
from noether_optics.observability_validation import validation_cases


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/optical_gain_observability_v1.yaml"
PROVENANCE = {
    "source_commit": "a" * 40,
    "source_sha256": "b" * 64,
    "runtime_sha256": "c" * 64,
}


@pytest.fixture
def ledger(tmp_path) -> ObservabilityLedger:
    return ObservabilityLedger.create(tmp_path / "run", CONFIG, PROVENANCE)


def _training_backend(ledger: ObservabilityLedger, failed: set[str] | None = None):
    failed = failed or set()

    def execute(case: dict) -> dict:
        case_id = _case_id(case)
        if case_id in failed:
            return {
                "case": case,
                "status": "failed",
                "artifact": None,
                "metadata": None,
                "error": "synthetic terminal training failure",
            }
        regime = int(case["regime_id"].split("-")[-1])
        score = np.full(ledger.config["loop"]["frames"], 10.0 + regime)
        frames, modes = ledger.config["loop"]["frames"], ledger.base["modes"]
        arrays = {
            "score": score.astype(np.float64),
            "variance": np.ones(frames, dtype=np.float64),
            "strehl": np.full(frames, 0.5, dtype=np.float64),
            "saturated": np.zeros(frames, dtype=bool),
            "fallback": np.zeros(frames, dtype=bool),
            "unavailable": np.zeros(frames, dtype=bool),
            "command": np.zeros((frames, modes), dtype=np.float64),
            "applied": np.zeros((frames, modes), dtype=np.float64),
        }
        if retain_full_wfs(ledger.config, case):
            shape = (frames, ledger.base["detector_n"], ledger.base["detector_n"])
            arrays.update(
                measurement=np.zeros(shape, dtype=np.float64),
                prediction=np.zeros(shape, dtype=np.float64),
            )
        reference = save_arrays(
            ledger,
            f"evidence/training/{case_id}.npz",
            arrays,
        )
        metadata = save_metadata(
            ledger,
            f"evidence/training/{case_id}.json",
            trial_metadata(ledger.config, ledger.base, case, reference, arrays),
        )
        return {
            "case": case,
            "status": "completed",
            "artifact": reference,
            "metadata": metadata,
            "error": None,
        }

    return execute


def _validation_backend(ledger: ObservabilityLedger, failed: set[str] | None = None):
    failed = failed or set()

    def execute(case: dict, threshold: float) -> dict:
        case_id = _case_id(case)
        if case_id in failed:
            return {
                "case": case,
                "status": "failed",
                "artifact": None,
                "metadata": None,
                "error": "synthetic terminal validation failure",
            }
        frames = ledger.config["loop"]["frames"]
        onset = ledger.config["loop"]["fault_onset"]
        score = np.full(frames, threshold + 1.0, dtype=np.float64)
        if case["condition"] == "gain_035":
            score[onset:] = threshold - 1.0
        trace = detector_trace(ledger.config, score, threshold)
        arrays = {
            "score": score,
            **trace,
            "variance": np.ones(frames, dtype=np.float64),
            "strehl": np.full(frames, 0.5, dtype=np.float64),
            "saturated": np.zeros(frames, dtype=bool),
            "fallback": np.zeros(frames, dtype=bool),
            "unavailable": np.zeros(frames, dtype=bool),
            "command": np.zeros((frames, ledger.base["modes"]), dtype=np.float64),
            "applied": np.zeros((frames, ledger.base["modes"]), dtype=np.float64),
        }
        if retain_full_wfs(ledger.config, case):
            shape = (frames, ledger.base["detector_n"], ledger.base["detector_n"])
            arrays.update(
                measurement=np.zeros(shape, dtype=np.float64),
                prediction=np.zeros(shape, dtype=np.float64),
            )
        reference = save_arrays(
            ledger,
            f"evidence/validation/{case_id}.npz",
            arrays,
        )
        metadata = save_metadata(
            ledger,
            f"evidence/validation/{case_id}.json",
            trial_metadata(ledger.config, ledger.base, case, reference, arrays),
        )
        return {
            "case": case,
            "status": "completed",
            "artifact": reference,
            "metadata": metadata,
            "error": None,
        }

    return execute


def _complete_training(ledger: ObservabilityLedger) -> None:
    progress = ledger.train_chunk(_training_backend(ledger), 64)
    assert progress == {
        "new_records": 64,
        "total_records": 64,
        "planned_records": 64,
        "complete": True,
    }
    bundle, _ = ledger.freeze_training()
    assert len(bundle["training_records"]) == 64


def test_manifest_binds_protocol_source_runtime_and_resume(ledger) -> None:
    resumed = ObservabilityLedger(ledger.root, CONFIG, PROVENANCE)
    assert resumed.manifest == ledger.manifest
    assert json.loads((ledger.root / "manifest.json").read_bytes()) == ledger.manifest
    changed = dict(PROVENANCE, runtime_sha256="d" * 64)
    with pytest.raises(ValueError, match="bindings changed"):
        ObservabilityLedger(ledger.root, CONFIG, changed)


def test_training_resume_retries_only_interruption_and_preserves_failure(ledger) -> None:
    backend = _training_backend(ledger)
    calls = []

    def interrupt(case):
        calls.append(case)
        record = backend(case)
        if len(calls) == 2:
            raise RuntimeError("synthetic interruption after raw publication")
        return record

    with pytest.raises(RuntimeError, match="synthetic interruption"):
        ledger.train_chunk(interrupt, 3)
    assert len(list((ledger.root / "training").glob("*.json"))) == 1

    interrupted_id = _case_id(calls[1])
    recovery = _training_backend(ledger, {interrupted_id})
    recovered = []

    def resume(case):
        recovered.append(case)
        return recovery(case)

    progress = ObservabilityLedger(ledger.root, CONFIG, PROVENANCE).train_chunk(resume, 63)
    assert progress["complete"] is True
    assert _case_id(recovered[0]) == interrupted_id
    assert len(recovered) == 63
    saved = json.loads((ledger.root / "training" / f"{interrupted_id}.json").read_text())
    assert saved["record"]["status"] == "failed"
    with pytest.raises(ValueError, match="failed training outcome"):
        ledger.freeze_training()
    assert not (ledger.root / "thresholds.json").exists()


def test_complete_training_freezes_once_and_blocks_late_records(ledger) -> None:
    _complete_training(ledger)
    original = (ledger.root / "thresholds.json").read_bytes()
    assert ledger.freeze_training()[0]["status"] == "frozen_before_validation"
    assert (ledger.root / "thresholds.json").read_bytes() == original
    backend = Mock()
    with pytest.raises(ValueError, match="Training is frozen"):
        ledger.train_chunk(backend, 1)
    backend.assert_not_called()


def test_validation_resume_retries_only_interrupted_case(ledger) -> None:
    _complete_training(ledger)
    backend = _validation_backend(ledger)
    calls = []

    def interrupt(case, threshold):
        calls.append(case)
        record = backend(case, threshold)
        if len(calls) == 2:
            raise RuntimeError("synthetic validation interruption")
        return record

    with pytest.raises(RuntimeError, match="validation interruption"):
        ledger.validate_chunk(interrupt, 3)
    assert len(list((ledger.root / "validation").glob("*.json"))) == 1
    recovered = []

    def resume(case, threshold):
        recovered.append(case)
        return backend(case, threshold)

    progress = ObservabilityLedger(ledger.root, CONFIG, PROVENANCE).validate_chunk(
        resume, 2
    )
    assert progress["total_records"] == 3
    assert recovered[0] == calls[1]
    assert recovered[1] != calls[0]


def test_complete_validation_releases_once_and_blocks_late_records(ledger) -> None:
    _complete_training(ledger)
    progress = ledger.validate_chunk(_validation_backend(ledger), 240)
    assert progress["complete"] is True
    bundle, _ = ledger.freeze_validation()
    assert bundle["evaluation_released"] is True
    original = (ledger.root / "validation_release.json").read_bytes()
    assert ledger.freeze_validation()[0] == bundle
    assert (ledger.root / "validation_release.json").read_bytes() == original
    backend = Mock()
    with pytest.raises(ValueError, match="release is frozen"):
        ledger.validate_chunk(backend, 1)
    backend.assert_not_called()


def test_terminal_validation_failure_is_skipped_and_withholds(ledger) -> None:
    _complete_training(ledger)
    first_case = validation_cases(ledger.config)[0]
    failed = {_case_id(first_case)}
    progress = ledger.validate_chunk(_validation_backend(ledger, failed), 240)
    assert progress["complete"] is True
    backend = Mock()
    assert ledger.validate_chunk(backend, 1)["new_records"] == 0
    backend.assert_not_called()
    bundle, _ = ledger.freeze_validation()
    assert bundle["evaluation_released"] is False
    assert bundle["validation_records"][0]["status"] == "failed"


@pytest.mark.parametrize("damage", ["binding", "artifact", "metadata"])
def test_changed_saved_training_evidence_blocks_resume(ledger, damage) -> None:
    ledger.train_chunk(_training_backend(ledger), 1)
    path = next((ledger.root / "training").glob("*.json"))
    saved = json.loads(path.read_text())
    if damage == "binding":
        saved["manifest_sha256"] = "0" * 64
        path.write_text(json.dumps(saved, sort_keys=True, indent=2) + "\n")
    elif damage == "artifact":
        artifact = ledger.root / saved["record"]["artifact"]["path"]
        artifact.write_bytes(artifact.read_bytes() + b"drift")
    else:
        metadata = ledger.root / saved["record"]["metadata"]["path"]
        metadata.write_bytes(metadata.read_bytes() + b"drift")
    backend = Mock()
    with pytest.raises(ValueError):
        ledger.train_chunk(backend, 1)
    backend.assert_not_called()


@pytest.mark.parametrize("damage", ["binding", "artifact", "metadata"])
def test_changed_saved_validation_evidence_blocks_resume(ledger, damage) -> None:
    _complete_training(ledger)
    ledger.validate_chunk(_validation_backend(ledger), 1)
    path = next((ledger.root / "validation").glob("*.json"))
    saved = json.loads(path.read_text())
    if damage == "binding":
        saved["threshold_sha256"] = "0" * 64
        path.write_text(json.dumps(saved, sort_keys=True, indent=2) + "\n")
    elif damage == "artifact":
        artifact = ledger.root / saved["record"]["artifact"]["path"]
        artifact.write_bytes(artifact.read_bytes() + b"drift")
    else:
        metadata = ledger.root / saved["record"]["metadata"]["path"]
        metadata.write_bytes(metadata.read_bytes() + b"drift")
    backend = Mock()
    with pytest.raises(ValueError):
        ledger.validate_chunk(backend, 1)
    backend.assert_not_called()


def test_retained_numerical_failure_is_terminal_and_keeps_raw_arrays(ledger) -> None:
    error = "Nonfinite physical trial output"

    def failed(case):
        frames, modes = ledger.config["loop"]["frames"], ledger.base["modes"]
        arrays = {
            "score": np.full(frames, np.nan, dtype=np.float64),
            "variance": np.ones(frames, dtype=np.float64),
            "strehl": np.full(frames, 0.5, dtype=np.float64),
            "saturated": np.zeros(frames, dtype=bool),
            "fallback": np.zeros(frames, dtype=bool),
            "unavailable": np.zeros(frames, dtype=bool),
            "command": np.zeros((frames, modes), dtype=np.float64),
            "applied": np.zeros((frames, modes), dtype=np.float64),
        }
        shape = (frames, ledger.base["detector_n"], ledger.base["detector_n"])
        arrays.update(
            measurement=np.zeros(shape, dtype=np.float64),
            prediction=np.zeros(shape, dtype=np.float64),
        )
        case_id = _case_id(case)
        raw = save_arrays(ledger, f"evidence/training/{case_id}.npz", arrays)
        metadata = save_metadata(
            ledger,
            f"evidence/training/{case_id}.json",
            trial_metadata(
                ledger.config,
                ledger.base,
                case,
                raw,
                arrays,
                status="failed",
                error=error,
            ),
        )
        return {
            "case": case,
            "status": "failed",
            "artifact": raw,
            "metadata": metadata,
            "error": error,
        }

    assert ledger.train_chunk(failed, 1)["total_records"] == 1
    saved = json.loads(next((ledger.root / "training").glob("*.json")).read_text())[
        "record"
    ]
    assert saved["status"] == "failed"
    assert saved["artifact"] is not None and saved["metadata"] is not None

    resumed = []
    backend = _training_backend(ledger)

    def next_case(case):
        resumed.append(case)
        return backend(case)

    ledger.train_chunk(next_case, 1)
    assert resumed[0]["atmosphere_seed"] == ledger.config["seeds"]["training"][1]


def test_record_directory_symlink_is_rejected_before_callback(ledger, tmp_path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (ledger.root / "training").symlink_to(outside, target_is_directory=True)
    backend = Mock()
    with pytest.raises(ValueError, match="regular directory"):
        ledger.train_chunk(backend, 1)
    backend.assert_not_called()
    assert list(outside.iterdir()) == []


def test_second_writer_and_invalid_limits_never_call_backend(ledger) -> None:
    backend = Mock()
    for limit in (0, True, -1):
        with pytest.raises(ValueError, match="positive integer"):
            ledger.train_chunk(backend, limit)
    with (ledger.root / ".lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            ledger.train_chunk(backend, 1)
    backend.assert_not_called()
