from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from noether_optics.observability_detector import (
    _case_id,
    detector_trace,
    freeze_thresholds,
    training_cases,
)
from noether_optics.observability_protocol import load_protocol
from noether_optics.observability_validation import (
    _regime_decision,
    _target_metrics,
    freeze_validation_release,
    validation_cases,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/optical_gain_observability_v1.yaml"
PROVENANCE = {
    "source_commit": "a" * 40,
    "source_sha256": "b" * 64,
    "runtime_sha256": "c" * 64,
}


@pytest.fixture
def protocol() -> dict:
    return load_protocol(CONFIG)


def _save(
    root: Path, relative: Path, arrays: dict[str, np.ndarray]
) -> dict[str, str]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return {
        "path": relative.as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _prepare(root: Path, protocol: dict) -> tuple[list[dict], dict[str, float]]:
    root.mkdir()
    training = []
    for index, case in enumerate(training_cases(protocol)):
        regime = int(case["regime_id"].split("-")[-1])
        score = np.full(protocol["loop"]["frames"], 10.0 + regime, dtype=np.float64)
        relative = Path(f"evidence/training/trial-{index:03d}.npz")
        reference = _save(root, relative, {"score": score})
        training.append(
            {
                "case": case,
                "status": "completed",
                "artifact": reference,
                "score": score,
                "error": None,
            }
        )
    thresholds, _ = freeze_thresholds(root, CONFIG, PROVENANCE, training)
    by_regime = {
        row["regime"]["id"]: row["threshold"] for row in thresholds["thresholds"]
    }

    records = []
    frames = protocol["loop"]["frames"]
    onset = protocol["loop"]["fault_onset"]
    for index, case in enumerate(validation_cases(protocol)):
        threshold = by_regime[case["regime_id"]]
        score = np.full(frames, threshold + 1.0, dtype=np.float64)
        if case["condition"] == "gain_035":
            score[onset:] = threshold - 1.0
        trace = detector_trace(protocol, score, threshold)
        arrays = {
            "score": score,
            **trace,
            "variance": np.ones(frames, dtype=np.float64),
            "strehl": np.full(frames, 0.5, dtype=np.float64),
            "saturated": np.zeros(frames, dtype=bool),
            "fallback": np.zeros(frames, dtype=bool),
            "unavailable": np.zeros(frames, dtype=bool),
            "command": np.zeros((frames, 2), dtype=np.float64),
            "applied": np.zeros((frames, 2), dtype=np.float64),
        }
        relative = Path(f"evidence/validation/trial-{index:03d}.npz")
        records.append(
            {
                "case": case,
                "status": "completed",
                "artifact": _save(root, relative, arrays),
                "error": None,
            }
        )
    return records, by_regime


def test_validation_cases_are_complete_paired_and_disjoint(protocol) -> None:
    cases = validation_cases(protocol)
    assert len(cases) == 240
    assert {case["role"] for case in cases} == {"validation"}
    assert {case["condition"] for case in cases} == {"clean", "gain_035"}
    assert {case["atmosphere_seed"] for case in cases} == set(
        protocol["seeds"]["validation"]
    )
    assert not {case["atmosphere_seed"] for case in cases} & set(
        protocol["seeds"]["training"] + protocol["seeds"]["evaluation"]
    )
    pairs = {
        (case["regime_id"], case["atmosphere_seed"]): set()
        for case in cases
    }
    for case in cases:
        pairs[(case["regime_id"], case["atmosphere_seed"])].add(case["condition"])
    assert set(map(frozenset, pairs.values())) == {frozenset({"clean", "gain_035"})}


def _trace_with_runs(*runs: tuple[int, int]) -> dict[str, np.ndarray]:
    frames = np.arange(131, 512, dtype=np.int64)
    crossing = np.zeros(len(frames), dtype=bool)
    for start, end in runs:
        crossing[(frames >= start) & (frames <= end)] = True
    return {"statistic_frame_id": frames, "raw_crossing": crossing}


def test_target_accounting_excludes_pre_onset_and_boundary_streaks(protocol) -> None:
    boundary = _target_metrics(protocol, _trace_with_runs((249, 270)))
    assert boundary == {
        "pre_onset_alarm": False,
        "boundary_straddling_alarm": True,
        "detected": False,
        "detection_delay_frames": None,
    }
    detected = _target_metrics(protocol, _trace_with_runs((256, 280)))
    assert detected["detected"] is True
    assert detected["detection_delay_frames"] == 7
    prior = _target_metrics(protocol, _trace_with_runs((248, 260), (270, 290)))
    assert prior["pre_onset_alarm"] is True
    assert prior["detected"] is False


def _decision_rows(protocol: dict) -> list[dict]:
    rows = []
    for case in validation_cases(protocol):
        if case["regime_id"] != "regime-00":
            continue
        metrics = {
            "all_finite": True,
            "alarm_run": False,
            "raw_crossings": 0,
            "raw_crossing_frames": 381,
            "alarm_frames": 0,
            "alarm_eligible_frames": 374,
            "fallback_frames": 0,
            "unavailable_frames": 0,
            "saturated_frames": 0,
            "saturation_fraction": 0.0,
            "pre_onset_alarm": False,
            "boundary_straddling_alarm": False,
            "detected": case["condition"] == "gain_035",
            "detection_delay_frames": 7 if case["condition"] == "gain_035" else None,
        }
        rows.append(
            {
                "id": _case_id(case),
                "case": case,
                "status": "completed",
                "artifact": {"path": "synthetic", "sha256": "0" * 64},
                "metrics": metrics,
                "error": None,
            }
        )
    return rows


@pytest.mark.parametrize(
    ("damage", "criterion"),
    [
        ("clean_alarm", "clean_alarm_runs"),
        ("raw_crossings", "clean_raw_crossing_fraction"),
        ("alarm_frames", "clean_alarm_frame_fraction"),
        ("pre_onset", "gain_pre_onset_alarm_runs"),
        ("misses", "gain_detected_runs"),
        ("delay", "maximum_detection_delay_frames"),
        ("nonfinite", "all_finite_outputs"),
        ("failed", "failed_runs"),
        ("fallback", "fallback_frames"),
        ("unavailable", "unavailable_frames"),
        ("saturation", "saturation_fraction"),
    ],
)
def test_each_validation_gate_can_independently_withhold(
    protocol, damage, criterion
) -> None:
    rows = _decision_rows(protocol)
    clean = next(row for row in rows if row["case"]["condition"] == "clean")
    gain = next(row for row in rows if row["case"]["condition"] == "gain_035")
    if damage == "clean_alarm":
        clean["metrics"]["alarm_run"] = True
    elif damage == "raw_crossings":
        clean["metrics"]["raw_crossings"] = 115
    elif damage == "alarm_frames":
        clean["metrics"]["alarm_frames"] = 12
    elif damage == "pre_onset":
        gain["metrics"]["pre_onset_alarm"] = True
    elif damage == "misses":
        for row in [row for row in rows if row["case"]["condition"] == "gain_035"][:2]:
            row["metrics"]["detected"] = False
            row["metrics"]["detection_delay_frames"] = None
    elif damage == "delay":
        gain["metrics"]["detection_delay_frames"] = 33
    elif damage == "nonfinite":
        clean["metrics"]["all_finite"] = False
    elif damage == "failed":
        clean.update(status="failed", artifact=None, metrics=None, error="synthetic failure")
    elif damage == "fallback":
        clean["metrics"]["fallback_frames"] = 1
    elif damage == "unavailable":
        clean["metrics"]["unavailable_frames"] = 1
    else:
        clean["metrics"]["saturated_frames"] = 1
    regime = protocol["regimes"]
    decision = _regime_decision(
        protocol,
        {"id": "regime-00", "photons": regime["photons"][0], "wind_scale": regime["wind_scales"][0]},
        rows,
    )
    assert decision["passed"] is False
    assert decision["criteria"][criterion]["passed"] is False
    if damage == "misses":
        assert decision["criteria"]["gain_missed_runs"]["passed"] is False


def test_all_validation_gates_pass_for_declared_clean_and_target_rows(protocol) -> None:
    regime = protocol["regimes"]
    decision = _regime_decision(
        protocol,
        {"id": "regime-00", "photons": regime["photons"][0], "wind_scale": regime["wind_scales"][0]},
        _decision_rows(protocol),
    )
    assert decision["passed"] is True
    assert all(row["passed"] for row in decision["criteria"].values())
    assert decision["gain_035"]["detected_runs"] == 30
    assert decision["gain_035"]["detected_only_delay_frames"] == [7] * 30


def test_complete_validation_publishes_one_idempotent_release(protocol, tmp_path) -> None:
    root = tmp_path / "run"
    records, _ = _prepare(root, protocol)
    bundle, reference = freeze_validation_release(root, CONFIG, PROVENANCE, records)
    original = (root / "validation_release.json").read_bytes()
    assert bundle["evaluation_released"] is True
    assert bundle["status"] == "evaluation_released"
    assert len(bundle["validation_records"]) == 240
    assert all(row["passed"] for row in bundle["regimes"])
    assert reference["sha256"] == hashlib.sha256(original).hexdigest()
    assert json.loads(original) == bundle
    assert freeze_validation_release(
        root, CONFIG, PROVENANCE, list(reversed(records))
    ) == (
        bundle,
        reference,
    )

    changed = deepcopy(records)
    changed[0].update(status="failed", artifact=None, error="synthetic terminal failure")
    with pytest.raises(ValueError, match="replace existing evidence"):
        freeze_validation_release(root, CONFIG, PROVENANCE, changed)
    assert (root / "validation_release.json").read_bytes() == original


def test_terminal_validation_failure_is_retained_and_withholds_evaluation(
    protocol, tmp_path
) -> None:
    root = tmp_path / "run"
    records, _ = _prepare(root, protocol)
    completed = deepcopy(records[0])
    records[0].update(status="failed", artifact=None, error="synthetic numerical failure")
    bundle, _ = freeze_validation_release(root, CONFIG, PROVENANCE, records)
    original = (root / "validation_release.json").read_bytes()
    assert bundle["evaluation_released"] is False
    assert bundle["status"] == "evaluation_withheld"
    regime = bundle["regimes"][0]
    assert regime["failed_runs"] == 1
    assert regime["criteria"]["failed_runs"]["passed"] is False
    assert bundle["validation_records"][0]["error"] == "synthetic numerical failure"
    records[0] = completed
    with pytest.raises(ValueError, match="replace existing evidence"):
        freeze_validation_release(root, CONFIG, PROVENANCE, records)
    assert (root / "validation_release.json").read_bytes() == original


def test_release_rejects_incomplete_duplicate_or_foreign_identities(
    protocol, tmp_path
) -> None:
    root = tmp_path / "run"
    records, _ = _prepare(root, protocol)
    incomplete = records[:-1]
    duplicate = deepcopy(records)
    duplicate[1] = deepcopy(duplicate[0])
    foreign = deepcopy(records)
    foreign[0]["case"]["atmosphere_seed"] += 1_000_000
    for damaged in (incomplete, duplicate, foreign):
        with pytest.raises(ValueError):
            freeze_validation_release(root, CONFIG, PROVENANCE, damaged)
    assert not (root / "validation_release.json").exists()


@pytest.mark.parametrize(
    "damage", ["validation_hash", "trace", "training_hash", "provenance"]
)
def test_release_rechecks_all_frozen_evidence(protocol, tmp_path, damage) -> None:
    root = tmp_path / "run"
    records, _ = _prepare(root, protocol)
    if damage == "training_hash":
        path = root / "evidence/training/trial-000.npz"
        path.write_bytes(path.read_bytes() + b"drift")
    elif damage != "provenance":
        path = root / records[0]["artifact"]["path"]
        if damage == "validation_hash":
            path.write_bytes(path.read_bytes() + b"drift")
        else:
            with np.load(path, allow_pickle=False) as archive:
                arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
            arrays["p2_alarm"][0] = True
            np.savez(path, **arrays)
            records[0]["artifact"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    provenance = (
        dict(PROVENANCE, runtime_sha256="d" * 64)
        if damage == "provenance"
        else PROVENANCE
    )
    with pytest.raises(ValueError):
        freeze_validation_release(root, CONFIG, provenance, records)
    assert not (root / "validation_release.json").exists()
