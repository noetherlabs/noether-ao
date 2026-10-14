from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from scipy.stats import beta as beta_distribution
from scipy.stats import t as student_t

from noether_optics.controller_evidence import save_arrays, save_metadata
from noether_optics.controller_selection import _bytes
from noether_optics.observability_detector import _case_id, detector_trace
from noether_optics.observability_evaluation import (
    clopper_pearson,
    evaluation_cases,
    paired_student_interval,
)
from noether_optics.observability_protocol import load_protocol
from noether_optics.observability_run import ObservabilityLedger
from noether_optics.observability_trials import retain_full_wfs, trial_metadata


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/optical_gain_observability_runner_smoke_v1.yaml"
V1_CONFIG = ROOT / "configs/optical_gain_observability_v1.yaml"
PROVENANCE = {
    "source_commit": "a" * 40,
    "source_sha256": "b" * 64,
    "runtime_sha256": "c" * 64,
}


def _arrays(
    ledger: ObservabilityLedger,
    case: dict,
    score: np.ndarray,
    *,
    threshold: float | None = None,
) -> dict[str, np.ndarray]:
    frames, modes = ledger.config["loop"]["frames"], ledger.base["modes"]
    variance_by_condition = {
        "clean": 1.0,
        "gain_035": 4.0,
        "gain_065": 6.25,
        "vibration": 9.0,
        "misregistration": 12.25,
    }
    strehl_by_condition = {
        "clean": 0.8,
        "gain_035": 0.6,
        "gain_065": 0.5,
        "vibration": 0.4,
        "misregistration": 0.3,
    }
    arrays: dict[str, np.ndarray] = {
        "score": np.asarray(score, dtype=np.float64),
        "variance": np.full(
            frames, variance_by_condition[case["condition"]], dtype=np.float64
        ),
        "strehl": np.full(
            frames, strehl_by_condition[case["condition"]], dtype=np.float64
        ),
        "saturated": np.zeros(frames, dtype=bool),
        "fallback": np.zeros(frames, dtype=bool),
        "unavailable": np.zeros(frames, dtype=bool),
        "command": np.zeros((frames, modes), dtype=np.float64),
        "applied": np.zeros((frames, modes), dtype=np.float64),
    }
    if threshold is not None:
        arrays = {"score": arrays["score"], **detector_trace(
            ledger.config, arrays["score"], threshold
        ), **{key: value for key, value in arrays.items() if key != "score"}}
    if retain_full_wfs(ledger.config, case):
        shape = (frames, ledger.base["detector_n"], ledger.base["detector_n"])
        arrays.update(
            measurement=np.zeros(shape, dtype=np.float64),
            prediction=np.zeros(shape, dtype=np.float64),
        )
    return arrays


def _record(
    ledger: ObservabilityLedger,
    case: dict,
    score: np.ndarray,
    *,
    threshold: float | None = None,
) -> dict:
    arrays = _arrays(ledger, case, score, threshold=threshold)
    case_id = _case_id(case)
    role = case["role"]
    artifact = save_arrays(
        ledger, f"evidence/{role}/trials/{case_id}.npz", arrays
    )
    metadata = save_metadata(
        ledger,
        f"evidence/{role}/trials/{case_id}.json",
        trial_metadata(ledger.config, ledger.base, case, artifact, arrays),
    )
    return {
        "case": case,
        "status": "completed",
        "artifact": artifact,
        "metadata": metadata,
        "error": None,
    }


def _training_backend(ledger: ObservabilityLedger):
    def execute(case: dict) -> dict:
        score = np.full(ledger.config["loop"]["frames"], 10.0, dtype=np.float64)
        return _record(ledger, case, score)

    return execute


def _post_threshold_backend(
    ledger: ObservabilityLedger,
    *,
    failed: set[str] | None = None,
    clean_alarm: bool = False,
):
    failed = failed or set()

    def execute(case: dict, threshold: float) -> dict:
        if _case_id(case) in failed:
            return {
                "case": case,
                "status": "failed",
                "artifact": None,
                "metadata": None,
                "error": f"synthetic terminal {case['role']} failure",
            }
        frames = ledger.config["loop"]["frames"]
        onset = ledger.config["loop"]["fault_onset"]
        score = np.full(frames, threshold + 0.01, dtype=np.float64)
        if case["condition"] in ledger.config["conditions"]["target_conditions"]:
            score[onset:] = 0.0
        elif clean_alarm and case["condition"] == "clean":
            score[:] = 0.0
        return _record(ledger, case, score, threshold=threshold)

    return execute


def _prepared_ledger(tmp_path: Path, *, released: bool = True) -> ObservabilityLedger:
    ledger = ObservabilityLedger.create(tmp_path / "run", CONFIG, PROVENANCE)
    assert ledger.train_chunk(_training_backend(ledger), 2)["complete"] is True
    ledger.freeze_training()
    assert ledger.validate_chunk(
        _post_threshold_backend(ledger, clean_alarm=not released), 4
    )["complete"] is True
    decision, _ = ledger.freeze_validation()
    assert decision["evaluation_released"] is released
    return ledger


def test_v1_evaluation_cases_are_complete_paired_and_disjoint() -> None:
    protocol = load_protocol(V1_CONFIG)
    cases = evaluation_cases(protocol)
    assert len(cases) == 1_200
    assert {case["role"] for case in cases} == {"evaluation"}
    assert {case["condition"] for case in cases} == set(
        protocol["conditions"]["evaluation"]
    )
    assert {case["atmosphere_seed"] for case in cases} == set(
        protocol["seeds"]["evaluation"]
    )
    assert not {case["atmosphere_seed"] for case in cases} & set(
        protocol["seeds"]["training"] + protocol["seeds"]["validation"]
    )
    pairs: dict[tuple[str, int], set[str]] = {}
    for case in cases:
        key = (case["regime_id"], case["atmosphere_seed"])
        pairs.setdefault(key, set()).add(case["condition"])
    assert len(pairs) == 240
    assert set(map(frozenset, pairs.values())) == {
        frozenset(protocol["conditions"]["evaluation"])
    }


def test_missing_or_withheld_release_never_opens_evaluation(
    tmp_path: Path,
) -> None:
    missing = ObservabilityLedger.create(tmp_path / "missing", CONFIG, PROVENANCE)
    backend = Mock()
    with pytest.raises(ValueError, match="frozen validation release"):
        missing.evaluate_chunk(backend, 1)
    backend.assert_not_called()
    assert not (missing.root / "evaluation").exists()

    withheld = _prepared_ledger(tmp_path / "withheld", released=False)
    backend = Mock()
    with pytest.raises(ValueError, match="withheld evaluation"):
        withheld.evaluate_chunk(backend, 1)
    backend.assert_not_called()
    assert not (withheld.root / "evaluation").exists()


def test_evaluation_resume_retries_interruption_and_never_replays_terminal_rows(
    tmp_path: Path,
) -> None:
    ledger = _prepared_ledger(tmp_path)
    backend = _post_threshold_backend(ledger)
    calls: list[dict] = []

    def interrupt(case: dict, threshold: float) -> dict:
        calls.append(case)
        record = backend(case, threshold)
        if len(calls) == 2:
            raise RuntimeError("synthetic evaluation interruption")
        return record

    with pytest.raises(RuntimeError, match="evaluation interruption"):
        ledger.evaluate_chunk(interrupt, 3)
    assert len(list((ledger.root / "evaluation").glob("*.json"))) == 1

    resumed: list[dict] = []

    def resume(case: dict, threshold: float) -> dict:
        resumed.append(case)
        return backend(case, threshold)

    progress = ObservabilityLedger(ledger.root, CONFIG, PROVENANCE).evaluate_chunk(
        resume, 2
    )
    assert progress["total_records"] == 3
    assert resumed[0] == calls[1]
    assert all(case != calls[0] for case in resumed)

    fourth = evaluation_cases(ledger.config)[3]
    failed = _post_threshold_backend(ledger, failed={_case_id(fourth)})
    assert ledger.evaluate_chunk(failed, 1)["total_records"] == 4
    later: list[dict] = []

    def after_failure(case: dict, threshold: float) -> dict:
        later.append(case)
        return backend(case, threshold)

    ledger.evaluate_chunk(after_failure, 1)
    assert later[0] != fourth


def test_retained_evaluation_numerical_failure_keeps_raw_arrays(
    tmp_path: Path,
) -> None:
    ledger = _prepared_ledger(tmp_path)
    error = "Nonfinite physical trial output"

    def failed(case: dict, threshold: float) -> dict:
        del threshold
        score = np.full(ledger.config["loop"]["frames"], np.nan, dtype=np.float64)
        arrays = _arrays(ledger, case, score)
        case_id = _case_id(case)
        artifact = save_arrays(
            ledger, f"evidence/evaluation/trials/{case_id}.npz", arrays
        )
        metadata = save_metadata(
            ledger,
            f"evidence/evaluation/trials/{case_id}.json",
            trial_metadata(
                ledger.config,
                ledger.base,
                case,
                artifact,
                arrays,
                status="failed",
                error=error,
            ),
        )
        return {
            "case": case,
            "status": "failed",
            "artifact": artifact,
            "metadata": metadata,
            "error": error,
        }

    assert ledger.evaluate_chunk(failed, 1)["total_records"] == 1
    saved = json.loads(next((ledger.root / "evaluation").glob("*.json")).read_bytes())[
        "record"
    ]
    assert saved["status"] == "failed"
    assert saved["artifact"] is not None and saved["metadata"] is not None

    resumed: list[dict] = []
    backend = _post_threshold_backend(ledger)

    def next_case(case: dict, threshold: float) -> dict:
        resumed.append(case)
        return backend(case, threshold)

    ledger.evaluate_chunk(next_case, 1)
    assert resumed[0] != saved["case"]


def test_upstream_drift_during_callback_prevents_terminal_publication(
    tmp_path: Path,
) -> None:
    ledger = _prepared_ledger(tmp_path)
    backend = _post_threshold_backend(ledger)

    def drift(case: dict, threshold: float) -> dict:
        record = backend(case, threshold)
        path = ledger.root / "validation_release.json"
        path.write_bytes(path.read_bytes() + b"drift")
        return record

    with pytest.raises(ValueError, match="upstream evidence changed"):
        ledger.evaluate_chunk(drift, 1)
    assert not (ledger.root / "evaluation").exists()


def test_evaluation_record_directory_symlink_is_rejected_before_callback(
    tmp_path: Path,
) -> None:
    ledger = _prepared_ledger(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (ledger.root / "evaluation").symlink_to(outside, target_is_directory=True)
    backend = Mock()
    with pytest.raises(ValueError, match="regular directory"):
        ledger.evaluate_chunk(backend, 1)
    backend.assert_not_called()
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize(
    "damage", ["binding", "artifact", "metadata", "threshold", "release"]
)
def test_evaluation_resume_rejects_changed_bindings_or_evidence(
    tmp_path: Path, damage: str
) -> None:
    ledger = _prepared_ledger(tmp_path)
    ledger.evaluate_chunk(_post_threshold_backend(ledger), 1)
    record_path = next((ledger.root / "evaluation").glob("*.json"))
    saved = json.loads(record_path.read_bytes())
    if damage == "binding":
        saved["validation_release_sha256"] = "0" * 64
        record_path.write_bytes(_bytes(saved))
    elif damage in {"artifact", "metadata"}:
        path = ledger.root / saved["record"][damage]["path"]
        path.write_bytes(path.read_bytes() + b"drift")
    else:
        path = ledger.root / (
            "thresholds.json" if damage == "threshold" else "validation_release.json"
        )
        path.write_bytes(path.read_bytes() + b"drift")
    backend = Mock()
    with pytest.raises(ValueError):
        ledger.evaluate_chunk(backend, 1)
    backend.assert_not_called()


def test_exact_intervals_use_frozen_methods() -> None:
    zero = clopper_pearson(0, 60, 0.95)
    all_events = clopper_pearson(60, 60, 0.95)
    assert zero["lower"] == 0.0
    assert zero["upper"] == pytest.approx(
        beta_distribution.ppf(0.975, 1, 60)
    )
    assert all_events["lower"] == pytest.approx(
        beta_distribution.ppf(0.025, 60, 1)
    )
    assert all_events["upper"] == 1.0

    values = [1.0, 2.0, 4.0]
    interval = paired_student_interval(values, 0.05, 16)
    critical = student_t.ppf(1.0 - (0.05 / 16) / 2.0, 2)
    half_width = critical * np.std(values, ddof=1) / np.sqrt(3)
    assert interval["confidence"] == pytest.approx(0.996875)
    assert interval["lower"] == pytest.approx(np.mean(values) - half_width)
    assert interval["upper"] == pytest.approx(np.mean(values) + half_width)


def test_complete_report_is_descriptive_exact_and_idempotent(tmp_path: Path) -> None:
    ledger = _prepared_ledger(tmp_path)
    assert ledger.evaluate_chunk(_post_threshold_backend(ledger), 10) == {
        "new_records": 10,
        "total_records": 10,
        "planned_records": 10,
        "complete": True,
    }
    report, reference = ledger.freeze_evaluation_report()
    original = (ledger.root / "evaluation_report.json").read_bytes()
    assert report["status"] == "complete_descriptive"
    assert report["record_counts"] == {"planned": 10, "completed": 10, "failed": 0}
    assert report["bindings"]["manifest_sha256"] == ledger.manifest_sha256
    assert report["evaluation_released"] is True
    assert report["second_pass_gate"] is False
    assert report["inference"]["paired_error_family_size"] == 4
    assert report["inference"]["paired_error_confidence"] == pytest.approx(0.9875)
    assert reference["sha256"] == hashlib.sha256(original).hexdigest()

    conditions = report["regimes"][0]["conditions"]
    clean = conditions["clean"]
    assert set(ledger.config["reporting"]["clean"]).issubset(clean)
    assert clean["alarm_runs"] == 0
    assert clean["alarm_run_fraction"]["events"] == 0
    assert clean["alarm_run_fraction"]["trials"] == 2
    assert clean["alarm_run_fraction"]["upper"] == pytest.approx(
        beta_distribution.ppf(0.975, 1, 2)
    )
    assert clean["raw_crossing_fraction"]["interval"] is None
    assert clean["alarm_frames"] == 0
    assert clean["alarm_frame_fraction"]["interval"] is None
    assert set(ledger.config["reporting"]["integrity"]).issubset(
        clean["integrity"]
    )
    assert clean["integrity"] == {
        "finite_runs": 2,
        "failed_runs": 0,
        "fallback_frames": 0,
        "unavailable_frames": 0,
        "saturated_frames": 0,
        "observed_frames": 48,
        "saturation_fraction": 0.0,
    }
    target = conditions["gain_035"]
    assert set(ledger.config["reporting"]["targets"]).issubset(target)
    assert target["detected_runs_without_prior_alarm"]["events"] == 2
    assert target["detected_runs_without_prior_alarm"]["trials"] == 2
    assert target["missed_runs"]["events"] == 0

    gain_cell = next(
        row
        for row in report["regimes"][0]["paired_controller_error"]
        if row["condition"] == "gain_035"
    )
    assert set(ledger.config["reporting"]["controller_error"]).issubset(gain_cell)
    scale = ledger.base["wavelength_m"] / (2 * np.pi) * 1e9
    assert gain_cell["interval_frames"] == [16, 24]
    assert gain_cell["complete_pairs"] == 2
    assert gain_cell["paired_wfe_nm_delta"]["values"] == pytest.approx(
        [scale, scale]
    )
    assert gain_cell["paired_wfe_nm_delta"]["confidence"] == pytest.approx(0.9875)

    assert ledger.freeze_evaluation_report()[0] == report
    assert (ledger.root / "evaluation_report.json").read_bytes() == original
    backend = Mock()
    with pytest.raises(ValueError, match="report is frozen"):
        ledger.evaluate_chunk(backend, 1)
    backend.assert_not_called()


def test_report_requires_complete_cohort(tmp_path: Path) -> None:
    ledger = _prepared_ledger(tmp_path)
    ledger.evaluate_chunk(_post_threshold_backend(ledger), 1)
    with pytest.raises(ValueError, match="all evaluation records"):
        ledger.freeze_evaluation_report()
    assert not (ledger.root / "evaluation_report.json").exists()


def test_report_keeps_failed_denominators_and_incomplete_pairs(tmp_path: Path) -> None:
    ledger = _prepared_ledger(tmp_path)
    cases = evaluation_cases(ledger.config)
    failed_cases = {
        _case_id(next(case for case in cases if case["condition"] == "gain_035")),
        _case_id(next(case for case in cases if case["condition"] == "vibration")),
    }
    ledger.evaluate_chunk(_post_threshold_backend(ledger, failed=failed_cases), 10)
    report, _ = ledger.freeze_evaluation_report()
    assert report["record_counts"] == {"planned": 10, "completed": 8, "failed": 2}
    assert len(report["failed_records"]) == 2

    conditions = report["regimes"][0]["conditions"]
    target = conditions["gain_035"]
    assert target["detected_runs_without_prior_alarm"]["events"] == 1
    assert target["detected_runs_without_prior_alarm"]["trials"] == 2
    assert target["missed_runs"]["events"] == 1
    nuisance = conditions["vibration"]
    assert set(ledger.config["reporting"]["nuisances"]).issubset(nuisance)
    assert nuisance["failed_runs"] == 1
    assert nuisance["quiet_runs"]["events"] == 1
    assert nuisance["quiet_runs"]["trials"] == 1
    assert "unclassified" in nuisance["denominator_note"]

    cells = {
        row["condition"]: row
        for row in report["regimes"][0]["paired_controller_error"]
    }
    assert cells["gain_035"]["complete_pairs"] == 1
    assert len(cells["gain_035"]["incomplete_pairs"]) == 1
    assert cells["vibration"]["complete_pairs"] == 1
    assert len(cells["vibration"]["incomplete_pairs"]) == 1
