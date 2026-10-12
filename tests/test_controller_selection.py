"""Synthetic orchestration fixtures only: no atmosphere or study outcomes generated."""
from copy import deepcopy
import fcntl
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from noether_optics.controller_protocol import load_study, study_plan
from noether_optics.controller_selection import StudyLedger, select_candidates, validation_cases
import noether_optics.controller_selection as selection_module

ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = {"source_commit": "a" * 40, "source_sha256": "b" * 64, "runtime_sha256": "c" * 64}


@pytest.fixture
def config():
    value = load_study(ROOT / "configs/controller_study_v1.yaml")
    value["study_id"] = "synthetic_ledger_test_only"
    for role, base in (("training", 31001), ("validation", 32001), ("evaluation", 33001)):
        value["seeds"][role] = list(range(base, base + len(value["seeds"][role])))
    value["seeds"]["initialization"] = 34001
    value["candidates_per_controller"] = 2
    for controller, grid in value["candidate_grids"].items():
        grid["horizon" if controller == "predictive" else "gain"] = [1] if controller == "predictive" else [.1]
    return value


def outcome(case, artifact=None):
    reference = artifact or {"path": "evidence/synthetic.bin", "sha256": "0" * 64}
    return {"case": deepcopy(case), "status": "completed", "error": None,
            "metrics": {"all_outputs_finite": True, "wfe_nm": 50., "open_wfe_nm": 100.,
                        "saturation_fraction": .01, "strehl": .8},
            "artifacts": [reference], "open_loop": reference}


def failure(case):
    return {"case": deepcopy(case), "status": "failed", "error": "synthetic execution failure",
            "metrics": None, "artifacts": [], "open_loop": None}


def evidence(ledger, path, data=b"synthetic fixture; not scientific evidence"):
    destination = ledger.root / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    return ledger.artifact(path)


@pytest.fixture
def ledger(tmp_path, config):
    run = StudyLedger.create(tmp_path / "run", config, PROVENANCE)
    ref = evidence(run, "evidence/synthetic.bin")
    models = {row["id"]: {"status": "ready", "artifacts": [ref], "error": None}
              for row in study_plan(config)["candidates"] if row["controller"] in ("predictive", "learned")}
    run.bind_training(models, [ref])
    return run


def complete_validation(ledger):
    ref = ledger.artifact("evidence/synthetic.bin")
    for case in validation_cases(ledger.config):
        ledger.record_validation(outcome(case, ref))


def test_selection_uses_global_relative_score_and_declared_exact_ties(config):
    records = [outcome(case) for case in validation_cases(config)]
    # Deliberately reversed input order must not change the declared tie-break order.
    selected = select_candidates(config, list(reversed(records)))
    assert all(row["id"].endswith("00") for row in selected["selected"].values())
    for row in records:
        bright = row["case"]["photons"] == 1e6
        row["metrics"]["open_wfe_nm"] = 1000. if bright else 10.
        # Candidate 00 has lower absolute average WFE, but a worse relative score.
        second = row["case"]["candidate"]["id"].endswith("01")
        row["metrics"]["wfe_nm"] = (200. if bright else 1.) if second else (100. if bright else 9.)
    selected = select_candidates(config, records)
    assert all(row["id"].endswith("01") for row in selected["selected"].values())
    assert selected["pairwise_family"] == 120


@pytest.mark.parametrize("damage", ["missing", "duplicate", "unexpected", "coerced_parameter"])
def test_selection_rejects_incomplete_or_wrong_identities(config, damage):
    records = [outcome(case) for case in validation_cases(config)]
    if damage == "missing": records.pop()
    elif damage == "duplicate": records.append(deepcopy(records[0]))
    elif damage == "unexpected": records[0]["case"]["seed"] += 1_000_000
    else:
        next(row for row in records if row["case"]["candidate"]["controller"] == "predictive")["case"]["candidate"]["parameters"]["horizon"] = True
    with pytest.raises(ValueError, match="validation identity|Incomplete validation"):
        select_candidates(config, records)


@pytest.mark.parametrize("damage,reason", [("saturation", "saturation_limit"), ("not_useful", "not_below_open_loop"),
                                          ("nonfinite", "nonfinite_output"), ("failed", "execution_failed")])
def test_every_trial_gate_and_failed_controller_withholding(config, damage, reason):
    records = [outcome(case) for case in validation_cases(config)]
    for i, row in enumerate(records):
        if row["case"]["candidate"]["controller"] != "learned":
            continue
        if damage == "saturation": row["metrics"]["saturation_fraction"] = .010001
        elif damage == "not_useful": row["metrics"]["wfe_nm"] = 100.
        elif damage == "nonfinite": row["metrics"].update(all_outputs_finite=False, wfe_nm=None)
        else: records[i] = failure(row["case"])
    result = select_candidates(config, records)
    assert result["selected"]["learned"] is None
    assert result["selected"]["integrator"] is not None
    assert result["pairwise_family"] == 120  # Do not shrink the comparison family.
    rejected = [row for row in result["candidates"] if row["candidate"]["controller"] == "learned"]
    assert all(row["score"] is None for row in rejected)
    assert all(trial["reasons"] == [reason] for row in rejected for trial in row["rejected_trials"])
    assert result["failed_validation_records"] == (40 if damage == "failed" else 0)


@pytest.mark.parametrize("key,value", [("wfe_nm", float("nan")), ("wfe_nm", float("inf")),
                                      ("wfe_nm", True), ("wfe_nm", None), ("strehl", 1.1),
                                      ("saturation_fraction", -.1), ("all_outputs_finite", 1)])
def test_invalid_metrics_are_never_silently_used(config, key, value):
    records = [outcome(case) for case in validation_cases(config)]
    records[0]["metrics"][key] = value
    with pytest.raises(ValueError, match="metric"):
        select_candidates(config, records)


@pytest.mark.parametrize("damage", ["value", "artifact"])
def test_candidates_must_share_the_same_open_reference(config, damage):
    records = [outcome(case) for case in validation_cases(config)]
    if damage == "value": records[0]["metrics"]["open_wfe_nm"] = 999.
    else: records[0]["open_loop"] = {"path": "evidence/different.bin", "sha256": "f" * 64}
    with pytest.raises(ValueError, match="paired open-loop"):
        select_candidates(config, records)


def test_missing_selection_never_calls_backend(ledger):
    backend = Mock()
    with pytest.raises(ValueError, match="frozen selection"):
        ledger.evaluate_chunk(backend, 1)
    backend.assert_not_called()
    assert not (ledger.root / "evaluation").exists()


def test_missing_training_and_incomplete_validation_cannot_advance(tmp_path, config):
    run = StudyLedger.create(tmp_path / "empty", config, PROVENANCE)
    with pytest.raises(FileNotFoundError):
        run.record_validation(failure(validation_cases(config)[0]))
    assert not (run.root / "validation").exists()


def test_incomplete_validation_cannot_freeze_and_retry_preserves_record(ledger):
    row = outcome(validation_cases(ledger.config)[0], ledger.artifact("evidence/synthetic.bin"))
    ledger.record_validation(row)
    path = next((ledger.root / "validation").glob("*.json"))
    original = path.read_bytes()
    ledger.record_validation(row)
    assert path.read_bytes() == original
    row["metrics"]["wfe_nm"] = 1.
    with pytest.raises(ValueError, match="replace existing evidence"):
        ledger.record_validation(row)
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="Incomplete validation"):
        ledger.freeze_selection()
    assert not (ledger.root / "selection.json").exists()


@pytest.mark.parametrize("field", ["source_commit", "source_sha256", "runtime_sha256", "config"])
def test_resume_requires_identical_config_and_provenance(ledger, field):
    config, provenance = deepcopy(ledger.config), deepcopy(PROVENANCE)
    if field == "config": config["training"]["learned_steps"] += 1
    else: provenance[field] = "d" * len(provenance[field])
    with pytest.raises(ValueError, match="provenance changed"):
        StudyLedger(ledger.root, config, provenance)


def test_validation_is_bound_to_original_training_bundle(ledger):
    ledger.record_validation(outcome(validation_cases(ledger.config)[0], ledger.artifact("evidence/synthetic.bin")))
    path = ledger.root / "training.json"
    bundle = json.loads(path.read_text())
    bundle["evidence"].append(evidence(ledger, "evidence/refitted.bin", b"changed training"))
    path.write_text(json.dumps(bundle))  # Simulate an accidental out-of-band edit.
    with pytest.raises(ValueError, match="training binding changed"):
        ledger.freeze_selection()


@pytest.mark.parametrize("damage", ["artifact", "selection", "validation", "training"])
def test_frozen_input_drift_blocks_evaluation_before_callback(ledger, damage):
    complete_validation(ledger)
    ledger.freeze_selection()
    if damage == "artifact": (ledger.root / "evidence/synthetic.bin").write_bytes(b"changed parameters or raw data")
    elif damage == "selection":
        path = ledger.root / "selection.json"
        data = json.loads(path.read_text())
        data["result"]["selected"]["integrator"] = None
        path.write_text(json.dumps(data))
    elif damage == "validation": next((ledger.root / "validation").glob("*.json")).unlink()
    else:
        path = ledger.root / "training.json"
        data = json.loads(path.read_text())
        data["models"]["learned-00"]["error"] = "changed"
        path.write_text(json.dumps(data))
    backend = Mock()
    with pytest.raises(ValueError):
        ledger.evaluate_chunk(backend, 1)
    backend.assert_not_called()
    assert not (ledger.root / "evaluation").exists()


def test_selection_is_idempotent_and_blocks_late_validation_or_refitting(ledger):
    complete_validation(ledger)
    result = ledger.freeze_selection()
    original = (ledger.root / "selection.json").read_bytes()
    assert ledger.freeze_selection() == result
    assert (ledger.root / "selection.json").read_bytes() == original
    with pytest.raises(ValueError, match="frozen"):
        ledger.record_validation(failure(validation_cases(ledger.config)[0]))
    with pytest.raises(ValueError, match="frozen"):
        ledger.bind_training({}, [])


def test_chunks_resume_skip_failed_records_and_preserve_bytes(ledger):
    complete_validation(ledger)
    ledger.freeze_selection()
    ref = ledger.artifact("evidence/synthetic.bin")
    called = []
    def execute(case, training):
        called.append(case)
        return failure(case) if len(called) == 1 else outcome(case, ref)
    first = ledger.evaluate_chunk(execute, 2)
    assert first == {"new_records": 2, "total_records": 2, "planned_records": 600, "complete": False}
    original = {path: path.read_bytes() for path in (ledger.root / "evaluation").glob("*.json")}
    resumed = StudyLedger(ledger.root, ledger.config, PROVENANCE)
    assert resumed.evaluate_chunk(execute, 3)["total_records"] == 5
    assert len(called) == len({json.dumps(case, sort_keys=True) for case in called}) == 5
    assert all(path.read_bytes() == data for path, data in original.items())
    assert any(json.loads(data)["record"]["status"] == "failed" for data in original.values())


def test_interrupted_callback_keeps_prior_trials_and_releases_lock(ledger):
    complete_validation(ledger)
    ledger.freeze_selection()
    ref = ledger.artifact("evidence/synthetic.bin")
    called = []
    def execute(case, training):
        called.append(case)
        if len(called) == 2:
            raise RuntimeError("simulated interruption")
        return outcome(case, ref)
    with pytest.raises(RuntimeError, match="interruption"):
        ledger.evaluate_chunk(execute, 3)
    assert len(list((ledger.root / "evaluation").glob("*.json"))) == 1
    recovered = []
    def recover(case, training):
        recovered.append(case)
        return outcome(case, ref)
    assert ledger.evaluate_chunk(recover, 1)["total_records"] == 2
    assert recovered[0] == called[1]  # Retry only the interrupted, unpublished trial.


def test_withheld_controllers_are_not_executed_and_all_withheld_is_complete(ledger):
    for case in validation_cases(ledger.config):
        ledger.record_validation(failure(case))
    ledger.freeze_selection()
    assert ledger.evaluation_cases() == []
    backend = Mock()
    assert ledger.evaluate_chunk(backend, 1) == {"new_records": 0, "total_records": 0, "planned_records": 0, "complete": True}
    backend.assert_not_called()


@pytest.mark.parametrize("damage", ["binding", "identity", "artifact"])
def test_corrupt_prior_evaluation_is_not_skipped_on_resume(ledger, damage):
    complete_validation(ledger)
    ledger.freeze_selection()
    ref = ledger.artifact("evidence/synthetic.bin")
    ledger.evaluate_chunk(lambda case, training: outcome(case, ref), 1)
    path = next((ledger.root / "evaluation").glob("*.json"))
    saved = json.loads(path.read_text())
    if damage == "binding": saved["selection_sha256"] = "0" * 64
    elif damage == "identity": saved["record"]["case"]["seed"] += 1
    else: saved["record"]["artifacts"][0]["sha256"] = "0" * 64
    path.write_text(json.dumps(saved))
    backend = Mock()
    with pytest.raises(ValueError):
        ledger.evaluate_chunk(backend, 1)
    backend.assert_not_called()


def test_backend_cannot_change_identity_or_frozen_models(ledger):
    complete_validation(ledger)
    ledger.freeze_selection()
    ref = ledger.artifact("evidence/synthetic.bin")
    def wrong_case(case, training):
        case["seed"] += 1
        return outcome(case, ref)
    with pytest.raises(ValueError, match="different evaluation identity"):
        ledger.evaluate_chunk(wrong_case, 1)
    def change_model(case, training):
        (ledger.root / "evidence/synthetic.bin").write_bytes(b"refitted")
        return failure(case)
    with pytest.raises(ValueError, match="Artifact hash"):
        ledger.evaluate_chunk(change_model, 1)
    assert not (ledger.root / "evaluation").exists()


def test_outside_paths_symlinks_and_missing_artifacts_rejected(ledger, tmp_path):
    outside = tmp_path / "private.bin"
    outside.write_bytes(b"outside")
    (ledger.root / "evidence/escape.bin").symlink_to(outside)
    for path in (str(outside), "evidence/../../private.bin", "manifest.json", "evidence/escape.bin", "evidence/missing.bin"):
        with pytest.raises(ValueError):
            ledger.artifact(path)


def test_second_writer_fails_without_publishing_or_calling_backend(ledger):
    with (ledger.root / ".lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            ledger.record_validation(failure(validation_cases(ledger.config)[0]))
        backend = Mock()
        with pytest.raises(BlockingIOError):
            ledger.evaluate_chunk(backend, 1)
        backend.assert_not_called()
    assert not (ledger.root / "validation").exists()


def test_failed_training_cannot_produce_successful_validation(tmp_path, config):
    run = StudyLedger.create(tmp_path / "failed-training", config, PROVENANCE)
    ref = evidence(run, "evidence/training-failure.bin")
    models = {row["id"]: {"status": "failed", "artifacts": [ref], "error": "nonfinite training loss"}
              for row in study_plan(config)["candidates"] if row["controller"] in ("predictive", "learned")}
    run.bind_training(models, [ref])
    case = next(case for case in validation_cases(config) if case["candidate"]["controller"] == "learned")
    with pytest.raises(ValueError, match="Failed training"):
        run.record_validation(outcome(case, ref))
    run.record_validation(failure(case))
    assert len(list((run.root / "validation").glob("*.json"))) == 1


def test_failed_atomic_publication_leaves_no_completed_record(ledger, monkeypatch):
    def disk_error(*args):
        raise OSError("simulated full disk")
    with monkeypatch.context() as patch:
        patch.setattr(selection_module.os, "link", disk_error)
        with pytest.raises(OSError, match="full disk"):
            ledger.record_validation(failure(validation_cases(ledger.config)[0]))
    assert list((ledger.root / "validation").iterdir()) == []
    ledger.record_validation(failure(validation_cases(ledger.config)[0]))
    assert len(list((ledger.root / "validation").glob("*.json"))) == 1


def test_one_withheld_controller_never_enters_evaluation_plan(ledger):
    ref = ledger.artifact("evidence/synthetic.bin")
    for case in validation_cases(ledger.config):
        ledger.record_validation(failure(case) if case["candidate"]["controller"] == "learned" else outcome(case, ref))
    ledger.freeze_selection()
    cases = ledger.evaluation_cases()
    assert len(cases) == 480
    assert {case["candidate"]["controller"] for case in cases} == {"integrator", "predictive", "forward", "fixed"}
    assert {case["seed"] for case in cases} == set(ledger.config["seeds"]["evaluation"])
    assert {case["fault"] for case in cases} == {"none", "optical_gain", "vibration"}


def test_full_evaluation_completion_and_invalid_chunk_budgets(ledger):
    complete_validation(ledger)
    ledger.freeze_selection()
    backend = Mock(side_effect=lambda case, training: failure(case))
    for limit in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            ledger.evaluate_chunk(backend, limit)
    backend.assert_not_called()
    result = ledger.evaluate_chunk(backend, 1000)
    assert result == {"new_records": 600, "total_records": 600, "planned_records": 600, "complete": True}
    assert backend.call_count == 600
    assert ledger.evaluate_chunk(backend, 1)["new_records"] == 0
    assert backend.call_count == 600


@pytest.mark.parametrize("damage", ["manifest", "selection"])
def test_json_boolean_is_not_an_integer_in_frozen_evidence(ledger, damage):
    complete_validation(ledger)
    ledger.freeze_selection()
    path = ledger.root / f"{damage}.json"
    data = json.loads(path.read_text())
    if damage == "manifest": data["schema_version"] = True
    else: data["result"]["selected"]["predictive"]["parameters"]["horizon"] = True
    path.write_text(json.dumps(data))
    backend = Mock()
    with pytest.raises(ValueError, match="changed"):
        ledger.evaluate_chunk(backend, 1)
    backend.assert_not_called()
