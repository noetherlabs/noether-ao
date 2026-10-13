"""Small physical wiring on separate seeds with explicitly synthetic model fixtures."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from noether_optics.artifacts import load_config
from noether_optics.controller_evidence import load_arrays, save_arrays, save_metadata
from noether_optics.controller_physics import episode_config, paired_open_loop, controller_kernel
from noether_optics.controller_protocol import load_study, study_plan
from noether_optics.controller_selection import StudyLedger, _bytes, _digest, _read, validation_cases
from noether_optics.controller_trials import evaluate_trials, validate_trials
from noether_optics import controller_trials as trials
from noether_optics import controller_physics as physics
from noether_optics import simulation, atmosphere

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def prepared(tmp_path):
    study = load_study(ROOT / "configs/controller_study_v1.yaml")
    study["study_id"] = "trial_adapter_fixture_only"
    for role, start in (("training", 81001), ("validation", 82001), ("evaluation", 83001)):
        study["seeds"][role] = list(range(start, start + len(study["seeds"][role])))
    study["seeds"]["initialization"] = 84001
    study["loop"].update(frames=12, burn_in=3, fault_onset=6, fault_end=12)
    study["candidates_per_controller"] = 1
    for grid in study["candidate_grids"].values():
        for key in grid: grid[key] = grid[key][:1]
    base = load_config(ROOT / "configs/simulation.yaml")
    base.update(n=8, modes=4, detector_n=16, screen_factor=2)
    source = {"commit": "a" * 40, "fixture": "synthetic provenance, not a study run"}
    runtime = {"fixture": "synthetic runtime"}
    provenance = {"source_commit": source["commit"], "source_sha256": _digest(_bytes(source)),
                  "runtime_sha256": _digest(_bytes(runtime))}
    ledger = StudyLedger.create(tmp_path / "run", study, provenance)
    refs = [save_metadata(ledger, f"evidence/provenance/{name}.json", data)
            for name, data in (("source", source), ("runtime", runtime), ("study", study), ("base", base))]
    models = {}
    for candidate in study_plan(study)["candidates"]:
        name = candidate["controller"]
        if name not in ("learned", "predictive"): continue
        arrays = {"coefficients": np.zeros((4, 2))} if name == "predictive" else {
            f"parameter_{i}": np.zeros(shape) for i, shape in enumerate(((3, 8), (8,), (8, 1), (1,)))}
        ref = save_arrays(ledger, f"evidence/models/{candidate['id']}/parameters.npz", arrays)
        models[candidate["id"]] = {"status": "ready", "artifacts": [ref], "error": None}
    return ledger, base, models, refs


@pytest.fixture
def ready(prepared):
    ledger, base, models, refs = prepared
    ledger.bind_training(models, refs)
    return ledger, base


def synthetic(case, ref):
    return {"case": deepcopy(case), "status": "completed", "error": None, "artifacts": [ref],
            "open_loop": ref, "metrics": {"all_outputs_finite": True, "wfe_nm": 1.,
            "open_wfe_nm": 2., "saturation_fraction": 0., "strehl": .9}}


def records(ledger, stage):
    return [_read(path)["record"] for path in sorted((ledger.root / stage).glob("*.json"))]


def block_physics(monkeypatch):
    forbidden = Mock(side_effect=AssertionError("Physics must remain inaccessible"))
    monkeypatch.setattr(simulation, "calibration", forbidden)
    monkeypatch.setattr(atmosphere, "phase_sequence", forbidden)
    monkeypatch.setattr(physics, "controller_kernel", forbidden)
    return forbidden


def test_missing_training_or_selection_never_starts_physics(prepared, monkeypatch):
    ledger, base, models, refs = prepared
    forbidden = block_physics(monkeypatch)
    with pytest.raises(FileNotFoundError): validate_trials(ledger, base, 1)
    with pytest.raises(ValueError, match="frozen selection"): evaluate_trials(ledger, base, 1)
    ledger.bind_training(models, refs)
    with pytest.raises(ValueError, match="frozen selection"): evaluate_trials(ledger, base, 1)
    forbidden.assert_not_called()
    assert not (ledger.root / "evidence/evaluation").exists()


def test_physical_validation_records_raw_metrics_and_resumes_without_reexecution(ready, monkeypatch):
    ledger, base = ready
    assert validate_trials(ledger, base, 1)["new_records"] == 1
    row = records(ledger, "validation")[0]
    assert row["status"] == "completed" and row["metrics"]["all_outputs_finite"]
    raw = load_arrays(ledger, row["artifacts"][0])
    assert physics.FULL_OUTPUT_KEYS.issubset(raw) and not raw["fallback"].any()
    detail = _read(ledger.root / row["artifacts"][1]["path"])
    assert detail["post_burn_frames"] == [3, 12] and detail["fault_window_frames"] == [6, 12]
    expected = physics.paired_metrics(raw, load_arrays(ledger, row["open_loop"]), detail["config"])
    assert detail["metrics"] == row["metrics"] == expected["metrics"]
    second = validate_trials(ledger, base, 1)
    assert second["total_records"] == 2
    assert {r["case"]["seed"] for r in records(ledger, "validation")} == {82001, 82002}


def test_candidates_share_the_same_saved_atmosphere_and_paired_reference(ready):
    ledger, base = ready
    validate_trials(ledger, base, 1)
    first = records(ledger, "validation")[0]
    cases = validation_cases(ledger.config)
    ref = ledger.artifact("evidence/provenance/source.json")
    for case in cases[1:20]: ledger.record_validation(synthetic(case, ref))
    assert cases[20]["candidate"]["controller"] == "predictive"
    validate_trials(ledger, base, 1)
    predictor = next(row for row in records(ledger, "validation") if row["case"] == cases[20])
    assert predictor["open_loop"] == first["open_loop"]
    assert predictor["artifacts"][2:4] == first["artifacts"][2:4]


def test_failed_training_records_failure_without_model_or_atmosphere_access(prepared, monkeypatch):
    ledger, base, models, refs = prepared
    predictor = next(key for key in models if key.startswith("predictive"))
    models[predictor] = {"status": "failed", "artifacts": [], "error": "retained synthetic training failure"}
    ledger.bind_training(models, refs)
    for case in validation_cases(ledger.config)[:20]: ledger.record_validation(synthetic(case, refs[0]))
    forbidden = block_physics(monkeypatch)
    validate_trials(ledger, base, 1)
    failed = next(row for row in records(ledger, "validation") if row["status"] == "failed")
    assert "training failure" in failed["error"] and failed["metrics"] is None
    forbidden.assert_not_called()


@pytest.mark.parametrize("damage", ["base", "missing_provenance", "wrong_model_schema", "wrong_model_shape"])
def test_bad_frozen_inputs_reject_before_physics(prepared, monkeypatch, damage):
    ledger, base, models, refs = prepared
    predictor = next(key for key in models if key.startswith("predictive"))
    if damage == "base": base["n"] += 1
    elif damage == "missing_provenance": refs = refs[:-1]
    else:
        arrays = {"wrong": np.zeros((4, 2))} if damage == "wrong_model_schema" else {"coefficients": np.zeros((3, 2))}
        ref = save_arrays(ledger, "evidence/bad/parameters.npz", arrays)
        models[predictor]["artifacts"] = [ref]
    ledger.bind_training(models, refs)
    if damage.startswith("wrong_model"):
        for case in validation_cases(ledger.config)[:20]: ledger.record_validation(synthetic(case, refs[0]))
    forbidden = block_physics(monkeypatch)
    with pytest.raises(ValueError): validate_trials(ledger, base, 1)
    forbidden.assert_not_called()


def test_disk_refusal_precedes_calibration_and_raw_acquisition(ready, monkeypatch):
    ledger, base = ready
    monkeypatch.setattr(trials.shutil, "disk_usage", lambda path: SimpleNamespace(free=1))
    forbidden = block_physics(monkeypatch)
    with pytest.raises(RuntimeError, match="free disk"): validate_trials(ledger, base, 1)
    forbidden.assert_not_called()
    assert not (ledger.root / "evidence/validation").exists()


def test_raw_before_metadata_interruption_retries_without_replacement(ready, monkeypatch):
    ledger, base = ready
    real = trials.save_metadata
    def interrupted(ledger, relative, data):
        if "/trials/" in relative: raise OSError("simulated metadata interruption")
        return real(ledger, relative, data)
    monkeypatch.setattr(trials, "save_metadata", interrupted)
    with pytest.raises(OSError): validate_trials(ledger, base, 1)
    assert records(ledger, "validation") == []
    raw = next((ledger.root / "evidence/validation/trials").glob("*.npz"))
    before = raw.read_bytes()
    monkeypatch.setattr(trials, "save_metadata", real)
    validate_trials(ledger, base, 1)
    assert raw.read_bytes() == before and len(records(ledger, "validation")) == 1


def test_nonfinite_full_output_is_retained_and_cannot_become_eligible(ready, monkeypatch):
    ledger, base = ready
    real = physics.controller_kernel
    def poisoned(*args, **kwargs):
        kernel = real(*args, **kwargs)
        def run(*inputs):
            output = dict(kernel(*inputs))
            output["measurement"] = output["measurement"].at[0, 0].set(np.nan)
            return output
        return run
    monkeypatch.setattr(physics, "controller_kernel", poisoned)
    validate_trials(ledger, base, 1)
    row = records(ledger, "validation")[0]
    assert row["status"] == "completed" and row["metrics"]["all_outputs_finite"] is False
    assert np.isnan(load_arrays(ledger, row["artifacts"][0])["measurement"][0, 0])


def test_shared_evidence_drift_blocks_resume_before_physics(ready, monkeypatch):
    ledger, base = ready
    validate_trials(ledger, base, 1)
    row = records(ledger, "validation")[0]
    (ledger.root / row["artifacts"][2]["path"]).write_bytes(b"changed atmosphere")
    forbidden = block_physics(monkeypatch)
    with pytest.raises(ValueError, match="hash or identity"): validate_trials(ledger, base, 1)
    forbidden.assert_not_called()


def test_gated_physical_evaluation_uses_fault_matched_open_loop(ready):
    ledger, base = ready
    ref = ledger.artifact("evidence/provenance/source.json")
    for case in validation_cases(ledger.config): ledger.record_validation(synthetic(case, ref))
    ledger.freeze_selection()
    # Synthetic prior records advance to the first vibration identity. No earlier
    # physical evaluation is generated by this correctness fixture.
    ledger.evaluate_chunk(lambda case, training: synthetic(case, ref), 20)
    result = evaluate_trials(ledger, base, 1)
    assert result["total_records"] == 21
    row = next(r for r in records(ledger, "evaluation") if r["case"]["fault"] == "vibration")
    case = row["case"]
    cfg = episode_config(base, ledger.config, "evaluation", case["photons"], case["wind_scale"], case["candidate"])
    cal = simulation.calibration(base)
    phase = load_arrays(ledger, row["artifacts"][2])["phase"]
    expected = controller_kernel(cfg | {"gain": 0.}, cal, "integrator", "vibration")(phase, case["seed"])
    actual = load_arrays(ledger, row["open_loop"])
    np.testing.assert_allclose(actual["variance"], expected["variance"], atol=1e-12, rtol=1e-12)
    clean = paired_open_loop(phase, cfg, cal, "none")
    assert not np.allclose(actual["variance"][6:], clean["variance"][6:])


def test_learned_trial_uses_the_frozen_saved_parameters(ready):
    ledger, base = ready
    ref = ledger.artifact("evidence/provenance/source.json")
    cases = validation_cases(ledger.config)
    for case in cases[:80]: ledger.record_validation(synthetic(case, ref))
    assert cases[80]["candidate"]["controller"] == "learned"
    validate_trials(ledger, base, 1)
    row = next(r for r in records(ledger, "validation") if r["case"] == cases[80])
    detail = _read(ledger.root / row["artifacts"][1]["path"])
    model = load_arrays(ledger, detail["model"]["artifacts"][0])
    phase = load_arrays(ledger, row["artifacts"][2])["phase"]
    expected = controller_kernel(detail["config"], simulation.calibration(base), "learned")(
        phase, row["case"]["seed"], tuple(model[f"parameter_{i}"] for i in range(4)))
    actual = load_arrays(ledger, row["artifacts"][0])
    for key in actual:
        np.testing.assert_allclose(actual[key], expected[key], atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("damage", ["schema", "shape", "duplicate"])
def test_malformed_learned_models_block_physics(prepared, monkeypatch, damage):
    ledger, base, models, refs = prepared
    name = next(key for key in models if key.startswith("learned"))
    arrays = {f"parameter_{i}": np.zeros(shape) for i, shape in enumerate(((3, 8), (8,), (8, 1), (1,)))}
    if damage == "schema": arrays.pop("parameter_3")
    if damage == "shape": arrays["parameter_3"] = np.zeros(2)
    ref = save_arrays(ledger, "evidence/bad/parameters.npz", arrays)
    if damage == "duplicate": models[name]["artifacts"].append(ref)
    else: models[name]["artifacts"] = [ref]
    ledger.bind_training(models, refs)
    for case in validation_cases(ledger.config)[:80]: ledger.record_validation(synthetic(case, refs[0]))
    forbidden = block_physics(monkeypatch)
    with pytest.raises(ValueError, match="parameters.npz|frozen learned"):
        validate_trials(ledger, base, 1)
    forbidden.assert_not_called()
