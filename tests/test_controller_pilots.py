"""Durable training-data checks on synthetic and reduced optical fixtures."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from noether_optics import atmosphere, simulation
from noether_optics import controller_physics as physics
from noether_optics import controller_pilots as pilots
from noether_optics.artifacts import load_config
from noether_optics.controller_evidence import load_arrays, save_metadata
from noether_optics.controller_protocol import load_study, study_plan
from noether_optics.controller_selection import StudyLedger, _bytes, _case_id, _digest, _read

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def prepared(tmp_path):
    study = load_study(ROOT / "configs/controller_study_v1.yaml")
    study["study_id"] = "pilot_records_correctness_only"
    for role, start in (("training", 101001), ("validation", 102001), ("evaluation", 103001)):
        study["seeds"][role] = list(range(start, start + len(study["seeds"][role])))
    study["seeds"]["initialization"] = 104001
    study["training"].update(frames=10, burn_in=2, learned_steps=1)
    base = load_config(ROOT / "configs/simulation.yaml")
    base.update(n=8, modes=4, detector_n=16, screen_factor=2)
    source = {"commit": "a" * 40, "fixture": "synthetic source, not study provenance"}
    runtime = {"fixture": "synthetic runtime"}
    provenance = {"source_commit": source["commit"], "source_sha256": _digest(_bytes(source)),
                  "runtime_sha256": _digest(_bytes(runtime))}
    ledger = StudyLedger.create(tmp_path / "run", study, provenance)
    refs = [save_metadata(ledger, f"evidence/provenance/{name}.json", value)
            for name, value in (("source", source), ("runtime", runtime), ("study", study), ("base", base))]
    candidate = next(c for c in study_plan(study)["candidates"] if c["controller"] == "predictive")
    return SimpleNamespace(ledger=ledger, base=base, refs=refs, candidate=candidate)


def acquire(p, limit=12):
    return pilots.acquire_pilots(p.ledger, p.base, p.refs, limit)


def fit(p):
    return pilots.fit_predictor_record(p.ledger, p.base, p.candidate, p.refs)


def rows(p):
    return [_read(path) for path in sorted((p.ledger.root / pilots.PILOTS / "records").glob("*.json"))]


def model_dir(p):
    return p.ledger.root / f"evidence/training/{p.candidate['id']}"


def synthetic_backend(monkeypatch, *, failure=None, interrupt=None):
    calls = {"calibration": 0, "phases": [], "pilots": []}
    def calibrate(cfg):
        calls["calibration"] += 1
        return {"mask": np.ones((cfg["n"], cfg["n"])),
                "reconstructor": np.eye(cfg["modes"], cfg["detector_n"]**2) * (1e308 if failure == "telemetry" else 1),
                "reference": np.zeros(cfg["detector_n"]**2)}
    def phase(cfg, seed):
        calls["phases"].append((seed, cfg["wind_scale"]))
        value = np.sin(np.arange(cfg["n"]**2).reshape(cfg["n"], cfg["n"])) * cfg["wind_scale"]
        return np.broadcast_to(value, (cfg["frames"], cfg["n"], cfg["n"])).copy() * (np.nan if failure == "phase" else 1)
    def build(cfg, cal, controller, fault):
        assert controller == "integrator" and fault == "none"
        assert (cfg["gain"], cfg["leak"], cfg["delay_frames"]) == (.3, .999, 2)
        assert cfg["faults"]["onset"] == cfg["faults"]["end"] == cfg["frames"]
        def run(phases, seed):
            calls["pilots"].append((seed, cfg["photons"], cfg["wind_scale"]))
            if len(calls["pilots"]) == interrupt: raise RuntimeError("pilot backend interrupted")
            frames, modes, pixels = cfg["frames"], cfg["modes"], cfg["detector_n"]**2
            t = np.arange(frames, dtype=float)[:, None]
            measured = np.zeros((frames, pixels))
            measured[:, :modes] = .1 * np.sin(t + np.arange(modes)[None, :]) + .01 * t * (seed % 7)
            if failure == "telemetry": measured[:] = 1e308
            out = {key: np.zeros(frames) for key in ("variance", "strehl", "score")}
            out.update({key: np.zeros(frames, dtype=bool) for key in ("alarm", "fallback", "saturated", "unavailable")})
            out.update(command=np.zeros((frames, modes)), applied=np.broadcast_to(t * .02, (frames, modes)).copy(),
                       estimate=np.full((frames, modes), 42.), measurement=measured, prediction=np.zeros_like(measured))
            if failure == "outputs": out["prediction"][0, 0] = np.nan
            if failure == "shape": out["measurement"] = measured[:-1]
            return out
        return run
    monkeypatch.setattr(simulation, "calibration", calibrate)
    monkeypatch.setattr(atmosphere, "phase_sequence", phase)
    monkeypatch.setattr(physics, "controller_kernel", build)
    return calls


def forbid_backend(monkeypatch):
    forbidden = Mock(side_effect=AssertionError("No new numerical execution"))
    for module, name in ((simulation, "calibration"), (atmosphere, "phase_sequence"),
                         (physics, "controller_kernel"), (pilots, "fit_measured_predictor")):
        monkeypatch.setattr(module, name, forbidden)
    return forbidden


def test_bounded_pilots_share_phases_and_completed_resume_skips_backend(prepared, monkeypatch):
    calls = synthetic_backend(monkeypatch)
    assert acquire(prepared, 5) == {"new_records": 5, "total_records": 5, "complete": False, "failed_records": 0}
    assert acquire(prepared, 20) == {"new_records": 7, "total_records": 12, "complete": True, "failed_records": 0}
    assert len(calls["pilots"]) == 12 and len(calls["phases"]) == 6
    records = rows(prepared)
    assert len({row["phase"]["path"] for row in records}) == 6
    assert len({row["calibration"]["sha256"] for row in records}) == 1
    before = {p: p.read_bytes() for p in prepared.ledger.root.rglob("*") if p.is_file()}
    forbidden = forbid_backend(monkeypatch)
    assert acquire(prepared)["new_records"] == 0
    forbidden.assert_not_called()
    assert all(p.read_bytes() == data for p, data in before.items())


def test_predictor_rebuild_uses_saved_measurements_and_separate_episodes(prepared, monkeypatch):
    synthetic_backend(monkeypatch)
    acquire(prepared)
    result = fit(prepared)
    assert result["model"]["status"] == "ready"
    episodes = []
    for row in rows(prepared):
        raw = load_arrays(prepared.ledger, row["outputs"])
        cal = load_arrays(prepared.ledger, row["calibration"])
        telemetry = load_arrays(prepared.ledger, row["telemetry"])["telemetry"]
        np.testing.assert_array_equal(telemetry, physics.measured_telemetry(raw, cal))
        assert not np.array_equal(telemetry, raw["estimate"])
        episodes.append({"case": row["case"], "telemetry": telemetry})
    expected = physics.fit_measured_predictor(prepared.ledger.config, prepared.candidate, episodes, 4)
    actual = load_arrays(prepared.ledger, result["model"]["artifacts"][0])
    history = load_arrays(prepared.ledger, result["model"]["artifacts"][1])
    np.testing.assert_array_equal(actual["coefficients"], expected["coefficients"])
    np.testing.assert_array_equal(history["episode_coefficients"], expected["episode_coefficients"])
    forbidden = forbid_backend(monkeypatch)
    assert fit(prepared) == result
    forbidden.assert_not_called()
    # The terminal model/evidence format can participate in aggregate binding.
    models = {c["id"]: {"status": "failed", "artifacts": [], "error": "unused correctness fixture"}
              for c in study_plan(prepared.ledger.config)["candidates"] if c["controller"] in ("predictive", "learned")}
    models[prepared.candidate["id"]] = result["model"]
    prepared.ledger.bind_training(models, result["evidence"])
    assert prepared.ledger._training()["models"] == models


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_limit_rejects_before_any_training_evidence(prepared, monkeypatch, limit):
    forbidden = forbid_backend(monkeypatch)
    with pytest.raises(ValueError): acquire(prepared, limit)
    forbidden.assert_not_called()
    assert not (prepared.ledger.root / "evidence/training").exists()


def test_incomplete_cohort_blocks_fitting(prepared, monkeypatch):
    synthetic_backend(monkeypatch)
    acquire(prepared, 1)
    forbidden = forbid_backend(monkeypatch)
    with pytest.raises(ValueError, match="Incomplete"): fit(prepared)
    forbidden.assert_not_called()
    assert not model_dir(prepared).exists()


@pytest.mark.parametrize("failure", ["outputs", "phase", "telemetry"])
def test_numerical_pilot_failure_is_retained_and_blocks_predictor_without_retry(prepared, monkeypatch, failure):
    calls = synthetic_backend(monkeypatch, failure=failure)
    with np.errstate(over="ignore", invalid="ignore"):
        result = acquire(prepared)
    assert result["complete"] and result["failed_records"] == 12
    failed = rows(prepared)[0]
    assert failed["status"] == "failed" and failed["error"]
    ref = failed[{"outputs": "outputs", "phase": "phase", "telemetry": "telemetry"}[failure]]
    assert any(not np.isfinite(a).all() for a in load_arrays(prepared.ledger, ref).values())
    if failure == "phase": assert not calls["pilots"]
    forbidden = forbid_backend(monkeypatch)
    fitted = fit(prepared)
    assert fitted["model"]["status"] == "failed" and "pilot failed" in fitted["model"]["error"]
    assert not (model_dir(prepared) / "parameters.npz").exists()
    assert acquire(prepared)["new_records"] == 0 and fit(prepared) == fitted
    forbidden.assert_not_called()


@pytest.mark.parametrize("failure", ["nonfinite", "mean_overflow", "solve"])
def test_predictor_numerical_failures_retain_coefficients_and_are_terminal(prepared, monkeypatch, failure):
    synthetic_backend(monkeypatch)
    acquire(prepared)
    calls = []
    real = physics.predict_fit
    def broken(telemetry, horizon, ridge):
        calls.append(1)
        if failure == "solve" and len(calls) == 2: raise np.linalg.LinAlgError("synthetic singular solve")
        if failure == "nonfinite": return np.full((4, 2), np.nan)
        if failure == "mean_overflow": return np.full((4, 2), 1e308)
        return real(telemetry, horizon, ridge)
    monkeypatch.setattr(physics, "predict_fit", broken)
    with np.errstate(over="ignore", invalid="ignore"): result = fit(prepared)
    assert result["model"]["status"] == "failed"
    history = load_arrays(prepared.ledger, result["model"]["artifacts"][0])
    if failure == "solve":
        assert history["episode_coefficients"].shape == (1, 4, 2) and len(calls) == 2
        assert _read(model_dir(prepared) / "history.json")["failed_case"] == physics.training_cases(prepared.ledger.config)[1]
    else: assert not np.isfinite(history["coefficients"]).all()
    assert not (model_dir(prepared) / "parameters.npz").exists()
    forbidden = forbid_backend(monkeypatch)
    assert fit(prepared) == result
    forbidden.assert_not_called()


def test_backend_interruption_preserves_completed_pilots_and_diagnostics(prepared, monkeypatch):
    synthetic_backend(monkeypatch, interrupt=4)
    with pytest.raises(RuntimeError): acquire(prepared)
    assert len(rows(prepared)) == 3
    before = {path: path.read_bytes() for path in prepared.ledger.root.rglob("*.npz")}
    calls = synthetic_backend(monkeypatch)
    assert acquire(prepared)["new_records"] == 9 and len(calls["pilots"]) == 9
    assert all(path.read_bytes() == data for path, data in before.items())
    result = fit(prepared)
    assert any("interruptions" in ref["path"] for ref in result["evidence"])


@pytest.mark.parametrize("kind", ["pilot", "predictor"])
def test_raw_before_record_interruption_accepts_identical_retry(prepared, monkeypatch, kind):
    calls = synthetic_backend(monkeypatch)
    if kind == "predictor": acquire(prepared)
    real = pilots.save_metadata
    def interrupted(ledger, relative, data):
        if ("/records/" in relative if kind == "pilot" else relative.endswith("/complete.json")):
            raise OSError("synthetic publication failure")
        return real(ledger, relative, data)
    monkeypatch.setattr(pilots, "save_metadata", interrupted)
    with pytest.raises(OSError):
        acquire(prepared, 1) if kind == "pilot" else fit(prepared)
    before = {path: path.read_bytes() for path in prepared.ledger.root.rglob("*.npz")}
    monkeypatch.setattr(pilots, "save_metadata", real)
    acquire(prepared, 1) if kind == "pilot" else fit(prepared)
    assert all(path.read_bytes() == data for path, data in before.items())
    assert len(calls["pilots"]) == (2 if kind == "pilot" else 12)


@pytest.mark.parametrize("damage", ["base", "source", "raw", "calibration", "phase", "case", "extra_case"])
def test_drift_blocks_resumed_acquisition_before_backend(prepared, monkeypatch, damage):
    synthetic_backend(monkeypatch)
    acquire(prepared, 1)
    row = rows(prepared)[0]
    if damage == "base": prepared.base["gain"] += .1
    elif damage == "source": (prepared.ledger.root / prepared.refs[0]["path"]).write_bytes(b"changed")
    elif damage in ("raw", "calibration", "phase"):
        (prepared.ledger.root / row["outputs" if damage == "raw" else damage]["path"]).write_bytes(b"changed")
    else:
        path = next((prepared.ledger.root / pilots.PILOTS / "records").glob("*.json"))
        if damage == "case":
            row["case"]["seed"] = 103001
            path.write_bytes(_bytes(row))
        else: path.with_name("foreign.json").write_bytes(path.read_bytes())
    forbidden = forbid_backend(monkeypatch)
    with pytest.raises(ValueError): acquire(prepared)
    forbidden.assert_not_called()


@pytest.mark.parametrize("frozen", ["training.json", "selection.json", "validation/prior.json"])
def test_frozen_stage_refuses_acquisition_and_fitting(prepared, monkeypatch, frozen):
    path = prepared.ledger.root / frozen
    path.parent.mkdir(exist_ok=True)
    path.write_text("{}")
    forbidden = forbid_backend(monkeypatch)
    with pytest.raises(ValueError, match="frozen"): acquire(prepared)
    with pytest.raises(ValueError, match="frozen"): fit(prepared)
    forbidden.assert_not_called()


def test_disk_refusal_precedes_calibration_and_phase_generation(prepared, monkeypatch):
    forbidden = forbid_backend(monkeypatch)
    monkeypatch.setattr(pilots.shutil, "disk_usage", lambda path: SimpleNamespace(free=1))
    with pytest.raises(RuntimeError, match="free disk"): acquire(prepared)
    forbidden.assert_not_called()
    assert rows(prepared) == []


def test_invalid_raw_shape_is_an_interruption_with_retained_raw(prepared, monkeypatch):
    synthetic_backend(monkeypatch, failure="shape")
    with pytest.raises(ValueError, match="shape"): acquire(prepared)
    assert rows(prepared) == []
    assert list((prepared.ledger.root / pilots.PILOTS / "outputs").glob("*.npz"))
    assert list((prepared.ledger.root / pilots.PILOTS / "interruptions").glob("*.json"))


@pytest.mark.parametrize("mutation", ["manifest", "training", "source"])
def test_backend_changes_run_binding_before_pilot_publication(prepared, monkeypatch, mutation):
    synthetic_backend(monkeypatch)
    real = physics.controller_kernel
    def build(*args):
        execute = real(*args)
        def run(*values):
            result = execute(*values)
            path = prepared.ledger.root / (prepared.refs[0]["path"] if mutation == "source" else mutation + ".json")
            path.write_text("{}")
            return result
        return run
    monkeypatch.setattr(physics, "controller_kernel", build)
    with pytest.raises(ValueError): acquire(prepared)
    assert rows(prepared) == []


def test_predictor_fit_holds_lock_and_rechecks_inputs_before_terminal_publication(prepared, monkeypatch):
    synthetic_backend(monkeypatch)
    acquire(prepared)
    real = pilots.fit_measured_predictor
    def mutate(*args):
        with pytest.raises(BlockingIOError):
            with prepared.ledger._locked(): raise AssertionError("Lock not held")
        result = real(*args)
        (prepared.ledger.root / "training.json").write_text("{}")
        return result
    monkeypatch.setattr(pilots, "fit_measured_predictor", mutate)
    with pytest.raises(ValueError, match="frozen"): fit(prepared)
    assert not (model_dir(prepared) / "complete.json").exists()
    assert not (model_dir(prepared) / "parameters.npz").exists()


def test_saved_terminal_predictor_rechecks_parameter_integrity(prepared, monkeypatch):
    synthetic_backend(monkeypatch)
    acquire(prepared)
    result = fit(prepared)
    (prepared.ledger.root / result["model"]["artifacts"][0]["path"]).write_bytes(b"changed")
    forbidden = forbid_backend(monkeypatch)
    with pytest.raises(ValueError): fit(prepared)
    forbidden.assert_not_called()


def test_foreign_predictor_candidate_is_rejected_before_acquisition(prepared, monkeypatch):
    prepared.candidate["parameters"]["ridge"] = .123
    forbidden = forbid_backend(monkeypatch)
    with pytest.raises(ValueError, match="declared predictive"): fit(prepared)
    forbidden.assert_not_called()
    assert not (prepared.ledger.root / "evidence/training").exists()


def test_unexpected_predictor_exception_is_preserved_as_interruption(prepared, monkeypatch):
    synthetic_backend(monkeypatch)
    acquire(prepared)
    real = pilots.fit_measured_predictor
    monkeypatch.setattr(pilots, "fit_measured_predictor", Mock(side_effect=RuntimeError("unexpected fit error")))
    with pytest.raises(RuntimeError, match="unexpected fit error"): fit(prepared)
    assert not (model_dir(prepared) / "complete.json").exists()
    diagnostic = _read(next((model_dir(prepared) / "interruptions").glob("*.json")))
    assert diagnostic["error_type"] == "RuntimeError"
    monkeypatch.setattr(pilots, "fit_measured_predictor", real)
    result = fit(prepared)
    assert result["model"]["status"] == "ready"
    assert any("interruptions" in ref["path"] for ref in result["evidence"])


def test_recomputed_calibration_must_match_before_more_pilot_execution(prepared, monkeypatch):
    synthetic_backend(monkeypatch)
    acquire(prepared, 1)
    original = simulation.calibration
    def changed(base):
        cal = original(base)
        cal["reference"] += .1
        return cal
    monkeypatch.setattr(simulation, "calibration", changed)
    forbidden = Mock(side_effect=AssertionError("No pilot on changed calibration"))
    monkeypatch.setattr(physics, "controller_kernel", forbidden)
    with pytest.raises(ValueError, match="replace"): acquire(prepared, 1)
    forbidden.assert_not_called()
    assert len(rows(prepared)) == 1


def test_learned_journal_reuses_pilot_atmosphere_files(prepared, monkeypatch):
    from noether_optics import controller_training as training
    from noether_optics.controller_training_journal import fit_learned_record
    synthetic_backend(monkeypatch)
    acquire(prepared)
    records = rows(prepared)
    episodes = [{"case": row["case"], "phases": load_arrays(prepared.ledger, row["phase"])["phase"]}
                for row in records]
    cal = load_arrays(prepared.ledger, records[0]["calibration"])
    candidate = next(c for c in study_plan(prepared.ledger.config)["candidates"] if c["controller"] == "learned")
    before = {path: path.read_bytes() for path in (prepared.ledger.root / "evidence/training/atmospheres").glob("*.npz")}
    # Analytic gradient isolates journal integration; this is not optical training.
    monkeypatch.setattr(training, "_episode_kernel", lambda *args: lambda params, *unused:
        ((np.asarray(1.), np.asarray(True)), tuple(np.full_like(p, .01) for p in params)))
    result = fit_learned_record(prepared.ledger, prepared.base, candidate, cal, episodes, prepared.refs)
    assert result["model"]["status"] == "ready"
    assert all(path.read_bytes() == data for path, data in before.items())
    learned = _read(prepared.ledger.root / f"evidence/training/{candidate['id']}/journal.json")
    assert {row["phase"]["path"] for row in learned["episodes"]} == {row["phase"]["path"] for row in records}


def test_reduced_physical_pilots_and_all_predictor_candidates(prepared):
    assert acquire(prepared)["complete"]
    records = rows(prepared)
    assert len(records) == 12 and all(row["status"] == "ready" for row in records)
    for row in records:
        raw = load_arrays(prepared.ledger, row["outputs"])
        telemetry = load_arrays(prepared.ledger, row["telemetry"])["telemetry"]
        np.testing.assert_allclose(telemetry, raw["estimate"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(raw["applied"][2:], raw["command"][:-2], rtol=1e-12, atol=1e-12)
    for candidate in study_plan(prepared.ledger.config)["candidates"]:
        if candidate["controller"] == "predictive":
            prepared.candidate = candidate
            result = fit(prepared)
            assert result["model"]["status"] == "ready"
            coeff = load_arrays(prepared.ledger, result["model"]["artifacts"][0])["coefficients"]
            assert coeff.shape == (4, 2) and np.isfinite(coeff).all()
