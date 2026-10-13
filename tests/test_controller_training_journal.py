"""Durable fit correctness on separate synthetic/reduced optical cohorts."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import jax.numpy as jnp
import numpy as np
import pytest

from noether_optics import controller_training as training
from noether_optics import controller_training_journal as journal
from noether_optics.artifacts import load_config
from noether_optics.atmosphere import phase_sequence
from noether_optics.controller_evidence import load_arrays, save_metadata
from noether_optics.controller_physics import episode_config, training_cases
from noether_optics.controller_protocol import load_study, study_plan
from noether_optics.controller_selection import StudyLedger, _bytes, _digest, _read
from noether_optics.simulation import calibration

ROOT = Path(__file__).resolve().parents[1]


def prepare(root, steps=2):
    study = load_study(ROOT / "configs/controller_study_v1.yaml")
    study["study_id"] = "training_journal_correctness_only"
    for role, start in (("training", 91001), ("validation", 92001), ("evaluation", 93001)):
        study["seeds"][role] = list(range(start, start + len(study["seeds"][role])))
    study["seeds"]["initialization"] = 94001
    study["training"].update(frames=10, burn_in=2, learned_steps=steps)
    base = load_config(ROOT / "configs/simulation.yaml")
    base.update(n=8, modes=4, detector_n=16, screen_factor=2)
    source = {"commit": "a" * 40, "fixture": "synthetic provenance, not a study run"}
    runtime = {"fixture": "synthetic runtime"}
    provenance = {"source_commit": source["commit"], "source_sha256": _digest(_bytes(source)),
                  "runtime_sha256": _digest(_bytes(runtime))}
    ledger = StudyLedger.create(root, study, provenance)
    refs = [save_metadata(ledger, f"evidence/provenance/{name}.json", data)
            for name, data in (("source", source), ("runtime", runtime), ("study", study), ("base", base))]
    candidate = next(c for c in study_plan(study)["candidates"] if c["controller"] == "learned")
    pattern = np.sin(np.arange(64).reshape(8, 8))
    episodes = [{"case": case, "phases": np.broadcast_to(
        pattern * case["wind_scale"] * (1 + case["seed"] % 3), (10, 8, 8)).copy()}
        for case in training_cases(study)]
    return SimpleNamespace(ledger=ledger, base=base, candidate=candidate, cal={"mask": jnp.ones((8, 8))},
                           episodes=episodes, refs=refs)


@pytest.fixture
def prepared(tmp_path):
    return prepare(tmp_path / "run")


def fit(p):
    return journal.fit_learned_record(p.ledger, p.base, p.candidate, p.cal, p.episodes, p.refs)


def directory(p):
    return p.ledger.root / f"evidence/training/{p.candidate['id']}"


def backend(monkeypatch, calls, *, interrupt=None, damage=None):
    def build(cfg, cal):
        def execute(params, phases, seed, denominator):
            calls.append(seed)
            if len(calls) == interrupt:
                raise RuntimeError("simulated interruption")
            target = .01 * (seed % 7) + cfg["gain"] * cfg["wind_scale"] + .01 * np.log10(cfg["photons"])
            loss = sum(jnp.mean((p - target)**2) for p in params) / denominator
            gradient = tuple(2 * (p - target) / p.size / denominator for p in params)
            bad = len(calls) == (26 if damage == "final" else 2)
            if bad and damage in ("loss", "final"):
                loss = jnp.asarray(np.nan)
            if bad and damage == "gradient":
                gradient = tuple(jnp.full_like(p, np.nan) for p in params)
            return (loss, jnp.asarray(not (bad and damage == "outputs"))), gradient
        return execute
    monkeypatch.setattr(training, "_episode_kernel", build)


def forbid_backend(monkeypatch):
    forbidden = Mock(side_effect=AssertionError("No optical gradient backend access"))
    monkeypatch.setattr(training, "_episode_kernel", forbidden)
    return forbidden


@pytest.mark.parametrize("interruption", [5, 17, 29])
def test_interrupted_replay_matches_uninterrupted_fit_at_every_stage(prepared, tmp_path, monkeypatch, interruption):
    p = prepared
    calls = []
    backend(monkeypatch, calls, interrupt=interruption)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        fit(p)
    completed = sorted(directory(p).glob("steps/*/*.json"))
    assert len(completed) == interruption - 1
    before = {path: path.read_bytes() for path in directory(p).rglob("*") if path.is_file()}
    interrupted = _read(next(directory(p).glob("interruptions/*.json")))
    assert interrupted["error_type"] == "RuntimeError"
    assert not (directory(p) / "complete.json").exists()
    calls.clear()
    backend(monkeypatch, calls)
    resumed = fit(p)
    assert len(calls) == 36 - len(completed)
    assert all(path.read_bytes() == value for path, value in before.items())
    assert any("/interruptions/" in ref["path"] for ref in resumed["evidence"])

    control = prepare(tmp_path / "control")
    calls.clear()
    complete = fit(control)
    assert len(calls) == 36
    for name in ("parameters.npz", "history.npz", "history.json"):
        assert (directory(p) / name).read_bytes() == (directory(control) / name).read_bytes()
    assert resumed["model"] == complete["model"]
    assert len(list(directory(p).glob("steps/*/*.json"))) == 36


def test_terminal_resume_skips_backend_and_optimizer_and_preserves_bytes(prepared, monkeypatch):
    backend(monkeypatch, [])
    saved = fit(prepared)
    before = {path: path.read_bytes() for path in prepared.ledger.root.rglob("*") if path.is_file()}
    forbidden = forbid_backend(monkeypatch)
    monkeypatch.setattr(training.optax, "adam", forbidden)
    prepared.episodes.reverse()
    assert fit(prepared) == saved
    assert before == {path: path.read_bytes() for path in prepared.ledger.root.rglob("*") if path.is_file()}
    forbidden.assert_not_called()


@pytest.mark.parametrize("damage", ["loss", "gradient", "outputs", "final"])
def test_numerical_failure_is_terminal_with_raw_failure_and_no_usable_parameters(prepared, monkeypatch, damage):
    calls = []
    backend(monkeypatch, calls, damage=damage)
    result = fit(prepared)
    assert result["model"]["status"] == "failed" and "Nonfinite" in result["model"]["error"]
    assert not (directory(prepared) / "parameters.npz").exists()
    assert len(calls) == (26 if damage == "final" else 2)
    raw = [load_arrays(prepared.ledger, _read(path)["raw"]) for path in directory(prepared).glob("steps/*/*.json")]
    if damage in ("loss", "final"):
        assert sum(np.isnan(r["loss"]) for r in raw) == 1
    elif damage == "gradient":
        assert sum(np.isnan(r["gradient_0"]).all() for r in raw) == 1
    else:
        assert sum(not r["outputs_finite"] for r in raw) == 1
    # Strict JSON metadata never substitutes a JSON NaN for retained raw failure.
    assert all(b"NaN" not in path.read_bytes() for path in directory(prepared).rglob("*.json"))
    forbidden = forbid_backend(monkeypatch)
    assert fit(prepared) == result
    forbidden.assert_not_called()


@pytest.mark.parametrize("damage", ["base", "calibration", "phase", "source", "manifest", "step_parameters", "raw", "identity"])
def test_changed_inputs_or_saved_episode_block_resume_before_backend(prepared, monkeypatch, damage):
    p = prepared
    backend(monkeypatch, [], interrupt=5)
    with pytest.raises(RuntimeError): fit(p)
    if damage == "base": p.base["gain"] += .1
    elif damage == "calibration": p.cal["mask"] = jnp.ones((8, 8)) * .5
    elif damage == "phase":
        for episode in p.episodes: episode["phases"] *= 2
    elif damage == "source": (p.ledger.root / p.refs[0]["path"]).write_bytes(b"changed")
    elif damage == "manifest": (p.ledger.root / "manifest.json").write_bytes(b"{}")
    elif damage == "step_parameters": (directory(p) / "steps/0000/parameters.npz").write_bytes(b"changed")
    else:
        meta = next(directory(p).glob("steps/0000/*.json"))
        row = _read(meta)
        if damage == "raw": (p.ledger.root / row["raw"]["path"]).write_bytes(b"changed")
        else:
            row["identity"]["case"]["seed"] += 1
            meta.write_bytes(_bytes(row))
    forbidden = forbid_backend(monkeypatch)
    with pytest.raises(ValueError): fit(p)
    forbidden.assert_not_called()
    assert not (directory(p) / "complete.json").exists()


@pytest.mark.parametrize("damage", ["parameters", "history", "episode", "binding"])
def test_terminal_resume_rechecks_all_saved_artifacts(prepared, monkeypatch, damage):
    backend(monkeypatch, [])
    fit(prepared)
    if damage == "binding":
        path = directory(prepared) / "complete.json"
        value = _read(path)
        value["journal"]["sha256"] = "0" * 64
        path.write_bytes(_bytes(value))
    else:
        name = {"parameters": "parameters.npz", "history": "history.npz"}.get(damage)
        path = directory(prepared) / name if name else next(directory(prepared).glob("steps/*/*.npz"))
        path.write_bytes(b"changed")
    forbidden = forbid_backend(monkeypatch)
    with pytest.raises(ValueError): fit(prepared)
    forbidden.assert_not_called()


@pytest.mark.parametrize("publication", ["episode", "terminal"])
def test_interrupted_publication_retains_raw_and_accepts_only_identical_retry(prepared, monkeypatch, publication):
    real = journal.save_metadata
    def interrupted(ledger, relative, data):
        if ("/steps/" in relative if publication == "episode" else relative.endswith("/complete.json")):
            raise OSError("simulated metadata interruption")
        return real(ledger, relative, data)
    monkeypatch.setattr(journal, "save_metadata", interrupted)
    calls = []
    backend(monkeypatch, calls)
    with pytest.raises(OSError): fit(prepared)
    saved = {p: p.read_bytes() for p in directory(prepared).rglob("*.npz")}
    assert saved and not (directory(prepared) / "complete.json").exists()
    monkeypatch.setattr(journal, "save_metadata", real)
    calls.clear()
    assert fit(prepared)["model"]["status"] == "ready"
    assert len(calls) == (36 if publication == "episode" else 0)
    assert all(p.read_bytes() == value for p, value in saved.items())


@pytest.mark.parametrize("frozen", ["training.json", "selection.json", "validation/prior.json"])
def test_frozen_stage_refuses_fit_before_backend_or_new_evidence(prepared, monkeypatch, frozen):
    path = prepared.ledger.root / frozen
    path.parent.mkdir(exist_ok=True)
    path.write_text("{}")
    forbidden = forbid_backend(monkeypatch)
    with pytest.raises(ValueError, match="frozen"): fit(prepared)
    forbidden.assert_not_called()
    assert not directory(prepared).exists()


@pytest.mark.parametrize("damage", ["missing", "duplicate", "held_out", "foreign_candidate"])
def test_bad_cohort_rejects_before_journal_writes(prepared, monkeypatch, damage):
    if damage == "missing": prepared.episodes.pop()
    elif damage == "duplicate": prepared.episodes.append(deepcopy(prepared.episodes[0]))
    elif damage == "held_out": prepared.episodes[0]["case"]["seed"] = 93001
    else: prepared.candidate["parameters"]["gain"] = .7
    forbidden = forbid_backend(monkeypatch)
    with pytest.raises(ValueError): fit(prepared)
    forbidden.assert_not_called()
    assert not (prepared.ledger.root / "evidence/training").exists()


def test_atmosphere_cannot_change_with_photon_count(prepared, monkeypatch):
    prepared.episodes[-1]["phases"] *= 2
    forbidden = forbid_backend(monkeypatch)
    with pytest.raises(ValueError, match="replace"): fit(prepared)
    forbidden.assert_not_called()
    assert not (directory(prepared) / "journal.json").exists()


def test_candidates_share_atmospheres_and_initialization(prepared, monkeypatch):
    backend(monkeypatch, [])
    first = fit(prepared)
    prior = directory(prepared)
    candidates = [c for c in study_plan(prepared.ledger.config)["candidates"] if c["controller"] == "learned"]
    prepared.candidate = candidates[-1]
    second = fit(prepared)
    a = _read(prior / "journal.json")
    b = _read(directory(prepared) / "journal.json")
    assert a["episodes"] == b["episodes"]
    assert len(list((prepared.ledger.root / "evidence/training/atmospheres").glob("*.npz"))) == 6
    assert (prior / "steps/0000/parameters.npz").read_bytes() == (directory(prepared) / "steps/0000/parameters.npz").read_bytes()
    assert first["model"]["artifacts"][0]["sha256"] != second["model"]["artifacts"][0]["sha256"]


def test_failed_optimizer_state_and_parameters_are_preserved(prepared, monkeypatch):
    backend(monkeypatch, [])
    class BrokenOptimizer:
        def init(self, params): return jnp.asarray(0.)
        def update(self, gradient, state, params):
            return tuple(jnp.full_like(p, np.inf) for p in params), jnp.asarray(np.nan)
    monkeypatch.setattr(training.optax, "adam", lambda rate: BrokenOptimizer())
    result = fit(prepared)
    assert result["model"]["status"] == "failed"
    history = load_arrays(prepared.ledger, result["model"]["artifacts"][0])
    assert np.isnan(history["step_0_failed_optimizer_0"])
    assert np.isinf(history["checkpoint_1_parameter_0"]).all()
    assert not (directory(prepared) / "parameters.npz").exists()


def test_nonfinite_initialization_is_a_retained_terminal_failure(prepared, monkeypatch):
    real = training.initialize_weights
    monkeypatch.setattr(training, "initialize_weights", lambda seed: tuple(jnp.full_like(p, np.nan) for p in real(seed)))
    forbidden = forbid_backend(monkeypatch)
    result = fit(prepared)
    assert result["model"]["status"] == "failed"
    assert "initialization" in result["model"]["error"]
    history = load_arrays(prepared.ledger, result["model"]["artifacts"][0])
    assert np.isnan(history["checkpoint_0_parameter_0"]).all()
    monkeypatch.setattr(training, "initialize_weights", forbidden)
    assert fit(prepared) == result
    forbidden.assert_not_called()


@pytest.mark.parametrize("mutation", ["manifest", "training", "base", "candidate"])
def test_mutation_during_fit_blocks_terminal_model_publication(prepared, monkeypatch, mutation):
    backend(monkeypatch, [])
    real = training._episode_kernel
    def build(cfg, cal):
        kernel = real(cfg, cal)
        def execute(*args):
            result = kernel(*args)
            if mutation == "manifest": (prepared.ledger.root / "manifest.json").write_text("{}")
            elif mutation == "training": (prepared.ledger.root / "training.json").write_text("{}")
            elif mutation == "base": prepared.base["gain"] += .01
            else: prepared.candidate["parameters"]["gain"] += .01
            return result
        return execute
    monkeypatch.setattr(training, "_episode_kernel", build)
    with pytest.raises(ValueError): fit(prepared)
    assert not (directory(prepared) / "complete.json").exists()
    assert not (directory(prepared) / "parameters.npz").exists()
    assert list(directory(prepared).glob("interruptions/*.json"))


def test_fit_holds_ledger_lock_through_backend_execution(prepared, monkeypatch):
    backend(monkeypatch, [])
    real = training._episode_kernel
    def build(cfg, cal):
        kernel = real(cfg, cal)
        def execute(*args):
            with pytest.raises(BlockingIOError):
                with prepared.ledger._locked():
                    raise AssertionError("Competing entry acquired the run lock")
            return kernel(*args)
        return execute
    monkeypatch.setattr(training, "_episode_kernel", build)
    assert fit(prepared)["model"]["status"] == "ready"


def test_reduced_optical_fit_and_terminal_resume(tmp_path, monkeypatch):
    p = prepare(tmp_path / "physical", steps=1)
    cfg = episode_config(p.base, p.ledger.config, "training", 1000., .5, p.candidate)
    p.cal = calibration(cfg)
    phases = {}
    for episode in p.episodes:
        case = episode["case"]
        key = (case["seed"], case["wind_scale"])
        if key not in phases:
            regime = episode_config(p.base, p.ledger.config, "training", case["photons"], case["wind_scale"], p.candidate)
            phases[key] = np.asarray(phase_sequence(regime, case["seed"]))
        episode["phases"] = phases[key]
    real = training._episode_kernel
    calls = []
    def interrupting_build(cfg, cal):
        kernel = real(cfg, cal)
        def execute(*args):
            calls.append(1)
            if len(calls) == 14:
                raise RuntimeError("optical fit interrupted after first Adam update")
            return kernel(*args)
        return execute
    monkeypatch.setattr(training, "_episode_kernel", interrupting_build)
    with pytest.raises(RuntimeError, match="optical fit interrupted"): fit(p)
    assert len(list(directory(p).glob("steps/*/*.json"))) == 13
    before = {path: path.read_bytes() for path in directory(p).rglob("*.npz")}
    result = fit(p)
    assert len(calls) == 25  # Thirteen cached episodes, one interruption, eleven new.
    assert all(path.read_bytes() == data for path, data in before.items())
    assert result["model"]["status"] == "ready"
    history = load_arrays(p.ledger, result["model"]["artifacts"][1])
    assert np.isfinite(history["final_mean_loss"]) and history["step_0_gradient_norm"] > 0
    assert any(not np.array_equal(history[f"checkpoint_0_parameter_{i}"], history[f"checkpoint_1_parameter_{i}"])
               for i in range(4))
    assert len(list(directory(p).glob("steps/*/*.json"))) == 24
    forbidden = forbid_backend(monkeypatch)
    assert fit(p) == result
    forbidden.assert_not_called()
