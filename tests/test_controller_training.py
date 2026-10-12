"""Analytic optimizer and small optical correctness checks, never study outcomes."""
from copy import deepcopy
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from noether_optics import controller_training as training
from noether_optics.artifacts import load_config
from noether_optics.atmosphere import phase_sequence
from noether_optics.controller_physics import episode_config, paired_open_loop, training_cases
from noether_optics.controller_protocol import load_study, study_plan
from noether_optics.simulation import calibration, initialize_weights, loop

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def inputs():
    study = load_study(ROOT / "configs/controller_study_v1.yaml")
    study["study_id"] = "learned_training_correctness_only"
    for role, start in (("training", 61001), ("validation", 62001), ("evaluation", 63001)):
        study["seeds"][role] = list(range(start, start + len(study["seeds"][role])))
    study["seeds"]["initialization"] = 64001
    study["training"].update(frames=10, burn_in=2)
    base = load_config(ROOT / "configs/simulation.yaml")
    base.update(n=8, modes=4, detector_n=16, screen_factor=2)
    candidate = next(c for c in study_plan(study)["candidates"] if c["controller"] == "learned")
    pattern = np.sin(np.arange(64).reshape(8, 8))
    # Unequal open-loop variances expose averaging before episode normalization.
    episodes = [{"case": case, "phases": np.broadcast_to(pattern * (i + 1), (10, 8, 8)).copy()}
                for i, case in enumerate(training_cases(study))]
    return base, study, candidate, {"mask": jnp.ones((8, 8))}, episodes


def analytic_backend(monkeypatch, calls):
    def build(cfg, cal):
        def execute(params, phases, seed, denominator):
            calls.append((seed, tuple(np.asarray(p).copy() for p in params)))
            target = .01 * (seed % 7) + cfg["gain"] * cfg["wind_scale"] + .01 * np.log10(cfg["photons"])
            scale = 1 + seed % 3
            loss = sum(jnp.mean((p - target)**2) for p in params) * scale / denominator
            gradient = tuple(2 * (p - target) / p.size * scale / denominator for p in params)
            return (loss, jnp.asarray(True)), gradient
        return execute
    monkeypatch.setattr(training, "_episode_kernel", build)


def test_fixed_budget_matches_independent_numpy_adam_and_normalization(inputs, monkeypatch):
    base, study, candidate, cal, episodes = inputs
    calls = []
    analytic_backend(monkeypatch, calls)
    fitted = training.fit_learned_controller(base, study, candidate, cal, list(reversed(episodes)))
    assert fitted["status"] == "ready" and fitted["error"] is None
    assert fitted["cases"] == training_cases(study)
    assert len(fitted["steps"]) == 20 and len(fitted["checkpoints"]) == 21
    assert len(calls) == 21 * 12  # Twenty updates, then final scoring (no update).
    initial = tuple(np.asarray(p) for p in initialize_weights(64001))
    parameters = tuple(p.copy() for p in initial)
    first_moment = tuple(np.zeros_like(p) for p in parameters)
    second_moment = tuple(np.zeros_like(p) for p in parameters)
    denominators = [np.mean(np.var(e["phases"], axis=(1, 2))[2:]) for e in episodes]
    np.testing.assert_allclose(fitted["open_variances"], denominators, rtol=1e-14)
    for step, record in enumerate(fitted["steps"]):
        gradients, losses = [], []
        for i, episode in enumerate(episodes):
            case = episode["case"]
            target = .01 * (case["seed"] % 7) + candidate["parameters"]["gain"] * case["wind_scale"] + .01 * np.log10(case["photons"])
            scale = 1 + case["seed"] % 3
            losses.append(scale * sum(np.mean((p - target)**2) for p in parameters) / denominators[i])
            gradients.append(tuple(scale * 2 * (p - target) / p.size / denominators[i] for p in parameters))
            assert calls[step * 12 + i][0] == case["seed"]
            for observed, expected in zip(calls[step * 12 + i][1], parameters):
                np.testing.assert_allclose(observed, expected, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(record["episode_losses"], losses, rtol=1e-12)
        assert record["mean_loss"] == pytest.approx(np.mean(losses), rel=1e-12)
        gradient = tuple(np.mean(np.stack(g), axis=0) for g in zip(*gradients))
        for actual, expected in zip(record["mean_gradient"], gradient):
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)
        first_moment = tuple(.9 * m + .1 * g for m, g in zip(first_moment, gradient))
        second_moment = tuple(.999 * v + .001 * g**2 for v, g in zip(second_moment, gradient))
        parameters = tuple(p - .02 * (m / (1 - .9**(step + 1))) /
                           (np.sqrt(v / (1 - .999**(step + 1))) + 1e-8)
                           for p, m, v in zip(parameters, first_moment, second_moment))
        for actual, expected in zip(fitted["checkpoints"][step + 1], parameters):
            np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-13)
    assert fitted["final"]["mean_loss"] != fitted["steps"][-1]["mean_loss"]
    for actual, expected in zip(fitted["parameters"], parameters):
        np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-13)
    assert all(np.array_equal(a, b) for a, b in zip(fitted["checkpoints"][0], initial))


def test_all_six_candidates_restart_identically_and_do_not_mutate_inputs(inputs, monkeypatch):
    base, study, _, cal, episodes = inputs
    original_base, original_study, original_episodes = deepcopy(base), deepcopy(study), deepcopy(episodes)
    calls = []
    analytic_backend(monkeypatch, calls)
    candidates = [c for c in study_plan(study)["candidates"] if c["controller"] == "learned"]
    fits = [training.fit_learned_controller(base, study, c, cal, episodes) for c in candidates]
    for fit in fits:
        assert fit["status"] == "ready" and len(fit["steps"]) == 20
        for actual, expected in zip(fit["checkpoints"][0], fits[0]["checkpoints"][0]):
            np.testing.assert_array_equal(actual, expected)
    again = training.fit_learned_controller(base, study, candidates[0], cal, list(reversed(episodes)))
    for actual, expected in zip(again["parameters"], fits[0]["parameters"]):
        np.testing.assert_array_equal(actual, expected)
    assert not np.array_equal(fits[0]["parameters"][0], fits[-1]["parameters"][0])
    assert base == original_base and study == original_study
    for actual, expected in zip(episodes, original_episodes):
        assert actual["case"] == expected["case"]
        np.testing.assert_array_equal(actual["phases"], expected["phases"])


@pytest.mark.parametrize("damage", ["missing", "duplicate", "heldout", "coerced_identity",
    "short", "nonfinite", "complex", "extra_field", "wrong_controller", "foreign_candidate",
    "zero_reference", "bad_reference", "linear_sensor"])
def test_bad_inputs_reject_before_initialization_or_gradient(inputs, monkeypatch, damage):
    base, study, candidate, cal, episodes = inputs
    if damage == "missing": episodes.pop()
    elif damage == "duplicate": episodes.append(deepcopy(episodes[0]))
    elif damage == "heldout": episodes[0]["case"]["seed"] = study["seeds"]["evaluation"][0]
    elif damage == "coerced_identity": episodes[0]["case"]["seed"] = float(episodes[0]["case"]["seed"])
    elif damage == "short": episodes[0]["phases"] = episodes[0]["phases"][:-1]
    elif damage == "nonfinite": episodes[0]["phases"][0, 0, 0] = np.nan
    elif damage == "complex": episodes[0]["phases"] = episodes[0]["phases"].astype(complex)
    elif damage == "extra_field": episodes[0]["telemetry"] = np.zeros(10)
    elif damage == "wrong_controller": candidate = study_plan(study)["candidates"][0]
    elif damage == "foreign_candidate": candidate["parameters"]["gain"] = .123
    elif damage == "zero_reference": episodes[0]["phases"][:] = 0
    elif damage == "bad_reference": cal["mask"] = jnp.full((8, 8), np.nan)
    elif damage == "linear_sensor": base["sensor"] = "linear"
    def forbidden(*args, **kwargs):
        raise AssertionError("Must reject before accessing training backend")
    monkeypatch.setattr(training, "initialize_weights", forbidden)
    monkeypatch.setattr(training, "_episode_kernel", forbidden)
    with pytest.raises(ValueError):
        training.fit_learned_controller(base, study, candidate, cal, episodes)


@pytest.mark.parametrize("damage", ["loss", "gradient", "outputs", "final"])
def test_numerical_failure_keeps_partial_evidence_without_usable_model(inputs, monkeypatch, damage):
    base, study, candidate, cal, episodes = inputs
    study["training"]["learned_steps"] = 1
    counter = 0
    def build(cfg, cal):
        def execute(params, phases, seed, denominator):
            nonlocal counter
            counter += 1
            fail = counter == (14 if damage == "final" else 2)
            loss = jnp.asarray(np.nan if fail and damage in ("loss", "final") else 1.)
            gradient = tuple(jnp.full_like(p, np.nan if fail and damage == "gradient" else .1) for p in params)
            return (loss, jnp.asarray(not (fail and damage == "outputs"))), gradient
        return execute
    monkeypatch.setattr(training, "_episode_kernel", build)
    result = training.fit_learned_controller(base, study, candidate, cal, episodes)
    assert result["status"] == "failed" and result["parameters"] is None
    assert "Nonfinite" in result["error"]
    record = result["final"] if damage == "final" else result["steps"][0]
    assert len(record["episode_losses"]) == len(record["episode_gradients"]) == 2
    assert counter == (14 if damage == "final" else 2)
    assert len(result["checkpoints"]) == (2 if damage == "final" else 1)
    if damage in ("loss", "final"): assert np.isnan(record["episode_losses"][-1])
    if damage == "gradient": assert np.isnan(record["episode_gradients"][-1][0]).all()
    if damage == "outputs": assert record["episode_outputs_finite"] == [True, False]


@pytest.mark.parametrize("damage", ["state", "parameters"])
def test_nonfinite_optimizer_retains_failed_checkpoint_and_state(inputs, monkeypatch, damage):
    base, study, candidate, cal, episodes = inputs
    analytic_backend(monkeypatch, [])
    class BrokenOptimizer:
        def init(self, params): return jnp.asarray(0.)
        def update(self, gradient, state, params):
            return tuple(jnp.full_like(p, np.inf if damage == "parameters" else .1) for p in params), jnp.asarray(np.nan)
    monkeypatch.setattr(training.optax, "adam", lambda rate: BrokenOptimizer())
    result = training.fit_learned_controller(base, study, candidate, cal, episodes)
    assert result["status"] == "failed" and result["parameters"] is None
    assert len(result["steps"]) == 1 and len(result["checkpoints"]) == 2
    assert np.isnan(result["steps"][0]["failed_optimizer_state"])
    if damage == "parameters": assert np.isinf(result["checkpoints"][-1][0]).all()


def test_unexpected_backend_exception_is_not_relabeled_scientific_failure(inputs, monkeypatch):
    base, study, candidate, cal, episodes = inputs
    def broken(*args): raise RuntimeError("backend implementation error")
    monkeypatch.setattr(training, "_episode_kernel", lambda *args: broken)
    with pytest.raises(RuntimeError, match="backend implementation error"):
        training.fit_learned_controller(base, study, candidate, cal, episodes)


def test_optical_objective_matches_gaussian_plant_and_finite_difference(inputs):
    base, study, candidate, _, _ = inputs
    cfg = episode_config(base, study, "training", 1000., .5, candidate)
    cal = calibration(cfg)
    phase = phase_sequence(cfg, 61001)
    params = initialize_weights(64001)
    denominator = paired_open_loop(phase, cfg, cal, "none")["variance"][2:].mean()
    (loss, finite), gradient = training._episode_kernel(cfg, cal)(params, phase, 61001, denominator)
    assert bool(finite) and np.isfinite(float(loss)) and training._finite(gradient)
    def direct(parameters, gaussian=True):
        return loop(cfg, cal, phase, 61001, "learned", jnp.zeros((4, 2)),
                    params=parameters, fault="none", fallback=False,
                    differentiable_noise=gaussian)["variance"][2:].mean() / denominator
    raw = jax.jit(direct)
    np.testing.assert_allclose(loss, raw(params), rtol=1e-12)
    assert not np.isclose(float(loss), float(jax.jit(lambda p: direct(p, False))(params)), rtol=1e-5)
    epsilon = 1e-5
    plus = params[:-1] + (params[-1] + epsilon,)
    minus = params[:-1] + (params[-1] - epsilon,)
    difference = float((raw(plus) - raw(minus)) / (2 * epsilon))
    assert abs(difference) > 1e-9
    np.testing.assert_allclose(gradient[-1][0], difference, rtol=2e-5, atol=1e-8)


def test_small_physical_fit_has_finite_changed_weights_and_complete_history(inputs):
    base, study, candidate, _, _ = inputs
    study["training"]["learned_steps"] = 1  # Wiring test, not a scientific training sweep.
    cfg = episode_config(base, study, "training", 1000., .5, candidate)
    cal = calibration(cfg)
    phases = {}
    episodes = []
    for case in training_cases(study):
        key = (case["seed"], case["wind_scale"])
        if key not in phases:
            regime = episode_config(base, study, "training", case["photons"], case["wind_scale"], candidate)
            phases[key] = np.asarray(phase_sequence(regime, case["seed"]))
        episodes.append({"case": case, "phases": phases[key]})
    result = training.fit_learned_controller(base, study, candidate, cal, episodes)
    assert result["status"] == "ready" and result["error"] is None
    assert len(result["steps"]) == 1 and len(result["checkpoints"]) == 2
    assert len(result["steps"][0]["episode_losses"]) == len(result["final"]["episode_losses"]) == 12
    assert all(result["steps"][0]["episode_outputs_finite"])
    assert all(result["final"]["episode_outputs_finite"])
    assert result["steps"][0]["gradient_norm"] > 0
    assert any(not np.array_equal(a, b) for a, b in zip(result["parameters"], result["checkpoints"][0]))


def test_nonfinite_initialization_is_retained_before_backend_access(inputs, monkeypatch):
    base, study, candidate, cal, episodes = inputs
    bad = tuple(jnp.full_like(p, np.nan) for p in initialize_weights(64001))
    monkeypatch.setattr(training, "initialize_weights", lambda seed: bad)
    def forbidden(*args): raise AssertionError("No gradient kernel for invalid initialization")
    monkeypatch.setattr(training, "_episode_kernel", forbidden)
    result = training.fit_learned_controller(base, study, candidate, cal, episodes)
    assert result["status"] == "failed" and result["parameters"] is None
    assert result["steps"] == [] and result["final"] is None
    assert np.isnan(result["checkpoints"][0][0]).all()


def test_mean_gradient_overflow_fails_without_update(inputs, monkeypatch):
    base, study, candidate, cal, episodes = inputs
    def build(*args):
        return lambda params, *unused: ((jnp.asarray(1.), jnp.asarray(True)),
                                        tuple(jnp.full_like(p, 1e308) for p in params))
    monkeypatch.setattr(training, "_episode_kernel", build)
    with np.errstate(over="ignore"):
        result = training.fit_learned_controller(base, study, candidate, cal, episodes)
    assert result["status"] == "failed" and result["parameters"] is None
    assert "mean loss or gradient" in result["error"]
    assert len(result["checkpoints"]) == 1
    assert len(result["steps"][0]["episode_gradients"]) == 12


def test_optical_kernel_checks_outputs_even_before_burn_in(inputs, monkeypatch):
    base, study, candidate, cal, _ = inputs
    cfg = episode_config(base, study, "training", 1000., .5, candidate)
    def poisoned_loop(config, calibration, phases, seed, controller, predictor, **kwargs):
        assert controller == "learned" and kwargs["differentiable_noise"] is True
        assert kwargs["fault"] == "none" and kwargs["fallback"] is False
        value = jnp.sum(kwargs["params"][0]**2)
        return {"variance": jnp.full(10, value), "measurement": jnp.ones((10, 3)).at[0, 0].set(jnp.nan)}
    monkeypatch.setattr(training, "loop", poisoned_loop)
    (loss, finite), gradient = training._episode_kernel(cfg, cal)(initialize_weights(64001), jnp.zeros((10, 8, 8)), 61001, 1.)
    assert np.isfinite(loss) and training._finite(gradient)
    assert not bool(finite)
