"""Small physical/causal checks on separate seeds, with no timing acquisition."""
from copy import deepcopy
import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from noether_optics.artifacts import load_config
from noether_optics.atmosphere import phase_sequence
from noether_optics.controller_physics import (controller_kernel, episode_config,
    fit_measured_predictor, measured_telemetry, paired_metrics, paired_open_loop, training_cases)
from noether_optics.controller_protocol import load_study, study_plan
from noether_optics.simulation import calibration, initialize_weights, loop, predict_fit

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def study():
    cfg = load_study(ROOT / "configs/controller_study_v1.yaml")
    cfg["study_id"] = "controller_physics_test_only"
    for role, start in (("training", 41001), ("validation", 42001), ("evaluation", 43001)):
        cfg["seeds"][role] = list(range(start, start + len(cfg["seeds"][role])))
    cfg["seeds"]["initialization"] = 44001
    cfg["training"].update(frames=16, burn_in=4)
    cfg["loop"].update(frames=16, burn_in=4, fault_onset=8, fault_end=16)
    return cfg


@pytest.fixture
def cfg(study):
    base = load_config(ROOT / "configs/simulation.yaml")
    base.update(n=16, modes=8, detector_n=32, screen_factor=2)
    candidate = study_plan(study)["candidates"][0]
    return episode_config(base, study, "validation", 1000., .5, candidate)


@pytest.fixture
def physical(cfg):
    cal = calibration(cfg)
    phase = phase_sequence(cfg, 42001)
    return cal, phase


def test_episode_overrides_preserve_physics_and_do_not_mutate_inputs(study):
    base = load_config(ROOT / "configs/simulation.yaml")
    original_base, original_study = deepcopy(base), deepcopy(study)
    predictive = next(c for c in study_plan(study)["candidates"] if c["controller"] == "predictive")
    cfg = episode_config(base, study, "evaluation", 1000., 2., predictive)
    assert cfg["delay_frames"] == base["delay_frames"]
    assert cfg["gain"] == base["gain"]
    assert cfg["seeds"] == study["seeds"]["evaluation"] and "seed" not in cfg
    assert cfg["training_seeds"] == study["seeds"]["training"]
    assert cfg["frames"] == 16 and cfg["faults"]["onset"] == 8
    pilot = episode_config(base, study, "training", 1e6, .5)
    assert (pilot["gain"], pilot["leak"]) == (.3, .999)
    assert pilot["faults"]["onset"] == pilot["faults"]["end"] == 16
    assert base == original_base and study == original_study


@pytest.mark.parametrize("change", ["stage", "regime", "candidate", "missing_candidate"])
def test_undeclared_episode_settings_rejected(study, change):
    base = load_config(ROOT / "configs/simulation.yaml")
    candidate = study_plan(study)["candidates"][0]
    stage, photons = "validation", 1000.
    if change == "stage": stage = "tuning"
    elif change == "regime": photons = 12345.
    elif change == "candidate": candidate["parameters"]["gain"] = .123
    else: candidate = None
    with pytest.raises(ValueError):
        episode_config(base, study, stage, photons, .5, candidate)


def test_pilot_reconstruction_uses_measurements_and_applied_command_only(cfg, physical):
    cal, phase = physical
    execute = controller_kernel(cfg, cal, "integrator")
    out = execute(phase, 42001)
    measured = measured_telemetry(out, cal)
    np.testing.assert_allclose(measured, out["estimate"], atol=1e-12, rtol=1e-12)
    poisoned = dict(out, estimate=np.full_like(measured, np.nan), phase_truth=np.full_like(phase, np.nan))
    np.testing.assert_array_equal(measured_telemetry(poisoned, cal), measured)
    altered = dict(out, measurement=np.asarray(out["measurement"]) + .01 * np.asarray(cal["interaction"][:, 0]))
    assert not np.array_equal(measured_telemetry(altered, cal), measured)
    later = execute(phase.at[10:].multiply(3), 42001)
    np.testing.assert_allclose(out["measurement"][:10], later["measurement"][:10], atol=1e-12)
    np.testing.assert_allclose(out["command"][:10], later["command"][:10], atol=1e-12)
    np.testing.assert_allclose(out["applied"][2:], out["command"][:-2], atol=1e-12)


def test_pilot_partial_modal_padding_and_invalid_measurements():
    cal = {"reconstructor": np.eye(2), "reference": np.ones(2)}
    out = {"measurement": np.array([[2., 3.], [4., 5.]]), "applied": np.ones((2, 3))}
    np.testing.assert_array_equal(measured_telemetry(out, cal), [[2., 3., 1.], [4., 5., 1.]])
    for bad in (dict(out, measurement=np.ones((3, 2))), dict(out, measurement=np.full((2, 2), np.nan))):
        with pytest.raises(ValueError):
            measured_telemetry(bad, cal)


def episodes(study):
    t = np.arange(study["training"]["frames"], dtype=float)
    # Different offsets/slopes expose any accidental concatenation across episodes.
    return [{"case": case, "telemetry": np.stack((t * (i + 1) + 100 * i, np.sin(t * (i + 1))), axis=1)}
            for i, case in enumerate(training_cases(study))]


def test_predictor_fits_each_burned_in_sequence_and_averages_equally(study):
    data = episodes(study)
    candidate = next(c for c in study_plan(study)["candidates"] if c["controller"] == "predictive")
    fitted = fit_measured_predictor(study, candidate, list(reversed(data)), 2)
    p = candidate["parameters"]
    expected = np.stack([predict_fit(row["telemetry"][4:], p["horizon"], p["ridge"]) for row in data])
    np.testing.assert_array_equal(fitted["episode_coefficients"], expected)
    np.testing.assert_array_equal(fitted["coefficients"], expected.mean(axis=0))
    assert fitted["cases"] == training_cases(study)
    contaminated = predict_fit(np.concatenate([row["telemetry"][4:] for row in data]), p["horizon"], p["ridge"])
    assert not np.allclose(fitted["coefficients"], contaminated)
    for row in data:
        row["telemetry"][:4] = 1e10
    np.testing.assert_array_equal(fit_measured_predictor(study, candidate, data, 2)["coefficients"], fitted["coefficients"])


@pytest.mark.parametrize("damage", ["missing", "duplicate", "heldout", "nonfinite", "short", "truth_field"])
def test_predictor_rejects_incomplete_or_leaking_telemetry(study, damage):
    data = episodes(study)
    candidate = next(c for c in study_plan(study)["candidates"] if c["controller"] == "predictive")
    if damage == "missing": data.pop()
    elif damage == "duplicate": data.append(deepcopy(data[0]))
    elif damage == "heldout": data[0]["case"]["seed"] = study["seeds"]["evaluation"][0]
    elif damage == "nonfinite": data[0]["telemetry"][5, 0] = np.nan
    elif damage == "short": data[0]["telemetry"] = data[0]["telemetry"][:-1]
    else: data[0]["phase_truth"] = np.zeros(2)
    with pytest.raises(ValueError):
        fit_measured_predictor(study, candidate, data, 2)


@pytest.mark.parametrize("fault", ["none", "optical_gain", "vibration"])
def test_faulted_open_loop_matches_zero_dm_plant(cfg, physical, fault):
    cal, phase = physical
    original = np.asarray(phase).copy()
    zero = controller_kernel(cfg | {"gain": 0.}, cal, "integrator", fault)(phase, 42001)
    reference = paired_open_loop(phase, cfg, cal, fault)
    np.testing.assert_allclose(reference["variance"], zero["variance"], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(reference["strehl"], zero["strehl"], atol=1e-12, rtol=1e-12)
    np.testing.assert_array_equal(phase, original)
    clean = paired_open_loop(phase, cfg, cal, "none")
    if fault == "vibration":
        np.testing.assert_array_equal(reference["variance"][:8], clean["variance"][:8])
        assert not np.allclose(reference["variance"][8:], clean["variance"][8:])
    else:
        np.testing.assert_array_equal(reference["variance"], clean["variance"])


def test_adapter_uses_exact_poisson_and_no_fallback(cfg, physical):
    cal, phase = physical
    actual = controller_kernel(cfg, cal, "integrator")(phase, 42001)
    predictor = jnp.zeros((cfg["modes"], 2))
    expected = jax.jit(lambda p: loop(cfg, cal, p, 42001, "integrator", predictor,
                                    fallback=False, differentiable_noise=False))(phase)
    gaussian = jax.jit(lambda p: loop(cfg, cal, p, 42001, "integrator", predictor,
                                    differentiable_noise=True))(phase)
    for key in actual:
        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-12, atol=1e-12)
    assert not np.array_equal(actual["measurement"], gaussian["measurement"])
    assert not np.any(actual["fallback"])


def test_predictive_and_learned_require_frozen_valid_models(cfg, physical):
    cal, phase = physical
    with pytest.raises(ValueError, match="fitted"):
        controller_kernel(cfg, cal, "predictive")
    with pytest.raises(ValueError, match="coefficients"):
        controller_kernel(cfg, cal, "predictive", predictor=np.zeros((1, 2)))
    execute = controller_kernel(cfg, cal, "learned")
    for params in (None, (np.ones(2),), tuple(np.full_like(p, np.nan) for p in initialize_weights(44001))):
        with pytest.raises(ValueError, match="frozen"):
            execute(phase, 42001, params)
    with pytest.raises(ValueError, match="atmosphere"):
        execute(phase[:-1], 42001, initialize_weights(44001))
    with pytest.raises(ValueError, match="uint32"):
        execute(phase, 2**32, initialize_weights(44001))
    with pytest.raises(ValueError, match="pyramid"):
        controller_kernel(cfg | {"sensor": "linear"}, cal, "integrator")


def simple_outputs(cfg):
    n = cfg["frames"]
    return {"variance": np.arange(1, n + 1, dtype=float), "strehl": np.full(n, .5),
            "saturated": np.arange(n) >= n // 2, "all_finite_full_outputs": np.asarray(True)}


def test_raw_metrics_have_exact_windows_units_and_strict_finite_status(cfg):
    closed = simple_outputs(cfg)
    opened = {key: value.copy() for key, value in closed.items()}
    opened["variance"] *= 4
    metrics = paired_metrics(closed, opened, cfg)
    expected = np.sqrt(np.arange(5, 17).mean()) * cfg["wavelength_m"] / (2 * np.pi) * 1e9
    assert metrics["metrics"]["wfe_nm"] == pytest.approx(expected)
    assert metrics["metrics"]["open_wfe_nm"] == pytest.approx(2 * expected)
    assert metrics["post_burn_frames"] == [4, 16]
    assert metrics["fault_window_frames"] == [8, 16]
    assert metrics["metrics"]["saturation_fraction"] == 8 / 12
    assert metrics["fault_window"]["saturation_fraction"] == 1
    closed["all_finite_full_outputs"] = np.asarray(False)
    assert paired_metrics(closed, opened, cfg)["metrics"]["all_outputs_finite"] is False
    closed["variance"][10] = np.nan
    result = paired_metrics(closed, opened, cfg)
    assert result["metrics"]["wfe_nm"] is None
    assert result["fault_window"]["wfe_nm"] is None


@pytest.mark.parametrize("damage", ["missing_flag", "nonboolean_flag", "short", "matrix_metric", "negative_variance", "bad_saturation"])
def test_metric_schema_does_not_hide_missing_or_invalid_raw_data(cfg, damage):
    closed, opened = simple_outputs(cfg), simple_outputs(cfg)
    if damage == "missing_flag": del closed["all_finite_full_outputs"]
    elif damage == "nonboolean_flag": closed["all_finite_full_outputs"] = np.asarray(1)
    elif damage == "short": closed["variance"] = closed["variance"][:-1]
    elif damage == "matrix_metric": closed["variance"] = closed["variance"][:, None]
    elif damage == "negative_variance": closed["variance"][0] = -1
    else: closed["saturated"] = closed["saturated"].astype(float)
    with pytest.raises(ValueError):
        paired_metrics(closed, opened, cfg)


def smoke_module():
    spec = importlib.util.spec_from_file_location("smoke_controller_physics", ROOT / "scripts/smoke_controller_physics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_smoke_config_records_actual_modified_base_hash():
    study, base, text = smoke_module().configuration(load_config(ROOT / "configs/controller_physics_smoke.yaml"))
    assert study["base_config_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    assert base["n"] == 16 and base["modes"] == 8
    assert study["study_id"] == "controller_physics_smoke"


@pytest.mark.parametrize("damage", ["reserved_root", "reserved_derived", "foreign_trial"])
def test_smoke_guard_rejects_reserved_streams_and_foreign_trial(damage):
    cfg = load_config(ROOT / "configs/controller_physics_smoke.yaml")
    if damage == "reserved_root": cfg["training_seeds"][0] = 21001
    elif damage == "reserved_derived": cfg["training_seeds"][0] = 121001
    else: cfg["trial_seed"] = 17
    with pytest.raises(ValueError, match="reserved|validation cohort"):
        smoke_module().configuration(cfg)


def test_ci_cannot_start_physical_smoke_artifact_run(tmp_path):
    out = tmp_path / "must-not-exist"
    result = subprocess.run([sys.executable, str(ROOT / "scripts/smoke_controller_physics.py"), "--out", str(out)],
                            env=dict(os.environ, CI="true"), capture_output=True, text=True)
    assert result.returncode != 0 and "outside CI" in result.stderr
    assert not out.exists()
