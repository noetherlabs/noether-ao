"""Independent equation, causal-boundary and physical-loop checks; no study cohort."""
from copy import deepcopy
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from noether_optics.artifacts import load_config
from noether_optics.identification import (fit_prediction_pipeline, fit_response,
    identification_telemetry, instrumented_kernel, one_step_prediction, stationary_noise)
from noether_optics.simulation import calibration, loop, predict_fit

ROOT = Path(__file__).resolve().parents[1]


def test_iv_removes_constructed_feedback_bias_for_nondiagonal_response():
    # Orthogonal deterministic columns make the moment equation exact, without
    # relying on a favorable random finite-sample estimate or simulator truth.
    q, _ = np.linalg.qr(np.column_stack((np.ones(80),
                                       np.random.default_rng(21).normal(size=(80, 6)))))
    instrument, disturbance = q[:, 1:3], q[:, 3:5]
    applied = instrument + disturbance
    response = np.array([[0.6, 0.2], [-0.1, 1.3]])
    residual = disturbance - applied @ response + np.array([3., -2.])
    iv = fit_response(residual, applied, instrument, method='iv')
    direct = fit_response(residual, applied, method='ols')
    assert iv['status'] == direct['status'] == 'ready'
    np.testing.assert_allclose(iv['response'], response, atol=2e-14, rtol=0)
    np.testing.assert_allclose(iv['intercept'], [3., -2.], atol=2e-14)
    np.testing.assert_allclose(direct['response'], response-0.5*np.eye(2), atol=2e-14)


def test_iv_matches_independent_two_stage_projection_without_normal_equations():
    rng = np.random.default_rng(22)
    z = rng.normal(size=(200, 4))
    x = z[:, :2] + rng.normal(size=(200, 2))
    y = -x @ np.array([[0.7, -0.2], [0.3, 1.2]]) + rng.normal(size=(200, 2))
    zc, xc, yc = z-z.mean(0), x-x.mean(0), y-y.mean(0)
    first_stage = zc @ np.linalg.lstsq(zc, xc, rcond=1e-10)[0]
    expected = np.linalg.lstsq(first_stage, -yc, rcond=1e-10)[0]
    actual = fit_response(y, x, z, method='iv')
    np.testing.assert_allclose(actual['response'], expected, atol=1e-14)


@pytest.mark.parametrize('scale', [1., 1e-9, 1e9])
def test_identification_relevance_is_invariant_to_physical_unit_scale(scale):
    rng = np.random.default_rng(23)
    z = rng.normal(size=(200, 2))
    x = 0.4*z + rng.normal(size=z.shape)
    y = -x + rng.normal(size=x.shape)
    original = fit_response(y, x, z, method='iv')
    scaled = fit_response(scale*y, scale*x, 17*scale*z, method='iv')
    np.testing.assert_allclose(scaled['response'], original['response'], rtol=1e-13)
    np.testing.assert_allclose(scaled['diagnostics']['canonical_correlations'],
                               original['diagnostics']['canonical_correlations'], rtol=1e-13)


def test_rank_and_relevance_failures_withhold_instead_of_regularizing_a_response():
    rng = np.random.default_rng(24)
    q, _ = np.linalg.qr(np.column_stack((np.ones(80), rng.normal(size=(80, 4)))))
    x, z = q[:, 1:3], q[:, 3:5]
    result = fit_response(-x, x, z, method='iv')
    assert result['status'] == 'withheld' and result['reason'] == 'weak_instruments'
    assert result['response'] is None
    result = fit_response(-x, x, np.zeros_like(x), method='iv')
    assert result['reason'] == 'rank_deficient_instruments'
    result = fit_response(-x, np.ones_like(x), z, method='iv')
    assert result['reason'] == 'rank_deficient_commands'


@pytest.mark.parametrize('damage', ['nan', 'infinity', 'strings', 'short', 'instrument_shape'])
def test_invalid_response_inputs_are_rejected(damage):
    rng = np.random.default_rng(25)
    x = rng.normal(size=(30, 2)); y = -x; z = x.copy()
    if damage == 'nan': y[0, 0] = np.nan
    elif damage == 'infinity': x[0, 0] = np.inf
    elif damage == 'strings': x = x.astype(str)
    elif damage == 'short': x, y, z = x[:2], y[:2], z[:2]
    else: z = z[:-1]
    with pytest.raises(ValueError):
        fit_response(y, x, z, method='iv')


def test_stationary_ar1_start_and_lag_covariances_match_the_declared_law():
    # Independent channels provide repeated trajectories, including stationary t=0.
    noise = stationary_noise(5, 40000, seed=26, rho=0.8, standard_deviation=0.3)
    expected = 0.3**2 * 0.8**np.abs(np.arange(5)[:, None]-np.arange(5)[None, :])
    np.testing.assert_allclose(np.cov(noise), expected, rtol=0.025, atol=0)
    np.testing.assert_array_equal(noise, stationary_noise(5, 40000, seed=26, rho=0.8,
                                                         standard_deviation=0.3))
    for rho in [-1., 1., np.nan, 0.8j, np.array([0.8])]:
        with pytest.raises(ValueError):
            stationary_noise(8, 2, seed=26, rho=rho, standard_deviation=1.)


def training_episodes():
    rng = np.random.default_rng(27)
    return [dict(residual=rng.normal(size=(50, 2))+offset,
                 applied=rng.normal(size=(50, 2)), instrument=rng.normal(size=(50, 2)))
            for offset in [0., 50.]]


def test_nominal_pipeline_reuses_independent_episode_fits_without_boundary_pairs():
    rows = training_episodes()
    model = fit_prediction_pipeline(rows, method='nominal', ridge=0.01)
    expected = np.stack([predict_fit(row['residual']+row['applied'], 1, 0.01) for row in rows])
    np.testing.assert_array_equal(model['episode_coefficients'], expected)
    np.testing.assert_array_equal(model['coefficients'], expected.mean(axis=0))
    joined = predict_fit(np.concatenate([row['residual']+row['applied'] for row in rows]), 1, 0.01)
    assert not np.allclose(model['coefficients'], joined)
    rows[0]['phase_truth'] = np.zeros((50, 2))
    with pytest.raises(ValueError, match='only residual'):
        fit_prediction_pipeline(rows, method='nominal', ridge=0.01)


def test_one_failed_training_response_withholds_the_entire_candidate():
    rows = training_episodes()
    rows[0]['instrument'][:] = 0
    model = fit_prediction_pipeline(rows, method='iv', ridge=0.01)
    assert model['status'] == 'withheld'
    assert len(model['episode_fits']) == 2
    assert model['response'] is None and model['coefficients'] is None


def test_nearly_canceling_response_epochs_do_not_become_an_invertible_calibration():
    rng = np.random.default_rng(20)
    x = rng.normal(size=(200, 1)); d = 0.1*rng.normal(size=(200, 1))
    rows = [dict(residual=d-x, applied=x, instrument=x),
            dict(residual=-d+(1+1e-14)*x, applied=x, instrument=x)]
    model = fit_prediction_pipeline(rows, method='iv', ridge=0.01)
    assert model['status'] == 'withheld'
    assert model['reason'] == 'singular_or_canceling_response'


def test_residual_extraction_does_not_cancel_small_signal_against_large_command():
    issued = np.full((8, 1), 1e16)
    applied = np.concatenate([np.zeros((2, 1)), issued[:-2]])
    outputs = dict(command=issued, applied=applied, probe_requested=np.ones((8, 1)),
                   measurement=np.full((8, 1), 1e-4))
    cal = dict(reconstructor=np.ones((1, 1)), reference=np.zeros(1))
    extracted = identification_telemetry(outputs, cal, delay_frames=2)
    np.testing.assert_array_equal(extracted['residual'], outputs['measurement'])


@pytest.mark.parametrize('delay', [1, 2, 4])
def test_forecast_obeys_after_issue_cutoff_and_correct_matrix_orientation(delay):
    rng = np.random.default_rng(28)
    response = np.array([[0.7, 0.2], [-0.1, 1.1]])
    coefficients = np.array([[0.5, 0.2], [0.4, -0.3]])
    model = dict(status='ready', horizon=1, response=response, coefficients=coefficients)
    issued = rng.normal(size=(16, 2)); applied = rng.normal(size=(16, 2))
    disturbance = np.empty((16, 2)); disturbance[:2] = rng.normal(size=(2, 2))
    for frame in range(2, 16):
        disturbance[frame] = coefficients[:, 0]*disturbance[frame-1] + coefficients[:, 1]*disturbance[frame-2]
    applied[delay:] = issued[:-delay]
    y = (disturbance-applied) @ response
    actual = one_step_prediction(model, y, applied, issued, delay_frames=delay)
    np.testing.assert_allclose(actual['prediction'], y[actual['target_indices']], atol=1e-14)
    altered_y, altered_x, altered_u = y.copy(), applied.copy(), issued.copy()
    altered_y[10:] += 100; altered_x[10:] -= 100; altered_u[10:] += 100
    changed = one_step_prediction(model, altered_y, altered_x, altered_u, delay_frames=delay)
    known = actual['target_indices'] <= 10  # frame 10 is forecast after frame 9 issue
    np.testing.assert_array_equal(actual['prediction'][known], changed['prediction'][known])


@pytest.fixture(scope='module')
def physical():
    cfg = load_config(ROOT/'configs/simulation.yaml')
    cfg.update(n=8, modes=2, sensor='linear', frames=20, burn_in=4, photons=1e6,
               gain=0.3, leak=0.99, command_clip_rad=0.2, delay_frames=2)
    cal = calibration(cfg)
    phase = jnp.zeros((20, 8, 8))
    return cfg, cal, phase


def test_opt_in_zero_inputs_preserve_every_legacy_output(physical):
    cfg, cal, phases = physical
    baseline = jax.jit(lambda p: loop(cfg, cal, p, 6100021, 'integrator', jnp.zeros((2, 2))))(phases)
    out = instrumented_kernel(cfg, cal)(phases, 6100021, np.zeros((20, 2)), np.zeros((20, 2)))
    assert set(out)-set(baseline) == {'feedback_requested', 'probe_requested',
                                     'combined_requested', 'measurement_perturbation'}
    for key in baseline:
        np.testing.assert_array_equal(out[key], baseline[key])


def test_probe_is_clipped_before_delay_and_measurement_noise_is_additive(physical):
    cfg, cal, phases = physical
    probe = np.zeros((20, 2)); probe[5] = [1., -1.]
    noise = np.zeros((20, 2)); noise[10] = [0.05, -0.03]
    execute = instrumented_kernel(cfg, cal)
    out = execute(phases, 6100022, probe, noise)
    unperturbed = execute(phases, 6100022, probe, np.zeros_like(noise))
    np.testing.assert_array_equal(out['combined_requested'], out['feedback_requested']+probe)
    np.testing.assert_array_equal(out['command'], np.clip(out['combined_requested'], -.2, .2))
    np.testing.assert_array_equal(out['applied'][2:], out['command'][:-2])
    np.testing.assert_array_equal(out['measurement'][:10], unperturbed['measurement'][:10])
    np.testing.assert_allclose(out['measurement'][10]-unperturbed['measurement'][10], noise[10], atol=1e-16)
    assert out['saturated'][5]
    telemetry = identification_telemetry(out, cal, delay_frames=2)
    np.testing.assert_array_equal(telemetry['instrument'][7], [1., -1.])
    assert not np.array_equal(telemetry['instrument'][7], telemetry['applied'][7])
    poisoned = dict(out, estimate=np.full((20, 2), np.nan), phase_truth=np.full((20, 8, 8), np.nan))
    for key, value in identification_telemetry(poisoned, cal, delay_frames=2).items():
        np.testing.assert_array_equal(value, telemetry[key])
    damaged = dict(out, applied=np.asarray(out['applied']).copy())
    damaged['applied'][7, 0] += 0.01
    with pytest.raises(ValueError, match='FIFO'):
        identification_telemetry(damaged, cal, delay_frames=2)


def test_collection_state_tracks_previously_issued_probe_and_clipping(physical):
    cfg, cal, phases = physical
    probe = np.full((20, 2), 0.1)
    out = instrumented_kernel(cfg, cal)(phases, 6100023, probe, np.zeros_like(probe))
    residual = np.asarray(out['measurement']) @ np.asarray(cal['reconstructor']).T
    previous = np.concatenate([np.zeros((1, 2)), np.asarray(out['command'])[:-1]])
    np.testing.assert_allclose(out['feedback_requested'], cfg['leak']*previous+cfg['gain']*residual, atol=1e-15)


def test_adapter_rejects_nonfinite_inputs_and_unsupported_physics(physical):
    cfg, cal, phases = physical
    execute = instrumented_kernel(cfg, cal)
    for bad in [np.full((20, 2), np.nan), np.zeros((19, 2)), np.zeros((20, 3))]:
        with pytest.raises(ValueError):
            execute(phases, 6100024, bad, np.zeros((20, 2)))
    with pytest.raises(ValueError):
        instrumented_kernel(cfg, cal, fault='stuck_actuator')
    changed = deepcopy(cfg); changed['sensor'] = 'pyramid'
    with pytest.raises(ValueError):
        instrumented_kernel(changed, cal)
