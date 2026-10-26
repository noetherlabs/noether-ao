"""Nominal instrument and deployment checks; no research evaluation identities."""
from copy import deepcopy
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from noether_optics.artifacts import load_config
from noether_optics.identification import (
    controller_filtered_instrument, instrumented_kernel, linear_sensor_transform,
    nominal_response, restore_nominal_response,
)
from noether_optics.simulation import calibration

ROOT = Path(__file__).resolve().parents[1]


def replay(probe, response, delay=2, clip=.2):
    return controller_filtered_instrument(probe, response, gain=.3, leak=.99,
                                          delay_frames=delay, command_clip_rad=clip)


@pytest.fixture(scope='module')
def physical():
    cfg = load_config(ROOT/'configs/simulation.yaml')
    cfg.update(n=8, modes=2, sensor='linear', frames=24, burn_in=4,
               photons=1e6, gain=.3, leak=.99, delay_frames=2, command_clip_rad=.2)
    return cfg, calibration(cfg)


@pytest.mark.parametrize('delay', [1, 2, 4])
def test_filtered_probe_has_exact_delay_and_clipped_command_memory(delay):
    probe = np.zeros((24, 2)); probe[0] = [1., -1.]
    response = np.array([[.7, .2], [-.1, 1.1]])
    out = replay(probe, response, delay)
    np.testing.assert_array_equal(out['instrument'][:delay], 0)
    np.testing.assert_array_equal(out['instrument'][delay], [.2, -.2])
    assert out['saturated'][0]
    if delay > 1:
        np.testing.assert_allclose(out['command'][1], [.198, -.198], atol=1e-16)
    previous = np.vstack((np.zeros((1, 2)), out['command'][:-1]))
    np.testing.assert_allclose(out['feedback_requested'],
                               .99*previous-.3*(out['instrument'] @ response), atol=1e-16)
    changed_probe = probe.copy(); changed_probe[10:] = 10
    changed = replay(changed_probe, response, delay)
    np.testing.assert_array_equal(changed['command'][:10], out['command'][:10])
    np.testing.assert_array_equal(changed['instrument'][:10+delay], out['instrument'][:10+delay])


def test_burn_in_does_not_reset_reference_controller_memory():
    probe = np.zeros((24, 2)); probe[0] = [.1, -.1]
    options = dict(gain=0., leak=1., delay_frames=2, command_clip_rad=.2)
    complete = controller_filtered_instrument(probe, np.eye(2), **options)['instrument'][8:]
    restarted = controller_filtered_instrument(probe[8:], np.eye(2), **options)['instrument']
    np.testing.assert_array_equal(complete, np.tile(probe[0], (16, 1)))
    np.testing.assert_array_equal(restarted, 0)


@pytest.mark.parametrize('delay', [1, 2, 4])
def test_reference_replay_matches_noiseless_physical_loop(physical, monkeypatch, delay):
    cfg, cal = physical; cfg = dict(cfg, delay_frames=delay, estimate_clip_rad=1e-9)
    # Test-only zero detector noise, with fresh JIT tracing. Finite photon count
    # keeps all detector score denominators finite; no infinite-flux shortcut.
    monkeypatch.setattr(jax.random, 'normal', lambda key, shape, **kwargs: jnp.zeros(shape))
    probe = np.random.default_rng(6100031).normal(scale=.3, size=(24, 2))
    physical_out = instrumented_kernel(cfg, cal)(np.zeros((24, 8, 8)), 6100031,
                                                probe, np.zeros_like(probe))
    reference = replay(probe, nominal_response(cal), delay)
    for reference_key, physical_key in [('instrument', 'applied'), ('command', 'command'),
                                        ('feedback_requested', 'feedback_requested')]:
        np.testing.assert_allclose(reference[reference_key], physical_out[physical_key], atol=2e-14, rtol=0)
    np.testing.assert_array_equal(reference['saturated'], physical_out['saturated'])
    assert all(np.isfinite(value).all() for value in physical_out.values())


def test_reference_instrument_is_independent_of_realized_sensor_noise(physical):
    cfg, cal = physical
    probe = np.random.default_rng(6100032).normal(scale=.02, size=(24, 2))
    execute = instrumented_kernel(cfg, cal)
    out0 = execute(np.zeros((24, 8, 8)), 6100032, probe, np.zeros_like(probe))
    out1 = execute(np.zeros((24, 8, 8)), 6100033, probe, np.full_like(probe, .1))
    assert not np.array_equal(out0['applied'], out1['applied'])
    np.testing.assert_array_equal(replay(out0['probe_requested'], nominal_response(cal))['instrument'],
                                  replay(out1['probe_requested'], nominal_response(cal))['instrument'])


def test_transform_changes_full_clean_residual_but_not_noise_or_physical_wfe(physical):
    cfg, cal = physical; cfg = dict(cfg, gain=0., leak=0.)
    transform = np.array([[.7, .2], [-.1, .9]])
    mismatched = linear_sensor_transform(cfg, cal, transform)
    t = np.arange(24)
    coefficients = np.column_stack((.3*np.sin(t), .1*np.cos(t)))
    phases = np.einsum('tm,mij->tij', coefficients, np.asarray(cal['basis']))
    probe = np.random.default_rng(6100034).normal(scale=.1, size=(24, 2))
    noise = np.full_like(probe, .03)
    baseline = instrumented_kernel(cfg, cal)(phases, 6100034, probe, noise)
    actual = instrumented_kernel(cfg, mismatched)(phases, 6100034, probe, noise)
    mask = np.asarray(cal['mask'])
    residual = (phases-np.einsum('tm,mij->tij', baseline['applied'], cal['basis']))*mask
    residual -= mask*(residual.sum(axis=(1, 2))/mask.sum())[:, None, None]
    clean = np.stack([np.asarray(cal['plant'](phase)) for phase in residual])
    np.testing.assert_allclose(actual['measurement']-baseline['measurement'],
                               clean @ (transform-np.eye(2)).T, atol=2e-15, rtol=0)
    for key in ('applied', 'command', 'variance', 'strehl', 'measurement_perturbation'):
        np.testing.assert_array_equal(actual[key], baseline[key])
    np.testing.assert_array_equal(mismatched['reconstructor'], cal['reconstructor'])
    np.testing.assert_array_equal(mismatched['interaction'], cal['interaction'])


def test_restoration_uses_nonsymmetric_matrix_order_and_known_calibration_only():
    nominal_h = np.array([[1., .3], [-.2, .8], [.1, .5]])
    nominal_r = np.array([[.7, .1, .2], [-.3, .9, .1]])
    physical_h = nominal_h @ np.array([[.7, .2], [-.1, .9]])
    cal = dict(reconstructor=nominal_r, interaction=nominal_h,
               reference=np.array([.2, -.1, .4]), plant=lambda _: pytest.fail('Truth accessed'))
    target = (nominal_r @ nominal_h).T
    response = (nominal_r @ physical_h).T
    result = restore_nominal_response(cal, response, maximum_reconstructor_gain=3.)
    assert result['status'] == 'ready'
    np.testing.assert_allclose((result['reconstructor'] @ physical_h).T, target, atol=5e-16)
    same = restore_nominal_response(cal, target, maximum_reconstructor_gain=1.)
    assert same['status'] == 'ready'
    np.testing.assert_array_equal(same['reconstructor'], nominal_r)
    np.testing.assert_array_equal(cal['reference'], [.2, -.1, .4])
    wrong_order = nominal_r[:, :2] @ np.linalg.inv(response.T)
    assert not np.allclose(result['reconstructor'][:, :2], wrong_order)


def test_restoration_withholds_rank_loss_and_excess_noise_amplification(physical):
    _, cal = physical
    for response, reason in [(np.zeros((2, 2)), 'singular_response'),
                              (.01*np.eye(2), 'reconstructor_gain_budget')]:
        result = restore_nominal_response(cal, response, maximum_reconstructor_gain=2.)
        assert result['status'] == 'withheld' and result['reason'] == reason
        assert result['reconstructor'] is None


def test_scalar_restoration_equals_conventional_gain_retuning_in_physical_loop(physical):
    cfg, cal = physical
    transform = np.array([[.7, .1], [-.05, .8]])
    mismatched = linear_sensor_transform(cfg, cal, transform)
    restored = restore_nominal_response(cal, .7*nominal_response(cal), maximum_reconstructor_gain=2.)
    assert restored['status'] == 'ready'
    corrected = dict(mismatched, reconstructor=jnp.asarray(restored['reconstructor']))
    phase = np.asarray(cal['basis'])[None, 0]*np.sin(np.arange(24))[:, None, None]
    probe = np.zeros((24, 2)); noise = np.full((24, 2), .03)
    actual = instrumented_kernel(cfg, corrected)(phase, 6100035, probe, noise)
    retuned = instrumented_kernel(dict(cfg, gain=cfg['gain']/.7), mismatched)(
        phase, 6100035, probe, noise)
    for key in ('command', 'applied', 'variance', 'strehl', 'measurement'):
        # Algebraically identical placement of a scalar across matrix products;
        # allow roundoff through 24 frames and spatial reductions in float64.
        np.testing.assert_allclose(actual[key], retuned[key], rtol=0, atol=64*np.finfo(float).eps)
    assert np.any(actual['saturated'])


def test_exact_coupled_response_restores_noiseless_physical_operation(physical, monkeypatch):
    cfg, cal = physical
    monkeypatch.setattr(jax.random, 'normal', lambda key, shape, **kwargs: jnp.zeros(shape))
    transform = np.array([[.6, .2], [-.15, .9]])
    mismatched = linear_sensor_transform(cfg, cal, transform)
    basis, mask = np.asarray(cal['basis']), np.asarray(cal['mask'])
    centered = basis*mask
    centered -= mask*(centered.sum(axis=(1, 2))/mask.sum())[:, None, None]
    # Evaluator-only exact response for this correctness oracle. Learned
    # candidates and their admission rules must never access this construction.
    actual_h = np.stack([np.asarray(mismatched['plant'](mode)) for mode in centered]).T
    response = (np.asarray(cal['reconstructor']) @ actual_h).T
    restored = restore_nominal_response(cal, response, maximum_reconstructor_gain=2.)
    assert restored['status'] == 'ready'
    corrected = dict(mismatched, reconstructor=jnp.asarray(restored['reconstructor']))
    t = np.arange(24)
    phases = np.einsum('tm,mij->tij', np.column_stack((.5*np.sin(t), .3*np.cos(t))), basis)
    zero = np.zeros((24, 2))
    nominal = instrumented_kernel(cfg, cal)(phases, 6100036, zero, zero)
    recovered = instrumented_kernel(cfg, corrected)(phases, 6100036, zero, zero)
    for key in ('command', 'applied', 'variance', 'strehl'):
        np.testing.assert_allclose(recovered[key], nominal[key], rtol=0, atol=2e-14)
    stale = instrumented_kernel(cfg, mismatched)(phases, 6100036, zero, zero)
    assert np.linalg.norm(stale['command']-nominal['command']) > .05


@pytest.mark.parametrize('damage', ['shape', 'nonfinite', 'complex', 'pyramid'])
def test_transform_rejects_unqualified_inputs(physical, damage):
    cfg, cal = physical; cfg = deepcopy(cfg); transform = np.eye(2)
    if damage == 'shape': transform = np.ones((2, 3))
    elif damage == 'nonfinite': transform[0, 0] = np.nan
    elif damage == 'complex': transform = transform.astype(complex)
    else: cfg['sensor'] = 'pyramid'
    with pytest.raises(ValueError):
        linear_sensor_transform(cfg, cal, transform)


def test_reference_parameters_and_overflow_fail_closed():
    probe = np.ones((8, 2)); response = np.eye(2)
    for parameters in [dict(gain=np.nan), dict(leak=1.1), dict(delay_frames=0),
                       dict(command_clip_rad=-1.), dict(gain=1j), dict(gain='0.3')]:
        options = dict(gain=.3, leak=.99, delay_frames=2, command_clip_rad=.2)
        options.update(parameters)
        with pytest.raises(ValueError):
            controller_filtered_instrument(probe, response, **options)
    with pytest.raises(ValueError, match='numerical range'):
        controller_filtered_instrument(probe*1e308, response*1e308, gain=1., leak=1.,
                                       delay_frames=2, command_clip_rad=1e308)
