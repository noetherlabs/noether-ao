"""Transfer mechanics against independent recurrences; no research cohorts."""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from noether_optics.artifacts import load_config
from noether_optics.recovery import scheduled_handover_kernel, validate_handover
from noether_optics.simulation import calibration, loop


@pytest.fixture
def linear(monkeypatch):
    cfg = load_config(Path(__file__).resolve().parents[1] / 'configs/simulation.yaml')
    cfg.update(n=2, modes=1, sensor='linear', frames=18, burn_in=2,
               photons=1e6, gain=.37, leak=.83, delay_frames=2, command_clip_rad=.24)
    cfg['faults'].update(onset=7, end=18)
    basis = jnp.asarray([[[-1., 1.], [-1., 1.]]])
    project = lambda phase: jnp.reshape(jnp.mean(phase * basis[0]), (1,))
    cal = dict(basis=basis, mask=jnp.ones((2, 2)), twin_modes=1,
               plant=project, model=lambda coefficient: coefficient,
               reconstructor=jnp.asarray([[1.3]]), reference=jnp.zeros(1),
               preconditioner=jnp.ones((1, 1)))
    coefficients = np.array([[1.2, -.15]])
    atmosphere = np.array([.1, -.15, .4, .35, -.5, .2, .7, -.3, .45,
                           -.6, .5, .25, -.2, .4, -.45, .3, .2, -.1])
    phases = atmosphere[:, None, None] * np.asarray(basis[0])
    monkeypatch.setattr(jax.random, 'normal', lambda key, shape, **kwargs: jnp.zeros(shape))
    return cfg, cal, coefficients, atmosphere, phases


def scalar_reference(atmosphere, *, delay, clip, trigger, matched):
    """Scalar plant and recurrence; no call into the implementation under test."""
    queue = [0.] * delay
    previous_command = previous_estimate = 0.
    rows = []
    for t, phase in enumerate(atmosphere):
        applied = queue.pop(0)
        measurement = phase - applied
        estimate = applied + 1.3 * measurement
        outgoing = 1.2 * estimate - .15 * previous_estimate
        integrator = .83 * previous_command + .37 * 1.3 * measurement
        target = t >= trigger + int(matched)
        requested = integrator if target else outgoing
        issued = np.clip(requested, -clip, clip)
        rows.append((applied, measurement, outgoing, requested, issued,
                     measurement**2, requested != issued))
        queue.append(issued)
        previous_command, previous_estimate = issued, estimate
    return np.asarray(rows)


@pytest.mark.parametrize('delay,clip', [(1, .24), (2, .24), (4, .24), (2, 10.)])
@pytest.mark.parametrize('mode', ['issued_state', 'command_matched'])
def test_matches_independent_scalar_queue_and_controller(linear, delay, clip, mode):
    cfg, cal, coefficients, atmosphere, phases = linear
    cfg.update(delay_frames=delay, command_clip_rad=clip)
    out = scheduled_handover_kernel(cfg, cal, coefficients, frame=7, mode=mode)(phases, 69000031)
    expected = scalar_reference(atmosphere, delay=delay, clip=clip, trigger=7,
                                matched=mode == 'command_matched')
    for column, key in enumerate(('applied', 'measurement', 'outgoing_requested',
                                  'handover_requested', 'command', 'variance', 'saturated')):
        np.testing.assert_allclose(np.asarray(out[key]).reshape(18), expected[:, column],
                                   rtol=0, atol=2e-15)
    np.testing.assert_array_equal(out['handover_triggered'], np.arange(18) >= 7)
    np.testing.assert_array_equal(out['handover_target_active'],
                                  np.arange(18) >= 7 + (mode == 'command_matched'))
    assert not np.any(out['fallback'])


@pytest.mark.parametrize('delay', [1, 2, 4])
def test_command_matching_is_exactly_one_frame_delayed_transfer(linear, delay):
    cfg, cal, coefficients, _, phases = linear
    cfg['delay_frames'] = delay
    matched = scheduled_handover_kernel(cfg, cal, coefficients, frame=7,
                                         mode='command_matched')(phases, 69000032)
    later = scheduled_handover_kernel(cfg, cal, coefficients, frame=8,
                                       mode='issued_state')(phases, 69000032)
    for key in matched:
        if key != 'handover_triggered':
            np.testing.assert_array_equal(matched[key], later[key])
    ordinary = scheduled_handover_kernel(cfg, cal, coefficients, frame=7,
                                          mode='issued_state')(phases, 69000032)
    baseline = loop(cfg, cal, jnp.asarray(phases), 69000032, 'predictive', jnp.asarray(coefficients))
    for result, extra in ((ordinary, 0), (matched, 1)):
        np.testing.assert_array_equal(result['command'][:7+extra], baseline['command'][:7+extra])
        for key in ('applied', 'measurement', 'variance', 'strehl'):
            np.testing.assert_allclose(result[key][:7+extra+delay],
                                       baseline[key][:7+extra+delay], rtol=0, atol=2e-15)
    assert matched['command'][7, 0] == np.clip(matched['outgoing_requested'][7, 0], -.24, .24)
    assert ordinary['command'][7, 0] != matched['command'][7, 0]


def test_future_phase_cannot_change_transfer_prefix_and_inputs_are_copied(linear):
    cfg, cal, coefficients, _, phases = linear
    execute = scheduled_handover_kernel(cfg, cal, coefficients, frame=7, mode='issued_state')
    out = execute(phases, 69000033)
    cfg['gain'] = 9.; coefficients[:] = 9.
    future = phases.copy(); future[11:] *= -3
    changed = execute(future, 69000033)
    for key in out:
        np.testing.assert_array_equal(changed[key][:11], out[key][:11])
    np.testing.assert_array_equal(changed['applied'][:13], out['applied'][:13])


@pytest.mark.parametrize('frame,mode', [(True, 'issued_state'), (-1, 'issued_state'),
                                       (18, 'issued_state'), (2., 'issued_state'),
                                       (None, 'issued_state'), (2, None), (2, 'reset')])
def test_rejects_undefined_transfer_settings(linear, frame, mode):
    cfg, cal, coefficients, _, _ = linear
    with pytest.raises(ValueError):
        scheduled_handover_kernel(cfg, cal, coefficients, frame=frame, mode=mode)


@pytest.mark.parametrize('name,value', [('gain', -1.), ('gain', np.inf), ('gain', True),
                                      ('leak', 0.), ('leak', 1.01), ('leak', np.nan),
                                      ('command_clip_rad', 0.), ('delay_frames', True),
                                      ('delay_frames', 1.5)])
def test_rejects_invalid_target_dynamics(linear, name, value):
    cfg, _, _, _, _ = linear
    cfg[name] = value
    with pytest.raises(ValueError):
        validate_handover(cfg, 18, 7, 'issued_state')


def test_rejects_mixed_supervisors_and_invalid_dynamic_inputs(linear):
    cfg, cal, coefficients, _, phases = linear
    for options in (dict(fallback=True), dict(command_probe=np.zeros((18, 1))),
                    dict(measurement_perturbation=np.zeros((18, 1))),
                    dict(differentiable_noise=True)):
        with pytest.raises(ValueError, match='uninstrumented predictive'):
            loop(cfg, cal, phases, 69000034, 'predictive', coefficients,
                 handover_frame=7, handover_mode='issued_state', **options)
    with pytest.raises(ValueError, match='uninstrumented predictive'):
        loop(cfg, cal, phases, 69000034, 'integrator', coefficients,
             handover_frame=7, handover_mode='issued_state')
    with pytest.raises(ValueError, match='predictor'):
        scheduled_handover_kernel(cfg, cal, np.full((1, 2), np.nan), frame=7, mode='issued_state')
    execute = scheduled_handover_kernel(cfg, cal, coefficients, frame=7, mode='issued_state')
    for bad in (phases[:-1], np.full_like(phases, np.nan), phases.astype(complex)):
        with pytest.raises(ValueError, match='atmosphere'):
            execute(bad, 69000034)
    for bad in (True, -1, 2**32-100000):
        with pytest.raises(ValueError, match='seed'):
            execute(phases, bad)


def test_boundary_frames_and_nonfinite_outputs_are_not_silently_removed(linear):
    cfg, cal, coefficients, _, phases = linear
    first = scheduled_handover_kernel(cfg, cal, coefficients, frame=0, mode='issued_state')(phases, 69000035)
    assert np.all(first['handover_target_active'])
    last = scheduled_handover_kernel(cfg, cal, coefficients, frame=17, mode='command_matched')(phases, 69000035)
    assert last['handover_triggered'][-1] and not np.any(last['handover_target_active'])
    broken = dict(cal, plant=lambda phase: jnp.full((1,), jnp.nan))
    result = scheduled_handover_kernel(cfg, broken, coefficients, frame=7, mode='issued_state')(phases, 69000035)
    assert result['measurement'].shape == (18, 1)
    assert np.isnan(result['measurement']).all()


def test_pyramid_delay_and_physical_residual_with_real_detector_noise():
    cfg = load_config(Path(__file__).resolve().parents[1] / 'configs/simulation.yaml')
    cfg.update(n=8, modes=2, frames=16, burn_in=2, detector_n=16,
               photons=1e4, gain=.3, leak=.99, delay_frames=2, command_clip_rad=.2)
    cfg['faults'].update(onset=6, end=16)
    cal = calibration(cfg)
    t = np.arange(16)
    phases = np.einsum('tm,mij->tij', np.column_stack((.2*np.sin(t), .1*np.cos(t))), cal['basis'])
    coefficients = np.array([[1.1, -.1], [1.1, -.1]])
    matched = scheduled_handover_kernel(cfg, cal, coefficients, frame=6,
                                         mode='command_matched', fault='optical_gain')(phases, 69000036)
    later = scheduled_handover_kernel(cfg, cal, coefficients, frame=7,
                                       mode='issued_state', fault='optical_gain')(phases, 69000036)
    for key in matched:
        if key != 'handover_triggered':
            np.testing.assert_array_equal(matched[key], later[key])
    applied = np.asarray(matched['applied'])
    np.testing.assert_array_equal(applied[:2], 0)
    np.testing.assert_array_equal(applied[2:], matched['command'][:-2])
    mask = np.asarray(cal['mask'])
    residual = (phases - np.einsum('tm,mij->tij', applied, cal['basis'])) * mask
    residual -= mask * (residual.sum(axis=(1, 2)) / mask.sum())[:, None, None]
    np.testing.assert_allclose(matched['variance'], (residual**2).sum(axis=(1, 2))/mask.sum(), atol=2e-15)
