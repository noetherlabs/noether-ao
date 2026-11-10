"""Analytic paired histories: event cost, causal prefix and censored recovery."""
from copy import deepcopy

import numpy as np
import pytest

from noether_optics.recovery_metrics import paired_recovery_outcomes, sustained_return


def fixture():
    config = dict(frames=12, modes=2, burn_in=2, faults=dict(onset=4, end=12),
                  delay_frames=2, command_clip_rad=4., dt_s=.25, wavelength_m=750e-9)
    continued = {key: np.zeros(12, dtype=bool) for key in
                 ('alarm', 'fallback', 'unavailable', 'saturated')}
    continued.update(variance=np.full(12, 4.), strehl=np.full(12, .5), score=np.zeros(12))
    for key in ('command', 'applied', 'estimate', 'measurement', 'prediction'):
        continued[key] = np.zeros((12, 2))
    intervened = deepcopy(continued)
    intervened['variance'][6:] = [9, 1, 1, 1, 5, 1]
    intervened['strehl'][6:] = [.1, .8, .8, .8, .4, .8]
    intervened['command'][4:7, 0] = [1, 4, 2]
    intervened['applied'][2:] = intervened['command'][:-2]
    intervened['handover_requested'] = intervened['command'].copy()
    intervened['handover_requested'][5, 0] = 5
    intervened['saturated'][5] = True
    reference = dict(variance=np.full(12, 10.), strehl=np.full(12, .1),
                     saturated=np.zeros(12, dtype=bool))
    return continued, intervened, reference, config


def evaluate(*args, **kwargs):
    settings = dict(effect_frame=6, threshold_multiplier=1.1, consecutive=3)
    settings.update(kwargs)
    return paired_recovery_outcomes(*args, **settings)


def test_analytic_horizon_cost_command_rates_and_relapse():
    pair = evaluate(*fixture())
    assert pair['status'] == 'finite_pair'
    outcomes = pair['outcomes']
    assert outcomes['integrated_excess_variance_rad2_s'] == -1.5
    assert outcomes['zero_dm_integral_rad2_s'] == 20
    assert outcomes['normalized_excess_variance'] == -.075
    arm = outcomes['arms']['intervened']
    assert arm['peak_variance_rad2'] == 9
    assert arm['integrated_variance_rad2_s'] == 6.5
    assert arm['wfe_nm'] == pytest.approx(np.sqrt(3.25) * 750 / (2*np.pi))
    assert arm['strehl'] == pytest.approx((1+.1+3*.8+.4+.8)/8)
    assert arm['saturation_fraction'] == 1/8
    assert arm['commands']['command']['peak_step_by_mode_rad'] == [3, 0]
    assert arm['commands']['applied']['peak_rate_by_mode_rad_s'] == [12, 0]
    assert arm['commands']['handover_requested']['peak_step_by_mode_rad'] == [4, 0]
    assert outcomes['arms']['continued']['commands']['handover_requested'] is None
    recovery = arm['reference_return']
    assert recovery['first_frame'] == 7
    assert recovery['confirmation_frame'] == 9
    assert recovery['onset_to_first_s'] == .75
    assert recovery['onset_to_confirmation_s'] == 1.25
    assert recovery['post_confirmation_above_fraction'] == .5
    assert recovery['post_confirmation_frames'] == 2
    assert not recovery['right_censored']


def test_same_event_horizon_for_later_effect_and_initial_jump_included():
    a, b, ref, cfg = fixture()
    # Make both physical prefixes agree through frame 6 to permit later effect.
    b['variance'][6] = a['variance'][6]
    b['strehl'][6] = a['strehl'][6]
    b['command'][:] = 0
    b['command'][5:] = 2
    b['applied'][2:] = b['command'][:-2]
    b['handover_requested'] = b['command'].copy()
    b['saturated'][:] = False
    late = evaluate(a, b, ref, cfg, effect_frame=7)
    assert late['horizon_frames'] == [4, 12]
    # Include frame 4's jump from frame 3, not merely differences inside [4,12).
    a, b, ref, cfg = fixture()
    b['command'][:] = 0
    b['command'][4:] = 3
    b['applied'][2:] = b['command'][:-2]
    b['handover_requested'] = b['command'].copy()
    b['saturated'][:] = False
    result = evaluate(a, b, ref, cfg)
    assert result['outcomes']['arms']['intervened']['commands']['command'][
        'peak_step_by_mode_rad'] == [3, 3]


@pytest.mark.parametrize('values,consecutive,first,confirmation,censored', [
    ([9, 1, 1, 1], 3, 1, 3, False),
    ([9, 9, 1, 1], 3, None, None, True),
    ([1, 1, 9, 1, 1], 3, None, None, True),
    ([1, 1, 1], 4, None, None, True),
    ([1, 1, 1], 1, 0, 0, False),
])
def test_sustained_return_boundaries(values, consecutive, first, confirmation, censored):
    outcome = sustained_return(values, start=0, stop=len(values), threshold=1.,
                               consecutive=consecutive)
    assert outcome['first_frame'] == first
    assert outcome['confirmation_frame'] == confirmation
    assert outcome['right_censored'] == censored
    if confirmation == len(values)-1:
        assert outcome['post_confirmation_frames'] == 0
        assert outcome['post_confirmation_above_fraction'] is None


@pytest.mark.parametrize('key,index', [('variance', 5), ('strehl', 3), ('applied', 4)])
def test_past_physical_changes_rejected(key, index):
    a, b, ref, cfg = fixture()
    b[key][index] += .1
    with pytest.raises(ValueError, match='histories differ|FIFO'):
        evaluate(a, b, ref, cfg)


@pytest.mark.parametrize('key', ['prediction', 'measurement', 'handover_requested', 'variance'])
def test_nonfinite_output_is_not_a_survivor_subset(key):
    a, b, ref, cfg = fixture()
    b[key][-1] = np.nan
    result = evaluate(a, b, ref, cfg)
    assert result['status'] == 'nonfinite_outputs'
    assert result['outcomes'] is None
    assert np.isnan(b[key][-1]).all()  # Caller retains the original evidence.


def test_nonfinite_reference_and_zero_normalizer_withhold_outcomes():
    a, b, ref, cfg = fixture()
    ref['variance'][-1] = np.inf
    assert evaluate(a, b, ref, cfg)['status'] == 'nonfinite_outputs'
    ref['variance'][:] = 0
    result = evaluate(a, b, ref, cfg)
    assert result['status'] == 'invalid_zero_dm_normalizer'
    assert result['outcomes'] is None


def test_missing_bulk_stream_cannot_be_replaced_by_a_boolean():
    a, b, ref, cfg = fixture()
    del b['measurement']
    b['all_finite_full_outputs'] = np.asarray(True)
    with pytest.raises(ValueError, match='complete physical'):
        evaluate(a, b, ref, cfg)


def test_wrong_fifo_and_clipped_command_rejected():
    a, b, ref, cfg = fixture()
    b['applied'][9] = 1
    with pytest.raises(ValueError, match='FIFO'):
        evaluate(a, b, ref, cfg)
    a, b, ref, cfg = fixture()
    b['command'][-1] = 5
    with pytest.raises(ValueError, match='modal clip'):
        evaluate(a, b, ref, cfg)


@pytest.mark.parametrize('corruption', ['request', 'saturated'])
def test_requested_command_and_saturation_must_match(corruption):
    a, b, ref, cfg = fixture()
    if corruption == 'request':
        b['handover_requested'][-1] = 1
    else:
        b['saturated'][-1] = True
    with pytest.raises(ValueError, match='Requested command|Saturation trace'):
        evaluate(a, b, ref, cfg)


@pytest.mark.parametrize('key,value', [('frames', True), ('delay_frames', 0),
                                      ('dt_s', 0), ('wavelength_m', float('nan'))])
def test_invalid_config_rejected(key, value):
    a, b, ref, cfg = fixture()
    cfg[key] = value
    with pytest.raises(ValueError):
        evaluate(a, b, ref, cfg)


@pytest.mark.parametrize('kwargs', [dict(effect_frame=True), dict(effect_frame=12),
                                   dict(threshold_multiplier=0), dict(consecutive=0)])
def test_invalid_recovery_definition_rejected(kwargs):
    with pytest.raises(ValueError):
        evaluate(*fixture(), **kwargs)


def test_finite_raw_overflow_is_reported_without_json_infinity():
    a, b, ref, cfg = fixture()
    a['variance'][:] = 1e307
    b['variance'][:] = 1e307
    with np.errstate(over='ignore', invalid='ignore'):
        result = evaluate(a, b, ref, cfg, threshold_multiplier=1e300)
    assert result['status'] == 'nonfinite_reduction'
    assert result['outcomes'] is None


def test_input_arrays_and_configuration_are_not_mutated():
    args = fixture()
    before = deepcopy(args)
    evaluate(*args)
    for current, old in zip(args[:3], before[:3]):
        assert current.keys() == old.keys()
        for key in current:
            np.testing.assert_array_equal(current[key], old[key])
    assert args[3] == before[3]
