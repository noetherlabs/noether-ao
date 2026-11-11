"""Statistical boundary checks using synthetic values, never physical trajectories."""
import copy
import math
import random

import numpy as np
import pytest
from scipy.stats import t

from noether_optics import recovery_study as study


def records():
    return [dict(zip(('seed', 'photons', 'wind_scale', 'fault', 'arm'), identity),
                 status='finite_pair', value=.01 + (identity[0] - study.SEEDS[0]) * .0001)
            for identity in study.identities()]


def test_protocol_counts_exposure_separation_and_defensive_copy():
    spec = study.protocol()
    assert len(study.identities()) == 1536
    assert spec['expected_terminal_records'] == 2304
    roots = set(spec['seeds'])
    exposed = set(spec['excluded_development_seeds'])
    assert len(roots) == 64 and not roots & exposed
    assert not (roots | {s + 100000 for s in roots}) & (exposed | {s + 100000 for s in exposed})
    assert all(0 <= seed < 2**32 - 100000 for seed in roots)
    spec['frozen_training_inputs'].clear()
    assert study.protocol()['frozen_training_inputs']


def test_paired_interval_uses_roots_not_frames_or_pooled_conditions():
    result = study.summarize(records())
    assert len(result['contrasts']) == 24
    row = result['contrasts'][0]
    values = .01 + np.arange(64) * .0001
    mean = values.mean()
    radius = t.isf(.05 / 48, 63) * values.std(ddof=1) / math.sqrt(64)
    assert row['mean'] == pytest.approx(mean)
    assert row['interval']['lower'] == pytest.approx(mean - radius)
    assert row['interval']['upper'] == pytest.approx(mean + radius)
    assert row['direction'] == 'increased_residual'
    assert row['positive_roots'] == 64


def test_order_invariance_and_sign_reversal():
    sample = records()
    baseline = study.summarize(sample)
    random.Random(23).shuffle(sample)
    assert study.summarize(sample) == baseline
    for row in sample:
        row['value'] *= -1
    opposite = study.summarize(sample)
    for original, reversed_row in zip(baseline['contrasts'], opposite['contrasts']):
        assert reversed_row['direction'] == 'decreased_residual'
        assert reversed_row['interval']['lower'] == pytest.approx(-original['interval']['upper'])


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'unexpected', 'bool_seed', 'extra_field', 'unhashable_fault'])
def test_incomplete_or_malformed_cohort_is_rejected(mutation):
    sample = records()
    if mutation == 'missing':
        sample.pop()
    elif mutation == 'duplicate':
        sample[-1] = copy.deepcopy(sample[0])
    elif mutation == 'unexpected':
        sample[0]['seed'] = 2026111001
    elif mutation == 'bool_seed':
        sample[0]['seed'] = True
    elif mutation == 'unhashable_fault':
        sample[0]['fault'] = []
    else:
        sample[0]['private_note'] = 'not part of the contract'
    with pytest.raises(ValueError):
        study.summarize(sample)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf'), True, None, '0.1'])
def test_invalid_finite_value_is_not_dropped(value):
    sample = records()
    sample[0]['value'] = value
    with pytest.raises(ValueError, match='finite real'):
        study.summarize(sample)


def test_one_nonfinite_root_withholds_whole_contrast_without_changing_family():
    sample = records()
    baseline = study.summarize(sample)
    sample[0].update(status='nonfinite_pair', value=None)
    result = study.summarize(sample)
    withheld = [row for row in result['contrasts'] if row['failed_roots']]
    assert len(withheld) == 1
    row = withheld[0]
    assert row['roots'] == 64 and row['failed_roots'] == [study.SEEDS[0]]
    assert row['status'] == 'withheld_nonfinite_pair'
    assert row['interval'] is row['mean'] is row['positive_roots'] is None
    assert result['family_size'] == 24
    for before, after in zip(baseline['contrasts'], result['contrasts']):
        if not after['failed_roots']:
            assert after == before


def test_identical_root_values_do_not_claim_certain_direction():
    sample = records()
    for row in sample:
        row['value'] = 1.
    result = study.summarize(sample)
    assert all(row['status'] == 'zero_observed_variance' and row['interval'] is None
               and row['direction'] == 'unresolved' and row['mean'] == 1.
               for row in result['contrasts'])


def test_interval_crossing_zero_is_unresolved():
    sample = records()
    for row in sample:
        row['value'] = -1. if row['seed'] % 2 else 1.
    assert all(row['direction'] == 'unresolved' for row in study.summarize(sample)['contrasts'])
