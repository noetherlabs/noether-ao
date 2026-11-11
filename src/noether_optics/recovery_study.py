"""Prospective scheduled-fallback contract and paired-root statistics.

This module does not acquire trajectories or certify raw artifacts. A caller
must reconstruct all trial outcomes before passing them to ``summarize``.
"""
from __future__ import annotations

from copy import deepcopy
from itertools import product
import math
import statistics

from scipy.stats import t

from .recovery_development import INPUTS, SEEDS as DEVELOPMENT_SEEDS

SEEDS = tuple(range(2026111101, 2026111165))
CONTRAST_ARMS = ('issued_state', 'command_matched')
PHOTONS = (1000., 1000000.)
WINDS = (.5, 2.)
FAULTS = ('none', 'optical_gain', 'vibration')
FAMILY_SIZE = len(PHOTONS) * len(WINDS) * len(FAULTS) * len(CONTRAST_ARMS)
ALPHA = .05


def protocol():
    """Return a detached copy of the complete, fixed statistical contract."""
    return deepcopy({
        'schema': 'noether-recovery-study/1', 'role': 'prospective_evaluation',
        'seeds': list(SEEDS), 'noise_offset': 100000,
        'excluded_development_seeds': list(DEVELOPMENT_SEEDS),
        'photons': list(PHOTONS), 'wind_scales': list(WINDS),
        'faults': list(FAULTS), 'arms': ['continued', *CONTRAST_ARMS],
        'frames': 512, 'burn_in': 128, 'onset': 256, 'end': 512,
        'threshold_multiplier': 1.1, 'consecutive_frames': 32,
        'predictor_id': 'predictive-01', 'target_id': 'integrator-04',
        'frozen_training_inputs': INPUTS,
        'expected_cases': 768, 'expected_terminal_records': 2304,
        'primary': {
            'metric': 'normalized_excess_variance',
            'sign': 'positive_means_intervention_worse',
            'unit': 'dimensionless', 'independent_unit': 'paired_root',
            'family_size': FAMILY_SIZE, 'family_alpha': ALPHA,
            'interval': 'bonferroni_student_t',
            'distribution': 'independent_root_differences; normal approximation',
            'nonfinite_policy': 'withhold_entire_affected_contrast',
            'zero_variance_policy': 'report_point_estimate_without_interval_or_direction',
            'practical_margin': None,
        },
        'secondary': 'descriptive_only_no_additional_superiority_tests',
        'stopping': 'fixed_cohort_no_outcome_dependent_extension_or_early_success',
        'freshness': 'scoped_exclusion_inventory_required_before_acquisition',
        'interpretation': 'Scheduled simulated handovers; no detector, deployment or safety certificate.',
    })


def identities():
    """The independent roots are shared across all conditions and both arms."""
    return tuple(product(SEEDS, PHOTONS, WINDS, FAULTS, CONTRAST_ARMS))


def _interval(values):
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    if sd == 0:
        return {'mean': mean, 'sample_sd': sd, 'interval': None,
                'direction': 'unresolved', 'status': 'zero_observed_variance'}
    radius = float(t.ppf(1 - ALPHA / (2 * FAMILY_SIZE), len(values) - 1)) * sd / math.sqrt(len(values))
    lower, upper = mean - radius, mean + radius
    if not all(math.isfinite(value) for value in (mean, sd, radius, lower, upper)):
        raise ValueError('Finite inputs overflowed the paired-root summary')
    return {
        'mean': mean, 'sample_sd': sd,
        'interval': {'lower': lower, 'upper': upper,
                     'marginal_confidence': 1 - ALPHA / FAMILY_SIZE},
        'direction': ('increased_residual' if lower > 0 else
                      'decreased_residual' if upper < 0 else 'unresolved'),
        'status': 'finite',
    }


def summarize(records):
    """Reduce exactly 1,536 reconstructed pairs without dropping failed roots.

    Each record has exactly ``seed``, ``photons``, ``wind_scale``, ``fault``,
    ``arm``, ``status`` and ``value``. Nonfinite physical pairs have status
    ``nonfinite_pair`` and a null value; they withhold their entire contrast.
    Missing, duplicate or unexpected identities are errors, not failed trials.
    This statistical boundary accepts caller-supplied records and does not
    authenticate them or establish that a physical experiment took place.
    """
    expected = set(identities())
    observed, groups = set(), {}
    fields = {'seed', 'photons', 'wind_scale', 'fault', 'arm', 'status', 'value'}
    for record in records:
        if not isinstance(record, dict) or set(record) != fields:
            raise ValueError('Unexpected paired-record fields')
        if (type(record['seed']) is not int
                or type(record['photons']) not in (int, float)
                or type(record['wind_scale']) not in (int, float)
                or type(record['fault']) is not str or type(record['arm']) is not str):
            raise ValueError('Paired identity requires ordinary finite numeric values')
        identity = tuple(record[name] for name in ('seed', 'photons', 'wind_scale', 'fault', 'arm'))
        if identity not in expected or identity in observed:
            raise ValueError('Unexpected or duplicate paired identity')
        observed.add(identity)
        value = record['value']
        if record['status'] == 'finite_pair':
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError('Finite pair requires a finite real value')
        elif record['status'] != 'nonfinite_pair' or value is not None:
            raise ValueError('Nonfinite pairs require an explicit status and null value')
        groups.setdefault(identity[1:], []).append(record)
    if observed != expected:
        raise ValueError('Complete predeclared cohort required; missing paired identities')
    rows = []
    for key, group in sorted(groups.items()):
        group = sorted(group, key=lambda row: row['seed'])
        failed = [row['seed'] for row in group if row['status'] != 'finite_pair']
        row = dict(zip(('photons', 'wind_scale', 'fault', 'arm'), key))
        row.update(roots=len(group), failed_roots=failed,
                   root_records=[{name: entry[name] for name in ('seed', 'status', 'value')}
                                 for entry in group])
        if failed:
            row.update(status='withheld_nonfinite_pair', mean=None, sample_sd=None,
                       interval=None, direction='unresolved', positive_roots=None,
                       negative_roots=None, zero_roots=None)
        else:
            values = [entry['value'] for entry in group]
            row.update(_interval(values), positive_roots=sum(value > 0 for value in values),
                       negative_roots=sum(value < 0 for value in values),
                       zero_roots=sum(value == 0 for value in values))
        rows.append(row)
    return {'schema': 'noether-recovery-statistics/1', 'protocol': protocol(),
            'contrasts': rows, 'family_size': FAMILY_SIZE,
            'limitations': [
                'Statistical summary of supplied reconstructed pairs; raw-evidence verification is separate.',
                'Bonferroni controls the declared family only if each Student-t interval has its nominal coverage.',
                'The interval is exact under normal independent root differences; otherwise it uses a normal approximation.',
                'Shared roots across conditions do not invalidate Bonferroni; frames and conditions are not additional independent roots.',
                'A nonzero direction establishes no practical instrument benefit or safety guarantee.',
                'All secondary outcomes and harmful-root counts are descriptive.']}
