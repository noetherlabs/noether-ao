"""Measured calibration and cost accounting for separate receiver recordings.

Commissioning and service are independent recordings with separate zero initial
states, not a continuous calibration handover. These helpers neither generate
random streams nor select a controller. A runner must preserve provenance and
serialize/hash the fitted decision before accessing service data. Read-only
array copies below prevent accidental mutation; they are not immutable custody.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log, sqrt
from numbers import Real

import numpy as np
from scipy.stats import t as student_t

from .identification import (fit_response, identification_telemetry,
                             restore_nominal_response)


@dataclass(frozen=True)
class CalibrationWindows:
    """Predeclared frame counts; the probe's complete delayed support is fitted."""

    commissioning_frames: int = 384
    probe_start: int = 128
    probe_stop: int = 256
    service_frames: int = 640
    delay_frames: int = 2

    def __post_init__(self):
        values = (self.commissioning_frames, self.probe_start, self.probe_stop,
                  self.service_frames, self.delay_frames)
        if (any(type(value) is not int for value in values)
                or not 0 <= self.probe_start < self.probe_stop
                or self.service_frames < 1 or self.delay_frames < 1
                or self.fit_stop > self.commissioning_frames):
            raise ValueError('Invalid commissioning, probe, service or delay windows')

    @property
    def fit_start(self):
        return self.probe_start + self.delay_frames

    @property
    def fit_stop(self):
        return self.probe_stop + self.delay_frames

    @property
    def charged_frames(self):
        return self.commissioning_frames - self.probe_start + self.service_frames

    @property
    def total_frames(self):
        return self.commissioning_frames + self.service_frames


DEFAULT_WINDOWS = CalibrationWindows()


def _finite_array(value, shape, name):
    array = np.asarray(value)
    if (array.shape != shape or array.dtype != np.dtype('float64')
            or not np.isfinite(array).all()):
        raise ValueError(f'{name} must be finite float64 with shape {shape}')
    return array


def _positive_scalar(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f'{name} must be a finite positive real scalar')
    value = float(value)
    if not isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be a finite positive real scalar')
    return value


def _readonly(value):
    if isinstance(value, np.ndarray):
        copied = value.copy()
        copied.setflags(write=False)
        return copied
    if isinstance(value, dict):
        return {key: _readonly(item) for key, item in value.items()}
    return value


def fit_commissioning_response(
    measurement, issued, applied, probe_requested, *, nominal_reconstructor,
    nominal_interaction, detector_reference, windows=DEFAULT_WINDOWS,
    maximum_reconstructor_gain=3., minimum_relevance=.05, rcond=1e-10,
):
    """Fit only measured commissioning rows and return a defensive decision copy.

    The default fit uses exactly frames [130,258), selected after full-record
    finite/FIFO checks. Inputs contain no physical plant, phase, receiver value,
    true response or service recording. Nominal H0, R0 and detector reference
    must be fixed before collection. Requested probes must be zero outside the
    declared window; their realization/amplitude and independent seed admission
    remain the runner's responsibility. A withheld fit/restoration returns the
    original reconstructor, retaining its reason and diagnostics.
    """
    if not isinstance(windows, CalibrationWindows):
        raise ValueError('Expected predeclared CalibrationWindows')
    recon = np.asarray(nominal_reconstructor)
    if recon.ndim != 2 or recon.shape[0] < 1 or recon.shape[0] != recon.shape[1]:
        raise ValueError('Full linear modal calibration must be nonempty and square')
    modes = recon.shape[0]
    recon = _finite_array(recon, (modes, modes), 'Nominal reconstructor')
    interaction = _finite_array(nominal_interaction, recon.shape, 'Nominal interaction')
    reference = _finite_array(detector_reference, (modes,), 'Detector reference')
    shape = (windows.commissioning_frames, modes)
    measured = _finite_array(measurement, shape, 'Measurements')
    commands = _finite_array(issued, shape, 'Issued commands')
    actual = _finite_array(applied, shape, 'Applied commands')
    probes = _finite_array(probe_requested, shape, 'Requested probes')
    if np.any(probes[:windows.probe_start] != 0) or np.any(probes[windows.probe_stop:] != 0):
        raise ValueError('Probe requests extend outside the declared window')
    gain_budget = _positive_scalar(maximum_reconstructor_gain, 'Reconstructor gain budget')
    # Validate these before fitting, including paths that withhold before SVD.
    if (isinstance(minimum_relevance, (bool, np.bool_))
            or not isinstance(minimum_relevance, Real)
            or not isfinite(float(minimum_relevance)) or not 0 <= minimum_relevance < 1):
        raise ValueError('Minimum relevance must be finite in [0,1)')
    cutoff = _positive_scalar(rcond, 'rcond')
    if cutoff >= 1:
        raise ValueError('rcond must be below one')
    nominal = {'reconstructor': recon, 'interaction': interaction, 'reference': reference}
    telemetry = identification_telemetry(
        {'measurement': measured, 'command': commands, 'applied': actual,
         'probe_requested': probes}, nominal, delay_frames=windows.delay_frames,
        burn_in=0)
    window = slice(windows.fit_start, windows.fit_stop)
    fit = fit_response(
        telemetry['residual'][window], telemetry['applied'][window],
        telemetry['instrument'][window], method='iv',
        minimum_relevance=float(minimum_relevance), rcond=cutoff)
    restoration = None
    if fit['status'] == 'ready':
        restoration = restore_nominal_response(
            nominal, fit['response'], maximum_reconstructor_gain=gain_budget,
            rcond=cutoff)
    ready = restoration is not None and restoration['status'] == 'ready'
    return _readonly({
        'status': 'ready' if ready else 'withheld',
        'reason': None if ready else (fit['reason'] if restoration is None else restoration['reason']),
        'response': fit['response'],
        'reconstructor': restoration['reconstructor'] if ready else recon,
        'detector_reference': reference,
        'fit': fit, 'restoration': restoration,
        'fit_rows': (windows.fit_start, windows.fit_stop),
    })


def paired_coupling_summary(
    nominal_commissioning, probed_commissioning, nominal_service,
    corrected_service, *, windows=DEFAULT_WINDOWS, dt_s=.001,
):
    """Charge probes, washout and every service frame; never clip optical outcomes.

    Coupling is dimensionless. Sums are coupling-frame utility; multiplying by
    dt gives coupling-time, not energy or received photons. Warm-up must match
    exactly, so its excluded numerator is zero. Both per-charged-frame and
    per-total-recorded-frame means are reported. A withheld correction is still
    represented by the probed commissioning trace and nominal service trace.
    """
    if not isinstance(windows, CalibrationWindows):
        raise ValueError('Expected predeclared CalibrationWindows')
    dt = _positive_scalar(dt_s, 'Frame duration')
    arrays = []
    for value, length, name in (
        (nominal_commissioning, windows.commissioning_frames, 'Nominal commissioning coupling'),
        (probed_commissioning, windows.commissioning_frames, 'Probed commissioning coupling'),
        (nominal_service, windows.service_frames, 'Nominal service coupling'),
        (corrected_service, windows.service_frames, 'Corrected service coupling'),
    ):
        array = _finite_array(value, (length,), name)
        if np.any((array < 0) | (array > 1)):
            raise ValueError(f'{name} must lie in [0,1] without clipping')
        arrays.append(array)
    baseline, probed, service0, service1 = arrays
    if not np.array_equal(baseline[:windows.probe_start], probed[:windows.probe_start]):
        raise ValueError('Uncharged commissioning warm-up must be identical')
    cost = float(np.sum((baseline - probed)[windows.probe_start:]))
    gain = float(np.sum(service1 - service0))
    delta = gain - cost
    time_delta = dt * delta
    if not isfinite(time_delta):
        raise ValueError('Coupling-time utility exceeds the numerical range')
    return {
        'commissioning_cost': cost, 'service_gain': gain, 'charged_delta': delta,
        'charged_mean_delta': delta / windows.charged_frames,
        'total_mean_delta': delta / windows.total_frames,
        'coupling_time_delta_s': time_delta,
        'charged_frames': windows.charged_frames, 'total_frames': windows.total_frames,
    }


def _statistics(values, alpha):
    values = np.asarray(values, dtype=np.float64)
    count = len(values)
    mean = float(np.mean(values))
    standard_error = float(np.std(values, ddof=1) / sqrt(count)) if count > 1 else None
    interval = None
    if standard_error is not None:
        critical = float(student_t.isf(alpha / 2, count - 1))
        if not isfinite(critical):
            raise ValueError('Interval level is outside the numerical range')
        width = critical * standard_error
        interval = [mean - width, mean + width]
    # Independent root blocks, each bounded in [-1,1]. No frame independence.
    width = sqrt(2 * (log(2) - log(alpha)) / count)
    return {
        'mean': mean, 'n_roots': count, 'degrees_of_freedom': count - 1,
        'standard_error': standard_error, 'alpha': alpha,
        'student_t_interval': interval,
        'hoeffding_interval': [max(-1., mean - width), min(1., mean + width)],
    }


def summarize_root_utilities(records, *, root_ids, condition_ids, alpha=.05):
    """Summarize an explicit root-by-condition grid without dropping failures.

    Each record has exactly root_id, condition_id and charged_mean_delta. A
    declared missing outcome is None, not NaN or a substituted zero. Missing or
    duplicate records are structural errors. Unknown outcomes contribute [-1,1]
    to identification bounds; an incomplete contrast gets no complete-case
    point estimate or confidence interval. Conditions share roots and are not
    independent replicates. Student-t intervals are model-dependent small-sample
    summaries; Hoeffding intervals require independent bounded root blocks.
    """
    roots, conditions = tuple(root_ids), tuple(condition_ids)
    for identities in (roots, conditions):
        if (not identities or any(not isinstance(item, str) or not item.strip() for item in identities)
                or len(set(identities)) != len(identities)):
            raise ValueError('Root and condition identities must be nonempty unique strings')
    level = _positive_scalar(alpha, 'alpha')
    if level >= 1 or level / (2 * len(conditions)) == 0:
        raise ValueError('alpha must be representable and below one')
    expected = {(root, condition) for root in roots for condition in conditions}
    cells = {}
    for record in records:
        if not isinstance(record, dict) or set(record) != {'root_id', 'condition_id', 'charged_mean_delta'}:
            raise ValueError('Expected explicit root, condition and utility records')
        if not isinstance(record['root_id'], str) or not isinstance(record['condition_id'], str):
            raise ValueError('Record identities must be strings')
        key = (record['root_id'], record['condition_id'])
        if key not in expected or key in cells:
            raise ValueError('Unknown or duplicate root-condition record')
        value = record['charged_mean_delta']
        if value is not None:
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                    or not isfinite(float(value)) or not -1 <= value <= 1):
                raise ValueError('Utility must be finite in [-1,1], or explicitly None')
            value = float(value)
        cells[key] = value
    if set(cells) != expected:
        raise ValueError('Every declared root-condition requires a record, including failures')

    def bounded_summary(values):
        observed = [value for value in values if value is not None]
        missing = len(values) - len(observed)
        total = sum(observed)
        return {
            'mean_delta': total / len(values) if missing == 0 else None,
            'bounds': [(total - missing) / len(values), (total + missing) / len(values)],
            'missing_outcomes': missing,
        }

    root_values = [dict(root_id=root, **bounded_summary(
        [cells[root, condition] for condition in conditions])) for root in roots]
    complete = all(row['missing_outcomes'] == 0 for row in root_values)
    means = [row['mean_delta'] for row in root_values]
    bounds = [sum(row['bounds'][side] for row in root_values) / len(roots) for side in (0, 1)]
    by_condition = {}
    for condition in conditions:
        values = [cells[root, condition] for root in roots]
        result = bounded_summary(values)
        result['statistics'] = (_statistics(values, level / len(conditions))
                                if result['missing_outcomes'] == 0 else None)
        by_condition[condition] = result
    return {
        'status': 'complete' if complete else 'incomplete',
        'n_roots': len(roots), 'n_conditions': len(conditions),
        'n_missing_cells': sum(value is None for value in cells.values()),
        'root_values': root_values,
        'mean_delta': sum(means) / len(roots) if complete else None,
        'mean_bounds': bounds,
        'primary': _statistics(means, level) if complete else None,
        'condition_statistics': by_condition,
    }
