"""Fixed-horizon outcomes for one paired controller intervention.

These reductions use complete saved streams, not independent frame samples.
The caller must bind the shared atmosphere, calibration, noise keys and source;
matching prefixes alone cannot establish that experimental provenance.
"""
from __future__ import annotations

from numbers import Real

import numpy as np

from .controller_physics import FULL_OUTPUT_KEYS, paired_metrics


def sustained_return(variance, *, start: int, stop: int, threshold: float,
                     consecutive: int) -> dict:
    """First complete below-threshold run in [start, stop), retaining relapse.

    A run ending at stop-1 counts. A shorter terminal run is right-censored.
    Confirmation occurs at the last frame of the first qualifying run.
    """
    values = np.asarray(variance)
    if (values.ndim != 1 or values.dtype.kind not in 'fiu'
            or not np.isfinite(values).all() or np.any(values < 0)):
        raise ValueError('Expected a finite nonnegative variance stream')
    if (any(type(x) is not int for x in (start, stop, consecutive))
            or not 0 <= start < stop <= len(values) or consecutive < 1
            or isinstance(threshold, (bool, np.bool_)) or not isinstance(threshold, Real)
            or not np.isfinite(threshold) or threshold < 0):
        raise ValueError('Invalid sustained-return window or threshold')
    below = values[start:stop] <= threshold
    count = 0
    first = None
    for offset, good in enumerate(below):
        count = count + 1 if good else 0
        if count == consecutive:
            first = start + offset - consecutive + 1
            break
    confirmation = None if first is None else first + consecutive - 1
    after = None if confirmation is None else values[confirmation + 1:stop]
    return {
        'first_frame': first,
        'confirmation_frame': confirmation,
        'right_censored': first is None,
        'censor_frame_exclusive': stop,
        'search_start_frame': start,
        'consecutive_frames': consecutive,
        'threshold_rad2': float(threshold),
        'post_confirmation_frames': None if after is None else len(after),
        'post_confirmation_above_fraction': (
            None if after is None or not len(after) else float(np.mean(after > threshold))),
    }


def _commands(outputs, config):
    frames, modes = config['frames'], config['modes']
    command, applied = (np.asarray(outputs[key]) for key in ('command', 'applied'))
    if any(x.shape != (frames, modes) or x.dtype.kind not in 'fiu'
           for x in (command, applied)):
        raise ValueError('Incomplete modal command history')
    # Nonfinite output is a retained numerical failure, not invalid schema.
    if not np.isfinite(command).all() or not np.isfinite(applied).all():
        return
    expected = np.zeros_like(command)
    delay = config['delay_frames']
    if delay < frames:
        expected[delay:] = command[:-delay]
    if not np.array_equal(applied, expected):
        raise ValueError('Applied command history violates the declared FIFO delay')
    if np.any(np.abs(command) > config['command_clip_rad']):
        raise ValueError('Issued command exceeds the modal clip')


def paired_recovery_outcomes(continued: dict, intervened: dict, reference: dict,
                             config: dict, *, effect_frame: int,
                             threshold_multiplier: float, consecutive: int) -> dict:
    """Compare one intervention to continuation over the same event horizon.

    Positive integrated excess variance means intervention was worse. The
    threshold uses the continuation's pre-event mean; physical prefixes must
    match exactly until the declared first possible effect. No confidence
    interval or cohort acceptance decision is calculated here.
    """
    frames, burn = config['frames'], config['burn_in']
    onset, end = config['faults']['onset'], config['faults']['end']
    delay, modes = config['delay_frames'], config['modes']
    if (any(type(x) is not int for x in (frames, burn, onset, end, delay, modes,
                                         effect_frame, consecutive))
            or not 0 <= burn < onset <= effect_frame < end <= frames
            or delay < 1 or modes < 1 or consecutive < 1):
        raise ValueError('Invalid fixed recovery horizon')
    for value in (config['dt_s'], config['wavelength_m'], config['command_clip_rad'],
                  threshold_multiplier):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not np.isfinite(value) or value <= 0):
            raise ValueError('Recovery scales must be finite and positive')
    for outputs in (continued, intervened):
        if not FULL_OUTPUT_KEYS.issubset(outputs):
            raise ValueError('Recovery requires complete physical output streams')
        _commands(outputs, config)
        if 'handover_requested' in outputs:
            requested = np.asarray(outputs['handover_requested'])
            if requested.shape != (frames, modes) or requested.dtype.kind not in 'fiu':
                raise ValueError('Incomplete requested-command history')
            if np.isfinite(requested).all() and np.isfinite(outputs['command']).all():
                clipped = np.clip(requested, -config['command_clip_rad'], config['command_clip_rad'])
                if not np.array_equal(clipped, outputs['command']):
                    raise ValueError('Requested command does not reproduce the issued command')
                if not np.array_equal(np.any(requested != clipped, axis=1), outputs['saturated']):
                    raise ValueError('Saturation trace differs from requested-command clipping')
    summaries = [paired_metrics(outputs, reference, config)
                 for outputs in (continued, intervened)]
    result = {
        'status': 'nonfinite_outputs', 'horizon_frames': [onset, end],
        'pre_event_frames': [burn, onset], 'first_possible_effect_frame': effect_frame,
        'outcomes': None,
    }
    if not all(row['metrics']['all_outputs_finite'] for row in summaries):
        return result
    if any(row['fault_window'][key] is None for row in summaries
           for key in ('wfe_nm', 'open_wfe_nm', 'strehl', 'saturation_fraction')):
        result['status'] = 'nonfinite_reduction'
        return result
    for name in ('variance', 'strehl', 'applied'):
        if not np.array_equal(np.asarray(continued[name])[:effect_frame],
                              np.asarray(intervened[name])[:effect_frame]):
            raise ValueError('Physical histories differ before the declared intervention effect')
    sl, dt = slice(onset, end), config['dt_s']
    opened = np.asarray(reference['variance'])
    normalizer = float(np.sum(opened[sl]) * dt)
    if not np.isfinite(normalizer) or normalizer <= 0:
        result['status'] = 'invalid_zero_dm_normalizer'
        return result
    variance = [np.asarray(x['variance']) for x in (continued, intervened)]
    threshold = float(np.mean(variance[0][burn:onset]) * threshold_multiplier)
    excess = float(np.sum(variance[1][sl] - variance[0][sl]) * dt)
    if not np.isfinite(threshold) or not np.isfinite(excess):
        result['status'] = 'nonfinite_reduction'
        return result
    arms = {}
    for label, outputs, values, summary in zip(
            ('continued', 'intervened'), (continued, intervened), variance, summaries, strict=True):
        commands = {}
        for key in ('command', 'applied', 'handover_requested'):
            if key not in outputs:
                commands[key] = None
                continue
            array = np.asarray(outputs[key])
            # onset > 0: include the jump INTO the event horizon.
            jumps = np.abs(np.diff(array[onset - 1:end], axis=0))
            commands[key] = {
                'peak_abs_by_mode_rad': np.max(np.abs(array[sl]), axis=0).tolist(),
                'peak_step_by_mode_rad': np.max(jumps, axis=0).tolist(),
                'peak_rate_by_mode_rad_s': (np.max(jumps, axis=0) / dt).tolist(),
            }
        recovery = sustained_return(values, start=effect_frame, stop=end,
                                    threshold=threshold, consecutive=consecutive)
        recovery['onset_to_first_s'] = (None if recovery['first_frame'] is None else
                                        (recovery['first_frame'] - onset) * dt)
        recovery['onset_to_confirmation_s'] = (
            None if recovery['confirmation_frame'] is None else
            (recovery['confirmation_frame'] - onset) * dt)
        arms[label] = {
            **summary['fault_window'], 'peak_variance_rad2': float(np.max(values[sl])),
            'integrated_variance_rad2_s': float(np.sum(values[sl]) * dt),
            'commands': commands, 'reference_return': recovery,
        }
    outcomes = {
        'integrated_excess_variance_rad2_s': excess,
        'zero_dm_integral_rad2_s': normalizer,
        'normalized_excess_variance': excess / normalizer,
        'threshold_multiplier': float(threshold_multiplier), 'arms': arms,
    }
    # Finite raw numbers can still overflow during reductions or unit scaling.
    def finite_tree(value):
        if isinstance(value, dict):
            return all(finite_tree(x) for x in value.values())
        if isinstance(value, list):
            return all(finite_tree(x) for x in value)
        return value is None or not isinstance(value, Real) or bool(np.isfinite(value))
    if not finite_tree(outcomes):
        result['status'] = 'nonfinite_reduction'
        return result
    return {**result, 'status': 'finite_pair', 'outcomes': outcomes}
