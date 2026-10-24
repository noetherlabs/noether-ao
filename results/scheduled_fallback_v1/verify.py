"""Reconstruct scheduled-fallback metrics from the released numerical evidence.

Requires NumPy only. No simulator, acquisition module or external file is loaded.
Numerical trace/metric functions retain the qualified reducer implementation;
public-file validation and Student-t quantile inversion are separate code.
Digests check retained-byte consistency, not acquisition-time authenticity.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import zipfile
import numpy as np

ABS_TOL = 1e-10
REL_TOL = 1e-9
CONDITIONS = ("nominal", "optical_gain_drift")
P_ACTIONS = ("continue_P", "switch_P_to_I")
I_ACTIONS = ("continue_I", "switch_I_to_I")
PRESENTATION_FILES = {'README.md', 'plot.py', 'comparison.png', 'comparison.svg'}
class ReductionError(ValueError):
    """Evidence fails the retained numerical or public file contract."""

def _require(condition, message):
    if not condition:
        raise ReductionError(message)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


def _json(path):
    def reject_constant(value):
        raise ReductionError("Nonfinite JSON literal: " + value)
    def finite_number(value):
        result = float(value)
        _require(math.isfinite(result), "Nonfinite JSON number")
        return result
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, "Duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), parse_constant=reject_constant,
                      parse_float=finite_number, object_pairs_hook=unique_pairs)


def _close(np, actual, expected, label):
    _require(actual.shape == expected.shape, label + ": shape differs")
    _require(bool(np.allclose(actual, expected, rtol=REL_TOL,
                              atol=ABS_TOL, equal_nan=True)),
             label + ": independently reconstructed values differ")


def physical_metrics(phases, applied, basis, mask, wavelength_m):
    """Reconstruct physical maps independently from actual applied commands.

    Inputs are stored float64 arrays. Nonfinite applied rows remain NaN; they
    cannot become scored valid rows. The returned residual map is transient.
    """
    import numpy as np

    for name, value in (("phases", phases), ("applied", applied),
                        ("basis", basis), ("mask", mask)):
        _require(isinstance(value, np.ndarray) and value.dtype == np.float64,
                 name + ": expected stored float64 array")
    _require(phases.ndim == 3 and mask.shape == phases.shape[1:],
             "Physical phase/pupil shape")
    _require(basis.ndim == 3 and basis.shape[1:] == mask.shape,
             "Physical basis shape")
    _require(applied.shape == (len(phases), len(basis)),
             "Applied-command shape")
    _require(bool(np.isfinite(phases).all() and np.isfinite(basis).all()),
             "Nonfinite phase or basis")
    _require(bool(np.isfinite(mask).all()
                  and np.all((mask == 0) | (mask == 1)) and mask.sum() > 0),
             "Pupil must be finite, nonempty and binary")
    _require(type(wavelength_m) in (int, float)
             and math.isfinite(wavelength_m) and wavelength_m > 0,
             "Invalid physical wavelength")
    count = float(mask.sum())
    dm = np.einsum("tm,mxy->txy", applied, basis)
    before_piston = (phases - dm) * mask
    piston = before_piston.sum(axis=(1, 2)) / count
    residual = before_piston - piston[:, None, None] * mask
    variance = (residual * residual).sum(axis=(1, 2)) / count
    real = (np.cos(residual) * mask).sum(axis=(1, 2)) / count
    imag = (np.sin(residual) * mask).sum(axis=(1, 2)) / count
    strehl = real * real + imag * imag
    open_map = phases * mask
    open_piston = open_map.sum(axis=(1, 2)) / count
    open_map = open_map - open_piston[:, None, None] * mask
    open_variance = (open_map * open_map).sum(axis=(1, 2)) / count
    open_real = (np.cos(open_map) * mask).sum(axis=(1, 2)) / count
    open_imag = (np.sin(open_map) * mask).sum(axis=(1, 2)) / count
    return {
        "residual": residual, "removed_piston_rad": piston,
        "variance": variance, "strehl": strehl,
        "wfe_nm": np.sqrt(variance) * wavelength_m / (2 * math.pi) * 1e9,
        "open_variance": open_variance,
        "open_strehl": open_real * open_real + open_imag * open_imag,
    }


def window_cost(variance, open_variance, start, end, dt):
    """Scalar integration; missing/nonfinite physical references are refused."""
    _require(type(start) is int and type(end) is int
             and 0 <= start < end <= len(variance) == len(open_variance),
             "Invalid fixed scoring window")
    _require(type(dt) in (int, float) and math.isfinite(dt) and dt > 0,
             "Invalid physical sampling interval")
    opened = [float(v) for v in open_variance[start:end]]
    _require(all(math.isfinite(v) and v >= 0 for v in opened),
             "Nonfinite or negative zero-DM reference")
    denominator = float(dt) * math.fsum(opened)
    _require(math.isfinite(denominator) and denominator > 0,
             "Nonpositive zero-DM reference")
    values = [float(v) for v in variance[start:end]]
    if not all(math.isfinite(v) and v >= 0 for v in values):
        return {"cost": None, "open_cost": denominator, "ratio": None}
    cost = float(dt) * math.fsum(values)
    _require(math.isfinite(cost), "Nonfinite integrated cost")
    return {"cost": cost, "open_cost": denominator, "ratio": cost / denominator}


def describe(values):
    """No implicit omission: incomplete paired vectors have no complete mean."""
    _require(all(v is None or (type(v) in (int, float) and math.isfinite(v))
                 for v in values), "Paired values must be finite numbers or explicit missing values")
    valid = [float(v) for v in values if v is not None]
    complete = len(valid) == len(values) and len(values) > 0
    return {
        "values": list(values), "expected": len(values), "available": len(valid),
        "complete": complete,
        "mean": math.fsum(valid) / len(valid) if complete else None,
        "median": statistics.median(valid) if complete else None,
        "minimum": min(valid) if complete else None,
        "maximum": max(valid) if complete else None,
    }


def check_trace(trace, phases, basis, mask, config, *, switch_frame=None):
    """Check sticky validity, physical metrics, clipping and delayed commands.

    Invalid outcomes are returned with their full denominator membership. A bad
    artifact raises; a correctly retained scientific invalidity does not.
    """
    import numpy as np

    nframes, modes = len(phases), config["modes"]
    required = {
        "trial_valid", "live_frame", "first_invalid_frame", "camera_valid",
        "measurement_valid", "policy_valid", "variance", "strehl",
        "raw_variance", "raw_strehl", "raw_residual", "applied", "command",
        "primary_candidate", "selected_candidate", "command_delta", "saturated",
        "measurement", "prediction", "raw_clean", "raw_count_rate",
        "fallback", "forced_intervention",
        "score", "alarm", "estimate", "unavailable", "fallback_candidate",
        "residual_estimate",
    }
    _require(switch_frame is None or (type(switch_frame) is int and 0 <= switch_frame < nframes),
             "Invalid scheduled switch index")
    _require(required == set(trace), "Incomplete or extra raw trace fields")
    _require(all(isinstance(v, np.ndarray) and v.ndim >= 1
                 and len(v) == nframes for v in trace.values()),
             "Every trace field must retain every frame")
    for name in ("trial_valid", "live_frame", "camera_valid", "measurement_valid",
                 "policy_valid", "saturated", "fallback", "forced_intervention",
                 "alarm", "unavailable"):
        _require(trace[name].shape == (nframes,)
                 and trace[name].dtype == np.bool_, name + ": invalid boolean trace")
    firsts = trace["first_invalid_frame"]
    _require(firsts.shape == (nframes,) and firsts.dtype == np.int64,
             "Invalid first-failure trace")
    flags = trace["trial_valid"]
    bad = np.flatnonzero(~flags)
    first = int(bad[0]) if len(bad) else None
    prefix = nframes if first is None else first
    _require(bool(flags[:prefix].all() and not flags[prefix:].any()),
             "Trial validity is not terminal")
    expected_first = np.full(nframes, -1, dtype=np.int64)
    if first is not None:
        expected_first[first:] = first
    _require(bool(np.array_equal(firsts, expected_first)),
             "Incorrect first-invalid-frame history")
    expected_live = np.arange(nframes) <= (nframes if first is None else first)
    _require(bool(np.array_equal(trace["live_frame"], expected_live)),
             "Invalid terminal/live-frame history")
    forced = np.zeros(nframes, dtype=bool) if switch_frame is None else (np.arange(nframes) >= switch_frame)
    forced &= trace["live_frame"]
    latched = np.maximum.accumulate(forced)
    _require(bool(np.array_equal(trace["fallback"], latched))
             and bool(np.array_equal(trace["forced_intervention"], forced)),
             "Scheduled live decision or retained fallback latch differs")
    for name, value in trace.items():
        _require(value.dtype in (np.dtype("float64"), np.dtype("bool"),
                                 np.dtype("int64")),
                 name + ": unsupported raw dtype")
        _require(bool(np.isfinite(value[:prefix]).all()),
                 name + ": nonfinite field classified valid")
        if first is not None and value.dtype == np.float64:
            _require(bool(np.isnan(value[first + 1:]).all()),
                     name + ": nonterminal float suffix")
    for name in ("camera_valid", "measurement_valid", "policy_valid"):
        _require(bool(trace[name][:prefix].all()),
                 name + ": valid trial contains invalid subsystem")
        if first is not None:
            _require(not bool(trace[name][first + 1:].any()),
                     name + ": terminal subsystem marked valid")
    if first is not None:
        for name in ("alarm", "saturated", "unavailable"):
            _require(not bool(trace[name][first + 1:].any()), name + ": terminal flag must be false")
    scalar_fields = ("variance", "strehl", "score", "raw_variance", "raw_strehl")
    for name in scalar_fields:
        _require(trace[name].shape == (nframes,) and trace[name].dtype == np.float64,
                 name + ": invalid scalar shape/type")
    for name in ("raw_residual",):
        _require(trace[name].shape == phases.shape and trace[name].dtype == np.float64,
                 name + ": invalid physical shape/type")
    for name in ("measurement", "prediction", "raw_clean", "raw_count_rate"):
        _require(trace[name].shape == (nframes, 4 * mask.size)
                 and trace[name].dtype == np.float64, name + ": invalid detector shape/type")
    for name in ("applied", "command", "primary_candidate",
                 "selected_candidate", "command_delta", "fallback_candidate",
                 "estimate", "residual_estimate"):
        _require(trace[name].shape == (nframes, modes) and trace[name].dtype == np.float64,
                 name + ": invalid modal shape")
    expected_camera = (trace["live_frame"]
                       & np.isfinite(trace["raw_residual"]).all(axis=(1, 2))
                       & np.isfinite(trace["raw_clean"]).all(axis=1)
                       & np.isfinite(trace["raw_count_rate"]).all(axis=1))
    _require(bool(np.array_equal(trace["camera_valid"], expected_camera)),
             "Camera draw validity differs from the live-frame sampling predicate")
    physical = physical_metrics(phases, trace["applied"], basis, mask,
                                config["wavelength_m"])
    for saved, reconstructed in (
        ("raw_residual", "residual"), ("raw_variance", "variance"),
        ("raw_strehl", "strehl"),
    ):
        _close(np, trace[saved], physical[reconstructed], saved)
    for name in ("variance", "strehl"):
        expected = physical[name].copy()
        expected[prefix:] = np.nan
        _close(np, trace[name], expected, name)
    delay, bound = config["delay_frames"], config["command_clip_rad"]
    _require(type(delay) is int and delay >= 1 and bound > 0, "Invalid FIFO config")
    expected_applied = np.zeros((nframes, modes), dtype=np.float64)
    expected_applied[delay:] = trace["command"][:-delay]
    last_live = nframes if first is None else first + 1
    _close(np, trace["applied"][:last_live], expected_applied[:last_live],
           "Actual FIFO-applied command")
    selected = np.where(forced[:, None], trace["fallback_candidate"], trace["primary_candidate"])
    _close(np, trace["selected_candidate"][:prefix], selected[:prefix],
           "Scheduled selected candidate")
    clipped = np.clip(selected[:prefix], -bound, bound)
    _close(np, trace["command"][:prefix], clipped, "Central clipped command")
    previous = np.vstack((np.zeros((1, modes)), trace["command"][:-1]))
    _close(np, trace["command_delta"][:prefix],
           trace["command"][:prefix] - previous[:prefix], "Command step")
    saturated = np.any(selected[:prefix] != clipped, axis=1)
    _require(bool(np.array_equal(trace["saturated"][:prefix], saturated)),
             "Incorrect saturation diagnostic")
    if first is not None:
        for name in ("command", "selected_candidate", "command_delta",
                     "variance", "strehl"):
            _require(bool(np.isnan(trace[name][first]).all()),
                     name + ": invalid frame was converted into a scored action")
    return {
        "valid": first is None, "first_invalid_frame": first,
        "physical": physical,
        "saturated_valid_frames": int(np.count_nonzero(trace["saturated"][:prefix])),
        "command_step_l2": np.sqrt(np.sum(trace["command_delta"]**2, axis=1)),
    }


def profile_contract(profile):
    """Fixed independent profile; no adapter or simulator import."""
    _require(profile in ("smoke", "study"), "Unknown recovery profile")
    smoke = profile == "smoke"
    return {"roots": [282000001] if smoke else list(range(282001001, 282001007)),
            "frames": 64 if smoke else 128, "burn_in": 16 if smoke else 32,
            "onset": 24 if smoke else 64, "end": 64 if smoke else 128,
            "drift_frames": 16 if smoke else 32, "optical_gain": .35,
            "switch_frame": 32 if smoke else 80, "first_application": 34 if smoke else 82,
            "conditions": list(CONDITIONS), "actions": list(P_ACTIONS + I_ACTIONS if smoke else P_ACTIONS)}


def return_to_reference(variance, reference, *, reference_start, reference_end,
                        start, end, valid, dt, paired_root_valid):
    """Eight-sample return to 1.1 times the continuing-P pre-event reference.

    This descriptive value cannot rescue an invalid whole trial or primary pair.
    A censored observation has no finite return time.
    """
    _require(type(valid) is bool and type(paired_root_valid) is bool, "Invalid return validity")
    _require(type(dt) in (int, float) and math.isfinite(dt) and dt > 0, "Invalid return sampling interval")
    _require(all(type(v) is int for v in (reference_start, reference_end, start, end))
             and 0 <= reference_start < reference_end <= start < end <= len(variance)
             and reference_end <= len(reference), "Invalid fixed reference-return windows")
    result = {"status": None, "paired_root_valid": paired_root_valid,
              "reference_window": [reference_start, reference_end], "observation_window": [start, end],
              "dwell_frames": 8, "threshold_multiplier": 1.1,
              "reference_mean_variance": None, "threshold_variance": None,
              "first_frame": None, "confirmation_frame": None,
              "elapsed_since_first_application_s": None,
              "observation_duration_s": (end-start)*dt, "descriptive_only": True}
    baseline = [float(v) for v in reference[reference_start:reference_end]]
    baseline_valid = all(math.isfinite(v) and v >= 0 for v in baseline)
    mean = math.fsum(v/len(baseline) for v in baseline) if baseline_valid else None
    baseline_valid = baseline_valid and math.isfinite(mean) and mean > 0 and math.isfinite(1.1*mean)
    if baseline_valid:
        result.update(reference_mean_variance=mean, threshold_variance=1.1*mean)
    if not valid:
        result['status'] = 'INVALID_TRIAL'
        return result
    if not baseline_valid:
        result['status'] = 'INVALID_REFERENCE'
        return result
    values = [float(v) for v in variance[start:end]]
    _require(all(math.isfinite(v) and v >= 0 for v in values), "Valid return trial has invalid physical samples")
    for offset in range(len(values)-8+1):
        if all(v <= 1.1*mean for v in values[offset:offset+8]):
            frame = start+offset
            result.update(status='RETURNED', first_frame=frame, confirmation_frame=frame+7,
                          elapsed_since_first_application_s=offset*dt)
            return result
    result['status'] = 'RIGHT_CENSORED'
    return result


def check_causal_pair(one, two, *, complete_end, physical_end):
    """Compare only the shared valid prefix; retain incomplete evidence explicitly."""
    import numpy as np
    _require(set(one) == set(two), "Paired raw inventories differ")
    n = len(one['trial_valid'])
    _require(len(two['trial_valid']) == n and 0 <= complete_end <= physical_end <= n,
             "Paired causal window differs")
    jointly_valid = one['trial_valid'] & two['trial_valid']
    failures = np.flatnonzero(~jointly_valid)
    valid_end = int(failures[0]) if len(failures) else n
    shared = min(complete_end, valid_end)
    for name in one:
        _require(bool(np.array_equal(one[name][:shared], two[name][:shared], equal_nan=True)),
                 "Causal complete prefix differs: " + name)
    physical_shared = min(physical_end, valid_end)
    for name in ('applied','measurement','raw_residual','raw_variance','raw_strehl','variance',
                 'strehl','raw_clean','raw_count_rate','camera_valid','measurement_valid'):
        _require(bool(np.array_equal(one[name][:physical_shared], two[name][:physical_shared], equal_nan=True)),
                 "Causal delayed physical prefix differs: " + name)
    return {'required_complete_prefix_end': complete_end, 'checked_complete_prefix_end': shared,
            'required_physical_prefix_end': physical_end, 'checked_physical_prefix_end': physical_shared,
            'complete': valid_end >= physical_end}


def assessment_from_reduced(rows, profile):
    """Fixed paired P-to-I minus continuing-P contrasts; no survivor estimate."""
    p = profile_contract(profile)
    actions = P_ACTIONS + I_ACTIONS if profile == 'smoke' else P_ACTIONS
    expected = {(root,condition,action) for root in p['roots']
                for condition in CONDITIONS for action in actions}
    _require(set(rows) == expected, 'Incomplete, duplicated or foreign assessment case')
    def contrast(root, condition, window):
        keep, switch = (rows[(root,condition,action)] for action in P_ACTIONS)
        a,b = keep['costs'][window],switch['costs'][window]
        _require(math.isclose(a['open_cost'],b['open_cost'],rel_tol=REL_TOL,abs_tol=ABS_TOL),
                 'Switch pair has unequal same-phase zero-DM denominator')
        if not keep['valid'] or not switch['valid']:
            return None
        _require(all(v is not None and math.isfinite(v) for v in (a['cost'],b['cost']))
                 and a['open_cost'] > 0, 'Valid pair has missing/nonfinite cost')
        return (b['cost']-a['cost'])/a['open_cost']
    primary,secondary,positive = {},{},{}
    for condition in CONDITIONS:
        values = [contrast(root,condition,'primary') for root in p['roots']]
        primary[condition] = {'root_seeds':p['roots'], **primary_interval(values,profile)}
        secondary[condition] = {'root_seeds':p['roots'], **describe([
            contrast(root,condition,'post_onset') for root in p['roots']]), 'descriptive_only':True}
        positive[condition] = [root for root,value in zip(p['roots'],values)
                               if value is not None and value > 0]
    delta = [b-a if a is not None and b is not None else None for a,b in
             zip(primary['nominal']['values'],primary['optical_gain_drift']['values'])]
    if profile == 'smoke':
        interpretation = 'ENGINEERING_ONLY'
    elif not all(v['complete'] for v in primary.values()):
        interpretation = 'INCOMPLETE_PRIMARY_PAIRS_NO_COMPLETE_PRIMARY_MEAN_OR_INTERVAL'
    elif any(v['interval'][0] > 0 for v in primary.values()):
        interpretation = 'SWITCH_HIGHER_COST_IN_AT_LEAST_ONE_CONDITION'
    elif all(v['interval'][1] < 0 for v in primary.values()):
        interpretation = 'SWITCH_LOWER_COST_IN_BOTH_CONDITIONS_NOT_SAFETY_EVIDENCE'
    else:
        interpretation = 'INCONCLUSIVE_NOT_EQUIVALENCE_OR_SAFETY'
    return {'primary_comparison':'scheduled_P_to_I_minus_continuing_P',
            'negative_primary_favors':'scheduled_switch', 'primary_regimes':primary,
            'secondary_post_onset':secondary,
            'paired_difference_of_primary_contrasts':{'root_seeds':p['roots'],**describe(delta),
                'direction':'optical_gain_drift_minus_nominal','interval':None,'descriptive_only':True},
            'observed_higher_switch_cost_roots':positive,
            'interpretation':interpretation,
            'safety_claim_supported':False}


def paired_descriptive_outcomes(rows, profile):
    """Expose adverse observed changes without promoting them to a safety test."""
    result = []
    for root in profile_contract(profile)['roots']:
        for condition in CONDITIONS:
            keep,switch = (rows[(root,condition,action)] for action in P_ACTIONS)
            valid = keep['valid'] and switch['valid']
            changes = {}
            for metric in ('primary_peak_residual_variance','post_onset_peak_residual_variance',
                           'primary_saturated_frames','post_onset_saturated_frames'):
                changes[metric] = switch[metric]-keep[metric] if valid else None
            result.append({'root_seed':root,'condition':condition,'paired_root_valid':valid,
                'direction':'scheduled_switch_minus_continuing_P','descriptive_only':True,
                'changes':changes,'observed_increases':[key for key,value in changes.items()
                    if value is not None and value>0],
                'return_status':{action:rows[(root,condition,action)]['return_to_reference']['status']
                                 for action in P_ACTIONS}})
    return result


def _finite_json(values):
    """Array rows keep explicit nulls for unavailable/nonfinite physical values."""
    import numpy as np
    values = np.asarray(values)
    if values.ndim == 0:
        value = float(values)
        return value if math.isfinite(value) else None
    return [_finite_json(value) for value in values]


def t5_quantile(probability=0.9875):
    """Invert the analytic Student-t(5) CDF by bounded bisection.

    With t=sqrt(5)*tan(theta), the density integral is proportional to
    integral cos(theta)^4 dtheta. This avoids a SciPy dependency.
    """
    _require(0.5 < probability < 1, "Unsupported Student-t probability")
    def cdf(x):
        theta = math.atan(x / math.sqrt(5))
        integral = 3 * theta / 8 + math.sin(2 * theta) / 4 + math.sin(4 * theta) / 32
        return 0.5 + 8 * integral / (3 * math.pi)
    low, high = 0., 100.
    _require(cdf(high) > probability, "Student-t bracket does not contain quantile")
    for _ in range(100):
        mid = (low + high) / 2
        if cdf(mid) < probability:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def primary_interval(values, profile):
    result = describe(values)
    result.update(interval=None, degrees_of_freedom=None, quantile=None)
    if profile == "smoke" or not result["complete"]:
        return result
    _require(profile == "study" and len(values) == 6, "Expected six independent roots")
    quantile = t5_quantile()
    halfwidth = quantile * statistics.stdev(values) / math.sqrt(6)
    result.update(interval=[result['mean'] - halfwidth, result['mean'] + halfwidth],
                  degrees_of_freedom=5, quantile=quantile, family_alpha=.05, regimes=2,
                  interpretation="Small-sample distribution-sensitive descriptive interval")
    return result


def relative_name(name):
    _require(isinstance(name, str) and name, "Expected a relative file name")
    path = PurePosixPath(name)
    _require(not path.is_absolute() and '..' not in path.parts and '\\' not in name
             and str(path) == name, "Noncanonical evidence path")
    return path


def file_bytes(root, name):
    path = root
    for part in relative_name(name).parts:
        path = path / part
        _require(not path.is_symlink(), "Symlinked evidence path")
    _require(path.is_file() and path.stat().st_size <= 8 * 1024**2,
             "Missing or oversized evidence file: " + name)
    return path.read_bytes()


def archive_bytes(data, expected):
    with zipfile.ZipFile(io.BytesIO(data)) as container:
        members = container.infolist()
        names = [entry.filename for entry in members]
        _require(len(names) == len(set(names)) and set(names) == {k + '.npy' for k in expected},
                 "Archive member inventory differs")
        _require(sum(entry.file_size for entry in members) <= 8 * 1024**2,
                 "Decoded archive exceeds retention bound")
        _require(all(entry.compress_type == zipfile.ZIP_STORED for entry in members),
                 "Expected original uncompressed NPZ members")
        for entry in members:
            with container.open(entry) as stream:
                version = np.lib.format.read_magic(stream)
                _require(version == (1,0), 'Unsupported NPY header version')
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
                _require(dtype in (np.dtype('float64'),np.dtype('bool'),np.dtype('int64'))
                         and len(shape) <= 3 and all(type(n) is int and n >= 0 for n in shape)
                         and math.prod(shape)*dtype.itemsize <= 8 * 1024**2
                         and stream.tell()+math.prod(shape)*dtype.itemsize == entry.file_size,
                         'Invalid or oversized decoded array shape')
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        arrays = {k: np.array(archive[k], copy=True) for k in archive.files}
    actual = {k: {'dtype': str(v.dtype), 'shape': list(v.shape)} for k, v in arrays.items()}
    _require(actual == expected and all(v.dtype in (np.dtype('float64'), np.dtype('bool'),
                                                   np.dtype('int64')) for v in arrays.values()),
             "Archive dtype or shape differs")
    return arrays


def parsed_bytes(data):
    def pairs(entries):
        result = {}
        for key,value in entries:
            _require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    def invalid(value):
        raise ReductionError('Nonfinite JSON constant')
    def real(value):
        parsed = float(value)
        _require(math.isfinite(parsed), 'Nonfinite JSON number')
        return parsed
    return json.loads(data, object_pairs_hook=pairs, parse_constant=invalid, parse_float=real)


def compare(actual, expected, label='assessment'):
    """Require complete structural equality with declared numerical tolerance."""
    if isinstance(expected, dict):
        _require(isinstance(actual, dict) and set(actual) == set(expected), label + ': keys differ')
        for key in expected:
            compare(actual[key], expected[key], label + '.' + key)
    elif isinstance(expected, list):
        _require(isinstance(actual, list) and len(actual) == len(expected), label + ': length differs')
        for index, (one, two) in enumerate(zip(actual, expected)):
            compare(one, two, label + '[' + str(index) + ']')
    elif type(expected) is float:
        _require(type(actual) in (int, float) and math.isfinite(actual)
                 and math.isclose(actual, expected, rel_tol=REL_TOL, abs_tol=ABS_TOL),
                 label + ': numerical value differs')
    else:
        _require(type(actual) is type(expected) and actual == expected, label + ': value differs')


def verify(root):
    """Verify a complete public package and return its reconstructed assessment."""
    root = Path(root).absolute()
    _require(root.is_dir() and not root.is_symlink(), 'Expected a regular evidence directory')
    manifest = parsed_bytes(file_bytes(root, 'manifest.json'))
    _require(manifest['schema'] == 'noether.scheduled-fallback.public-manifest.v1',
             'Unsupported release schema')
    files = manifest['artifacts']
    expected_npz = {'calibration.npz', 'models.npz'} | {
        f'phases/{root_seed}.npz' for root_seed in range(282001001, 282001007)} | {
        f'raw/R{i}-{condition}-{action}.npz' for i in range(1, 7)
        for condition in CONDITIONS for action in P_ACTIONS}
    expected_files = expected_npz | {'config.json', 'policies.json', 'assessment.json',
                                     'verify.py', 'test_verify.py'}
    _require(set(files) == expected_files, 'Public artifact inventory differs')
    for path in root.rglob('*'):
        _require(not path.is_symlink(), 'Symlink in evidence directory')
        relative = path.relative_to(root)
        if '__pycache__' in relative.parts:
            continue  # Interpreter cache is never a release artifact.
        _require(not any(part.startswith('.') for part in relative.parts), 'Hidden release file')
        if path.is_file():
            _require(relative.as_posix() in expected_files | {'manifest.json'} | PRESENTATION_FILES,
                     'Unlisted evidence or presentation file')
        else:
            _require(relative.as_posix() in {'raw','phases'}, 'Unlisted evidence directory')
    arrays, contents = {}, {}
    for name, metadata in files.items():
        data = file_bytes(root, name)
        _require(len(data) == metadata['bytes'] and hashlib.sha256(data).hexdigest() == metadata['sha256'],
                 'Artifact digest or size differs: ' + name)
        if name in expected_npz:
            arrays[name] = archive_bytes(data, metadata['arrays'])
            _require(metadata['native_sha256'] == metadata['sha256'], 'Raw native bytes were changed')
        else:
            contents[name] = data
    cfg = parsed_bytes(contents['config.json'])
    expected = parsed_bytes(contents['assessment.json'])
    policies = parsed_bytes(contents['policies.json'])
    p = profile_contract('study')
    _require(set(cfg) == {'schema','experiment_id','profile','cohort','cases','physical','metric_contract'}
             and cfg['schema'] == 'noether.scheduled-fallback.public-protocol.v1'
             and cfg['experiment_id'] == 'ao-scheduled-fallback-recovery-v1-c3-20261024'
             and cfg['profile'] == 'study' and cfg['cohort'] == p, 'Study cohort differs')
    metric = {'denominator':'same-root same-phase zero-DM residual-variance integral over primary window',
        'invalidity':'Any invalid frame withholds the entire paired-root cost and any complete mean or interval; no survivor mean, substitute seed or hidden exclusion.',
        'primary':'scheduled_P_to_I_minus_continuing_P',
        'primary_interval':{'conditions':2,'degrees_of_freedom':5,'family_alpha':.05,
            'interpretation':'distribution-sensitive descriptive interval','student_t_quantile_probability':.9875},
        'primary_window':[82,128],
        'reference_return':{'baseline':'continuing P mean physical variance','baseline_window':[32,64],
            'consecutive_frames':8,'search_window':[82,128],'threshold_multiplier':1.1,'unobserved':'RIGHT_CENSORED'},
        'safety_claim_supported':False,'secondary_post_onset_window':[64,128],
        'smoke_requires_all_eight_valid_and_exact_noop':False}
    _require(cfg['metric_contract'] == metric, 'Metric protocol differs')
    cases = [dict(id=f'R{i}-{condition}-{action}', stage='trial', root_seed=seed,
                  candidate_id=action, condition=condition)
             for i, seed in enumerate(p['roots'], 1) for condition in CONDITIONS for action in P_ACTIONS]
    _require(cfg['cases'] == cases, 'Case identities or order differ')
    physical = {'n':16,'modes':8,'sensor':'pyramid','detector_n':32,
                'wavelength_m':750e-9,'wavelengths_m':[750e-9],'diameter_m':1.,'r0_m':.4,
                'screen_factor':4,'wind_scale':2.,'photons':1e5,'read_noise_e':.3,
                'dt_s':.001,'delay_frames':2,'command_clip_rad':5.,'frames':128,'burn_in':32}
    _require(cfg['physical'] == physical, 'Physical protocol differs')
    cal = arrays['calibration.npz']
    _require(set(cal) == {'basis','mask','projection','reference','interaction','reconstructor',
                         'preconditioner','stuck_patch','common_reconstructor','common_reference'},
             'Calibration inventory differs')
    _require(all(v.dtype == np.float64 and np.isfinite(v).all() for v in cal.values()),
             'Invalid calibration dtype or values')
    basis, mask, c, interaction = (cal[k] for k in ('basis','mask','common_reconstructor','interaction'))
    _require(basis.shape == (8,16,16) and mask.shape == (16,16)
             and c.shape == (8,1024) and interaction.shape == (1024,8), 'Calibration shapes differ')
    u, singular, vt = np.linalg.svd(interaction, full_matrices=False)
    _require(singular[-1] > 0 and singular[0]/singular[-1] <= 1e6
             and np.linalg.norm(c @ interaction - np.eye(8), ord=2) <= 1e-10
             and np.allclose(c, (vt.T/singular) @ u.T, rtol=1e-12, atol=1e-12)
             and np.array_equal(cal['common_reference'], cal['reference']), 'Common observation differs')
    models = arrays['models.npz']
    _require(set(models) == {'predictor_coefficients'} and models['predictor_coefficients'].shape == (8,2)
             and models['predictor_coefficients'].dtype == np.float64
             and np.isfinite(models['predictor_coefficients']).all(), 'Selected predictor differs')
    for family in ('integrator', 'predictive'):
        candidate = deepcopy(policies[family])
        if family == 'predictive':
            coeff = models['predictor_coefficients']
            candidate['parameters']['coefficients'] = {'shape':[8,2], 'dtype':'float64',
                'sha256':hashlib.sha256(coeff.tobytes(order='C')).hexdigest()}
        _require(hashlib.sha256(_canonical(candidate)).hexdigest()
                 == policies['provenance']['parameter_sha256'][family], 'Policy fingerprint differs')
    trial_refs = manifest['trials']
    _require([row['case'] for row in trial_refs] == cases, 'Trial inventory differs')
    rows, raws, physical_rows = {}, {}, {}
    for ref in trial_refs:
        case = ref['case']; seed, condition, action = (case[k] for k in ('root_seed','condition','candidate_id'))
        phase = arrays[f'phases/{seed}.npz']
        _require(set(phase) == {'phase'} and phase['phase'].shape == (128,16,16)
                 and phase['phase'].dtype == np.float64 and np.isfinite(phase['phase']).all(), 'Phase shape differs')
        _require(ref['raw'] == f"raw/{case['id']}.npz", 'Trial raw binding differs')
        trace = arrays[ref['raw']]
        checked = check_trace(trace, phase['phase'], basis, mask, physical,
                              switch_frame=80 if action == 'switch_P_to_I' else None)
        _require(ref['validity']['valid'] == checked['valid']
                 and ref['validity']['first_invalid_frame'] == (-1 if checked['first_invalid_frame'] is None
                                                               else checked['first_invalid_frame']),
                 'Trial validity metadata differs')
        phys = checked['physical']; valid = checked['valid']
        costs = {name:window_cost(phys['variance'], phys['open_variance'], start, 128, .001)
                 for name,start in (('primary',82),('post_onset',64))}
        if not valid:
            for value in costs.values():
                value.update(cost=None,ratio=None)
        row = {'case':case,'record_sha256':ref['native_record_sha256'],'valid':valid,
               'reason':ref['validity']['reason'],'first_invalid_frame':checked['first_invalid_frame'],
               'costs':costs,'logical_frames':128,'camera_draw_frames':int(np.count_nonzero(trace['camera_valid'])),
               'valid_prefix_frames':128 if valid else checked['first_invalid_frame'],
               'saturated_valid_frames':checked['saturated_valid_frames'],
               'switch_step_command_delta':_finite_json(trace['command_delta'][80]),
               'switch_step_command_delta_l2':_finite_json(np.sqrt(np.sum(trace['command_delta'][80]**2))),
               'physical':{name:_finite_json(phys[name]) for name in ('variance','strehl','wfe_nm','open_variance','open_strehl')},
               'command_step_l2':_finite_json(checked['command_step_l2'])}
        for window, start in (('primary',82),('post_onset',64)):
            row[window+'_peak_residual_variance'] = float(np.max(phys['variance'][start:])) if valid else None
            row[window+'_saturated_frames'] = int(np.count_nonzero(trace['saturated'][start:])) if valid else None
        key = seed,condition,action
        rows[key], raws[key], physical_rows[key] = row,trace,phys
    causal = []
    for seed in p['roots']:
        for condition in CONDITIONS:
            one,two = (raws[seed,condition,a] for a in P_ACTIONS)
            causal.append({'root_seed':seed,'condition':condition,'actions':list(P_ACTIONS),
                           **check_causal_pair(one,two,complete_end=80,physical_end=82)})
        for action in P_ACTIONS:
            causal.append({'root_seed':seed,'action':action,'conditions':list(CONDITIONS),
                           **check_causal_pair(raws[seed,'nominal',action],raws[seed,'optical_gain_drift',action],
                                               complete_end=64,physical_end=64)})
        for condition in CONDITIONS:
            reference = physical_rows[seed,condition,'continue_P']['variance']
            paired_valid = all(rows[seed,condition,a]['valid'] for a in P_ACTIONS)
            for action in P_ACTIONS:
                row = rows[seed,condition,action]
                row['return_to_reference'] = return_to_reference(physical_rows[seed,condition,action]['variance'], reference,
                    reference_start=32,reference_end=64,start=82,end=128,valid=row['valid'],dt=.001,
                    paired_root_valid=paired_valid)
    actual = {'schema':'noether.scheduled-fallback.public-assessment.v1',
              'status':'COMPLETE_EVIDENCE_REDUCTION','profile':'study','experiment_id':cfg['experiment_id'],
              'stage':'assessment','comparison_tolerances':{'absolute':ABS_TOL,'relative':REL_TOL},
              **assessment_from_reduced(rows,'study'),
              'paired_descriptive_outcomes':paired_descriptive_outcomes(rows,'study'),
              'individual_trials':[rows[k] for k in sorted(rows)],'causal_prefix_checks':causal,
              'no_op_checks':[], 'engineering_smoke_complete':False,
              'failures':{'trials':len(rows),'invalid_trials':sum(not row['valid'] for row in rows.values()),
                          'invalid_case_ids':[row['case']['id'] for row in rows.values() if not row['valid']],
                          'logical_frames':sum(row['logical_frames'] for row in rows.values()),
                          'camera_draw_frames':sum(row['camera_draw_frames'] for row in rows.values())}}
    compare(actual, expected)
    return actual


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    report = verify(args.root)
    print(json.dumps({'status':'verified','trials':report['failures']['trials'],
                      'pairs':len(report['paired_descriptive_outcomes']),
                      'interpretation':report['interpretation'],
                      'primary_regimes':report['primary_regimes']},sort_keys=True))
