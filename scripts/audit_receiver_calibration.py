"""Reconstruct receiver utility from a retained commissioning/service packet.

This reads raw arrays without running the controller or importing its acquisition
runner. It checks finite-grid accounting, not instrument fidelity or causality
from hashes alone. The receiver overlap model is shared with the qualified
physics package; this is not an independent propagation-model comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
import zipfile

import numpy as np

from noether_physics import gaussian_fiber_coupling
from audit_controller_physical_metrics import check_queue, physical_metrics


RTOL, ATOL = 5e-12, 5e-14
MATRICES = {'command', 'applied', 'estimate', 'measurement', 'prediction',
            'feedback_requested', 'probe_requested', 'combined_requested',
            'measurement_perturbation'}
SCALARS = {'variance', 'strehl', 'score'}
FLAGS = {'alarm', 'fallback', 'saturated', 'unavailable'}
ARMS = ('commissioning-nominal', 'commissioning-probed',
        'service-nominal', 'service-corrected')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    def invalid(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 1024**2,
            'Expected bounded regular JSON')
    def pairs(items):
        result = {}
        for key,value in items:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    return json.loads(path.read_bytes(), parse_constant=invalid, object_pairs_hook=pairs)


def close(actual, expected, label, *, rtol=RTOL, atol=ATOL):
    a, b = np.asarray(actual), np.asarray(expected)
    require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(),
            'Nonfinite or incompatible ' + label)
    require(np.allclose(a, b, rtol=rtol, atol=atol), 'Raw reconstruction differs: ' + label)
    return float(np.max(np.abs(a-b))) if a.size else 0.


def audit_trial(directory, *, config_path, seed_audit_path, expected_result_sha256,
                expected_config_sha256, expected_seed_audit_sha256):
    directory = Path(directory)
    require(not directory.is_symlink() and directory.is_dir(), 'Expected a regular packet directory')
    result_path = directory / 'result.json'
    require(result_path.is_file() and not result_path.is_symlink(), 'Expected regular terminal record')
    require(digest(result_path) == expected_result_sha256, 'Terminal record identity differs')
    result = read_json(result_path)
    require(result['schema'] == 'noether-receiver-calibration-trial/1'
            and result['status'] in {'completed', 'failed'}, 'Unknown terminal schema/status')
    inventory = result['files']
    require(isinstance(inventory, dict) and 'result.json' not in inventory, 'Invalid artifact inventory')
    observed = {p.name for p in directory.iterdir()}
    require(observed == set(inventory) | {'result.json'}, 'Unexpected or missing packet files')
    for name, row in inventory.items():
        path = directory / name
        require(Path(name).name == name and name not in {'.', '..'}
                and path.is_file() and not path.is_symlink(), 'Unsafe artifact path')
        require(path.stat().st_size == row['bytes'] and digest(path) == row['sha256'],
                'Artifact bytes differ: ' + name)
    require(sum(p.stat().st_size for p in directory.iterdir()) <= 32*1024**2,
            'Packet exceeds declared retained byte bound')
    config_path, seed_audit_path = Path(config_path), Path(seed_audit_path)
    protocol, seed_audit = read_json(config_path), read_json(seed_audit_path)
    require(digest(config_path) == expected_config_sha256
            and digest(seed_audit_path) == expected_seed_audit_sha256,
            'External protocol or seed audit identity differs')
    inputs = None
    input_binding_error = None
    try:
        require('inputs.json' in inventory, 'Inputs artifact is absent')
        inputs = read_json(directory / 'inputs.json')
        require(inputs['config_sha256'] == expected_config_sha256
                and inputs['seed_audit_sha256'] == expected_seed_audit_sha256
                and inputs['protocol'] == protocol, 'Embedded protocol differs from admitted bytes')
        require(all(result[k] == inputs[k] for k in ('root_id', 'condition_id')), 'Trial identity differs')
        require(result['study_id'] == protocol['study_id'] == seed_audit['namespace']
                and result['root_id'] in protocol['roots'], 'Undeclared study/root identity')
        rows = [row for row in seed_audit['rows'] if row['root'] == result['root_id']]
        require(len(rows) == 1 and inputs['streams'] == rows[0], 'Selected stream row differs')
        conditions = [row for row in protocol['conditions'] if row['id'] == result['condition_id']]
        require(len(conditions) == 1 and inputs['condition'] == conditions[0], 'Condition identity differs')
    except (ValueError, KeyError, TypeError) as error:
        if result['status'] != 'failed':
            raise
        input_binding_error = type(error).__name__ + ': ' + str(error)
    cfg_base, windows = protocol['base_config'], protocol['windows']

    def finish(report):
        require(digest(result_path) == expected_result_sha256, 'Terminal changed during audit')
        for name,row in inventory.items():
            require(digest(directory/name) == row['sha256'], 'Artifact changed during audit')
        require(digest(config_path) == expected_config_sha256
                and digest(seed_audit_path) == expected_seed_audit_sha256, 'Admission changed during audit')
        return {'schema':'noether-receiver-calibration-audit/1', **report,
                'root_id':result['root_id'],'condition_id':result['condition_id'],
                'result_sha256':expected_result_sha256,
                'scope':'Raw finite-grid receiver and utility accounting only. Fit numerics, sensor and controller recurrence, historical ordering and hardware fidelity are not established by this reader.'}

    if result['status'] == 'failed':
        require(result['charged_mean_delta'] is None and result['utility_bounds'] == [-1.,1.]
                and result['error'] is not None, 'Failure was converted to a numerical success')
        missing = result.get('missing_artifacts', [])
        require(isinstance(missing,list)
                and all(isinstance(row,dict) and row.keys() == {'file','original_sha256'}
                        and isinstance(row['file'],str) and Path(row['file']).name == row['file']
                        and row['file'] not in {'.','..'} and row['file'] not in observed
                        and isinstance(row['original_sha256'],str)
                        and re.fullmatch('[0-9a-f]{64}',row['original_sha256']) for row in missing)
                and len({row['file'] for row in missing}) == len(missing),
                'Invalid missing-artifact inventory')
        # A failed write may be truncated, structurally invalid or nonfinite.
        # Verify retained bytes without interpreting them as scientific arrays.
        return finish({'status':'retained_failed_trial','charged_mean_delta':None,
                       'utility_bounds':[-1.,1.], 'failure_stage':result['stage'],
                       'input_binding_verified':input_binding_error is None,
                       'input_binding_error':input_binding_error,
                       'scientific_payload_reconstructed':False,
                       'retained_files_verified':len(inventory), 'missing_artifacts':missing})
    require(inputs is not None and not result.get('missing_artifacts', []), 'Completed trial lacks inputs/artifacts')

    loaded = {}
    logical_bytes = 0

    def arrays(name):
        nonlocal logical_bytes
        require(name in inventory and 'arrays' in inventory[name], 'Missing numeric archive: ' + name)
        with zipfile.ZipFile(directory / name) as archive:
            members = archive.infolist()
            require(len({entry.filename for entry in members}) == len(members), 'Duplicate array member')
            require(all(Path(entry.filename).name == entry.filename and entry.filename.endswith('.npy')
                        for entry in members), 'Unsafe array member')
            require(sum(entry.file_size for entry in members) <= 16*1024**2,
                    'Array archive exceeds logical bound before decompression')
            declared = inventory[name]['arrays']
            require({entry.filename[:-4] for entry in members} == set(declared), 'NPY member inventory differs')
            for entry in members:
                with archive.open(entry) as stream:
                    version = np.lib.format.read_magic(stream)
                    require(version in {(1,0),(2,0)}, 'Unsupported NPY header version')
                    reader = (np.lib.format.read_array_header_1_0 if version == (1,0)
                              else np.lib.format.read_array_header_2_0)
                    shape, _, dtype = reader(stream, max_header_size=4096)
                    require(dtype.kind in 'fbiuc' and dtype.itemsize <= 16 and len(shape) <= 4
                            and all(type(dimension) is int and dimension >= 0 for dimension in shape),
                            'Unsafe NPY dtype or shape')
                    size = math.prod(shape) * dtype.itemsize
                    require(size <= 16*1024**2 and entry.file_size - stream.tell() == size,
                            'NPY declared allocation differs from bounded payload')
                    require(declared[entry.filename[:-4]] == {'shape':list(shape), 'dtype':dtype.str,'bytes':size},
                            'NPY header differs from manifest')
        with np.load(directory / name, allow_pickle=False) as archive:
            values = {key: archive[key] for key in archive.files}
        declared = inventory[name]['arrays']
        require(set(values) == set(declared), 'Array inventory differs: ' + name)
        for key, array in values.items():
            require(array.dtype.kind in 'fbiuc', 'Unsafe array dtype')
            require(declared[key] == {'shape': list(array.shape), 'dtype': array.dtype.str,
                                      'bytes': array.nbytes}, 'Array descriptor differs')
            logical_bytes += array.nbytes
        require(logical_bytes <= 16*1024**2, 'Scientific payload exceeds bound')
        loaded[name] = values
        return values

    # Inspect every retained numerical payload of a completed trial.
    for name in inventory:
        if name.endswith('.npz'):
            arrays(name)
    require(logical_bytes == result['logical_scientific_bytes'], 'Logical payload accounting differs')
    calibration = loaded['calibration.npz']
    modes, n = cfg_base['modes'], cfg_base['n']
    basis, mask = calibration['basis'], calibration['mask']
    require(basis.shape == (modes, n, n) and mask.shape == (n, n), 'Calibration shape differs')
    require(basis.dtype == np.float64 and mask.dtype.kind in 'fbiu', 'Physical calibration dtype differs')
    require(np.isfinite(basis).all() and np.isfinite(mask).all()
            and np.isin(mask, (0, 1)).all() and mask.sum() > 0, 'Invalid physical calibration')
    requested = loaded['probe.npz']['requested']
    begin, end = windows['probe_start'], windows['probe_stop']
    amplitude = protocol['probe']['amplitude_rad']
    require(requested.shape == (windows['commissioning_frames'], modes)
            and requested.dtype == np.float64
            and np.all(requested[:begin] == 0) and np.all(requested[end:] == 0)
            and np.all(np.abs(requested[begin:end]) == amplitude), 'Probe budget/window differs')
    fit = read_json(directory / 'fit.json')
    freeze = read_json(directory / 'fit-freeze.json')
    fit_sha = inventory['fit.json']['sha256']
    require(fit['schema'] == 'noether-receiver-calibration-fit/1'
            and freeze['schema'] == 'noether-receiver-calibration-fit-freeze/1'
            and freeze['fit_sha256'] == fit_sha == result['fit_sha256'], 'Frozen fit identity differs')
    for field, filename in (('inputs_sha256', 'inputs.json'),
                            ('calibration_sha256', 'calibration.npz'),
                            ('commissioning_probed_sha256', 'commissioning-probed.npz'),
                            ('probe_sha256', 'probe.npz')):
        require(fit[field] == inventory[filename]['sha256'], 'Fit input binding differs')
    decision = fit['decision']
    require(decision['status'] in {'ready', 'withheld'}
            and decision['status'] == result['fit_status'], 'Fit status differs')
    close(decision['detector_reference'], calibration['reference'], 'original detector reference', rtol=0, atol=0)
    if decision['status'] == 'withheld':
        close(decision['reconstructor'], calibration['reconstructor'], 'withheld reconstructor', rtol=0, atol=0)
        left, right = loaded['service-nominal.npz'], loaded['service-corrected.npz']
        require(left.keys() == right.keys() and all(np.array_equal(left[k], right[k]) for k in left),
                'Withheld service differs from incumbent operation')
    eta, errors = {}, {}
    for arm in ARMS:
        recording = arm.split('-')[0]
        frames = windows[recording + '_frames']
        cfg = dict(cfg_base, frames=frames, photons=inputs['condition']['photons'])
        cfg['faults'] = dict(cfg_base['faults'], onset=frames, end=frames)
        raw = loaded[arm + '.npz']
        require(raw.keys() == MATRICES | SCALARS | FLAGS, 'Original loop schema differs')
        for key, value in raw.items():
            shape = (frames, modes) if key in MATRICES else (frames,)
            dtype = np.bool_ if key in FLAGS else np.float64
            require(value.shape == shape and value.dtype == dtype and np.isfinite(value).all(),
                    'Nonfinite or incompatible loop stream: ' + arm + '/' + key)
        check_queue(raw['command'], raw['applied'], windows['delay_frames'], cfg['command_clip_rad'])
        expected_probe = requested if arm == 'commissioning-probed' else np.zeros((frames, modes))
        require(np.array_equal(raw['probe_requested'], expected_probe)
                and not np.any(raw['measurement_perturbation']), 'Injected inputs differ')
        require(np.array_equal(raw['combined_requested'], raw['feedback_requested'] + expected_probe)
                and np.array_equal(raw['command'], np.clip(raw['combined_requested'],
                                   -cfg['command_clip_rad'], cfg['command_clip_rad']))
                and np.array_equal(raw['saturated'], np.any(raw['command'] != raw['combined_requested'],axis=1)),
                'Controller command stages or saturation differ')
        require(not np.any(raw['fallback']) and not np.any(raw['unavailable']), 'Unexpected intervention')
        phase_archive = loaded[recording + '-phase.npz']
        require(set(phase_archive) == {'phase'}, 'Unexpected phase archive')
        phase = phase_archive['phase']
        require(phase.dtype == np.float64, 'Phase dtype differs')
        physical = physical_metrics(phase, raw['applied'], basis, mask, cfg, 'none')
        for key in ('variance', 'strehl'):
            errors[arm + '/' + key] = close(raw[key], physical[key], arm + '/' + key)
        # Reconstruct one spatial frame at a time, independent of receiver.py.
        efficiencies = np.empty(frames, dtype=np.float64)
        receiver = protocol['receiver']
        for frame in range(frames):
            mirror = np.sum(raw['applied'][frame, :, None, None] * basis, axis=0)
            residual = (phase[frame] - mirror) * mask
            residual -= mask * (math.fsum(residual.ravel()) / float(mask.sum()))
            opd = residual * (cfg['wavelength_m'] / (2 * math.pi))
            efficiencies[frame] = gaussian_fiber_coupling(
                mask, opd, pixel_pitch_m=cfg['diameter_m']/n,
                wavelength_m=receiver['receiver_wavelength_m'],
                focal_length_m=receiver['focal_length_m'], mode_radius_m=receiver['mode_radius_m'])
        stored = loaded['receiver-' + arm + '.npz']
        require(set(stored) == {'coupling','frame_index','time_s','pupil_axis_m'}, 'Unexpected receiver archive')
        require(np.all((efficiencies >= 0) & (efficiencies <= 1)), 'Coupling outside physical bound')
        errors[arm + '/coupling'] = close(stored['coupling'], efficiencies, arm + '/coupling')
        require(np.array_equal(stored['frame_index'], np.arange(frames)), 'Receiver frame identity differs')
        close(stored['time_s'], np.arange(frames)*cfg['dt_s'], 'receiver times', rtol=0, atol=0)
        close(stored['pupil_axis_m'], (np.arange(n)-(n-1)/2)*cfg['diameter_m']/n, 'pupil axes')
        eta[arm] = efficiencies
    require(np.array_equal(loaded['receiver-commissioning-nominal.npz']['coupling'][:begin],
                           loaded['receiver-commissioning-probed.npz']['coupling'][:begin]),
            'Excluded warm-up differs')
    cost = math.fsum((eta['commissioning-nominal'] - eta['commissioning-probed'])[begin:])
    gain = math.fsum(eta['service-corrected'] - eta['service-nominal'])
    charged = windows['commissioning_frames'] - begin + windows['service_frames']
    total = windows['commissioning_frames'] + windows['service_frames']
    utility = {'commissioning_cost':cost,'service_gain':gain,'charged_delta':gain-cost,
               'charged_mean_delta':(gain-cost)/charged,'total_mean_delta':(gain-cost)/total,
               'coupling_time_delta_s':(gain-cost)*cfg_base['dt_s'],
               'charged_frames':charged,'total_frames':total}
    summary = read_json(directory / 'summary.json')
    require(summary['schema'] == 'noether-receiver-calibration-summary/1'
            and summary['fit_status'] == decision['status'], 'Summary status differs')
    require(summary['utility'].keys() == utility.keys(), 'Utility fields differ')
    for key, value in utility.items():
        close(summary['utility'][key], value, 'utility/' + key, atol=2e-10)
    close(result['charged_mean_delta'], utility['charged_mean_delta'], 'terminal utility', atol=2e-12)
    close(result['utility_bounds'], [result['charged_mean_delta']]*2,
          'completed utility bounds', rtol=0, atol=0)
    require(result['error'] is None, 'Completed trial has an error')
    report = {'status':'raw_receiver_utility_reconstructed','charged_mean_delta':utility['charged_mean_delta'],
              'fit_status':decision['status'],'utility':utility,'max_absolute_errors':errors}
    return finish(dict(report, logical_scientific_bytes=logical_bytes))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--seed-audit', type=Path, required=True)
    parser.add_argument('--result-sha256', required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--seed-audit-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.resolve().is_relative_to(args.directory.resolve()),
            'Audit output must be outside the retained packet')
    result = audit_trial(args.directory, config_path=args.config, seed_audit_path=args.seed_audit,
                         expected_result_sha256=args.result_sha256,
                         expected_config_sha256=args.config_sha256,
                         expected_seed_audit_sha256=args.seed_audit_sha256)
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
