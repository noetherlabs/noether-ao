"""One fresh commissioning/service trial with retained physical-loop evidence.

The caller supplies an independently reviewed seed audit and resource admission.
This module does not supervise processes, reserve identities, retry trials or
infer performance from successful execution. Use a separate fresh namespace for
capacity checks; comparison recordings must never serve as resource probes.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re

import jax
import jax.numpy as jnp
import numpy as np

import noether_optics
from .atmosphere import phase_sequence
from .controller_evidence import _runtime, _source
from .identification import instrumented_kernel, linear_sensor_transform
from .receiver import evaluate_receiver
from .receiver_calibration import (CalibrationWindows, fit_commissioning_response,
                                  paired_coupling_summary)
from .simulation import calibration


MATRIX_OUTPUTS = ('command', 'applied', 'estimate', 'measurement', 'prediction',
                  'feedback_requested', 'probe_requested', 'combined_requested',
                  'measurement_perturbation')
SCALAR_OUTPUTS = ('variance', 'strehl', 'score')
BOOLEAN_OUTPUTS = ('alarm', 'fallback', 'saturated', 'unavailable')
ROLES = ('commissioning-phase', 'commissioning-detector-base',
         'service-phase', 'service-detector-base', 'probe')
TRIAL_FILES = {
    'inputs.json', 'calibration.npz', 'probe.npz', 'commissioning-phase.npz',
    'commissioning-nominal.npz', 'commissioning-probed.npz', 'fit.json',
    'fit-freeze.json', 'service-phase.npz', 'service-nominal.npz',
    'service-corrected.npz', 'summary.json',
    'receiver-commissioning-nominal.npz', 'receiver-commissioning-probed.npz',
    'receiver-service-nominal.npz', 'receiver-service-corrected.npz',
}


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _json_bytes(value):
    return (json.dumps(_plain(value), sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def _checked_json(path, expected_sha256):
    if not re.fullmatch('[0-9a-f]{64}', expected_sha256):
        raise ValueError('Expected an explicit SHA256 input identity')
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024**2:
        raise ValueError('Input must be a bounded regular JSON file')
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError('Input SHA256 differs from its admission')
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result
    value = json.loads(data, object_pairs_hook=pairs)
    _json_bytes(value)  # Reject nonfinite JSON constants before any acquisition.
    return value


def validate_protocol(protocol):
    """Check the narrow two-recording contract and pre-allocation byte bounds."""
    if protocol.get('schema') != 'noether-receiver-calibration-development/1':
        raise ValueError('Unknown receiver calibration protocol')
    windows = CalibrationWindows(**protocol['windows'])
    base, controller, probe, fit = (protocol[key] for key in ('base_config', 'controller', 'probe', 'fit'))
    n, modes = base['n'], base['modes']
    if (type(n) is not int or type(modes) is not int or n < 4 or modes < 1
            or modes + 1 >= windows.probe_stop - windows.probe_start
            or base['sensor'] != 'linear' or base['delay_frames'] != windows.delay_frames
            or controller['kind'] != 'integrator'
            or any(controller[key] != base[key] for key in ('gain', 'leak', 'command_clip_rad'))
            or probe['distribution'] != 'rademacher'
            or probe['additional_measurement_perturbation'] != 0
            or fit['method'] != 'iv' or fit['instrument'] != 'delayed_requested_probe'):
        raise ValueError('Protocol differs from the qualified linear integrator experiment')
    if (not 0 < probe['amplitude_rad'] <= base['command_clip_rad']
            or not 0 < fit['rcond'] < 1 or not 0 <= fit['minimum_relevance'] < 1
            or fit['maximum_reconstructor_gain'] <= 0):
        raise ValueError('Invalid probe or fit settings')
    endpoint = protocol['endpoint']
    if (endpoint['commissioning_window'] != [windows.probe_start, windows.commissioning_frames]
            or endpoint['service_window'] != [0, windows.service_frames]
            or endpoint['charged_frames'] != windows.charged_frames):
        raise ValueError('Endpoint windows disagree with acquisition windows')
    for names in (protocol['roots'], [row['id'] for row in protocol['conditions']]):
        if (not names or len(set(names)) != len(names)
                or any(not isinstance(name, str) or not re.fullmatch('[a-z0-9_-]+', name) for name in names)):
            raise ValueError('Protocol identities must be unique simple labels')
    for condition in protocol['conditions']:
        if condition['sensor_response'] not in ('nominal', 'coupled') or condition['photons'] <= 0:
            raise ValueError('Unknown sensor condition or nonpositive sensor flux')
    limits = protocol['artifact_limits']
    for key, ceiling in (('logical_scientific_bytes_per_condition', 16*1024**2),
                         ('retained_bytes_per_condition', 32*1024**2), ('study_bytes', 1024**3)):
        if type(limits[key]) is not int or not 0 < limits[key] <= ceiling:
            raise ValueError('Artifact limits exceed the bounded trial contract')
    # Two unique phase cubes and both arms' complete original loop schema.
    lower_bound = windows.total_frames * (8*n*n + 2*(9*modes*8 + 3*8 + 4))
    if lower_bound > limits['logical_scientific_bytes_per_condition']:
        raise ValueError('Declared raw arrays exceed the logical byte budget')
    _json_bytes(protocol)
    return windows


def _streams(protocol, audit, root_id):
    if (audit.get('schema') != 'receiver-calibration-seed-audit/1'
            or audit.get('status') != 'scoped_disjoint' or audit.get('collisions') != []
            or audit.get('unique') is not True or audit.get('uint32_bounds_valid') is not True
            or audit.get('detector_offset_no_wrap') is not True
            or audit.get('namespace') != protocol['study_id']):
        raise ValueError('Require the matching scoped, collision-free seed audit')
    expected = []
    identities = []
    for root in protocol['roots']:
        row = {'root': root}
        for role in ROLES:
            label = f"{protocol['study_id']}/{root}/{role}".encode()
            row[role] = int.from_bytes(hashlib.sha256(label).digest()[:4], 'big')
        for recording in ('commissioning', 'service'):
            base = row[f'{recording}-detector-base']
            if base >= 2**32 - 100000:
                raise ValueError('Detector seed offset would wrap')
            row[f'{recording}-detector'] = base + 100000
        identities.extend(value for key, value in row.items() if key != 'root')
        expected.append(row)
    if (len(set(identities)) != len(identities) or audit.get('rows') != expected
            or audit.get('integer_identity_count') != len(identities)
            or audit.get('active_rng_root_count') != 5*len(expected)
            or audit.get('detector_base_argument_count') != 2*len(expected)):
        raise ValueError('Seed table differs from the declared role derivation')
    return next(row for row in expected if row['root'] == root_id)


def _packages(repository, source, protocol):
    package = Path(noether_optics.__file__).resolve().parent
    expected = {name.removeprefix('src/noether_optics/'): digest for name, digest in source['files'].items()
                if name.startswith('src/noether_optics/') and name.endswith('.py')}
    actual = {path.relative_to(package).as_posix() for path in package.rglob('*.py')}
    if actual != set(expected) or any(_hash(package / name) != digest for name, digest in expected.items()):
        raise ValueError('Imported AO package differs from the committed source')
    spec = importlib.util.find_spec('noether_physics')
    if spec is None or spec.origin is None:
        raise ValueError('Install the pinned receiver package before acquisition')
    physics = Path(spec.origin).resolve().parent
    required = protocol['receiver_package']
    names = ('__init__.py', 'coupling.py', 'differentiable.py')
    distribution = importlib.metadata.distribution('noether-physics')
    physics_files = {name: _hash(physics / name) for name in names}
    if (required['distribution'] != 'noether-physics'
            or distribution.version != required['version']
            or set(required['files']) != set(names) or physics_files != required['files']
            or {path.relative_to(physics).as_posix() for path in physics.rglob('*.py')} != set(names)):
        raise ValueError('Installed receiver package differs from the qualified protocol source')
    direct_url = distribution.read_text('direct_url.json')
    installed_commit = json.loads(direct_url).get('vcs_info', {}).get('commit_id') if direct_url else None
    if installed_commit is not None and installed_commit != required['git_commit']:
        raise ValueError('Installed receiver VCS identity differs from the qualified source')
    return {
        'noether_optics': {'version': importlib.metadata.version('noether-optics'), 'files': expected},
        'noether_physics': {'version': distribution.version, 'files': physics_files,
                           'declared_git_commit': required['git_commit'],
                           'installed_vcs_commit': installed_commit},
    }


def capture_inputs(config_path, seed_audit_path, *, root_id, condition_id,
                   expected_config_sha256, expected_seed_audit_sha256, repository):
    """Bind committed source, protocol, audited streams and actual CPU runtime."""
    protocol = _checked_json(config_path, expected_config_sha256)
    audit = _checked_json(seed_audit_path, expected_seed_audit_sha256)
    validate_protocol(protocol)
    conditions = {row['id']: row for row in protocol['conditions']}
    if root_id not in protocol['roots'] or condition_id not in conditions:
        raise ValueError('Requested root or condition is not declared')
    streams = _streams(protocol, audit, root_id)
    repository = Path(repository).resolve(strict=True)
    source = _source(repository)
    relative = Path(config_path).resolve(strict=True).relative_to(repository).as_posix()
    if source['files'].get(relative) != expected_config_sha256:
        raise ValueError('Protocol is not the exact committed execution input')
    packages = _packages(repository, source, protocol)
    runtime = _runtime()
    if runtime['machine']['jax_backend'] != 'cpu' or runtime['machine']['x64'] is not True:
        raise ValueError('Receiver calibration requires the admitted float64 CPU runtime')
    return {'protocol': protocol, 'condition': conditions[condition_id], 'streams': streams,
            'source': source, 'runtime': runtime, 'packages': packages,
            'config_sha256': expected_config_sha256, 'seed_audit_sha256': expected_seed_audit_sha256,
            'root_id': root_id, 'condition_id': condition_id}


def _validate_outputs(outputs, config, requested):
    frames, modes = config['frames'], config['modes']
    schema = {key: ((frames, modes), np.dtype('float64')) for key in MATRIX_OUTPUTS}
    schema.update({key: ((frames,), np.dtype('float64')) for key in SCALAR_OUTPUTS})
    schema.update({key: ((frames,), np.dtype('bool')) for key in BOOLEAN_OUTPUTS})
    if set(outputs) != set(schema):
        raise ValueError('Physical loop output fields changed')
    for name, (shape, dtype) in schema.items():
        value = outputs[name]
        if value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
            raise ValueError(f'Invalid or nonfinite physical loop output: {name}')
    delay = config['delay_frames']
    if (np.any(outputs['applied'][:delay] != 0)
            or not np.array_equal(outputs['applied'][delay:], outputs['command'][:-delay])
            or not np.array_equal(outputs['probe_requested'], requested)
            or np.any(outputs['measurement_perturbation'] != 0)
            or not np.array_equal(outputs['combined_requested'], outputs['feedback_requested'] + requested)
            or not np.array_equal(outputs['command'], np.clip(outputs['combined_requested'],
                                 -config['command_clip_rad'], config['command_clip_rad']))
            or not np.array_equal(outputs['saturated'],
                                  np.any(outputs['combined_requested'] != outputs['command'], axis=1))):
        raise ValueError('Physical loop command stages or FIFO disagree')
    if np.any(outputs['variance'] < 0) or np.any(outputs['fallback']) or np.any(outputs['unavailable']):
        raise ValueError('Invalid variance or undeclared fallback/unavailability')


def run_trial(config_path, seed_audit_path, *, root_id, condition_id, output,
              expected_config_sha256, expected_seed_audit_sha256, repository):
    """Acquire one never-overwritten trial; terminal scientific failures raise.

    Existing output directories are refused. Cross-directory identity admission
    belongs to the caller's study ledger, not a hidden global registry here.
    Produced bounded numeric artifacts survive a failed validation. Oversized or
    nonnumeric artifacts are refused without unsafe serialization. No raw stream
    is silently repaired, clipped, dropped from a successful record or rerolled.
    """
    output = Path(output).absolute()
    repository = Path(repository).resolve(strict=True)
    if output.is_relative_to(repository) or any(path.is_symlink() for path in (output, *output.parents)):
        raise ValueError('Raw trial output must be outside the source repository without symlinks')
    output.mkdir(exist_ok=False)
    files, logical_bytes = {}, 0
    retained_limit, logical_limit = 32*1024**2, 16*1024**2
    stage, inputs, decision, fit_hash = 'input_admission', None, None, None

    def account(name, arrays=None):
        path = output / name
        record = {'sha256': _hash(path), 'bytes': path.stat().st_size}
        if arrays is not None:
            record['arrays'] = {key: {'shape': list(value.shape), 'dtype': value.dtype.str,
                                      'bytes': value.nbytes} for key, value in sorted(arrays.items())}
        files[name] = record
        return record['sha256']

    def room(additional):
        if sum(path.stat().st_size for path in output.iterdir() if path.is_file()) + additional + 65536 > retained_limit:
            raise ValueError('Trial retained-byte budget exhausted')

    def write_json(name, value):
        data = _json_bytes(value)
        room(len(data))
        with (output / name).open('xb') as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        return account(name)

    def retain(name, values):
        nonlocal logical_bytes
        arrays = {key: np.asarray(value) for key, value in values.items()}
        if (not arrays or len(arrays) > 32 or any(not re.fullmatch('[a-z][a-z0-9_]{0,63}', key)
                or value.dtype.kind not in 'biufc' or value.ndim > 4 for key, value in arrays.items())):
            raise ValueError('Unsafe or unexpected numeric artifact schema')
        amount = sum(value.nbytes for value in arrays.values())
        if logical_bytes + amount > logical_limit:
            raise ValueError('Trial logical scientific-byte budget exhausted')
        room(amount + 4096*(len(arrays) + 1))
        with (output / name).open('xb') as stream:
            np.savez(stream, **dict(sorted(arrays.items())))
            stream.flush(); os.fsync(stream.fileno())
        logical_bytes += amount
        account(name, arrays)
        return arrays

    arguments = dict(root_id=root_id, condition_id=condition_id,
                     expected_config_sha256=expected_config_sha256,
                     expected_seed_audit_sha256=expected_seed_audit_sha256,
                     repository=repository)
    try:
        inputs = capture_inputs(config_path, seed_audit_path, **arguments)
        protocol, condition, streams = (inputs[key] for key in ('protocol', 'condition', 'streams'))
        windows = validate_protocol(protocol)
        logical_limit = protocol['artifact_limits']['logical_scientific_bytes_per_condition']
        retained_limit = protocol['artifact_limits']['retained_bytes_per_condition']
        input_hash = write_json('inputs.json', inputs)

        def configuration(frames):
            cfg = deepcopy(protocol['base_config'])
            cfg.update(frames=frames, photons=condition['photons'])
            cfg['faults'].update(onset=frames, end=frames)
            return cfg

        comm_cfg, service_cfg = configuration(windows.commissioning_frames), configuration(windows.service_frames)
        stage = 'nominal_calibration'
        nominal = calibration(comm_cfg)
        modes, n = comm_cfg['modes'], comm_cfg['n']
        transform = np.eye(modes, dtype=np.float64)
        if condition['sensor_response'] == 'coupled':
            spec = protocol['coupled_response']
            transform = (np.diag(np.linspace(spec['diagonal_first'], spec['diagonal_last'], modes))
                         + np.diag(np.full(modes-1, spec['upper_diagonal']), 1)
                         + np.diag(np.full(modes-1, spec['lower_diagonal']), -1))
        numeric_cal = {key: np.asarray(value) for key, value in nominal.items() if not callable(value)}
        retain('calibration.npz', numeric_cal | {'sensor_transform': transform})
        calibration_shapes = {'basis': (modes, n, n), 'mask': (n, n), 'projection': (modes, n*n),
                              'reference': (modes,), 'interaction': (modes, modes),
                              'reconstructor': (modes, modes), 'stuck_patch': (n, n),
                              'preconditioner': (modes, modes), 'twin_modes': ()}
        if set(numeric_cal) != set(calibration_shapes):
            raise ValueError('Nominal numeric calibration fields changed')
        for name, shape in calibration_shapes.items():
            value = numeric_cal[name]
            dtype = np.dtype('int64') if name == 'twin_modes' else np.dtype('float64')
            if value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
                raise ValueError(f'Invalid or nonfinite nominal calibration: {name}')
        if numeric_cal['twin_modes'].item() != modes:
            raise ValueError('Nominal calibration is not full modal sensing')
        physical = linear_sensor_transform(comm_cfg, nominal, transform)
        stage = 'probe_schedule'
        probe = np.zeros((windows.commissioning_frames, modes), dtype=np.float64)
        rng = np.random.default_rng(streams['probe'])
        probe[windows.probe_start:windows.probe_stop] = protocol['probe']['amplitude_rad'] * rng.choice(
            np.array([-1., 1.]), size=(windows.probe_stop-windows.probe_start, modes))
        retain('probe.npz', {'requested': probe})
        schedule = probe[windows.probe_start:windows.probe_stop]
        schedule_singular = np.linalg.svd(schedule - schedule.mean(axis=0), compute_uv=False)
        schedule_rank = int(np.sum(schedule_singular > protocol['fit']['rcond']*schedule_singular[0]))

        def phase_record(config, role):
            phase = np.asarray(jax.block_until_ready(phase_sequence(config, streams[f'{role}-phase'])))
            retain(f'{role}-phase.npz', {'phase': phase})
            if phase.shape != (config['frames'], n, n) or phase.dtype != np.dtype('float64') or not np.isfinite(phase).all():
                raise ValueError('Invalid or nonfinite phase sequence')
            phase.setflags(write=False)
            return phase

        def loop_record(config, cal, phase, requested, role, arm):
            phase_before, probe_before = phase.copy(), requested.copy()
            execute = instrumented_kernel(config, cal, fault='none')
            outputs = execute(phase, streams[f'{role}-detector-base'], requested, np.zeros_like(requested))
            host = retain(f'{role}-{arm}.npz', outputs)
            if not np.array_equal(phase, phase_before) or not np.array_equal(requested, probe_before):
                raise ValueError('Physical loop mutated its inputs')
            _validate_outputs(host, config, requested)
            return host

        stage = 'commissioning_phase'
        comm_phase = phase_record(comm_cfg, 'commissioning')
        stage = 'commissioning_nominal'
        comm0 = loop_record(comm_cfg, physical, comm_phase, np.zeros_like(probe), 'commissioning', 'nominal')
        stage = 'commissioning_probed'
        comm1 = loop_record(comm_cfg, physical, comm_phase, probe, 'commissioning', 'probed')
        stage = 'measured_response_fit'
        fit_settings = {key: protocol['fit'][key] for key in
                        ('maximum_reconstructor_gain', 'minimum_relevance', 'rcond')}
        decision = fit_commissioning_response(
            comm1['measurement'], comm1['command'], comm1['applied'], comm1['probe_requested'],
            nominal_reconstructor=np.asarray(nominal['reconstructor']),
            nominal_interaction=np.asarray(nominal['interaction']),
            detector_reference=np.asarray(nominal['reference']), windows=windows, **fit_settings)
        if schedule_rank < modes:
            decision = dict(decision, status='withheld', reason='rank_deficient_requested_schedule',
                            reconstructor=np.asarray(nominal['reconstructor']).copy())
        fit_hash = write_json('fit.json', {
            'schema': 'noether-receiver-calibration-fit/1', 'inputs_sha256': input_hash,
            'calibration_sha256': files['calibration.npz']['sha256'],
            'commissioning_probed_sha256': files['commissioning-probed.npz']['sha256'],
            'probe_sha256': files['probe.npz']['sha256'], 'decision': decision})

        def frozen_decision():
            value = _checked_json(output / 'fit.json', fit_hash)
            if (value['inputs_sha256'] != input_hash
                    or value['calibration_sha256'] != files['calibration.npz']['sha256']
                    or value['commissioning_probed_sha256'] != files['commissioning-probed.npz']['sha256']
                    or value['probe_sha256'] != files['probe.npz']['sha256']):
                raise ValueError('Fit artifact no longer binds commissioning inputs')
            return value['decision']

        decision = frozen_decision()
        write_json('fit-freeze.json', {'schema': 'noether-receiver-calibration-fit-freeze/1', 'fit_sha256': fit_hash})
        stage = 'service_phase'
        service_phase = phase_record(service_cfg, 'service')
        decision = frozen_decision()
        candidate = dict(physical, reconstructor=jnp.asarray(decision['reconstructor'], dtype=jnp.float64))
        zero_service = np.zeros((windows.service_frames, modes), dtype=np.float64)
        stage = 'service_nominal'
        service0 = loop_record(service_cfg, physical, service_phase, zero_service, 'service', 'nominal')
        frozen_decision()
        stage = 'service_corrected'
        if decision['status'] == 'withheld':
            service1 = retain('service-corrected.npz', service0)
        else:
            service1 = loop_record(service_cfg, candidate, service_phase, zero_service, 'service', 'corrected')
        frozen_decision()

        stage = 'receiver_evaluation'
        traces, receiver_metadata = {}, {}
        for label, cfg, cal, phase, outputs in (
            ('commissioning-nominal', comm_cfg, physical, comm_phase, comm0),
            ('commissioning-probed', comm_cfg, physical, comm_phase, comm1),
            ('service-nominal', service_cfg, physical, service_phase, service0),
            ('service-corrected', service_cfg, candidate, service_phase, service1),
        ):
            before = {key: value.copy() for key, value in outputs.items()}
            observed = evaluate_receiver(
                cfg, cal, phase, outputs, phase_reference_wavelength_m=cfg['wavelength_m'],
                source_id=f"{protocol['study_id']}/{root_id}/{condition_id}/{label}",
                fault='none', **protocol['receiver'])
            arrays = {key: observed[key] for key in ('coupling', 'frame_index', 'time_s', 'pupil_axis_m')}
            retain(f'receiver-{label}.npz', arrays)
            expected_axis = (np.arange(n, dtype=np.float64) - (n-1)/2) * (cfg['diameter_m']/n)
            expected_frames = np.arange(cfg['frames'], dtype=np.int64)
            if (np.asarray(observed['coupling']).shape != (cfg['frames'],)
                    or np.asarray(observed['coupling']).dtype != np.dtype('float64')
                    or not np.isfinite(observed['coupling']).all()
                    or np.any((observed['coupling'] < 0) | (observed['coupling'] > 1))
                    or np.asarray(observed['frame_index']).dtype != np.dtype('int64')
                    or not np.array_equal(observed['frame_index'], expected_frames)
                    or np.asarray(observed['time_s']).dtype != np.dtype('float64')
                    or not np.array_equal(observed['time_s'], expected_frames*cfg['dt_s'])
                    or np.asarray(observed['pupil_axis_m']).dtype != np.dtype('float64')
                    or not np.array_equal(observed['pupil_axis_m'], expected_axis)
                    or observed['evaluation_only'] is not True or observed['fault'] != 'none'):
                raise ValueError('Invalid receiver trace, frame identity or evaluator metadata')
            if not all(np.array_equal(outputs[key], value) for key, value in before.items()):
                raise ValueError('Receiver evaluation mutated physical loop outputs')
            traces[label] = observed['coupling']
            receiver_metadata[label] = {key: value for key, value in observed.items() if key not in arrays}
        stage = 'paired_summary'
        utility = paired_coupling_summary(
            traces['commissioning-nominal'], traces['commissioning-probed'],
            traces['service-nominal'], traces['service-corrected'], windows=windows, dt_s=comm_cfg['dt_s'])
        diagnostics = {
            'requested_schedule_rank': schedule_rank,
            'requested_schedule_singular_values': schedule_singular,
            'requested_probe_sum_squared_rad2': float(np.sum(probe**2)),
            'commissioning_final_command_difference_rad': comm1['command'][-1] - comm0['command'][-1],
            'commissioning_final_applied_difference_rad': comm1['applied'][-1] - comm0['applied'][-1],
            'service_copied_from_nominal_after_withholding': decision['status'] == 'withheld',
        }
        write_json('summary.json', {'schema': 'noether-receiver-calibration-summary/1',
                   'fit_status': decision['status'], 'fit_reason': decision['reason'],
                   'utility': utility, 'receiver_metadata': receiver_metadata, 'diagnostics': diagnostics})
        stage = 'input_and_artifact_recheck'
        if capture_inputs(config_path, seed_audit_path, **arguments) != inputs:
            raise ValueError('Source, protocol, seed audit or runtime changed during trial')
        if any(_hash(output / name) != record['sha256'] for name, record in files.items()):
            raise ValueError('Retained trial artifact changed')
        result = {'status': 'completed', 'charged_mean_delta': utility['charged_mean_delta'],
                  'utility_bounds': [utility['charged_mean_delta']]*2, 'error': None}
    except Exception as error:
        result = {'status': 'failed', 'charged_mean_delta': None, 'utility_bounds': [-1., 1.],
                  'error': {'type': type(error).__name__, 'message': str(error)[:2048]}}
        caught = error
    else:
        caught = None
    # Include partial numeric writes on failure, without pretending they passed
    # schema validation. The terminal manifest does not hash itself.
    present, changed_artifacts = set(), []
    for path in output.iterdir():
        if path.is_file():
            present.add(path.name)
            previous = files.get(path.name)
            if previous is None:
                account(path.name)
            elif _hash(path) != previous['sha256']:
                account(path.name)
                files[path.name]['original_sha256'] = previous['sha256']
                changed_artifacts.append(path.name)
    missing_artifacts = [{'file': name, 'original_sha256': files.pop(name)['sha256']}
                         for name in sorted(set(files) - present)]
    if result['status'] == 'completed' and (changed_artifacts or missing_artifacts or present != TRIAL_FILES):
        stage = 'terminal_artifact_inventory'
        caught = ValueError('Terminal artifact inventory differs from the completed trial')
        result = {'status': 'failed', 'charged_mean_delta': None, 'utility_bounds': [-1., 1.],
                  'error': {'type': type(caught).__name__, 'message': str(caught)}}
    result.update(schema='noether-receiver-calibration-trial/1',
                  study_id=inputs['protocol']['study_id'] if inputs is not None else None,
                  root_id=root_id, condition_id=condition_id, stage=stage,
                  fit_status=decision['status'] if decision is not None else None,
                  fit_sha256=fit_hash, files=files, logical_scientific_bytes=logical_bytes,
                  missing_artifacts=missing_artifacts)
    # The 64 KiB reserve in room() is specifically for this terminal record.
    terminal = _json_bytes(result)
    if len(terminal) > 65536:
        raise RuntimeError('Terminal evidence exceeds the reserved metadata budget')
    with (output / 'result.json').open('xb') as stream:
        stream.write(terminal); stream.flush(); os.fsync(stream.fileno())
    if caught is not None:
        raise RuntimeError(f'Receiver trial failed at {stage}; retained result.json') from caught
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--seed-audit', required=True, type=Path)
    parser.add_argument('--seed-audit-sha256', required=True)
    parser.add_argument('--repository', required=True, type=Path)
    parser.add_argument('--root', required=True)
    parser.add_argument('--condition', required=True)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    result = run_trial(args.config, args.seed_audit, root_id=args.root, condition_id=args.condition,
                       output=args.out, repository=args.repository,
                       expected_config_sha256=args.config_sha256,
                       expected_seed_audit_sha256=args.seed_audit_sha256)
    print(json.dumps({'status': result['status'], 'root_id': result['root_id'],
                      'condition_id': result['condition_id'], 'fit_status': result['fit_status']}))
