"""Reconstruct the complete scheduled-fallback study from retained raw packets.

The independent NumPy physical reducer checks finite-grid phase accounting.
It does not validate the sensor model, instrument fidelity or a safety claim.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from audit_controller_physical_metrics import ATOL, RTOL, check_queue, physical_metrics
from noether_optics import recovery_development as acquisition
from noether_optics.controller_evidence import _hash_file, capture_inputs
from noether_optics.controller_physics import episode_config
from noether_optics.controller_selection import _publish
from noether_optics.recovery_acquisition import (
    collection_manifest, read_json, root_indices, verify_packet, verify_root)
from noether_optics.recovery_metrics import paired_recovery_outcomes
from noether_optics.recovery_study import protocol, summarize


def reconstruct_root(root, collection, collection_sha, seed, study, frozen_cal, predictor, selection):
    """Authenticate each identity, reduce raw streams, and check physical metrics."""
    identity, report = verify_root(root, collection, collection_sha, seed)
    complete, initial_sha = verify_packet(root)
    spec = collection['protocol']

    def bound(ref):
        if (set(ref) != {'path', 'sha256'} or ref['path'] not in complete['files']
                or ref['sha256'] != complete['files'][ref['path']]):
            raise ValueError('Unbound recovery artifact reference')
        return root / ref['path']

    def arrays(ref):
        with np.load(bound(ref), allow_pickle=False) as archive:
            return {name: archive[name] for name in archive.files}

    calibration = arrays(report['calibration'])
    saved_predictor = arrays(report['predictor'])
    if read_json(root / 'evidence/original-selection.json') != selection:
        raise ValueError('Root selection differs from frozen training')
    if (calibration.keys() != frozen_cal.keys() or any(
            not np.array_equal(value, frozen_cal[key]) for key, value in calibration.items())
            or set(saved_predictor) != {'coefficients'}
            or not np.array_equal(saved_predictor['coefficients'], predictor)):
        raise ValueError('Root calibration or predictor differs from frozen training')
    errors = dict.fromkeys(('variance', 'strehl', 'open_variance', 'open_strehl'), 0.)
    records, reasons, nonfinite_references = [], [], []
    phases, references = {}, {}
    finite = nonfinite = 0
    target = {'id': 'integrator-04', 'controller': 'integrator', 'parameters': {'gain': .5, 'leak': .99}}
    for index, row in zip(root_indices(spec, seed), report['cases'], strict=True):
        if row != read_json(root / f'evidence/outcomes/{index:03d}.json'):
            raise ValueError('Reported recovery outcome differs from its retained record')
        case = row['case']
        phase_key, reference_key = (seed, case['wind_scale']), (seed, case['wind_scale'], case['fault'])
        if (phases.setdefault(phase_key, row['phase']) != row['phase']
                or references.setdefault(reference_key, row['reference']) != row['reference']):
            raise ValueError('Paired conditions do not share their declared physical streams')
        cfg = episode_config(collection['base'], study, 'evaluation',
                             case['photons'], case['wind_scale'], target)
        cfg.update(seeds=list(spec['seeds']))
        cfg.pop('training_seeds', None)
        if row['config'] != cfg:
            raise ValueError('Recovery configuration differs from frozen physical settings')
        phase = arrays(row['phase'])
        if set(phase) != {'phase'}:
            raise ValueError('Unexpected phase archive')
        phase = phase['phase']
        reference = arrays(row['reference'])
        reference_finite = all(np.isfinite(value).all() for value in reference.values())
        if not reference_finite:
            nonfinite_references.append(case)
        values = {}
        for arm in acquisition.ARMS:
            name = f'{index:03d}-{arm}'
            terminal = read_json(bound(row['arms'][arm]))
            intent = read_json(root / f'evidence/trials/{name}-intent.json')
            if (terminal['identity'] != name or intent['identity'] != name
                    or intent['seed'] != seed or intent['config'] != cfg
                    or intent['warm_kernel'] != terminal['warm_kernel']
                    or terminal['raw']['path'] != f'evidence/trials/{name}.npz'):
                raise ValueError('Recovery dispatch/terminal identity differs')
            outputs = values[arm] = arrays(terminal['raw'])
            is_finite = all(np.isfinite(value).all() for value in outputs.values())
            if terminal['status'] != ('finite' if is_finite else 'nonfinite'):
                raise ValueError('Recovery terminal status differs from raw arrays')
            if arm != 'continued':
                frames = np.arange(cfg['frames'])
                for key, expected in {
                    'handover_triggered': frames >= spec['onset'],
                    'handover_target_active': frames >= spec['onset'] + (arm == 'command_matched'),
                }.items():
                    if key not in outputs or not np.array_equal(outputs[key], expected):
                        raise ValueError('Recovery handover schedule differs from declared arm')
            if not is_finite:
                nonfinite += 1
                continue
            finite += 1
            check_queue(outputs['command'], outputs['applied'], cfg['delay_frames'], cfg['command_clip_rad'])
            actual = physical_metrics(phase, outputs['applied'], calibration['basis'],
                                      calibration['mask'], cfg, case['fault'])
            saved = {key: outputs[key] for key in ('variance', 'strehl')}
            if reference_finite:
                actual.update({'open_' + key: value for key, value in physical_metrics(
                    phase, np.zeros_like(outputs['applied']), calibration['basis'],
                    calibration['mask'], cfg, case['fault']).items()})
                saved.update({'open_' + key: reference[key] for key in ('variance', 'strehl')})
            for key, value in actual.items():
                if saved[key].shape != value.shape or not np.allclose(saved[key], value, atol=ATOL, rtol=RTOL):
                    raise ValueError('Recovery physical accounting differs: ' + name + ': ' + key)
                errors[key] = max(errors[key], float(np.max(np.abs(saved[key] - value))))
        for arm in acquisition.ARMS[1:]:
            result = paired_recovery_outcomes(values['continued'], values[arm], reference, cfg,
                effect_frame=spec['onset'] + cfg['delay_frames'] + (arm == 'command_matched'),
                threshold_multiplier=spec['threshold_multiplier'], consecutive=spec['consecutive_frames'])
            if result != row['contrasts'][arm]:
                raise ValueError('Recovery reduction does not reproduce from raw streams')
            if result['status'] not in ('finite_pair', 'nonfinite_outputs', 'nonfinite_reduction'):
                raise ValueError('Recovery outcome is outside the frozen nonfinite policy')
            valid = result['status'] == 'finite_pair'
            records.append({**case, 'arm': arm, 'status': 'finite_pair' if valid else 'nonfinite_pair',
                            'value': result['outcomes']['normalized_excess_variance'] if valid else None})
            if not valid:
                reasons.append({**case, 'arm': arm, 'raw_status': result['status']})
    final_complete, final_sha = verify_packet(root)
    if final_sha != initial_sha or final_complete != complete:
        raise ValueError('Recovery packet changed during raw reconstruction')
    return records, {**identity, 'finite_arm_records': finite, 'nonfinite_arm_records': nonfinite,
                     'nonfinite_reference_cases': nonfinite_references,
                     'withheld_pairs': reasons, 'max_absolute_error': errors}


def audit(root, controller_run):
    root = Path(root)
    if root.is_symlink() or any(p.is_symlink() for p in root.iterdir()):
        raise ValueError('Symlinked recovery collection')
    root = root.resolve()
    manifest_path, receipt_path = root / 'manifest.json', root / 'acquisition.json'
    manifest_sha, receipt_sha = _hash_file(manifest_path), _hash_file(receipt_path)
    collection, receipt = read_json(manifest_path), read_json(receipt_path)
    expected = collection_manifest(controller_run, collection['exclusion_inventory_sha256'])
    if collection != expected or collection['protocol'] != protocol():
        raise ValueError('Collection differs from the frozen study or current source/runtime')
    seeds = protocol()['seeds']
    if (set(p.name for p in root.iterdir()) != {'.lock', 'manifest.json', 'roots', 'acquisition.json'}
            or {p.name for p in (root / 'roots').iterdir()} != {str(seed) for seed in seeds}
            or receipt['status'] != 'complete_acquisition'
            or receipt['manifest_sha256'] != manifest_sha
            or receipt['completed_roots'] != len(seeds) or receipt['expected_roots'] != len(seeds)):
        raise ValueError('Complete declared recovery cohort required before statistics')
    predictor, frozen_cal, selection = acquisition.load_training(controller_run)
    inputs = capture_inputs(acquisition.ROOT / 'configs/controller_study_v1.yaml', acquisition.ROOT)
    records, audits = [], []
    for seed in seeds:
        pairs, checks = reconstruct_root(root / 'roots' / str(seed), collection, manifest_sha,
                                         seed, inputs['study'], frozen_cal, predictor, selection)
        records.extend(pairs)
        audits.append(checks)
    if receipt['root_packets'] != [
            {key: row[key] for key in ('seed', 'complete_sha256')} for row in audits]:
        raise ValueError('Acquisition receipt differs from audited root identities')
    result = summarize(records)
    if (_hash_file(manifest_path) != manifest_sha or _hash_file(receipt_path) != receipt_sha
            or collection_manifest(controller_run, collection['exclusion_inventory_sha256']) != collection):
        raise ValueError('Study binding changed during reconstruction')
    # Recheck prior packets too: an earlier root can change while a later one is audited.
    for row in audits:
        _, digest = verify_packet(root / 'roots' / str(row['seed']))
        if digest != row['complete_sha256']:
            raise ValueError('Recovery root changed during complete-cohort reconstruction')
    return {'schema': 'noether-recovery-report/1', 'status': 'complete_raw_reconstructed_report',
            'manifest_sha256': manifest_sha, 'acquisition_sha256': receipt_sha,
            'root_audits': audits, 'statistics': result,
            'physical_tolerance': {'absolute': ATOL, 'relative': RTOL},
            'audit_source_sha256': _hash_file(Path(__file__)),
            'physical_reducer_sha256': _hash_file(Path(__file__).with_name('audit_controller_physical_metrics.py')),
            'limitations': ['Scheduled simulated interventions; no detector or hardware result.',
                            'Independent NumPy phase accounting uses the retained basis, pupil and phase.',
                            'This does not independently validate the optical sensor or atmosphere model.',
                            'Nonfinite raw arms are retained and withhold their affected statistical contrast.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--controller-run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.out.resolve().is_relative_to(args.run.resolve()):
        raise ValueError('Use a fresh audit result outside the retained collection')
    _publish(args.out, audit(args.run, args.controller_run))


if __name__ == '__main__':
    main()
