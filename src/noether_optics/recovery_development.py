"""Fresh-only development acquisition for three scheduled recovery arms.

This pilot estimates resource use and paired variability. It has no held-out
cohort, detector, stopping-on-success rule or controller acceptance criterion.
Run under an external wall/memory/storage supervisor with exclusive compute.
"""
from __future__ import annotations

import argparse
import hashlib
from itertools import product
import json
from pathlib import Path
import shutil
import time

import numpy as np

from .controller_evidence import (_hash_file, _source, capture_inputs,
                                  save_arrays, save_metadata)
from .controller_selection import _bytes, _publish

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (2026111001, 2026111002, 2026111003, 2026111004)
ARMS = ('continued', 'issued_state', 'command_matched')
MIN_FREE_BYTES = 105033365984
INPUTS = {
    'training.json': 'fa92d7f58709bb6c64a740d87192206422fd8a00eb2e1fd9a0a866a9222c87eb',
    'selection.json': 'af0e54d45635fe008f8f14476cb727fa0e4abe79961096e53e74dea53fffbb3f',
    'evidence/provenance/base.json': 'be55790a001a36162e5c43c66b0ea10837f2d1398e0bce5935bfadd45045ebad',
    'evidence/provenance/study.json': '6f5df8fe1074f5369e69205423b4e977ced9d332af5c9bccb721d0a30bb8a8b6',
    'evidence/training/predictive-01/parameters.npz': '718ae87c8bf66d51d164c22a5f42864edbca8a1f61cbb4b31eec306490b681f5',
    'evidence/training/learned-00/calibration.npz': '40813ecb3b0ea609f8aaa2514f72635efeb169a81ce9e0e0a2498f75dbd4ffee',
}


def protocol(*, smoke=False):
    if type(smoke) is not bool:
        raise ValueError('smoke must be boolean')
    return {
        'schema': 'noether-recovery-development/1', 'role': 'development',
        'smoke': smoke, 'seeds': list(SEEDS[:1] if smoke else SEEDS),
        'noise_offset': 100000, 'photons': [1000.] if smoke else [1000., 1000000.],
        'wind_scales': [.5] if smoke else [.5, 2.],
        'faults': ['none'] if smoke else ['none', 'optical_gain', 'vibration'],
        'arms': list(ARMS), 'frames': 512, 'burn_in': 128, 'onset': 256, 'end': 512,
        'threshold_multiplier': 1.1, 'consecutive_frames': 32,
        'frozen_training_inputs': INPUTS,
        'predictor_id': 'predictive-01', 'target_id': 'integrator-04',
        'minimum_free_bytes': MIN_FREE_BYTES,
        'interpretation': 'Development only; no confidence interval, detector or hardware claim.',
        'timing': 'First call includes JIT; subsequent calls are warm. Host conversion included; archive writes excluded.',
        'stream_inventory_scope': 'Development namespace; not a complete historical exclusion certificate.',
    }


def cases(spec):
    return [dict(seed=seed, photons=photons, wind_scale=wind, fault=fault)
            for seed, photons, wind, fault in product(
                spec['seeds'], spec['photons'], spec['wind_scales'], spec['faults'])]


class DevelopmentRun:
    """Artifact adapter reusing the existing exclusive numeric/JSON writer."""

    def __init__(self, root, manifest):
        self.root = Path(root).resolve()
        self.root.mkdir(exist_ok=False)
        _publish(self.root / 'manifest.json', manifest)
        self.references = {'manifest.json': _hash_file(self.root / 'manifest.json')}

    def artifact(self, relative):
        path = self.root / relative
        if path.is_symlink() or not path.resolve().is_relative_to(self.root):
            raise ValueError('Artifact path escapes the development run')
        digest = _hash_file(path)
        if relative in self.references and self.references[relative] != digest:
            raise ValueError('Previously published development artifact changed')
        self.references[relative] = digest
        return {'path': relative, 'sha256': digest}

    def verify(self):
        actual = {p.relative_to(self.root).as_posix() for p in self.root.rglob('*') if p.is_file()}
        if actual != self.references.keys():
            raise ValueError('Development artifact inventory changed')
        for name, expected in self.references.items():
            path = self.root / name
            if path.is_symlink() or _hash_file(path) != expected:
                raise ValueError('Development artifact changed: ' + name)


def load_training(run):
    """Load only exact frozen training/selection inputs, never old evaluation."""
    run = Path(run).resolve()
    for name, expected in INPUTS.items():
        path = run / name
        if not path.resolve().is_relative_to(run) or _hash_file(path) != expected:
            raise ValueError('Frozen recovery input changed: ' + name)
    with np.load(run / 'evidence/training/predictive-01/parameters.npz', allow_pickle=False) as saved:
        if saved.files != ['coefficients']:
            raise ValueError('Unexpected frozen predictor archive')
        predictor = saved['coefficients']
    with np.load(run / 'evidence/training/learned-00/calibration.npz', allow_pickle=False) as saved:
        calibration = {name: saved[name] for name in saved.files}
    selected = json.loads((run / 'selection.json').read_bytes())
    # The artifact hash, rather than this schema access, is the selection binding.
    return predictor, calibration, selected


def acquire_arm(run, identity, config, execute, phases, seed, *, warm):
    """Durable intent precedes dispatch; nonfinite output still receives raw data."""
    stem = 'evidence/trials/' + identity
    save_metadata(run, stem + '-intent.json', {
        'identity': identity, 'seed': seed, 'config': config, 'warm_kernel': warm})
    before = time.perf_counter()
    try:
        outputs = {name: np.asarray(value) for name, value in execute(phases, seed).items()}
        elapsed = time.perf_counter() - before
    except BaseException as error:
        save_metadata(run, stem + '-interrupted.json', {
            'identity': identity, 'status': 'interrupted',
            'exception_type': type(error).__name__, 'seconds': time.perf_counter() - before})
        raise
    raw = save_arrays(run, stem + '.npz', outputs)
    finite = all(np.isfinite(value).all() for value in outputs.values())
    terminal = save_metadata(run, stem + '.json', {
        'identity': identity, 'status': 'finite' if finite else 'nonfinite',
        'raw': raw, 'kernel_and_conversion_seconds': elapsed, 'warm_kernel': warm})
    return outputs, terminal


def run_development(output, controller_run, *, smoke=False):
    from .atmosphere import phase_sequence
    from .controller_physics import controller_kernel, episode_config, paired_open_loop
    from .recovery import scheduled_handover_kernel
    from .recovery_metrics import paired_recovery_outcomes
    from .simulation import calibration

    output, controller_run = Path(output).resolve(), Path(controller_run).resolve()
    if output.exists() or output.is_relative_to(controller_run):
        raise ValueError('Use a fresh output outside the retained controller study')
    if not output.parent.is_dir() or shutil.disk_usage(output.parent).free < MIN_FREE_BYTES:
        raise RuntimeError('Development output requires the declared free-space reserve')
    spec = protocol(smoke=smoke)
    predictor, frozen_cal, selection = load_training(controller_run)
    inputs = capture_inputs(ROOT / 'configs/controller_study_v1.yaml', ROOT)
    for name in ('base', 'study'):
        if hashlib.sha256(_bytes(inputs[name])).hexdigest() != INPUTS[f'evidence/provenance/{name}.json']:
            raise ValueError('Development configuration differs from the frozen controller study')
    run = DevelopmentRun(output, {'protocol': spec, 'source': inputs['source'],
                                 'runtime': inputs['runtime'], 'base': inputs['base']})
    before = time.perf_counter()
    try:
        # Recreate callables, then require exact numeric agreement with the saved
        # calibration before any atmosphere or controller acquisition.
        cal = calibration(inputs['base'])
        numeric_cal = {key: np.asarray(value) for key, value in cal.items() if not callable(value)}
        calibration_ref = save_arrays(run, 'evidence/calibration.npz', numeric_cal)
        if numeric_cal.keys() != frozen_cal.keys() or any(
                not np.array_equal(value, frozen_cal[key]) for key, value in numeric_cal.items()):
            raise ValueError('Reconstructed calibration differs from frozen numeric arrays')
        predictor_ref = save_arrays(run, 'evidence/predictor.npz', {'coefficients': predictor})
        target = {'id': 'integrator-04', 'controller': 'integrator',
                  'parameters': {'gain': .5, 'leak': .99}}
        # The full selection file is retained even though only this fixed pair is used.
        save_metadata(run, 'evidence/original-selection.json', selection)
        kernels, phases_by_key, references = {}, {}, {}
        outcomes, terminals = [], []
        for index, case in enumerate(cases(spec)):
            if shutil.disk_usage(output).free < MIN_FREE_BYTES:
                raise RuntimeError('Free-space reserve would be violated before the next case')
            cfg = episode_config(inputs['base'], inputs['study'], 'evaluation',
                                 case['photons'], case['wind_scale'], target)
            cfg.update(seeds=list(spec['seeds']))
            cfg.pop('training_seeds', None)
            key = (case['seed'], case['wind_scale'])
            if key not in phases_by_key:
                phase = np.asarray(phase_sequence(cfg, case['seed']))
                ref = save_arrays(run, f'evidence/atmospheres/{len(phases_by_key):03d}.npz', {'phase': phase})
                phases_by_key[key] = phase, ref
            phase, phase_ref = phases_by_key[key]
            reference_key = (*key, case['fault'])
            if reference_key not in references:
                ref_values = {name: np.asarray(value) for name, value in paired_open_loop(
                    phase, cfg, cal, case['fault']).items()}
                ref = save_arrays(run, f'evidence/references/{len(references):03d}.npz', ref_values)
                references[reference_key] = ref_values, ref
            reference, reference_ref = references[reference_key]
            arm_outputs = {}
            arm_refs = {}
            for arm in ARMS:
                kernel_key = (case['photons'], case['wind_scale'], case['fault'], arm)
                warm = kernel_key in kernels
                if not warm:
                    if arm == 'continued':
                        kernels[kernel_key] = controller_kernel(cfg, cal, 'predictive', case['fault'], predictor)
                    else:
                        kernels[kernel_key] = scheduled_handover_kernel(
                            cfg, cal, predictor, frame=spec['onset'], mode=arm, fault=case['fault'])
                identity = f'{index:03d}-{arm}'
                values, terminal = acquire_arm(run, identity, cfg, kernels[kernel_key],
                                               phase, case['seed'], warm=warm)
                arm_outputs[arm], arm_refs[arm] = values, terminal
                terminals.append(terminal)
            contrasts = {}
            for arm in ARMS[1:]:
                effect = spec['onset'] + cfg['delay_frames'] + (arm == 'command_matched')
                contrasts[arm] = paired_recovery_outcomes(
                    arm_outputs['continued'], arm_outputs[arm], reference, cfg,
                    effect_frame=effect, threshold_multiplier=spec['threshold_multiplier'],
                    consecutive=spec['consecutive_frames'])
            row = {'case': case, 'config': cfg, 'phase': phase_ref, 'reference': reference_ref,
                   'arms': arm_refs, 'contrasts': contrasts}
            outcomes.append(row)
            save_metadata(run, f'evidence/outcomes/{index:03d}.json', row)
        if _source(ROOT) != inputs['source']:
            raise ValueError('Source changed during development acquisition')
        for name, digest in INPUTS.items():
            if _hash_file(controller_run / name) != digest:
                raise ValueError('Frozen training input changed during acquisition')
        run.verify()
        status = 'complete_finite_development' if all(
            result['status'] == 'finite_pair' for row in outcomes for result in row['contrasts'].values()
        ) else 'complete_development_with_withheld_outcomes'
        result = {
            'schema': spec['schema'], 'status': status, 'smoke': smoke,
            'cases': outcomes, 'terminal_records': len(terminals),
            'expected_terminal_records': len(cases(spec)) * len(ARMS),
            'calibration': calibration_ref, 'predictor': predictor_ref,
            'acquisition_inclusive_seconds': time.perf_counter() - before,
            'interpretation': spec['interpretation'],
        }
        _publish(output / 'report.json', result)
        # Bind every retained file; result absence/partial intent never means success.
        inventory = {p.relative_to(output).as_posix(): _hash_file(p)
                     for p in sorted(output.rglob('*')) if p.is_file()}
        _publish(output / 'complete.json', {'status': status, 'files': inventory})
        return result
    except BaseException as error:
        _publish(output / 'stopped.json', {'status': 'stopped', 'exception_type': type(error).__name__,
                                          'seconds': time.perf_counter() - before})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controller-run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    result = run_development(args.out, args.controller_run, smoke=args.smoke)
    print(json.dumps({key: result[key] for key in ('status', 'terminal_records', 'smoke')}))


if __name__ == '__main__':
    main()
