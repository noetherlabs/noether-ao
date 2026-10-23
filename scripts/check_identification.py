"""Retain bounded component-check evidence; this does not run a held-out study."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('JAX_ENABLE_X64', '1')

import jax
import numpy as np

import noether_optics
from noether_optics.artifacts import load_config
from noether_optics.atmosphere import phase_sequence
from noether_optics.controller_evidence import _source
from noether_optics.controller_selection import _publish
from noether_optics.identification import (fit_prediction_pipeline, fit_response,
    identification_telemetry, instrumented_kernel, one_step_prediction, stationary_noise)
from noether_optics.simulation import calibration

ROOT = Path(__file__).resolve().parents[1]


def plain(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [plain(item) for item in value]
    return value


def run(out: Path):
    """Use only fixed development identities, with every case and failure retained."""
    if Path(noether_optics.__file__).resolve().parent != ROOT/'src/noether_optics':
        raise RuntimeError('Imported package must resolve to the checked source repository')
    inventory = _source(ROOT)
    source = inventory['commit']
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=normal'], cwd=ROOT):
        raise RuntimeError('Component evidence requires clean committed source')
    if not jax.config.jax_enable_x64 or jax.default_backend() != 'cpu':
        raise RuntimeError('Component checks require the declared float64 CPU runtime')
    config_path = ROOT/'configs/identification_smoke.yaml'
    spec = load_config(config_path)
    base_path = ROOT/spec['base']
    base = load_config(base_path)
    input_hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (config_path, base_path)}
    base.update(spec['overrides'])
    for key in ('seed', 'seeds', 'training_seeds'):
        base.pop(key, None)
    base['faults'].update(onset=base['frames'], end=base['frames'])
    roots = spec['training_roots']+spec['forecast_roots']
    streams = [root+offset for root in roots for offset in
               (0, 100000, spec['probe_seed_offset'], spec['noise_seed_offset'])]
    if len(streams) != len(set(streams)) or not all(0 <= seed < 2**32 for seed in streams):
        raise ValueError('Overlapping or invalid development random-stream identities')
    ancestor = out.resolve().parent
    while not ancestor.exists(): ancestor = ancestor.parent
    # Explicit, conservative allowance for this component check only; no study
    # resource gate is changed. Raw uncompressed logical payload is under 64 MiB.
    if shutil.disk_usage(ancestor).free < 512*1024**2:
        raise RuntimeError('Component check requires 512 MiB free space')
    out.mkdir(parents=True, exist_ok=False)
    artifacts = []

    def retain(name, arrays):
        path = out/(name+'.npz')
        with path.open('xb') as handle:
            np.savez_compressed(handle, **{key: np.asarray(value) for key, value in sorted(arrays.items())})
        artifacts.append({'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                          'bytes': path.stat().st_size})

    def write(name, data):
        _publish(out/name, plain(data))

    cal = calibration(base)
    execute = instrumented_kernel(base, cal)
    changed = deepcopy(base); changed['faults'].update(spec['drift'])
    execute_drift = instrumented_kernel(changed, cal, fault='optical_gain')
    # Evaluate the physical masked, piston-removed response, including the actual
    # reconstructor. This reference is never an input to a fitted candidate.
    basis, mask = np.asarray(cal['basis']), np.asarray(cal['mask'])
    centered = basis*mask
    centered -= mask*(centered.sum(axis=(-2, -1))/mask.sum())[:, None, None]
    physical_response = np.stack([np.asarray(cal['plant'](mode)) for mode in centered])
    reference_response = physical_response @ np.asarray(cal['reconstructor']).T
    models, forecasts, checks = [], [], []
    start = base['burn_in']; shape = (base['frames'], base['modes'])

    def acquire(root, rho, drift=False, zero_probe=False):
        phase = phase_sequence(base, root)
        probe = np.random.default_rng(root+spec['probe_seed_offset']).normal(
            scale=spec['probe_standard_deviation_rad'], size=shape)
        if zero_probe: probe[:] = 0
        noise = stationary_noise(*shape, seed=root+spec['noise_seed_offset'], rho=rho,
                                 standard_deviation=spec['noise_standard_deviation'])
        values = (execute_drift if drift else execute)(phase, root, probe, noise)
        host = {key: np.asarray(value) for key, value in values.items()}
        label = f'{root}-rho{rho:g}-drift{int(drift)}-probe{int(not zero_probe)}'
        retain(label, host | {'phase_truth': np.asarray(phase)})
        checks.append({'case': label, 'finite': all(np.isfinite(value).all() for value in host.values())})
        return host

    for rho in spec['correlations']:
        rows = [identification_telemetry(acquire(root, rho), cal,
                                        delay_frames=base['delay_frames'], burn_in=start)
                for root in spec['training_roots']]
        fitted = {method: fit_prediction_pipeline(rows, method=method,
                    ridge=spec['predictor_ridge'], minimum_relevance=spec['minimum_relevance'],
                    rcond=spec['rcond']) for method in ('nominal', 'ols', 'iv')}
        for method, model in fitted.items():
            models.append({'rho': rho, 'model': model,
                           'response_relative_error': (float(np.linalg.norm(model['response']-reference_response)
                            / np.linalg.norm(reference_response)) if model['status'] == 'ready' else None)})
        zero = fit_response(rows[0]['residual'], rows[0]['applied'], np.zeros_like(rows[0]['instrument']), method='iv')
        checks.append({'rho': rho, 'zero_excitation_withheld': zero['status'] == 'withheld'})
        for drift in ([False, True] if rho == spec['correlations'][-1] else [False]):
            for root in spec['forecast_roots']:
                values, control = acquire(root, rho, drift), acquire(root, rho, drift, zero_probe=True)
                telemetry = identification_telemetry(values, cal, delay_frames=base['delay_frames'])
                record = {'root': root, 'rho': rho, 'drift': drift,
                    'probe_rms_rad': float(np.sqrt(np.mean(values['probe_requested'][start:]**2))),
                    'paired_variance_change_rad2': float(np.mean(values['variance'][start:]-control['variance'][start:])),
                    'saturation_fraction': float(values['saturated'][start:].mean()), 'predictions': {}}
                for method, model in fitted.items():
                    if model['status'] != 'ready':
                        record['predictions'][method] = {'status': 'withheld', 'reason': model['reason']}
                        continue
                    prediction = one_step_prediction(model, telemetry['residual'], telemetry['applied'],
                                                      values['command'], delay_frames=base['delay_frames'])
                    indices = prediction['target_indices']
                    error = prediction['prediction']-telemetry['residual'][indices]
                    retain(f'forecast-{root}-rho{rho:g}-drift{int(drift)}-{method}', prediction | {'error': error})
                    # Before onset and after settled gain are distinct; no adaptation
                    # or pooled-stationary interpretation of the transition window.
                    windows = {'post_burn': indices >= start}
                    if drift:
                        windows |= {'before_drift': (indices >= start) & (indices < spec['drift']['onset']),
                                    'settled_drift': indices >= spec['drift']['onset']+spec['drift']['drift_frames']}
                    record['predictions'][method] = {'status': 'ready',
                        'mse_by_window': {name: float(np.mean(error[mask]**2)) for name, mask in windows.items()}}
                forecasts.append(record)
    write('evidence.json', {'scope': 'component smoke on development identities; no held-out performance claim',
                           'reference_response': reference_response, 'models': models,
                           'forecasts': forecasts, 'checks': checks})
    evidence = out/'evidence.json'
    artifacts.append({'file': evidence.name, 'sha256': hashlib.sha256(evidence.read_bytes()).hexdigest(),
                      'bytes': evidence.stat().st_size})
    # Outputs belong in the ignored runs directory until reviewed for publication.
    if (_source(ROOT) != inventory
            or subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=normal'], cwd=ROOT)
            or any(hashlib.sha256((ROOT/path).read_bytes()).hexdigest() != digest
                   for path, digest in input_hashes.items())):
        raise RuntimeError('Source or input changed during acquisition; no completion manifest published')
    write('manifest.json', {'source_commit': source, 'source_inventory': inventory,
         'config': spec, 'base': base,
         'input_hashes': input_hashes,
         'runtime': {name: importlib.metadata.version(name) for name in ('numpy', 'jax', 'jaxlib', 'dLux')},
         'jax_backend': jax.default_backend(), 'jax_x64': bool(jax.config.jax_enable_x64),
         'artifacts': artifacts, 'all_implementation_checks_pass': all(
             row.get('finite', row.get('zero_excitation_withheld', False)) for row in checks)})
    print(json.dumps({'source_commit': source, 'artifacts': len(artifacts),
                      'all_implementation_checks_pass': all(row.get('finite', row.get('zero_excitation_withheld', False)) for row in checks)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args().out)
