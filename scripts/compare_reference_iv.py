"""Compare instruments on retained development telemetry; acquire no trajectories."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('JAX_ENABLE_X64', '1')

import jax
import numpy as np

import noether_optics
from noether_optics.controller_evidence import _source
from noether_optics.controller_selection import _publish
from noether_optics.identification import (
    controller_filtered_instrument, fit_prediction_pipeline,
    identification_telemetry, nominal_response, one_step_prediction,
)
from noether_optics.simulation import calibration

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT/'results/identification_component_smoke_v1'


def plain(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [plain(item) for item in value]
    return value


def inputs():
    manifest = json.loads((INPUT/'manifest.json').read_text())
    paths = [INPUT/'manifest.json', INPUT/'verify.py']
    for item in manifest['artifacts']:
        if Path(item['file']).name != item['file']:
            raise ValueError('Invalid retained artifact path')
        paths.append(INPUT/item['file'])
    hashes = {}
    for path in paths:
        name = path.relative_to(ROOT).as_posix()
        data = path.read_bytes()
        committed = subprocess.check_output(['git', 'show', f'HEAD:{name}'], cwd=ROOT)
        if data != committed:
            raise ValueError(f'Retained input differs from committed bytes: {name}')
        hashes[name] = hashlib.sha256(data).hexdigest()
    return manifest, hashes


def run(out):
    if Path(noether_optics.__file__).resolve().parent != ROOT/'src/noether_optics':
        raise RuntimeError('Imported package must resolve to the checked source repository')
    source = _source(ROOT)
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT):
        raise RuntimeError('Reanalysis requires clean committed source')
    if not jax.config.jax_enable_x64 or jax.default_backend() != 'cpu':
        raise RuntimeError('Reanalysis requires float64 CPU execution')
    original, hashes = inputs()
    subprocess.run([sys.executable, str(INPUT/'verify.py')], cwd=ROOT, check=True)
    ancestor = out.resolve().parent
    while not ancestor.exists(): ancestor = ancestor.parent
    # New derived arrays occupy under 1 MiB. No study/acquisition gate changes.
    if shutil.disk_usage(ancestor).free < 256*1024**2:
        raise RuntimeError('Reanalysis requires 256 MiB free space')
    out.mkdir(parents=True, exist_ok=False)
    spec, base = original['config'], original['base']
    evidence = json.loads((INPUT/'evidence.json').read_text())
    cal = calibration(base)
    # Reanalysis must retain the acquisition coordinate system even if the
    # current implementation of calibration changes in a later revision.
    if base['sensor'] != 'linear':
        raise ValueError('Retained comparison requires the declared linear calibration')
    np.testing.assert_array_equal(cal['reference'], np.zeros(base['modes']))
    np.testing.assert_array_equal(cal['interaction'], np.eye(base['modes']))
    np.testing.assert_array_equal(cal['reconstructor'],
                                  np.eye(base['modes'])/(1+base['reconstructor_ridge']))
    response0 = nominal_response(cal)  # declared model, never the evaluator response
    start, delay = base['burn_in'], base['delay_frames']
    artifacts, models, forecasts = [], [], []

    def retain(name, data):
        path = out/(name+'.npz')
        with path.open('xb') as handle:
            np.savez_compressed(handle, **data)
        artifacts.append({'file': path.name, 'bytes': path.stat().st_size,
                          'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})

    def read(seed, rho, drift=False):
        path = INPUT/f'{seed}-rho{rho:g}-drift{int(drift)}-probe1.npz'
        # Never load phase truth or simulator estimates into the fitting adapter.
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key] for key in ('measurement', 'command', 'applied', 'probe_requested')}

    for rho in spec['correlations']:
        rows, original_rows = [], []
        for seed in spec['training_roots']:
            raw = read(seed, rho)
            telemetry = identification_telemetry(raw, cal, delay_frames=delay, burn_in=start)
            original_rows.append(telemetry)
            reference = controller_filtered_instrument(raw['probe_requested'], response0,
                gain=base['gain'], leak=base['leak'], delay_frames=delay,
                command_clip_rad=base['command_clip_rad'])
            retain(f'reference-{seed}-rho{rho:g}', reference)
            rows.append(dict(telemetry, instrument=reference['instrument'][start:]))
        for retained in [row['model'] for row in evidence['models'] if row['rho'] == rho]:
            reproduced = fit_prediction_pipeline(original_rows, method=retained['method'],
                ridge=spec['predictor_ridge'], minimum_relevance=spec['minimum_relevance'],
                rcond=spec['rcond'])
            if reproduced['status'] != retained['status']:
                raise ValueError('Reanalysis does not reproduce the retained baseline status')
            for field in ('response', 'coefficients'):
                np.testing.assert_allclose(reproduced[field], retained[field], rtol=1e-12, atol=1e-14)
        model = fit_prediction_pipeline(rows, method='iv', ridge=spec['predictor_ridge'],
            minimum_relevance=spec['minimum_relevance'], rcond=spec['rcond'])
        models.append({'rho': rho, 'instrument': 'controller_filtered', 'model': model,
            'response_relative_error': (float(np.linalg.norm(model['response']-evidence['reference_response'])
                / np.linalg.norm(evidence['reference_response'])) if model['status'] == 'ready' else None)})
        for old in [row for row in evidence['forecasts'] if row['rho'] == rho]:
            raw = read(old['root'], rho, old['drift'])
            telemetry = identification_telemetry(raw, cal, delay_frames=delay)
            record = dict(old)  # Preserve all baseline outcomes and probing costs.
            record['predictions'] = dict(old['predictions'])
            if model['status'] != 'ready':
                result = {'status': 'withheld', 'reason': model['reason']}
            else:
                prediction = one_step_prediction(model, telemetry['residual'], telemetry['applied'],
                                                  raw['command'], delay_frames=delay)
                indices = prediction['target_indices']
                error = prediction['prediction']-telemetry['residual'][indices]
                retain(f"forecast-{old['root']}-rho{rho:g}-drift{int(old['drift'])}",
                       prediction | {'error': error})
                windows = {'post_burn': indices >= start}
                if old['drift']:
                    windows.update(before_drift=(indices >= start) & (indices < spec['drift']['onset']),
                                   settled_drift=indices >= spec['drift']['onset']+spec['drift']['drift_frames'])
                result = {'status': 'ready', 'mse_by_window': {
                    name: float(np.mean(error[selected]**2)) for name, selected in windows.items()}}
            record['predictions']['controller_filtered_iv'] = result
            forecasts.append(record)
    _publish(out/'evidence.json', plain({'scope': 'exploratory reanalysis of existing development recordings',
        'nominal_response': response0, 'models': models, 'forecasts': forecasts,
        'baseline_models': evidence['models'], 'reference_response': evidence['reference_response']}))
    path = out/'evidence.json'
    artifacts.append({'file': path.name, 'bytes': path.stat().st_size,
                      'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    if (_source(ROOT) != source or inputs()[1] != hashes
            or subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT)):
        raise RuntimeError('Source or retained inputs changed; no completion manifest published')
    _publish(out/'manifest.json', {'analysis_source': source,
        'acquisition_source_commit': original['source_commit'], 'input_sha256': hashes,
        'runtime': {name: importlib.metadata.version(name) for name in ('numpy', 'jax', 'jaxlib', 'dLux')},
        'jax_backend': jax.default_backend(), 'jax_x64': bool(jax.config.jax_enable_x64),
        'artifacts': artifacts, 'new_trajectories': 0})
    print(json.dumps({'analysis_commit': source['commit'], 'artifacts': len(artifacts),
                      'model_statuses': [row['model']['status'] for row in models]}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args().out)
