"""Reconstruct this fixed linear-sensor experiment using NumPy, without acquisition."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import yaml


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-12)


def verify(directory: Path):
    manifest = json.loads((directory / 'manifest.json').read_text())
    inventory = manifest['source_inventory']
    if manifest['source_commit'] != inventory['commit']:
        raise ValueError('Inconsistent source identity')
    root = Path(__file__).resolve().parents[2]
    source_bytes = {}
    for name, digest in inventory['files'].items():
        data = subprocess.check_output(['git', 'show', f"{inventory['commit']}:{name}"], cwd=root)
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f'Source digest mismatch: {name}')
        source_bytes[name] = data
    spec_path = 'configs/identification_smoke.yaml'
    source_spec = yaml.safe_load(source_bytes[spec_path])
    input_hashes = {name: hashlib.sha256(source_bytes[name]).hexdigest()
                    for name in (spec_path, source_spec['base'])}
    if manifest['input_hashes'] != input_hashes:
        raise ValueError('Input configuration digest mismatch')
    resolved_base = yaml.safe_load(source_bytes[source_spec['base']])
    resolved_base.update(source_spec['overrides'])
    for key in ('seed', 'seeds', 'training_seeds'):
        resolved_base.pop(key, None)
    resolved_base['faults'].update(onset=resolved_base['frames'], end=resolved_base['frames'])
    if manifest['config'] != source_spec or manifest['base'] != resolved_base:
        raise ValueError('Embedded configuration does not reproduce the committed inputs')
    names = []
    for item in manifest['artifacts']:
        name = item['file']
        if Path(name).name != name or name in names:
            raise ValueError('Invalid or repeated artifact name')
        names.append(name)
        data = (directory / name).read_bytes()
        if len(data) != item['bytes'] or hashlib.sha256(data).hexdigest() != item['sha256']:
            raise ValueError(f'Artifact digest mismatch: {name}')
    if len(names) != 35:
        raise ValueError('Expected the complete 35-artifact development run')
    evidence = json.loads((directory / 'evidence.json').read_text())
    spec, base = manifest['config'], manifest['base']
    if base['sensor'] != 'linear' or base['modes'] != 4:
        raise ValueError('This reconstruction is specific to the retained linear experiment')
    start, delay, modes = base['burn_in'], base['delay_frames'], base['modes']
    # The full linear sensor has zero reference and identity interaction matrix.
    scale = 1 / (1 + base['reconstructor_ridge'])
    close(evidence['reference_response'], np.eye(modes) * scale)

    def raw(root, rho, drift=False, probe=True):
        name = f'{root}-rho{rho:g}-drift{int(drift)}-probe{int(probe)}.npz'
        if name not in names:
            raise ValueError(f'Unbound recording: {name}')
        with np.load(directory / name, allow_pickle=False) as archive:
            row = {key: archive[key] for key in archive.files}
        if not all(np.isfinite(value).all() for value in row.values()):
            raise ValueError(f'Nonfinite recording: {name}')
        close(row['combined_requested'], row['feedback_requested'] + row['probe_requested'])
        close(row['command'], np.clip(row['combined_requested'], -base['command_clip_rad'], base['command_clip_rad']))
        np.testing.assert_array_equal(row['applied'][:delay], 0)
        np.testing.assert_array_equal(row['applied'][delay:], row['command'][:-delay])
        return row

    models = {}
    for record in evidence['models']:
        rho, model = record['rho'], record['model']
        method = model['method']
        if model['status'] != 'ready':
            raise ValueError('This retained run has six ready models')
        training, fitted = [], []
        for seed in spec['training_roots']:
            row = raw(seed, rho)
            x, y = row['applied'][start:], row['measurement'][start:] * scale
            z = np.zeros_like(row['probe_requested'])
            z[delay:] = row['probe_requested'][:-delay]
            z = z[start:]
            xc, yc, zc = (value - value.mean(axis=0) for value in (x, y, z))
            if method == 'nominal':
                response = np.eye(modes)
            elif method == 'ols':
                response = -np.linalg.lstsq(xc, yc, rcond=spec['rcond'])[0]
            elif method == 'iv':
                # Explicit two-stage least squares, independent of the fitted
                # implementation's SVD-basis projection.
                projected = zc @ np.linalg.lstsq(zc, xc, rcond=spec['rcond'])[0]
                response = -np.linalg.lstsq(projected, yc, rcond=spec['rcond'])[0]
            else:
                raise ValueError('Unknown fitted method')
            fitted.append(response)
            training.append((x, y))
        close(fitted, [fit['response'] for fit in model['episode_fits']])
        response = np.mean(fitted, axis=0)
        close(response, model['response'])
        coefficients = []
        for x, y in training:
            q = np.linalg.solve(response.T, y.T).T + x
            per_mode = []
            for mode in range(modes):
                design = np.column_stack((q[1:-1, mode], q[:-2, mode]))
                # Ridge regression as an augmented least-squares problem.
                augmented = np.vstack((design, np.sqrt(spec['predictor_ridge']) * np.eye(2)))
                target = np.concatenate((q[2:, mode], np.zeros(2)))
                per_mode.append(np.linalg.lstsq(augmented, target, rcond=None)[0])
            coefficients.append(per_mode)
        close(coefficients, model['episode_coefficients'])
        coefficients = np.mean(coefficients, axis=0)
        close(coefficients, model['coefficients'])
        reference = np.asarray(evidence['reference_response'])
        close(np.linalg.norm(response - reference) / np.linalg.norm(reference), record['response_relative_error'])
        models[rho, method] = response, coefficients
    if len(models) != 6:
        raise ValueError('Incomplete fitted-model collection')

    for record in evidence['forecasts']:
        seed, rho, drift = record['root'], record['rho'], record['drift']
        row, control = raw(seed, rho, drift), raw(seed, rho, drift, probe=False)
        close(np.sqrt(np.mean(row['probe_requested'][start:]**2)), record['probe_rms_rad'])
        close(np.mean(row['variance'][start:] - control['variance'][start:]), record['paired_variance_change_rad2'])
        close(row['saturated'][start:].mean(), record['saturation_fraction'])
        x, y, issued = row['applied'], row['measurement'] * scale, row['command']
        indices = np.arange(max(2, delay), len(y))
        for method in ('nominal', 'ols', 'iv'):
            response, coefficients = models[rho, method]
            # Only past residual/application rows and already issued commands.
            current = np.linalg.solve(response.T, y[indices-1].T).T + x[indices-1]
            previous = np.linalg.solve(response.T, y[indices-2].T).T + x[indices-2]
            prediction = (current * coefficients[:, 0] + previous * coefficients[:, 1]
                          - issued[indices-delay]) @ response
            error = prediction - y[indices]
            name = f'forecast-{seed}-rho{rho:g}-drift{int(drift)}-{method}.npz'
            if name not in names:
                raise ValueError(f'Unbound forecast: {name}')
            with np.load(directory / name, allow_pickle=False) as saved:
                np.testing.assert_array_equal(indices, saved['target_indices'])
                close(prediction, saved['prediction'])
                close(error, saved['error'])
            windows = {'post_burn': indices >= start}
            if drift:
                windows.update(before_drift=(indices >= start) & (indices < spec['drift']['onset']),
                               settled_drift=indices >= spec['drift']['onset'] + spec['drift']['drift_frames'])
            for name, selected in windows.items():
                close(np.mean(error[selected]**2), record['predictions'][method]['mse_by_window'][name])
    if len(evidence['forecasts']) != 6:
        raise ValueError('Incomplete development forecast collection')
    print(json.dumps({'verified_artifacts': len(names), 'reconstructed_models': len(models),
                      'reconstructed_forecasts': 18, 'source_commit': manifest['source_commit']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path(__file__).resolve().parent)
    verify(parser.parse_args().directory)
