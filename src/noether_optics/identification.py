"""Closed-loop response identification from measured modal telemetry.

    independent requested probe -> controller + clip + delay -> applied command
                              instruments Z                 regressors X
    residual observation Y = disturbance - X B + noise
        -> nominal / ordinary least squares / instrumental variables response
        -> measured pseudo-open-loop history -> existing two-tap predictor

This module fits a static command-to-observation response, not atmospheric
dynamics by instrumental variables. Exogeneity is an experimental assumption;
rank and relevance diagnostics cannot establish it from the same recording.
"""
from __future__ import annotations

from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np

from .simulation import loop, predict_fit


def _matrix(value, name):
    array = np.asarray(value)
    if array.ndim != 2 or array.dtype.kind not in 'fiu' or min(array.shape) < 1:
        raise ValueError(f'{name} must be a nonempty real numeric matrix')
    with np.errstate(over='ignore', invalid='ignore'):
        array = array.astype(np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f'{name} must be finite in float64')
    return array


def stationary_noise(frames: int, modes: int, *, seed: int, rho: float,
                     standard_deviation: float) -> np.ndarray:
    """Generate stationary Gaussian AR(1) noise with covariance sigma² rho^|i-j|.

    This is a declared additive disturbance, not a detector-noise replacement.
    Channels are independent. The first sample has stationary variance sigma².
    """
    if (type(frames) is not int or frames < 1 or type(modes) is not int or modes < 1
            or type(seed) is not int or not 0 <= seed < 2**32
            or not np.isscalar(rho) or not np.isrealobj(rho)
            or not np.isscalar(standard_deviation) or not np.isrealobj(standard_deviation)
            or not np.isfinite(rho) or abs(rho) >= 1
            or not np.isfinite(standard_deviation) or standard_deviation < 0):
        raise ValueError('Invalid stationary AR(1) dimensions, seed or parameters')
    white = np.random.default_rng(seed).standard_normal((frames, modes))
    result = np.empty_like(white)
    result[0] = standard_deviation * white[0]
    innovation_scale = standard_deviation * np.sqrt((1-rho)*(1+rho))
    for index in range(1, frames):
        result[index] = rho * result[index-1] + innovation_scale * white[index]
    if not np.isfinite(result).all():
        raise ValueError('AR(1) noise exceeds the numerical range')
    return result


def instrumented_kernel(config: dict, cal: dict, *, fault: str = 'none'):
    """Reuse the physical loop with finite probes and additive sensor noise.

    Only the linear modal sensor and an integrator collection controller are
    qualified by this adapter. Instrumented pyramid trials require a separate
    observation model. Commands are radians on the reference pupil; additive
    noise is in the linear sensor's modal output units. Probe and noise arrays
    must be generated independently of atmosphere and detector random streams.
    """
    cfg = deepcopy(config)
    if (cfg['sensor'] != 'linear' or cal['twin_modes'] != cfg['modes']
            or fault not in ('none', 'optical_gain')):
        raise ValueError('Identification requires the full linear modal sensor and a supported fault')
    frames, modes = cfg['frames'], cfg['modes']
    compiled = jax.jit(lambda phases, seed, probe, noise: loop(
        cfg, cal, phases, seed, 'integrator', jnp.zeros((modes, 2)),
        fault=fault, fallback=False, differentiable_noise=False,
        command_probe=probe, measurement_perturbation=noise))

    def execute(phases, seed, probe, noise):
        phase = np.asarray(phases)
        probe = _matrix(probe, 'Probe request')
        noise = _matrix(noise, 'Additive measurement noise')
        if (phase.shape != (frames, cfg['n'], cfg['n']) or phase.dtype.kind != 'f'
                or not np.isfinite(phase).all()
                or probe.shape != (frames, modes) or noise.shape != (frames, modes)
                or type(seed) is not int or not 0 <= seed < 2**32-100000):
            raise ValueError('Incompatible or nonfinite instrumented loop inputs')
        return jax.block_until_ready(compiled(jnp.asarray(phase), seed,
                                             jnp.asarray(probe), jnp.asarray(noise)))
    return execute


def identification_telemetry(outputs: dict, cal: dict, *, delay_frames: int,
                             burn_in: int = 0) -> dict[str, np.ndarray]:
    """Select observations, actual modal application and delayed probe requests.

    No phase truth, simulator estimate or prediction is used. Delayed probe
    requests are instruments, not an inferred physical contribution after
    clipping. The initial FIFO is zero, as declared by simulation.loop.
    """
    applied = _matrix(outputs['applied'], 'Applied command')
    issued = _matrix(outputs['command'], 'Issued command')
    probe = _matrix(outputs['probe_requested'], 'Requested probe')
    if (applied.shape != issued.shape or applied.shape != probe.shape
            or type(delay_frames) is not int or not 1 <= delay_frames < len(applied)
            or type(burn_in) is not int or not 0 <= burn_in < len(applied)-2):
        raise ValueError('Invalid telemetry shapes, delay or burn-in')
    if (np.any(applied[:delay_frames] != 0)
            or not np.array_equal(applied[delay_frames:], issued[:-delay_frames])):
        raise ValueError('Applied commands do not reproduce the declared FIFO')
    delayed_probe = np.zeros_like(probe)
    delayed_probe[delay_frames:] = probe[:-delay_frames]
    measurement = _matrix(outputs['measurement'], 'Measurements')
    recon = _matrix(cal['reconstructor'], 'Reconstructor')
    reference = np.asarray(cal['reference'])
    if (measurement.shape[0] != len(applied) or recon.shape != (applied.shape[1], measurement.shape[1])
            or reference.shape != (measurement.shape[1],) or reference.dtype.kind not in 'fiu'
            or not np.isfinite(reference).all()):
        raise ValueError('Identification needs full compatible modal observations')
    # Same calibrated residual as measured_telemetry, before adding the command:
    # adding and subtracting a large command here would lose small residuals.
    residual = (measurement-reference) @ recon.T
    if not np.isfinite(residual).all():
        raise ValueError('Nonfinite reconstructed residual observations')
    return {name: np.array(array[burn_in:], copy=True) for name, array in (
        ('residual', residual), ('applied', applied), ('instrument', delayed_probe))}


def _basis(array, rcond):
    u, singular, _ = np.linalg.svd(array, full_matrices=False)
    rank = int(np.sum(singular > rcond * singular[0])) if singular[0] > 0 else 0
    return u[:, :rank], singular, rank


def fit_response(residual, applied, instrument=None, *, method: str,
                 minimum_relevance: float = 0.05, rcond: float = 1e-10) -> dict:
    """Fit Y = disturbance - X B; rows are frames and columns are modal channels.

    B maps row applied commands to row observations. Both OLS and IV remove
    training means and retain an intercept. IV solves the projected least-squares
    system Qz.T @ X B = -Qz.T @ Y without forming an n-by-n projection.
    Minimum canonical correlation is a descriptive relevance gate, not an iid
    F statistic, confidence level, or proof of independent excitation.
    Rank-deficient or weak-instrument fits return a withheld result.
    """
    y, x = _matrix(residual, 'Residual observations'), _matrix(applied, 'Applied commands')
    if (y.shape != x.shape or len(y) <= x.shape[1]+1 or method not in ('ols', 'iv')
            or not np.isfinite(rcond) or not 0 < rcond < 1
            or not np.isfinite(minimum_relevance) or not 0 <= minimum_relevance < 1):
        raise ValueError('Invalid response fit dimensions, method or tolerances')
    x_mean, y_mean = x.mean(axis=0), y.mean(axis=0)
    xc, yc = x-x_mean, y-y_mean
    if not np.isfinite(xc).all() or not np.isfinite(yc).all():
        raise ValueError('Centered telemetry exceeds the numerical range')
    qx, sx, rank_x = _basis(xc, rcond)
    result = {'method': method, 'status': 'withheld', 'reason': None,
              'response': None, 'intercept': None,
              'diagnostics': {'frames': len(y), 'modes': x.shape[1],
                              'command_rank': rank_x,
                              'command_singular_values': sx,
                              'minimum_relevance': minimum_relevance, 'rcond': rcond}}
    if rank_x < x.shape[1]:
        result['reason'] = 'rank_deficient_commands'
        return result
    design, target = xc, -yc
    if method == 'iv':
        z = _matrix(instrument, 'Probe instruments')
        if z.shape[0] != len(y) or z.shape[1] < x.shape[1] or len(y) <= z.shape[1]+1:
            raise ValueError('Instrument dimensions cannot identify the response')
        zc = z-z.mean(axis=0)
        if not np.isfinite(zc).all():
            raise ValueError('Centered instruments exceed the numerical range')
        qz, sz, rank_z = _basis(zc, rcond)
        result['diagnostics'].update(instrument_rank=rank_z, instrument_singular_values=sz)
        if rank_z < x.shape[1]:
            result['reason'] = 'rank_deficient_instruments'
            return result
        canonical = np.linalg.svd(qz.T @ qx, compute_uv=False)
        result['diagnostics']['canonical_correlations'] = canonical
        if float(canonical[-1]) < minimum_relevance:
            result['reason'] = 'weak_instruments'
            return result
        design, target = qz.T @ xc, -qz.T @ yc
    response, _, rank, singular = np.linalg.lstsq(design, target, rcond=rcond)
    result['diagnostics'].update(projected_command_rank=int(rank), fit_singular_values=singular)
    if rank < x.shape[1]:
        result['reason'] = 'rank_deficient_projected_commands'
        return result
    intercept = y_mean + x_mean @ response
    if not np.isfinite(response).all() or not np.isfinite(intercept).all():
        result['reason'] = 'nonfinite_fit'
        return result
    result.update(status='ready', response=response, intercept=intercept)
    return result


def fit_prediction_pipeline(episodes: list[dict], *, method: str, ridge: float,
                            minimum_relevance: float = 0.05, rcond: float = 1e-10) -> dict:
    """Fit equal-weight, per-episode response and two-tap prediction stages.

    The nominal baseline uses response I, reproducing measured pseudo-open-loop
    reconstruction. OLS and IV change only that response; every method uses the
    existing predict_fit(horizon=1) on each independent training episode. No
    episodes are concatenated or silently dropped. Cohort/seed access belongs to
    the experiment runner, not this numerical adapter.
    """
    if (not isinstance(episodes, list) or not episodes or method not in ('nominal', 'ols', 'iv')
            or not np.isfinite(ridge) or ridge <= 0 or not np.isfinite(rcond) or not 0 < rcond < 1
            or not np.isfinite(minimum_relevance) or not 0 <= minimum_relevance < 1):
        raise ValueError('A prediction pipeline needs training episodes, a method and positive ridge')
    rows, fits, modes = [], [], None
    for episode in episodes:
        if not isinstance(episode, dict) or set(episode) != {'residual', 'applied', 'instrument'}:
            raise ValueError('Training episodes accept only residual, applied and instrument arrays')
        row = {name: _matrix(value, name) for name, value in episode.items()}
        if (len({value.shape for value in row.values()}) != 1 or len(row['residual']) < 4
                or (modes is not None and row['residual'].shape[1] != modes)):
            raise ValueError('Training episodes need compatible complete modal telemetry')
        modes = row['residual'].shape[1]
        rows.append(row)
        fits.append({'status': 'ready', 'response': np.eye(modes)} if method == 'nominal'
                    else fit_response(row['residual'], row['applied'], row['instrument'],
                                      method=method, minimum_relevance=minimum_relevance, rcond=rcond))
    result = {'method': method, 'status': 'withheld', 'reason': None, 'response': None,
              'coefficients': None, 'episode_fits': fits, 'episode_coefficients': None,
              'horizon': 1, 'ridge': ridge, 'rcond': rcond, 'minimum_relevance': minimum_relevance}
    if any(fit['status'] != 'ready' for fit in fits):
        result['reason'] = 'withheld_training_response'
        return result
    # Divide before summing to avoid an overflowing intermediate sum of finite fits.
    with np.errstate(over='ignore', invalid='ignore'):
        response = np.sum(np.stack([fit['response'] for fit in fits])/len(fits), axis=0)
    if not np.isfinite(response).all():
        result['reason'] = 'nonfinite_mean_response'
        return result
    _, singular, rank = _basis(response, rcond)
    result['response_singular_values'] = singular
    individual_scale = max(float(np.linalg.svd(fit['response'], compute_uv=False)[0]) for fit in fits)
    result['individual_response_scale'] = individual_scale
    if rank != modes or singular[-1] <= rcond*individual_scale:
        result['reason'] = 'singular_or_canceling_response'
        return result
    coefficients = []
    for row in rows:
        pseudo = np.linalg.solve(response.T, row['residual'].T).T + row['applied']
        if not np.isfinite(pseudo).all():
            result['reason'] = 'nonfinite_reconstruction'
            return result
        coefficient = predict_fit(pseudo, horizon=1, ridge=ridge)
        if not np.isfinite(coefficient).all():
            result['reason'] = 'nonfinite_predictor'
            return result
        coefficients.append(coefficient)
    with np.errstate(over='ignore', invalid='ignore'):
        mean_coefficients = np.sum(np.stack(coefficients)/len(coefficients), axis=0)
    if not np.isfinite(mean_coefficients).all():
        result['reason'] = 'nonfinite_mean_predictor'
        return result
    result.update(status='ready', response=response, coefficients=mean_coefficients,
                  episode_coefficients=np.stack(coefficients))
    return result


def one_step_prediction(model: dict, residual, applied, issued, *, delay_frames: int) -> dict:
    """Forecast after command issue using only the then-known FIFO and past data.

    Forecast of frame k uses residual[k-1], residual[k-2] and issued[k-delay].
    delay>=1 makes that command available at forecast time. Early targets whose
    applied command belongs to the unspecified initial FIFO are omitted.
    Recorded future applied commands and future residuals are never predictors.
    """
    if model['status'] != 'ready' or model['horizon'] != 1:
        raise ValueError('Require a ready one-step prediction model')
    y, x, command = (_matrix(value, name) for value, name in (
        (residual, 'Residual'), (applied, 'Applied command'), (issued, 'Issued command')))
    response = _matrix(model['response'], 'Response')
    coefficients = _matrix(model['coefficients'], 'Predictor')
    if (x.shape != y.shape or command.shape != y.shape
            or response.shape != (y.shape[1], y.shape[1]) or coefficients.shape != (y.shape[1], 2)
            or type(delay_frames) is not int or not 1 <= delay_frames < len(y)-1):
        raise ValueError('Invalid one-step forecast shapes or delay')
    target = np.arange(max(2, delay_frames), len(y))
    # Do not reconstruct future rows, even though the batched API accepts a trajectory.
    now = np.linalg.solve(response.T, y[target-1].T).T + x[target-1]
    before = np.linalg.solve(response.T, y[target-2].T).T + x[target-2]
    disturbance = coefficients[:, 0]*now + coefficients[:, 1]*before
    predicted = (disturbance - command[target-delay_frames]) @ response
    if not np.isfinite(predicted).all():
        raise ValueError('Nonfinite sensor forecast')
    return {'target_indices': target, 'prediction': predicted}
