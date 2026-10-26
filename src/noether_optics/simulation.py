"""Small modal AO plant, causal controllers and empirical fallback logic.

    atmosphere[t] - FIFO[0] -> noisy sensor -> estimate -> command -> FIFO[-1]
             past estimates -> predicted sensor -> detector -> latched fallback

Truth phase is used only by the plant and evaluation loss. Controllers receive
measurements, their own command history and calibrated model parameters.
"""
from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from math import factorial

import dLux as dl
import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import t as student_t

from .optics import aperture

CONTROLLERS = ('integrator', 'predictive', 'forward', 'fixed', 'learned')
FAULTS = ('none', 'stuck_actuator', 'optical_gain', 'misregistration',
          'vibration', 'model_mismatch', 'late_twin', 'missing_twin')


def _zernikes(n: int, modes: int) -> tuple[np.ndarray, np.ndarray]:
    x = (np.arange(n) - (n - 1)/2) / n
    xx, yy = np.meshgrid(2*x, 2*x)
    rho, phi = np.hypot(xx, yy), np.arctan2(yy, xx)
    mask = rho <= 1
    arrays = []
    radial = 1
    while len(arrays) < modes:
        for az in range(-radial, radial+1, 2):
            m = abs(az)
            poly = sum((-1)**k * factorial(radial-k) / (factorial(k) *
                       factorial((radial+m)//2-k) * factorial((radial-m)//2-k)) *
                       rho**(radial-2*k) for k in range((radial-m)//2+1))
            angular = np.sin(m*phi) if az < 0 else np.cos(m*phi)
            arrays.append(poly * angular * np.sqrt((radial+1)*(1 if az == 0 else 2)))
            if len(arrays) == modes:
                break
        radial += 1
    return np.stack(arrays), mask


@lru_cache(maxsize=24)
def modal_basis(n: int, modes: int, reference_n: int | None = None) -> np.ndarray:
    """Sample a piston-free Zernike basis orthonormal on one reference pupil.

    A single reference-grid QR defines coefficient meaning on every fidelity grid.
    Coefficients are radians RMS on the reference pupil, not mirror displacement.
    """
    ref, ref_mask = _zernikes(reference_n or n, modes)
    means = ref[:, ref_mask].mean(axis=1)
    matrix = (ref[:, ref_mask] - means[:, None]).T
    _, r = np.linalg.qr(matrix / np.sqrt(ref_mask.sum()), mode='reduced')
    raw, mask = _zernikes(n, modes)
    b = np.linalg.solve(r.T, (raw - means[:, None, None]).reshape(modes, -1))
    return (b.reshape(modes, n, n) * mask).astype(np.float64)


def pyramid_sensor(n: int, wavelengths: tuple[float, ...], reference_wavelength: float,
                   detector_n: int, separation: float = .6,
                   pupil_shift_pixels: int = 0) -> Callable:
    """Return an unmodulated ideal four-face pyramid using dLux FFT propagations.

    Pupil is padded to four diameters. Each face shifts a pupil by separation*D.
    A nondispersive refractive wedge gives wavelength-independent pupil separation.
    Detector bins conserve photon flux; all fidelities share the detector extent.
    """
    if (4*n) % detector_n:
        raise ValueError('Detector must divide the padded pyramid grid')
    pad = 4
    amp = aperture(n)
    amp = amp / jnp.sqrt(jnp.sum(amp**2))
    base = dl.Wavefront(wavelength=reference_wavelength, npixels=n, diameter=1.0).set(phasor=amp.astype(jnp.complex128))
    k = jnp.fft.fftshift(jnp.fft.fftfreq(pad*n))
    ramp = -2*jnp.pi * separation*n * (jnp.abs(k[:, None]) + jnp.abs(k[None, :]))
    wavelength_array = jnp.asarray(wavelengths)
    binning = pad*n // detector_n

    def mono(phase, wl):
        wf = base.set(wavelength=wl).add_phase(phase * reference_wavelength/wl)
        if pupil_shift_pixels:
            wf = wf.resize(pad*n)
            wf = wf.set(phasor=jnp.roll(wf.phasor,pupil_shift_pixels,axis=1))
            focal = wf.propagate_FFT(pad=1)
        else:
            focal = wf.propagate_FFT(pad=pad)
        detector = focal.add_phase(ramp).propagate_FFT(pad=1, inverse=True)
        intensity = detector.psf
        return intensity.reshape(detector_n, binning, detector_n, binning).sum(axis=(1,3))

    def sensor(phase):
        return jax.vmap(mono, in_axes=(None,0))(phase, wavelength_array).mean(axis=0).ravel()
    return sensor


def calibration(config: dict, twin: dict | None = None) -> dict:
    """Build immutable sensor models, an interaction matrix and reconstructor."""
    n, modes = config['n'], config['modes']
    twin = twin or {'n': n, 'modes': modes, 'wavelengths_m': config['wavelengths_m']}
    nb, mb = twin['n'], twin['modes']
    if mb > modes:
        raise ValueError('Twin cannot estimate more modes than the DM')
    basis = jnp.asarray(modal_basis(n, modes))
    twin_basis = jnp.asarray(modal_basis(nb, modes, n)[:mb])
    mask = aperture(n)
    projection = basis.reshape(modes, -1) / mask.sum()
    if config['sensor'] == 'linear':
        plant = lambda phase: projection @ phase.ravel()
        model = lambda c: c
        misregistered_plant = lambda phase: projection @ jnp.roll(phase,config['faults']['shift_pixels'],axis=1).ravel()
        reference = jnp.zeros(mb)
        interaction = jnp.eye(mb)
    elif config['sensor'] == 'pyramid':
        plant = pyramid_sensor(n, tuple(config['wavelengths_m']), config['wavelength_m'], config['detector_n'])
        misregistered_plant = pyramid_sensor(n,tuple(config['wavelengths_m']),config['wavelength_m'],config['detector_n'],pupil_shift_pixels=config['faults']['shift_pixels'])
        sensor = pyramid_sensor(nb, tuple(twin['wavelengths_m']), config['wavelength_m'], config['detector_n'])
        model = lambda c: sensor(jnp.tensordot(c, twin_basis, axes=1))
        reference = model(jnp.zeros(mb))
        interaction = jax.jit(lambda: jax.lax.map(
            lambda direction: jax.jvp(model, (jnp.zeros(mb),), (direction,))[1],
            jnp.eye(mb), batch_size=4).T)()
    else:
        raise ValueError('Sensor must be linear or pyramid')
    # Fixed calibration, independent of flux and evaluation measurements.
    ridge = config['reconstructor_ridge']
    reconstructor = jnp.linalg.solve(interaction.T @ interaction + ridge*jnp.eye(mb), interaction.T)
    x = (jnp.arange(n)-(n-1)/2)/n
    xx, yy = jnp.meshgrid(x, x)
    stuck_patch = jnp.exp(-((xx-.2)**2+(yy-.1)**2)/(2*config['faults']['stuck_radius_d']**2))
    return {'basis': basis, 'mask': mask, 'projection': projection,
            'plant': plant, 'misregistered_plant':misregistered_plant, 'model': model, 'reference': reference,
            'interaction': interaction, 'reconstructor': reconstructor,
            'stuck_patch': stuck_patch, 'twin_modes': mb,
            'preconditioner': jnp.linalg.inv(interaction.T@interaction +
                config['forward_ridge']*jnp.eye(mb))}


def predict_fit(telemetry: np.ndarray, horizon: int, ridge: float) -> np.ndarray:
    """Fit a causal per-mode two-tap predictor using training-only telemetry."""
    a = np.asarray(telemetry)
    x = np.stack((a[1:-horizon], a[:-horizon-1]), axis=-1)
    y = a[horizon+1:]
    gram = np.einsum('tmi,tmj->mij', x, x)
    rhs = np.einsum('tmi,tm->mi', x, y)
    return np.linalg.solve(gram + ridge*np.eye(2), rhs[...,None])[...,0]


def weights(params: tuple[jax.Array, ...], residual: jax.Array, forward: jax.Array,
            variance: jax.Array) -> jax.Array:
    """Shared small network produces a forward-command weight for each mode."""
    w1, b1, w2, b2 = params
    features = jnp.stack((jnp.log1p(residual**2/(variance+1e-8)),
                          jnp.log1p(forward**2), jnp.broadcast_to(jnp.log1p(variance),residual.shape)), axis=-1)
    return jax.nn.sigmoid(jnp.tanh(features @ w1 + b1) @ w2 + b2).squeeze(-1)


def initialize_weights(seed: int, width: int = 8) -> tuple:
    """Seed a small per-mode weighting network without any test telemetry."""
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    return (jax.random.normal(k1,(3,width))*.1, jnp.zeros(width),
            jax.random.normal(k2,(width,1))*.1, jnp.zeros(1))


def loop(config: dict, cal: dict, phases: jax.Array, seed: int, controller: str,
         predictor: jax.Array, threshold: float = 1e30, fallback: bool = False,
         fault: str = 'none', params: tuple | None = None,
         differentiable_noise: bool = False, *, command_probe: jax.Array | None = None,
         measurement_perturbation: jax.Array | None = None) -> dict[str,jax.Array]:
    """Run a causal delayed AO loop via lax.scan.

    Evaluation uses exact Poisson shot noise plus Gaussian read noise. Training's
    reparameterized Gaussian approximation is explicitly selected. Discrete fallback
    and Poisson sampling are not differentiable; the optical/control path is.

    Optional finite, externally generated probes enter after the controller and
    before clipping. The next controller state follows the issued command,
    including the probe. Additive measurement perturbations enter after detector
    noise, in the sensor's output units. Neither input replaces detector noise.
    Opting in adds command-stage and perturbation telemetry; the default path and
    output schema remain unchanged. Callers must establish independent streams.
    """
    if controller not in CONTROLLERS or fault not in FAULTS:
        raise ValueError('Unknown controller or fault')
    delay = config['delay_frames']
    if delay < 1:
        raise ValueError('Delay must be at least one frame')
    m, mb = config['modes'], cal['twin_modes']
    instrumented = command_probe is not None or measurement_perturbation is not None
    if instrumented:
        if fault == 'stuck_actuator':
            raise ValueError('Instrumented modal commands do not describe a stuck-actuator surface')
        sensor_size = m if config['sensor'] == 'linear' else cal['reference'].size
        command_probe = jnp.zeros((len(phases), m)) if command_probe is None else jnp.asarray(command_probe)
        measurement_perturbation = (jnp.zeros((len(phases), sensor_size))
                                   if measurement_perturbation is None else jnp.asarray(measurement_perturbation))
        if (command_probe.shape != (len(phases), m)
                or measurement_perturbation.shape != (len(phases), sensor_size)
                or command_probe.dtype.kind not in 'fiu'
                or measurement_perturbation.dtype.kind not in 'fiu'):
            raise ValueError('Probe and measurement perturbation shapes/dtypes must match the loop')
    flux, read = config['photons'], config['read_noise_e']
    gain, leak = config['gain'], config['leak']
    basis, mask = cal['basis'], cal['mask']
    plant, model, recon = cal['plant'], cal['model'], cal['reconstructor']
    ref = cal['reference']
    zero = jnp.zeros(m)
    keys = jax.random.split(jax.random.PRNGKey(seed+100000), len(phases))
    fcfg = config['faults']
    step_gain = config['forward_step']
    modal_noise = (1/flux + read**2/flux**2) if config['sensor']=='linear' else jnp.sum(recon**2 * (ref[None,:]/flux + read**2/flux**2),axis=1)

    def frame(carry, inputs):
        fifo, command, previous, predicted, latched = carry
        if instrumented:
            t, phase, key, probe, perturbation = inputs
        else:
            t, phase, key = inputs
        active = (t >= fcfg['onset']) & (t < fcfg['end'])
        applied = fifo[0]
        dm = jnp.tensordot(applied, basis, axes=1)
        if fault == 'stuck_actuator':
            dm = dm * (1-jnp.where(active, cal['stuck_patch'], 0))
        if fault == 'vibration':
            phase = phase + jnp.where(active, fcfg['vibration_rad']*jnp.sin(2*jnp.pi*fcfg['vibration_hz']*t*config['dt_s']),0)*basis[0]
        residual = (phase-dm)*mask
        residual = residual - mask*jnp.sum(residual)/mask.sum()
        if fault == 'misregistration':
            # Shift the complete complex pupil inside its padded support.
            clean = jax.lax.cond(active,cal['misregistered_plant'],plant,residual)
        else:
            clean = plant(residual)
        if fault == 'optical_gain':
            flat = plant(jnp.zeros_like(residual))
            progress = jnp.clip((t-fcfg['onset'])/fcfg['drift_frames'],0,1)
            drift_gain = 1 + progress*(fcfg['optical_gain']-1)
            clean = flat + jnp.where(active, drift_gain,1)*(clean-flat)
        shot_key, read_key = jax.random.split(key)
        if config['sensor'] == 'linear':
            measured = clean + jnp.sqrt(1/flux + read**2/flux**2)*jax.random.normal(shot_key,clean.shape)
        else:
            counts = jnp.maximum(clean*flux, 0)
            if differentiable_noise:
                observed = counts + jnp.sqrt(counts+read**2)*jax.random.normal(shot_key, counts.shape)
            else:
                observed = jax.random.poisson(shot_key, counts).astype(jnp.float64) + read*jax.random.normal(read_key, counts.shape)
            measured = observed/flux
        if instrumented:
            measured = measured + perturbation
        model_scale = jnp.where(active, fcfg['model_phase_scale'],1) if fault == 'model_mismatch' else 1.0
        expected = model((predicted[:mb]-applied[:mb])*model_scale)
        data = measured[:mb] if config['sensor']=='linear' else measured
        noise_var = jnp.ones_like(expected)*(1/flux+read**2/flux**2) if config['sensor']=='linear' else jnp.maximum(expected,0)/flux + read**2/flux**2 + 1e-16
        score = jnp.mean((data-expected)**2/noise_var)
        missing = active if fault=='missing_twin' else jnp.asarray(False)
        age = jnp.where(active,2*config['dt_s'],0) if fault=='late_twin' else 0.
        unavailable = missing | (age > config['dt_s'])
        alarm = ((score > threshold) | unavailable) & (t >= config['burn_in'])
        latched = latched | (alarm & fallback)
        estimate_linear = recon @ (data-ref)
        # Fixed-preconditioner gradient descent on nonlinear twin image mismatch.
        # J^T J at calibration supplies a stable scale; gradient is from the full twin.
        prior = previous[:mb]-applied[:mb]
        def objective(coeff):
            diff = model(coeff*model_scale)-data
            return .5*jnp.sum(diff**2)
        preconditioner = cal['preconditioner']
        def improve(_, estimate):
            gradient = jax.grad(objective)(estimate)
            return jnp.clip(estimate - step_gain*(preconditioner@gradient), -config['estimate_clip_rad'], config['estimate_clip_rad'])
        if controller in ('forward','fixed','learned'):
            fit = jax.lax.fori_loop(0, config['forward_steps'], improve, prior)
        else:
            fit = estimate_linear
        # Classical state follows actual issued command, so switching is bumpless.
        ci = leak*command + gain*jnp.pad(estimate_linear,(0,m-mb))
        pseudo = applied + jnp.pad(fit,(0,m-mb))
        cf = leak*command + gain*jnp.pad(fit,(0,m-mb))
        cp = predictor[:,0]*pseudo + predictor[:,1]*previous
        if controller == 'integrator': candidate = ci
        elif controller == 'predictive': candidate = cp
        elif controller == 'forward': candidate = cf
        elif controller == 'fixed':
            snr = fit**2/(modal_noise+1e-8)
            w = snr/(1+snr)/(1+fit**2)
            candidate = ci.at[:mb].set((1-w)*ci[:mb]+w*cf[:mb])
        else:
            w = weights(params,estimate_linear,fit,modal_noise)
            candidate = ci.at[:mb].set((1-w)*ci[:mb]+w*cf[:mb])
        if controller in ('forward','fixed','learned'):
            candidate = jnp.where(unavailable,command,candidate)
        candidate = jnp.where(latched, ci, candidate)
        if instrumented:
            feedback_requested = candidate
            candidate = candidate + probe
        clipped = jnp.clip(candidate,-config['command_clip_rad'],config['command_clip_rad'])
        next_fifo = jnp.concatenate((fifo[1:],clipped[None,:]),axis=0)
        # One-frame model for the next measurement. Never sees future phase.
        next_prediction = 2*pseudo-previous
        variance = jnp.sum(residual**2)/mask.sum()
        strehl = jnp.abs(jnp.sum(mask*jnp.exp(1j*residual))/mask.sum())**2
        output = {'variance':variance,'strehl':strehl,'score':score,'alarm':alarm,
                  'fallback':latched,'saturated':jnp.any(candidate!=clipped),
                  'command':clipped,'applied':applied,'estimate':pseudo,'unavailable':unavailable,
                  'measurement':data,'prediction':expected}
        if instrumented:
            output.update(feedback_requested=feedback_requested, probe_requested=probe,
                          combined_requested=candidate, measurement_perturbation=perturbation)
        return (next_fifo,clipped,pseudo,next_prediction,latched),output

    initial = (jnp.zeros((delay,m)),zero,zero,zero,jnp.asarray(False))
    inputs = (jnp.arange(len(phases)),phases,keys)
    if instrumented:
        inputs = (*inputs, command_probe, measurement_perturbation)
    return jax.lax.scan(frame,initial,inputs)[1]


def summarize(outputs: dict, config: dict) -> dict[str,float]:
    """Compute post-burn-in metrics; WFE is sqrt(mean spatial variance)."""
    start = config['burn_in']
    variance = np.asarray(outputs['variance'])[start:]
    return {'residual_variance_rad2':float(variance.mean()),
            'all_outputs_finite':bool(all(np.isfinite(np.asarray(v)).all() for v in outputs.values()) and outputs.get('all_finite_full_outputs',True)),
            'wfe_nm':float(np.sqrt(variance.mean())*config['wavelength_m']/(2*np.pi)*1e9),
            'strehl':float(np.asarray(outputs['strehl'])[start:].mean()),
            'saturation_fraction':float(np.asarray(outputs['saturated'])[start:].mean())}


def paired_interval(a: list[float], b: list[float]) -> tuple[float,float,float]:
    """Return paired mean difference and Student-t interval across seeds."""
    aa,bb=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    if aa.ndim!=1 or aa.shape!=bb.shape or len(aa)<2 or not np.isfinite(aa).all() or not np.isfinite(bb).all():
        raise ValueError('Paired intervals need matching finite vectors with at least two seeds')
    delta = aa-bb
    mean = float(delta.mean())
    half = float(student_t.ppf(.975,len(delta)-1)*delta.std(ddof=1)/np.sqrt(len(delta)))
    return mean, mean-half, mean+half
