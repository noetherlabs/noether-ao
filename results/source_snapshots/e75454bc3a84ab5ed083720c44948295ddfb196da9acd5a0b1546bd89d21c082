"""Seeded finite von Karman screens and differentiable frozen-flow sampling."""
from __future__ import annotations

import jax
import jax.numpy as jnp


def phase_sequence(config: dict, seed: int, frames: int | None = None) -> jax.Array:
    """Sample two or more periodic phase layers, in radians at reference wavelength.

    The PSD is in cycles/metre: 0.0229 r0^(-5/3) (f² + L0^-2)^(-11/6).
    Filtering real unit white noise preserves Hermitian symmetry. A screen spans
    screen_factor pupil diameters; low-frequency subharmonics are not included.
    Translation is a Fourier phase ramp, with no per-frame screen resampling.
    """
    n = int(config['n']); factor = int(config['screen_factor']); big = n * factor
    frames = int(frames or config['frames'])
    diameter = float(config['diameter_m']); dx = diameter / n
    freq = jnp.fft.fftfreq(big, d=dx)
    fx, fy = jnp.meshgrid(freq, freq)
    f2 = fx**2 + fy**2
    keys = jax.random.split(jax.random.PRNGKey(seed), len(config['layers']))
    if len(config['layers']) < 2:
        raise ValueError('At least two turbulence layers are required')
    if abs(sum(layer['weight'] for layer in config['layers']) - 1) > 1e-8:
        raise ValueError('Layer weights must sum to one')
    spectra, rates = [], []
    for key, layer in zip(keys, config['layers']):
        psd = .0229 * config['r0_m']**(-5/3) * (f2 + layer['outer_scale_m']**(-2))**(-11/6)
        psd = psd.at[0, 0].set(0)
        white = jax.random.normal(key, (big, big), dtype=jnp.float64)
        spectra.append(jnp.fft.fft2(white) * jnp.sqrt(psd * layer['weight']) / dx)
        vx, vy = layer['velocity_m_s']
        rates.append(-2j * jnp.pi * (fx * vx + fy * vy) * config.get('wind_scale', 1.0))
    spectra, rates = jnp.stack(spectra), jnp.stack(rates)
    start = (big - n) // 2

    def step(_, t):
        layers = jnp.fft.ifft2(spectra * jnp.exp(rates * t), axes=(-2, -1)).real
        crop = layers[:, start:start+n, start:start+n].sum(axis=0)
        return None, crop

    return jax.lax.scan(step, None, jnp.arange(frames) * config['dt_s'])[1]
