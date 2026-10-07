"""Explicit scalar Fourier optics with unit discrete input power."""
from __future__ import annotations

from collections.abc import Callable
import dLux as dl
import jax
import jax.numpy as jnp
import numpy as np


def aperture(n: int, dtype: str = "float64") -> jax.Array:
    """Return an interpixel-centred hard circular pupil of diameter one."""
    x = (jnp.arange(n, dtype=dtype) - (n - 1) / 2) / n
    return ((x[:, None] ** 2 + x[None, :] ** 2) <= 0.25).astype(dtype)


def propagation(n: int, dtype: str, engine: str) -> Callable:
    """Build a batched dynamic-OPD pupil-to-focal complex-field function.

    The documented dLux Wavefront.propagate_FFT path is compiled as a whole.
    Outputs are separate wavelength-scaled focal grids, not one summed PSF.
    """
    amp = aperture(n, dtype)
    amp = amp / jnp.sqrt(jnp.sum(amp ** 2))
    if engine == "dlux":
        wf = dl.Wavefront(wavelength=1.0, npixels=n, diameter=1.0).set(
            phasor=amp.astype("complex64" if dtype == "float32" else "complex128"),
            pixel_scale=jnp.asarray(1 / n, dtype=dtype))

        def mono(opd: jax.Array, wavelength: jax.Array) -> jax.Array:
            return wf.set(wavelength=wavelength).add_opd(opd).propagate_FFT(pad=1).phasor
    elif engine == "jax":
        def mono(opd: jax.Array, wavelength: jax.Array) -> jax.Array:
            field = amp * jnp.exp((2j * jnp.pi / wavelength) * opd)
            return jnp.fft.fftshift(jnp.fft.fft2(jnp.fft.ifftshift(field))) / n
    else:
        raise ValueError(f"Unknown propagation engine {engine}")
    return jax.vmap(mono, in_axes=(None, 0))


def relative_l2(actual: np.ndarray, expected: np.ndarray) -> float:
    """Compare complex fields without fitting scale, phase, or conjugation."""
    return float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))
