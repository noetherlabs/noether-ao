"""Independent HCIPy matrix-quadrature reference for the stated ideal pyramid.

Physical pupil -> matrix Fourier integral -> thin surface -> inverse integral
    -> physical pixel powers -> HCIPy detector bin sums -> spectral average.

This module deliberately imports neither dLux/JAX nor the production sensor.
Conventions and the distinction from HCIPy's native defaults are documented in
docs/pyramid_validation.md. Output power is never renormalized.
"""
from collections.abc import Callable, Sequence

import hcipy as hp
import numpy as np


def pyramid_reference(n: int, wavelengths: Sequence[float], reference_wavelength: float,
                      detector_n: int, separation: float = 0.6) -> Callable:
    """Construct an even-grid, unit-diameter, nondispersive pyramid reference."""
    if not isinstance(n, int) or n < 2 or n % 2:
        raise ValueError("Pupil grid must be a positive even integer")
    if not isinstance(detector_n, int) or detector_n < 1 or (4 * n) % detector_n:
        raise ValueError("Detector size must divide the four-diameter pupil grid")
    wavelengths = np.asarray(wavelengths, dtype=float)
    if wavelengths.ndim != 1 or not len(wavelengths) or not np.all(
        np.isfinite(wavelengths) & (wavelengths > 0)
    ):
        raise ValueError("Wavelengths must be a nonempty positive finite vector")
    if not np.isfinite(reference_wavelength) or reference_wavelength <= 0:
        raise ValueError("Reference wavelength must be positive and finite")
    if not np.isfinite(separation) or separation <= 0:
        raise ValueError("Pupil separation must be positive and finite")

    pupil_grid = hp.make_pupil_grid(n, 1.0)
    output_grid = hp.make_pupil_grid(4 * n, 4.0)
    aperture = hp.make_circular_aperture(1.0)(pupil_grid)
    amplitude = aperture / np.sqrt(np.sum(aperture**2 * pupil_grid.weights))
    optics = []
    for wavelength in wavelengths:
        # Zero is sampled on this even focal grid: no empirical alignment fit.
        focal_grid = hp.make_uniform_grid([4 * n, 4 * n], n * wavelength,
                                          has_center=True)
        angular_frequency_grid = focal_grid.scaled(2 * np.pi / wavelength)
        outward = hp.MatrixFourierTransform(pupil_grid, angular_frequency_grid)
        inward = hp.MatrixFourierTransform(output_grid, angular_frequency_grid)
        refractive_index = 1.5
        sag = -separation * (np.abs(focal_grid.x) + np.abs(focal_grid.y)) / (
            refractive_index - 1
        )
        surface = hp.SurfaceApodizer(hp.Field(sag, focal_grid), refractive_index)
        optics.append((wavelength, focal_grid, angular_frequency_grid,
                       outward, inward, surface))

    def sensor(phase):
        phase = np.asarray(phase)
        if phase.shape != (n, n) or not np.isrealobj(phase) or not np.isfinite(phase).all():
            raise ValueError(f"Phase must be a finite real array with shape {(n, n)}")
        powers = []
        for wavelength, focal_grid, frequency_grid, outward, inward, surface in optics:
            field = hp.Field(amplitude * np.exp(
                1j * phase.ravel() * reference_wavelength / wavelength
            ), pupil_grid)
            # Physical Fraunhofer normalization at focal length one metre.
            focal = hp.Wavefront(hp.Field(
                outward.forward(field) / (1j * wavelength), focal_grid
            ), wavelength)
            masked = surface.forward(focal)
            returned = inward.backward(hp.Field(masked.electric_field, frequency_grid))
            detector = hp.Wavefront(returned * (1j * wavelength), wavelength)
            powers.append(np.asarray(hp.subsample_field(
                detector.power, (4 * n) // detector_n, statistic="sum"
            )))
        return np.mean(powers, axis=0)

    return sensor


def analytic_modes(n: int, names: Sequence[str]) -> dict[str, np.ndarray]:
    """Independent, piston-free unit-RMS pupil inputs; no production basis code."""
    grid = hp.make_pupil_grid(n, 1.0)
    u, v = 2 * grid.x, 2 * grid.y
    rho2 = u**2 + v**2
    mask = np.asarray(hp.make_circular_aperture(1.0)(grid), dtype=bool)
    polynomials = {"tilt_x": u, "tilt_y": v, "defocus": 2 * rho2 - 1,
                   "astigmatism_oblique": 2 * u * v, "coma_x": (3 * rho2 - 2) * u}
    modes = {}
    for name in names:
        if name not in polynomials:
            raise ValueError(f"Unknown analytic mode: {name}")
        values = np.array(polynomials[name], copy=True)
        values -= values[mask].mean()
        values /= np.sqrt(np.mean(values[mask]**2))
        modes[name] = np.where(mask, values, 0).reshape(n, n)
    return modes
