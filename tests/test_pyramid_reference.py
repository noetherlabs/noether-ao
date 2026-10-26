"""Cross-library pixel agreement, independent derivatives and error sensitivity."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from noether_optics.optics import relative_l2
from noether_optics.pyramid_reference import analytic_modes, pyramid_reference
from noether_optics.simulation import pyramid_sensor


@pytest.mark.parametrize("n,wavelengths,detector_n", [
    (16, (750e-9,), 64), (32, (650e-9, 750e-9, 850e-9), 64)
])
def test_pixels_absolute_flux_and_independent_derivative(n, wavelengths, detector_n):
    reference = pyramid_reference(n, wavelengths, 750e-9, detector_n)
    production = jax.jit(pyramid_sensor(n, wavelengths, 750e-9, detector_n))
    modes = analytic_modes(n, ("tilt_x", "defocus", "coma_x"))
    phase = .05 * (modes["tilt_x"] + .3 * modes["coma_x"])
    for value in (np.zeros_like(phase), phase):
        actual, expected = np.asarray(production(value)), reference(value)
        assert relative_l2(actual, expected) <= 1e-10
        assert abs(actual.sum() - 1) <= 1e-12
        assert abs(expected.sum() - 1) <= 1e-12
    direction = modes["defocus"]
    step = 1e-5
    finite = (reference(phase + step * direction) - reference(phase - step * direction)) / (2 * step)
    derivative = jax.jvp(production, (jnp.asarray(phase),), (jnp.asarray(direction),))[1]
    assert relative_l2(derivative, finite) <= 1e-6


def test_comparison_rejects_sign_geometry_orientation_and_scale_errors():
    n, detector_n = 16, 32
    modes = analytic_modes(n, ("tilt_x", "tilt_y", "defocus", "astigmatism_oblique", "coma_x"))
    phase = .05 * sum(weight * mode for weight, mode in zip((.5, .2, .3, .4, .1), modes.values()))
    reference = pyramid_reference(n, (750e-9,), 750e-9, detector_n)(phase)
    production = pyramid_sensor(n, (750e-9,), 750e-9, detector_n)
    value = np.asarray(production(phase))
    wrong = (production(-phase),
             pyramid_sensor(n, (750e-9,), 750e-9, detector_n, separation=.66)(phase),
             value.reshape(detector_n, detector_n).T.ravel(), value * 1.1)
    assert all(relative_l2(candidate, reference) > 1e-4 for candidate in wrong)


@pytest.mark.parametrize("kwargs", [
    {"n": 15}, {"detector_n": 13}, {"wavelengths": ()},
    {"wavelengths": (0.,)}, {"wavelengths": (np.nan,)},
    {"reference_wavelength": -1.}, {"separation": np.inf}
])
def test_reference_rejects_invalid_optical_contract(kwargs):
    with pytest.raises(ValueError):
        pyramid_reference(**({"n": 16, "wavelengths": (750e-9,),
                             "reference_wavelength": 750e-9, "detector_n": 32} | kwargs))


@pytest.mark.parametrize("phase", [np.zeros((15, 16)), np.full((16, 16), np.nan),
                                    np.ones((16, 16), dtype=complex)])
def test_reference_rejects_invalid_phase(phase):
    reference = pyramid_reference(16, (750e-9,), 750e-9, 32)
    with pytest.raises(ValueError, match="finite real array"):
        reference(phase)
