import numpy as np
import pytest

from noether_optics import shift_prediction as sp


def test_past_only_refuses_future_and_long_history():
    view = sp.PastOnly(np.zeros((3000, 17)))
    idx = np.arange(sp.HIST, 2000)
    with pytest.raises(ValueError):
        view.window(idx, [-1], [0])
    with pytest.raises(ValueError):
        view.window(idx, [sp.HIST], [0])
    assert view.window(idx, [0, 5], range(17)).shape == (len(idx), 2 * 17)


@pytest.mark.parametrize("h", sp.HORIZONS)
def test_split_keeps_test_windows_after_training(h):
    tr, te = sp.split(h)
    assert tr.max() + h < sp.NTRAIN
    assert te.min() - (sp.HIST - 1) >= sp.NTRAIN


def test_kernels_recover_a_pure_tilt():
    c = (np.arange(sp.NPIX) - (sp.NPIX - 1) / 2) * sp.DX
    X, Y = np.meshgrid(c, c, indexing="ij")
    phase = 3.0 * X + 0.7 * Y + 1.2  # rad; slope 3 rad/m along the wind
    slopes = np.einsum("ij,kij->k", phase, sp.kernels())
    np.testing.assert_allclose(slopes, 3.0, rtol=1e-12)


def test_simulated_image_motion_is_physical():
    target, measured = sp.simulate(sp.REGIMES["one layer, 10 m/s"], seed=0, n=4000)
    assert measured.shape == (4000, 1 + sp.SUB * sp.SUB)
    # Kolmogorov Z-tilt for D = 0.38 m, r0 = 4 cm is 0.76 arcsec; the outer scale lowers it
    assert 0.3 < target.std() < 0.9


def test_linear_prediction_beats_no_change_ten_frames_ahead():
    target, measured = sp.simulate(sp.REGIMES["one layer, 10 m/s"], seed=0)
    view = sp.PastOnly(measured)
    tr, te = sp.split(10)
    linear = sp.GlobalLinear().fit(view, tr, target[tr + 10])
    s_linear = sp.skill(linear.predict(view, te), target[te + 10])
    s_still = sp.skill(sp.NoChange().predict(view, te), target[te + 10])
    assert s_linear > 0.8 and s_linear > s_still
