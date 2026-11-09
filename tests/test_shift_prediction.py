import numpy as np
import pytest
import importlib.util
import json
from pathlib import Path
import sys

from noether_optics import shift_prediction as sp


def test_past_only_refuses_future_and_long_history():
    view = sp.PastOnly(np.zeros((3000, 17)))
    idx = np.arange(sp.HIST, 2000)
    with pytest.raises(ValueError):
        view.window(idx, [-1], [0])
    with pytest.raises(ValueError):
        view.window(idx, [sp.HIST], [0])
    assert view.window(idx, [0, 5], range(17)).shape == (len(idx), 2 * 17)


def test_past_only_rejects_history_before_recording():
    measured = np.zeros((8, 2))
    measured[-1] = 12345  # NumPy's negative indexing would expose this future frame.
    with pytest.raises(ValueError):
        sp.PastOnly(measured).window(np.array([0]), [1], [0])


@pytest.mark.parametrize("field,value", [
    ("idx", [-1]), ("idx", [8]), ("idx", [True]), ("idx", [3.0]),
    ("idx", [[3]]), ("idx", []),
    ("lags", [True]), ("lags", [1.0]), ("lags", [[1]]), ("lags", []),
    ("channels", [-1]), ("channels", [2]), ("channels", [True]),
    ("channels", [0.0]), ("channels", [[0]]), ("channels", []),
    ("idx", np.array([2**64 - 1], dtype=np.uint64)),
    ("lags", np.array([2**64 - 1], dtype=np.uint64)),
])
def test_past_only_rejects_invalid_indices(field, value):
    request = dict(idx=np.array([3]), lags=[0, 1], channels=[0, 1])
    request[field] = value
    with pytest.raises(ValueError):
        sp.PastOnly(np.zeros((8, 2))).window(**request)


@pytest.mark.parametrize("measured", [np.zeros(8), np.zeros((0, 2)), np.zeros((8, 0))])
def test_past_only_requires_nonempty_frame_channel_array(measured):
    with pytest.raises(ValueError):
        sp.PastOnly(measured)


def test_past_only_preserves_requested_order_and_returns_independent_values():
    measured = np.arange(24).reshape(8, 3)
    idx = np.array([7, 3], dtype=np.uint64)
    lags = np.array([2, 0, 1], dtype=np.uint64)
    channels = [2, 0]
    actual = sp.PastOnly(measured).window(idx, lags, channels)
    expected = [[measured[int(t) - int(lag), c] for lag in lags for c in channels] for t in idx]
    np.testing.assert_array_equal(actual, expected)
    actual[:] = -1
    np.testing.assert_array_equal(measured, np.arange(24).reshape(8, 3))


@pytest.mark.parametrize("h", [0, -1, True, 1.5, sp.NTRAIN - sp.HIST, 10**100])
def test_split_rejects_invalid_horizon(h):
    with pytest.raises(ValueError):
        sp.split(h)


def _runner():
    path = Path(__file__).resolve().parents[1] / "scripts/run_shift_prediction.py"
    spec = importlib.util.spec_from_file_location("run_shift_prediction", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_preserves_existing_evidence_before_acquisition(tmp_path, monkeypatch):
    runner = _runner()
    evidence = tmp_path / "recording"
    evidence.mkdir()
    original = evidence / "results.json"
    original.write_bytes(b'{"retained": true}\n')

    def unexpected_acquisition():
        pytest.fail("acquisition started before checking the output destination")

    monkeypatch.setattr(sp, "run_benchmark", unexpected_acquisition)
    monkeypatch.setattr(sys, "argv", ["run_shift_prediction.py", "--out", str(evidence)])
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == 2
    assert original.read_bytes() == b'{"retained": true}\n'
    assert list(evidence.iterdir()) == [original]


def test_current_report_accepts_retained_results_without_acquisition():
    archive = Path(__file__).resolve().parents[1] / "results/shift_prediction_v1"
    bench = json.loads((archive / "results.json").read_text())
    posthoc = json.loads((archive / "posthoc.json").read_text())
    env = json.loads((archive / "manifest.json").read_text())["environment"]
    report = _runner()._report(bench, posthoc, env)
    # Preserve the numerical comparison while removing unsupported conclusions.
    assert "full-sensor linear, 20 frames: +0.173 [+0.162, +0.184]" in report
    assert "positive paired interval: 0/6" in report
    assert "does not establish equivalence" in report
    assert "overlapping forecast-window/time pairs" in report


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
