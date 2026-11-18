"""Receiver units and timing, with exact finite-cell and actual-loop checks."""
from copy import deepcopy
import math
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("noether_physics")

from noether_physics import gaussian_fiber_coupling
from noether_optics.artifacts import load_config
from noether_optics.receiver import _residual_frame, evaluate_receiver
from noether_optics.simulation import calibration, loop

PHASE_WAVELENGTH = 750e-9
RECEIVER_WAVELENGTH = 1550e-9
FOCAL_LENGTH = 0.1
PUPIL_MODE_RADIUS = 1e-3
MODE_RADIUS = RECEIVER_WAVELENGTH * FOCAL_LENGTH / (np.pi * PUPIL_MODE_RADIUS)


def analytic_case(command=0.07):
    """An exact zero-piston, unit-variance two-column modal field."""
    cfg = dict(n=2, modes=1, frames=6, delay_frames=2, wavelength_m=PHASE_WAVELENGTH,
               diameter_m=0.6e-3, dt_s=1e-3)
    b = np.array([[-1.0, 1.0], [-1.0, 1.0]])
    a = 0.23 + 0.03 * np.arange(cfg["frames"])
    issued = np.full((cfg["frames"], 1), command, dtype=np.float64)
    applied = np.zeros_like(issued)
    applied[2:] = issued[:-2]
    outputs = {"command": issued, "applied": applied,
               "variance": (a - applied[:, 0])**2}
    return cfg, {"basis": b[None], "mask": np.ones((2, 2))}, a[:, None, None] * b, outputs


def observe(case, **overrides):
    cfg, cal, phase, outputs = case
    options = dict(phase_reference_wavelength_m=cfg["wavelength_m"],
                   receiver_wavelength_m=RECEIVER_WAVELENGTH, focal_length_m=FOCAL_LENGTH,
                   mode_radius_m=MODE_RADIUS, source_id="receiver-conformance", fault="none")
    options.update(overrides)
    return evaluate_receiver(cfg, cal, phase, outputs, **options)


def exact_prefactor(pitch, receiver_wavelength=RECEIVER_WAVELENGTH):
    wp = receiver_wavelength * FOCAL_LENGTH / (math.pi * MODE_RADIUS)
    h = pitch / wp
    integral = math.sqrt(math.pi) / 2 * math.erf(h)
    return 4 * integral**4 / (h*h * (math.pi / 2))


def test_exact_finite_cell_modal_sign_and_normalization():
    case = analytic_case()
    cfg, _, phase, outputs = case
    delta = phase[:, 0, 1] - outputs["applied"][:, 0]
    eta0 = exact_prefactor(cfg["diameter_m"] / cfg["n"])
    expected = eta0 * np.cos(PHASE_WAVELENGTH / RECEIVER_WAVELENGTH * delta)**2
    trace = observe(case)
    np.testing.assert_allclose(trace["coupling"], expected, rtol=2e-13, atol=0)
    np.testing.assert_array_equal(trace["frame_index"], np.arange(6))
    np.testing.assert_array_equal(trace["time_s"], np.arange(6) * cfg["dt_s"])
    assert trace["evaluation_only"] is True
    assert trace["source_id"] == "receiver-conformance"
    assert trace["receiver"]["pixel_pitch_m"] == cfg["diameter_m"] / cfg["n"]
    assert trace["phase_reference_wavelength_m"] == PHASE_WAVELENGTH


def test_two_phase_reference_wavelengths_preserve_the_same_physical_field():
    original = analytic_case()
    reference = observe(original)["coupling"]
    for wavelength in (650e-9, 1000e-9):
        cfg, cal, phase, outputs = deepcopy(original)
        ratio = PHASE_WAVELENGTH / wavelength
        cfg["wavelength_m"] = wavelength
        phase *= ratio
        outputs["command"] *= ratio
        outputs["applied"] *= ratio
        outputs["variance"] *= ratio**2
        np.testing.assert_allclose(observe((cfg, cal, phase, outputs))["coupling"], reference,
                                   rtol=2e-13, atol=0)
    with pytest.raises(ValueError, match="reference wavelength"):
        observe(original, phase_reference_wavelength_m=RECEIVER_WAVELENGTH)
    # Changing the receiver wavelength is a physically different overlap.
    changed = observe(original, receiver_wavelength_m=1100e-9)["coupling"]
    cfg, _, phase, outputs = original
    expected = exact_prefactor(cfg["diameter_m"] / cfg["n"], 1100e-9) * np.cos(
        PHASE_WAVELENGTH / 1100e-9 * (phase[:, 0, 1] - outputs["applied"][:, 0]))**2
    np.testing.assert_allclose(changed, expected, rtol=2e-13, atol=0)
    assert not np.allclose(changed, reference)


def test_modal_derivative_and_physical_surface_factor_two_agree():
    c, step = 0.07, 1e-4
    case = analytic_case(c)
    cfg, cal, phase, outputs = case
    frame = 3
    a = phase[frame, 0, 1]  # Nonzero incoming aberration distinguishes the sign.
    eta0 = exact_prefactor(cfg["diameter_m"] / cfg["n"])
    ratio = PHASE_WAVELENGTH / RECEIVER_WAVELENGTH
    expected_dc = eta0 * ratio * math.sin(2 * ratio * (a - c))
    finite_dc = (observe(analytic_case(c + step))["coupling"][frame]
                 - observe(analytic_case(c - step))["coupling"][frame]) / (2 * step)
    assert expected_dc > 0
    assert finite_dc == pytest.approx(expected_dc, rel=2e-8, abs=0)
    incoming_opd = PHASE_WAVELENGTH * a / (2 * np.pi)
    surface = -PHASE_WAVELENGTH * c / (4 * np.pi)

    def physical_receiver(surface_m):
        # A physical reflective mirror adds twice its surface displacement.
        opd = (incoming_opd + 2 * surface_m) * cal["basis"][0]
        return gaussian_fiber_coupling(
            cal["mask"], opd, pixel_pitch_m=cfg["diameter_m"] / cfg["n"],
            wavelength_m=RECEIVER_WAVELENGTH, focal_length_m=FOCAL_LENGTH, mode_radius_m=MODE_RADIUS)

    assert physical_receiver(surface) == pytest.approx(observe(case)["coupling"][frame], rel=2e-13, abs=0)
    surface_step = PHASE_WAVELENGTH * step / (4 * np.pi)
    finite_ds = (physical_receiver(surface + surface_step)
                 - physical_receiver(surface - surface_step)) / (2 * surface_step)
    delta = incoming_opd + 2 * surface
    expected_ds = -4 * np.pi / RECEIVER_WAVELENGTH * eta0 * math.sin(4 * np.pi * delta / RECEIVER_WAVELENGTH)
    assert finite_ds == pytest.approx(expected_ds, rel=2e-8, abs=0)
    assert finite_dc == pytest.approx(finite_ds * (-PHASE_WAVELENGTH / (4 * np.pi)), rel=2e-8, abs=0)


def test_asymmetric_spatial_opd_and_axis_order_are_preserved():
    phase = np.array([[0.0, 0.03, 0.11, 0.07], [0.13, 0.02, 0.17, 0.05],
                      [0.19, 0.23, 0.01, 0.29], [0.31, 0.04, 0.37, 0.41]])
    basis = np.zeros((1, 4, 4))
    mask = np.ones((4, 4))
    expected = (phase - phase.mean()) * PHASE_WAVELENGTH / (2 * np.pi)
    opd, variance = _residual_frame(phase, np.zeros(1), basis, mask, PHASE_WAVELENGTH)
    np.testing.assert_allclose(opd, expected, rtol=2e-15, atol=0)
    assert opd[0, 2] != opd[2, 0]
    cfg = dict(n=4, modes=1, frames=1, delay_frames=2, wavelength_m=PHASE_WAVELENGTH,
               diameter_m=4e-3, dt_s=1e-3)
    outputs = dict(command=np.zeros((1, 1)), applied=np.zeros((1, 1)), variance=np.array([variance]))
    trace = observe((cfg, dict(basis=basis, mask=mask), phase[None], outputs))
    np.testing.assert_allclose(trace["pupil_axis_m"], [-1.5e-3, -0.5e-3, 0.5e-3, 1.5e-3], atol=0, rtol=0)


@pytest.fixture
def loop_case():
    cfg = load_config(Path(__file__).parents[1] / "configs/simulation.yaml")
    cfg.update(n=8, modes=3, frames=8, burn_in=0, sensor="linear", diameter_m=6e-3)
    cfg["faults"].update(onset=3, end=8, drift_frames=2)
    cal = calibration(cfg)
    t = np.arange(cfg["frames"])
    phase = (0.2 * np.sin(0.7 * t)[:, None, None] * np.asarray(cal["basis"])[0]
             + 0.13 * np.asarray(cal["basis"])[1])
    return cfg, cal, jnp.asarray(phase)


def run_loop(cfg, cal, phase, **options):
    return loop(cfg, cal, phase, 91027, "integrator", jnp.zeros((cfg["modes"], 2)), **options)


@pytest.mark.parametrize("fault", ["none", "optical_gain"])
def test_actual_loop_observation_is_passive_and_preserves_every_output(loop_case, fault):
    cfg, cal, phase = loop_case
    cfg_before = deepcopy(cfg)
    phase_before = np.array(phase)
    cal_before = {key: np.array(value) for key, value in cal.items() if not callable(value)}
    outputs = run_loop(cfg, cal, phase, fault=fault)
    original = {key: np.array(value) for key, value in outputs.items()}
    trace = observe((cfg, cal, phase, outputs), fault=fault)
    assert np.isfinite(trace["coupling"]).all()
    assert np.all((trace["coupling"] >= 0) & (trace["coupling"] <= 1))
    again = run_loop(cfg, cal, phase, fault=fault)
    assert cfg == cfg_before
    np.testing.assert_array_equal(phase, phase_before)
    for key, value in original.items():
        np.testing.assert_array_equal(outputs[key], value)
        np.testing.assert_array_equal(again[key], value)
    for key, value in cal_before.items():
        np.testing.assert_array_equal(cal[key], value)


@pytest.mark.parametrize("delay", [1, 2, 3])
def test_probe_impulse_affects_coupling_only_when_the_command_arrives(loop_case, delay):
    cfg, cal, phase = loop_case
    cfg.update(gain=0.0, leak=0.0, delay_frames=delay)
    phase = jnp.zeros_like(phase)
    probes = jnp.zeros((cfg["frames"], cfg["modes"])).at[1, 1].set(0.2)
    outputs = run_loop(cfg, cal, phase, command_probe=probes)
    trace = observe((cfg, cal, phase, outputs))
    eta = trace["coupling"]
    arrival = 1 + delay
    assert eta[arrival] < eta[0]
    np.testing.assert_array_equal(np.delete(eta, arrival), np.full(cfg["frames"] - 1, eta[0]))
    np.testing.assert_array_equal(np.asarray(outputs["applied"])[arrival], probes[1])


def test_future_phase_and_commands_cannot_change_earlier_receiver_values(loop_case):
    cfg, cal, phase = loop_case
    cut = 4
    first = run_loop(cfg, cal, phase)
    future = phase.at[cut:].multiply(3)
    second = run_loop(cfg, cal, future)
    a = observe((cfg, cal, phase, first))
    b = observe((cfg, cal, future, second))
    np.testing.assert_array_equal(a["coupling"][:cut], b["coupling"][:cut])
    for key in first:
        np.testing.assert_array_equal(first[key][:cut], second[key][:cut])


@pytest.mark.parametrize("field", ["command", "applied", "variance"])
def test_wrong_frame_or_scaled_loop_evidence_is_rejected(field):
    cfg, cal, phase, outputs = analytic_case()
    if field == "applied":
        outputs[field] = outputs["command"].copy()
    elif field == "command":
        outputs[field][0] += 0.01
    else:
        outputs[field] *= (PHASE_WAVELENGTH / (2 * np.pi))**2  # OPD m² is not rad².
    with pytest.raises(ValueError, match="FIFO|variance"):
        observe((cfg, cal, phase, outputs))


@pytest.mark.parametrize("fault", ["stuck_actuator", "vibration", "misregistration", "model_mismatch",
                                   "late_twin", "missing_twin", "unknown"])
def test_unsupported_faults_are_rejected_even_for_consistent_arrays(fault):
    with pytest.raises(ValueError, match="supports only"):
        observe(analytic_case(), fault=fault)


@pytest.mark.parametrize("field", ["phase", "basis", "mask", "command", "applied", "variance"])
@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_nonfinite_inputs_are_rejected(field, bad):
    cfg, cal, phase, outputs = analytic_case()
    value = phase if field == "phase" else cal[field] if field in cal else outputs[field]
    value.flat[0] = bad
    with pytest.raises(ValueError, match="finite|binary"):
        observe((cfg, cal, phase, outputs))


@pytest.mark.parametrize("field", ["phase", "basis", "mask", "command", "applied", "variance"])
def test_wrong_shapes_are_rejected(field):
    cfg, cal, phase, outputs = analytic_case()
    if field == "phase":
        phase = phase[:-1]
    elif field in cal:
        cal[field] = cal[field][:-1]
    else:
        outputs[field] = outputs[field][:-1]
    with pytest.raises(ValueError, match="shape|mask"):
        observe((cfg, cal, phase, outputs))


@pytest.mark.parametrize("dtype", [np.float32, np.complex128, np.int64])
def test_wrong_phase_dtype_is_rejected(dtype):
    cfg, cal, phase, outputs = analytic_case()
    with pytest.raises(ValueError, match="float64"):
        observe((cfg, cal, phase.astype(dtype), outputs))


@pytest.mark.parametrize("bad", [0.0, -1.0, np.inf, np.nan, True, "0.1"])
def test_invalid_receiver_geometry_is_rejected(bad):
    with pytest.raises(ValueError, match="finite and positive"):
        observe(analytic_case(), mode_radius_m=bad)


def test_unrepresentable_phase_conversion_is_rejected():
    case = analytic_case()
    case[0]["wavelength_m"] = np.nextafter(0.0, 1.0)
    with pytest.raises(ValueError, match="unrepresentable OPD scale"):
        observe(case)


def test_mask_amplitude_and_source_identity_boundaries():
    case = analytic_case()
    for amplitude in (np.zeros((2, 2)), -np.ones((2, 2)), np.full((2, 2), np.nan)):
        with pytest.raises(ValueError, match="Pupil amplitude"):
            observe(case, pupil_amplitude=amplitude)
    case[1]["mask"][0, 0] = 0
    with pytest.raises(ValueError, match="supported by the loop mask"):
        observe(case, pupil_amplitude=np.ones((2, 2)))
    with pytest.raises(ValueError, match="source_id"):
        observe(analytic_case(), source_id=" ")
    missing = analytic_case()
    del missing[3]["variance"]
    with pytest.raises(ValueError, match="Missing receiver inputs"):
        observe(missing)


def test_amplitude_scaling_preserves_coupling_without_mutating_input():
    case = analytic_case()
    amplitude = np.array([[0.4, 0.7], [1.0, 0.2]])
    original = amplitude.copy()
    a = observe(case, pupil_amplitude=amplitude)
    b = observe(case, pupil_amplitude=amplitude * 1e100)
    np.testing.assert_allclose(a["coupling"], b["coupling"], rtol=2e-13, atol=0)
    np.testing.assert_array_equal(amplitude, original)
