"""Passive Gaussian-receiver evaluation of the existing delayed modal plant.

    phase[t] - applied[t] @ basis -> piston-free residual -> OPD -> coupling
                   ^
             command[t-delay]

This consumes evaluator truth after a run. It supplies no controller observation
and changes neither the simulator nor its retained output schema.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def _positive(value, name: str) -> float:
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))):
        raise ValueError(f"{name} must be finite and positive")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be finite and positive") from error
    if not np.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def _float64(value, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or array.dtype != np.dtype("float64"):
        raise ValueError(f"{name} must have shape {shape} and real float64 dtype")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _residual_frame(phase, applied, basis, mask, phase_wavelength_m):
    """Reconstruct one validated frame in the loop's [y, x] array order."""
    residual = (phase - np.tensordot(applied, basis, axes=1)) * mask
    residual = residual - mask * np.sum(residual) / mask.sum()
    variance = float(np.sum(residual**2) / mask.sum())
    return residual * (phase_wavelength_m / (2 * np.pi)), variance


def evaluate_receiver(
    config: Mapping,
    cal: Mapping,
    phases,
    outputs: Mapping,
    *,
    phase_reference_wavelength_m: float,
    receiver_wavelength_m: float,
    focal_length_m: float,
    mode_radius_m: float,
    source_id: str,
    fault: str,
    pupil_amplitude=None,
) -> dict:
    """Evaluate coupling from a complete, float64 modal-loop history.

    ``phases`` contains radians at ``config['wavelength_m']``. The explicit
    phase-reference wavelength must agree with that configuration. Receiver
    wavelength is independent. The square pupil pitch is diameter/n; array
    columns are x and rows are y, both increasing along the returned axis.

    Only no fault and optical-gain drift are supported. Both preserve the
    physical residual reconstructed from the input phase and applied modal
    command. Other faults are rejected, including those outside a fault window.
    Commands are phase coefficients, so no mirror-surface factor two is added.

    The complete episode must start with the loop's zero command queue. Exact
    FIFO agreement and float64 variance agreement are required. These checks
    detect timing/scale inconsistencies; they do not authenticate arbitrary
    historical arrays or prove their calibration. ``source_id`` is a required
    caller-supplied run identity, not an independent provenance certificate.

    A fixed nonnegative pupil amplitude may be supplied, with zero transmission
    outside the loop mask. The default is that binary mask. Coupling is a
    fraction of incident pupil power in an ideal centered Gaussian mode, not
    measured received power, delivered bits or an on-sky performance result.
    Only scalar frame traces are retained. Inputs are never modified.
    """
    if fault not in ("none", "optical_gain"):
        raise ValueError("Receiver reconstruction supports only none and optical_gain faults")
    if not isinstance(source_id, str) or not source_id.strip():
        raise ValueError("A nonempty source_id is required")
    for value, required, label in (
        (config, {"n", "modes", "frames", "delay_frames", "wavelength_m", "diameter_m", "dt_s"}, "config"),
        (cal, {"basis", "mask"}, "calibration"),
        (outputs, {"command", "applied", "variance"}, "outputs"),
    ):
        if not isinstance(value, Mapping) or not required.issubset(value):
            raise ValueError(f"Missing receiver inputs in {label}")
    for key, lower in (("n", 2), ("modes", 1), ("frames", 1), ("delay_frames", 1)):
        if type(config[key]) is not int or config[key] < lower:
            raise ValueError(f"{key} must be an integer >= {lower}")
    n, modes, frames, delay = (config[key] for key in ("n", "modes", "frames", "delay_frames"))
    wavelength = _positive(phase_reference_wavelength_m, "phase_reference_wavelength_m")
    if wavelength != _positive(config["wavelength_m"], "config wavelength_m"):
        raise ValueError("Phase reference wavelength differs from the loop configuration")
    if wavelength / (2 * np.pi) == 0:
        raise ValueError("Phase reference wavelength has an unrepresentable OPD scale")
    receiver_wavelength = _positive(receiver_wavelength_m, "receiver_wavelength_m")
    focal_length = _positive(focal_length_m, "focal_length_m")
    mode_radius = _positive(mode_radius_m, "mode_radius_m")
    pitch = _positive(_positive(config["diameter_m"], "diameter_m") / n, "pupil pixel pitch")
    dt = _positive(config["dt_s"], "dt_s")
    phase = _float64(phases, (frames, n, n), "phases")
    basis = _float64(cal["basis"], (modes, n, n), "basis")
    mask = np.asarray(cal["mask"])
    if (mask.shape != (n, n) or mask.dtype.kind not in "bifu"
            or not np.isfinite(mask).all() or not np.isin(mask, (0, 1)).all() or not mask.any()):
        raise ValueError("The loop pupil mask must be nonempty and binary")
    mask = mask.astype(np.float64, copy=False)
    command = _float64(outputs["command"], (frames, modes), "command")
    applied = _float64(outputs["applied"], (frames, modes), "applied")
    variance = _float64(outputs["variance"], (frames,), "variance")
    if np.any(variance < 0):
        raise ValueError("Loop variance must be nonnegative")
    if (np.any(applied[:delay] != 0)
            or not np.array_equal(applied[delay:], command[:-delay])):
        raise ValueError("Applied commands disagree with the zero-initialized delay FIFO")
    for name, value in outputs.items():
        array = np.asarray(value)
        if array.dtype.kind not in "bifu" or not np.isfinite(array).all():
            raise ValueError(f"Loop output {name} contains nonfinite or nonnumeric values")
    if "all_finite_full_outputs" in outputs and not bool(outputs["all_finite_full_outputs"]):
        raise ValueError("Loop output records an earlier nonfinite failure")
    amplitude = mask if pupil_amplitude is None else np.asarray(pupil_amplitude)
    if (amplitude.shape != (n, n) or amplitude.dtype.kind not in "fiu"
            or not np.isfinite(amplitude).all() or np.any(amplitude < 0)
            or not np.any(amplitude > 0) or np.any(amplitude[mask == 0] != 0)):
        raise ValueError("Pupil amplitude must be finite, nonnegative, nonzero and supported by the loop mask")

    # Optional dependency: existing simulation users need not install a receiver.
    from noether_physics import gaussian_fiber_coupling

    efficiency = np.empty(frames, dtype=np.float64)
    reconstructed_variance = np.empty(frames, dtype=np.float64)
    for frame in range(frames):
        opd, reconstructed_variance[frame] = _residual_frame(
            phase[frame], applied[frame], basis, mask, wavelength)
        if (not np.isfinite(opd).all() or not np.isfinite(reconstructed_variance[frame])
                or not np.isclose(reconstructed_variance[frame], variance[frame], rtol=5e-12, atol=5e-14)):
            raise ValueError(f"Reconstructed residual variance disagrees with loop frame {frame}")
        efficiency[frame] = gaussian_fiber_coupling(
            amplitude, opd, pixel_pitch_m=pitch, wavelength_m=receiver_wavelength,
            focal_length_m=focal_length, mode_radius_m=mode_radius)
    frame_index = np.arange(frames, dtype=np.int64)
    times = frame_index * dt
    axis = (np.arange(n, dtype=np.float64) - (n - 1) / 2) * pitch
    if not np.isfinite(times).all() or not np.isfinite(axis).all() or not np.isfinite(efficiency).all():
        raise ValueError("Receiver geometry, time or coupling is unrepresentable")
    return {
        "coupling": efficiency,
        "frame_index": frame_index,
        "time_s": times,
        "pupil_axis_m": axis,
        "source_id": source_id,
        "evaluation_only": True,
        "fault": fault,
        "phase_reference_wavelength_m": wavelength,
        "receiver": {"wavelength_m": receiver_wavelength, "focal_length_m": focal_length,
                     "mode_radius_m": mode_radius, "pixel_pitch_m": pitch},
    }
