"""Numerical adapters for the predeclared study; no persistence or cohort execution.

    noisy pilot measurement + applied command -> measured pseudo-open loop
       -> independent per-episode two-tap fits -> equal-weight coefficients

    shared atmosphere + physical disturbance -> zero-DM reference
    full closed-loop outputs + reference -> paired post-burn/fault-window metrics

These helpers reuse the unchanged simulator. A stage runner must enforce the
ledger gate, frozen models, real provenance and reserved-cohort access policy.
"""
from __future__ import annotations

from copy import deepcopy
from itertools import product
import json

import jax
import jax.numpy as jnp
import numpy as np

from .controller_protocol import study_plan, validate_study
from .simulation import CONTROLLERS, loop, predict_fit

STUDY_FAULTS = ("none", "optical_gain", "vibration")
FULL_OUTPUT_KEYS = frozenset(("variance", "strehl", "saturated", "score", "alarm",
                              "fallback", "command", "applied", "estimate", "unavailable",
                              "measurement", "prediction"))


def _identity(value) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False)


def training_cases(study: dict) -> list[dict]:
    validate_study(study)
    return [dict(seed=seed, photons=photons, wind_scale=wind)
            for photons, wind, seed in product(study["conditions"]["photons"],
                                               study["conditions"]["wind_scales"], study["seeds"]["training"])]


def episode_config(base: dict, study: dict, stage: str, photons: float,
                   wind_scale: float, candidate: dict | None = None) -> dict:
    """Apply only declared overrides; callers must verify the pinned base on load."""
    validate_study(study)
    if stage not in ("training", "validation", "evaluation"):
        raise ValueError("Unknown study stage")
    if photons not in study["conditions"]["photons"] or wind_scale not in study["conditions"]["wind_scales"]:
        raise ValueError("Undeclared study regime")
    if candidate is not None and _identity(candidate) not in {
        _identity(row) for row in study_plan(study)["candidates"]
    }:
        raise ValueError("Undeclared candidate")
    if stage != "training" and candidate is None:
        raise ValueError("Validation/evaluation requires a declared candidate")
    cfg = deepcopy(base)
    # The loop takes an explicit seed; never leave misleading legacy cohort labels.
    cfg.pop("seed", None)
    cfg.update(seeds=list(study["seeds"][stage]), training_seeds=list(study["seeds"]["training"]))
    settings = study["training"] if stage == "training" else study["loop"]
    cfg.update(frames=settings["frames"], burn_in=settings["burn_in"],
               photons=photons, wind_scale=wind_scale)
    if stage == "training":
        cfg.update(gain=settings["pilot_gain"], leak=settings["pilot_leak"])
        # Training is clean. No fault window lies within a training episode.
        cfg["faults"].update(onset=cfg["frames"], end=cfg["frames"])
    else:
        cfg["faults"].update(onset=settings["fault_onset"], end=settings["fault_end"])
    if candidate is not None:
        cfg.update({key: value for key, value in candidate["parameters"].items() if key in ("gain", "leak")})
    return cfg


def measured_telemetry(outputs: dict, cal: dict) -> np.ndarray:
    """Reconstruct pilot telemetry using observations and applied DM only.

    Deliberately does not consume phase truth or the simulator's estimate output.
    For the pilot integrator this must agree with its retained estimate stream.
    """
    measured, applied = np.asarray(outputs["measurement"]), np.asarray(outputs["applied"])
    recon, reference = np.asarray(cal["reconstructor"]), np.asarray(cal["reference"])
    if (measured.ndim != 2 or applied.ndim != 2 or measured.shape[0] != applied.shape[0]
            or recon.ndim != 2 or reference.shape != (recon.shape[1],)
            or measured.shape[1] != recon.shape[1] or recon.shape[0] > applied.shape[1]):
        raise ValueError("Incompatible pilot measurement/calibration shapes")
    if not all(np.isfinite(value).all() for value in (measured, applied, recon, reference)):
        raise ValueError("Nonfinite measured pilot telemetry")
    result = applied.astype(np.float64, copy=True)
    result[:, :recon.shape[0]] += (measured - reference) @ recon.T
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite reconstructed pilot telemetry")
    return result


def fit_measured_predictor(study: dict, candidate: dict, episodes: list[dict], modes: int) -> dict:
    """Require the complete training cohort; fit each burned-in sequence separately.

    Episodes contain only a training case and its measured telemetry. Individual
    coefficients are returned so their equal-weight average is auditable.
    """
    expected = training_cases(study)
    if (_identity(candidate) not in {_identity(row) for row in study_plan(study)["candidates"]}
            or candidate["controller"] != "predictive"):
        raise ValueError("Expected a declared predictive candidate")
    if type(modes) is not int or modes < 1:
        raise ValueError("Mode count must be a positive integer")
    found = {}
    for episode in episodes:
        if set(episode) != {"case", "telemetry"}:
            raise ValueError("Predictor episodes accept measured telemetry only")
        key = _identity(episode["case"])
        if key not in {_identity(case) for case in expected} or key in found:
            raise ValueError("Foreign or duplicate training episode")
        array = np.asarray(episode["telemetry"])
        if array.shape != (study["training"]["frames"], modes) or not np.isfinite(array).all():
            raise ValueError("Incomplete or nonfinite training telemetry")
        found[key] = array
    if len(found) != len(expected):
        raise ValueError("Incomplete training cohort")
    parameters = candidate["parameters"]
    coefficients = np.stack([predict_fit(found[_identity(case)][study["training"]["burn_in"]:],
                                        parameters["horizon"], parameters["ridge"]) for case in expected])
    mean = coefficients.mean(axis=0)
    if not np.isfinite(coefficients).all() or not np.isfinite(mean).all():
        raise ValueError("Nonfinite predictor coefficients")
    return {"cases": expected, "episode_coefficients": coefficients, "coefficients": mean}


def controller_kernel(cfg: dict, cal: dict, controller: str, fault: str = "none",
                      predictor: np.ndarray | None = None):
    """Compile a reusable exact-Poisson/read-noise kernel, with fallback disabled.

    Construct this only after the appropriate runner stage gate. Network weights
    stay dynamic within a fixed condition, gain/leak setting and parameter shape.
    Gaussian learned training will use a separate, explicit training path.
    """
    if controller not in CONTROLLERS or fault not in STUDY_FAULTS:
        raise ValueError("Unsupported study controller or fault")
    if cfg["sensor"] != "pyramid":
        raise ValueError("The study requires the pyramid sensor and Poisson/read noise")
    config = deepcopy(cfg)
    if predictor is None:
        if controller == "predictive":
            raise ValueError("Predictive trials require fitted coefficients")
        predictor = np.zeros((config["modes"], 2))
    predictor = np.array(predictor, dtype=np.float64, copy=True)
    if predictor.shape != (config["modes"], 2) or not np.isfinite(predictor).all():
        raise ValueError("Invalid frozen predictor coefficients")
    compiled = jax.jit(lambda phases, seed, params: loop(
        config, cal, phases, seed, controller, jnp.asarray(predictor),
        fault=fault, params=params, fallback=False, differentiable_noise=False))

    def execute(phases, seed, params=None):
        if np.shape(phases) != (config["frames"], config["n"], config["n"]):
            raise ValueError("Incomplete or incompatible atmosphere array")
        if type(seed) is not int or not 0 <= seed < 2**32 - 100000:
            raise ValueError("Atmosphere/detector seed must fit uint32")
        if controller == "learned":
            shapes = ((3, 8), (8,), (8, 1), (1,))
            if (params is None or len(params) != 4 or any(np.shape(p) != shape for p, shape in zip(params, shapes))
                    or not all(np.isfinite(np.asarray(p)).all() for p in params)):
                raise ValueError("Learned trials require the frozen width-eight model")
        elif params is not None:
            raise ValueError("Only learned trials accept network parameters")
        return jax.block_until_ready(compiled(jnp.asarray(phases), seed, params))
    return execute


def paired_open_loop(phases, cfg: dict, cal: dict, fault: str) -> dict:
    """Zero-DM metrics with the same physical disturbance as simulation.loop."""
    if fault not in STUDY_FAULTS:
        raise ValueError("Unsupported study fault")
    if np.shape(phases) != (cfg["frames"], cfg["n"], cfg["n"]):
        raise ValueError("Incomplete or incompatible atmosphere array")
    phase = jnp.asarray(phases)
    if fault == "vibration":
        frame = jnp.arange(cfg["frames"])
        fcfg = cfg["faults"]
        active = (frame >= fcfg["onset"]) & (frame < fcfg["end"])
        amplitude = jnp.where(active, fcfg["vibration_rad"] * jnp.sin(
            2 * jnp.pi * fcfg["vibration_hz"] * frame * cfg["dt_s"]), 0)
        phase = phase + amplitude[:, None, None] * cal["basis"][0]
    # Optical-gain drift changes the sensor only, not the atmospheric truth.
    mask = cal["mask"]
    residual = phase * mask
    residual = residual - mask * jnp.sum(residual, axis=(-2, -1))[:, None, None] / mask.sum()
    return {"variance": jnp.sum(residual**2, axis=(-2, -1)) / mask.sum(),
            "strehl": jnp.abs(jnp.sum(mask * jnp.exp(1j * residual), axis=(-2, -1)) / mask.sum())**2,
            "saturated": jnp.zeros(cfg["frames"], dtype=bool)}


def paired_metrics(outputs: dict, reference: dict, cfg: dict) -> dict:
    """Derive ledger metrics plus interval summaries, retaining nonfinite failures.

    If bulky streams were omitted from a saved trial, a retained scalar boolean
    all_finite_full_outputs is mandatory. A false flag cannot be hidden by finite
    post-burn arrays. The caller owns reference identity/physical pairing checks.
    """
    def inspect(values, require_full):
        arrays = {key: np.asarray(value) for key, value in values.items()}
        if not {"variance", "strehl", "saturated"}.issubset(arrays):
            raise ValueError("Missing per-frame metrics")
        if require_full and not FULL_OUTPUT_KEYS.issubset(arrays) and "all_finite_full_outputs" not in arrays:
            raise ValueError("Missing full-output finite evidence")
        for key, value in arrays.items():
            if key == "all_finite_full_outputs":
                if value.shape != () or value.dtype != np.dtype(bool):
                    raise ValueError("Full-output finite flag must be a scalar boolean")
            elif value.ndim < 1 or value.shape[0] != cfg["frames"]:
                raise ValueError("Incomplete per-frame output")
        if any(arrays[key].shape != (cfg["frames"],) for key in ("variance", "strehl", "saturated")):
            raise ValueError("Metric streams must be one-dimensional")
        if arrays["saturated"].dtype != np.dtype(bool):
            raise ValueError("Saturation stream must be boolean")
        if np.any(arrays["variance"] < 0):
            raise ValueError("Negative physical variance")
        finite = all(np.isfinite(value).all() for value in arrays.values())
        return arrays, bool(finite and arrays.get("all_finite_full_outputs", True))

    closed, cf = inspect(outputs, True)
    opened, of = inspect(reference, False)
    start, frames = cfg["burn_in"], cfg["frames"]
    if not 0 <= start < frames:
        raise ValueError("Invalid post-burn interval")
    def scalar(value):
        return float(value) if np.isfinite(value) else None
    def interval(start, stop):
        sl = slice(start, stop)
        return {"all_outputs_finite": cf and of,
                "wfe_nm": scalar(np.sqrt(closed["variance"][sl].mean()) * cfg["wavelength_m"] / (2 * np.pi) * 1e9),
                "open_wfe_nm": scalar(np.sqrt(opened["variance"][sl].mean()) * cfg["wavelength_m"] / (2 * np.pi) * 1e9),
                "saturation_fraction": scalar(closed["saturated"][sl].mean()),
                "strehl": scalar(closed["strehl"][sl].mean())}
    onset, end = cfg["faults"]["onset"], cfg["faults"]["end"]
    if not 0 <= onset <= end <= frames:
        raise ValueError("Fault window is outside the episode")
    return {"metrics": interval(start, frames),
            "post_burn_frames": [start, frames],
            "fault_window": interval(onset, end) if onset < end else None,
            "fault_window_frames": [onset, end]}
