"""Independent WFS adapters and calibration metrics; no controller fitting.

The adapters share only phase inputs and units, never a propagation kernel.
Native models retain their conventions; all calibration transfer is reported.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess

import numpy as np

ENGINES = ("noether", "hcipy_matched", "hcipy_native", "specula_native")


def relative_error(actual, expected):
    """Relative Euclidean error, rejecting a zero reference instead of hiding it."""
    denominator = np.linalg.norm(expected)
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("Expected a finite nonzero reference")
    return float(np.linalg.norm(np.asarray(actual) - expected) / denominator)


def phase_rms(coefficients, gram):
    """Pupil RMS for a sampled, not necessarily orthogonal, modal basis."""
    variance = np.einsum("...i,ij,...j->...", coefficients, gram, coefficients)
    return np.sqrt(np.maximum(variance, 0))


def reconstruct(interaction, rcond):
    """Fixed-cutoff SVD inversion with rank and conditioning diagnostics."""
    interaction = np.asarray(interaction)
    if interaction.ndim != 2 or not np.isfinite(interaction).all():
        raise ValueError("Interaction matrix must be a finite two-dimensional array")
    if not 0 < rcond < 1:
        raise ValueError("SVD cutoff must lie between zero and one")
    u, s, vh = np.linalg.svd(interaction, full_matrices=False)
    if not len(s) or s[0] <= 0:
        raise ValueError("Interaction matrix has no observable modes")
    retained = s > rcond * s[0]
    inverse = np.divide(1, s, out=np.zeros_like(s), where=retained)
    return (vh.T * inverse) @ u.T, s, int(retained.sum())


def make_inputs(config, n):
    # This independent polynomial generator is shared input, not sensor code.
    from .pyramid_reference import analytic_modes
    modes = np.stack(list(analytic_modes(n, config["modes"]).values()))
    x = (np.arange(n) - (n - 1) / 2) / n
    mask = x[:, None] ** 2 + x[None, :] ** 2 <= 0.25
    gram = modes[:, mask] @ modes[:, mask].T / mask.sum()
    rng = np.random.default_rng(config["heldout_seed"])
    directions = rng.normal(size=(config["heldout_count"], len(modes)))
    directions /= phase_rms(directions, gram)[:, None]
    coefficients = np.concatenate([rms * directions for rms in config["phase_rms_rad"]])
    return {"modes": modes, "mask": mask, "gram": gram,
            "coefficients": coefficients,
            "phase_rms": np.repeat(config["phase_rms_rad"], config["heldout_count"])}


def backend(engine, config, n, mask):
    """Construct a native public API adapter with explicit units and geometry."""
    wavelength = config["wavelength_m"]
    detector_n = config["detector_n"]
    separation = config["separation_diameters"]
    metadata = {"engine": engine, "pupil_n": n, "detector_n": detector_n}
    if engine == "noether":
        import jax
        from .simulation import pyramid_sensor
        if not jax.config.jax_enable_x64:
            raise ValueError("Calibration comparison requires JAX_ENABLE_X64=1")
        sensor = jax.jit(pyramid_sensor(n, (wavelength,), wavelength, detector_n, separation))
    elif engine == "hcipy_matched":
        from .pyramid_reference import pyramid_reference
        sensor = pyramid_reference(n, (wavelength,), wavelength, detector_n, separation)
    elif engine == "hcipy_native":
        import hcipy as hp
        pupil = hp.make_pupil_grid(n, 1)
        output = hp.make_pupil_grid(4 * n, 4)
        optic = hp.PyramidWavefrontSensorOptics(
            pupil, output, separation=2 * separation, pupil_diameter=1,
            wavelength_0=wavelength, q=4, num_airy=n / 2)
        amplitude = hp.Field(mask.ravel() / np.sqrt(mask.sum() / n**2), pupil)

        def sensor(phase):
            wave = hp.Wavefront(amplitude * np.exp(1j * phase.ravel()), wavelength)
            return np.asarray(hp.subsample_field(
                optic.forward(wave).power, 4 * n // detector_n, statistic="sum"))

        metadata.update(focal_stop="circular", focal_center="interpixel", return_sign="forward")
    elif engine == "specula_native":
        import specula
        specula.init(-1, precision=0)
        from specula.data_objects.electric_field import ElectricField
        from specula.data_objects.simul_params import SimulParams
        from specula.processing_objects.modulated_pyramid import ModulatedPyramid
        field = ElectricField(n, n, 1 / n, S0=1 / (mask.sum() / n**2),
                              target_device_idx=-1, precision=0)
        field.A = mask.astype(float)
        optic = ModulatedPyramid(
            SimulParams(pixel_pupil=n, pixel_pitch=1 / n),
            wavelengthInNm=wavelength * 1e9,
            fov=wavelength * n * specula.RAD2ASEC,
            pup_diam=detector_n / 4, output_resolution=detector_n,
            pup_dist=2 * separation * detector_n / 4, pup_margin=0,
            mod_amp=0, mod_step=1, fft_res=4, target_device_idx=-1, precision=0)
        optic.inputs["in_ef"].set(field)
        optic.setup()
        tick = 0

        def sensor(phase):
            nonlocal tick
            tick += 1
            field.phaseInNm = phase * wavelength * 1e9 / (2 * np.pi)
            field.generation_time = tick
            optic.check_ready(tick)
            optic.trigger()
            optic.post_trigger()
            return np.array(optic.out_i.i, copy=True).ravel()

        metadata.update(fft_sampling=optic.fft_sampling, fft_totsize=optic.fft_totsize,
                        toccd_side=optic.toccd_side, fov_res=optic.fov_res,
                        focal_stop="circular", modulation=0, opd_unit="nm")
    else:
        raise ValueError(f"Unknown optical backend: {engine}")
    return sensor, metadata


def acquire(sensor, inputs, config, autodiff=False):
    """Retain probe images, held-out images and projected noise realizations."""
    modes = inputs["modes"]
    zero = np.zeros_like(modes[0])
    reference = np.asarray(sensor(zero))
    probes = np.stack([[np.asarray(sensor(sign * step * mode))
                        for mode in modes for sign in (1, -1)]
                       for step in (config["poke_rad"], config["poke_rad"] / 2)])
    derivatives = np.stack([(images[::2] - images[1::2]).T / (2 * step)
                            for images, step in zip(probes, (config["poke_rad"], config["poke_rad"] / 2))])
    interaction = derivatives[0]
    if autodiff:
        import jax
        import jax.numpy as jnp
        jvp = jax.jit(lambda m: jax.jvp(sensor, (jnp.asarray(zero),), (m,))[1])
        interaction = np.stack([np.asarray(jvp(jnp.asarray(mode))) for mode in modes], axis=1)
    reconstructor, singular_values, rank = reconstruct(interaction, config["pinv_rcond"])
    images = np.stack([np.asarray(sensor(np.einsum("i,ijk->jk", c, modes)))
                       for c in inputs["coefficients"]])
    piston = np.asarray(sensor(np.ones_like(zero) * 0.123))
    if not np.isfinite(reference).all() or np.min(reference) < 0:
        raise ValueError("Cannot simulate photon counts from invalid reference power")
    flux, read = config["photons"], config["read_noise_e"]
    rng = np.random.default_rng(config["noise_seed"])
    # Retain modal draws; generate one detector frame at a time to bound memory.
    noise = np.stack([reconstructor @ ((rng.poisson(reference * flux)
                      + rng.normal(0, read, reference.shape)) / flux - reference)
                      for _ in range(config["noise_count"])])
    return {"reference": reference, "probes": probes, "derivatives": derivatives,
            "interaction": interaction, "reconstructor": reconstructor,
            "singular_values": singular_values, "rank": np.asarray(rank),
            "images": images, "piston": piston, "noise_coefficients": noise}


def metrics(raw, inputs, config):
    """Derive local calibration verdicts and noise diagnostics from saved arrays."""
    matrix, ref, images = raw["interaction"], raw["reference"], raw["images"]
    recon, s, rank = reconstruct(matrix, config["pinv_rcond"])
    coeff, gram = inputs["coefficients"], inputs["gram"]
    estimates = (images - ref) @ recon.T
    errors = phase_rms(estimates - coeff, gram) / inputs["phase_rms"]
    small = inputs["phase_rms"] == min(config["phase_rms_rad"])
    derivative_error = max(relative_error(raw["derivatives"][0], raw["derivatives"][1]),
                           relative_error(matrix, raw["derivatives"][1]))
    piston_error = relative_error(raw["piston"], ref)
    # These deliberate calibration errors must be caught, not calibrated away.
    controls = {"reversed_sign": -estimates,
                "nm_as_radians": estimates * (2 * np.pi / (config["wavelength_m"] * 1e9))}
    control_errors = {name: float(np.min(phase_rms(value[small] - coeff[small], gram)
                                               / inputs["phase_rms"][small]))
                      for name, value in controls.items()}
    all_images = np.concatenate([images, raw["probes"].reshape(-1, ref.size),
                                 ref[None], raw["piston"][None]])
    finite = bool(np.isfinite(all_images).all() and (all_images >= 0).all())
    powers = all_images.sum(axis=1)
    t = config["tolerances"]
    gates = {"finite_nonnegative": finite,
             "power": bool(np.all((powers > 0) & (powers <= 1 + 1e-10))),
             "full_rank": rank == len(coeff[0]),
             "derivative_convergence": derivative_error <= t["derivative_step_relative_l2"],
             "piston_invariance": piston_error <= t["piston_relative_l2"],
             "small_signal": bool(errors[small].max() <= t["small_signal_phase_relative_rms"]),
             "negative_controls": min(control_errors.values()) > t["negative_control_minimum_relative_rms"]}
    variance = ref / config["photons"] + (config["read_noise_e"] / config["photons"])**2
    predicted_cov = (recon * variance) @ recon.T
    empirical_cov = np.cov(raw["noise_coefficients"], rowvar=False, ddof=1)
    return {"gates": gates, "all_pass": all(gates.values()), "rank": rank,
            "condition_number": float(s[0] / s[-1]) if rank == len(coeff[0]) else None,
            "singular_values": s.tolist(), "reference_power": float(ref.sum()),
            "derivative_relative_l2": derivative_error, "piston_relative_l2": piston_error,
            "phase_relative_rms": errors.tolist(), "negative_controls": control_errors,
            "predicted_noise_covariance": predicted_cov.tolist(),
            "empirical_noise_covariance": empirical_cov.tolist(),
            "noise_variance_ratio": (np.diag(empirical_cov) / np.diag(predicted_cov)).tolist()}


def transfer(source, target, inputs, config):
    """Test a fixed source calibration on another backend without alignment fits."""
    recon, _, _ = reconstruct(source["interaction"], config["pinv_rcond"])
    result = {}
    for name, zero in (("matrix_only", target["reference"]),
                       ("matrix_and_reference", source["reference"])):
        estimates = (target["images"] - zero) @ recon.T
        result[name] = (phase_rms(estimates - inputs["coefficients"], inputs["gram"])
                        / inputs["phase_rms"]).tolist()
    result["reference_relative_l2"] = relative_error(target["reference"], source["reference"])
    result["interaction_relative_l2"] = relative_error(target["interaction"], source["interaction"])
    return result


def source_metadata(engine):
    """Hash installed library Python sources and record separate runtime versions."""
    libraries = ("jax", "jaxlib", "dLux") if engine == "noether" else (
        ("specula",) if engine == "specula_native" else ("hcipy",))
    roots = {}
    for name in libraries:
        module = __import__(name)
        root = Path(module.__file__).resolve().parent
        files = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(root.rglob("*.py"))}
        info = {"python_sources_sha256": hashlib.sha256(
            json.dumps(files, sort_keys=True).encode()).hexdigest(), "source_files": files}
        if (root.parent / ".git").exists():
            info["git_commit"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root.parent, text=True).strip()
            info["tracked_dirty"] = bool(subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"], cwd=root.parent, text=True))
        roots[name] = info
    return {"python": platform.python_version(), "libraries": roots,
            "packages": dict(sorted((d.metadata["Name"], d.version)
                                    for d in importlib.metadata.distributions()))}
