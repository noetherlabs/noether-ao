"""Run or rebuild the predeclared independent ideal-pyramid comparison."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import hcipy as hp
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from noether_optics.artifacts import load_config, start_run, write_json, write_table
from noether_optics.optics import relative_l2
from noether_optics.pyramid_reference import analytic_modes, pyramid_reference
from noether_optics.simulation import pyramid_sensor

CONTROL_NAMES = ("phase_sign", "separation_plus_10_percent", "transpose", "power_plus_10_percent")


def run(config: dict, out: Path) -> None:
    """Store independent pixel arrays before deriving any verdicts or plots."""
    path = start_run(config, out)
    (path / "raw").mkdir()
    provenance = json.loads((path / "provenance.json").read_text())
    provenance["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    provenance["reference_library_sources"] = {}
    for obj in (hp.PyramidWavefrontSensorOptics, hp.FraunhoferPropagator,
                hp.MatrixFourierTransform, hp.SurfaceApodizer, hp.make_uniform_grid,
                hp.Wavefront, hp.subsample_field):
        source = Path(inspect.getsourcefile(obj))
        provenance["reference_library_sources"][obj.__name__] = {
            "file": str(source.relative_to(Path(hp.__file__).parent)),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }
    write_json(path / "provenance.json", provenance)
    ref_wl = config["reference_wavelength_m"]
    detector_n = config["detector_n"]
    separation = config["separation_diameters"]
    step = config["derivative_step_rad"]
    for n in config["grids"]:
        modes = analytic_modes(n, config["modes"])
        directions = np.stack(list(modes.values()))
        phases = np.concatenate([np.zeros((1, n, n)), config["phase_rms_rad"] * directions])
        asymmetric = config["phase_rms_rad"] * np.einsum(
            "i,ijk->jk", np.array([.5, .2, .3, .4, .1]), directions
        )
        for spectrum, wavelengths in enumerate(config["wavelength_sets_m"]):
            reference = pyramid_reference(n, wavelengths, ref_wl, detector_n, separation)
            production = jax.jit(pyramid_sensor(n, tuple(wavelengths), ref_wl, detector_n, separation))
            derivative = jax.jit(lambda phase, direction: jax.jvp(
                production, (phase,), (direction,)
            )[1])
            actual = np.stack([np.asarray(production(phase)) for phase in phases])
            expected = np.stack([reference(phase) for phase in phases])
            actual_derivatives = np.stack([
                np.asarray(derivative(jnp.asarray(phase), jnp.asarray(direction)))
                for phase, direction in zip(phases[1:], directions)
            ])
            expected_derivatives = np.stack([
                (reference(phase + step * direction) - reference(phase - step * direction)) / (2 * step)
                for phase, direction in zip(phases[1:], directions)
            ])
            baseline = np.asarray(production(asymmetric))
            wrong_geometry = jax.jit(pyramid_sensor(
                n, tuple(wavelengths), ref_wl, detector_n, separation * 1.1
            ))
            controls = np.stack([
                np.asarray(production(-asymmetric)), np.asarray(wrong_geometry(asymmetric)),
                baseline.reshape(detector_n, detector_n).T.ravel(), baseline * 1.1,
            ])
            np.savez_compressed(path / "raw" / f"n{n}_spectrum{spectrum}.npz",
                                phases=phases, directions=directions, actual=actual,
                                expected=expected, actual_derivatives=actual_derivatives,
                                expected_derivatives=expected_derivatives,
                                control_phase=asymmetric, control_actual=controls,
                                control_expected=reference(asymmetric))
            print(f"Stored n={n}, spectrum={spectrum}", flush=True)


def rebuild(path: Path) -> dict:
    """Recompute every reported value from retained arrays and frozen settings."""
    config = json.loads((path / "config.json").read_text())
    tolerance = config["tolerances"]
    pixels, derivatives, controls = [], [], []
    for n in config["grids"]:
        for spectrum, wavelengths in enumerate(config["wavelength_sets_m"]):
            with np.load(path / "raw" / f"n{n}_spectrum{spectrum}.npz") as raw:
                for index, mode in enumerate(["flat", *config["modes"]]):
                    actual, expected = raw["actual"][index], raw["expected"][index]
                    error = relative_l2(actual, expected)
                    power_error = max(abs(float(actual.sum()) - 1), abs(float(expected.sum()) - 1))
                    finite_positive = bool(np.isfinite(actual).all() and np.isfinite(expected).all()
                                           and actual.min() >= 0 and expected.min() >= 0)
                    pixels.append({"n": n, "spectrum": spectrum, "wavelength_count": len(wavelengths),
                                   "mode": mode, "relative_l2": error,
                                   "production_power": float(actual.sum()),
                                   "reference_power": float(expected.sum()),
                                   "max_power_error": power_error,
                                   "pass": finite_positive and error <= tolerance["intensity_relative_l2"]
                                   and power_error <= tolerance["total_power_absolute"]})
                for index, mode in enumerate(config["modes"]):
                    actual = raw["actual_derivatives"][index]
                    expected = raw["expected_derivatives"][index]
                    error = relative_l2(actual, expected)
                    derivatives.append({"n": n, "spectrum": spectrum, "mode": mode,
                                        "relative_l2": error,
                                        "pass": error <= tolerance["derivative_relative_l2"]})
                for index, name in enumerate(CONTROL_NAMES):
                    error = relative_l2(raw["control_actual"][index], raw["control_expected"])
                    controls.append({"n": n, "spectrum": spectrum, "fault": name,
                                     "relative_l2": error,
                                     "detected": error > tolerance["minimum_control_relative_l2"]})
    for name, rows in (("pixels", pixels), ("derivatives", derivatives), ("controls", controls)):
        write_table(path / name, rows)
    summary = {
        "pixel_cases": len(pixels), "derivative_cases": len(derivatives),
        "negative_controls": len(controls),
        "max_intensity_relative_l2": max(row["relative_l2"] for row in pixels),
        "max_derivative_relative_l2": max(row["relative_l2"] for row in derivatives),
        "max_power_error": max(row["max_power_error"] for row in pixels),
        "min_control_relative_l2": min(row["relative_l2"] for row in controls),
        "all_pass": all(row["pass"] for row in pixels + derivatives)
                    and all(row["detected"] for row in controls),
        "scope": "Matched finite-grid ideal pyramid only; not native HCIPy defaults or instrument validation",
    }
    write_json(path / "summary.json", summary)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
    for axis, rows, key, title in (
        (axes[0], pixels, "intensity_relative_l2", "Detector pixel agreement"),
        (axes[1], derivatives, "derivative_relative_l2", "Independent derivative agreement"),
    ):
        for spectrum in range(len(config["wavelength_sets_m"])):
            selected = [row for row in rows if row["spectrum"] == spectrum]
            axis.scatter([row["n"] for row in selected], [row["relative_l2"] for row in selected],
                         s=18, alpha=.7, label=f"{len(config['wavelength_sets_m'][spectrum])} wavelength(s)")
        axis.axhline(tolerance[key], color="black", linestyle="--", label="Predeclared tolerance")
        axis.set(title=title, xlabel="Pupil grid per side", ylabel="Relative L2 error", yscale="log")
        axis.legend(fontsize=7)
    fig.savefig(path / "agreement.png", dpi=150)
    plt.close(fig)
    (path / "report.md").write_text(
        "# Independent ideal-pyramid comparison\n\n"
        f"All predeclared gates passed: **{summary['all_pass']}**. "
        f"{len(pixels)} pixel cases, {len(derivatives)} derivative cases, and "
        f"{len(controls)} deliberate error controls.\n\n"
        f"Maximum intensity relative L2 error: {summary['max_intensity_relative_l2']:.9g}. "
        f"Maximum derivative relative L2 error: {summary['max_derivative_relative_l2']:.9g}. "
        f"Maximum absolute power error: {summary['max_power_error']:.9g}. "
        f"Smallest deliberate error: {summary['min_control_relative_l2']:.9g}.\n\n"
        "![Agreement against predeclared tolerances](agreement.png)\n\n"
        "See [pixel cases](pixels.md), [derivatives](derivatives.md), "
        "[negative controls](controls.md), [configuration](config.json), "
        "[provenance](provenance.json), and retained arrays in `raw/`.\n\n"
        "The independent reference uses HCIPy matrix integrals and a physical thin surface. "
        "The production sensor and all M0–M7 baseline data are unchanged. Agreement covers "
        "the matched finite-grid ideal contract; it does not establish convergence to an "
        "unbounded focal plane, agreement with native HCIPy defaults, on-sky behavior, "
        "controller fairness or safe fault recovery.\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/pyramid_validation.yaml"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    if not args.render_only:
        run(load_config(args.config), args.out)
    summary = rebuild(args.out)
    print(json.dumps(summary, indent=2))
    if not summary["all_pass"]:
        raise SystemExit("Independent pyramid validation failed; see retained evidence")


if __name__ == "__main__":
    main()
