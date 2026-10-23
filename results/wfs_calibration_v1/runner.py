"""Acquire or rebuild independent pyramid-sensor calibration comparisons."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

from noether_optics.wfs_calibration import (
    ENGINES, acquire, backend, make_inputs, metrics, source_metadata, transfer,
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return dict(archive)


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def validate(config):
    for n in config["grids"]:
        if not isinstance(n, int) or n < 16 or n % 2 or (4 * n) % config["detector_n"]:
            raise ValueError("Use even pupil grids >=16 and an exactly divisible detector")
    if len(set(config["grids"])) != len(config["grids"]):
        raise ValueError("Duplicate pupil grids")
    if len(set(config["modes"])) != len(config["modes"]):
        raise ValueError("Duplicate calibration modes")
    if config["heldout_seed"] == config["noise_seed"]:
        raise ValueError("Held-out and noise streams must be separate")
    for name in ("wavelength_m", "separation_diameters", "poke_rad", "photons"):
        if not np.isfinite(config[name]) or config[name] <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if config["noise_count"] < 2 or config["heldout_count"] < 1:
        raise ValueError("Insufficient samples")
    if not np.isfinite(config["read_noise_e"]) or config["read_noise_e"] < 0:
        raise ValueError("Read noise must be finite and nonnegative")
    if not np.isfinite(config["pinv_rcond"]) or not 0 < config["pinv_rcond"] < 1:
        raise ValueError("SVD cutoff must lie between zero and one")
    if len(set(config["phase_rms_rad"])) != len(config["phase_rms_rad"]):
        raise ValueError("Duplicate phase amplitudes")
    if not config["phase_rms_rad"] or not all(np.isfinite(x) and x > 0 for x in config["phase_rms_rad"]):
        raise ValueError("Phase amplitudes must be positive and finite")


def worker(path, engine, n):
    config = json.loads((path / "config.json").read_text())
    inputs = arrays(path / "raw" / f"inputs_n{n}.npz")
    sensor, geometry = backend(engine, config, n, inputs["mask"])
    raw = acquire(sensor, inputs, config, autodiff=engine == "noether")
    # Exclusive creation prevents a repeated worker from replacing raw evidence.
    with (path / "raw" / f"{engine}_n{n}.npz").open("xb") as handle:
        np.savez_compressed(handle, **raw)
    dump(path / "raw" / f"{engine}_n{n}.json",
         {"geometry": geometry, "runtime": source_metadata(engine)})


def run(config, path, specula_python):
    from noether_optics.artifacts import start_run
    validate(config)
    if shutil.disk_usage(path.parent).free < 512 * 1024**2:
        raise RuntimeError("Requires 512 MiB free reserve before bounded calibration acquisition")
    start_run(config, path)
    (path / "raw").mkdir()
    runner = Path(__file__).resolve()
    for n in config["grids"]:
        np.savez_compressed(path / "raw" / f"inputs_n{n}.npz", **make_inputs(config, n))
        for engine in ENGINES:
            python = specula_python if engine == "specula_native" else sys.executable
            subprocess.run([str(python), str(runner), "--out", str(path),
                            "--worker", engine, "--n", str(n)], check=True)
            print(f"Retained {engine}, pupil {n}", flush=True)
    # Save the exact runner too; the existing run helper hashes all source modules.
    (path / "runner.py").write_bytes(runner.read_bytes())
    files = [path / "config.json", path / "provenance.json", path / "machine.json",
             path / "runner.py", *sorted((path / "raw").iterdir())]
    dump(path / "manifest.json", {str(p.relative_to(path)): sha(p) for p in files})


def rebuild(path):
    """Verify a complete inventory before deriving any report from saved arrays."""
    manifest = json.loads((path / "manifest.json").read_text())
    config = json.loads((path / "config.json").read_text())
    validate(config)
    expected = {"config.json", "provenance.json", "machine.json", "runner.py"}
    for n in config["grids"]:
        expected.add(f"raw/inputs_n{n}.npz")
        for engine in ENGINES:
            expected.update({f"raw/{engine}_n{n}.npz", f"raw/{engine}_n{n}.json"})
    if set(manifest) != expected:
        raise ValueError("Incomplete or unexpected calibration artifact inventory")
    for name, digest in manifest.items():
        if sha(path / name) != digest:
            raise ValueError(f"Calibration evidence hash mismatch: {name}")
    local, cross = [], []
    for n in config["grids"]:
        inputs = arrays(path / "raw" / f"inputs_n{n}.npz")
        raw = {engine: arrays(path / "raw" / f"{engine}_n{n}.npz") for engine in ENGINES}
        for engine in ENGINES:
            local.append({"n": n, "engine": engine, **metrics(raw[engine], inputs, config)})
        for source in ENGINES:
            for target in ENGINES:
                if source != target:
                    cross.append({"n": n, "source": source, "target": target,
                                  **transfer(raw[source], raw[target], inputs, config)})
    matched = [row for row in cross if {row["source"], row["target"]} == {"noether", "hcipy_matched"}]
    count = config["heldout_count"]
    t = config["tolerances"]
    # Match the small-amplitude block by value, not by its position in the file.
    small_index = config["phase_rms_rad"].index(min(config["phase_rms_rad"]))
    start, end = small_index * count, (small_index + 1) * count
    matched_pass = all(
        row["reference_relative_l2"] <= t["matched_reference_relative_l2"]
        and row["interaction_relative_l2"] <= t["matched_interaction_relative_l2"]
        and max(row["matrix_and_reference"][start:end]) <= t["small_signal_phase_relative_rms"]
        for row in matched)
    summary = {"all_required_gates_pass": all(r["all_pass"] for r in local) and matched_pass,
               "matched_transfer_pass": matched_pass, "local_calibrations": local,
               "cross_calibrations": cross,
               "native_equivalence_claimed": False}
    dump(path / "summary.json", summary)
    plot(path, local, cross, config, start, end)
    lines = ["# Independent WFS calibration comparison", "",
             f"All required gates passed: **{summary['all_required_gates_pass']}**.", "",
             "Four independently propagated/calibrated optical paths; "
             "native-model transfer is diagnostic, not an equivalence claim. "
             "See the [frozen protocol](../../docs/wfs_calibration.md).", "",
             "![Calibration accuracy and transfer](comparison.png)", "",
             f"| Pupil | Backend | Power | Condition | Max error at {min(config['phase_rms_rad']):g} rad | Gates |",
             "| --- | --- | --- | --- | --- | --- |"]
    for row in local:
        condition = row['condition_number']
        lines.append(f"| {row['n']} | {row['engine']} | {row['reference_power']:.8g} | "
                     f"{condition if condition is None else f'{condition:.6g}'} | "
                     f"{max(row['phase_relative_rms'][start:end]):.6g} | {row['all_pass']} |")
    lines.extend(["", "Errors are relative pupil RMS, not percentages. "
                  "All per-case errors, singular values, gate verdicts and noise covariances "
                  "are retained in [summary.json](summary.json).", "",
                  "## Calibration transfer", "",
                  "Each row applies the source reconstructor to target images. Both subtracting "
                  "the target zero image (matrix only) and using the source zero image (full) are shown. "
                  "No fitted alignment or scaling is applied.", "",
                  "| Pupil | Source → target | Interaction relative L2 | Small-signal matrix-only error | Small-signal full error |",
                  "| --- | --- | --- | --- | --- |"])
    for row in cross:
        lines.append(f"| {row['n']} | {row['source']} → {row['target']} | "
                     f"{row['interaction_relative_l2']:.6g} | "
                     f"{max(row['matrix_only'][start:end]):.6g} | "
                     f"{max(row['matrix_and_reference'][start:end]):.6g} |")
    lines.extend(["", "## Nonlinearity and noise", "",
                  "| Pupil | Backend | Phase RMS (rad) | Max relative reconstruction error |",
                  "| --- | --- | --- | --- |"])
    for row in local:
        for index, rms in enumerate(config["phase_rms_rad"]):
            error = max(row["phase_relative_rms"][index * count:(index + 1) * count])
            lines.append(f"| {row['n']} | {row['engine']} | {rms:g} | {error:.6g} |")
    lines.extend(["", "Noise ratios compare sampled to analytic modal variances at the flat "
                  f"wavefront ({config['noise_count']} independent detector realizations, "
                  f"{config['photons']:g} incident photons, {config['read_noise_e']:g} electron read noise). "
                  "Finite-sample deviations from one are diagnostic; this is not instrument noise.", "",
                  "| Pupil | Backend | Minimum variance ratio | Maximum variance ratio |",
                  "| --- | --- | --- | --- |"])
    for row in local:
        ratio = row["noise_variance_ratio"]
        lines.append(f"| {row['n']} | {row['engine']} | {min(ratio):.6g} | {max(ratio):.6g} |")
    lines.extend(["", "## Scope", "",
                  "HCIPy matched uses matrix Fourier integrals and a physical thin surface; "
                  "Noether uses dLux/JAX FFTs and automatic derivatives. HCIPy native and "
                  "SPECULA native retain their focal stops, centering and propagation conventions. "
                  "Native disagreement is not evidence of a library bug. No controller, "
                  "on-sky, modulation, runtime or hardware-validity claim follows from this run.", "",
                  "[Configuration](config.json) · [Noether provenance](provenance.json) · "
                  "[File hashes](manifest.json). Per-backend geometry, package versions and library "
                  f"source hashes accompany each raw archive in `raw/`. All {count * len(config['phase_rms_rad'])} held-out images per "
                  "backend and grid, both push-pull step sizes and modal noise draws are retained.", ""])
    (path / "report.md").write_text("\n".join(lines))
    return summary


def plot(path, local, cross, config, start, end):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    n = max(config["grids"])
    count = config["heldout_count"]
    for row in local:
        if row["n"] == n:
            values = [max(row["phase_relative_rms"][i * count:(i + 1) * count])
                      for i in range(len(config["phase_rms_rad"]))]
            axes[0].loglog(config["phase_rms_rad"], values, "o-", label=row["engine"])
    axes[0].set(title=f"Own-calibration error (pupil {n})", xlabel="Input phase RMS (rad)",
                ylabel="Maximum relative phase RMS error")
    axes[0].legend(fontsize=8)
    values = np.zeros((len(ENGINES), len(ENGINES)))
    for i, source in enumerate(ENGINES):
        for j, target in enumerate(ENGINES):
            if source == target:
                row = next(r for r in local if r["n"] == n and r["engine"] == source)
                values[i, j] = max(row["phase_relative_rms"][start:end])
            else:
                row = next(r for r in cross if r["n"] == n and r["source"] == source and r["target"] == target)
                values[i, j] = max(row["matrix_only"][start:end])
    heat = axes[1].imshow(values, norm=LogNorm(), cmap="viridis")
    for (i, j), value in np.ndenumerate(values):
        color = "black" if heat.norm(value) > .6 else "white"
        axes[1].text(j, i, f"{value:.2g}", ha="center", va="center", color=color, fontsize=9)
    labels = ["Noether", "HCIPy matched", "HCIPy native", "SPECULA native"]
    axes[1].set(xticks=range(4), yticks=range(4), xticklabels=labels, yticklabels=labels,
                xlabel="Target measurements", ylabel="Source calibration",
                title=f"Small-signal transfer; target zero subtracted (pupil {n})")
    axes[1].tick_params(axis="x", rotation=30)
    fig.colorbar(heat, ax=axes[1], label="Maximum relative phase RMS error")
    fig.savefig(path / "comparison.png", dpi=160, metadata={"Software": "Noether AO"})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/wfs_calibration.yaml"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--specula-python", type=Path)
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--worker", choices=ENGINES, help=argparse.SUPPRESS)
    parser.add_argument("--n", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    path = args.out.resolve()
    if args.worker:
        worker(path, args.worker, args.n)
        return
    if not args.render_only:
        from noether_optics.artifacts import load_config
        if args.specula_python is None:
            parser.error("--specula-python is required for acquisition; no backend is silently skipped")
        # Resolving a venv's Python symlink bypasses its pyvenv.cfg and packages.
        run(load_config(args.config), path, args.specula_python.absolute())
    summary = rebuild(path)
    print(json.dumps({k: v for k, v in summary.items() if not isinstance(v, list)}, indent=2))
    if not summary["all_required_gates_pass"]:
        raise SystemExit("Calibration gates failed; all outcomes retained")


if __name__ == "__main__":
    main()
