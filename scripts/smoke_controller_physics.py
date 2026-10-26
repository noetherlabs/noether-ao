"""Run/rebuild a small physical wiring check; never run reserved study cohorts."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import numpy as np
import yaml

from noether_optics.artifacts import load_config, start_run, write_json
from noether_optics.atmosphere import phase_sequence
from noether_optics.controller_physics import (controller_kernel, episode_config, fit_measured_predictor,
    measured_telemetry, paired_metrics, paired_open_loop, training_cases, STUDY_FAULTS)
from noether_optics.controller_protocol import load_study, study_plan, validate_study
from noether_optics.simulation import calibration, initialize_weights, CONTROLLERS

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def configuration(smoke):
    if smoke["kind"] != "controller_physics_smoke":
        raise ValueError("This command only accepts a labeled component smoke configuration")
    reserved = load_study(ROOT / "configs/controller_study_v1.yaml")
    study = deepcopy(reserved)
    study["study_id"] = "controller_physics_smoke"
    study["base_config"] = "derived-smoke-base.yaml"
    base = load_config(ROOT / reserved["base_config"])
    base.update(smoke["base_overrides"])
    base_text = yaml.safe_dump(base, sort_keys=True)
    study["base_config_sha256"] = hashlib.sha256(base_text.encode()).hexdigest()
    for role in ("training", "validation", "evaluation"):
        study["seeds"][role] = smoke[f"{role}_seeds"]
    study["seeds"]["initialization"] = smoke["initialization_seed"]
    study["training"].update(frames=smoke["frames"], burn_in=smoke["burn_in"])
    study["loop"].update(frames=smoke["frames"], burn_in=smoke["burn_in"],
                         fault_onset=smoke["fault_onset"], fault_end=smoke["fault_end"])
    validate_study(study)
    def streams(config):
        seeds = config["seeds"]
        roots = {seed for role in ("training", "validation", "evaluation") for seed in seeds[role]}
        return roots | {seed + seeds["noise_offset"] for seed in roots} | {seeds["initialization"]}
    if streams(study) & streams(reserved):
        raise ValueError("Smoke streams overlap reserved study streams")
    if smoke["trial_seed"] not in study["seeds"]["validation"]:
        raise ValueError("Smoke trial seed must belong to its separate validation cohort")
    return study, base, base_text


def run(config_path, out):
    if os.environ.get("CI"):
        raise RuntimeError("Run physical smoke artifacts locally, outside CI")
    if not jax.config.jax_enable_x64:
        raise RuntimeError("The smoke contract requires JAX x64")
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip():
        raise RuntimeError("Commit tracked source changes before the physical smoke run")
    if shutil.disk_usage(out.parent.resolve()).free < 512 * 1024**2:
        raise RuntimeError("Require at least 512 MiB free for this bounded smoke run")
    smoke = load_config(config_path)
    study, base, base_text = configuration(smoke)
    path = start_run({"kind": smoke["kind"], "smoke": smoke, "study": study, "base": base}, out)
    (path / "raw").mkdir()
    (path / "derived-smoke-base.yaml").write_text(base_text)
    provenance = json.loads((path / "provenance.json").read_text())
    provenance.update(runner_sha256=digest(Path(__file__)), smoke_config_sha256=digest(config_path),
                      source_study_sha256=digest(ROOT / "configs/controller_study_v1.yaml"))
    write_json(path / "provenance.json", provenance)
    cal = calibration(base)
    np.savez_compressed(path / "raw/calibration.npz",
                        **{key: np.asarray(cal[key]) for key in ("basis", "mask", "reconstructor", "reference")})
    phases, kernels, episodes = {}, {}, []
    index = {"atmospheres": [], "pilots": [], "predictors": [], "zero_dm": [], "trials": []}
    for i, case in enumerate(training_cases(study)):
        cfg = episode_config(base, study, "training", case["photons"], case["wind_scale"])
        key = (case["wind_scale"], case["seed"])
        if key not in phases:
            phases[key] = phase_sequence(cfg, case["seed"])
            filename = f"raw/atmosphere_{len(phases):02d}.npz"
            np.savez_compressed(path / filename, phase=np.asarray(phases[key]))
            index["atmospheres"].append({"wind_scale": case["wind_scale"], "seed": case["seed"], "file": filename})
        kernel_key = (case["photons"], case["wind_scale"])
        if kernel_key not in kernels:
            kernels[kernel_key] = controller_kernel(cfg, cal, "integrator")
        output = kernels[kernel_key](phases[key], case["seed"])
        telemetry = measured_telemetry(output, cal)
        filename = f"raw/pilot_{i:02d}.npz"
        np.savez_compressed(path / filename, **{key: np.asarray(value) for key, value in output.items()},
                            measured_pseudo_open_loop=telemetry)
        index["pilots"].append({"case": case, "file": filename})
        episodes.append({"case": case, "telemetry": telemetry})
    print("Stored all 12 separate smoke pilot episodes.", flush=True)
    candidates = study_plan(study)["candidates"]
    fitted = {}
    for candidate in (row for row in candidates if row["controller"] == "predictive"):
        result = fit_measured_predictor(study, candidate, episodes, base["modes"])
        filename = f"raw/{candidate['id']}.npz"
        np.savez_compressed(path / filename, coefficients=result["coefficients"],
                            episode_coefficients=result["episode_coefficients"])
        fitted[candidate["id"]] = result["coefficients"]
        index["predictors"].append({"candidate": candidate, "file": filename})
    initial = initialize_weights(study["seeds"]["initialization"])
    np.savez_compressed(path / "raw/untrained_network.npz",
                        **{f"parameter_{i}": np.asarray(value) for i, value in enumerate(initial)})
    choices = {name: next(row for row in candidates if row["controller"] == name) for name in CONTROLLERS}
    cfg = episode_config(base, study, "validation", smoke["trial_photons"], smoke["trial_wind_scale"], choices["integrator"])
    phase = phase_sequence(cfg, smoke["trial_seed"])
    np.savez_compressed(path / "raw/trial_atmosphere.npz", phase=np.asarray(phase))
    for fault in STUDY_FAULTS:
        reference = paired_open_loop(phase, cfg, cal, fault)
        zero = controller_kernel(cfg | {"gain": 0.}, cal, "integrator", fault)(phase, smoke["trial_seed"])
        ref_file = f"raw/open_{fault}.npz"
        np.savez_compressed(path / ref_file, **{key: np.asarray(value) for key, value in reference.items()})
        zero_file = f"raw/zero_dm_{fault}.npz"
        np.savez_compressed(path / zero_file, **{key: np.asarray(value) for key, value in zero.items()})
        index["zero_dm"].append({"fault": fault, "open_file": ref_file, "zero_file": zero_file})
        for controller, candidate in choices.items():
            current = episode_config(base, study, "validation", smoke["trial_photons"], smoke["trial_wind_scale"], candidate)
            kernel = controller_kernel(current, cal, controller, fault, fitted.get(candidate["id"]))
            output = kernel(phase, smoke["trial_seed"], initial if controller == "learned" else None)
            filename = f"raw/trial_{controller}_{fault}.npz"
            np.savez_compressed(path / filename, **{key: np.asarray(value) for key, value in output.items()})
            index["trials"].append({"controller": controller, "candidate": candidate, "fault": fault,
                                    "seed": smoke["trial_seed"], "config": current, "file": filename,
                                    "open_file": ref_file, "learned_weights": "untrained smoke initialization" if controller == "learned" else None})
        print(f"Stored five controller paths and zero-DM check: {fault}.", flush=True)
    write_json(path / "index.json", index)
    protected = {str(file.relative_to(path)): digest(file) for file in sorted(path.rglob("*")) if file.is_file()}
    write_json(path / "protected_sha256.json", protected)
    return rebuild(path)


def rebuild(path):
    protected = json.loads((path / "protected_sha256.json").read_text())
    if any(digest(path / name) != value for name, value in protected.items()):
        raise ValueError("Protected smoke evidence changed")
    config = json.loads((path / "config.json").read_text())
    index = json.loads((path / "index.json").read_text())
    cal = arrays(path / "raw/calibration.npz")
    pilot_error, episodes = [], []
    for row in index["pilots"]:
        raw = arrays(path / row["file"])
        telemetry = measured_telemetry(raw, cal)
        pilot_error.append(float(np.max(np.abs(telemetry - raw["estimate"]))))
        pilot_error.append(float(np.max(np.abs(telemetry - raw["measured_pseudo_open_loop"]))))
        episodes.append({"case": row["case"], "telemetry": telemetry})
    predictor_error = []
    for row in index["predictors"]:
        actual = arrays(path / row["file"])
        expected = fit_measured_predictor(config["study"], row["candidate"], episodes, config["base"]["modes"])
        predictor_error.extend(float(np.max(np.abs(actual[key] - expected[key]))) for key in ("coefficients", "episode_coefficients"))
    zero_error = []
    for row in index["zero_dm"]:
        reference, zero = arrays(path / row["open_file"]), arrays(path / row["zero_file"])
        zero_error.extend(float(np.max(np.abs(reference[key] - zero[key]))) for key in ("variance", "strehl"))
    clean = arrays(path / "raw/open_none.npz")
    vibration = arrays(path / "raw/open_vibration.npz")
    omitted_error = float(np.max(np.abs(vibration["variance"] - clean["variance"])))
    rows = []
    for row in index["trials"]:
        raw, reference = arrays(path / row["file"]), arrays(path / row["open_file"])
        rows.append({"controller": row["controller"], "fault": row["fault"],
                     **paired_metrics(raw, reference, row["config"]),
                     "fallback_used": bool(raw["fallback"].any()), "learned_weights": row["learned_weights"]})
    tolerance = config["smoke"]["tolerances"]
    gates = {"pilot_reconstruction": max(pilot_error) <= tolerance["pilot_absolute"],
             "predictor_rebuild": max(predictor_error) <= tolerance["pilot_absolute"],
             "paired_zero_dm": max(zero_error) <= tolerance["zero_dm_absolute"],
             "omitted_vibration_rejected": omitted_error > tolerance["minimum_omitted_vibration_error"],
             "all_trial_outputs_finite": all(row["metrics"]["all_outputs_finite"] for row in rows),
             "fallback_disabled": not any(row["fallback_used"] for row in rows),
             "predictor_candidate_coverage": len(index["predictors"]) == config["study"]["candidates_per_controller"]
                 and {row["candidate"]["id"] for row in index["predictors"]}
                 == {row["id"] for row in study_plan(config["study"])["candidates"] if row["controller"] == "predictive"},
             "zero_dm_fault_coverage": len(index["zero_dm"]) == len(STUDY_FAULTS)
                 and {row["fault"] for row in index["zero_dm"]} == set(STUDY_FAULTS),
             "controller_fault_coverage": len(rows) == 15 and {(row["controller"], row["fault"]) for row in rows}
                 == {(controller, fault) for controller in CONTROLLERS for fault in STUDY_FAULTS}}
    summary = {"kind": "component_smoke", "pilot_episodes": len(episodes), "predictor_candidates": len(index["predictors"]),
               "controller_trials": len(rows), "max_pilot_absolute_error": max(pilot_error),
               "max_predictor_rebuild_error": max(predictor_error), "max_zero_dm_absolute_error": max(zero_error),
               "omitted_vibration_variance_error": omitted_error, "gates": gates, "all_pass": all(gates.values()),
               "scope": "Small numerical wiring check; no full-study outcomes, learned training, fair selection, timing or superiority claim."}
    write_json(path / "trials.json", rows)
    write_json(path / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/controller_physics_smoke.yaml")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--rebuild", type=Path)
    args = parser.parse_args()
    if (args.out is None) == (args.rebuild is None):
        parser.error("Choose exactly one of --out or --rebuild")
    summary = rebuild(args.rebuild) if args.rebuild else run(args.config, args.out)
    print(json.dumps(summary, indent=2, allow_nan=False))
    if not summary["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
