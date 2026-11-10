"""Recompute saved controller metrics from phase and applied mirror coefficients.

This NumPy-only audit does not call the simulator or validate its sensor model.
It checks the retained finite-grid physical accounting, not hardware fidelity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ATOL = 5e-12
RTOL = 5e-12


def physical_metrics(phase, applied, basis, mask, config, fault):
    """Spatial phase variance and coherent pupil peak, after removing piston."""
    frames, n, modes = config["frames"], config["n"], config["modes"]
    shapes = ((frames, n, n), (frames, modes), (modes, n, n), (n, n))
    values = tuple(np.asarray(x, dtype=np.float64) for x in (phase, applied, basis, mask))
    if any(x.shape != shape or not np.isfinite(x).all()
           for x, shape in zip(values, shapes, strict=True)):
        raise ValueError("Physical arrays have incompatible shape or nonfinite values")
    phase, applied, basis, mask = values
    if not np.isin(mask, (0, 1)).all() or mask.sum() == 0:
        raise ValueError("Expected a nonempty binary pupil")
    if fault not in ("none", "optical_gain", "vibration"):
        raise ValueError("Unsupported physical disturbance")
    physical_phase = phase.copy()
    if fault == "vibration":
        f = config["faults"]
        t = np.arange(frames)
        amplitude = f["vibration_rad"] * np.sin(2*np.pi*f["vibration_hz"]*t*config["dt_s"])
        amplitude *= (t >= f["onset"]) & (t < f["end"])
        physical_phase += amplitude[:, None, None] * basis[0]
    mirror = (applied @ basis.reshape(modes, -1)).reshape(frames, n, n)
    residual = (physical_phase - mirror) * mask
    residual -= (residual.sum(axis=(1, 2)) / mask.sum())[:, None, None] * mask
    return {
        "variance": np.sum(residual**2, axis=(1, 2)) / mask.sum(),
        "strehl": np.abs(np.sum(mask*np.exp(1j*residual), axis=(1, 2))/mask.sum())**2,
    }


def check_queue(command, applied, delay, clip):
    """Check issued-command bounds and the actual delayed command history."""
    if (type(delay) is not int or delay < 1 or not np.isfinite(clip) or clip <= 0
            or command.ndim != 2 or applied.shape != command.shape
            or not np.isfinite(command).all() or not np.isfinite(applied).all()):
        raise ValueError("Invalid command history")
    expected = np.zeros_like(command)
    if delay < len(command):
        expected[delay:] = command[:-delay]
    if not np.array_equal(applied, expected):
        raise ValueError("Applied commands violate the declared delay")
    if np.any(np.abs(command) > clip):
        raise ValueError("Issued commands exceed the declared modal bound")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def audit(run, report_path):
    run = Path(run).resolve()
    report_path = Path(report_path).resolve()
    report_sha256 = digest(report_path)
    report = json.loads(report_path.read_bytes())
    if report["status"] != "complete_descriptive_report":
        raise ValueError("A completed raw-verified study report is required")
    for name in ("manifest", "training", "selection"):
        if digest(run / (name + ".json")) != report[name + "_sha256"]:
            raise ValueError("Study identity differs from the completed report")
    verified = {}

    def bound(ref):
        path = (run / ref["path"]).resolve()
        if not path.is_relative_to(run) or path == run:
            raise ValueError("Artifact path escapes study")
        key = (ref["path"], ref["sha256"])
        if key not in verified:
            if digest(path) != ref["sha256"]:
                raise ValueError("Artifact digest mismatch")
            verified[key] = path
        return path

    training = json.loads((run / "training.json").read_bytes())
    calibration_ref = next(x for x in training["evidence"]
                           if x["path"] == "evidence/training/learned-00/calibration.npz")
    with np.load(bound(calibration_ref), allow_pickle=False) as archive:
        basis, mask = archive["basis"], archive["mask"]
    errors = {key: 0.0 for key in ("variance", "strehl", "open_variance", "open_strehl")}
    rows = []
    for stage in ("validation", "evaluation"):
        files = sorted((run / stage).glob("*.json"))
        if len(files) != report[stage + "_records"]:
            raise ValueError("Terminal cohort differs from the completed report")
        expected_cases = {json.dumps(x["case"], sort_keys=True): x
                          for x in report[stage + "_trials"]}
        if len(expected_cases) != len(files):
            raise ValueError("Repeated or incomplete report cases")
        seen = set()
        for terminal in files:
            row = json.loads(terminal.read_bytes())["record"]
            identity = json.dumps(row["case"], sort_keys=True)
            if identity in seen or identity not in expected_cases:
                raise ValueError("Repeated or unexpected terminal identity")
            seen.add(identity)
            if row["status"] != "completed" or expected_cases[identity]["status"] != "completed":
                raise ValueError("Physical audit requires complete finite trial evidence")
            detail_ref = next(x for x in row["artifacts"]
                              if x["path"].startswith(f"evidence/{stage}/trials/")
                              and x["path"].endswith(".json"))
            detail = json.loads(bound(detail_ref).read_bytes())
            if detail["case"] != row["case"]:
                raise ValueError("Trial identity mismatch")
            cfg = detail["config"]
            with np.load(bound(detail["atmosphere"][0]), allow_pickle=False) as archive:
                phase = archive["phase"]
            with np.load(bound(detail["raw"]), allow_pickle=False) as archive:
                command, applied = archive["command"], archive["applied"]
                saved = {key: archive[key] for key in ("variance", "strehl")}
            check_queue(command, applied, cfg["delay_frames"], cfg["command_clip_rad"])
            actual = physical_metrics(phase, applied, basis, mask, cfg, row["case"]["fault"])
            with np.load(bound(detail["open_loop"][0]), allow_pickle=False) as archive:
                saved.update({"open_" + key: archive[key] for key in ("variance", "strehl")})
            actual.update({"open_" + key: value for key, value in physical_metrics(
                phase, np.zeros_like(applied), basis, mask, cfg, row["case"]["fault"]).items()})
            for key, value in actual.items():
                if saved[key].shape != value.shape or not np.allclose(
                        saved[key], value, atol=ATOL, rtol=RTOL, equal_nan=False):
                    raise ValueError(f"Physical metric mismatch: {terminal.name}: {key}")
                errors[key] = max(errors[key], float(np.max(np.abs(saved[key] - value))))
            rows.append({"stage": stage, "terminal": terminal.name, "sha256": digest(terminal)})
    # A second digest pass detects input mutation during the audit.
    for (_, expected), path in verified.items():
        if digest(path) != expected:
            raise ValueError("Artifact changed during audit")
    for row in rows:
        if digest(run / row["stage"] / row["terminal"]) != row["sha256"]:
            raise ValueError("Terminal record changed during audit")
    if digest(report_path) != report_sha256 or any(
            digest(run / (name + ".json")) != report[name + "_sha256"]
            for name in ("manifest", "training", "selection")):
        raise ValueError("Report or frozen study identity changed during audit")
    return {
        "schema_version": 1, "status": "physical_accounting_verified",
        "report_sha256": report_sha256, "calibration": calibration_ref,
        "tolerance": {"absolute": ATOL, "relative": RTOL}, "max_absolute_error": errors,
        "validation_records": report["validation_records"],
        "evaluation_records": report["evaluation_records"], "terminal_records": rows,
        "audit_source_sha256": digest(Path(__file__)), "numpy_version": np.__version__,
        "scope": "Finite-grid residual variance, coherent pupil Strehl, zero-DM reference, command delay and modal clipping.",
        "limitations": ["Uses the acquisition's saved basis, pupil and phase screens.",
                        "Does not reproduce the sensor, controller, atmospheric statistics or hardware.",
                        "Numerical tolerances were fixed before this audit, after the main report was observed."],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.resolve().is_relative_to(args.run.resolve()) or args.out.exists():
        raise ValueError("Use a fresh output outside the retained run")
    result = audit(args.run, args.report)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: value for key, value in result.items() if key != "terminal_records"}))


if __name__ == "__main__":
    main()
