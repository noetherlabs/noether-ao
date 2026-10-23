"""Independently reconstruct the retained controller-filtered IV comparison.

Requires NumPy, PyYAML and a Git checkout containing both recorded commits.
Imports no production optics code and acquires no trajectories. Hashes establish
internal consistency with the recorded commits, not external authenticity.
"""
from __future__ import annotations

import argparse
from collections import deque
from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

import numpy as np
import yaml


ORIGINAL = PurePosixPath("results/identification_component_smoke_v1")
EXECUTION_PATHS = ("src", "scripts", "configs", "pyproject.toml", "uv.lock")
RTOL, ATOL = 1e-9, 1e-12


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")

    return json.loads(path.read_text(), object_pairs_hook=unique, parse_constant=invalid)


def relative_name(name):
    require(isinstance(name, str) and name != "", "Empty or non-string path")
    path = PurePosixPath(name)
    require(not path.is_absolute() and ".." not in path.parts and str(path) == name
            and "\\" not in name, f"Noncanonical relative path: {name}")
    return path


def file_bytes(root, name):
    relative = relative_name(name)
    path = root
    for part in relative.parts:
        path = path / part
        require(not path.is_symlink(), f"Symlinked evidence path: {name}")
    require(path.is_file(), f"Missing evidence file: {name}")
    return path.read_bytes()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root)


def commit_id(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", value),
            "Invalid pinned commit identity")
    return value


def source_inventory(root, inventory):
    require(inventory.get("schema_version") == 1, "Unsupported source inventory")
    commit = commit_id(inventory["commit"])
    entries = git(root, "ls-tree", "-r", "-z", commit, "--", *EXECUTION_PATHS)
    names = set()
    for entry in entries.split(b"\0"):
        if not entry:
            continue
        metadata, raw_name = entry.split(b"\t", 1)
        mode, kind, _ = metadata.split()
        require(kind == b"blob" and mode in (b"100644", b"100755"),
                "Nonregular execution input in the recorded source")
        names.add(raw_name.decode())
    require(names == set(inventory["files"]) and {"pyproject.toml", "uv.lock"} <= names,
            "Incomplete source inventory")
    for name, expected in inventory["files"].items():
        relative_name(name)
        data = git(root, "show", f"{commit}:{name}")
        require(digest(data) == expected, f"Source hash mismatch: {name}")
    return commit


class Comparison:
    """Compare shapes and finite values without NumPy's broadcasting shortcuts."""

    def __init__(self):
        self.maximum_absolute_error = 0.0

    def array(self, actual, expected, label, *, exact=False):
        actual, expected = np.asarray(actual), np.asarray(expected)
        require(actual.shape == expected.shape, f"Shape mismatch: {label}")
        require(actual.dtype.kind in "biuf" and expected.dtype.kind in "biuf",
                f"Nonnumeric value: {label}")
        require(np.isfinite(actual).all() and np.isfinite(expected).all(),
                f"Nonfinite value: {label}")
        if exact:
            np.testing.assert_array_equal(actual, expected, err_msg=label)
        else:
            np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL,
                                       equal_nan=False, err_msg=label)
            if actual.size:
                self.maximum_absolute_error = max(self.maximum_absolute_error,
                    float(np.max(np.abs(actual.astype(float)-expected.astype(float)))))

    def tree(self, actual, expected, label):
        if isinstance(expected, dict):
            require(isinstance(actual, dict) and set(actual) == set(expected),
                    f"Field mismatch: {label}")
            for key in expected:
                self.tree(actual[key], expected[key], f"{label}.{key}")
        elif isinstance(expected, (str, bool)) or expected is None:
            require(type(actual) is type(expected) and actual == expected,
                    f"Value mismatch: {label}")
        elif isinstance(expected, list) and expected and isinstance(expected[0], dict):
            require(isinstance(actual, list) and len(actual) == len(expected),
                    f"Collection mismatch: {label}")
            for index, (left, right) in enumerate(zip(actual, expected)):
                self.tree(left, right, f"{label}[{index}]")
        elif isinstance(expected, (int, np.integer)):
            require(type(actual) is int and actual == expected, f"Integer mismatch: {label}")
        else:
            self.array(actual, expected, label)


def artifact_inventory(directory, entries):
    artifacts = {}
    for item in entries:
        name = str(relative_name(item["file"]))
        require(PurePosixPath(name).name == name and name not in artifacts,
                "Repeated or nested derived artifact")
        data = file_bytes(directory, name)
        require(type(item["bytes"]) is int and len(data) == item["bytes"]
                and digest(data) == item["sha256"], f"Derived artifact hash mismatch: {name}")
        artifacts[name] = item
    return artifacts


def archive(path, names):
    with np.load(path, allow_pickle=False) as saved:
        require(set(saved.files) == set(names), f"Archive fields differ: {path.name}")
        arrays = {name: saved[name] for name in names}
    for name, value in arrays.items():
        require(value.dtype.kind in "biuf" and np.isfinite(value).all(),
                f"Invalid numeric archive field: {path.name}:{name}")
    return arrays


def replay(probe, response, base):
    """Independent FIFO implementation of the nominal, probe-only recurrence."""
    frames, modes = probe.shape
    queue = deque(np.zeros(modes) for _ in range(base["delay_frames"]))
    previous = np.zeros(modes)
    applied, feedback, issued = (np.empty_like(probe) for _ in range(3))
    saturated = np.empty(frames, dtype=bool)
    for index, request in enumerate(probe):
        applied[index] = queue.popleft()
        residual = -applied[index] @ response
        feedback[index] = base["leak"]*previous + base["gain"]*residual
        combined = feedback[index] + request
        require(np.isfinite(combined).all(), "Nonfinite independent reference recurrence")
        issued[index] = np.clip(combined, -base["command_clip_rad"], base["command_clip_rad"])
        saturated[index] = np.any(issued[index] != combined)
        previous = issued[index].copy()
        queue.append(previous.copy())
    return {"instrument": applied, "command": issued,
            "feedback_requested": feedback, "saturated": saturated}


def spectrum(matrix, rcond):
    singular = np.linalg.svd(matrix, compute_uv=False)
    rank = int(np.count_nonzero(singular > rcond*singular[0])) if singular[0] > 0 else 0
    return singular, rank


def response_fit(row, method, spec):
    """Full-column-rank QR diagnostics and independent two-stage least squares."""
    x, y, z = (row[key] for key in ("applied", "residual", "instrument"))
    modes, rcond = x.shape[1], spec["rcond"]
    if method == "nominal":
        return {"status": "ready", "response": np.eye(modes)}
    xm, ym = x.mean(0), y.mean(0)
    xc, yc = x-xm, y-ym
    sx, rx = spectrum(xc, rcond)
    diagnostics = {"frames": len(x), "modes": modes, "command_rank": rx,
                   "command_singular_values": sx, "minimum_relevance": spec["minimum_relevance"],
                   "rcond": rcond}
    result = {"method": method, "status": "withheld", "reason": None,
              "response": None, "intercept": None, "diagnostics": diagnostics}
    if rx < modes:
        result["reason"] = "rank_deficient_commands"
        return result
    design = xc
    if method == "iv":
        zc = z-z.mean(0)
        sz, rz = spectrum(zc, rcond)
        diagnostics.update(instrument_rank=rz, instrument_singular_values=sz)
        if rz < modes:
            result["reason"] = "rank_deficient_instruments"
            return result
        qz = np.linalg.qr(zc, mode="reduced")[0]
        qx = np.linalg.qr(xc, mode="reduced")[0]
        canonical = np.linalg.svd(qz.T @ qx, compute_uv=False)
        diagnostics["canonical_correlations"] = canonical
        if canonical[-1] < spec["minimum_relevance"]:
            result["reason"] = "weak_instruments"
            return result
        # Predict the actual commands from the instruments, then regress the
        # observations on these predicted commands. No n-by-n projector.
        design = zc @ np.linalg.lstsq(zc, xc, rcond=rcond)[0]
    response, _, rank, singular = np.linalg.lstsq(design, -yc, rcond=rcond)
    diagnostics.update(projected_command_rank=int(rank), fit_singular_values=singular)
    if rank < modes:
        result["reason"] = "rank_deficient_projected_commands"
        return result
    intercept = ym + xm @ response
    if not np.isfinite(response).all() or not np.isfinite(intercept).all():
        result["reason"] = "nonfinite_fit"
        return result
    result.update(status="ready", reason=None, response=response, intercept=intercept)
    return result


def pipeline(rows, method, spec):
    fits = [response_fit(row, method, spec) for row in rows]
    result = {"method": method, "status": "withheld", "reason": None, "response": None,
              "coefficients": None, "episode_fits": fits, "episode_coefficients": None,
              "horizon": 1, "ridge": spec["predictor_ridge"], "rcond": spec["rcond"],
              "minimum_relevance": spec["minimum_relevance"]}
    if any(fit["status"] != "ready" for fit in fits):
        result["reason"] = "withheld_training_response"
        return result
    with np.errstate(over="ignore", invalid="ignore"):
        response = sum(fit["response"]/len(fits) for fit in fits)
    if not np.isfinite(response).all():
        result["reason"] = "nonfinite_mean_response"
        return result
    singular, rank = spectrum(response, spec["rcond"])
    individual = max(np.linalg.svd(fit["response"], compute_uv=False)[0] for fit in fits)
    result.update(response_singular_values=singular, individual_response_scale=float(individual))
    if rank != response.shape[0] or singular[-1] <= spec["rcond"]*individual:
        result["reason"] = "singular_or_canceling_response"
        return result
    coefficients = []
    for row in rows:
        q = np.linalg.solve(response.T, row["residual"].T).T + row["applied"]
        if not np.isfinite(q).all():
            result["reason"] = "nonfinite_reconstruction"
            return result
        per_mode = []
        for mode in range(q.shape[1]):
            design = np.column_stack((q[1:-1, mode], q[:-2, mode]))
            augmented = np.vstack((design, np.sqrt(spec["predictor_ridge"])*np.eye(2)))
            target = np.concatenate((q[2:, mode], np.zeros(2)))
            per_mode.append(np.linalg.lstsq(augmented, target, rcond=None)[0])
        if not np.isfinite(per_mode).all():
            result["reason"] = "nonfinite_predictor"
            return result
        coefficients.append(np.asarray(per_mode))
    with np.errstate(over="ignore", invalid="ignore"):
        mean = sum(value/len(coefficients) for value in coefficients)
    if not np.isfinite(mean).all():
        result["reason"] = "nonfinite_mean_predictor"
        return result
    result.update(status="ready", response=response, coefficients=mean,
                  episode_coefficients=np.asarray(coefficients))
    return result


def prediction(model, raw, scale, delay):
    y = raw["measurement"]*scale
    indices = np.arange(max(2, delay), len(y))
    response, coefficient = model["response"], model["coefficients"]
    predicted = np.empty((len(indices), y.shape[1]))
    for row, target in enumerate(indices):
        current = np.linalg.solve(response.T, y[target-1]) + raw["applied"][target-1]
        previous = np.linalg.solve(response.T, y[target-2]) + raw["applied"][target-2]
        disturbance = coefficient[:, 0]*current + coefficient[:, 1]*previous
        predicted[row] = (disturbance-raw["command"][target-delay]) @ response
    return {"target_indices": indices, "prediction": predicted, "error": predicted-y[indices]}


def verify(root, directory):
    compare = Comparison()
    manifest = read_json(directory/"manifest.json")
    analysis = source_inventory(root, manifest["analysis_source"])
    require("scripts/compare_reference_iv.py" in manifest["analysis_source"]["files"],
            "Analysis source does not include the comparison runner")
    require(manifest["jax_backend"] == "cpu" and manifest["jax_x64"] is True
            and type(manifest["new_trajectories"]) is int and manifest["new_trajectories"] == 0,
            "Unexpected execution scope or numeric runtime")
    original_dir = root/ORIGINAL
    original = read_json(original_dir/"manifest.json")
    acquisition = source_inventory(root, original["source_inventory"])
    require(acquisition == original["source_commit"] == manifest["acquisition_source_commit"],
            "Acquisition source identity mismatch")
    input_names = {str(ORIGINAL/"manifest.json"), str(ORIGINAL/"verify.py")}
    input_names.update(str(ORIGINAL/relative_name(item["file"])) for item in original["artifacts"])
    require(len(input_names) == 37 and set(manifest["input_sha256"]) == input_names,
            "Incomplete original input binding")
    for name, expected in manifest["input_sha256"].items():
        data = file_bytes(root, name)
        require(digest(data) == expected and data == git(root, "show", f"{analysis}:{name}"),
                f"Original input differs from the analysis commit: {name}")
    # Explicitly bind the embedded configurations before using any dimensions.
    spec_path = "configs/identification_smoke.yaml"
    source_spec = git(root, "show", f"{acquisition}:{spec_path}")
    spec = yaml.safe_load(source_spec)
    base_path = str(relative_name(spec["base"]))
    source_base = git(root, "show", f"{acquisition}:{base_path}")
    require(original["input_hashes"] == {spec_path: digest(source_spec), base_path: digest(source_base)},
            "Acquisition input hash mismatch")
    base = yaml.safe_load(source_base)
    base.update(spec["overrides"])
    for key in ("seed", "seeds", "training_seeds"):
        base.pop(key, None)
    base["faults"].update(onset=base["frames"], end=base["frames"])
    require(original["config"] == spec and original["base"] == base,
            "Embedded configuration differs from committed acquisition inputs")
    require(base["sensor"] == "linear" and base["modes"] == 4,
            "This verifier is specific to the retained full linear experiment")
    # Run only after verifying these exact Python bytes against the analysis
    # commit. This original NumPy verifier checks all old fits, arrays and costs.
    completed = subprocess.run([sys.executable, str(original_dir/"verify.py")], cwd=root,
                               check=True, text=True, capture_output=True)
    receipt = json.loads(completed.stdout)
    require(receipt == {"verified_artifacts": 35, "reconstructed_models": 6,
                        "reconstructed_forecasts": 18, "source_commit": acquisition},
            "Original reconstruction receipt differs")
    artifacts = artifact_inventory(directory, manifest["artifacts"])
    require("evidence.json" in artifacts, "Unbound derived evidence")
    old = read_json(original_dir/"evidence.json")
    evidence = read_json(directory/"evidence.json")
    require(set(evidence) == {"scope", "nominal_response", "models", "forecasts",
                              "baseline_models", "reference_response"}, "Unexpected evidence fields")
    require(evidence["scope"] == "exploratory reanalysis of existing development recordings",
            "Unexpected reanalysis scope")
    require(evidence["baseline_models"] == old["models"]
            and evidence["reference_response"] == old["reference_response"],
            "Retained baseline models or evaluator reference changed")
    frames, modes, start, delay = (base[k] for k in ("frames", "modes", "burn_in", "delay_frames"))
    scale = 1/(1+base["reconstructor_ridge"])
    response0 = scale*np.eye(modes)
    compare.array(evidence["nominal_response"], response0, "nominal calibration")
    original_files = {item["file"] for item in original["artifacts"]}

    def raw(seed, rho, drift=False):
        name = f"{seed}-rho{rho:g}-drift{int(drift)}-probe1.npz"
        require(name in original_files, f"Unbound original trajectory: {name}")
        keys = ("measurement", "command", "applied", "probe_requested")
        with np.load(original_dir/name, allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in keys}
        for key, value in arrays.items():
            require(value.dtype == np.dtype("float64") and value.shape == (frames, modes)
                    and np.isfinite(value).all(), f"Invalid retained modal array: {name}:{key}")
        compare.array(arrays["applied"][:delay], np.zeros((delay, modes)), name+":initial FIFO", exact=True)
        compare.array(arrays["applied"][delay:], arrays["command"][:-delay], name+":FIFO", exact=True)
        return arrays

    require(len(evidence["models"]) == len(spec["correlations"]), "Missing or duplicate models")
    new_models = {row["rho"]: row for row in evidence["models"]}
    require(set(new_models) == set(spec["correlations"]), "Model correlation identities differ")
    expected_artifacts = {"evidence.json"}
    reconstructed = {}
    baseline_count = 0
    for rho in spec["correlations"]:
        rows, raw_rows = [], []
        for seed in spec["training_roots"]:
            recording = raw(seed, rho)
            reference = replay(recording["probe_requested"], response0, base)
            name = f"reference-{seed}-rho{rho:g}.npz"
            expected_artifacts.add(name)
            require(name in artifacts, f"Missing reference artifact: {name}")
            saved = archive(directory/name, reference)
            require(saved["saturated"].dtype.kind == "b", "Reference saturation is not boolean")
            for field in reference:
                compare.array(saved[field], reference[field], name+":"+field, exact=field == "saturated")
            shifted = np.zeros_like(recording["probe_requested"])
            shifted[delay:] = recording["probe_requested"][:-delay]
            common = {"residual": recording["measurement"][start:]*scale,
                      "applied": recording["applied"][start:]}
            rows.append(dict(common, instrument=reference["instrument"][start:]))
            raw_rows.append(dict(common, instrument=shifted[start:]))
        originals = [row["model"] for row in old["models"] if row["rho"] == rho]
        require(len(originals) == 3 and {model["method"] for model in originals} == {"nominal", "ols", "iv"},
                "Baseline model identities differ")
        for baseline in originals:
            expected = pipeline(raw_rows, baseline["method"], spec)
            compare.tree(baseline, expected, f"baseline {rho:g}:{baseline['method']}")
            baseline_count += 1
        model = pipeline(rows, "iv", spec)
        reconstructed[rho] = model
        reference = np.asarray(old["reference_response"])
        relative_error = (float(np.linalg.norm(model["response"]-reference)/np.linalg.norm(reference))
                          if model["status"] == "ready" else None)
        compare.tree(new_models[rho], {"rho": rho, "instrument": "controller_filtered", "model": model,
                                      "response_relative_error": relative_error}, f"filtered IV {rho:g}")

    expected_cases = {(seed, rho, drift) for rho in spec["correlations"]
                      for seed in spec["forecast_roots"]
                      for drift in ([False, True] if rho == spec["correlations"][-1] else [False])}
    identity = lambda row: (row["root"], row["rho"], row["drift"])
    old_cases, new_cases = ({identity(row): row for row in collection}
                          for collection in (old["forecasts"], evidence["forecasts"]))
    require(len(old["forecasts"]) == len(evidence["forecasts"]) == len(expected_cases)
            and set(old_cases) == set(new_cases) == expected_cases,
            "Missing, duplicate or unexpected forecast case")
    forecast_count = 0
    for case, previous in old_cases.items():
        seed, rho, drift = case
        actual = new_cases[case]
        preserved = deepcopy(actual)
        result = preserved["predictions"].pop("controller_filtered_iv")
        require(preserved == previous, f"Baseline forecast or probing cost changed: {case}")
        model = reconstructed[rho]
        if model["status"] != "ready":
            require(result == {"status": "withheld", "reason": model["reason"]},
                    f"Withheld forecast differs: {case}")
            continue
        predicted = prediction(model, raw(seed, rho, drift), scale, delay)
        name = f"forecast-{seed}-rho{rho:g}-drift{int(drift)}.npz"
        expected_artifacts.add(name)
        require(name in artifacts, f"Missing forecast artifact: {name}")
        saved = archive(directory/name, predicted)
        require(saved["target_indices"].dtype.kind in "iu", "Forecast indices are not integers")
        for field in predicted:
            compare.array(saved[field], predicted[field], name+":"+field, exact=field == "target_indices")
        targets = predicted["target_indices"]
        windows = {"post_burn": targets >= start}
        if drift:
            windows.update(before_drift=(targets >= start) & (targets < spec["drift"]["onset"]),
                           settled_drift=targets >= spec["drift"]["onset"]+spec["drift"]["drift_frames"])
        losses = {name: float(np.mean(predicted["error"][selected]**2)) for name, selected in windows.items()}
        compare.tree(result, {"status": "ready", "mse_by_window": losses}, f"forecast metrics {case}")
        forecast_count += 1
    require(set(artifacts) == expected_artifacts, "Unexpected or incomplete derived artifact collection")
    # README and verifier are publication support; all result payloads must be bound.
    payload = {path.name for path in directory.iterdir() if path.suffix in (".npz", ".json")}
    require(payload == expected_artifacts | {"manifest.json"}, "Unbound result payloads in directory")
    return {"analysis_source_commit": analysis, "acquisition_source_commit": acquisition,
            "verified_inputs": len(input_names), "verified_artifacts": len(artifacts),
            "reconstructed_references": len(spec["correlations"])*len(spec["training_roots"]),
            "reconstructed_baselines": baseline_count, "reconstructed_models": len(reconstructed),
            "model_statuses": {str(rho): model["status"] for rho, model in reconstructed.items()},
            "reconstructed_forecasts": forecast_count,
            "maximum_absolute_numerical_difference": compare.maximum_absolute_error,
            "new_trajectories": 0}


def repository_root(explicit):
    candidates = [explicit] if explicit else [Path(__file__).resolve().parent, Path.cwd()]
    for candidate in candidates:
        try:
            root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"],
                        cwd=candidate, stderr=subprocess.DEVNULL, text=True).strip()).resolve()
        except (OSError, subprocess.CalledProcessError):
            continue
        require(explicit is None or root == explicit.resolve(), "--root must identify the Git repository root")
        return root
    raise ValueError("Run inside the repository or provide --root")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="Git repository root (otherwise discovered)")
    parser.add_argument("--directory", type=Path, help="Derived result directory to verify")
    args = parser.parse_args()
    root = repository_root(args.root)
    here = Path(__file__).resolve().parent
    directory = args.directory or (here if (here/"manifest.json").is_file() else root/"runs/reference-iv-v1")
    print(json.dumps(verify(root, directory.resolve()), sort_keys=True))
