"""Immutable learned-episode journal; replay gradients through the existing Adam loop.

No pickled optimizer state, reserved-cohort generation or timing measurement.
Callers supply the verified run, calibration and complete training atmosphere set.
"""
from __future__ import annotations

import math

import jax
import numpy as np

from .controller_evidence import (check_provenance_binding, check_training_open,
                                  load_arrays, record_interruption, save_arrays, save_metadata)
from .controller_physics import _identity, training_cases
from .controller_protocol import study_plan
from .controller_selection import _bytes, _case_id, _digest, _read
from .controller_training import _training_inputs, fit_learned_controller


def fit_learned_record(ledger, base, candidate, cal, episodes, shared_evidence):
    """Fit/resume one candidate and return its terminal model/evidence references.

    The ledger lock covers input publication, cached replay, missing computations
    and final publication. A bound training set cannot be refit. Caller must open
    the ledger with freshly captured provenance before entering this function.
    """
    with ledger._locked():
        check_training_open(ledger)
        check_provenance_binding(ledger, base, shared_evidence)
        ledger._verify_artifacts(shared_evidence)
        saved = _load_terminal(ledger, candidate, shared_evidence)
        if saved is not None:
            return saved
        cases, _, denominators, found = _training_inputs(base, ledger.config, candidate, cal, episodes)
        prefix = f"evidence/training/{candidate['id']}"
        numeric_cal = {key: np.asarray(value) for key, value in cal.items() if not callable(value)}
        calibration = save_arrays(ledger, prefix + "/calibration.npz", numeric_cal)
        inputs = []
        for case, denominator in zip(cases, denominators):
            identity = {"seed": case["seed"], "wind_scale": case["wind_scale"]}
            path = f"evidence/training/atmospheres/{_case_id(identity)}.npz"
            ref = save_arrays(ledger, path, {"phase": found[_identity(case)]})
            inputs.append({"case": case, "phase": ref, "open_variance": denominator})
        binding = {"manifest_sha256": _digest(_bytes(ledger.manifest)), "candidate": candidate,
                   "shared_evidence": shared_evidence, "calibration": calibration, "episodes": inputs}
        journal = save_metadata(ledger, prefix + "/journal.json", binding)
        cache = _EpisodeJournal(ledger, prefix, journal, inputs)
        try:
            result = fit_learned_controller(base, ledger.config, candidate, cal,
                [{"case": case, "phases": found[_identity(case)]} for case in cases],
                episode_evaluator=cache)
            check_training_open(ledger)
            check_provenance_binding(ledger, base, shared_evidence)
            if (_digest(_bytes(ledger.manifest)) != binding["manifest_sha256"]
                    or _bytes(result["candidate"]) != _bytes(binding["candidate"])):
                raise ValueError("Training inputs changed during fitting")
            model, history = _save_result(ledger, prefix, result)
            refs = [*shared_evidence, calibration, journal, *[row["phase"] for row in inputs], *cache.references, *history]
            refs.extend(ledger.artifact(path.relative_to(ledger.root).as_posix())
                        for path in sorted((ledger.root / prefix / "interruptions").glob("*.json")))
            # Atmospheres/step parameters are intentionally shared across episodes.
            unique = {ref["path"]: ref for ref in refs}
            terminal = {"model": model, "evidence": [unique[key] for key in sorted(unique)]}
            _verify_result(ledger, terminal)
            save_metadata(ledger, prefix + "/complete.json", {"journal": journal, "result": terminal})
            return terminal
        except BaseException as error:
            record_interruption(ledger, prefix, journal, error)
            raise


def load_learned_record(ledger, base, candidate, shared_evidence):
    """Return a verified terminal learned record without running optical kernels."""
    with ledger._locked():
        check_training_open(ledger)
        check_provenance_binding(ledger, base, shared_evidence)
        ledger._verify_artifacts(shared_evidence)
        return _load_terminal(ledger, candidate, shared_evidence)


def _load_terminal(ledger, candidate, shared_evidence):
    declared = {_identity(row) for row in study_plan(ledger.config)["candidates"]
                if row["controller"] == "learned"}
    if candidate.get("controller") != "learned" or _identity(candidate) not in declared:
        raise ValueError("Expected a declared learned candidate")
    prefix = f"evidence/training/{candidate['id']}"
    terminal_path = ledger.root / (prefix + "/complete.json")
    if not terminal_path.exists():
        return None

    binding = _read(ledger.root / (prefix + "/journal.json"))
    calibration = binding.get("calibration") if isinstance(binding, dict) else None
    episodes = binding.get("episodes") if isinstance(binding, dict) else None
    if (not isinstance(binding, dict)
            or set(binding) != {"manifest_sha256", "candidate", "shared_evidence", "calibration", "episodes"}
            or binding["manifest_sha256"] != _digest(_bytes(ledger.manifest))
            or _bytes(binding["candidate"]) != _bytes(candidate)
            or binding["shared_evidence"] != shared_evidence
            or not isinstance(calibration, dict)
            or calibration.get("path") != prefix + "/calibration.npz"
            or not isinstance(episodes, list)):
        raise ValueError("Terminal training input binding changed")
    cases = training_cases(ledger.config)
    if len(episodes) != len(cases):
        raise ValueError("Terminal training cohort changed")
    required = [*shared_evidence, calibration]
    for case, row in zip(cases, episodes):
        identity = {"seed": case["seed"], "wind_scale": case["wind_scale"]}
        expected_path = f"evidence/training/atmospheres/{_case_id(identity)}.npz"
        variance = row.get("open_variance") if isinstance(row, dict) else None
        phase = row.get("phase") if isinstance(row, dict) else None
        if (not isinstance(row, dict) or set(row) != {"case", "phase", "open_variance"}
                or _bytes(row["case"]) != _bytes(case) or not isinstance(phase, dict)
                or phase.get("path") != expected_path or type(variance) not in (int, float)
                or not math.isfinite(variance) or variance <= 0):
            raise ValueError("Terminal training episode binding changed")
        required.append(phase)
    unique = {ref["path"]: ref for ref in required}
    ledger._verify_artifacts([unique[path] for path in sorted(unique)])

    journal = ledger.artifact(prefix + "/journal.json")
    saved = _read(terminal_path)
    if (not isinstance(saved, dict) or set(saved) != {"journal", "result"}
            or saved["journal"] != journal):
        raise ValueError("Terminal training binding changed")
    _verify_result(ledger, saved["result"])
    evidence_paths = {ref["path"] for ref in saved["result"].get("evidence", [])}
    if not (set(unique) | {journal["path"]}).issubset(evidence_paths):
        raise ValueError("Terminal training evidence is incomplete")
    return saved["result"]


def _verify_result(ledger, result):
    if set(result) != {"model", "evidence"}:
        raise ValueError("Invalid terminal training record")
    model = result["model"]
    if set(model) != {"status", "artifacts", "error"}:
        raise ValueError("Invalid terminal model")
    ledger._verify_artifacts(result["evidence"])
    ledger._verify_artifacts(model["artifacts"])
    if model["status"] == "ready":
        if model["error"] is not None or not model["artifacts"]:
            raise ValueError("Ready model requires finite parameters")
        parameters = load_arrays(ledger, model["artifacts"][0])
        shapes = ((3, 8), (8,), (8, 1), (1,))
        if set(parameters) != {f"parameter_{i}" for i in range(4)} or any(
                parameters[f"parameter_{i}"].shape != shape or not np.isfinite(parameters[f"parameter_{i}"]).all()
                for i, shape in enumerate(shapes)):
            raise ValueError("Invalid terminal learned parameters")
    elif model["status"] != "failed" or not isinstance(model["error"], str) or not model["error"].strip():
        raise ValueError("Invalid terminal training status")


class _EpisodeJournal:
    def __init__(self, ledger, prefix, journal, inputs):
        self.ledger, self.prefix, self.journal = ledger, prefix, journal
        self.inputs = {_identity(row["case"]): row for row in inputs}
        self.references = []

    def __call__(self, step, case, params, phases, denominator, compute):
        row = self.inputs[_identity(case)]
        if denominator != row["open_variance"]:
            raise ValueError("Training normalization changed during replay")
        prefix = self.prefix + f"/steps/{step:04d}"
        parameter_ref = save_arrays(self.ledger, prefix + "/parameters.npz",
                                   {f"parameter_{i}": np.asarray(p) for i, p in enumerate(params)})
        identity = {"journal": self.journal, "step": step, "case": case, "parameters": parameter_ref,
                    "phase": row["phase"], "open_variance": denominator}
        relative = prefix + "/" + _case_id(case)
        path = self.ledger.root / (relative + ".json")
        if path.exists():
            saved = _read(path)
            if (set(saved) != {"identity", "raw"} or _bytes(saved["identity"]) != _bytes(identity)
                    or saved["raw"]["path"] != relative + ".npz"):
                raise ValueError("Saved training episode identity changed")
            arrays = load_arrays(self.ledger, saved["raw"])
            raw = saved["raw"]
            metadata = self.ledger.artifact(relative + ".json")
        else:
            (loss, finite), gradient = compute()
            arrays = {"loss": np.asarray(loss), "outputs_finite": np.asarray(finite),
                      **{f"gradient_{i}": np.asarray(g) for i, g in enumerate(gradient)}}
            _check_episode(arrays, params)
            raw = save_arrays(self.ledger, relative + ".npz", arrays)
            metadata = save_metadata(self.ledger, relative + ".json", {"identity": identity, "raw": raw})
        _check_episode(arrays, params)
        self.references.extend((parameter_ref, raw, metadata))
        return (arrays["loss"], arrays["outputs_finite"]), tuple(arrays[f"gradient_{i}"] for i in range(len(params)))


def _check_episode(arrays, params):
    if set(arrays) != {"loss", "outputs_finite", *[f"gradient_{i}" for i in range(len(params))]}:
        raise ValueError("Invalid saved training episode schema")
    if (arrays["loss"].shape != () or arrays["loss"].dtype.kind != "f"
            or arrays["outputs_finite"].shape != () or arrays["outputs_finite"].dtype != np.dtype(bool)):
        raise ValueError("Invalid saved loss or finite flag")
    if any(arrays[f"gradient_{i}"].shape != np.shape(p)
           or arrays[f"gradient_{i}"].dtype != np.asarray(p).dtype for i, p in enumerate(params)):
        raise ValueError("Invalid saved gradient shape or dtype")
    # Nonfinite values are preserved for the numerical helper's failure decision.


def _save_result(ledger, prefix, result):
    arrays = {"open_variances": result["open_variances"]}
    for k, checkpoint in enumerate(result["checkpoints"]):
        arrays.update({f"checkpoint_{k}_parameter_{i}": value for i, value in enumerate(checkpoint)})
    steps = []
    for record in result["steps"]:
        step = record["step"]
        row = {"step": step, "episode_count": len(record["episode_losses"])}
        for key in ("mean_loss", "gradient_norm"):
            if key in record:
                name = f"step_{step}_{key}"
                arrays[name] = np.asarray(record[key])
                row[key] = name
        for i, gradient in enumerate(record.get("mean_gradient", ())):
            arrays[f"step_{step}_mean_gradient_{i}"] = gradient
        if "failed_optimizer_state" in record:
            leaves, structure = jax.tree.flatten(record["failed_optimizer_state"])
            row["failed_optimizer_structure"] = str(structure)
            arrays.update({f"step_{step}_failed_optimizer_{i}": value for i, value in enumerate(leaves)})
        steps.append(row)
    if result["final"] is not None and "mean_loss" in result["final"]:
        arrays["final_mean_loss"] = np.asarray(result["final"]["mean_loss"])
    raw = save_arrays(ledger, prefix + "/history.npz", arrays)
    summary = save_metadata(ledger, prefix + "/history.json", {
        "status": result["status"], "error": result["error"], "candidate": result["candidate"],
        "cases": result["cases"], "steps": steps, "checkpoint_count": len(result["checkpoints"]),
        "final_episode_count": len(result["final"]["episode_losses"]) if result["final"] is not None else 0,
        "raw": raw})
    artifacts = [raw, summary]
    if result["status"] == "ready":
        parameters = save_arrays(ledger, prefix + "/parameters.npz",
                                 {f"parameter_{i}": value for i, value in enumerate(result["parameters"])})
        artifacts.insert(0, parameters)
    return {"status": result["status"], "artifacts": artifacts, "error": result["error"]}, [raw, summary]
