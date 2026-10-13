"""Durable training pilots and predictor fits, before aggregate training freeze.

    admitted pilot -> shared phase + calibration -> exact-noise integrator
      -> full raw outputs -> measured telemetry -> immutable ready/failed record
    complete pilot cohort -> independent predictor fits -> terminal model/evidence

Callers must open the run with fresh source/runtime provenance. No timings, stage
CLI, learned-fit scheduling or aggregate training binding are performed here.
"""
from __future__ import annotations

from copy import deepcopy
import shutil

import numpy as np

from .controller_evidence import (check_provenance_binding, check_training_open,
    load_arrays, record_interruption, save_arrays, save_metadata)
from .controller_physics import (FULL_OUTPUT_KEYS, NumericalTrainingFailure, _identity,
    episode_config, fit_measured_predictor, measured_telemetry, training_cases)
from .controller_protocol import study_plan
from .controller_selection import _bytes, _case_id, _digest, _read

PILOTS = "evidence/training/pilots"


def _admit(ledger, base, shared):
    check_training_open(ledger)
    check_provenance_binding(ledger, base, shared)
    ledger._verify_artifacts(shared)


def _journal(ledger, shared):
    return save_metadata(ledger, PILOTS + "/journal.json", {
        "manifest_sha256": _digest(_bytes(ledger.manifest)), "shared_evidence": shared})


def _phase_path(case):
    identity = {key: case[key] for key in ("seed", "wind_scale")}
    return f"evidence/training/atmospheres/{_case_id(identity)}", identity


def _references(row):
    return [row[key] for key in ("calibration", "phase", "phase_metadata", "outputs", "telemetry")
            if row[key] is not None]


def _audit_pilots(ledger, base, journal):
    expected = {_case_id(case): case for case in training_cases(ledger.config)}
    records = {}
    for path in sorted((ledger.root / PILOTS / "records").glob("*.json")):
        row = _read(path)
        if (set(row) != {"case", "journal", "config", "status", "error", "calibration",
                         "phase", "phase_metadata", "outputs", "telemetry"}
                or path.stem not in expected or _bytes(row["case"]) != _bytes(expected[path.stem])
                or row["journal"] != journal):
            raise ValueError("Saved pilot identity changed")
        case = expected[path.stem]
        cfg = episode_config(base, ledger.config, "training", case["photons"], case["wind_scale"])
        phase_path, identity = _phase_path(case)
        if (_bytes(row["config"]) != _bytes(cfg)
                or row["calibration"]["path"] != PILOTS + "/calibration.npz"
                or row["phase"]["path"] != phase_path + ".npz"
                or row["phase_metadata"]["path"] != phase_path + ".json"):
            raise ValueError("Saved pilot input binding changed")
        ledger._verify_artifacts(_references(row))
        phase = _read(ledger.root / row["phase_metadata"]["path"])
        if phase != {"identity": identity, "journal": journal, "raw": row["phase"]}:
            raise ValueError("Saved pilot atmosphere identity changed")
        for key in ("outputs", "telemetry"):
            if row[key] is not None and row[key]["path"] != f"{PILOTS}/{key}/{path.stem}.npz":
                raise ValueError("Saved pilot artifact identity changed")
        if row["status"] == "ready":
            if row["error"] is not None or row["outputs"] is None or row["telemetry"] is None:
                raise ValueError("Ready pilot requires raw observations and telemetry")
        elif row["status"] != "failed" or not isinstance(row["error"], str) or not row["error"].strip():
            raise ValueError("Invalid saved pilot status")
        records[path.stem] = row
    return records


def acquire_pilots(ledger, base, shared_evidence, limit):
    """Audit old records and acquire at most limit new training pilot episodes."""
    ledger._check_limit(limit)
    base, shared_evidence = deepcopy(base), deepcopy(shared_evidence)
    with ledger._locked():
        _admit(ledger, base, shared_evidence)
        journal = _journal(ledger, shared_evidence)
        records = _audit_pilots(ledger, base, journal)
        execute = _PilotExecutor(ledger, base, journal)
        written = 0
        try:
            for case in training_cases(ledger.config):
                key = _case_id(case)
                if key in records:
                    continue
                row = execute(case)
                _admit(ledger, base, shared_evidence)
                if _journal(ledger, shared_evidence) != journal:
                    raise ValueError("Pilot journal changed during execution")
                ledger._verify_artifacts(_references(row))
                save_metadata(ledger, f"{PILOTS}/records/{key}.json", row)
                records[key] = row
                written += 1
                if written == limit:
                    break
        except BaseException as error:
            record_interruption(ledger, PILOTS, journal, error)
            raise
        return {"new_records": written, "total_records": len(records),
                "complete": len(records) == len(training_cases(ledger.config)),
                "failed_records": sum(row["status"] == "failed" for row in records.values())}


class _PilotExecutor:
    def __init__(self, ledger, base, journal):
        self.ledger, self.base, self.journal = ledger, base, journal
        self.cal = None
        self.kernels = {}

    def __call__(self, case):
        from .atmosphere import phase_sequence
        from .controller_physics import controller_kernel
        from .simulation import calibration

        cfg = episode_config(self.base, self.ledger.config, "training", case["photons"], case["wind_scale"])
        estimate = cfg["frames"] * (cfg["n"]**2 + 2 * cfg["detector_n"]**2 + 6 * cfg["modes"] + 16) * 8
        if shutil.disk_usage(self.ledger.root).free < max(512 * 1024**2, 2 * estimate):
            raise RuntimeError("Insufficient free disk before pilot acquisition")
        if self.cal is None:
            self.cal = calibration(self.base)
            numeric = {key: np.asarray(value) for key, value in self.cal.items() if not callable(value)}
            self.cal_ref = save_arrays(self.ledger, PILOTS + "/calibration.npz", numeric)
            if not all(np.isfinite(value).all() for value in numeric.values()):
                raise ValueError("Nonfinite pilot calibration; retained for inspection")
        path, identity = _phase_path(case)
        if (self.ledger.root / (path + ".json")).exists():
            saved = _read(self.ledger.root / (path + ".json"))
            if (set(saved) != {"identity", "journal", "raw"} or saved["identity"] != identity
                    or saved["journal"] != self.journal or saved["raw"]["path"] != path + ".npz"):
                raise ValueError("Training atmosphere identity changed")
            phase_ref = saved["raw"]
            phases = load_arrays(self.ledger, phase_ref)
            if set(phases) != {"phase"}:
                raise ValueError("Invalid training atmosphere schema")
            phases = phases["phase"]
            phase_metadata = self.ledger.artifact(path + ".json")
        else:
            phases = np.asarray(phase_sequence(cfg, case["seed"]))
            phase_ref = save_arrays(self.ledger, path + ".npz", {"phase": phases})
            phase_metadata = save_metadata(self.ledger, path + ".json", {
                "identity": identity, "journal": self.journal, "raw": phase_ref})
        if phases.shape != (cfg["frames"], cfg["n"], cfg["n"]) or phases.dtype.kind != "f":
            raise ValueError("Invalid training atmosphere shape or dtype")
        row = {"case": deepcopy(case), "journal": self.journal, "config": cfg,
               "calibration": self.cal_ref, "phase": phase_ref, "phase_metadata": phase_metadata,
               "outputs": None, "telemetry": None, "status": "failed", "error": None}
        if not np.isfinite(phases).all():
            row["error"] = "Nonfinite training atmosphere"
            return row
        regime = (case["photons"], case["wind_scale"])
        if regime not in self.kernels:
            self.kernels[regime] = controller_kernel(cfg, self.cal, "integrator", "none")
        outputs = {key: np.asarray(value) for key, value in self.kernels[regime](phases, case["seed"]).items()}
        key = _case_id(case)
        row["outputs"] = save_arrays(self.ledger, f"{PILOTS}/outputs/{key}.npz", outputs)
        _check_outputs(outputs, cfg)
        if not all(np.isfinite(value).all() for value in outputs.values()):
            row["error"] = "Nonfinite pilot outputs"
            return row
        try:
            telemetry = measured_telemetry(outputs, self.cal)
        except NumericalTrainingFailure as error:
            telemetry = error.evidence["telemetry"]
            row["error"] = str(error)
        row["telemetry"] = save_arrays(self.ledger, f"{PILOTS}/telemetry/{key}.npz", {"telemetry": telemetry})
        if row["error"] is None:
            row["status"] = "ready"
        return row


def _check_outputs(outputs, cfg):
    if set(outputs) != FULL_OUTPUT_KEYS:
        raise ValueError("Pilot requires all raw controller outputs")
    for key, value in outputs.items():
        width = cfg["modes"] if key in ("command", "applied", "estimate") else (
            cfg["detector_n"]**2 if key in ("measurement", "prediction") else None)
        shape = (cfg["frames"], width) if width is not None else (cfg["frames"],)
        if value.shape != shape:
            raise ValueError("Invalid pilot output shape: " + key)
        if key in ("saturated", "alarm", "fallback", "unavailable") and value.dtype != np.dtype(bool):
            raise ValueError("Invalid pilot boolean output: " + key)
    if outputs["fallback"].any() or outputs["unavailable"].any():
        raise ValueError("Clean training pilot used fallback or unavailable model")


def fit_predictor_record(ledger, base, candidate, shared_evidence):
    """Fit one declared predictor from the complete saved measured pilot cohort."""
    base, candidate, shared_evidence = deepcopy(base), deepcopy(candidate), deepcopy(shared_evidence)
    if (candidate.get("controller") != "predictive" or _identity(candidate) not in
            {_identity(row) for row in study_plan(ledger.config)["candidates"]}):
        raise ValueError("Expected a declared predictive candidate")
    with ledger._locked():
        _admit(ledger, base, shared_evidence)
        pilot_journal = _journal(ledger, shared_evidence)
        records = _audit_pilots(ledger, base, pilot_journal)
        cases = training_cases(ledger.config)
        if len(records) != len(cases):
            raise ValueError("Incomplete pilot cohort; predictor fitting is blocked")
        pilots = [records[_case_id(case)] for case in cases]
        pilot_refs = [ledger.artifact(f"{PILOTS}/records/{_case_id(case)}.json") for case in cases]
        prefix = f"evidence/training/{candidate['id']}"
        journal = save_metadata(ledger, prefix + "/journal.json", {
            "pilot_journal": pilot_journal, "pilots": pilot_refs, "candidate": candidate, "modes": base["modes"]})
        terminal_path = ledger.root / (prefix + "/complete.json")
        if terminal_path.exists():
            saved = _read(terminal_path)
            if set(saved) != {"journal", "result"} or saved["journal"] != journal:
                raise ValueError("Terminal predictor binding changed")
            _verify_predictor(ledger, saved["result"], base["modes"])
            return saved["result"]
        try:
            failure = [row["case"] for row in pilots if row["status"] == "failed"]
            fitted, error = None, "Training pilot failed; complete cohort is required" if failure else None
            if not failure:
                episodes = []
                for row in pilots:
                    outputs = load_arrays(ledger, row["outputs"])
                    _check_outputs(outputs, row["config"])
                    cal = load_arrays(ledger, row["calibration"])
                    measured = measured_telemetry(outputs, cal)
                    saved = load_arrays(ledger, row["telemetry"])
                    if set(saved) != {"telemetry"} or not np.array_equal(saved["telemetry"], measured):
                        raise ValueError("Saved pilot telemetry differs from measured reconstruction")
                    episodes.append({"case": row["case"], "telemetry": measured})
                try:
                    fitted = fit_measured_predictor(ledger.config, candidate, episodes, base["modes"])
                except NumericalTrainingFailure as failed:
                    fitted, error = failed.evidence, str(failed)
            _admit(ledger, base, shared_evidence)
            ledger._verify_artifacts(pilot_refs)
            refs = [*shared_evidence, pilot_journal, journal, *pilot_refs,
                    *[ref for row in pilots for ref in _references(row)]]
            arrays = {} if fitted is None else {key: fitted[key] for key in ("episode_coefficients", "coefficients")
                                               if fitted[key] is not None}
            raw = save_arrays(ledger, prefix + "/history.npz", arrays) if arrays else None
            summary = save_metadata(ledger, prefix + "/history.json", {
                "candidate": candidate, "cases": cases, "status": "failed" if error else "ready",
                "error": error, "failed_pilots": failure,
                "failed_case": fitted.get("failed_case") if fitted else None, "raw": raw})
            artifacts = ([raw] if raw else []) + [summary]
            if error is None:
                artifacts.insert(0, save_arrays(ledger, prefix + "/parameters.npz", {"coefficients": fitted["coefficients"]}))
            refs += artifacts
            for folder in (PILOTS, prefix):
                refs.extend(ledger.artifact(p.relative_to(ledger.root).as_posix())
                            for p in sorted((ledger.root / folder / "interruptions").glob("*.json")))
            unique = {ref["path"]: ref for ref in refs}
            result = {"model": {"status": "failed" if error else "ready", "error": error, "artifacts": artifacts},
                      "evidence": [unique[key] for key in sorted(unique)]}
            _verify_predictor(ledger, result, base["modes"])
            save_metadata(ledger, prefix + "/complete.json", {"journal": journal, "result": result})
            return result
        except BaseException as error:
            record_interruption(ledger, prefix, journal, error)
            raise


def _verify_predictor(ledger, result, modes):
    if set(result) != {"model", "evidence"} or set(result["model"]) != {"status", "error", "artifacts"}:
        raise ValueError("Invalid terminal predictor schema")
    model = result["model"]
    ledger._verify_artifacts(result["evidence"])
    ledger._verify_artifacts(model["artifacts"])
    if model["status"] == "ready":
        if model["error"] is not None or not model["artifacts"]:
            raise ValueError("Ready predictor requires finite coefficients")
        params = load_arrays(ledger, model["artifacts"][0])
        if (set(params) != {"coefficients"} or params["coefficients"].shape != (modes, 2)
                or not np.isfinite(params["coefficients"]).all()):
            raise ValueError("Invalid terminal predictor coefficients")
    elif model["status"] != "failed" or not isinstance(model["error"], str) or not model["error"].strip():
        raise ValueError("Invalid terminal predictor status")
