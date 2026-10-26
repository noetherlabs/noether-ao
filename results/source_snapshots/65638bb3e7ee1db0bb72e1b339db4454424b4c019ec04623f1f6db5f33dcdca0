"""Immutable study records and the validation-to-evaluation barrier.

This module performs no simulation and imports no optical or random-number code.
The future physics adapter supplies raw artifacts, metrics and fresh provenance.

    config + provenance -> training bundle -> complete validation -> selection
                                                                   |
    fresh provenance + rechecked evidence -------------------------> gate
                                                                   |
                                              resumable evaluation callbacks

All public mutations hold one process lock. A callback runs under that lock;
another process cannot freeze training or append trials concurrently. Files are
published atomically without replacement. An interrupted temporary file is never
a completed trial. Hash checks detect accidental drift, not a malicious writer
with permission to replace the entire run directory.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
from itertools import product
import json
import math
import os
from pathlib import Path
import re
import tempfile

from .controller_protocol import PARAMETERS, study_plan, validate_study


def _bytes(value) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read(path: Path):
    return json.loads(path.read_bytes())


def _publish(path: Path, value) -> None:
    """Atomic, exclusive publication; an identical retry is an idempotent no-op."""
    data = _bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(name, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise ValueError(f"Refusing to replace existing evidence: {path.name}") from None
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(name).unlink(missing_ok=True)


def _provenance(value: dict) -> None:
    # The adapter must calculate these from the actual checkout/environment.
    if set(value) != {"source_commit", "source_sha256", "runtime_sha256"}:
        raise ValueError("Require source commit, source inventory hash and runtime hash")
    for key, length in (("source_commit", 40), ("source_sha256", 64), ("runtime_sha256", 64)):
        if not isinstance(value[key], str) or not re.fullmatch(f"[0-9a-f]{{{length}}}", value[key]):
            raise ValueError(f"Invalid provenance: {key}")


def _cases(config: dict, stage: str, candidates: list[dict]) -> list[dict]:
    return [dict(stage=stage, candidate=candidate, photons=photons, wind_scale=wind,
                 fault=fault, seed=seed, detector_seed=seed + config["seeds"]["noise_offset"])
            for candidate, photons, wind, fault, seed in product(
                candidates, config["conditions"]["photons"], config["conditions"]["wind_scales"],
                config["conditions"][f"{stage}_faults"], config["seeds"][stage])]


def _case_id(case: dict) -> str:
    return _digest(_bytes(case))


def validation_cases(config: dict) -> list[dict]:
    """Return the complete declared validation identities without accessing arrays."""
    return _cases(config, "validation", study_plan(config)["candidates"])


def _outcome(record: dict) -> None:
    if set(record) != {"case", "status", "metrics", "artifacts", "open_loop", "error"}:
        raise ValueError("Unexpected trial fields")
    if record["status"] == "failed":
        if record["metrics"] is not None or not isinstance(record["error"], str) or not record["error"].strip():
            raise ValueError("A failed trial needs an error and no invented metrics")
    elif record["status"] == "completed":
        if record["error"] is not None or not record["artifacts"] or record["open_loop"] is None:
            raise ValueError("A completed trial needs raw artifacts and no error")
        metrics = record["metrics"]
        if not isinstance(metrics, dict) or set(metrics) != {
            "all_outputs_finite", "wfe_nm", "open_wfe_nm", "saturation_fraction", "strehl"
        } or type(metrics["all_outputs_finite"]) is not bool:
            raise ValueError("Invalid metric schema")
        for key in ("wfe_nm", "open_wfe_nm", "saturation_fraction", "strehl"):
            value = metrics[key]
            if value is None and not metrics["all_outputs_finite"]:
                continue  # JSON null preserves a nonfinite result without using JSON NaN.
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid metric: {key}")
            if key in ("saturation_fraction", "strehl") and value > 1:
                raise ValueError(f"Invalid metric: {key}")
    else:
        raise ValueError("Unknown trial status")


def select_candidates(config: dict, records: list[dict]) -> dict:
    """Apply the declared gates only after every candidate/case has a record."""
    candidates = study_plan(config)["candidates"]
    expected = {_case_id(case): case for case in validation_cases(config)}
    found, references = {}, {}
    for record in records:
        _outcome(record)
        key = _case_id(record["case"])
        if key not in expected or record["case"] != expected[key]:
            raise ValueError("Unexpected validation identity")
        if key in found:
            raise ValueError("Duplicate validation identity")
        if record["status"] == "completed":
            condition = {name: value for name, value in record["case"].items() if name != "candidate"}
            reference = _bytes([record["open_loop"], record["metrics"]["open_wfe_nm"]])
            condition_key = _case_id(condition)
            if condition_key in references and references[condition_key] != reference:
                raise ValueError("Candidates must share the paired open-loop reference")
            references[condition_key] = reference
        found[key] = record
    if found.keys() != expected.keys():
        raise ValueError(f"Incomplete validation: {len(found)} of {len(expected)} trials")
    scores = []
    for candidate in candidates:
        rows = [found[key] for key, case in expected.items() if case["candidate"] == candidate]
        rejected, ratios = [], []
        for row in rows:
            reasons = []
            m = row["metrics"]
            if row["status"] == "failed":
                reasons.append("execution_failed")
            elif not m["all_outputs_finite"]:
                reasons.append("nonfinite_output")
            else:
                if m["saturation_fraction"] > config["selection"]["max_saturation_fraction"]:
                    reasons.append("saturation_limit")
                if m["open_wfe_nm"] <= 0 or m["wfe_nm"] >= m["open_wfe_nm"]:
                    reasons.append("not_below_open_loop")
                else:
                    ratios.append(m["wfe_nm"] / m["open_wfe_nm"])
            if reasons:
                rejected.append({"case_id": _case_id(row["case"]), "reasons": reasons})
        scores.append({"candidate": candidate, "eligible": not rejected,
                       "score": math.fsum(ratios) / len(rows) if not rejected else None,
                       "rejected_trials": rejected})
    selected = {}
    for controller in PARAMETERS:
        eligible = [row for row in scores if row["candidate"]["controller"] == controller and row["eligible"]]
        # min is stable in the predeclared candidate order; exact ties retain the first.
        selected[controller] = min(eligible, key=lambda row: row["score"])["candidate"] if eligible else None
    return {"selected": selected, "candidates": scores,
            "validation_records": len(found),
            "completed_validation_trials": sum(row["status"] == "completed" for row in records),
            "failed_validation_records": sum(row["status"] == "failed" for row in records),
            "pairwise_family": study_plan(config)["predeclared_pairwise_family"]}


class StudyLedger:
    """One run's stage boundaries; the physics adapter owns array/metric semantics."""

    def __init__(self, root: Path, config: dict, current_provenance: dict):
        validate_study(config)
        _provenance(current_provenance)
        self.root = Path(root).resolve()
        # Detach from caller-owned mutable dictionaries.
        self.config = json.loads(_bytes(config))
        self.provenance = json.loads(_bytes(current_provenance))
        self.manifest = {"schema_version": 1, "config": self.config, "provenance": self.provenance}
        if _bytes(_read(self.root / "manifest.json")) != _bytes(self.manifest):
            raise ValueError("Run configuration or source/runtime provenance changed")

    @classmethod
    def create(cls, root: Path, config: dict, current_provenance: dict):
        validate_study(config)
        _provenance(current_provenance)
        root = Path(root)
        root.mkdir(parents=True, exist_ok=False)
        _publish(root / "manifest.json", {"schema_version": 1, "config": config, "provenance": current_provenance})
        return cls(root, config, current_provenance)

    @contextmanager
    def _locked(self):
        with (self.root / ".lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                if _bytes(_read(self.root / "manifest.json")) != _bytes(self.manifest):
                    raise ValueError("Run manifest changed")
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def artifact(self, relative: str) -> dict:
        """Hash an existing raw/model/provenance file within evidence/."""
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "evidence":
            raise ValueError("Artifacts must stay inside evidence/")
        target = self.root / path
        if not target.resolve().is_relative_to(self.root / "evidence"):
            raise ValueError("Artifact symlink escapes evidence/")
        if not target.is_file():
            raise ValueError(f"Missing artifact: {relative}")
        digest = hashlib.sha256()
        with target.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return {"path": path.as_posix(), "sha256": digest.hexdigest()}

    def _verify_artifacts(self, artifacts: list[dict]) -> None:
        if not isinstance(artifacts, list):
            raise ValueError("Artifacts must be a list")
        paths = []
        for ref in artifacts:
            if not isinstance(ref, dict) or set(ref) != {"path", "sha256"} or self.artifact(ref["path"]) != ref:
                raise ValueError("Artifact hash or identity changed")
            paths.append(ref["path"])
        if len(set(paths)) != len(paths):
            raise ValueError("Duplicate artifact reference")

    def bind_training(self, models: dict, evidence: list[dict]) -> None:
        """Freeze trained candidate bindings before the first validation trial."""
        with self._locked():
            if (self.root / "selection.json").exists():
                raise ValueError("Selection is frozen")
            bundle = {"models": models, "evidence": evidence}
            self._check_training(bundle)
            _publish(self.root / "training.json", bundle)

    def _check_training(self, bundle: dict) -> None:
        trained = {row["id"] for row in study_plan(self.config)["candidates"]
                   if row["controller"] in ("predictive", "learned")}
        if set(bundle) != {"models", "evidence"} or set(bundle["models"]) != trained or not bundle["evidence"]:
            raise ValueError("Training must cover every fitted candidate and include evidence")
        self._verify_artifacts(bundle["evidence"])
        for model in bundle["models"].values():
            if set(model) != {"status", "artifacts", "error"}:
                raise ValueError("Invalid trained-model schema")
            self._verify_artifacts(model["artifacts"])
            if model["status"] == "ready":
                if not model["artifacts"] or model["error"] is not None:
                    raise ValueError("A ready model needs parameters and no error")
            elif model["status"] == "failed":
                if not isinstance(model["error"], str) or not model["error"].strip():
                    raise ValueError("A failed model needs an error")
            else:
                raise ValueError("Unknown model status")

    def _training(self) -> dict:
        bundle = _read(self.root / "training.json")
        self._check_training(bundle)
        return bundle

    def _check_record(self, record: dict, training: dict) -> None:
        _outcome(record)
        self._verify_artifacts(record["artifacts"])
        if record["open_loop"] is not None:
            self._verify_artifacts([record["open_loop"]])
        model = training["models"].get(record["case"]["candidate"]["id"])
        if model and model["status"] == "failed" and record["status"] != "failed":
            raise ValueError("Failed training cannot yield a completed trial")

    def record_validation(self, record: dict) -> None:
        with self._locked():
            if (self.root / "selection.json").exists():
                raise ValueError("Selection is frozen")
            expected = {_case_id(case) for case in validation_cases(self.config)}
            if _case_id(record["case"]) not in expected:
                raise ValueError("Unexpected validation identity")
            training = self._training()
            self._check_record(record, training)
            _publish(self.root / "validation" / f"{_case_id(record['case'])}.json",
                     {"training_sha256": _digest(_bytes(training)), "record": record})

    def _selection(self) -> dict:
        training = self._training()
        records = []
        for path in sorted((self.root / "validation").glob("*.json")):
            saved = _read(path)
            if set(saved) != {"training_sha256", "record"} or saved["training_sha256"] != _digest(_bytes(training)):
                raise ValueError("Validation training binding changed")
            record = saved["record"]
            if path.stem != _case_id(record["case"]):
                raise ValueError("Validation file identity changed")
            self._check_record(record, training)
            records.append(record)
        return {"manifest_sha256": _digest(_bytes(self.manifest)),
                "training_sha256": _digest(_bytes(training)),
                "validation_sha256": _digest(_bytes(records)),
                "result": select_candidates(self.config, records)}

    def freeze_selection(self) -> dict:
        with self._locked():
            selection = self._selection()
            _publish(self.root / "selection.json", selection)
            return selection

    def _gate(self) -> tuple[dict, list[dict]]:
        if not (self.root / "selection.json").is_file():
            raise ValueError("Evaluation requires a frozen selection")
        selection = _read(self.root / "selection.json")
        if _bytes(selection) != _bytes(self._selection()):
            raise ValueError("Frozen selection or its evidence changed")
        chosen = [selection["result"]["selected"][name] for name in PARAMETERS
                  if selection["result"]["selected"][name] is not None]
        return selection, _cases(self.config, "evaluation", chosen)

    def evaluation_cases(self) -> list[dict]:
        """No evaluation identities are released until all frozen inputs recheck."""
        with self._locked():
            return self._gate()[1]

    def validate_chunk(self, execute, limit: int) -> dict:
        """Audit and append a bounded validation chunk, retaining saved failures."""
        self._check_limit(limit)
        with self._locked():
            if (self.root / "selection.json").exists():
                raise ValueError("Selection is frozen")
            training = self._training()
            return self._execute_chunk(execute, limit, "validation", validation_cases(self.config),
                                       training, "training_sha256", _digest(_bytes(training)))

    def evaluate_chunk(self, execute, limit: int) -> dict:
        """Run lazy callbacks only after frozen selection and all evidence recheck.

        Unexpected exceptions propagate. Adapters explicitly record scientific
        failures, and all existing outcomes are retained and skipped on resume.
        """
        self._check_limit(limit)
        with self._locked():
            selection, cases = self._gate()
            return self._execute_chunk(execute, limit, "evaluation", cases, self._training(),
                                       "selection_sha256", _digest(_bytes(selection)))

    @staticmethod
    def _check_limit(limit):
        if type(limit) is not int or limit < 1:
            raise ValueError("Chunk limit must be a positive integer")

    def _execute_chunk(self, execute, limit, stage, cases, training, binding, seal):
        # Called only while the stage's public entry point holds the process lock.
        expected = {_case_id(case): case for case in cases}
        completed = set()
        for path in sorted((self.root / stage).glob("*.json")):
            saved = _read(path)
            if set(saved) != {binding, "record"} or saved[binding] != seal:
                raise ValueError(f"{stage.title()} {binding.split('_')[0]} binding changed")
            record = saved["record"]
            if path.stem not in expected or _case_id(record["case"]) != path.stem:
                raise ValueError(f"Unexpected {stage} identity")
            self._check_record(record, training)
            completed.add(path.stem)
        written = 0
        for key, case in expected.items():
            if key in completed:
                continue
            record = execute(json.loads(_bytes(case)), json.loads(_bytes(training)))
            if _case_id(record["case"]) != key:
                raise ValueError(f"Backend returned a different {stage} identity")
            if (_bytes(_read(self.root / "training.json")) != _bytes(training)
                    or _bytes(_read(self.root / "manifest.json")) != _bytes(self.manifest)):
                raise ValueError("Backend changed frozen inputs")
            if stage == "evaluation" and _digest(_bytes(_read(self.root / "selection.json"))) != seal:
                raise ValueError("Backend changed frozen selection")
            if stage == "validation" and (self.root / "selection.json").exists():
                raise ValueError("Selection became frozen during validation")
            self._check_training(training)
            self._check_record(record, training)
            _publish(self.root / stage / f"{key}.json", {binding: seal, "record": record})
            written += 1
            if written == limit:
                break
        return {"new_records": written, "total_records": len(completed) + written,
                "planned_records": len(cases), "complete": len(completed) + written == len(cases)}
