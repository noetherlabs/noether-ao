"""Durable, locked stage records for the optical-gain observability study.

The ledger owns identity ordering and immutable terminal records. Numerical callbacks
own artifact production. Saved successes and scientific failures are never replayed;
an interrupted callback without a terminal JSON record is retried on the next chunk.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
from typing import Any

from .controller_selection import _bytes, _digest, _provenance, _publish, _read
from .observability_detector import (
    _artifact_arrays,
    _case_id,
    _file_digest,
    _load_protocol_binding,
    _score,
    freeze_thresholds,
    training_cases,
)
from .observability_validation import (
    _normalize_validation_record,
    _threshold_bundle,
    freeze_validation_release,
    validation_cases,
)


class ObservabilityLedger:
    """One protocol-bound run with resumable training and validation records."""

    def __init__(
        self,
        root: Path,
        protocol_path: Path,
        provenance: dict[str, Any],
        repository: Path | None = None,
    ):
        requested_root = Path(root)
        if not requested_root.is_dir() or requested_root.is_symlink():
            raise ValueError("Observability root must be an existing non-symlink directory")
        _provenance(provenance)
        config, protocol_sha256, protocol_file, repository = _load_protocol_binding(
            protocol_path, repository
        )
        self.root = requested_root.resolve()
        self.config = json.loads(_bytes(config))
        self.provenance = json.loads(_bytes(provenance))
        self.protocol_sha256 = protocol_sha256
        self.protocol_file = protocol_file
        self.repository = repository
        self.manifest = {
            "schema_version": 1,
            "kind": "optical_gain_observability_run",
            "study_id": self.config["study_id"],
            "protocol": self.config,
            "protocol_sha256": self.protocol_sha256,
            "provenance": self.provenance,
        }
        self.manifest_sha256 = _digest(_bytes(self.manifest))
        self._check_manifest()

    @classmethod
    def create(
        cls,
        root: Path,
        protocol_path: Path,
        provenance: dict[str, Any],
        repository: Path | None = None,
    ) -> "ObservabilityLedger":
        """Create a new empty run after validating every immutable input."""
        _provenance(provenance)
        config, protocol_sha256, _, resolved_repository = _load_protocol_binding(
            protocol_path, repository
        )
        root = Path(root)
        if root.is_symlink():
            raise ValueError("Observability root cannot be a symlink")
        root.mkdir(parents=True, exist_ok=False)
        manifest = {
            "schema_version": 1,
            "kind": "optical_gain_observability_run",
            "study_id": config["study_id"],
            "protocol": config,
            "protocol_sha256": protocol_sha256,
            "provenance": provenance,
        }
        _publish(root / "manifest.json", manifest)
        return cls(root, protocol_path, provenance, resolved_repository)

    def _check_manifest(self) -> None:
        path = self.root / "manifest.json"
        if path.is_symlink() or not path.is_file() or path.read_bytes() != _bytes(self.manifest):
            raise ValueError("Run manifest or its source/runtime bindings changed")
        if _file_digest(self.protocol_file) != self.protocol_sha256:
            raise ValueError("Protocol changed after the run was bound")

    @contextmanager
    def _locked(self):
        with (self.root / ".lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                self._check_manifest()
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def artifact(self, relative: str) -> dict[str, str]:
        """Hash one regular, canonical artifact below this run's evidence tree."""
        if not isinstance(relative, str):
            raise ValueError("Artifact path must be a string")
        path = Path(relative)
        if (
            path.is_absolute()
            or path.as_posix() != relative
            or ".." in path.parts
            or len(path.parts) < 2
            or path.parts[0] != "evidence"
        ):
            raise ValueError("Artifact must use a canonical path below evidence/")
        target = self.root / path
        for current in (target, *target.parents):
            if current == self.root:
                break
            if current.is_symlink():
                raise ValueError("Artifact path cannot contain a symlink")
        if not target.is_file() or not target.resolve().is_relative_to(self.root / "evidence"):
            raise ValueError(f"Missing or unsafe artifact: {relative}")
        return {"path": relative, "sha256": _file_digest(target)}

    @staticmethod
    def _check_limit(limit: int) -> None:
        if type(limit) is not int or limit < 1:
            raise ValueError("Chunk limit must be a positive integer")

    @staticmethod
    def _case(record: Any) -> dict[str, Any]:
        if not isinstance(record, dict) or not isinstance(record.get("case"), dict):
            raise ValueError("Terminal record must contain one case mapping")
        return record["case"]

    def _training_record(
        self, record: dict[str, Any], expected_case: dict[str, Any]
    ) -> tuple[dict[str, Any], str | None]:
        if not isinstance(record, dict) or set(record) != {
            "case",
            "status",
            "artifact",
            "error",
        }:
            raise ValueError("Invalid training record schema")
        if record["case"] != expected_case:
            raise ValueError("Training record identity changed")
        if record["status"] == "completed":
            if record["error"] is not None:
                raise ValueError("Completed training record cannot contain an error")
            artifact, arrays = _artifact_arrays(
                self.root, record["artifact"], "training"
            )
            if "score" not in arrays:
                raise ValueError("Completed training artifact is missing its score")
            _score(self.config, arrays["score"])
            return {
                "case": expected_case,
                "status": "completed",
                "artifact": artifact,
                "error": None,
            }, artifact["path"]
        if record["status"] == "failed":
            if (
                record["artifact"] is not None
                or not isinstance(record["error"], str)
                or not record["error"].strip()
            ):
                raise ValueError("Failed training record requires an error and no artifact")
            return {
                "case": expected_case,
                "status": "failed",
                "artifact": None,
                "error": record["error"],
            }, None
        raise ValueError("Unknown training record status")

    def _stage_directory(self, stage: str) -> Path:
        directory = self.root / stage
        if directory.is_symlink() or (
            directory.exists() and not directory.is_dir()
        ):
            raise ValueError(f"{stage.title()} record directory must be a regular directory")
        return directory

    def _saved(
        self,
        stage: str,
        expected_cases: list[dict[str, Any]],
        binding: dict[str, str],
        thresholds: dict[str, float] | None = None,
    ) -> tuple[dict[str, dict[str, Any]], set[str]]:
        expected = {_case_id(case): case for case in expected_cases}
        found: dict[str, dict[str, Any]] = {}
        artifact_paths: set[str] = set()
        directory = self._stage_directory(stage)
        for path in sorted(directory.glob("*.json")):
            if path.is_symlink():
                raise ValueError(f"{stage.title()} record cannot be a symlink")
            saved = _read(path)
            if path.read_bytes() != _bytes(saved) or set(saved) != {*binding, "record"}:
                raise ValueError(f"Invalid canonical {stage} record")
            if any(saved[key] != value for key, value in binding.items()):
                raise ValueError(f"{stage.title()} evidence binding changed")
            case = self._case(saved["record"])
            try:
                case_id = _case_id(case)
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid {stage} record identity") from error
            if path.stem != case_id or case_id not in expected or case_id in found:
                raise ValueError(f"Unexpected or duplicate {stage} identity")
            if stage == "training":
                record, artifact_path = self._training_record(
                    saved["record"], expected[case_id]
                )
            else:
                assert thresholds is not None
                normalized = _normalize_validation_record(
                    self.root,
                    self.config,
                    thresholds,
                    saved["record"],
                    expected[case_id],
                )
                record = json.loads(_bytes(saved["record"]))
                artifact_path = (
                    None
                    if normalized["artifact"] is None
                    else normalized["artifact"]["path"]
                )
            if artifact_path is not None:
                if artifact_path in artifact_paths:
                    raise ValueError(f"{stage.title()} records share a raw artifact")
                artifact_paths.add(artifact_path)
            found[case_id] = record
        return found, artifact_paths

    def _publish_record(
        self,
        stage: str,
        case_id: str,
        record: dict[str, Any],
        binding: dict[str, str],
    ) -> None:
        directory = self._stage_directory(stage)
        path = directory / f"{case_id}.json"
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError(f"{stage.title()} record must be a regular file")
        _publish(path, {**binding, "record": record})

    def train_chunk(self, execute, limit: int) -> dict[str, Any]:
        """Append terminal training records in declared order without replay."""
        self._check_limit(limit)
        with self._locked():
            if (
                (self.root / "thresholds.json").exists()
                or (self.root / "validation_release.json").exists()
                or any((self.root / "validation").glob("*.json"))
            ):
                raise ValueError("Training is frozen")
            cases = training_cases(self.config)
            binding = {"manifest_sha256": self.manifest_sha256}
            found, artifact_paths = self._saved("training", cases, binding)
            written = 0
            for case in cases:
                case_id = _case_id(case)
                if case_id in found:
                    continue
                record = execute(json.loads(_bytes(case)))
                if _case_id(self._case(record)) != case_id:
                    raise ValueError("Backend returned a different training identity")
                self._check_manifest()
                if (self.root / "thresholds.json").exists():
                    raise ValueError("Backend froze training during acquisition")
                found, artifact_paths = self._saved("training", cases, binding)
                normalized, artifact_path = self._training_record(record, case)
                if artifact_path is not None and artifact_path in artifact_paths:
                    raise ValueError("Training records cannot share a raw artifact")
                self._publish_record("training", case_id, normalized, binding)
                found[case_id] = normalized
                if artifact_path is not None:
                    artifact_paths.add(artifact_path)
                written += 1
                if written == limit:
                    break
            total = len(found)
            return {
                "new_records": written,
                "total_records": total,
                "planned_records": len(cases),
                "complete": total == len(cases),
            }

    def freeze_training(self) -> tuple[dict[str, Any], dict[str, str]]:
        """Freeze thresholds only after all terminal training records recheck."""
        with self._locked():
            if (
                (self.root / "validation_release.json").exists()
                or any((self.root / "validation").glob("*.json"))
            ):
                raise ValueError("Validation already started")
            cases = training_cases(self.config)
            binding = {"manifest_sha256": self.manifest_sha256}
            found, _ = self._saved("training", cases, binding)
            if len(found) != len(cases):
                raise ValueError("Threshold freeze requires all training records")
            if any(row["status"] == "failed" for row in found.values()):
                raise ValueError("Threshold freeze cannot replace a failed training outcome")
            records = []
            for case in cases:
                row = found[_case_id(case)]
                artifact, arrays = _artifact_arrays(
                    self.root, row["artifact"], "training"
                )
                records.append(
                    {
                        "case": case,
                        "status": "completed",
                        "artifact": artifact,
                        "score": _score(self.config, arrays["score"]),
                        "error": None,
                    }
                )
            return freeze_thresholds(
                self.root,
                self.protocol_file,
                self.provenance,
                records,
                self.repository,
            )

    def validate_chunk(self, execute, limit: int) -> dict[str, Any]:
        """Append terminal validation records against the frozen thresholds."""
        self._check_limit(limit)
        with self._locked():
            if (self.root / "validation_release.json").exists():
                raise ValueError("Validation release is frozen")
            _, threshold_reference, thresholds = _threshold_bundle(
                self.root, self.config, self.protocol_sha256, self.provenance
            )
            cases = validation_cases(self.config)
            binding = {
                "manifest_sha256": self.manifest_sha256,
                "threshold_sha256": threshold_reference["sha256"],
            }
            found, artifact_paths = self._saved(
                "validation", cases, binding, thresholds
            )
            written = 0
            for case in cases:
                case_id = _case_id(case)
                if case_id in found:
                    continue
                threshold = thresholds[case["regime_id"]]
                record = execute(json.loads(_bytes(case)), threshold)
                if _case_id(self._case(record)) != case_id:
                    raise ValueError("Backend returned a different validation identity")
                self._check_manifest()
                if (self.root / "validation_release.json").exists():
                    raise ValueError("Backend froze validation during acquisition")
                _, current_reference, current_thresholds = _threshold_bundle(
                    self.root, self.config, self.protocol_sha256, self.provenance
                )
                if current_reference != threshold_reference:
                    raise ValueError("Frozen threshold binding changed")
                found, artifact_paths = self._saved(
                    "validation", cases, binding, current_thresholds
                )
                normalized = _normalize_validation_record(
                    self.root, self.config, current_thresholds, record, case
                )
                artifact_path = (
                    None
                    if normalized["artifact"] is None
                    else normalized["artifact"]["path"]
                )
                if artifact_path is not None and artifact_path in artifact_paths:
                    raise ValueError("Validation records cannot share a raw artifact")
                detached = json.loads(_bytes(record))
                self._publish_record("validation", case_id, detached, binding)
                found[case_id] = detached
                if artifact_path is not None:
                    artifact_paths.add(artifact_path)
                written += 1
                if written == limit:
                    break
            total = len(found)
            return {
                "new_records": written,
                "total_records": total,
                "planned_records": len(cases),
                "complete": total == len(cases),
            }

    def freeze_validation(self) -> tuple[dict[str, Any], dict[str, str]]:
        """Publish the release decision from complete, reverified saved records."""
        with self._locked():
            _, threshold_reference, thresholds = _threshold_bundle(
                self.root, self.config, self.protocol_sha256, self.provenance
            )
            cases = validation_cases(self.config)
            binding = {
                "manifest_sha256": self.manifest_sha256,
                "threshold_sha256": threshold_reference["sha256"],
            }
            found, _ = self._saved("validation", cases, binding, thresholds)
            if len(found) != len(cases):
                raise ValueError("Release requires all validation records")
            records = [found[_case_id(case)] for case in cases]
            return freeze_validation_release(
                self.root,
                self.protocol_file,
                self.provenance,
                records,
                self.repository,
            )
