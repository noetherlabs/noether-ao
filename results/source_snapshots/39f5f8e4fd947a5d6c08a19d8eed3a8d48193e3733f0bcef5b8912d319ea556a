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

from .controller_evidence import save_arrays
from .controller_selection import _bytes, _digest, _provenance, _publish, _read
from .observability_detector import (
    _artifact_arrays,
    _build_threshold_bundle,
    _case_id,
    _file_digest,
    _load_protocol_binding,
    _publish_threshold_bundle,
    _score,
    detector_trace,
    training_cases,
)
from .observability_evaluation import (
    build_evaluation_report,
    evaluation_cases,
    normalize_evaluation_record,
    publish_evaluation_report,
)
from .observability_trials import (
    training_artifact_layout,
    verify_failed_evidence,
    verify_training_evidence,
    verify_training_trace,
)
from .observability_resources import load_resource_inputs
from .observability_validation import (
    _normalize_validation_record,
    _threshold_bundle,
    freeze_validation_release,
    validation_cases,
)


class ObservabilityLedger:
    """One protocol-bound run with resumable, one-way stage records."""

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
        checked_config, base = load_resource_inputs(
            self.protocol_file, repository
        )
        if _bytes(checked_config) != _bytes(self.config):
            raise ValueError("Protocol changed while the pinned base was loaded")
        if not isinstance(base, dict):
            raise ValueError("Pinned base configuration must be a mapping")
        self.base = json.loads(_bytes(base))
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
            "metadata",
            "error",
        }:
            raise ValueError("Invalid training record schema")
        if record["case"] != expected_case:
            raise ValueError("Training record identity changed")
        if record["status"] == "completed":
            if record["error"] is not None:
                raise ValueError("Completed training record cannot contain an error")
            artifact, _, metadata = verify_training_evidence(
                self.root,
                self.config,
                self.base,
                expected_case,
                record["artifact"],
                record["metadata"],
            )
            return {
                "case": expected_case,
                "status": "completed",
                "artifact": artifact,
                "metadata": metadata,
                "error": None,
            }, artifact["path"]
        if record["status"] == "failed":
            if not isinstance(record["error"], str) or not record["error"].strip():
                raise ValueError("Failed training record requires an error")
            artifact, metadata = record["artifact"], record["metadata"]
            if (artifact is None) != (metadata is None):
                raise ValueError("Failed training raw evidence and metadata must be paired")
            if artifact is not None:
                artifact, metadata = verify_failed_evidence(
                    self.root,
                    self.config,
                    self.base,
                    expected_case,
                    artifact,
                    metadata,
                    "training",
                    record["error"],
                )
            return {
                "case": expected_case,
                "status": "failed",
                "artifact": artifact,
                "metadata": metadata,
                "error": record["error"],
            }, None if artifact is None else artifact["path"]
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
            elif stage == "validation":
                assert thresholds is not None
                normalized = _normalize_validation_record(
                    self.root,
                    self.config,
                    self.base,
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
            elif stage == "evaluation":
                assert thresholds is not None
                normalized = normalize_evaluation_record(
                    self.root,
                    self.config,
                    self.base,
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
            else:
                raise ValueError(f"Unknown observability stage: {stage}")
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

    def _check_saved_snapshot(
        self,
        stage: str,
        binding: dict[str, str],
        found: dict[str, dict[str, Any]],
    ) -> None:
        """Recheck immutable bytes after a callback without decoding old arrays."""
        directory = self._stage_directory(stage)
        paths = sorted(directory.glob("*.json"))
        if {path.stem for path in paths} != set(found) or any(
            path.is_symlink() for path in paths
        ):
            raise ValueError(f"{stage.title()} records changed during callback")
        for path in paths:
            record = found[path.stem]
            if path.read_bytes() != _bytes({**binding, "record": record}):
                raise ValueError(f"{stage.title()} record changed during callback")
            for key in ("artifact", "metadata"):
                reference = record.get(key)
                if reference is not None and self.artifact(reference["path"]) != reference:
                    raise ValueError(
                        f"{stage.title()} artifact changed during callback"
                    )

    def _check_threshold_snapshot(
        self, bundle: dict[str, Any], reference: dict[str, str]
    ) -> None:
        path = self.root / reference["path"]
        if (
            path.is_symlink()
            or not path.is_file()
            or _file_digest(path) != reference["sha256"]
        ):
            raise ValueError("Frozen threshold binding changed")
        for row in bundle["training_records"]:
            for key in ("artifact", "detector_artifact", "metadata"):
                retained = row.get(key)
                if retained is not None and self.artifact(retained["path"]) != retained:
                    raise ValueError("Frozen training evidence changed during callback")

    def _snapshot_paths(
        self, relatives: list[str], label: str
    ) -> dict[str, tuple[int, int, int, int, int]]:
        """Capture cheap post-hash file identities for callback-time drift checks."""
        snapshot: dict[str, tuple[int, int, int, int, int]] = {}
        for relative in relatives:
            if not isinstance(relative, str):
                raise ValueError(f"Invalid or duplicate {label} snapshot path")
            path = Path(relative)
            if (
                path.is_absolute()
                or path.as_posix() != relative
                or ".." in path.parts
                or relative in snapshot
            ):
                raise ValueError(f"Invalid or duplicate {label} snapshot path")
            target = self.root / path
            for current in (target, *target.parents):
                if current == self.root:
                    break
                if current.is_symlink():
                    raise ValueError(f"{label} snapshot path cannot contain a symlink")
            if not target.is_file():
                raise ValueError(f"Missing {label} snapshot file")
            state = target.stat()
            snapshot[relative] = (
                state.st_dev,
                state.st_ino,
                state.st_size,
                state.st_mtime_ns,
                state.st_ctime_ns,
            )
        return snapshot

    def _check_path_snapshot(
        self,
        snapshot: dict[str, tuple[int, int, int, int, int]],
        label: str,
    ) -> None:
        if self._snapshot_paths(list(snapshot), label) != snapshot:
            raise ValueError(f"{label} changed during callback")

    @staticmethod
    def _record_reference_paths(record: dict[str, Any]) -> list[str]:
        paths = []
        for key in ("artifact", "metadata"):
            reference = record.get(key)
            if isinstance(reference, dict) and isinstance(reference.get("path"), str):
                paths.append(reference["path"])
        return paths

    def _evaluation_stage_snapshot(
        self, found: dict[str, dict[str, Any]]
    ) -> dict[str, tuple[int, int, int, int, int]]:
        paths = [f"evaluation/{case_id}.json" for case_id in found]
        for record in found.values():
            paths.extend(self._record_reference_paths(record))
        return self._snapshot_paths(paths, "Evaluation evidence")

    def _check_evaluation_stage_snapshot(
        self,
        found: dict[str, dict[str, Any]],
        snapshot: dict[str, tuple[int, int, int, int, int]],
    ) -> None:
        directory = self._stage_directory("evaluation")
        records = sorted(directory.glob("*.json"))
        if {path.stem for path in records} != set(found) or any(
            path.is_symlink() for path in records
        ):
            raise ValueError("Evaluation records changed during callback")
        self._check_path_snapshot(snapshot, "Evaluation evidence")

    def _check_validation_release_snapshot(
        self,
        threshold_bundle: dict[str, Any],
        threshold_reference: dict[str, str],
        release_bundle: dict[str, Any],
        release_reference: dict[str, str],
    ) -> dict[str, tuple[int, int, int, int, int]]:
        """Recheck the complete upstream chain after evaluation work."""
        references = [threshold_reference["path"], release_reference["path"]]
        for row in threshold_bundle["training_records"]:
            for key in ("artifact", "detector_artifact", "metadata"):
                retained = row.get(key)
                if retained is not None:
                    references.append(retained["path"])
        for row in release_bundle["validation_records"]:
            for key in ("artifact", "metadata"):
                retained = row.get(key)
                if retained is not None:
                    references.append(retained["path"])
        snapshot = self._snapshot_paths(references, "Frozen upstream evidence")
        self._check_threshold_snapshot(threshold_bundle, threshold_reference)
        path = self.root / release_reference["path"]
        if (
            path.is_symlink()
            or not path.is_file()
            or path.read_bytes() != _bytes(release_bundle)
            or _file_digest(path) != release_reference["sha256"]
        ):
            raise ValueError("Frozen validation release changed")
        for row in release_bundle["validation_records"]:
            for key in ("artifact", "metadata"):
                reference = row.get(key)
                if reference is not None and self.artifact(reference["path"]) != reference:
                    raise ValueError("Frozen validation evidence changed during callback")
        self._check_path_snapshot(snapshot, "Frozen upstream evidence")
        return snapshot

    def _evaluation_gate(
        self,
    ) -> tuple[
        dict[str, str],
        dict[str, float],
        dict[str, str],
        dict[str, tuple[int, int, int, int, int]],
    ]:
        """Rebuild and verify the published validation release without opening P2."""
        release_path = self.root / "validation_release.json"
        if release_path.is_symlink() or not release_path.is_file():
            raise ValueError("Evaluation requires a frozen validation release")
        threshold_bundle, threshold_reference, thresholds = _threshold_bundle(
            self.root,
            self.config,
            self.base,
            self.protocol_sha256,
            self.provenance,
        )
        validation = validation_cases(self.config)
        validation_binding = {
            "manifest_sha256": self.manifest_sha256,
            "threshold_sha256": threshold_reference["sha256"],
        }
        found, _ = self._saved(
            "validation", validation, validation_binding, thresholds
        )
        if len(found) != len(validation):
            raise ValueError("Evaluation requires all validation records")
        records = [found[_case_id(case)] for case in validation]
        release_bundle, release_reference = freeze_validation_release(
            self.root,
            self.protocol_file,
            self.provenance,
            records,
            self.repository,
        )
        if not release_bundle["evaluation_released"]:
            raise ValueError("Validation withheld evaluation")
        upstream_snapshot = self._check_validation_release_snapshot(
            threshold_bundle,
            threshold_reference,
            release_bundle,
            release_reference,
        )
        return (
            threshold_reference,
            thresholds,
            release_reference,
            upstream_snapshot,
        )

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
                self._check_saved_snapshot("training", binding, found)
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
        """Freeze thresholds and derived traces after all raw records recheck."""
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
            retained: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
            for case in cases:
                case_id = _case_id(case)
                row = found[case_id]
                artifact, arrays = _artifact_arrays(
                    self.root, row["artifact"], "training"
                )
                retained[case_id] = (arrays, row["metadata"])
                records.append(
                    {
                        "case": case,
                        "status": "completed",
                        "artifact": artifact,
                        "score": _score(self.config, arrays["score"]),
                        "error": None,
                    }
                )
            bundle = _build_threshold_bundle(
                self.config, self.protocol_sha256, self.provenance, records
            )
            thresholds = {
                row["regime"]["id"]: float(row["threshold"])
                for row in bundle["thresholds"]
            }
            rows = {row["id"]: row for row in bundle["training_records"]}
            for case in cases:
                case_id = _case_id(case)
                arrays, metadata = retained[case_id]
                trace = detector_trace(
                    self.config, arrays["score"], thresholds[case["regime_id"]]
                )
                detector_artifact = save_arrays(
                    self,
                    f"evidence/training/detectors/{case_id}.npz",
                    trace,
                )
                rows[case_id]["detector_artifact"] = verify_training_trace(
                    self.root,
                    self.config,
                    case,
                    arrays["score"],
                    thresholds[case["regime_id"]],
                    detector_artifact,
                )
                rows[case_id]["metadata"] = metadata
            bundle["schema_version"] = 2
            bundle["training_artifact_layout"] = training_artifact_layout(
                self.config
            )
            bundle["training_manifest_sha256"] = _digest(
                _bytes(bundle["training_records"])
            )
            if _file_digest(self.protocol_file) != self.protocol_sha256:
                raise ValueError("Protocol changed before threshold publication")
            return bundle, _publish_threshold_bundle(self.root, bundle)

    def validate_chunk(self, execute, limit: int) -> dict[str, Any]:
        """Append terminal validation records against the frozen thresholds."""
        self._check_limit(limit)
        with self._locked():
            if (self.root / "validation_release.json").exists():
                raise ValueError("Validation release is frozen")
            threshold_bundle, threshold_reference, thresholds = _threshold_bundle(
                self.root, self.config, self.base, self.protocol_sha256, self.provenance
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
                self._check_threshold_snapshot(threshold_bundle, threshold_reference)
                self._check_saved_snapshot("validation", binding, found)
                normalized = _normalize_validation_record(
                    self.root,
                    self.config,
                    self.base,
                    thresholds,
                    record,
                    case,
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
                self.root, self.config, self.base, self.protocol_sha256, self.provenance
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

    def evaluate_chunk(self, execute, limit: int) -> dict[str, Any]:
        """Append released evaluation records in declared order without replay."""
        self._check_limit(limit)
        with self._locked():
            report_path = self.root / "evaluation_report.json"
            if report_path.exists() or report_path.is_symlink():
                raise ValueError("Evaluation report is frozen")
            (
                threshold_reference,
                thresholds,
                release_reference,
                upstream_snapshot,
            ) = self._evaluation_gate()
            cases = evaluation_cases(self.config)
            binding = {
                "manifest_sha256": self.manifest_sha256,
                "threshold_sha256": threshold_reference["sha256"],
                "validation_release_sha256": release_reference["sha256"],
            }
            found, artifact_paths = self._saved(
                "evaluation", cases, binding, thresholds
            )
            evaluation_snapshot = self._evaluation_stage_snapshot(found)
            self._check_saved_snapshot("evaluation", binding, found)
            self._check_evaluation_stage_snapshot(found, evaluation_snapshot)
            written = 0
            for case in cases:
                case_id = _case_id(case)
                if case_id in found:
                    continue
                threshold = thresholds[case["regime_id"]]
                record = execute(json.loads(_bytes(case)), threshold)
                if _case_id(self._case(record)) != case_id:
                    raise ValueError("Backend returned a different evaluation identity")
                self._check_manifest()
                if report_path.exists() or report_path.is_symlink():
                    raise ValueError("Backend froze evaluation during acquisition")
                self._check_path_snapshot(
                    upstream_snapshot, "Frozen upstream evidence"
                )
                self._check_evaluation_stage_snapshot(found, evaluation_snapshot)
                new_snapshot = self._snapshot_paths(
                    self._record_reference_paths(record), "New evaluation evidence"
                )
                normalized = normalize_evaluation_record(
                    self.root,
                    self.config,
                    self.base,
                    thresholds,
                    record,
                    case,
                )
                self._check_path_snapshot(new_snapshot, "New evaluation evidence")
                artifact_path = (
                    None
                    if normalized["artifact"] is None
                    else normalized["artifact"]["path"]
                )
                if artifact_path is not None and artifact_path in artifact_paths:
                    raise ValueError("Evaluation records cannot share a raw artifact")
                detached = json.loads(_bytes(record))
                self._publish_record("evaluation", case_id, detached, binding)
                found[case_id] = detached
                if artifact_path is not None:
                    artifact_paths.add(artifact_path)
                evaluation_snapshot.update(new_snapshot)
                evaluation_snapshot.update(
                    self._snapshot_paths(
                        [f"evaluation/{case_id}.json"], "Evaluation record"
                    )
                )
                self._check_evaluation_stage_snapshot(found, evaluation_snapshot)
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

    def freeze_evaluation_report(self) -> tuple[dict[str, Any], dict[str, str]]:
        """Publish one immutable descriptive report from the complete cohort."""
        with self._locked():
            (
                threshold_reference,
                thresholds,
                release_reference,
                upstream_snapshot,
            ) = self._evaluation_gate()
            cases = evaluation_cases(self.config)
            binding = {
                "manifest_sha256": self.manifest_sha256,
                "threshold_sha256": threshold_reference["sha256"],
                "validation_release_sha256": release_reference["sha256"],
            }
            found, _ = self._saved("evaluation", cases, binding, thresholds)
            if len(found) != len(cases):
                raise ValueError("Report requires all evaluation records")
            evaluation_snapshot = self._evaluation_stage_snapshot(found)
            records = [found[_case_id(case)] for case in cases]
            report = build_evaluation_report(
                self.root,
                self.config,
                self.base,
                self.manifest_sha256,
                self.protocol_sha256,
                self.provenance,
                threshold_reference,
                release_reference,
                thresholds,
                records,
            )
            self._check_manifest()
            self._check_path_snapshot(upstream_snapshot, "Frozen upstream evidence")
            self._check_evaluation_stage_snapshot(found, evaluation_snapshot)
            return report, publish_evaluation_report(self.root, report)
