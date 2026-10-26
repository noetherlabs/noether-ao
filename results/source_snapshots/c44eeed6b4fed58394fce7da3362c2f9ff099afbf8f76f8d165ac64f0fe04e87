"""Physical trial adapter and retained-evidence contract for P2 observability.

Training raw streams are acquired before their regime threshold exists.  They are
therefore retained separately from the threshold-derived detector trace, which is
published by the one-way training freeze.  Validation already has a frozen threshold
and retains the complete frame contract in one numeric archive.

This module imports no JAX or simulator module at import time.  Physical construction
starts only inside a ledger-admitted callback.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import re
import shutil
from typing import Any

import numpy as np

from .controller_evidence import (
    ROOT,
    _runtime,
    _source,
    save_arrays,
    save_metadata,
)
from .controller_selection import _bytes, _digest, _read
from .observability_detector import (
    _artifact_arrays,
    _case_id,
    _file_digest,
    detector_trace,
)
from .observability_protocol import protocol_plan, validate_protocol
from .observability_resources import (
    estimate_observability_resources,
    load_resource_inputs,
    resource_preflight,
)


TRACE_ARRAYS = (
    "statistic_frame_id",
    "lower_statistic",
    "raw_crossing",
    "alarm_frame_id",
    "p2_alarm",
)
CORE_ARRAYS = (
    "score",
    "variance",
    "strehl",
    "saturated",
    "fallback",
    "unavailable",
    "command",
    "applied",
)


def capture_observability_inputs(
    protocol_path: Path, repository: Path | None = None
) -> dict[str, Any]:
    """Capture clean source/runtime bindings without constructing physical data."""
    repository = Path(repository or ROOT).resolve()
    source = _source(repository)
    path = Path(protocol_path)
    path = path if path.is_absolute() else repository / path
    path = path.resolve()
    if (
        not path.is_relative_to(repository)
        or path.relative_to(repository).as_posix() not in source["files"]
    ):
        raise ValueError("Protocol must be a tracked execution input")
    protocol, base = load_resource_inputs(path, repository)
    base_path = (repository / protocol["base_config"]).resolve()
    if (
        not base_path.is_relative_to(repository)
        or base_path.relative_to(repository).as_posix() not in source["files"]
    ):
        raise ValueError("Base must be a tracked execution input")
    runtime = _runtime()
    if not runtime["machine"]["x64"]:
        raise ValueError("The observability study requires JAX x64")
    if _source(repository) != source:
        raise ValueError("Source changed during observability provenance capture")
    return {
        "protocol": protocol,
        "base": base,
        "source": source,
        "runtime": runtime,
        "provenance": {
            "source_commit": source["commit"],
            "source_sha256": _digest(_bytes(source)),
            "runtime_sha256": _digest(_bytes(runtime)),
        },
    }


def _provenance_references(ledger) -> list[dict[str, str]]:
    return [
        ledger.artifact(f"evidence/provenance/{name}.json")
        for name in ("source", "runtime", "protocol", "base")
    ]


def verify_observability_provenance(ledger, base: dict[str, Any]) -> None:
    """Recheck persisted source/runtime/protocol/base evidence before physics."""
    references = {ref["path"]: ref for ref in _provenance_references(ledger)}
    prefix = "evidence/provenance/"
    source = _read(ledger.root / (prefix + "source.json"))
    runtime = _read(ledger.root / (prefix + "runtime.json"))
    protocol = _read(ledger.root / (prefix + "protocol.json"))
    saved_base = _read(ledger.root / (prefix + "base.json"))
    if (
        _digest(_bytes(source)) != ledger.provenance["source_sha256"]
        or _digest(_bytes(runtime)) != ledger.provenance["runtime_sha256"]
        or source.get("commit") != ledger.provenance["source_commit"]
    ):
        raise ValueError("Stored source/runtime evidence differs from the run binding")
    if _bytes(protocol) != _bytes(ledger.config) or _bytes(saved_base) != _bytes(base):
        raise ValueError("Stored protocol/base evidence differs from the run binding")
    if set(references) != {
        prefix + "source.json",
        prefix + "runtime.json",
        prefix + "protocol.json",
        prefix + "base.json",
    }:
        raise ValueError("Incomplete observability provenance evidence")


def open_observability_run(
    root: Path,
    protocol_path: Path = Path("configs/optical_gain_observability_v1.yaml"),
    *,
    create: bool = False,
    repository: Path | None = None,
):
    """Create or resume a storage-admitted run bound to clean source/runtime."""
    repository = Path(repository or ROOT).resolve()
    root = Path(root).expanduser().resolve()
    if create:
        protocol, base = load_resource_inputs(protocol_path, repository)
        admission = resource_preflight(protocol, base, root)
        if not admission["admitted"]:
            raise RuntimeError("Insufficient free disk for the observability study")
    inputs = capture_observability_inputs(protocol_path, repository)
    from .observability_run import ObservabilityLedger

    factory = ObservabilityLedger.create if create else ObservabilityLedger
    ledger = factory(root, protocol_path, inputs["provenance"], repository)
    with ledger._locked():
        for name in ("source", "runtime", "protocol", "base"):
            save_metadata(
                ledger,
                f"evidence/provenance/{name}.json",
                inputs[name],
            )
    if _source(repository) != inputs["source"]:
        raise ValueError("Source changed while the observability run was opened")
    verify_observability_provenance(ledger, inputs["base"])
    return ledger, inputs["base"], _provenance_references(ledger)


def _expected_case(config: dict[str, Any], case: Any) -> dict[str, Any]:
    validate_protocol(config)
    keys = {
        "role",
        "condition",
        "regime_id",
        "photons",
        "wind_scale",
        "atmosphere_seed",
        "detector_seed",
    }
    if (
        not isinstance(case, dict)
        or set(case) != keys
        or case.get("role") not in {
            "training",
            "validation",
            "evaluation",
        }
    ):
        raise ValueError("Unknown observability trial role")
    plan = protocol_plan(config)
    regimes = [row for row in plan["regimes"] if row["id"] == case["regime_id"]]
    streams = [
        row
        for row in plan["seed_streams"][case["role"]]
        if row["atmosphere"] == case["atmosphere_seed"]
        and row["detector"] == case["detector_seed"]
    ]
    if (
        len(regimes) != 1
        or len(streams) != 1
        or case["condition"] not in config["conditions"][case["role"]]
        or case["photons"] != regimes[0]["photons"]
        or case["wind_scale"] != regimes[0]["wind_scale"]
    ):
        raise ValueError("Foreign observability trial identity")
    return json.loads(_bytes(case))


def training_artifact_layout(config: dict[str, Any]) -> dict[str, Any]:
    """Return the canonical split necessitated by post-cohort calibration."""
    validate_protocol(config)
    return {
        "raw_before_threshold": [
            key
            for key in config["retention"]["all_trial_frame_arrays"]
            if key not in set(TRACE_ARRAYS)
        ],
        "detector_after_threshold": list(TRACE_ARRAYS),
        "per_run_metrics": config["retention"]["all_trial_metrics"],
    }


def retain_full_wfs(config: dict[str, Any], case: dict[str, Any]) -> bool:
    """Return the frozen first-seed retention decision for one declared case."""
    checked = _expected_case(config, case)
    if config["retention"]["full_wfs_policy"] != "first_seed_per_role_regime_condition":
        raise ValueError("Unsupported full-WFS retention policy")
    return checked["atmosphere_seed"] == config["seeds"][checked["role"]][0]


def trial_config(
    base: dict[str, Any], config: dict[str, Any], case: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    """Apply only frozen P2 overrides and return the simulator fault identity."""
    checked = _expected_case(config, case)
    if not isinstance(base, dict) or base.get("sensor") != "pyramid":
        raise ValueError("Observability trials require the pinned pyramid base")
    cfg = deepcopy(base)
    cfg.pop("seed", None)
    cfg.update(
        frames=config["loop"]["frames"],
        burn_in=config["loop"]["burn_in"],
        photons=checked["photons"],
        wind_scale=checked["wind_scale"],
        gain=config["controller"]["gain"],
        leak=config["controller"]["leak"],
    )
    cfg["seeds"] = list(config["seeds"][checked["role"]])
    cfg["training_seeds"] = list(config["seeds"]["training"])
    cfg["faults"].update(
        onset=config["loop"]["fault_onset"],
        end=config["loop"]["fault_end"],
        drift_frames=config["loop"]["optical_gain_drift_frames"],
        optical_gain=config["faults"]["gain_035"]["terminal_response_multiplier"],
        vibration_rad=config["faults"]["vibration"]["amplitude_rad"],
        vibration_hz=config["faults"]["vibration"]["frequency_hz"],
        shift_pixels=config["faults"]["misregistration"]["shift_pixels"],
    )
    condition = checked["condition"]
    if condition == "clean":
        fault = "none"
    else:
        declaration = config["faults"].get(condition)
        if declaration is None:
            raise ValueError("Undeclared observability condition")
        fault = declaration["kind"]
        if condition in {"gain_035", "gain_065"}:
            cfg["faults"]["optical_gain"] = declaration[
                "terminal_response_multiplier"
            ]
        elif condition == "vibration":
            cfg["faults"].update(
                vibration_rad=declaration["amplitude_rad"],
                vibration_hz=declaration["frequency_hz"],
            )
        elif condition == "misregistration":
            cfg["faults"]["shift_pixels"] = declaration["shift_pixels"]
    return cfg, fault


def _float_array(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "fiu":
        raise ValueError(f"Physical {label} has an incompatible shape or dtype")
    with np.errstate(over="ignore", invalid="ignore"):
        return np.array(array, dtype=np.float64, copy=True, order="C")


def _bool_array(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or array.dtype != np.dtype(bool):
        raise ValueError(f"Physical {label} must be a boolean frame array")
    return np.array(array, dtype=bool, copy=True, order="C")


def physical_core_arrays(
    config: dict[str, Any],
    base: dict[str, Any],
    case: dict[str, Any],
    outputs: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Normalize simulator outputs without hiding numerical failures."""
    checked = _expected_case(config, case)
    if not isinstance(outputs, dict):
        raise ValueError("Physical trial outputs must be a mapping")
    frames, modes = config["loop"]["frames"], base["modes"]
    required = set(CORE_ARRAYS)
    if not required.issubset(outputs):
        raise ValueError(f"Physical trial is missing outputs: {sorted(required - set(outputs))}")
    arrays = {
        "score": _float_array(outputs["score"], (frames,), "score"),
        "variance": _float_array(outputs["variance"], (frames,), "variance"),
        "strehl": _float_array(outputs["strehl"], (frames,), "strehl"),
        "saturated": _bool_array(outputs["saturated"], (frames,), "saturated"),
        "fallback": _bool_array(outputs["fallback"], (frames,), "fallback"),
        "unavailable": _bool_array(outputs["unavailable"], (frames,), "unavailable"),
        "command": _float_array(outputs["command"], (frames, modes), "command"),
        "applied": _float_array(outputs["applied"], (frames, modes), "applied"),
    }
    if retain_full_wfs(config, checked):
        detector_n = base["detector_n"]
        for key in config["retention"]["full_wfs_arrays"]:
            if key not in outputs:
                raise ValueError(f"Full-WFS trial is missing {key}")
            value = np.asarray(outputs[key])
            flat = (frames, detector_n * detector_n)
            cube = (frames, detector_n, detector_n)
            if value.shape == flat:
                value = value.reshape(cube)
            arrays[key] = _float_array(value, cube, key)
    return arrays


def numerical_failure(arrays: dict[str, np.ndarray]) -> str | None:
    """Classify produced but unusable physical streams as retained outcomes."""
    if any(not np.isfinite(value).all() for value in arrays.values()):
        return "Nonfinite physical trial output"
    if np.any(arrays["score"] < 0):
        return "Negative detector score in physical trial output"
    if np.any(arrays["variance"] < 0):
        return "Negative residual variance in physical trial output"
    return None


def complete_trial_arrays(
    config: dict[str, Any],
    base: dict[str, Any],
    case: dict[str, Any],
    outputs: dict[str, Any],
    threshold: float,
) -> dict[str, np.ndarray]:
    """Return the exact validation archive, including the frozen detector trace."""
    core = physical_core_arrays(config, base, case, outputs)
    failure = numerical_failure(core)
    if failure is not None:
        raise FloatingPointError(failure)
    return _complete_from_core(config, core, threshold)


def _complete_from_core(
    config: dict[str, Any], core: dict[str, np.ndarray], threshold: float
) -> dict[str, np.ndarray]:
    trace = detector_trace(config, core["score"], threshold)
    return {name: (trace[name] if name in trace else core[name])
            for name in config["retention"]["all_trial_frame_arrays"]} | {
                name: core[name] for name in config["retention"]["full_wfs_arrays"]
                if name in core
            }


def trial_metrics(
    config: dict[str, Any], base: dict[str, Any], arrays: dict[str, np.ndarray]
) -> dict[str, float]:
    """Compute the retained post-burn-in scalars from frame arrays."""
    start, stop = config["loop"]["burn_in"], config["loop"]["frames"]
    variance = np.asarray(arrays["variance"])[start:stop]
    strehl = np.asarray(arrays["strehl"])[start:stop]
    saturated = np.asarray(arrays["saturated"])[start:stop]
    if (
        variance.shape != (stop - start,)
        or strehl.shape != variance.shape
        or saturated.shape != variance.shape
        or saturated.dtype != np.dtype(bool)
        or not np.isfinite(variance).all()
        or not np.isfinite(strehl).all()
        or np.any(variance < 0)
    ):
        raise ValueError("Cannot derive finite post-burn-in trial metrics")
    metrics = {
        "wfe_nm": float(
            np.sqrt(variance.mean()) * base["wavelength_m"] / (2 * np.pi) * 1e9
        ),
        "strehl": float(strehl.mean()),
        "saturation_fraction": float(saturated.mean()),
    }
    if list(metrics) != config["retention"]["all_trial_metrics"] or not all(
        math.isfinite(value) for value in metrics.values()
    ):
        raise ValueError("Invalid retained trial metrics")
    return metrics


def trial_metadata(
    config: dict[str, Any],
    base: dict[str, Any],
    case: dict[str, Any],
    raw: dict[str, str],
    arrays: dict[str, np.ndarray],
    *,
    status: str = "completed",
    error: str | None = None,
) -> dict[str, Any]:
    """Build the canonical per-run metadata envelope."""
    checked = _expected_case(config, case)
    if status == "completed" and error is None:
        metrics = trial_metrics(config, base, arrays)
    elif status == "failed" and isinstance(error, str) and error.strip():
        metrics = None
    else:
        raise ValueError("Invalid trial metadata outcome")
    return {
        "schema_version": 1,
        "kind": "optical_gain_observability_trial_metadata",
        "case": checked,
        "status": status,
        "raw": raw,
        "metric_interval": {
            "name": "post_burn_in",
            "start_frame": config["loop"]["burn_in"],
            "stop_frame_exclusive": config["loop"]["frames"],
        },
        "metrics": metrics,
        "error": error,
    }


def _metadata_artifact(
    root: Path, reference: Any, role: str
) -> tuple[dict[str, str], dict[str, Any]]:
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise ValueError(f"{role.title()} record requires one metadata reference")
    path, digest = reference["path"], reference["sha256"]
    if (
        not isinstance(path, str)
        or not path.startswith(f"evidence/{role}/")
        or Path(path).is_absolute()
        or Path(path).as_posix() != path
        or ".." in Path(path).parts
        or Path(path).suffix != ".json"
        or not isinstance(digest, str)
        or not re.fullmatch(r"[0-9a-f]{64}", digest)
    ):
        raise ValueError(f"{role.title()} metadata must be a canonical hashed JSON reference")
    target = root / path
    for current in (target, *target.parents):
        if current == root:
            break
        if current.is_symlink():
            raise ValueError(f"{role.title()} metadata path cannot contain a symlink")
    if not target.is_file() or not target.resolve().is_relative_to(root / "evidence" / role):
        raise ValueError(f"{role.title()} metadata is missing or outside evidence/{role}")
    if _file_digest(target) != digest:
        raise ValueError(f"{role.title()} metadata hash changed")
    try:
        value = json.loads(target.read_bytes())
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f"{role.title()} metadata is not valid JSON") from exc
    if target.read_bytes() != _bytes(value) or _file_digest(target) != digest:
        raise ValueError(f"{role.title()} metadata is not canonical or changed while read")
    return {"path": path, "sha256": digest}, value


def verify_trial_metadata(
    root: Path,
    config: dict[str, Any],
    base: dict[str, Any],
    case: dict[str, Any],
    raw: dict[str, str],
    arrays: dict[str, np.ndarray],
    reference: Any,
    role: str,
    *,
    status: str = "completed",
    error: str | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    artifact, saved = _metadata_artifact(root, reference, role)
    expected = trial_metadata(
        config, base, case, raw, arrays, status=status, error=error
    )
    if saved != expected:
        raise ValueError(f"{role.title()} trial metadata differs from retained arrays")
    return artifact, saved


def _expected_keys(
    config: dict[str, Any], case: dict[str, Any], *, complete: bool
) -> set[str]:
    names = (
        set(config["retention"]["all_trial_frame_arrays"])
        if complete
        else set(CORE_ARRAYS)
    )
    if retain_full_wfs(config, case):
        names.update(config["retention"]["full_wfs_arrays"])
    return names


def _verify_array_contract(
    config: dict[str, Any],
    base: dict[str, Any],
    case: dict[str, Any],
    arrays: dict[str, np.ndarray],
    *,
    complete: bool,
    require_finite: bool,
) -> None:
    expected = _expected_keys(config, case, complete=complete)
    if set(arrays) != expected:
        difference = sorted(set(arrays) ^ expected)
        raise ValueError(f"Retained trial arrays differ from contract: {difference}")
    frames, modes, detector_n = (
        config["loop"]["frames"],
        base["modes"],
        base["detector_n"],
    )
    statistics = (
        frames - config["loop"]["burn_in"] - config["detector"]["window_frames"] + 1
    )
    alarms = statistics - config["detector"]["persistence_windows"] + 1
    float_shapes = {
        "score": (frames,),
        "variance": (frames,),
        "strehl": (frames,),
        "command": (frames, modes),
        "applied": (frames, modes),
        "lower_statistic": (statistics,),
        "measurement": (frames, detector_n, detector_n),
        "prediction": (frames, detector_n, detector_n),
    }
    bool_shapes = {
        "saturated": (frames,),
        "fallback": (frames,),
        "unavailable": (frames,),
        "raw_crossing": (statistics,),
        "p2_alarm": (alarms,),
    }
    int_shapes = {
        "statistic_frame_id": (statistics,),
        "alarm_frame_id": (alarms,),
    }
    for key, shape in float_shapes.items():
        if key in arrays and (
            arrays[key].shape != shape or arrays[key].dtype != np.dtype(np.float64)
        ):
            raise ValueError(f"Retained {key} must be float64 with shape {shape}")
    for key, shape in bool_shapes.items():
        if key in arrays and (arrays[key].shape != shape or arrays[key].dtype != np.dtype(bool)):
            raise ValueError(f"Retained {key} must be boolean with shape {shape}")
    for key, shape in int_shapes.items():
        if key in arrays and (
            arrays[key].shape != shape or arrays[key].dtype != np.dtype(np.int64)
        ):
            raise ValueError(f"Retained {key} must be int64 with shape {shape}")
    if require_finite and numerical_failure(arrays) is not None:
        raise ValueError("Completed trial contains an invalid numerical output")


def verify_training_evidence(
    root: Path,
    config: dict[str, Any],
    base: dict[str, Any],
    case: dict[str, Any],
    artifact: Any,
    metadata: Any,
) -> tuple[dict[str, str], dict[str, np.ndarray], dict[str, str]]:
    raw, arrays = _artifact_arrays(root, artifact, "training")
    _verify_array_contract(config, base, case, arrays, complete=False, require_finite=True)
    detail, _ = verify_trial_metadata(
        root, config, base, case, raw, arrays, metadata, "training"
    )
    return raw, arrays, detail


def verify_validation_evidence(
    root: Path,
    config: dict[str, Any],
    base: dict[str, Any],
    case: dict[str, Any],
    artifact: Any,
    metadata: Any,
    threshold: float,
) -> tuple[dict[str, str], dict[str, np.ndarray], dict[str, str], dict[str, Any]]:
    raw, arrays = _artifact_arrays(root, artifact, "validation")
    _verify_array_contract(config, base, case, arrays, complete=True, require_finite=True)
    trace = detector_trace(config, arrays["score"], threshold)
    for key in TRACE_ARRAYS:
        if not np.array_equal(arrays[key], trace[key]):
            raise ValueError(f"Stored {key} differs from the frozen detector")
    detail, saved = verify_trial_metadata(
        root, config, base, case, raw, arrays, metadata, "validation"
    )
    return raw, arrays, detail, saved


def verify_failed_evidence(
    root: Path,
    config: dict[str, Any],
    base: dict[str, Any],
    case: dict[str, Any],
    artifact: Any,
    metadata: Any,
    role: str,
    error: str,
) -> tuple[dict[str, str], dict[str, str]]:
    raw, arrays = _artifact_arrays(root, artifact, role)
    _verify_array_contract(config, base, case, arrays, complete=False, require_finite=False)
    detail, _ = verify_trial_metadata(
        root,
        config,
        base,
        case,
        raw,
        arrays,
        metadata,
        role,
        status="failed",
        error=error,
    )
    return raw, detail


def verify_training_trace(
    root: Path,
    config: dict[str, Any],
    case: dict[str, Any],
    score: np.ndarray,
    threshold: float,
    reference: Any,
) -> dict[str, str]:
    artifact, arrays = _artifact_arrays(root, reference, "training")
    expected = detector_trace(config, score, threshold)
    if set(arrays) != set(TRACE_ARRAYS):
        raise ValueError("Training detector artifact has an invalid array contract")
    for key in TRACE_ARRAYS:
        if not np.array_equal(arrays[key], expected[key]):
            raise ValueError(f"Training detector artifact differs at {key}")
    return artifact


def _require_storage_admission(ledger, base: dict[str, Any]) -> None:
    """Preserve the preflight capacity as free bytes become retained evidence."""
    if _bytes(base) != _bytes(ledger.base):
        raise ValueError("Physical base differs from the protocol-pinned base")
    allocated = 0
    inodes: set[tuple[int, int]] = set()
    for path in ledger.root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Observability run cannot contain symlinks")
        if (
            path.is_file()
            and not path.name.startswith(".pending-")
            and (path.suffix in {".json", ".npz"} or path.name == ".lock")
        ):
            stat = path.stat()
            identity = (stat.st_dev, stat.st_ino)
            if identity in inodes:
                continue
            inodes.add(identity)
            allocated += getattr(stat, "st_blocks", (stat.st_size + 511) // 512) * 512
    free = shutil.disk_usage(ledger.root).free
    required = estimate_observability_resources(ledger.config, base)["budget"][
        "required_free_bytes"
    ]
    if free + allocated < required:
        raise RuntimeError("Observability run has not passed full storage admission")


def train_trials(ledger, base: dict[str, Any], limit: int) -> dict[str, Any]:
    """Advance physical training only through the ledger's locked stage gate."""
    _require_storage_admission(ledger, base)
    return ledger.train_chunk(_TrialExecutor(ledger, base), limit)


def validate_trials(ledger, base: dict[str, Any], limit: int) -> dict[str, Any]:
    """Advance physical validation only after the threshold gate opens it."""
    _require_storage_admission(ledger, base)
    return ledger.validate_chunk(_TrialExecutor(ledger, base), limit)


class _TrialExecutor:
    def __init__(self, ledger, base: dict[str, Any]):
        if _bytes(base) != _bytes(ledger.base):
            raise ValueError("Physical base differs from the protocol-pinned base")
        self.ledger = ledger
        self.base = deepcopy(base)
        self.calibration = None
        budget = estimate_observability_resources(ledger.config, self.base)["budget"]
        self.per_trial_floor = (
            budget["largest_single_archive_envelope_bytes"]
            + budget["temporary_write_headroom_bytes"]
        )

    def __call__(self, case: dict[str, Any], threshold: float | None = None) -> dict[str, Any]:
        role = case.get("role")
        if (role == "training") != (threshold is None) or role not in {"training", "validation"}:
            raise ValueError("Trial callback received an incompatible stage threshold")
        verify_observability_provenance(self.ledger, self.base)
        if shutil.disk_usage(self.ledger.root).free < self.per_trial_floor:
            raise RuntimeError("Insufficient free disk before observability trial")

        # Numerical imports and construction remain downstream of the ledger gate.
        import jax
        import jax.numpy as jnp
        from .atmosphere import phase_sequence
        from .simulation import calibration, loop

        cfg, fault = trial_config(self.base, self.ledger.config, case)
        if self.calibration is None:
            self.calibration = calibration(self.base)
        phases = phase_sequence(cfg, case["atmosphere_seed"])
        outputs = jax.block_until_ready(
            loop(
                cfg,
                self.calibration,
                phases,
                case["atmosphere_seed"],
                "forward",
                jnp.zeros((cfg["modes"], 2), dtype=jnp.float64),
                threshold=1e30,
                fallback=False,
                fault=fault,
                differentiable_noise=False,
            )
        )
        core = physical_core_arrays(self.ledger.config, self.base, case, outputs)
        case_id = _case_id(case)
        relative = f"evidence/{role}/trials/{case_id}"
        failure = numerical_failure(core)
        if failure is not None:
            raw = save_arrays(self.ledger, relative + ".npz", core)
            detail = save_metadata(
                self.ledger,
                relative + ".json",
                trial_metadata(
                    self.ledger.config,
                    self.base,
                    case,
                    raw,
                    core,
                    status="failed",
                    error=failure,
                ),
            )
            return {
                "case": case,
                "status": "failed",
                "artifact": raw,
                "metadata": detail,
                "error": failure,
            }
        arrays = core if threshold is None else _complete_from_core(
            self.ledger.config, core, threshold
        )
        raw = save_arrays(self.ledger, relative + ".npz", arrays)
        detail = save_metadata(
            self.ledger,
            relative + ".json",
            trial_metadata(self.ledger.config, self.base, case, raw, arrays),
        )
        return {
            "case": case,
            "status": "completed",
            "artifact": raw,
            "metadata": detail,
            "error": None,
        }
