"""Static whole-run storage sizing for the predeclared controller study.

This module does not import JAX, construct the optical plant, access a study
cohort or collect elapsed time.  It converts the frozen case plan and the saved
array schemas into a conservative storage envelope, then compares that envelope
with the target filesystem when requested.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import shutil

import yaml

from .controller_protocol import load_study, study_plan, validate_study

ROOT = Path(__file__).resolve().parents[2]
MIB = 1024**2
GIB = 1024**3
FLOAT64_BYTES = 8
BOOL_BYTES = 1

# Every numeric archive is DEFLATE-compressed one member at a time.  Compression
# is data-dependent, so the budget takes no compression credit.  Twice the raw
# payload plus 4 KiB per member is a deliberately loose envelope for NPY headers,
# DEFLATE expansion, ZIP64 records and directory entries.
ARCHIVE_PAYLOAD_FACTOR = 2
ARCHIVE_MEMBER_ALLOWANCE_BYTES = 4096
METADATA_MINIMUM_BYTES = 512 * MIB
WRITE_HEADROOM_MINIMUM_BYTES = 512 * MIB
OPERATING_RESERVE_MINIMUM_BYTES = 4 * GIB


def load_resource_inputs(path: Path, repository: Path | None = None) -> tuple[dict, dict]:
    """Load the hash-pinned study and base configuration without importing JAX."""
    root = (repository or ROOT).resolve()
    study = load_study(Path(path), repository=root)
    base_bytes = (root / study["base_config"]).read_bytes()
    if hashlib.sha256(base_bytes).hexdigest() != study["base_config_sha256"]:
        raise ValueError("Base simulation configuration hash changed during resource loading")
    base = yaml.safe_load(base_bytes)
    if not isinstance(base, dict):
        raise ValueError("Base simulation configuration must be a mapping")
    return study, base


def _positive_integer(value, label: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _round_up(value: int, quantum: int = MIB) -> int:
    return ((value + quantum - 1) // quantum) * quantum


def _archive_category(name: str, count: int, payload_per_archive: int,
                      members_per_archive: int) -> dict:
    count = _positive_integer(count, name + " archive count")
    payload_per_archive = _positive_integer(payload_per_archive, name + " payload")
    members_per_archive = _positive_integer(members_per_archive, name + " member count")
    numeric = count * payload_per_archive
    per_archive = (ARCHIVE_PAYLOAD_FACTOR * payload_per_archive
                   + ARCHIVE_MEMBER_ALLOWANCE_BYTES * members_per_archive)
    return {
        "name": name,
        "archives": count,
        "members_per_archive": members_per_archive,
        "numeric_payload_bytes": numeric,
        "archive_envelope_bytes": count * per_archive,
        "largest_archive_envelope_bytes": per_archive,
    }


def _full_output_bytes(frames: int, modes: int, detector_n: int) -> int:
    # Three scalar float streams, three modal float streams, two detector float
    # streams and four scalar boolean streams are retained for every frame.
    floats = frames * (3 + 3 * modes + 2 * detector_n**2)
    booleans = frames * 4
    return FLOAT64_BYTES * floats + BOOL_BYTES * booleans


def _calibration_bytes(n: int, modes: int, detector_n: int) -> int:
    # Numeric members returned by simulation.calibration for the pinned pyramid
    # plant: basis, mask, projection, reference, interaction, reconstructor,
    # stuck_patch, twin_modes and preconditioner.  Callables are never serialized.
    values = (2 * modes * n**2 + 2 * n**2 + detector_n**2
              + 2 * modes * detector_n**2 + modes**2 + 1)
    return FLOAT64_BYTES * values


def estimate_study_resources(study: dict, base: dict) -> dict:
    """Return a deterministic upper envelope for one complete planned run.

    The estimate assumes every fitted model succeeds and every selected
    controller reaches evaluation, which maximizes retained numeric evidence.
    Failed candidates and withheld controllers can only reduce trial storage.
    """
    validate_study(study)
    if base.get("sensor") != "pyramid":
        raise ValueError("Controller-study sizing requires the pinned pyramid sensor")
    n = _positive_integer(base.get("n"), "pupil width")
    modes = _positive_integer(base.get("modes"), "mode count")
    detector_n = _positive_integer(base.get("detector_n"), "detector width")
    training_frames = _positive_integer(study["training"]["frames"], "training frames")
    trial_frames = _positive_integer(study["loop"]["frames"], "trial frames")
    learned_steps = _positive_integer(study["training"]["learned_steps"], "learned steps")

    plan = study_plan(study)
    candidates = plan["candidates"]
    predictors = sum(row["controller"] == "predictive" for row in candidates)
    learned = sum(row["controller"] == "learned" for row in candidates)
    training_cases = plan["planned_training_episodes"]
    validation_trials = plan["planned_validation_trials"]
    evaluation_trials = plan["maximum_planned_evaluation_trials"]
    winds = len(study["conditions"]["wind_scales"])
    validation_seeds = len(study["seeds"]["validation"])
    evaluation_seeds = len(study["seeds"]["evaluation"])
    validation_conditions = (len(study["conditions"]["photons"])
                             * winds * len(study["conditions"]["validation_faults"])
                             * validation_seeds)
    evaluation_conditions = (len(study["conditions"]["photons"])
                             * winds * len(study["conditions"]["evaluation_faults"])
                             * evaluation_seeds)

    parameter_values = 3 * 8 + 8 + 8 * 1 + 1  # width-eight learned network
    parameter_bytes = parameter_values * FLOAT64_BYTES
    learned_evaluations = learned_steps + 1  # fixed steps plus final objective
    learned_history_bytes = (
        training_cases * FLOAT64_BYTES
        + (learned_steps + 1) * parameter_bytes
        + learned_steps * (2 * FLOAT64_BYTES + parameter_bytes)
        + FLOAT64_BYTES
        # One late optimizer failure is larger than the successful history.  The
        # envelope retains both this state allowance and final parameters.
        + 2 * parameter_bytes + FLOAT64_BYTES
    )
    learned_history_members = (
        1 + (learned_steps + 1) * 4 + learned_steps * (2 + 4) + 1 + 9
    )
    calibration_bytes = _calibration_bytes(n, modes, detector_n)
    phase_training = training_frames * n**2 * FLOAT64_BYTES
    phase_trial = trial_frames * n**2 * FLOAT64_BYTES
    open_loop_trial = trial_frames * (2 * FLOAT64_BYTES + BOOL_BYTES)
    predictor_history = training_cases * modes * 2 * FLOAT64_BYTES + modes * 2 * FLOAT64_BYTES

    categories = [
        _archive_category("training_pilot_calibration", 1, calibration_bytes, 9),
        _archive_category("training_shared_atmospheres",
                          len(study["seeds"]["training"]) * winds, phase_training, 1),
        _archive_category("training_pilot_outputs", training_cases,
                          _full_output_bytes(training_frames, modes, detector_n), 12),
        _archive_category("training_measured_telemetry", training_cases,
                          training_frames * modes * FLOAT64_BYTES, 1),
        _archive_category("predictor_fit_histories", predictors, predictor_history, 2),
        _archive_category("predictor_parameters", predictors,
                          modes * 2 * FLOAT64_BYTES, 1),
        _archive_category("learned_calibrations", learned, calibration_bytes, 9),
        _archive_category("learned_step_parameters", learned * learned_evaluations,
                          parameter_bytes, 4),
        _archive_category("learned_episode_results",
                          learned * learned_evaluations * training_cases,
                          FLOAT64_BYTES + BOOL_BYTES + parameter_bytes, 6),
        _archive_category("learned_fit_histories", learned,
                          learned_history_bytes, learned_history_members),
        _archive_category("learned_parameters", learned, parameter_bytes, 4),
        _archive_category("validation_shared_atmospheres",
                          validation_seeds * winds, phase_trial, 1),
        _archive_category("validation_open_loops", validation_conditions,
                          open_loop_trial, 3),
        _archive_category("validation_trial_outputs", validation_trials,
                          _full_output_bytes(trial_frames, modes, detector_n), 12),
        _archive_category("evaluation_shared_atmospheres",
                          evaluation_seeds * winds, phase_trial, 1),
        _archive_category("evaluation_open_loops", evaluation_conditions,
                          open_loop_trial, 3),
        _archive_category("evaluation_trial_outputs", evaluation_trials,
                          _full_output_bytes(trial_frames, modes, detector_n), 12),
    ]
    numeric = sum(row["numeric_payload_bytes"] for row in categories)
    archives = sum(row["archive_envelope_bytes"] for row in categories)
    largest = max(row["largest_archive_envelope_bytes"] for row in categories)
    # JSON evidence contains repeated immutable references by design.  Reserve an
    # additional one eighth of all archive envelopes, with a 512 MiB floor, for
    # manifests, hashes, interruption diagnostics and filesystem allocation.
    metadata = max(METADATA_MINIMUM_BYTES, _round_up((archives + 7) // 8))
    retained = archives + metadata
    write_headroom = max(WRITE_HEADROOM_MINIMUM_BYTES, _round_up(largest))
    operating_reserve = max(OPERATING_RESERVE_MINIMUM_BYTES,
                            _round_up((retained + 9) // 10))
    required = retained + write_headroom + operating_reserve
    return {
        "schema_version": 1,
        "kind": "controller_study_resource_budget",
        "study_id": study["study_id"],
        "status": "derived_not_measured",
        "counts": {
            "training_pilot_records": training_cases,
            "fitted_predictor_candidates": predictors,
            "fitted_learned_candidates": learned,
            "validation_trials": validation_trials,
            "maximum_evaluation_trials": evaluation_trials,
            "numeric_archives": sum(row["archives"] for row in categories),
        },
        "budget": {
            "numeric_payload_bytes": numeric,
            "archive_envelope_bytes": archives,
            "metadata_filesystem_envelope_bytes": metadata,
            "retained_evidence_envelope_bytes": retained,
            "largest_single_archive_envelope_bytes": largest,
            "temporary_write_headroom_bytes": write_headroom,
            "operating_reserve_bytes": operating_reserve,
            "required_free_bytes": required,
            "compression_credit_bytes": 0,
        },
        "categories": categories,
        "assumptions": {
            "numeric_float_dtype": "float64",
            "numeric_boolean_bytes": BOOL_BYTES,
            "archive_payload_factor": ARCHIVE_PAYLOAD_FACTOR,
            "archive_member_allowance_bytes": ARCHIVE_MEMBER_ALLOWANCE_BYTES,
            "metadata_filesystem_fraction": "1/8",
            "maximum_selected_controllers": len({row["controller"] for row in candidates}),
            "failed_or_withheld_candidates_reduce_storage": True,
            "elapsed_costs_included": False,
        },
        "outcomes_generated": False,
    }


def resource_preflight(study: dict, base: dict, target: Path) -> dict:
    """Compare the static budget with a target filesystem without creating it."""
    target = Path(target).expanduser().resolve()
    if target.exists() and not target.is_dir():
        raise ValueError("Study target must be a directory or an uncreated path")
    ancestor = target
    while not ancestor.exists():
        ancestor = ancestor.parent
    budget = estimate_study_resources(study, base)
    free = shutil.disk_usage(ancestor).free
    required = budget["budget"]["required_free_bytes"]
    return {
        **budget,
        "kind": "controller_study_resource_preflight",
        "status": "sufficient" if free >= required else "insufficient",
        "target": str(target),
        "filesystem": {"checked_path": str(ancestor), "free_bytes": free},
        "admitted": free >= required,
    }
