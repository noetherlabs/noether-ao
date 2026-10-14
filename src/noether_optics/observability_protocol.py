"""Validate and enumerate the P2 optical-gain protocol without running physics."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml


UINT32_LIMIT = 2**32
BASE_CONFIG = "configs/simulation.yaml"
BASE_CONFIG_SHA256 = "71e98a0ab39ba8ec668247beaae2bab450bd767be80d2942a086a43051bf1731"
ROLE_COUNTS = {"training": 16, "validation": 30, "evaluation": 60}
ROLES = tuple(ROLE_COUNTS)
SMOKE_STUDY_ID = "optical_gain_observability_runner_smoke_v1"
PRIOR_ROOTS = frozenset(
    {11, 12, 13, 17, 101, 102, 103, 104, 105, 20261008}
    | set(range(21001, 21004))
    | set(range(22001, 22006))
    | set(range(23001, 23011))
    | {24001}
    | set(range(51001, 51004))
    | set(range(52001, 52006))
    | set(range(53001, 53011))
    | {54001}
    | set(range(61001, 61004))
    | set(range(62001, 62006))
    | set(range(63001, 63011))
    | {64001}
)
V1_ROOTS = frozenset(
    range(310001, 310017)
) | frozenset(
    range(320001, 320031)
) | frozenset(
    range(330001, 330061)
)

PROFILES = {
    "optical_gain_observability_v1": {
        "base_config": BASE_CONFIG,
        "base_config_sha256": BASE_CONFIG_SHA256,
        "role_counts": ROLE_COUNTS,
        "reserved_roots": PRIOR_ROOTS,
        "loop": {
            "frames": 512,
            "burn_in": 128,
            "fault_onset": 256,
            "fault_end": 512,
            "optical_gain_drift_frames": 32,
        },
        "regimes": {
            "photons": [1000.0, 1000000.0],
            "wind_scales": [0.5, 2.0],
            "common_random_numbers_across_regimes": True,
        },
        "validation": {
            "scope": "each_regime",
            "integrity_interval": "full_run",
            "saturation_interval": "full_run",
            "release_evaluation_only_if_all_regimes_pass": True,
            "max_clean_alarm_runs": 0,
            "max_clean_raw_crossing_fraction": 0.01,
            "max_clean_alarm_frame_fraction": 0.001,
            "max_gain_pre_onset_alarm_runs": 0,
            "min_gain_detected_runs": 29,
            "max_gain_missed_runs": 1,
            "max_detection_delay_frames": 32,
            "require_all_finite": True,
            "require_no_fallback": True,
            "require_no_unavailable": True,
            "max_saturation_fraction": 0.0,
            "failure_action": "withhold_evaluation_and_retain_failure",
        },
    },
    SMOKE_STUDY_ID: {
        "base_config": "configs/controller_runner_smoke_base.yaml",
        "base_config_sha256": (
            "72f646b4c9df8777c7cc9c0fa333f1031212348743f007949116b598063ba3d1"
        ),
        "role_counts": {"training": 2, "validation": 2, "evaluation": 2},
        "reserved_roots": PRIOR_ROOTS | V1_ROOTS,
        "loop": {
            "frames": 24,
            "burn_in": 4,
            "fault_onset": 16,
            "fault_end": 24,
            "optical_gain_drift_frames": 4,
        },
        "regimes": {
            "photons": [1000000.0],
            "wind_scales": [1.0],
            "common_random_numbers_across_regimes": True,
        },
        "validation": {
            "scope": "each_regime",
            "integrity_interval": "full_run",
            "saturation_interval": "full_run",
            "release_evaluation_only_if_all_regimes_pass": True,
            "max_clean_alarm_runs": 0,
            "max_clean_raw_crossing_fraction": 0.01,
            "max_clean_alarm_frame_fraction": 0.001,
            "max_gain_pre_onset_alarm_runs": 0,
            "min_gain_detected_runs": 2,
            "max_gain_missed_runs": 0,
            "max_detection_delay_frames": 8,
            "require_all_finite": True,
            "require_no_fallback": True,
            "require_no_unavailable": True,
            "max_saturation_fraction": 0.0,
            "failure_action": "withhold_evaluation_and_retain_failure",
        },
    },
}


def _keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{label} keys must be exactly {sorted(expected)}")
    return value


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _seed_list(values: Any, label: str, minimum: int) -> set[int]:
    if not isinstance(values, list) or len(values) < minimum:
        raise ValueError(f"{label} requires at least {minimum} seeds")
    for value in values:
        _integer(value, f"{label} seed")
        if value >= UINT32_LIMIT:
            raise ValueError(f"{label} seeds must fit uint32")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} seeds must be unique")
    return set(values)


def _exact(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"Unsupported {label} contract")


def validate_protocol(config: dict[str, Any]) -> None:
    """Reject leakage or drift in a supported observability contract."""
    top = {
        "schema_version", "study_id", "base_config", "base_config_sha256", "seeds",
        "loop", "regimes", "controller", "detector", "conditions", "faults",
        "validation", "evaluation", "reporting", "retention", "escalation",
    }
    _keys(config, top, "protocol")
    if config["schema_version"] != 1 or config["study_id"] not in PROFILES:
        raise ValueError("Unsupported optical-gain protocol identity")
    profile = PROFILES[config["study_id"]]
    if (
        config["base_config"] != profile["base_config"]
        or config["base_config_sha256"] != profile["base_config_sha256"]
    ):
        raise ValueError("Unsupported base simulation binding")

    seeds = _keys(
        config["seeds"],
        {"training", "validation", "evaluation", "noise_offset", "excluded_prior_roots"},
        "seed",
    )
    if type(seeds["noise_offset"]) is not int or seeds["noise_offset"] != 100000:
        raise ValueError("noise_offset must match the simulation sensor-key offset")
    offset = seeds["noise_offset"]
    excluded = _seed_list(seeds["excluded_prior_roots"], "prior exclusion", 1)
    if not excluded.issuperset(PRIOR_ROOTS):
        raise ValueError("Prior exclusions must retain every recorded scientific seed root")
    reserved = excluded | profile["reserved_roots"]
    used = reserved | {seed + offset for seed in reserved}
    for role, count in profile["role_counts"].items():
        roots = _seed_list(seeds[role], role, count)
        if len(roots) != count:
            raise ValueError(f"{role} requires exactly {count} seeds")
        if any(seed + offset >= UINT32_LIMIT for seed in roots):
            raise ValueError("Derived detector seeds would overflow uint32")
        streams = roots | {seed + offset for seed in roots}
        if len(streams) != 2 * len(roots) or streams & used:
            raise ValueError(
                f"{role} atmosphere/detector streams overlap prior data or another role"
            )
        used |= streams

    _exact(config["loop"], profile["loop"], "loop")
    _exact(config["regimes"], profile["regimes"], "regime")
    _exact(
        config["controller"],
        {"name": "forward", "gain": 0.3, "leak": 0.999, "refit": False,
         "noise": "poisson_plus_read", "fallback": False,
         "detector_output_affects_command": False},
        "controller",
    )
    _exact(
        config["detector"],
        {"source_signal": "existing_whitened_residual_score",
         "statistic": "causal_arithmetic_mean", "window_frames": 4, "tail": "lower",
         "training_quantile": 0.005, "quantile_method": "linear",
         "comparison": "strict_less_than", "threshold_scope": "per_regime",
         "calibration_source": "training_clean_only", "persistence_windows": 8,
         "target_streak_must_begin_at_or_after_onset": True,
         "upper_tail_union": False, "freeze_before_validation": True,
         "recalibration_after_validation": False},
        "detector",
    )
    _exact(
        config["conditions"],
        {"training": ["clean"], "validation": ["clean", "gain_035"],
         "evaluation": ["clean", "gain_035", "gain_065", "vibration", "misregistration"],
         "target_conditions": ["gain_035", "gain_065"],
         "nuisance_conditions": ["vibration", "misregistration"],
         "pair_same_seed_across_conditions": True},
        "condition",
    )
    _exact(
        config["faults"],
        {"gain_035": {"kind": "optical_gain", "terminal_response_multiplier": 0.35},
         "gain_065": {"kind": "optical_gain", "terminal_response_multiplier": 0.65},
         "vibration": {"kind": "vibration", "amplitude_rad": 0.7, "frequency_hz": 40.0},
         "misregistration": {"kind": "misregistration", "shift_pixels": 2}},
        "fault",
    )
    _exact(config["validation"], profile["validation"], "validation gate")
    _exact(
        config["evaluation"],
        {"state_before_validation": "sealed", "descriptive_no_pass_claim": True,
         "run_interval_method": "clopper_pearson_exact_two_sided",
         "run_interval_confidence": 0.95,
         "frame_interval_method": "none_temporally_correlated_descriptive_only",
         "paired_error_method": "paired_student_t_bonferroni",
         "primary_paired_error_metric": "wfe_nm",
         "paired_error_interval": "fault_onset_to_fault_end",
         "paired_error_familywise_alpha": 0.05,
         "paired_error_family": "all_nonclean_conditions_by_regime",
         "secondary_error_metrics": ["strehl"],
         "secondary_interval_method": "descriptive_only"},
        "evaluation",
    )
    _exact(
        config["reporting"],
        {"clean": ["alarm_runs", "alarm_run_fraction", "alarm_frames",
                   "alarm_frame_fraction", "raw_crossing_fraction"],
         "targets": ["detected_runs_without_prior_alarm", "missed_runs",
                     "pre_onset_alarm_runs", "boundary_straddling_alarm_runs",
                     "detected_only_delay_frames"],
         "nuisances": ["alarm_runs_without_prior_alarm", "quiet_runs",
                       "pre_onset_alarm_runs", "boundary_straddling_alarm_runs",
                       "alarm_delay_frames"],
         "controller_error": ["paired_wfe_nm_delta", "paired_strehl_delta",
                              "saturation_fraction"],
         "integrity": ["finite_runs", "fallback_frames", "unavailable_frames", "failed_runs"]},
        "reporting",
    )
    _exact(
        config["retention"],
        {"all_trial_frame_arrays": ["score", "statistic_frame_id", "lower_statistic",
                                    "raw_crossing", "alarm_frame_id", "p2_alarm", "variance",
                                    "strehl", "saturated", "fallback", "unavailable", "command",
                                    "applied"],
         "all_trial_metrics": ["wfe_nm", "strehl", "saturation_fraction"],
         "full_wfs_arrays": ["measurement", "prediction"],
         "full_wfs_policy": "first_seed_per_role_regime_condition",
         "retain_failed_runs": True, "compression_credit": False},
        "retention",
    )
    _exact(
        config["escalation"],
        {"next_protocol": "optical_gain_observability_v2", "next_candidate": "modal_dither",
         "trigger": "v1_validation_failure_or_inadequate_v1_evaluation",
         "requires_new_training_validation_evaluation_seeds": True,
         "requires_dither_free_paired_science_control": True,
         "science_impact_metrics": ["wfe_nm", "strehl", "command_rms"],
         "v1_held_out_outcomes_may_not_select_escalation": True,
         "independent_sensing_requires_separate_later_protocol": True},
        "escalation",
    )


def load_protocol(path: Path, repository: Path | None = None) -> dict[str, Any]:
    """Load the protocol and reject drift in the inherited plant configuration."""
    root = repository or Path(__file__).resolve().parents[2]
    config = yaml.safe_load(path.read_text())
    validate_protocol(config)
    base = (root / config["base_config"]).read_bytes()
    if hashlib.sha256(base).hexdigest() != config["base_config_sha256"]:
        raise ValueError("Base simulation configuration hash changed")
    return config


def _zero_event_upper_bound(samples: int, alpha: float = 0.05) -> float:
    return 1 - alpha ** (1 / samples)


def protocol_plan(config: dict[str, Any]) -> dict[str, Any]:
    """Enumerate declared budgets and identities without opening any cohort."""
    validate_protocol(config)
    regimes = [
        {"id": f"regime-{index:02d}", "photons": photons, "wind_scale": wind}
        for index, (photons, wind) in enumerate(
            (p, w) for p in config["regimes"]["photons"] for w in config["regimes"]["wind_scales"]
        )
    ]
    seeds = config["seeds"]
    conditions = config["conditions"]
    regime_count = len(regimes)
    training = regime_count * len(conditions["training"]) * len(seeds["training"])
    validation = regime_count * len(conditions["validation"]) * len(seeds["validation"])
    evaluation = regime_count * len(conditions["evaluation"]) * len(seeds["evaluation"])
    paired_family = regime_count * (len(conditions["evaluation"]) - 1)
    return {
        "kind": "predeclared_plan",
        "status": "not_executed",
        "study_id": config["study_id"],
        "regimes": regimes,
        "conditions": {role: conditions[role] for role in ROLES},
        "condition_classes": {
            "target": conditions["target_conditions"],
            "nuisance": conditions["nuisance_conditions"],
        },
        "seed_streams": {
            role: [{"atmosphere": seed, "detector": seed + seeds["noise_offset"]}
                   for seed in seeds[role]]
            for role in ROLES
        },
        "planned_training_trials": training,
        "planned_validation_trials": validation,
        "maximum_planned_evaluation_trials": evaluation,
        "maximum_planned_trials": training + validation + evaluation,
        "planned_evaluation_fault_clean_pairs": paired_family * len(seeds["evaluation"]),
        "predeclared_paired_error_family": paired_family,
        "simultaneous_interval_confidence": (
            1 - config["evaluation"]["paired_error_familywise_alpha"] / paired_family
        ),
        "zero_clean_alarm_run_upper_bounds": {
            "validation_one_sided_95": _zero_event_upper_bound(len(seeds["validation"])),
            "evaluation_one_sided_95": _zero_event_upper_bound(len(seeds["evaluation"])),
        },
        "thresholds_generated": False,
        "outcomes_generated": False,
        "validation_decision_generated": False,
        "evaluation_released": False,
    }
