"""Validate and enumerate a controller study without generating any outcomes."""
from __future__ import annotations

from itertools import product
import hashlib
import math
from pathlib import Path

import yaml

PARAMETERS = {"integrator": ("gain", "leak"), "predictive": ("horizon", "ridge"),
              "forward": ("gain", "leak"), "fixed": ("gain", "leak"),
              "learned": ("gain", "leak")}
MIN_SEEDS = {"training": 3, "validation": 5, "evaluation": 10}
UINT32_LIMIT = 2**32
LEGACY_SEEDS = frozenset((11, 12, 13, 17, 101, 102, 103, 104, 105, 20261008))


def _integer(value, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _positive(value, label: str, maximum: float = math.inf) -> None:
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= maximum:
        raise ValueError(f"{label} must be finite and in (0, {maximum}]")


def _seed_list(values, label: str, minimum: int) -> set[int]:
    if not isinstance(values, list) or len(values) < minimum:
        raise ValueError(f"{label} requires at least {minimum} seeds")
    for value in values:
        _integer(value, label)
        if value >= UINT32_LIMIT:
            raise ValueError(f"{label} seeds must fit uint32")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} seeds must be unique")
    return set(values)


def validate_study(config: dict) -> None:
    """Reject leakage, inert search parameters and unequal candidate budgets."""
    if config["schema_version"] != 1:
        raise ValueError("Unsupported controller-study schema")
    seeds = config["seeds"]
    # This is the existing simulation.loop sensor-key offset, not a tunable knob.
    if type(seeds["noise_offset"]) is not int or seeds["noise_offset"] != 100000:
        raise ValueError("noise_offset must match simulation.loop: 100000")
    offset = seeds["noise_offset"]
    legacy = _seed_list(seeds["excluded_legacy"], "legacy exclusion", 1)
    if not legacy.issuperset(LEGACY_SEEDS):
        raise ValueError("Legacy exclusions must retain all recorded baseline seeds")
    used = legacy | {seed + offset for seed in legacy}
    for role, minimum in MIN_SEEDS.items():
        roots = _seed_list(seeds[role], role, minimum)
        if any(seed + offset >= UINT32_LIMIT for seed in roots):
            raise ValueError("Derived detector seeds would overflow uint32")
        streams = roots | {seed + offset for seed in roots}
        if len(streams) != 2 * len(roots) or streams & used:
            raise ValueError(f"{role} atmosphere/detector streams overlap another role or legacy data")
        used |= streams
    initial = _integer(seeds["initialization"], "initialization seed")
    if initial >= UINT32_LIMIT or initial in used:
        raise ValueError("Initialization seed overlaps another stream or exceeds uint32")

    loop, training = config["loop"], config["training"]
    for name, settings in (("loop", loop), ("training", training)):
        frames = _integer(settings["frames"], f"{name} frames", 2)
        burn = _integer(settings["burn_in"], f"{name} burn-in", 1)
        if burn >= frames:
            raise ValueError("Burn-in must leave an evaluation interval")
    if not loop["burn_in"] < _integer(loop["fault_onset"], "fault onset") < _integer(
        loop["fault_end"], "fault end"
    ) <= loop["frames"]:
        raise ValueError("Fault window must follow burn-in and fit inside the run")
    _positive(training["pilot_gain"], "pilot gain", 1)
    _positive(training["pilot_leak"], "pilot leak", 1)
    _integer(training["learned_steps"], "learned training steps", 1)
    _positive(training["learned_learning_rate"], "learned learning rate")
    if training["predictor_telemetry"] != "measured_pseudo_open_loop" or training["learned_noise"] != "gaussian_pathwise":
        raise ValueError("Training telemetry/noise contract changed")

    conditions = config["conditions"]
    for name in ("photons", "wind_scales"):
        values = conditions[name]
        if not isinstance(values, list) or len(values) < 2:
            raise ValueError(f"At least two {name} settings required")
        for value in values:
            _positive(value, name)
        if len(set(values)) != len(values):
            raise ValueError(f"Duplicate {name} settings")
    if conditions["validation_faults"] != ["none"] or conditions["evaluation_faults"] != ["none", "optical_gain", "vibration"]:
        raise ValueError("Fault cohort changed; validation must remain clean")

    budget = _integer(config["candidates_per_controller"], "candidate budget", 1)
    if set(config["candidate_grids"]) != set(PARAMETERS):
        raise ValueError("Every declared controller needs a candidate grid")
    for controller, names in PARAMETERS.items():
        grid = config["candidate_grids"][controller]
        if set(grid) != set(names):
            raise ValueError(f"Unsupported or inert candidate parameters for {controller}")
        count = 1
        for name in names:
            values = grid[name]
            if not isinstance(values, list) or not values:
                raise ValueError("Candidate axes must be nonempty lists")
            for value in values:
                if name == "horizon":
                    _integer(value, "prediction horizon", 1)
                    if value >= training["frames"] - training["burn_in"] - 2:
                        raise ValueError("Horizon leaves insufficient fitting samples")
                else:
                    _positive(value, name, 1 if name in ("gain", "leak") else math.inf)
            if len(set(values)) != len(values):
                raise ValueError("Duplicate candidate settings inflate the search budget")
            count *= len(values)
        if count != budget:
            raise ValueError("Candidate budgets must be equal across controllers")

    selection = config["selection"]
    fixed_selection = {"metric": "mean_relative_wfe_to_open", "scope": "global_across_clean_regimes",
                       "require_each_trial_below_open": True, "no_eligible_candidate": "withhold_evaluation",
                       "tie_break": "candidate_order"}
    if any(selection.get(key) != value for key, value in fixed_selection.items()):
        raise ValueError("Unsupported selection contract")
    if selection["require_each_trial_below_open"] is not True:
        raise ValueError("Each validation trial must beat its paired open loop")
    _positive(selection["max_saturation_fraction"], "maximum saturation fraction", 1)
    evaluation = config["evaluation"]
    if evaluation["noise"] != "poisson_plus_read" or evaluation["fallback"] is not False or evaluation["refit_after_selection"] is not False:
        raise ValueError("Evaluation must use fixed models, Poisson noise and no fallback")
    if evaluation["interval_method"] != "paired_student_t_bonferroni":
        raise ValueError("Unsupported simultaneous interval method")
    _positive(evaluation["familywise_alpha"], "familywise alpha")
    if evaluation["familywise_alpha"] >= 1:
        raise ValueError("Familywise alpha must be less than one")


def load_study(path: Path, repository: Path | None = None) -> dict:
    """Load a study and reject silent changes to its inherited plant settings."""
    root = repository or Path(__file__).resolve().parents[2]
    config = yaml.safe_load(path.read_text())
    base = (root / config["base_config"]).read_bytes()
    if hashlib.sha256(base).hexdigest() != config["base_config_sha256"]:
        raise ValueError("Base simulation configuration hash changed")
    validate_study(config)
    return config


def study_plan(config: dict) -> dict:
    """Enumerate only planned budgets and candidate identities; never simulate."""
    validate_study(config)
    candidates = []
    for controller, names in PARAMETERS.items():
        grid = config["candidate_grids"][controller]
        for index, values in enumerate(product(*(grid[name] for name in names))):
            candidates.append({"id": f"{controller}-{index:02d}", "controller": controller,
                               "parameters": dict(zip(names, values))})
    c, s = config["conditions"], config["seeds"]
    regimes = len(c["photons"]) * len(c["wind_scales"])
    validation_conditions = regimes * len(c["validation_faults"])
    evaluation_conditions = regimes * len(c["evaluation_faults"])
    family = evaluation_conditions * math.comb(len(PARAMETERS), 2)
    return {
        "kind": "predeclared_plan", "status": "not_executed", "study_id": config["study_id"],
        "candidates": candidates,
        "seed_streams": {role: [{"atmosphere": seed, "detector": seed + s["noise_offset"]}
                               for seed in s[role]] for role in MIN_SEEDS},
        "planned_training_episodes": regimes * len(s["training"]),
        "planned_validation_trials_per_controller": config["candidates_per_controller"] * validation_conditions * len(s["validation"]),
        "planned_validation_trials": len(candidates) * validation_conditions * len(s["validation"]),
        "maximum_planned_evaluation_trials": len(PARAMETERS) * evaluation_conditions * len(s["evaluation"]),
        "predeclared_pairwise_family": family,
        "simultaneous_interval_confidence": 1 - config["evaluation"]["familywise_alpha"] / family,
        "outcomes_generated": False,
    }
