"""Guard study leakage and selection budgets before any outcomes are produced."""
from copy import deepcopy
from pathlib import Path
import json
import subprocess
import sys

import pytest
import yaml

from noether_optics.controller_protocol import load_study, study_plan, validate_study
from noether_optics.simulation import CONTROLLERS

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def protocol():
    return load_study(ROOT / "configs/controller_study_v1.yaml")


def test_plan_is_unexecuted_balanced_and_deterministic(protocol):
    original = deepcopy(protocol)
    plan = study_plan(protocol)
    assert protocol == original
    assert plan == study_plan(protocol)
    assert plan["status"] == "not_executed" and plan["outcomes_generated"] is False
    assert len(plan["candidates"]) == 30
    assert len({row["id"] for row in plan["candidates"]}) == 30
    assert set(row["controller"] for row in plan["candidates"]) == set(CONTROLLERS)
    assert all(sum(row["controller"] == name for row in plan["candidates"]) == 6 for name in CONTROLLERS)
    assert plan["planned_training_episodes"] == 12
    assert plan["planned_validation_trials_per_controller"] == 120
    assert plan["planned_validation_trials"] == 600
    assert plan["maximum_planned_evaluation_trials"] == 600
    assert plan["predeclared_pairwise_family"] == 120
    assert plan["simultaneous_interval_confidence"] == pytest.approx(1 - .05 / 120)


@pytest.mark.parametrize("role", ["validation", "evaluation"])
def test_direct_cross_role_seed_reuse_is_rejected(protocol, role):
    protocol["seeds"][role][0] = protocol["seeds"]["training"][0]
    with pytest.raises(ValueError, match="overlap"):
        validate_study(protocol)


def test_derived_sensor_key_collision_is_rejected(protocol):
    # The root lists are disjoint, but evaluation atmosphere would reuse training noise.
    protocol["seeds"]["evaluation"][0] = protocol["seeds"]["training"][0] + 100000
    with pytest.raises(ValueError, match="overlap"):
        validate_study(protocol)


@pytest.mark.parametrize("value", [True, -1, 2.5, 2**32 - 1, 101, 100101])
def test_invalid_overflow_and_legacy_seeds_are_rejected(protocol, value):
    protocol["seeds"]["evaluation"][0] = value
    with pytest.raises(ValueError):
        validate_study(protocol)


@pytest.mark.parametrize("fault", ["duplicate", "short", "initialization", "offset", "legacy_omission"])
def test_seed_contract_cannot_be_silently_relaxed(protocol, fault):
    seeds = protocol["seeds"]
    if fault == "duplicate": seeds["evaluation"][1] = seeds["evaluation"][0]
    elif fault == "short": seeds["validation"].pop()
    elif fault == "initialization": seeds["initialization"] = seeds["validation"][0] + 100000
    elif fault == "offset": seeds["noise_offset"] = 200000
    else: seeds["excluded_legacy"].remove(101)
    with pytest.raises(ValueError):
        validate_study(protocol)


@pytest.mark.parametrize("fault", ["unequal", "duplicate", "nan", "ignored_gain", "missing_controller", "long_horizon"])
def test_invalid_or_inert_candidate_search_is_rejected(protocol, fault):
    grids = protocol["candidate_grids"]
    if fault == "unequal": grids["integrator"]["gain"].pop()
    elif fault == "duplicate": grids["forward"]["gain"] = [.1, .1, .5]
    elif fault == "nan": grids["fixed"]["leak"][0] = float("nan")
    elif fault == "ignored_gain": grids["predictive"]["gain"] = [1.]
    elif fault == "missing_controller": del grids["learned"]
    else: grids["predictive"]["horizon"][-1] = protocol["training"]["frames"]
    with pytest.raises(ValueError):
        validate_study(protocol)


@pytest.mark.parametrize("fault", ["empty_interval", "fault_during_burn", "fault_selection", "oracle", "refit", "noise"])
def test_phase_boundaries_and_frozen_evaluation_contract(protocol, fault):
    if fault == "empty_interval": protocol["loop"]["burn_in"] = protocol["loop"]["frames"]
    elif fault == "fault_during_burn": protocol["loop"]["fault_onset"] = 10
    elif fault == "fault_selection": protocol["conditions"]["validation_faults"].append("optical_gain")
    elif fault == "oracle": protocol["training"]["predictor_telemetry"] = "true_phase"
    elif fault == "refit": protocol["evaluation"]["refit_after_selection"] = True
    else: protocol["evaluation"]["noise"] = "gaussian"
    with pytest.raises(ValueError):
        validate_study(protocol)


def test_base_configuration_drift_is_rejected(protocol, tmp_path):
    base = tmp_path / protocol["base_config"]
    base.parent.mkdir(parents=True)
    base.write_text((ROOT / protocol["base_config"]).read_text() + "\n# drift\n")
    config = tmp_path / "study.yaml"
    config.write_text(yaml.safe_dump(protocol))
    with pytest.raises(ValueError, match="hash changed"):
        load_study(config, repository=tmp_path)


def test_planning_command_preserves_an_existing_frozen_plan(tmp_path):
    output = tmp_path / "plan.json"
    command = [sys.executable, str(ROOT / "scripts/plan_controller_study.py"),
               "--config", str(ROOT / "configs/controller_study_v1.yaml"), "--out", str(output)]
    subprocess.run(command, cwd=tmp_path, check=True, capture_output=True)
    original = output.read_bytes()
    assert json.loads(original)["outcomes_generated"] is False
    repeated = subprocess.run(command, cwd=tmp_path, capture_output=True)
    assert repeated.returncode != 0
    assert output.read_bytes() == original
