from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from noether_optics.observability_protocol import load_protocol, protocol_plan, validate_protocol


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/optical_gain_observability_v1.yaml"
PLAN = ROOT / "plans/optical_gain_observability_v1.json"


@pytest.fixture
def protocol() -> dict:
    return load_protocol(CONFIG)


def test_committed_plan_is_balanced_deterministic_and_contains_no_outcomes(protocol) -> None:
    plan = protocol_plan(protocol)
    expected = json.loads(PLAN.read_text())
    rebuilt = plan | {
        "protocol_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
        "base_config_sha256": protocol["base_config_sha256"],
    }
    assert rebuilt == expected
    assert len(plan["regimes"]) == 4
    assert plan["condition_classes"] == {
        "target": ["gain_035", "gain_065"],
        "nuisance": ["vibration", "misregistration"],
    }
    assert plan["planned_training_trials"] == 64
    assert plan["planned_validation_trials"] == 240
    assert plan["maximum_planned_evaluation_trials"] == 1200
    assert plan["maximum_planned_trials"] == 1504
    assert plan["planned_evaluation_fault_clean_pairs"] == 960
    assert plan["predeclared_paired_error_family"] == 16
    assert plan["simultaneous_interval_confidence"] == 0.996875
    assert plan["zero_clean_alarm_run_upper_bounds"] == pytest.approx({
        "validation_one_sided_95": 0.09503385285530419,
        "evaluation_one_sided_95": 0.04870291331009746,
    })
    assert all(plan[key] is False for key in (
        "thresholds_generated", "outcomes_generated", "validation_decision_generated",
        "evaluation_released",
    ))


def test_all_new_atmosphere_and_detector_streams_are_disjoint(protocol) -> None:
    streams = protocol_plan(protocol)["seed_streams"]
    sets = {
        role: {value for pair in pairs for value in pair.values()}
        for role, pairs in streams.items()
    }
    assert all(len(values) == 2 * len(streams[role]) for role, values in sets.items())
    assert not sets["training"] & sets["validation"]
    assert not sets["training"] & sets["evaluation"]
    assert not sets["validation"] & sets["evaluation"]


@pytest.mark.parametrize("collision", ["direct", "derived", "prior", "missing_prior"])
def test_seed_leakage_is_rejected(protocol, collision) -> None:
    damaged = deepcopy(protocol)
    if collision == "direct":
        damaged["seeds"]["validation"][0] = damaged["seeds"]["training"][0]
    elif collision == "derived":
        damaged["seeds"]["validation"][0] = (
            damaged["seeds"]["training"][0] + damaged["seeds"]["noise_offset"]
        )
    elif collision == "prior":
        damaged["seeds"]["evaluation"][0] = 101
    else:
        damaged["seeds"]["excluded_prior_roots"].remove(101)
    with pytest.raises(ValueError, match="overlap|Prior exclusions"):
        validate_protocol(damaged)


def test_frozen_role_counts_cannot_be_expanded(protocol) -> None:
    damaged = deepcopy(protocol)
    damaged["seeds"]["validation"].append(320031)
    with pytest.raises(ValueError, match="exactly 30"):
        validate_protocol(damaged)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("detector", "window_frames", 5),
        ("detector", "tail", "two_sided"),
        ("detector", "training_quantile", 0.01),
        ("detector", "persistence_windows", 7),
        ("detector", "target_streak_must_begin_at_or_after_onset", False),
        ("detector", "threshold_scope", "global"),
        ("detector", "upper_tail_union", True),
        ("loop", "fault_onset", 255),
        ("controller", "fallback", True),
        ("controller", "detector_output_affects_command", True),
        ("conditions", "target_conditions", ["gain_035"]),
        ("faults", "gain_035", {"kind": "optical_gain", "terminal_response_multiplier": 0.4}),
    ],
)
def test_candidate_and_plant_semantics_are_frozen(protocol, section, key, value) -> None:
    damaged = deepcopy(protocol)
    damaged[section][key] = value
    with pytest.raises(ValueError, match="Unsupported"):
        validate_protocol(damaged)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("max_clean_alarm_runs", 1),
        ("max_clean_raw_crossing_fraction", 0.02),
        ("min_gain_detected_runs", 28),
        ("max_gain_missed_runs", 2),
        ("max_detection_delay_frames", 33),
        ("integrity_interval", "post_burn"),
        ("failure_action", "continue_evaluation"),
    ],
)
def test_validation_gates_cannot_be_relaxed(protocol, key, value) -> None:
    damaged = deepcopy(protocol)
    damaged["validation"][key] = value
    with pytest.raises(ValueError, match="validation gate"):
        validate_protocol(damaged)


def test_reporting_intervals_and_escalation_are_frozen(protocol) -> None:
    interval = deepcopy(protocol)
    interval["evaluation"]["frame_interval_method"] = "iid_wilson"
    with pytest.raises(ValueError, match="evaluation"):
        validate_protocol(interval)
    escalation = deepcopy(protocol)
    escalation["escalation"]["next_candidate"] = "inspect_v1_then_choose"
    with pytest.raises(ValueError, match="escalation"):
        validate_protocol(escalation)


def test_base_configuration_drift_is_rejected(tmp_path) -> None:
    base = tmp_path / "configs/simulation.yaml"
    base.parent.mkdir(parents=True)
    base.write_text("changed: true\n")
    with pytest.raises(ValueError, match="hash changed"):
        load_protocol(CONFIG, repository=tmp_path)


def test_base_configuration_path_is_frozen(protocol) -> None:
    damaged = deepcopy(protocol)
    damaged["base_config"] = "../simulation.yaml"
    with pytest.raises(ValueError, match="base simulation binding"):
        validate_protocol(damaged)


def test_plan_writer_refuses_to_overwrite(tmp_path) -> None:
    destination = tmp_path / "plan.json"
    command = [
        sys.executable,
        str(ROOT / "scripts/plan_optical_gain_observability.py"),
        "--config",
        str(CONFIG),
        "--out",
        str(destination),
    ]
    first = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
    assert "no threshold or outcome generated" in first.stdout
    assert json.loads(destination.read_text()) == json.loads(PLAN.read_text())
    second = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    assert second.returncode != 0
    assert destination.read_text() == PLAN.read_text()


def test_yaml_round_trip_preserves_protocol_identity(protocol) -> None:
    loaded = yaml.safe_load(CONFIG.read_text())
    assert loaded == protocol
