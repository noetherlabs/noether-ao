"""Pre-acquisition checks for the nonreserved observability runner smoke."""
from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest

from noether_optics.observability_protocol import protocol_plan, validate_protocol
from noether_optics.observability_resources import (
    estimate_observability_resources,
    load_resource_inputs,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/optical_gain_observability_runner_smoke_v1.yaml"


def runner_module():
    spec = importlib.util.spec_from_file_location(
        "run_observability_smoke", ROOT / "scripts/run_observability_smoke.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_observability_smoke_is_reduced_isolated_and_storage_bounded():
    protocol, base = load_resource_inputs(CONFIG, repository=ROOT)
    plan = protocol_plan(protocol)
    audit = runner_module().seed_audit(protocol)
    resources = estimate_observability_resources(protocol, base)

    assert protocol["study_id"] == "optical_gain_observability_runner_smoke_v1"
    assert (base["n"], base["modes"], base["detector_n"]) == (8, 4, 8)
    assert protocol["loop"] == {
        "frames": 24,
        "burn_in": 4,
        "fault_onset": 16,
        "fault_end": 24,
        "optical_gain_drift_frames": 4,
    }
    assert plan["planned_training_trials"] == 2
    assert plan["planned_validation_trials"] == 4
    assert plan["maximum_planned_evaluation_trials"] == 10
    assert audit["separate"] is True
    assert audit["reserved_v1_overlap"] == audit["prior_experiment_overlap"] == []
    assert resources["counts"] == {
        "training_trials": 2,
        "validation_trials": 4,
        "maximum_evaluation_trials": 10,
        "full_wfs_trial_archives": 8,
        "numeric_archives": 18,
    }
    assert resources["budget"]["numeric_payload_bytes"] == 237_616
    assert resources["budget"]["required_free_bytes"] == 5_370_101_856
    assert resources["outcomes_generated"] is False


@pytest.mark.parametrize(("role", "seed"), [
    ("training", 310001),  # frozen P2 v1 atmosphere root
    ("training", 410001),  # frozen P2 v1 derived detector stream
    ("validation", 61001),  # controller runner-smoke atmosphere root
])
def test_observability_smoke_rejects_protected_stream_reuse(role, seed):
    protocol, _ = load_resource_inputs(CONFIG, repository=ROOT)
    damaged = deepcopy(protocol)
    damaged["seeds"][role][0] = seed
    with pytest.raises(ValueError, match="overlap"):
        validate_protocol(damaged)


def test_ci_cannot_start_observability_smoke(tmp_path):
    out = tmp_path / "must-not-exist"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_observability_smoke.py"),
            "--out",
            str(out),
        ],
        env=dict(os.environ, CI="true"),
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "outside CI" in result.stderr
    assert not out.exists()


@pytest.mark.parametrize("damage", ["changed", "missing", "extra", "symlink"])
def test_observability_smoke_rebuild_rejects_inventory_drift(tmp_path, damage):
    runner = runner_module()
    (tmp_path / "run").mkdir()
    (tmp_path / "commands.json").write_text("{}\n")
    evidence = tmp_path / "run/evidence.json"
    evidence.write_text("retained\n")
    runner._write_json(
        tmp_path / "protected_sha256.json", runner._protected_files(tmp_path)
    )
    assert runner._verify_protected(tmp_path)

    if damage == "changed":
        evidence.write_text("changed\n")
    elif damage == "missing":
        evidence.unlink()
    elif damage == "extra":
        (tmp_path / "run/extra.json").write_text("unlisted\n")
    else:
        evidence.unlink()
        evidence.symlink_to(tmp_path / "commands.json")

    with pytest.raises(ValueError, match="changed|symlink"):
        runner._verify_protected(tmp_path)


def test_smoke_reports_paired_controller_error_without_ranking():
    rows = []
    for seed, clean_wfe, gain_wfe in ((720001, 10.0, 14.0), (720002, 20.0, 18.0)):
        for condition, wfe, strehl in (
            ("clean", clean_wfe, 0.8),
            ("gain_035", gain_wfe, 0.7),
        ):
            rows.append({
                "status": "completed",
                "case": {
                    "condition": condition,
                    "regime_id": "regime-00",
                    "atmosphere_seed": seed,
                    "detector_seed": seed + 100000,
                },
                "metrics": {
                    "retained_trial_metrics": {
                        "wfe_nm": wfe,
                        "strehl": strehl,
                        "saturation_fraction": 0.0,
                    }
                },
            })

    pairs = runner_module()._paired_controller_error({"validation_records": rows})

    assert [row["wfe_nm_delta"] for row in pairs] == [4.0, -2.0]
    assert [row["strehl_delta"] for row in pairs] == pytest.approx([-0.1, -0.1])
