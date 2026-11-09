"""Pre-acquisition checks for the nonreserved end-to-end runner smoke."""
from copy import deepcopy
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from noether_optics.controller_protocol import load_study, study_plan
from noether_optics.controller_resources import estimate_study_resources, load_resource_inputs

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/controller_runner_smoke_v1.yaml"


def runner_module():
    spec = importlib.util.spec_from_file_location(
        "run_controller_study_smoke", ROOT / "scripts/run_controller_study_smoke.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_smoke_plan_is_reduced_isolated_and_storage_bounded():
    study, base = load_resource_inputs(CONFIG, repository=ROOT)
    plan = study_plan(study)
    audit = runner_module().seed_audit(study)
    resources = estimate_study_resources(study, base)

    assert study["study_id"] == "controller_runner_smoke_v1"
    assert (base["n"], base["modes"], base["detector_n"]) == (8, 4, 8)
    assert len(plan["candidates"]) == 5
    assert plan["planned_training_episodes"] == 12
    assert plan["planned_validation_trials"] == 100
    assert plan["maximum_planned_evaluation_trials"] == 600
    assert audit["separate"] is True
    assert audit["reserved_overlap"] == audit["component_overlap"] == []
    assert resources["budget"]["numeric_payload_bytes"] == 6_740_040
    assert resources["budget"]["required_free_bytes"] == 5_419_929_744
    assert resources["outcomes_generated"] is False


@pytest.mark.parametrize(("role", "seed"), [
    ("training", 21001),       # reserved atmosphere root
    ("training", 151001),      # component-smoke derived detector stream
])
def test_runner_seed_audit_detects_cross_experiment_reuse(role, seed):
    study = load_study(CONFIG)
    changed = deepcopy(study)
    changed["seeds"][role][0] = seed
    assert runner_module().seed_audit(changed)["separate"] is False


def test_ci_cannot_start_full_runner_smoke(tmp_path):
    out = tmp_path / "must-not-exist"
    result = subprocess.run([
        sys.executable, str(ROOT / "scripts/run_controller_study_smoke.py"),
        "--out", str(out),
    ], env=dict(os.environ, CI="true"), capture_output=True, text=True)
    assert result.returncode != 0 and "outside CI" in result.stderr
    assert not out.exists()


@pytest.mark.parametrize("damage", ["changed", "missing", "extra"])
def test_runner_smoke_rebuild_rejects_inventory_drift(tmp_path, damage):
    runner = runner_module()
    (tmp_path / "run").mkdir()
    (tmp_path / "commands.json").write_text("{}\n")
    evidence = tmp_path / "run/evidence.json"
    evidence.write_text("retained\n")
    runner._write_json(tmp_path / "protected_sha256.json", runner._protected_files(tmp_path))
    assert runner._verify_protected(tmp_path)

    if damage == "changed":
        evidence.write_text("changed\n")
    elif damage == "missing":
        evidence.unlink()
    else:
        (tmp_path / "run/extra.json").write_text("unlisted\n")

    with pytest.raises(ValueError, match="changed"):
        runner._verify_protected(tmp_path)


@pytest.mark.parametrize("damage", [None, "source", "manifest", "incomplete", "extra", "sequence",
                                     "missing_compile", "negative_time", "interrupted", "synthetic"])
def test_cost_gate_requires_complete_bound_nonsynthetic_attempts(tmp_path, damage):
    runner = runner_module()
    manifest = {"provenance": {"source_commit": "fixture"}, "config": {"study_id": "fixture"}}
    directory = tmp_path / "costs"
    directory.mkdir()
    for stage in ("training", "validation", "evaluation"):
        rows = [{"sequence": 0, "kind": "attempt_start", "stage": stage,
                 "clock": "perf_counter_ns", "provenance": manifest["provenance"],
                 "manifest_sha256": hashlib.sha256(runner._json_bytes(manifest)).hexdigest()}]
        rows += [{"sequence": i, "kind": "operation", "operation": operation,
                  "status": "completed", "elapsed_ns": 100}
                 for i, operation in enumerate(("input_placement", "lower_and_compile", "kernel_execution"), 1)]
        rows.append({"sequence": 4, "kind": "attempt_end", "status": "completed", "elapsed_ns": 400})
        if stage == "training":
            if damage == "source":
                rows[0]["provenance"] = {"source_commit": "changed"}
            elif damage == "manifest":
                rows[0]["manifest_sha256"] = "changed"
            elif damage == "incomplete":
                rows.pop()
            elif damage == "sequence":
                rows[2]["sequence"] = 4
            elif damage == "missing_compile":
                rows[2]["operation"] = "kernel_execution"
            elif damage == "negative_time":
                rows[1]["elapsed_ns"] = -1
            elif damage == "interrupted":
                rows[-1]["status"] = "interrupted"
            elif damage == "synthetic":
                rows[0]["clock"] = "synthetic_test_clock"
        (directory / (stage + ".jsonl")).write_text("".join(json.dumps(row) + "\n" for row in rows))
    if damage == "extra":
        (directory / "interrupted.jsonl").write_text("{}\n")
    assert runner._cost_records_valid(tmp_path, manifest) is (damage is None)
