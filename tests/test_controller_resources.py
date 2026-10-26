"""Static storage-budget checks that never import or execute the simulator."""
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from noether_optics import controller_resources as resources

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def inputs():
    return resources.load_resource_inputs(
        ROOT / "configs/controller_study_v1.yaml", repository=ROOT)


def test_frozen_study_budget_is_deterministic_auditable_and_outcome_free(inputs):
    study, base = inputs
    original_study, original_base = deepcopy(study), deepcopy(base)

    first = resources.estimate_study_resources(study, base)

    assert first == resources.estimate_study_resources(study, base)
    assert study == original_study and base == original_base
    assert first["kind"] == "controller_study_resource_budget"
    assert first["status"] == "derived_not_measured"
    assert first["outcomes_generated"] is False
    assert first["counts"] == {
        "training_pilot_records": 12,
        "fitted_predictor_candidates": 6,
        "fitted_learned_candidates": 6,
        "validation_trials": 600,
        "maximum_evaluation_trials": 600,
        "numeric_archives": 3069,
    }
    categories = {row["name"]: row for row in first["categories"]}
    assert categories["training_shared_atmospheres"]["archives"] == 6
    assert categories["learned_episode_results"]["archives"] == 1512
    assert categories["validation_shared_atmospheres"]["archives"] == 10
    assert categories["validation_open_loops"]["archives"] == 20
    assert categories["evaluation_shared_atmospheres"]["archives"] == 20
    assert categories["evaluation_open_loops"]["archives"] == 120
    budget = first["budget"]
    assert budget == {
        "numeric_payload_bytes": 42_166_977_776,
        "archive_envelope_bytes": 84_440_381_920,
        "metadata_filesystem_envelope_bytes": 10_556_014_592,
        "retained_evidence_envelope_bytes": 94_996_396_512,
        "largest_single_archive_envelope_bytes": 69_644_288,
        "temporary_write_headroom_bytes": 536_870_912,
        "operating_reserve_bytes": 9_500_098_560,
        "required_free_bytes": 105_033_365_984,
        "compression_credit_bytes": 0,
    }
    assert budget["archive_envelope_bytes"] >= 2 * budget["numeric_payload_bytes"]
    assert budget["retained_evidence_envelope_bytes"] == (
        budget["archive_envelope_bytes"] + budget["metadata_filesystem_envelope_bytes"])
    assert budget["required_free_bytes"] == (
        budget["retained_evidence_envelope_bytes"]
        + budget["temporary_write_headroom_bytes"]
        + budget["operating_reserve_bytes"])
    assert first["assumptions"]["elapsed_costs_included"] is False


def test_resource_module_import_does_not_load_jax_or_simulator():
    check = subprocess.run([
        sys.executable, "-c",
        "import sys; import noether_optics.controller_resources; "
        "assert 'jax' not in sys.modules; "
        "assert 'noether_optics.simulation' not in sys.modules",
    ], cwd=ROOT, capture_output=True, text=True)
    assert check.returncode == 0, check.stderr


def test_resource_loader_rechecks_the_exact_base_bytes_it_parses(
        inputs, tmp_path, monkeypatch):
    study, _ = inputs
    base_path = tmp_path / study["base_config"]
    base_path.parent.mkdir(parents=True)
    base_path.write_bytes((ROOT / study["base_config"]).read_bytes() + b"\n# changed after load\n")
    monkeypatch.setattr(resources, "load_study", lambda path, repository: deepcopy(study))

    with pytest.raises(ValueError, match="hash changed during resource loading"):
        resources.load_resource_inputs(tmp_path / "study.yaml", repository=tmp_path)


def test_budget_uses_frozen_dimensions_and_trial_frames(inputs):
    study, base = inputs
    baseline = resources.estimate_study_resources(study, base)
    changed = deepcopy(study)
    changed["loop"].update(frames=1024, fault_end=1024)

    larger = resources.estimate_study_resources(changed, base)

    assert larger["budget"]["numeric_payload_bytes"] > baseline["budget"]["numeric_payload_bytes"]
    assert larger["budget"]["required_free_bytes"] > baseline["budget"]["required_free_bytes"]


@pytest.mark.parametrize(("field", "value"), [
    ("sensor", "linear"), ("n", True), ("modes", 0), ("detector_n", 2.5),
])
def test_incompatible_base_contract_is_rejected(inputs, field, value):
    study, base = inputs
    base[field] = value
    with pytest.raises(ValueError):
        resources.estimate_study_resources(study, base)


def test_preflight_checks_nearest_existing_ancestor_without_creating_target(
        inputs, tmp_path, monkeypatch):
    study, base = inputs
    target = tmp_path / "new" / "study"
    required = resources.estimate_study_resources(study, base)["budget"]["required_free_bytes"]
    checked = []

    def disk_usage(path):
        checked.append(Path(path))
        return SimpleNamespace(free=required)

    monkeypatch.setattr(resources.shutil, "disk_usage", disk_usage)
    report = resources.resource_preflight(study, base, target)

    assert report["admitted"] is True and report["status"] == "sufficient"
    assert report["filesystem"] == {"checked_path": str(tmp_path), "free_bytes": required}
    assert report["target"] == str(target.resolve())
    assert checked == [tmp_path]
    assert not target.exists()


def test_preflight_refuses_insufficient_space_and_file_target(inputs, tmp_path, monkeypatch):
    study, base = inputs
    monkeypatch.setattr(resources.shutil, "disk_usage", lambda path: SimpleNamespace(free=1))
    report = resources.resource_preflight(study, base, tmp_path / "study")
    assert report["admitted"] is False and report["status"] == "insufficient"

    file_target = tmp_path / "file"
    file_target.write_text("occupied")
    with pytest.raises(ValueError, match="directory"):
        resources.resource_preflight(study, base, file_target)
