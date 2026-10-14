"""Outcome-free storage admission checks for the frozen P2 protocol."""
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from noether_optics import observability_resources as resources

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/optical_gain_observability_v1.yaml"


@pytest.fixture
def inputs():
    return resources.load_resource_inputs(CONFIG, repository=ROOT)


def test_frozen_budget_is_deterministic_auditable_and_outcome_free(inputs) -> None:
    protocol, base = inputs
    original_protocol, original_base = deepcopy(protocol), deepcopy(base)

    first = resources.estimate_observability_resources(protocol, base)

    assert first == resources.estimate_observability_resources(protocol, base)
    assert protocol == original_protocol and base == original_base
    assert first["kind"] == "optical_gain_observability_resource_budget"
    assert first["status"] == "derived_not_measured"
    assert first["outcomes_generated"] is False
    assert first["counts"] == {
        "training_trials": 64,
        "validation_trials": 240,
        "maximum_evaluation_trials": 1200,
        "full_wfs_trial_archives": 32,
        "numeric_archives": 1504,
    }
    categories = {row["name"]: row for row in first["categories"]}
    assert {name: row["archives"] for name, row in categories.items()} == {
        "training_standard_trial_outputs": 60,
        "training_full_wfs_trial_outputs": 4,
        "validation_standard_trial_outputs": 232,
        "validation_full_wfs_trial_outputs": 8,
        "evaluation_standard_trial_outputs": 1180,
        "evaluation_full_wfs_trial_outputs": 20,
    }
    assert first["budget"] == {
        "numeric_payload_bytes": 2_341_413_792,
        "archive_envelope_bytes": 4_763_174_720,
        "metadata_filesystem_envelope_bytes": 595_591_168,
        "retained_evidence_envelope_bytes": 5_358_765_888,
        "largest_single_archive_envelope_bytes": 68_856_038,
        "temporary_write_headroom_bytes": 536_870_912,
        "operating_reserve_bytes": 4_294_967_296,
        "required_free_bytes": 10_190_604_096,
        "compression_credit_bytes": 0,
    }
    assert first["assumptions"]["elapsed_costs_included"] is False
    assert first["assumptions"]["maximum_evaluation_release_assumed"] is True


def test_schema_matches_detector_endpoints_and_pinned_plant_dimensions(inputs) -> None:
    protocol, base = inputs
    schemas = resources.estimate_observability_resources(protocol, base)["schemas"]
    arrays = {row["name"]: row for row in schemas["all_trial_frame_arrays"]}
    full = {row["name"]: row for row in schemas["full_wfs_arrays"]}

    assert list(arrays) == protocol["retention"]["all_trial_frame_arrays"]
    assert arrays["score"]["shape"] == [512]
    assert arrays["statistic_frame_id"]["shape"] == [381]
    assert arrays["alarm_frame_id"]["shape"] == [374]
    assert arrays["command"]["shape"] == [512, 100]
    assert arrays["applied"]["bytes_per_trial"] == 409_600
    assert list(full) == protocol["retention"]["full_wfs_arrays"]
    assert full["measurement"]["shape"] == [512, 64, 64]
    assert full["prediction"]["bytes_per_trial"] == 16_777_216
    assert schemas["all_trial_metrics"] == ["wfe_nm", "strehl", "saturation_fraction"]


def test_resource_module_import_does_not_load_jax_or_simulator() -> None:
    check = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import noether_optics.observability_resources; "
            "assert 'jax' not in sys.modules; "
            "assert 'noether_optics.simulation' not in sys.modules",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert check.returncode == 0, check.stderr


def test_loader_rechecks_the_exact_base_bytes_it_parses(inputs, tmp_path, monkeypatch) -> None:
    protocol, _ = inputs
    (tmp_path / "protocol.yaml").write_text("fixture: true\n")
    base_path = tmp_path / protocol["base_config"]
    base_path.parent.mkdir(parents=True)
    base_path.write_bytes((ROOT / protocol["base_config"]).read_bytes() + b"\n# drift\n")
    monkeypatch.setattr(resources, "load_protocol", lambda path, repository: deepcopy(protocol))

    with pytest.raises(ValueError, match="hash changed during resource loading"):
        resources.load_resource_inputs(tmp_path / "protocol.yaml", repository=tmp_path)


def test_loader_rejects_protocol_outside_repository(tmp_path) -> None:
    outside = tmp_path / "protocol.yaml"
    outside.write_text(CONFIG.read_text())
    with pytest.raises(ValueError, match="inside the source repository"):
        resources.load_resource_inputs(outside, repository=ROOT)


@pytest.mark.parametrize(
    ("field", "value"),
    [("sensor", "linear"), ("modes", True), ("modes", 0), ("detector_n", 2.5)],
)
def test_incompatible_base_contract_is_rejected(inputs, field, value) -> None:
    protocol, base = inputs
    base[field] = value
    with pytest.raises(ValueError):
        resources.estimate_observability_resources(protocol, base)


def test_preflight_checks_nearest_existing_ancestor_without_creating_target(
    inputs, tmp_path, monkeypatch
) -> None:
    protocol, base = inputs
    target = tmp_path / "new" / "observability"
    required = resources.estimate_observability_resources(protocol, base)["budget"][
        "required_free_bytes"
    ]
    checked = []

    def disk_usage(path):
        checked.append(Path(path))
        return SimpleNamespace(free=required)

    monkeypatch.setattr(resources.shutil, "disk_usage", disk_usage)
    report = resources.resource_preflight(protocol, base, target)

    assert report["admitted"] is True and report["status"] == "sufficient"
    assert report["filesystem"] == {"checked_path": str(tmp_path), "free_bytes": required}
    assert report["target"] == str(target.resolve())
    assert checked == [tmp_path]
    assert not target.exists()


def test_preflight_refuses_insufficient_space_and_file_target(
    inputs, tmp_path, monkeypatch
) -> None:
    protocol, base = inputs
    monkeypatch.setattr(resources.shutil, "disk_usage", lambda path: SimpleNamespace(free=1))
    report = resources.resource_preflight(protocol, base, tmp_path / "run")
    assert report["admitted"] is False and report["status"] == "insufficient"

    file_target = tmp_path / "file"
    file_target.write_text("occupied")
    with pytest.raises(ValueError, match="directory"):
        resources.resource_preflight(protocol, base, file_target)
