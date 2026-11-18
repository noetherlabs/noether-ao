"""Tiny physical recordings and refusal checks for commissioning/service custody."""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("noether_physics")

from noether_optics import receiver_calibration_runner as runner


ROOT = Path(__file__).resolve().parents[1]
MATRIX_FIELDS = {
    "command", "applied", "estimate", "measurement", "prediction",
    "feedback_requested", "probe_requested", "combined_requested", "measurement_perturbation",
}
SCALAR_FIELDS = {"variance", "strehl", "score"}
BOOLEAN_FIELDS = {"alarm", "fallback", "saturated", "unavailable"}


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def read_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def inventory(directory):
    return {path.relative_to(directory).as_posix(): file_sha(path)
            for path in sorted(directory.rglob("*")) if path.is_file()}


def fixture_audit(protocol):
    """Derive only the independent unit namespace's metadata, without sampling."""
    row = {"root": protocol["roots"][0]}
    for role in ("commissioning-phase", "commissioning-detector-base", "service-phase",
                 "service-detector-base", "probe"):
        label = f"{protocol['study_id']}/{row['root']}/{role}".encode()
        row[role] = int.from_bytes(hashlib.sha256(label).digest()[:4], "big")
    for recording in ("commissioning", "service"):
        assert row[f"{recording}-detector-base"] < 2**32 - 100000
        row[f"{recording}-detector"] = row[f"{recording}-detector-base"] + 100000
    assert len(set(value for key, value in row.items() if key != "root")) == 7
    return dict(schema="receiver-calibration-seed-audit/1", status="scoped_disjoint",
                collisions=[], unique=True, uint32_bounds_valid=True, detector_offset_no_wrap=True,
                namespace=protocol["study_id"], rows=[row], integer_identity_count=7,
                active_rng_root_count=5, detector_base_argument_count=2,
                scope="Unit-fixture namespace; not a scientific comparison-stream audit")


@pytest.fixture
def tiny_case(tmp_path, monkeypatch):
    # This patch replaces only provenance admission for functional unit fixtures.
    # Calibration, atmosphere, loop, response fit and receiver remain real code.
    # No comparison stream table or retained scientific recording is loaded.
    protocol = read_json(ROOT / "configs/receiver_calibration_development_v1.json")
    protocol["study_id"] = "receiver-calibration-runner-unit-fixture-v1"
    protocol["roots"] = ["fixture-root"]
    protocol["windows"] = dict(commissioning_frames=16, probe_start=2, probe_stop=10,
                               service_frames=8, delay_frames=2)
    protocol["base_config"].update(n=8, modes=3, detector_n=8, screen_factor=2,
                                   r0_m=1e6)
    protocol["conditions"] = [dict(id="fixture-condition", sensor_response="nominal", photons=1e12)]
    protocol["probe"]["amplitude_rad"] = .5
    protocol["fit"]["maximum_reconstructor_gain"] = 1e-12
    protocol["endpoint"].update(commissioning_window=[2, 16], service_window=[0, 8],
                                charged_frames=22)
    audit_value = fixture_audit(protocol)
    streams = audit_value["rows"][0]
    config = tmp_path / "fixture-config.json"
    audit = tmp_path / "fixture-seeds.json"
    config.write_text(json.dumps(protocol, sort_keys=True) + "\n")
    audit.write_text(json.dumps(audit_value, sort_keys=True) + "\n")
    captured = dict(protocol=protocol, condition=protocol["conditions"][0], streams=streams,
                    source={"schema_version": 1, "commit": "unit-fixture", "files": {}},
                    runtime={"fixture": True}, packages={}, config_sha256=file_sha(config),
                    seed_audit_sha256=file_sha(audit), root_id="fixture-root",
                    condition_id="fixture-condition")
    monkeypatch.setattr(runner, "capture_inputs", lambda *args, **kwargs: deepcopy(captured))
    options = dict(root_id="fixture-root", condition_id="fixture-condition",
                   expected_config_sha256=file_sha(config), expected_seed_audit_sha256=file_sha(audit),
                   repository=ROOT)
    return config, audit, options, captured


def test_existing_output_refuses_before_preflight_and_preserves_every_byte(tmp_path, monkeypatch):
    output = tmp_path / "existing"
    output.mkdir()
    (output / "sentinel").write_bytes(b"preserve this recording")
    before = inventory(output)

    def forbidden(*args, **kwargs):
        pytest.fail("An existing destination must refuse before preflight or acquisition")

    monkeypatch.setattr(runner, "capture_inputs", forbidden)
    with pytest.raises(FileExistsError):
        runner.run_trial(tmp_path / "absent-config", tmp_path / "absent-audit",
                         root_id="fixture-root", condition_id="fixture-condition", output=output,
                         expected_config_sha256="0" * 64, expected_seed_audit_sha256="0" * 64,
                         repository=ROOT)
    assert inventory(output) == before


@pytest.mark.parametrize("mismatch", ["config", "seed_audit"])
def test_input_hash_mismatch_refuses_before_runtime_or_acquisition(tmp_path, monkeypatch, mismatch):
    config, audit = tmp_path / "config.json", tmp_path / "audit.json"
    config.write_text("{}\n")
    audit.write_text("{}\n")
    options = dict(root_id="fixture-root", condition_id="fixture-condition", repository=ROOT,
                   expected_config_sha256=file_sha(config), expected_seed_audit_sha256=file_sha(audit))
    options["expected_config_sha256" if mismatch == "config" else "expected_seed_audit_sha256"] = "0" * 64
    monkeypatch.setattr(runner, "_source", lambda *args: pytest.fail("Source must follow input hashes"))
    monkeypatch.setattr(runner, "_runtime", lambda: pytest.fail("Runtime must follow input hashes"))
    with pytest.raises((ValueError, RuntimeError), match="hash|SHA|sha|changed|identity"):
        runner.capture_inputs(config, audit, **options)


def admission_fixture(tmp_path):
    """Metadata-only provenance fixture with no sampled atmosphere or detector."""
    protocol = read_json(ROOT / "configs/receiver_calibration_development_v1.json")
    protocol["study_id"] = "receiver-runner-admission-unit-fixture-v1"
    protocol["roots"] = ["fixture-root"]
    audit = fixture_audit(protocol)
    config_path, audit_path = tmp_path / "config.json", tmp_path / "audit.json"
    config_path.write_text(json.dumps(protocol) + "\n")
    audit_path.write_text(json.dumps(audit) + "\n")
    options = dict(root_id="fixture-root", condition_id=protocol["conditions"][0]["id"],
                   repository=tmp_path, expected_config_sha256=file_sha(config_path),
                   expected_seed_audit_sha256=file_sha(audit_path))
    return config_path, audit_path, options


@pytest.mark.parametrize("mismatch", ["root", "condition", "seed", "source"])
def test_admission_identity_mismatch_precedes_packages_and_runtime(tmp_path, monkeypatch, mismatch):
    config, audit, options = admission_fixture(tmp_path)
    if mismatch in ("root", "condition"):
        options[f"{mismatch}_id"] = "undeclared"
    elif mismatch == "seed":
        value = read_json(audit)
        value["rows"][0]["probe"] ^= 1
        audit.write_text(json.dumps(value) + "\n")
        options["expected_seed_audit_sha256"] = file_sha(audit)
    monkeypatch.setattr(runner, "_source", lambda *args: {"files": {"config.json": "0" * 64}})
    monkeypatch.setattr(runner, "_packages", lambda *args: pytest.fail("Wrong identities reached packages"))
    monkeypatch.setattr(runner, "_runtime", lambda: pytest.fail("Wrong identities reached runtime"))
    with pytest.raises(ValueError, match="declared|Seed table|committed execution input"):
        runner.capture_inputs(config, audit, **options)


def direct_coupling(phase, applied, calibration, protocol):
    """Independent dimensional pupil integral, without the receiver adapter."""
    mask, basis = calibration["mask"], calibration["basis"]
    base, receiver = protocol["base_config"], protocol["receiver"]
    pitch = base["diameter_m"] / base["n"]
    wavelength = receiver["receiver_wavelength_m"]
    radius = wavelength * receiver["focal_length_m"] / (math.pi * receiver["mode_radius_m"])
    edges = (np.arange(base["n"] + 1) - base["n"] / 2) * pitch
    integrals = np.array([radius * math.sqrt(math.pi) / 2 *
                          (math.erf(right / radius) - math.erf(left / radius))
                          for left, right in zip(edges[:-1], edges[1:])])
    weights = np.outer(integrals, integrals)
    pupil_power = np.sum(mask**2) * pitch**2
    mode_power = math.pi * radius**2 / 2
    values = []
    for frame, coefficients in zip(phase, applied):
        residual = frame.copy()
        for coefficient, mode in zip(coefficients, basis):
            residual -= coefficient * mode
        residual *= mask
        residual -= mask * residual.sum() / mask.sum()
        receiver_phase = residual * base["wavelength_m"] / wavelength
        overlap = np.sum(mask * np.exp(1j * receiver_phase) * weights)
        values.append(abs(overlap)**2 / (pupil_power * mode_power))
    return np.asarray(values)


def assert_terminal_inventory(output, result):
    assert set(result["files"]) == set(inventory(output)) - {"result.json"}
    assert result["missing_artifacts"] == []
    for name, record in result["files"].items():
        assert record["sha256"] == file_sha(output / name)
        assert record["bytes"] == (output / name).stat().st_size


def test_real_trial_freezes_before_service_and_retains_negative_probe_utility(tiny_case, tmp_path, monkeypatch):
    config, audit, options, captured = tiny_case
    protocol, streams = captured["protocol"], captured["streams"]
    output = tmp_path / "trial"
    real_phase, real_kernel = runner.phase_sequence, runner.instrumented_kernel
    real_fit, real_receiver = runner.fit_commissioning_response, runner.evaluate_receiver
    phase_calls, kernel_calls, receiver_calls, fit_calls = [], [], [], []

    def assert_frozen():
        freeze = read_json(output / "fit-freeze.json")
        assert freeze["fit_sha256"] == file_sha(output / "fit.json")
        return freeze["fit_sha256"]

    def phase(config, seed):
        if seed == streams["service-phase"]:
            assert fit_calls and not receiver_calls
            assert_frozen()
        phase_calls.append(seed)
        return real_phase(config, seed)

    def fit(measurement, issued, applied, requested, **kwargs):
        assert not (output / "service-phase.npz").exists()
        recorded = read_arrays(output / "commissioning-probed.npz")
        for value, key in ((measurement, "measurement"), (issued, "command"),
                           (applied, "applied"), (requested, "probe_requested")):
            np.testing.assert_array_equal(value, recorded[key])
        assert set(kwargs) == {"nominal_reconstructor", "nominal_interaction", "detector_reference",
                               "windows", "maximum_reconstructor_gain", "minimum_relevance", "rcond"}
        fit_calls.append(True)
        return real_fit(measurement, issued, applied, requested, **kwargs)

    def kernel(config, cal, **kwargs):
        execute = real_kernel(config, cal, **kwargs)

        def observed(phase, seed, requested, perturbation):
            result = execute(phase, seed, requested, perturbation)
            kernel_calls.append(dict(seed=seed, phase=np.asarray(phase).copy(),
                                     requested=requested.copy(), perturbation=perturbation.copy(),
                                     outputs={key: np.asarray(value).copy() for key, value in result.items()}))
            return result
        return observed

    def receiver(*args, **kwargs):
        receiver_calls.append(assert_frozen())
        return real_receiver(*args, **kwargs)

    monkeypatch.setattr(runner, "phase_sequence", phase)
    monkeypatch.setattr(runner, "instrumented_kernel", kernel)
    monkeypatch.setattr(runner, "fit_commissioning_response", fit)
    monkeypatch.setattr(runner, "evaluate_receiver", receiver)
    result = runner.run_trial(config, audit, output=output, **options)
    assert result == read_json(output / "result.json")
    assert result["status"] == "completed" and result["fit_status"] == "withheld"
    assert phase_calls == [streams["commissioning-phase"], streams["service-phase"]]
    assert len(fit_calls) == 1 and receiver_calls == [assert_frozen()] * 4
    assert len(kernel_calls) == 3  # Withholding transparently reuses the nominal service recording.
    assert [row["seed"] for row in kernel_calls] == [streams["commissioning-detector-base"]] * 2 + [streams["service-detector-base"]]
    np.testing.assert_array_equal(kernel_calls[0]["phase"], kernel_calls[1]["phase"])
    assert np.count_nonzero(kernel_calls[2]["requested"]) == 0
    assert all(np.count_nonzero(row["perturbation"]) == 0 for row in kernel_calls)
    calibration = read_arrays(output / "calibration.npz")
    decision = read_json(output / "fit.json")["decision"]
    np.testing.assert_array_equal(decision["reconstructor"], calibration["reconstructor"])
    np.testing.assert_array_equal(decision["detector_reference"], calibration["reference"])
    traces = {}
    for index, label in enumerate(("commissioning-nominal", "commissioning-probed", "service-nominal", "service-corrected")):
        role = label.split("-")[0]
        phase = read_arrays(output / f"{role}-phase.npz")["phase"]
        raw = read_arrays(output / f"{label}.npz")
        assert set(raw) == MATRIX_FIELDS | SCALAR_FIELDS | BOOLEAN_FIELDS
        for key, value in raw.items():
            np.testing.assert_array_equal(value, kernel_calls[min(index, 2)]["outputs"][key])
            assert value.dtype == np.dtype(bool if key in BOOLEAN_FIELDS else np.float64)
        assert not raw["applied"][:2].any()
        np.testing.assert_array_equal(raw["applied"][2:], raw["command"][:-2])
        reconstructed = direct_coupling(phase, raw["applied"], calibration, protocol)
        traces[label] = read_arrays(output / f"receiver-{label}.npz")["coupling"]
        np.testing.assert_allclose(traces[label], reconstructed, rtol=2e-12, atol=2e-14)
    cost = float(np.sum(traces["commissioning-nominal"][2:] - traces["commissioning-probed"][2:]))
    gain = float(np.sum(traces["service-corrected"] - traces["service-nominal"]))
    assert cost > 0 and gain == 0
    summary = read_json(output / "summary.json")
    assert summary["diagnostics"]["service_copied_from_nominal_after_withholding"] is True
    assert summary["utility"]["commissioning_cost"] == pytest.approx(cost)
    assert summary["utility"]["service_gain"] == 0
    assert summary["utility"]["charged_frames"] == 22
    assert summary["utility"]["total_frames"] == 24
    assert summary["utility"]["total_mean_delta"] == pytest.approx(-cost / 24)
    assert summary["utility"]["coupling_time_delta_s"] == pytest.approx(-cost * protocol["base_config"]["dt_s"])
    assert result["charged_mean_delta"] == pytest.approx(-cost / 22)
    assert result["utility_bounds"] == [result["charged_mean_delta"]] * 2
    assert_terminal_inventory(output, result)
    assert result["logical_scientific_bytes"] == sum(
        row["bytes"] for item in result["files"].values() for row in item.get("arrays", {}).values())
    assert sum(path.stat().st_size for path in output.iterdir()) < protocol["artifact_limits"]["retained_bytes_per_condition"]


def test_nonfinite_real_output_is_retained_and_cannot_be_overwritten(tiny_case, tmp_path, monkeypatch):
    config, audit, options, captured = tiny_case
    output = tmp_path / "nonfinite"
    real_kernel, real_phase = runner.instrumented_kernel, runner.phase_sequence
    produced = {}

    def phase(config, seed):
        assert seed != captured["streams"]["service-phase"], "Failed commissioning reached service"
        return real_phase(config, seed)

    def kernel(*args, **kwargs):
        execute = real_kernel(*args, **kwargs)

        def corrupted(*arguments):
            result = {key: np.asarray(value).copy() for key, value in execute(*arguments).items()}
            result["variance"][-1] = np.nan
            produced.update(result)
            return result
        return corrupted

    monkeypatch.setattr(runner, "phase_sequence", phase)
    monkeypatch.setattr(runner, "instrumented_kernel", kernel)
    with pytest.raises(RuntimeError, match="retained result.json") as error:
        runner.run_trial(config, audit, output=output, **options)
    assert isinstance(error.value.__cause__, ValueError)
    raw = read_arrays(output / "commissioning-nominal.npz")
    for key in produced:
        np.testing.assert_array_equal(raw[key], produced[key])
    assert np.isnan(raw["variance"][-1])
    result = read_json(output / "result.json")
    assert result["status"] == "failed" and result["charged_mean_delta"] is None
    assert result["utility_bounds"] == [-1, 1] and result["stage"] == "commissioning_nominal"
    assert not (output / "fit.json").exists() and not (output / "service-phase.npz").exists()
    assert_terminal_inventory(output, result)
    before = inventory(output)
    with pytest.raises(FileExistsError):
        runner.run_trial(config, audit, output=output, **options)
    assert inventory(output) == before


def test_injected_ready_decision_reaches_only_the_fourth_physical_loop(tiny_case, tmp_path, monkeypatch):
    """Exercise ready-decision wiring; the injected fit is not fit-quality evidence."""
    config, audit, options, captured = tiny_case
    captured["protocol"]["fit"]["maximum_reconstructor_gain"] = 3.
    config.write_text(json.dumps(captured["protocol"], sort_keys=True) + "\n")
    options["expected_config_sha256"] = captured["config_sha256"] = file_sha(config)
    output = tmp_path / "ready-wiring"
    real_kernel = runner.instrumented_kernel
    calibrations, recordings = [], []

    def known_decision(*args, nominal_reconstructor, detector_reference, windows, **kwargs):
        matrix = .9 * nominal_reconstructor
        return dict(status="ready", reason=None, reconstructor=matrix,
                    detector_reference=detector_reference.copy(),
                    response=np.eye(matrix.shape[0], dtype=np.float64) / .9,
                    fit={"status": "ready", "fixture": "injected wiring decision"},
                    restoration={"status": "ready", "fixture": "injected wiring decision"},
                    fit_rows=(windows.fit_start, windows.fit_stop))

    def kernel(config, cal, **kwargs):
        number = len(calibrations)
        calibrations.append({key: np.asarray(cal[key]).copy()
                             for key in ("reconstructor", "reference", "interaction", "basis", "mask")})
        if number >= 2:
            freeze = read_json(output / "fit-freeze.json")
            assert freeze["fit_sha256"] == file_sha(output / "fit.json")
            nominal = read_arrays(output / "calibration.npz")["reconstructor"]
            expected = nominal if number == 2 else np.asarray(read_json(output / "fit.json")["decision"]["reconstructor"])
            np.testing.assert_array_equal(cal["reconstructor"], expected)
        execute = real_kernel(config, cal, **kwargs)

        def observed(phase, seed, requested, perturbation):
            recordings.append((np.asarray(phase).copy(), seed, requested.copy(), perturbation.copy()))
            return execute(phase, seed, requested, perturbation)
        return observed

    monkeypatch.setattr(runner, "fit_commissioning_response", known_decision)
    monkeypatch.setattr(runner, "instrumented_kernel", kernel)
    result = runner.run_trial(config, audit, output=output, **options)
    assert result["status"] == "completed" and result["fit_status"] == "ready"
    assert len(calibrations) == len(recordings) == 4
    for index in range(3):
        np.testing.assert_array_equal(calibrations[index]["reconstructor"], calibrations[0]["reconstructor"])
    np.testing.assert_array_equal(calibrations[3]["reconstructor"], .9 * calibrations[0]["reconstructor"])
    for key in ("reference", "interaction", "basis", "mask"):
        np.testing.assert_array_equal(calibrations[3][key], calibrations[2][key])
    np.testing.assert_array_equal(recordings[2][0], recordings[3][0])
    assert recordings[2][1] == recordings[3][1] == captured["streams"]["service-detector-base"]
    for recording in recordings[2:]:
        assert not recording[2].any() and not recording[3].any()
    for label in ("service-nominal", "service-corrected"):
        raw = read_arrays(output / f"{label}.npz")
        assert not raw["applied"][:2].any()
    assert read_json(output / "summary.json")["diagnostics"]["service_copied_from_nominal_after_withholding"] is False
    assert_terminal_inventory(output, result)


def test_changed_frozen_fit_refuses_service_loop_and_retains_changed_bytes(tiny_case, tmp_path, monkeypatch):
    config, audit, options, captured = tiny_case
    output = tmp_path / "changed-fit"
    real_phase, real_kernel = runner.phase_sequence, runner.instrumented_kernel
    original_hash = []

    def phase(config, seed):
        if seed == captured["streams"]["service-phase"]:
            original_hash.append(read_json(output / "fit-freeze.json")["fit_sha256"])
            fit = read_json(output / "fit.json")
            fit["decision"]["reconstructor"][0][0] += 1
            (output / "fit.json").write_text(json.dumps(fit) + "\n")
        return real_phase(config, seed)

    def kernel(config, *args, **kwargs):
        assert config["frames"] != 8, "Changed fit reached a service controller"
        return real_kernel(config, *args, **kwargs)

    monkeypatch.setattr(runner, "phase_sequence", phase)
    monkeypatch.setattr(runner, "instrumented_kernel", kernel)
    with pytest.raises(RuntimeError) as error:
        runner.run_trial(config, audit, output=output, **options)
    assert isinstance(error.value.__cause__, ValueError)
    result = read_json(output / "result.json")
    assert result["status"] == "failed" and result["stage"] == "service_phase"
    assert (output / "service-phase.npz").exists() and not (output / "service-nominal.npz").exists()
    assert result["files"]["fit.json"]["original_sha256"] == original_hash[0]
    assert result["files"]["fit.json"]["sha256"] != original_hash[0]
    assert result["fit_sha256"] == original_hash[0]
    assert_terminal_inventory(output, result)


@pytest.mark.parametrize("budget", ["logical", "retained"])
def test_too_small_artifact_budget_refuses_before_calibration(tiny_case, tmp_path, monkeypatch, budget):
    config, audit, options, captured = tiny_case
    key = f"{budget}_scientific_bytes_per_condition" if budget == "logical" else "retained_bytes_per_condition"
    captured["protocol"]["artifact_limits"][key] = 1 if budget == "logical" else 65536
    monkeypatch.setattr(runner, "calibration", lambda *args: pytest.fail("Insufficient budget reached optics"))
    output = tmp_path / "too-small"
    with pytest.raises(RuntimeError) as error:
        runner.run_trial(config, audit, output=output, **options)
    assert isinstance(error.value.__cause__, ValueError)
    result = read_json(output / "result.json")
    assert result["status"] == "failed" and result["files"] == {}
    assert result["charged_mean_delta"] is None and result["utility_bounds"] == [-1, 1]
    assert set(inventory(output)) == {"result.json"}
    assert (output / "result.json").stat().st_size < 65536


def valid_loop_fixture():
    config = {"frames": 6, "modes": 2, "delay_frames": 2, "command_clip_rad": 1.}
    requested = np.zeros((6, 2), dtype=np.float64)
    requested[1, 0] = 2.
    outputs = {key: np.zeros((6, 2), dtype=np.float64) for key in MATRIX_FIELDS}
    outputs.update({key: np.zeros(6, dtype=np.float64) for key in SCALAR_FIELDS})
    outputs.update({key: np.zeros(6, dtype=bool) for key in BOOLEAN_FIELDS})
    outputs["probe_requested"] = requested.copy()
    outputs["combined_requested"] = requested.copy()
    outputs["command"][1, 0] = 1.
    outputs["applied"][3, 0] = 1.
    outputs["saturated"][1] = True
    return config, requested, outputs


@pytest.mark.parametrize("corruption", ["missing", "extra", "object", "float32", "shape", "nonfinite",
                                         "fifo", "requested", "perturbation", "clipping", "saturation", "unavailable"])
def test_loop_schema_refuses_corrupted_original_fields(corruption):
    config, requested, raw = valid_loop_fixture()
    runner._validate_outputs(raw, config, requested)
    if corruption == "missing":
        del raw["prediction"]
    elif corruption == "extra":
        raw["new_field"] = np.zeros(6)
    elif corruption in ("object", "float32"):
        raw["variance"] = raw["variance"].astype(corruption)
    elif corruption == "shape":
        raw["measurement"] = raw["measurement"][:-1]
    elif corruption == "nonfinite":
        raw["estimate"][0, 0] = np.inf
    elif corruption == "fifo":
        raw["applied"][3, 0] = 0
    elif corruption == "requested":
        raw["probe_requested"][0, 0] = 1
    elif corruption == "perturbation":
        raw["measurement_perturbation"][0, 0] = .01
    elif corruption == "clipping":
        raw["command"][1, 0] = 2
        raw["applied"][3, 0] = 2
    elif corruption == "saturation":
        raw["saturated"][1] = False
    else:
        raw["unavailable"][0] = True
    with pytest.raises(ValueError):
        runner._validate_outputs(raw, config, requested)
