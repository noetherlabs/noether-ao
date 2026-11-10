"""Analytic physical-accounting checks independent of the JAX plant."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "physical_audit", Path(__file__).resolve().parents[1] / "scripts/audit_controller_physical_metrics.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def fixture():
    return dict(frames=4, n=2, modes=1, dt_s=.25,
                faults=dict(onset=1, end=3, vibration_rad=.5, vibration_hz=1))


def test_analytic_piston_removal_and_coherent_peak():
    cfg = fixture()
    basis = np.array([[[-1., 1.], [-1., 1.]]])
    phase = np.repeat((7 + .4*basis)[None, 0], 4, axis=0)
    result = audit.physical_metrics(phase, np.full((4, 1), .1), basis, np.ones((2, 2)), cfg, "none")
    np.testing.assert_allclose(result["variance"], .3**2, atol=1e-15)
    np.testing.assert_allclose(result["strehl"], np.cos(.3)**2, atol=1e-15)
    corrupted = audit.physical_metrics(phase, np.full((4, 1), .2), basis, np.ones((2, 2)), cfg, "none")
    assert not np.allclose(corrupted["variance"], result["variance"], atol=audit.ATOL, rtol=audit.RTOL)


def test_vibration_window_and_sensor_gain_do_not_conflate():
    cfg = fixture()
    basis = np.array([[[-1., 1.], [-1., 1.]]])
    args = (np.zeros((4, 2, 2)), np.zeros((4, 1)), basis, np.ones((2, 2)), cfg)
    result = audit.physical_metrics(*args, "vibration")
    np.testing.assert_allclose(result["variance"], [0, .25, 0, 0], atol=1e-15)
    np.testing.assert_array_equal(audit.physical_metrics(*args, "optical_gain")["variance"], 0)


def test_delay_and_modal_bounds():
    command = np.array([[.1], [-.2], [.3], [.4]])
    applied = np.array([[0.], [0.], [.1], [-.2]])
    audit.check_queue(command, applied, 2, .4)
    for delay, limit, changed in [(1, .4, applied), (2, .39, applied),
                                  (2, .4, applied + 1e-12)]:
        with pytest.raises(ValueError):
            audit.check_queue(command, changed, delay, limit)


@pytest.mark.parametrize("fault", ["stuck_actuator", "unknown"])
def test_unsupported_fault_not_silently_reinterpreted(fault):
    with pytest.raises(ValueError, match="Unsupported"):
        audit.physical_metrics(np.zeros((4, 2, 2)), np.zeros((4, 1)),
                               np.ones((1, 2, 2)), np.ones((2, 2)), fixture(), fault)


@pytest.mark.parametrize("corruption", [None, "variance", "delay", "digest"])
def test_retained_artifact_audit(tmp_path, corruption):
    run = tmp_path / "run"
    run.mkdir()

    def write(relative, value):
        path = run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".npz":
            np.savez(path, **value)
        else:
            path.write_text(json.dumps(value))
        return {"path": relative, "sha256": audit.digest(path)}

    basis = np.array([[[-1., 1.], [-1., 1.]]])
    cal = write("evidence/training/learned-00/calibration.npz",
                {"basis": basis, "mask": np.ones((2, 2))})
    manifest = write("manifest.json", {})
    training = write("training.json", {"evidence": [cal]})
    selection = write("selection.json", {})
    report = {"status": "complete_descriptive_report",
              "manifest_sha256": manifest["sha256"],
              "training_sha256": training["sha256"], "selection_sha256": selection["sha256"]}
    cfg = dict(fixture(), delay_frames=2, command_clip_rad=1.)
    for stage in ("validation", "evaluation"):
        case = {"stage": stage, "fault": "none"}
        phase = write(f"evidence/{stage}/phase.npz", {"phase": np.repeat(.4*basis, 4, axis=0)})
        physical = {"variance": np.full(4, .16), "strehl": np.full(4, np.cos(.4)**2)}
        raw = dict(physical, command=np.zeros((4, 1)), applied=np.zeros((4, 1)))
        if stage == "evaluation" and corruption == "variance":
            raw["variance"] = raw["variance"] + .01
        if stage == "evaluation" and corruption == "delay":
            raw["applied"][0, 0] = .1
        raw_ref = write(f"evidence/{stage}/raw.npz", raw)
        reference = write(f"evidence/{stage}/open.npz", physical)
        detail = write(f"evidence/{stage}/trials/one.json", {
            "case": case, "config": cfg, "atmosphere": [phase],
            "raw": raw_ref, "open_loop": [reference]})
        write(f"{stage}/one.json", {"record": {
            "status": "completed", "case": case, "artifacts": [detail]}})
        report[stage + "_records"] = 1
        report[stage + "_trials"] = [{"case": case, "status": "completed"}]
    if corruption == "digest":
        (run / cal["path"]).write_bytes(b"changed")
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report))
    if corruption:
        messages = {"variance": "Physical metric mismatch", "delay": "declared delay", "digest": "digest mismatch"}
        with pytest.raises(ValueError, match=messages[corruption]):
            audit.audit(run, report_path)
    else:
        result = audit.audit(run, report_path)
        assert result["status"] == "physical_accounting_verified"
        assert len(result["terminal_records"]) == 2
        assert result["report_sha256"] == audit.digest(report_path)
