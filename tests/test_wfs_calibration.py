"""Calibration semantics and independent optical checks on non-study inputs."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from noether_optics.artifacts import load_config
from noether_optics.wfs_calibration import (
    acquire, backend, make_inputs, metrics, phase_rms, reconstruct, transfer,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    c = load_config(ROOT / "configs/wfs_calibration.yaml")
    return c | {"grids": [16], "detector_n": 32, "heldout_seed": 734,
                "noise_seed": 735, "heldout_count": 3, "noise_count": 32,
                "phase_rms_rad": [0.001, 0.04]}


def test_phase_error_uses_sampled_gram():
    basis = np.array([[1., -1, 1, -1], [2, 0, -2, 0]])
    basis[1] += basis[0]
    coefficients = np.array([[.2, .4], [.1, -.3]])
    expected = np.sqrt(np.mean((coefficients @ basis)**2, axis=1))
    assert np.allclose(phase_rms(coefficients, basis @ basis.T / 4), expected)
    assert not np.allclose(expected, np.linalg.norm(coefficients, axis=1))


def test_svd_rank_and_cutoff():
    recon, _, rank = reconstruct(np.diag([2., 0.]), 1e-10)
    assert rank == 1
    np.testing.assert_allclose(recon, np.diag([.5, 0.]))
    with pytest.raises(ValueError, match="no observable modes"):
        reconstruct(np.zeros((3, 2)), 1e-10)
    with pytest.raises(ValueError, match="finite"):
        reconstruct(np.array([[np.nan]]), 1e-10)


def test_linear_calibration_controls_noise_and_reference_transfer(config):
    inputs = make_inputs(config, 16)
    basis = inputs["modes"][:, inputs["mask"]]
    projection = np.linalg.solve(inputs["gram"], basis / basis.shape[1])
    response = np.eye(len(basis)) * .01
    reference = np.ones(len(basis)) * .1

    def sensor(phase):
        return reference + response @ (projection @ phase[inputs["mask"]])

    raw = acquire(sensor, inputs, config)
    result = metrics(raw, inputs, config)
    assert result["all_pass"]
    assert max(result["phase_relative_rms"]) < 1e-9
    assert min(result["negative_controls"].values()) > .9
    expected = np.diag((.1 / config["photons"] + (1 / config["photons"])**2) / .01**2 * np.ones(5))
    np.testing.assert_allclose(result["predicted_noise_covariance"], expected, rtol=1e-9)
    shifted = raw | {"reference": raw["reference"] + .01, "images": raw["images"] + .01}
    cross = transfer(raw, shifted, inputs, config)
    assert max(cross["matrix_only"]) < 1e-9
    assert min(cross["matrix_and_reference"]) > 1


@pytest.mark.parametrize("engine", ["noether", "hcipy_matched", "hcipy_native"])
def test_independent_optical_calibration(engine, config):
    inputs = make_inputs(config, 16)
    sensor, _ = backend(engine, config, 16, inputs["mask"])
    raw = acquire(sensor, inputs, config, autodiff=engine == "noether")
    result = metrics(raw, inputs, config)
    assert result["all_pass"], result["gates"]


def runner():
    spec = importlib.util.spec_from_file_location("wfs_runner", ROOT / "scripts/compare_wfs_calibration.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_protocol_rejects_duplicate_and_overlapping_inputs(config):
    module = runner()
    for invalid in (config | {"grids": [16, 16]},
                    config | {"noise_seed": config["heldout_seed"]},
                    config | {"poke_rad": 0}):
        with pytest.raises(ValueError):
            module.validate(invalid)


def test_no_clobber_and_resource_refusal_precede_acquisition(config, tmp_path, monkeypatch):
    module = runner()
    from collections import namedtuple
    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(module.shutil, "disk_usage", lambda p: Usage(10, 9, 1))
    path = tmp_path / "no-run"
    with pytest.raises(RuntimeError, match="reserve"):
        module.run(config, path, Path("missing-python"))
    assert not path.exists()
    monkeypatch.setattr(module.shutil, "disk_usage", lambda p: Usage(10**10, 0, 10**10))
    path.mkdir()
    with pytest.raises(FileExistsError):
        module.run(config, path, Path("missing-python"))
    assert not list(path.iterdir())


def test_rebuild_rejects_tampering_and_incomplete_cohort(config, tmp_path):
    module = runner()
    module.dump(tmp_path / "config.json", config)
    module.dump(tmp_path / "manifest.json", {})
    with pytest.raises(ValueError, match="inventory"):
        module.rebuild(tmp_path)
    expected = {"config.json", "provenance.json", "machine.json", "runner.py", "raw/inputs_n16.npz"}
    for engine in module.ENGINES:
        expected.update({f"raw/{engine}_n16.npz", f"raw/{engine}_n16.json"})
    for name in expected - {"config.json"}:
        p = tmp_path / name
        p.parent.mkdir(exist_ok=True)
        p.write_bytes(b"test evidence")
    module.dump(tmp_path / "manifest.json", {name: module.sha(tmp_path / name) for name in expected})
    (tmp_path / "raw/inputs_n16.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        module.rebuild(tmp_path)


def test_cli_preserves_virtual_environment_python_symlink(tmp_path, monkeypatch):
    module = runner()
    python = tmp_path / "python"
    executable = tmp_path / "real-python"
    executable.touch()
    python.symlink_to(executable)
    calls = []
    monkeypatch.setattr(module, "run", lambda cfg, path, runtime: calls.append(runtime))
    monkeypatch.setattr(module, "rebuild", lambda path: {"all_required_gates_pass": True})
    monkeypatch.setattr(module.sys, "argv", ["compare", "--config", str(ROOT / "configs/wfs_calibration.yaml"),
                        "--out", str(tmp_path / "out"), "--specula-python", str(python)])
    module.main()
    assert calls == [python]
