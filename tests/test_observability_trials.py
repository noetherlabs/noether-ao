"""Physical-adapter contract checks that do not acquire a declared cohort."""
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from noether_optics.artifacts import load_config
from noether_optics.controller_evidence import save_metadata
from noether_optics.controller_selection import _bytes, _digest
from noether_optics.observability_detector import detector_trace, training_cases
from noether_optics.observability_protocol import load_protocol, protocol_plan
from noether_optics.observability_run import ObservabilityLedger
from noether_optics.observability_trials import (
    CORE_ARRAYS,
    _TrialExecutor,
    complete_trial_arrays,
    numerical_failure,
    open_observability_run,
    physical_core_arrays,
    retain_full_wfs,
    train_trials,
    trial_config,
    trial_metadata,
    trial_metrics,
    verify_observability_provenance,
)
from noether_optics.observability_validation import validation_cases


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/optical_gain_observability_v1.yaml"
PROVENANCE = {
    "source_commit": "a" * 40,
    "source_sha256": "b" * 64,
    "runtime_sha256": "c" * 64,
}


@pytest.fixture
def protocol() -> dict:
    return load_protocol(CONFIG)


@pytest.fixture
def base() -> dict:
    return load_config(ROOT / "configs/simulation.yaml")


def _light_base(base: dict) -> dict:
    value = deepcopy(base)
    value.update(modes=3, detector_n=2)
    return value


def _outputs(protocol: dict, base: dict, *, wfs: bool = False) -> dict:
    frames = protocol["loop"]["frames"]
    result = {
        "score": np.linspace(2.0, 3.0, frames, dtype=np.float64),
        "variance": np.full(frames, 4.0, dtype=np.float64),
        "strehl": np.full(frames, 0.6, dtype=np.float64),
        "saturated": np.zeros(frames, dtype=bool),
        "fallback": np.zeros(frames, dtype=bool),
        "unavailable": np.zeros(frames, dtype=bool),
        "command": np.zeros((frames, base["modes"]), dtype=np.float64),
        "applied": np.zeros((frames, base["modes"]), dtype=np.float64),
    }
    if wfs:
        pixels = base["detector_n"] ** 2
        result["measurement"] = np.ones((frames, pixels), dtype=np.float64)
        result["prediction"] = np.full((frames, pixels), 0.5, dtype=np.float64)
    return result


def _bound_ledger(tmp_path: Path, protocol: dict, base: dict) -> ObservabilityLedger:
    source = {"schema_version": 1, "commit": "a" * 40, "files": {}}
    runtime = {"machine": {"x64": True}, "fixture": True}
    provenance = {
        "source_commit": source["commit"],
        "source_sha256": _digest(_bytes(source)),
        "runtime_sha256": _digest(_bytes(runtime)),
    }
    ledger = ObservabilityLedger.create(tmp_path / "run", CONFIG, provenance)
    values = {
        "source": source,
        "runtime": runtime,
        "protocol": protocol,
        "base": base,
    }
    for name, value in values.items():
        save_metadata(
            ledger,
            f"evidence/provenance/{name}.json",
            value,
        )
    return ledger


def _evaluation_case(protocol: dict, condition: str) -> dict:
    plan = protocol_plan(protocol)
    regime, stream = plan["regimes"][0], plan["seed_streams"]["evaluation"][0]
    return {
        "role": "evaluation",
        "condition": condition,
        "regime_id": regime["id"],
        "photons": regime["photons"],
        "wind_scale": regime["wind_scale"],
        "atmosphere_seed": stream["atmosphere"],
        "detector_seed": stream["detector"],
    }


@pytest.mark.parametrize(
    ("condition", "fault", "field", "value"),
    [
        ("clean", "none", None, None),
        ("gain_035", "optical_gain", "optical_gain", 0.35),
        ("gain_065", "optical_gain", "optical_gain", 0.65),
        ("vibration", "vibration", "vibration_rad", 0.7),
        ("misregistration", "misregistration", "shift_pixels", 2),
    ],
)
def test_trial_config_maps_only_frozen_conditions(
    protocol, base, condition, fault, field, value
) -> None:
    case = (
        training_cases(protocol)[0]
        if condition == "clean"
        else _evaluation_case(protocol, condition)
    )
    original = deepcopy(base)
    cfg, selected = trial_config(base, protocol, case)

    assert base == original
    assert selected == fault
    assert cfg["frames"] == 512 and cfg["burn_in"] == 128
    assert cfg["gain"] == 0.3 and cfg["leak"] == 0.999
    assert cfg["faults"]["onset"] == 256 and cfg["faults"]["end"] == 512
    if field is not None:
        assert cfg["faults"][field] == value


def test_wfs_policy_reshapes_only_the_first_seed(protocol, base) -> None:
    small = _light_base(base)
    first, second = training_cases(protocol)[:2]
    first_arrays = physical_core_arrays(
        protocol, small, first, _outputs(protocol, small, wfs=True)
    )
    second_arrays = physical_core_arrays(
        protocol, small, second, _outputs(protocol, small, wfs=True)
    )

    assert retain_full_wfs(protocol, first) is True
    assert first_arrays["measurement"].shape == (512, 2, 2)
    assert first_arrays["prediction"].dtype == np.dtype(np.float64)
    assert retain_full_wfs(protocol, second) is False
    assert set(second_arrays) == set(CORE_ARRAYS)


def test_validation_archive_and_metrics_reproduce_frozen_detector(protocol, base) -> None:
    small = _light_base(base)
    case = validation_cases(protocol)[1]
    outputs = _outputs(protocol, small)
    threshold = 2.5
    arrays = complete_trial_arrays(protocol, small, case, outputs, threshold)
    trace = detector_trace(protocol, outputs["score"], threshold)

    assert list(arrays) == protocol["retention"]["all_trial_frame_arrays"]
    assert all(np.array_equal(arrays[key], value) for key, value in trace.items())
    metrics = trial_metrics(protocol, small, arrays)
    assert metrics["wfe_nm"] == pytest.approx(
        2.0 * small["wavelength_m"] / (2 * np.pi) * 1e9
    )
    assert metrics == {
        "wfe_nm": metrics["wfe_nm"],
        "strehl": pytest.approx(0.6),
        "saturation_fraction": 0.0,
    }
    metadata = trial_metadata(
        protocol,
        small,
        case,
        {"path": "evidence/validation/trial.npz", "sha256": "a" * 64},
        arrays,
    )
    assert metadata["metric_interval"] == {
        "name": "post_burn_in",
        "start_frame": 128,
        "stop_frame_exclusive": 512,
    }
    assert metadata["metrics"] == metrics


def test_numerical_failure_is_distinct_from_structural_failure(protocol, base) -> None:
    small = _light_base(base)
    case = training_cases(protocol)[1]
    outputs = _outputs(protocol, small)
    outputs["score"][200] = np.nan
    arrays = physical_core_arrays(protocol, small, case, outputs)
    assert numerical_failure(arrays) == "Nonfinite physical trial output"
    with pytest.raises(FloatingPointError, match="Nonfinite"):
        complete_trial_arrays(protocol, small, validation_cases(protocol)[1], outputs, 1.0)

    missing = _outputs(protocol, small)
    missing.pop("command")
    with pytest.raises(ValueError, match="missing outputs"):
        physical_core_arrays(protocol, small, case, missing)


def test_trial_module_import_is_lazy() -> None:
    check = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import noether_optics.observability_trials; "
            "assert 'jax' not in sys.modules; "
            "assert 'noether_optics.simulation' not in sys.modules; "
            "assert 'noether_optics.atmosphere' not in sys.modules",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert check.returncode == 0, check.stderr


def test_executor_connects_exact_physics_only_inside_callback(
    protocol, base, tmp_path, monkeypatch
) -> None:
    ledger = _bound_ledger(tmp_path, protocol, base)
    case = training_cases(protocol)[1]
    calls = {}

    import jax
    from noether_optics import atmosphere, observability_trials, simulation

    monkeypatch.setattr(
        observability_trials.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(free=2**50),
    )
    monkeypatch.setattr(simulation, "calibration", lambda cfg: {"fixed": True})
    monkeypatch.setattr(
        atmosphere,
        "phase_sequence",
        lambda cfg, seed: calls.setdefault("phase", (cfg, seed)) and "synthetic-phase",
    )

    def loop(cfg, cal, phases, seed, controller, predictor, **kwargs):
        calls["loop"] = {
            "config": cfg,
            "calibration": cal,
            "phases": phases,
            "seed": seed,
            "controller": controller,
            "predictor": np.asarray(predictor),
            **kwargs,
        }
        return _outputs(protocol, base)

    monkeypatch.setattr(simulation, "loop", loop)
    monkeypatch.setattr(jax, "block_until_ready", lambda value: value)

    record = _TrialExecutor(ledger, base)(case)
    assert record["status"] == "completed"
    assert record["metadata"] is not None
    assert calls["phase"][1] == case["atmosphere_seed"]
    assert calls["loop"]["seed"] == case["atmosphere_seed"]
    assert calls["loop"]["controller"] == "forward"
    assert calls["loop"]["fallback"] is False
    assert calls["loop"]["differentiable_noise"] is False
    assert calls["loop"]["threshold"] == 1e30
    assert np.array_equal(calls["loop"]["predictor"], np.zeros((base["modes"], 2)))
    ledger._training_record(record, case)


def test_physics_rechecks_persisted_provenance(protocol, base, tmp_path) -> None:
    ledger = _bound_ledger(tmp_path, protocol, base)
    verify_observability_provenance(ledger, base)

    path = ledger.root / "evidence/provenance/runtime.json"
    path.write_bytes(_bytes({"machine": {"x64": True}, "fixture": False}))
    with pytest.raises(ValueError, match="source/runtime evidence"):
        verify_observability_provenance(ledger, base)


def test_public_physics_refuses_before_callback_when_full_storage_is_unavailable(
    base, tmp_path, monkeypatch
) -> None:
    ledger = ObservabilityLedger.create(tmp_path / "run", CONFIG, PROVENANCE)
    callback = Mock()
    monkeypatch.setattr(ledger, "train_chunk", callback)
    monkeypatch.setattr(
        "noether_optics.observability_trials.shutil.disk_usage",
        lambda path: SimpleNamespace(free=1),
    )

    with pytest.raises(RuntimeError, match="full storage admission"):
        train_trials(ledger, base, 1)
    callback.assert_not_called()


def test_run_creation_refuses_storage_before_source_or_runtime_capture(
    tmp_path, monkeypatch
) -> None:
    target = tmp_path / "unadmitted"
    capture = Mock()
    monkeypatch.setattr(
        "noether_optics.observability_resources.shutil.disk_usage",
        lambda path: SimpleNamespace(free=1),
    )
    monkeypatch.setattr(
        "noether_optics.observability_trials.capture_observability_inputs",
        capture,
    )

    with pytest.raises(RuntimeError, match="Insufficient free disk"):
        open_observability_run(target, CONFIG, create=True, repository=ROOT)
    capture.assert_not_called()
    assert not target.exists()


def test_run_opening_persists_and_rechecks_source_runtime_bindings(
    tmp_path, monkeypatch
) -> None:
    repository = tmp_path / "repository"
    config_dir = repository / "configs"
    config_dir.mkdir(parents=True)
    protocol_path = config_dir / CONFIG.name
    base_path = config_dir / "simulation.yaml"
    protocol_path.write_bytes(CONFIG.read_bytes())
    base_path.write_bytes((ROOT / "configs/simulation.yaml").read_bytes())
    source = {
        "schema_version": 1,
        "commit": "d" * 40,
        "files": {
            "configs/optical_gain_observability_v1.yaml": "protocol-object",
            "configs/simulation.yaml": "base-object",
        },
    }
    runtime = {"machine": {"x64": True}, "fixture": "runtime"}
    monkeypatch.setattr(
        "noether_optics.observability_trials._source",
        lambda root: deepcopy(source),
    )
    monkeypatch.setattr(
        "noether_optics.observability_trials._runtime",
        lambda: deepcopy(runtime),
    )
    monkeypatch.setattr(
        "noether_optics.observability_resources.shutil.disk_usage",
        lambda path: SimpleNamespace(free=2**50),
    )

    ledger, loaded_base, references = open_observability_run(
        tmp_path / "run",
        protocol_path,
        create=True,
        repository=repository,
    )
    assert {row["path"] for row in references} == {
        f"evidence/provenance/{name}.json"
        for name in ("source", "runtime", "protocol", "base")
    }
    assert ledger.provenance == {
        "source_commit": source["commit"],
        "source_sha256": _digest(_bytes(source)),
        "runtime_sha256": _digest(_bytes(runtime)),
    }
    verify_observability_provenance(ledger, loaded_base)
