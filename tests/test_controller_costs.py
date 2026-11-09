"""Deterministic accounting checks; synthetic clocks, no benchmark acquisition."""
import fcntl
from itertools import count
import json
import os
from pathlib import Path
import subprocess
import sys
from textwrap import dedent
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from noether_optics import cli
from noether_optics.controller_costs import (
    cost_identity, measure_call, measured_jit, record_costs,
)
from noether_optics.controller_selection import _bytes, _digest


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    monkeypatch.delenv("CI", raising=False)
    return SimpleNamespace(root=tmp_path, manifest={"study_id": "cost_test"},
                           provenance={"source_commit": "fixture", "runtime_sha256": "fixture"})


def rows(ledger):
    paths = list((ledger.root / "costs").glob("*.jsonl"))
    assert len(paths) == 1
    return [json.loads(line) for line in paths[0].read_text().splitlines()]


def clock():
    return count(0, 100).__next__


def test_disabled_measurement_is_noop_even_in_ci(ledger, monkeypatch):
    monkeypatch.setenv("CI", "true")
    with record_costs(ledger, "training"):
        assert measure_call("fit", {}, lambda: 7) == 7
    assert not (ledger.root / "costs").exists()


def test_costs_bound_to_manifest_and_preserve_operation_identity(ledger):
    with record_costs(ledger, "validation", enabled=True, clock=clock()):
        with cost_identity(stage="validation", case={"seed": 1}):
            assert measure_call("trial", {"candidate": "fixed"}, lambda: 9) == 9
    header, operation, end = rows(ledger)
    assert header["manifest_sha256"] == _digest(_bytes(ledger.manifest))
    assert header["provenance"] == ledger.provenance
    assert header["clock"] == "synthetic_test_clock"
    assert operation["identity"] == {"stage": "validation", "case": {"seed": 1}, "candidate": "fixed"}
    assert operation["elapsed_ns"] == 100 and operation["status"] == "completed"
    assert end["status"] == "completed" and end["elapsed_ns"] == 300
    assert [row["sequence"] for row in (header, operation, end)] == [0, 1, 2]


def test_interruption_is_retained_and_recorder_context_is_reset(ledger):
    def fail():
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt), record_costs(ledger, "training", enabled=True, clock=clock()):
        measure_call("fit", {}, fail)
    saved = rows(ledger)
    assert saved[-1]["status"] == saved[-2]["status"] == "interrupted"
    assert saved[-1]["error_type"] == saved[-2]["error_type"] == "KeyboardInterrupt"
    assert measure_call("after", {}, lambda: 3) == 3
    assert rows(ledger) == saved


def test_declared_numerical_failure_completes_cost_span_without_masking_failure(ledger):
    from noether_optics.controller_physics import NumericalTrainingFailure

    def fail():
        raise NumericalTrainingFailure("Nonfinite predictor coefficients", {"coefficients": None})

    with record_costs(ledger, "training", enabled=True, clock=clock()):
        with pytest.raises(NumericalTrainingFailure):
            measure_call("predictor_fit", {}, fail,
                         completed_errors=(NumericalTrainingFailure,))
    operation = rows(ledger)[1]
    assert operation["status"] == "completed"
    assert operation["error_type"] == "NumericalTrainingFailure"
    assert rows(ledger)[-1]["status"] == "completed"


def test_cost_session_lock_refuses_other_cooperating_writer(ledger):
    with (ledger.root / ".cost-acquisition.lock").open("a+b") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError), record_costs(ledger, "training", enabled=True, clock=clock()):
            pytest.fail("second cost writer entered")
    assert not list((ledger.root / "costs").glob("*.jsonl"))


@pytest.mark.parametrize("name", ["costs", ".cost-acquisition.lock"])
def test_cost_artifacts_do_not_follow_symlinks(ledger, tmp_path, name):
    destination = tmp_path / "untouched"
    if name == "costs":
        destination.mkdir()
    else:
        destination.write_bytes(b"retain")
    (ledger.root / name).symlink_to(destination)
    with pytest.raises((ValueError, OSError)), record_costs(ledger, "training", enabled=True, clock=clock()):
        pytest.fail("followed a cost symlink")
    if destination.is_file():
        assert destination.read_bytes() == b"retain"
    else:
        assert list(destination.iterdir()) == []


@pytest.mark.parametrize("command, limits", [
    ("study-train", ["--create", "--pilot-limit", "1", "--model-limit", "1"]),
    ("study-validate", ["--limit", "1"]),
    ("study-evaluate", ["--limit", "1"]),
])
def test_ci_rejects_costs_before_creating_or_loading_study(tmp_path, monkeypatch, command, limits):
    monkeypatch.setenv("CI", "true")
    out = tmp_path / "absent"
    monkeypatch.setattr("sys.argv", ["ao", command, "--config", "missing.yaml",
                                     "--out", str(out), "--measure-costs", *limits])
    with pytest.raises(RuntimeError, match="outside CI"):
        cli.main()
    assert not out.exists()


def test_measured_jit_reuses_shapes_not_values_and_preserves_random_outputs(ledger):
    kernel = jax.jit(lambda x, seed, unused: x * x + jax.random.normal(jax.random.PRNGKey(seed), x.shape))
    wrapped = measured_jit(kernel, "fixture")
    inputs = [(np.arange(3, dtype=np.float64), 1, None),
              (np.arange(3, dtype=np.float64) + 1, 2, None),
              (np.arange(4, dtype=np.float64), 1, None)]
    expected = [np.asarray(kernel(*args)) for args in inputs]
    with record_costs(ledger, "validation", enabled=True, clock=clock()):
        actual = [np.asarray(wrapped(*args)) for args in inputs]
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)
    operations = [row for row in rows(ledger) if row["kind"] == "operation"]
    assert [row["operation"] for row in operations] == [
        "input_placement", "lower_and_compile", "kernel_execution",
        "input_placement", "kernel_execution",
        "input_placement", "lower_and_compile", "kernel_execution"]
    assert all(row["elapsed_ns"] == 100 for row in operations)
    executions = [row for row in operations if row["operation"] == "kernel_execution"]
    assert executions[0]["identity"] == executions[1]["identity"]
    assert executions[1]["identity"] != executions[2]["identity"]


def test_measured_jit_does_not_reuse_executable_across_devices():
    script = dedent("""\
        from itertools import count
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from types import SimpleNamespace
        import jax
        import numpy as np
        from noether_optics.controller_costs import measured_jit, record_costs

        devices = jax.devices()
        assert len(devices) >= 2
        kernel = measured_jit(jax.jit(lambda x: x + 1), "device_test")
        with TemporaryDirectory() as root:
            ledger = SimpleNamespace(root=Path(root), manifest={"study_id": "device_test"},
                                     provenance={"source_commit": "fixture"})
            with record_costs(ledger, "validation", enabled=True, clock=count(0, 100).__next__):
                outputs = [np.asarray(kernel(jax.device_put(np.array([1.]), device))).tolist()
                           for device in devices[:2]]
            rows = [json.loads(line) for line in
                    next((Path(root) / "costs").glob("*.jsonl")).read_text().splitlines()]
            compiles = [row["identity"]["input_signature_sha256"] for row in rows
                        if row.get("operation") == "lower_and_compile"]
        print(json.dumps({"compiles": compiles, "outputs": outputs}))
    """)
    env = os.environ.copy()
    env.pop("CI", None)  # synthetic clock, isolated from the CI process
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    env["JAX_PLATFORMS"] = "cpu"
    result = subprocess.run([sys.executable, "-B", "-c", script], check=True,
                            capture_output=True, text=True, env=env)
    observed = json.loads(result.stdout)
    assert len(observed["compiles"]) == len(set(observed["compiles"])) == 2
    assert observed["outputs"] == [[2.0], [2.0]]


def test_measured_gradient_and_nonfinite_output_are_not_relabelled(ledger):
    def loss(params, inputs):
        return jnp.sum(params[0] * inputs**2)
    kernel = jax.jit(jax.value_and_grad(loss))
    wrapped = measured_jit(kernel, "gradient")
    args = ((jnp.array([2., 3.]),), np.array([4., 5.]))
    expected = kernel(*args)
    with record_costs(ledger, "training", enabled=True, clock=clock()):
        actual = wrapped(*args)
        failed = wrapped(args[0], np.array([np.nan, 5.]))
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_array_equal(a, b)
    assert not np.isfinite(np.asarray(failed[0]))
    assert rows(ledger)[-1]["status"] == "completed"  # measurement completed, science failed


def test_execution_timer_waits_for_result_readiness(ledger, monkeypatch):
    calls = []
    class FakeKernel:
        def lower(self, *args):
            calls.append("lower")
            return self
        def compile(self):
            calls.append("compile")
            return self
        def __call__(self, *args):
            calls.append("dispatch")
            return jnp.array([7.])
    original = jax.block_until_ready
    def sync(value):
        calls.append("synchronize")
        return original(value)
    def tick():
        calls.append("clock")
        return len(calls)
    monkeypatch.setattr(jax, "block_until_ready", sync)
    with record_costs(ledger, "validation", enabled=True, clock=tick):
        measured_jit(FakeKernel(), "fake")(np.ones(2))
    i = calls.index("dispatch")
    assert calls[i - 1:i + 3] == ["clock", "dispatch", "synchronize", "clock"]


def test_optical_controller_outputs_are_exact_with_cost_measurement(ledger):
    from noether_optics.artifacts import load_config
    from noether_optics.controller_physics import controller_kernel
    from noether_optics.simulation import calibration

    cfg = load_config(Path(__file__).parents[1] / "configs/controller_runner_smoke_base.yaml")
    cfg.update(frames=8, burn_in=2)
    cal = calibration(cfg)
    phases = np.linspace(-0.1, 0.1, 8 * cfg["n"]**2).reshape(8, cfg["n"], cfg["n"])
    kernel = controller_kernel(cfg, cal, "integrator")
    expected = kernel(phases, 61001)
    with record_costs(ledger, "training", enabled=True, clock=clock()):
        actual = kernel(phases, 61001)
    assert set(actual) == set(expected)
    for name in actual:
        np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)
