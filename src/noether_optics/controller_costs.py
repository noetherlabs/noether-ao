"""Opt-in, synchronized study costs; separate from scientific stage decisions.

One exclusive JSONL file records each invocation, including interrupted attempts.
Kernel costs distinguish input placement, lowering/compilation and execution.
Enclosing fit costs are inclusive: never add them to their nested kernel costs.
No replayed gradient or already-terminal trial creates a new kernel measurement.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import time
from uuid import uuid4

from .controller_selection import _bytes, _digest

_ACTIVE = ContextVar("controller_cost_recorder", default=None)
_IDENTITY = ContextVar("controller_cost_identity", default={})


def require_local_measurement() -> None:
    if os.environ.get("CI"):
        raise RuntimeError("Acquire study costs locally, outside CI and without concurrent compute")


@contextmanager
def cost_identity(**identity):
    """Attach declared case/candidate identities without changing numerical inputs."""
    token = _IDENTITY.set({**_IDENTITY.get(), **identity})
    try:
        yield
    finally:
        _IDENTITY.reset(token)


class _Recorder:
    def __init__(self, stream, clock):
        self.stream, self.clock = stream, clock
        self.sequence = 0

    def write(self, row):
        payload = {"sequence": self.sequence, **row}
        self.stream.write((json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                      allow_nan=False) + "\n").encode())
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.sequence += 1

    def call(self, operation, metadata, function, completed_errors=()):
        start = self.clock()
        status, error_type = "completed", None
        try:
            return function()
        except completed_errors as error:
            error_type = type(error).__name__
            raise
        except BaseException as error:
            status, error_type = "interrupted", type(error).__name__
            raise
        finally:
            stop = self.clock()
            if stop < start:
                raise RuntimeError("Monotonic cost clock moved backwards")
            self.write({"kind": "operation", "operation": operation,
                        "identity": {**_IDENTITY.get(), **metadata},
                        "elapsed_ns": stop - start, "status": status,
                        "error_type": error_type})


def measure_call(operation, metadata, function, *, completed_errors=()):
    recorder = _ACTIVE.get()
    return function() if recorder is None else recorder.call(
        operation, metadata, function, completed_errors)


@contextmanager
def record_costs(ledger, stage, *, enabled=False, clock=None):
    """Retain one attempt, bound to the admitted manifest, source and runtime.

The lock serializes cooperating measured writers for this run. The operator must
exclude other compute on the host. A missing attempt-end record means incomplete
measurement; it must never be interpreted as zero cost. The caller opens/audits
the scientific ledger before entering, so admission is outside these costs.
An injected clock is for deterministic tests only, never a physical measurement.
"""
    if not enabled:
        yield
        return
    require_local_measurement()
    if stage not in {"training", "validation", "evaluation"}:
        raise ValueError("Unknown cost acquisition stage")
    if _ACTIVE.get() is not None:
        raise RuntimeError("Nested cost acquisition is not supported")
    root = Path(ledger.root)
    directory = root / "costs"
    if directory.is_symlink():
        raise ValueError("Cost directory must not be a symlink")
    directory.mkdir(exist_ok=True)
    root_fd = os.open(root, os.O_RDONLY)
    try:
        os.fsync(root_fd)
    finally:
        os.close(root_fd)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    lock_path = root / ".cost-acquisition.lock"
    with os.fdopen(os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600), "a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = directory / (uuid4().hex + ".jsonl")
        with os.fdopen(os.open(path, flags, 0o600), "wb") as stream:
            directory_fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            recorder = _Recorder(stream, time.perf_counter_ns if clock is None else clock)
            recorder.write({"kind": "attempt_start", "schema_version": 1, "stage": stage,
                            "started_at": datetime.now(timezone.utc).isoformat(),
                            "manifest_sha256": _digest(_bytes(ledger.manifest)),
                            "provenance": ledger.provenance,
                            "clock": "perf_counter_ns" if clock is None else "synthetic_test_clock",
                            "units": "ns", "enclosing_costs": "inclusive",
                            "scope": "Study resource costs, not frame latency or deadlines"})
            token = _ACTIVE.set(recorder)
            status, error_type = "completed", None
            start = recorder.clock()
            try:
                yield
            except BaseException as error:
                status, error_type = "interrupted", type(error).__name__
                raise
            finally:
                _ACTIVE.reset(token)
                recorder.write({"kind": "attempt_end", "status": status,
                                "error_type": error_type,
                                "elapsed_ns": recorder.clock() - start})


def measured_jit(kernel, label, metadata=None):
    """Wrap a JIT callable; explicitly compile once per abstract input signature.

Input placement is synchronized before lowering or execution. Execution ends only
after every result leaf is ready. Host output conversion stays in the caller and
is excluded from kernel execution cost. Cache hits still retain executions; a
fresh process retains its own lowering/compile overhead, even with a JAX cache.
"""
    executables = {}
    metadata = {} if metadata is None else metadata

    def execute(*args):
        recorder = _ACTIVE.get()
        if recorder is None:
            return kernel(*args)
        import jax

        identity = {"kernel": label, **metadata}
        device_args = recorder.call("input_placement", identity,
                                    lambda: jax.block_until_ready(jax.device_put(args)))
        leaves, structure = jax.tree.flatten(device_args)
        devices = tuple((value.device.platform, value.device.id) for value in leaves)
        signature = (structure, tuple((tuple(value.shape), str(value.dtype),
                                       bool(getattr(value, "weak_type", False))) for value in leaves),
                     devices)
        abstract = {"tree": str(structure), "leaves": signature[1], "devices": devices}
        identity = {**identity, "input_signature_sha256": _digest(_bytes(abstract))}
        if signature not in executables:
            executables[signature] = recorder.call(
                "lower_and_compile", {**identity, "abstract_inputs": abstract},
                lambda: kernel.lower(*device_args).compile())
        return recorder.call("kernel_execution", identity,
                             lambda: jax.block_until_ready(executables[signature](*device_args)))

    return execute
