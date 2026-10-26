"""CI may summarize saved results, but must never start timing workers."""
import importlib.util
from pathlib import Path
import sys

import pytest

from noether_optics.latency import run


@pytest.fixture
def dragrace_runner():
    source = Path(__file__).resolve().parents[1] / "scripts" / "run_dragrace.py"
    spec = importlib.util.spec_from_file_location("run_dragrace", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_latency_ci_rejection_precedes_artifact_creation(monkeypatch, tmp_path):
    monkeypatch.setenv("CI", "true")
    out = tmp_path / "never-created"
    with pytest.raises(RuntimeError, match="forbidden in CI"):
        run({"repeats": 10000}, str(out))
    assert not out.exists()


def test_dragrace_ci_rejection_precedes_files_and_workers(
    monkeypatch, tmp_path, dragrace_runner
):
    monkeypatch.setenv("CI", "true")
    out = tmp_path / "never-created"
    monkeypatch.setattr(sys, "argv", ["run_dragrace", "--out", str(out),
                                     "--repo", str(tmp_path / "absent")])
    monkeypatch.setattr(dragrace_runner.subprocess, "run",
                        lambda *a, **k: pytest.fail("started timing worker"))
    with pytest.raises(RuntimeError, match="No benchmark timings in CI"):
        dragrace_runner.main()
    assert not out.exists()


def test_dragrace_saved_results_can_render_in_ci(
    monkeypatch, tmp_path, dragrace_runner
):
    monkeypatch.setenv("CI", "true")
    repo, out = tmp_path / "upstream", tmp_path / "saved"
    rendered = []
    monkeypatch.setattr(dragrace_runner, "render_results",
                        lambda path, source: rendered.append((path, source)))
    monkeypatch.setattr(dragrace_runner.subprocess, "run",
                        lambda *a, **k: pytest.fail("started timing worker"))
    monkeypatch.setattr(sys, "argv", ["run_dragrace", "--out", str(out),
                                     "--repo", str(repo), "--render-only"])
    dragrace_runner.main()
    assert rendered == [(out.resolve(), repo.resolve())]
