"""Command-boundary checks without running scientific backends."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from noether_optics import artifacts, cli, config
from noether_optics import controller_evidence as evidence
from noether_optics import controller_training_stage as training_stage
from noether_optics import controller_trials as trials
from noether_optics.experiments import runners


@pytest.mark.parametrize("create", [False, True])
def test_study_train_dispatches_bound_chunk_and_prints_json(monkeypatch, capsys, create):
    ledger, base, refs = SimpleNamespace(), {"fixture": "base"}, [{"fixture": "evidence"}]
    summary = {"complete": False, "new_pilot_records": 2, "new_model_records": 3}
    opened = Mock(return_value=(ledger, base, refs))
    trained = Mock(return_value=summary)
    monkeypatch.setattr(evidence, "open_study_run", opened)
    monkeypatch.setattr(training_stage, "run_training_chunk", trained)
    argv = ["noether-optics", "study-train", "--config", "study.yaml", "--out", "run",
            "--pilot-limit", "2", "--model-limit", "3"]
    if create:
        argv.append("--create")
    monkeypatch.setattr("sys.argv", argv)

    cli.main()

    opened.assert_called_once_with(Path("run"), Path("study.yaml"), create=create)
    trained.assert_called_once_with(ledger, base, refs, pilot_limit=2, model_limit=3)
    assert json.loads(capsys.readouterr().out) == summary


@pytest.mark.parametrize("extra", [
    [], ["--pilot-limit", "0", "--model-limit", "1"],
    ["--pilot-limit", "1", "--model-limit", "0"], ["--pilot-limit", "1"]])
def test_study_train_rejects_missing_or_invalid_limits_before_open(monkeypatch, extra):
    monkeypatch.setattr(evidence, "open_study_run",
                        lambda *args, **kwargs: pytest.fail("run opened after invalid CLI input"))
    monkeypatch.setattr(
        "sys.argv", ["noether-optics", "study-train", "--config", "study.yaml", "--out", "run", *extra])
    with pytest.raises(SystemExit):
        cli.main()


def test_study_validate_dispatches_bound_chunk_and_prints_json(monkeypatch, capsys):
    ledger, base = SimpleNamespace(), {"fixture": "base"}
    summary = {"complete": False, "new_records": 4,
               "total_records": 4, "planned_records": 600}
    opened = Mock(return_value=(ledger, base, [{"fixture": "evidence"}]))
    validated = Mock(return_value=summary)
    monkeypatch.setattr(evidence, "open_study_run", opened)
    monkeypatch.setattr(trials, "validate_trials", validated)
    monkeypatch.setattr("sys.argv", [
        "noether-optics", "study-validate", "--config", "study.yaml",
        "--out", "run", "--limit", "4"])

    cli.main()

    opened.assert_called_once_with(Path("run"), Path("study.yaml"))
    validated.assert_called_once_with(ledger, base, 4)
    assert json.loads(capsys.readouterr().out) == summary


@pytest.mark.parametrize("extra", [
    [], ["--limit", "0"], ["--limit", "-1"], ["--limit", "1", "--create"]])
def test_study_validate_rejects_missing_invalid_or_creation_options_before_open(
        monkeypatch, extra):
    monkeypatch.setattr(evidence, "open_study_run",
                        lambda *args, **kwargs: pytest.fail("run opened after invalid CLI input"))
    monkeypatch.setattr("sys.argv", [
        "noether-optics", "study-validate", "--config", "study.yaml", "--out", "run", *extra])
    with pytest.raises(SystemExit):
        cli.main()


def test_existing_experiment_rejects_study_flags_and_preserves_dispatch(monkeypatch):
    monkeypatch.setattr("sys.argv", [
        "noether-optics", "validate", "--config", "missing.yaml", "--out", "run", "--create"])
    with pytest.raises(SystemExit):
        cli.main()

    loaded = Mock(return_value={"fixture": "config"})
    checked, executed = Mock(), Mock()
    monkeypatch.setattr(artifacts, "load_config", loaded)
    monkeypatch.setattr(config, "validate_simulation", checked)
    monkeypatch.setattr(runners, "validate", executed)
    monkeypatch.setattr("sys.argv", [
        "noether-optics", "validate", "--config", "simulation.yaml", "--out", "run"])
    cli.main()
    loaded.assert_called_once_with("simulation.yaml")
    checked.assert_called_once_with({"fixture": "config"})
    executed.assert_called_once_with({"fixture": "config"}, "run")
