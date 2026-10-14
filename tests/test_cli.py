"""Command-boundary checks without running scientific backends."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from noether_optics import artifacts, cli, config
from noether_optics import controller_evidence as evidence
from noether_optics import controller_resources as resources
from noether_optics import controller_training_stage as training_stage
from noether_optics import controller_trials as trials
from noether_optics import observability_resources, observability_trials
from noether_optics.experiments import runners


def test_study_resources_reports_admitted_target_without_opening_run(monkeypatch, capsys):
    study, base = {"study_id": "fixture"}, {"fixture": "base"}
    report = {"kind": "controller_study_resource_preflight", "admitted": True}
    loaded = Mock(return_value=(study, base))
    checked = Mock(return_value=report)
    monkeypatch.setattr(resources, "load_resource_inputs", loaded)
    monkeypatch.setattr(resources, "resource_preflight", checked)
    monkeypatch.setattr(evidence, "open_study_run",
                        lambda *args, **kwargs: pytest.fail("resource preflight opened a run"))
    monkeypatch.setattr("sys.argv", [
        "noether-optics", "study-resources", "--config", "study.yaml",
        "--target", "planned-run"])

    cli.main()

    loaded.assert_called_once_with(Path("study.yaml"))
    checked.assert_called_once_with(study, base, Path("planned-run"))
    assert json.loads(capsys.readouterr().out) == report


def test_study_resources_returns_two_after_printing_insufficient_report(monkeypatch, capsys):
    report = {"kind": "controller_study_resource_preflight", "admitted": False}
    monkeypatch.setattr(resources, "load_resource_inputs", Mock(return_value=({}, {})))
    monkeypatch.setattr(resources, "resource_preflight", Mock(return_value=report))
    monkeypatch.setattr("sys.argv", [
        "noether-optics", "study-resources", "--config", "study.yaml",
        "--target", "planned-run"])

    with pytest.raises(SystemExit) as error:
        cli.main()

    assert error.value.code == 2
    assert json.loads(capsys.readouterr().out) == report


@pytest.mark.parametrize("admitted", [True, False])
def test_observability_resources_reports_without_opening_a_run(
    monkeypatch, capsys, admitted
):
    protocol, base = {"study_id": "fixture"}, {"fixture": "base"}
    report = {"kind": "optical_gain_observability_resource_preflight",
              "admitted": admitted}
    loaded = Mock(return_value=(protocol, base))
    checked = Mock(return_value=report)
    monkeypatch.setattr(observability_resources, "load_resource_inputs", loaded)
    monkeypatch.setattr(observability_resources, "resource_preflight", checked)
    monkeypatch.setattr(evidence, "open_study_run",
                        lambda *args, **kwargs: pytest.fail("resource preflight opened a run"))
    monkeypatch.setattr("sys.argv", [
        "noether-optics", "observability-resources", "--config", "protocol.yaml",
        "--target", "planned-run"])

    if admitted:
        cli.main()
    else:
        with pytest.raises(SystemExit) as error:
            cli.main()
        assert error.value.code == 2

    loaded.assert_called_once_with(Path("protocol.yaml"))
    checked.assert_called_once_with(protocol, base, Path("planned-run"))
    assert json.loads(capsys.readouterr().out) == report


@pytest.mark.parametrize("create", [False, True])
def test_observability_train_dispatches_admitted_physical_chunk(
    monkeypatch, capsys, create
):
    ledger, base = SimpleNamespace(), {"fixture": "base"}
    summary = {"complete": False, "new_records": 1, "total_records": 1}
    opened = Mock(return_value=(ledger, base, [{"fixture": "evidence"}]))
    trained = Mock(return_value=summary)
    loaded = Mock(return_value=({"study_id": "fixture"}, base))
    preflight = {"admitted": True, "budget": {"required_free_bytes": 123456}}
    checked = Mock(return_value=preflight)
    monkeypatch.setattr(observability_trials, "open_observability_run", opened)
    monkeypatch.setattr(observability_trials, "train_trials", trained)
    monkeypatch.setattr(observability_resources, "load_resource_inputs", loaded)
    monkeypatch.setattr(observability_resources, "resource_preflight", checked)
    argv = [
        "noether-optics", "observability-train", "--config", "protocol.yaml",
        "--out", "run", "--limit", "1",
    ]
    if create:
        argv.append("--create")
    monkeypatch.setattr("sys.argv", argv)

    cli.main()

    if create:
        loaded.assert_called_once_with(Path("protocol.yaml"))
        checked.assert_called_once_with({"study_id": "fixture"}, base, Path("run"))
    else:
        loaded.assert_not_called()
        checked.assert_not_called()
    opened.assert_called_once_with(
        Path("run"), Path("protocol.yaml"), create=create
    )
    trained.assert_called_once_with(ledger, base, 1)
    assert json.loads(capsys.readouterr().out) == summary


def test_observability_train_creation_refuses_insufficient_capacity(
    monkeypatch, capsys
):
    report = {"kind": "optical_gain_observability_resource_preflight", "admitted": False}
    monkeypatch.setattr(
        observability_resources, "load_resource_inputs", Mock(return_value=({}, {}))
    )
    monkeypatch.setattr(
        observability_resources, "resource_preflight", Mock(return_value=report)
    )
    monkeypatch.setattr(
        observability_trials,
        "open_observability_run",
        lambda *args, **kwargs: pytest.fail("run opened after disk refusal"),
    )
    monkeypatch.setattr("sys.argv", [
        "noether-optics", "observability-train", "--config", "protocol.yaml",
        "--out", "run", "--limit", "1", "--create",
    ])

    with pytest.raises(SystemExit) as error:
        cli.main()

    assert error.value.code == 2
    assert json.loads(capsys.readouterr().out) == report


@pytest.mark.parametrize("command", [
    "observability-freeze-training", "observability-freeze-validation"
])
def test_observability_freeze_dispatches_exact_bound_stage(
    monkeypatch, capsys, command
):
    result = {"study_id": "fixture", "status": "frozen"}
    artifact = {"path": "frozen.json", "sha256": "a" * 64}
    method = Mock(return_value=(result, artifact))
    ledger = SimpleNamespace()
    setattr(
        ledger,
        "freeze_training" if command.endswith("training") else "freeze_validation",
        method,
    )
    opened = Mock(return_value=(ledger, {"fixture": "base"}, []))
    monkeypatch.setattr(observability_trials, "open_observability_run", opened)
    monkeypatch.setattr("sys.argv", [
        "noether-optics", command, "--config", "protocol.yaml", "--out", "run",
    ])

    cli.main()

    opened.assert_called_once_with(Path("run"), Path("protocol.yaml"))
    method.assert_called_once_with()
    assert json.loads(capsys.readouterr().out) == {
        "artifact": artifact,
        "result": result,
    }


def test_observability_validate_dispatches_bound_physical_chunk(monkeypatch, capsys):
    ledger, base = SimpleNamespace(), {"fixture": "base"}
    summary = {"complete": False, "new_records": 2, "total_records": 2}
    opened = Mock(return_value=(ledger, base, [{"fixture": "evidence"}]))
    validated = Mock(return_value=summary)
    monkeypatch.setattr(observability_trials, "open_observability_run", opened)
    monkeypatch.setattr(observability_trials, "validate_trials", validated)
    monkeypatch.setattr("sys.argv", [
        "noether-optics", "observability-validate", "--config", "protocol.yaml",
        "--out", "run", "--limit", "2",
    ])

    cli.main()

    opened.assert_called_once_with(Path("run"), Path("protocol.yaml"))
    validated.assert_called_once_with(ledger, base, 2)
    assert json.loads(capsys.readouterr().out) == summary


@pytest.mark.parametrize("command", ["observability-train", "observability-validate"])
@pytest.mark.parametrize("limit", ["0", "-1"])
def test_observability_chunks_reject_invalid_limit_before_open(
    monkeypatch, command, limit
):
    monkeypatch.setattr(
        observability_trials,
        "open_observability_run",
        lambda *args, **kwargs: pytest.fail("run opened after invalid CLI input"),
    )
    monkeypatch.setattr("sys.argv", [
        "noether-optics", command, "--config", "protocol.yaml",
        "--out", "run", "--limit", limit,
    ])
    with pytest.raises(SystemExit):
        cli.main()


@pytest.mark.parametrize("extra", [[], ["--target", "run", "--limit", "1"],
                                   ["--config", "study.yaml"]])
def test_study_resources_rejects_missing_or_cross_stage_options(monkeypatch, extra):
    monkeypatch.setattr(resources, "load_resource_inputs",
                        lambda *args, **kwargs: pytest.fail("inputs loaded after invalid CLI input"))
    monkeypatch.setattr("sys.argv", ["noether-optics", "study-resources", *extra])
    with pytest.raises(SystemExit):
        cli.main()


@pytest.mark.parametrize("create", [False, True])
def test_study_train_dispatches_bound_chunk_and_prints_json(monkeypatch, capsys, create):
    ledger, base, refs = SimpleNamespace(), {"fixture": "base"}, [{"fixture": "evidence"}]
    summary = {"complete": False, "new_pilot_records": 2, "new_model_records": 3}
    opened = Mock(return_value=(ledger, base, refs))
    trained = Mock(return_value=summary)
    loaded = Mock(return_value=({"study_id": "fixture"}, base))
    preflight = {"admitted": True, "budget": {"required_free_bytes": 123456}}
    checked = Mock(return_value=preflight)
    monkeypatch.setattr(evidence, "open_study_run", opened)
    monkeypatch.setattr(resources, "load_resource_inputs", loaded)
    monkeypatch.setattr(resources, "resource_preflight", checked)
    monkeypatch.setattr(training_stage, "run_training_chunk", trained)
    argv = ["noether-optics", "study-train", "--config", "study.yaml", "--out", "run",
            "--pilot-limit", "2", "--model-limit", "3"]
    if create:
        argv.append("--create")
    monkeypatch.setattr("sys.argv", argv)

    cli.main()

    if create:
        loaded.assert_called_once_with(Path("study.yaml"))
        checked.assert_called_once_with({"study_id": "fixture"}, base, Path("run"))
        opened.assert_called_once_with(
            Path("run"), Path("study.yaml"), create=True, min_free_bytes=123456)
    else:
        loaded.assert_not_called()
        checked.assert_not_called()
        opened.assert_called_once_with(Path("run"), Path("study.yaml"), create=False)
    trained.assert_called_once_with(ledger, base, refs, pilot_limit=2, model_limit=3)
    assert json.loads(capsys.readouterr().out) == summary


def test_study_train_creation_refuses_insufficient_full_run_capacity(
        monkeypatch, capsys):
    report = {"kind": "controller_study_resource_preflight", "admitted": False}
    monkeypatch.setattr(resources, "load_resource_inputs", Mock(return_value=({}, {})))
    monkeypatch.setattr(resources, "resource_preflight", Mock(return_value=report))
    monkeypatch.setattr(evidence, "open_study_run",
                        lambda *args, **kwargs: pytest.fail("run opened after disk refusal"))
    monkeypatch.setattr(training_stage, "run_training_chunk",
                        lambda *args, **kwargs: pytest.fail("training ran after disk refusal"))
    monkeypatch.setattr("sys.argv", [
        "noether-optics", "study-train", "--config", "study.yaml", "--out", "run",
        "--pilot-limit", "1", "--model-limit", "1", "--create"])

    with pytest.raises(SystemExit) as error:
        cli.main()

    assert error.value.code == 2
    assert json.loads(capsys.readouterr().out) == report


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


def test_study_select_freezes_bound_selection_and_prints_json(monkeypatch, capsys):
    summary = {"manifest_sha256": "a" * 64, "result": {"selected": {"learned": None}}}
    frozen = Mock(return_value=summary)
    ledger = SimpleNamespace(freeze_selection=frozen)
    opened = Mock(return_value=(ledger, {"fixture": "base"}, [{"fixture": "evidence"}]))
    monkeypatch.setattr(evidence, "open_study_run", opened)
    monkeypatch.setattr("sys.argv", [
        "noether-optics", "study-select", "--config", "study.yaml", "--out", "run"])

    cli.main()

    opened.assert_called_once_with(Path("run"), Path("study.yaml"))
    frozen.assert_called_once_with()
    assert json.loads(capsys.readouterr().out) == summary


@pytest.mark.parametrize("extra", [
    ["--create"], ["--limit", "1"], ["--model-limit", "1"]])
def test_study_select_rejects_other_stage_options_before_open(monkeypatch, extra):
    monkeypatch.setattr(evidence, "open_study_run",
                        lambda *args, **kwargs: pytest.fail("run opened after invalid CLI input"))
    monkeypatch.setattr("sys.argv", [
        "noether-optics", "study-select", "--config", "study.yaml", "--out", "run", *extra])
    with pytest.raises(SystemExit):
        cli.main()


def test_study_evaluate_dispatches_bound_chunk_and_prints_json(monkeypatch, capsys):
    ledger, base = SimpleNamespace(), {"fixture": "base"}
    summary = {"complete": False, "new_records": 3,
               "total_records": 3, "planned_records": 480}
    opened = Mock(return_value=(ledger, base, [{"fixture": "evidence"}]))
    evaluated = Mock(return_value=summary)
    monkeypatch.setattr(evidence, "open_study_run", opened)
    monkeypatch.setattr(trials, "evaluate_trials", evaluated)
    monkeypatch.setattr("sys.argv", [
        "noether-optics", "study-evaluate", "--config", "study.yaml",
        "--out", "run", "--limit", "3"])

    cli.main()

    opened.assert_called_once_with(Path("run"), Path("study.yaml"))
    evaluated.assert_called_once_with(ledger, base, 3)
    assert json.loads(capsys.readouterr().out) == summary


@pytest.mark.parametrize("extra", [
    [], ["--limit", "0"], ["--limit", "-1"], ["--limit", "1", "--create"]])
def test_study_evaluate_rejects_missing_invalid_or_creation_options_before_open(
        monkeypatch, extra):
    monkeypatch.setattr(evidence, "open_study_run",
                        lambda *args, **kwargs: pytest.fail("run opened after invalid CLI input"))
    monkeypatch.setattr("sys.argv", [
        "noether-optics", "study-evaluate", "--config", "study.yaml", "--out", "run", *extra])
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
