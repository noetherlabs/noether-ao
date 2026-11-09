"""Synthetic reporting checks; no reserved streams or simulator execution."""
from copy import deepcopy
from itertools import product
import math
import json
from pathlib import Path

import numpy as np
import pytest

from noether_optics.artifacts import load_config
from noether_optics.controller_evaluation import (
    _comparisons, _condition_rows, _reconstruct_trial, _metrics_match, paired_interval,
    build_evaluation_report, cost_summary,
)
from noether_optics.controller_evidence import save_arrays, save_metadata
from noether_optics.controller_physics import FULL_OUTPUT_KEYS, episode_config, paired_metrics
from noether_optics.controller_protocol import PARAMETERS, load_study, study_plan
from noether_optics.controller_selection import StudyLedger, _bytes, _case_id, _digest, validation_cases

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    cfg = load_study(ROOT / "configs/controller_study_v1.yaml")
    cfg["study_id"] = "synthetic_controller_report_test_only"
    for role, base in (("training", 71001), ("validation", 72001), ("evaluation", 73001)):
        cfg["seeds"][role] = list(range(base, base + len(cfg["seeds"][role])))
    cfg["seeds"]["initialization"] = 74001
    return cfg


def test_two_pairs_match_closed_form_cauchy_quantile():
    interval = paired_interval([2.0, 4.0], alpha=.05)
    radius = 1 / math.tan(math.pi * .05 / 2)
    assert interval["mean_nm"] == 3.0
    assert interval["low_nm"] == pytest.approx(3 - radius)
    assert interval["high_nm"] == pytest.approx(3 + radius)


@pytest.mark.parametrize("values", [[1.0], [1.0, float("nan")], [1.0, float("inf")], [True, 2.0]])
def test_no_dropping_invalid_pairs(values):
    with pytest.raises(ValueError):
        paired_interval(values, alpha=.05)


def test_constant_differences_are_point_interval_not_deleted():
    result = paired_interval([-2.0] * 10, alpha=.05 / 120)
    assert result["low_nm"] == result["high_nm"] == -2.0
    assert result["n"] == 10


@pytest.mark.parametrize("gate", ["saturation", "usefulness"])
def test_float_tolerance_cannot_hide_changed_eligibility(gate):
    raw = dict(all_outputs_finite=True, wfe_nm=20.0, open_wfe_nm=20.0,
               saturation_fraction=.01, strehl=.8)
    saved = deepcopy(raw)
    if gate == "saturation":
        saved["saturation_fraction"] += 1e-13
    else:
        saved["wfe_nm"] -= 1e-13
    with pytest.raises(ValueError, match="eligibility gate"):
        _metrics_match(saved, raw, .01)


def synthetic_rows(config):
    candidates = study_plan(config)["candidates"]
    selected = {name: next(c for c in candidates if c["controller"] == name) for name in PARAMETERS}
    rows = []
    for index, name in enumerate(PARAMETERS):
        for photons, wind, fault, seed in product(
                config["conditions"]["photons"], config["conditions"]["wind_scales"],
                config["conditions"]["evaluation_faults"], config["seeds"]["evaluation"]):
            metrics = dict(all_outputs_finite=True, wfe_nm=20.0 + index,
                           open_wfe_nm=100.0, saturation_fraction=0.0, strehl=.8)
            case = dict(stage="evaluation", candidate=selected[name], photons=photons, wind_scale=wind,
                        fault=fault, seed=seed, detector_seed=seed + config["seeds"]["noise_offset"])
            rows.append(dict(case=case, status="completed", error=None,
                             post_burn=deepcopy(metrics), fault_window=deepcopy(metrics)))
    return selected, rows


def test_all_pairs_and_withheld_controllers_keep_original_family(config):
    selected, rows = synthetic_rows(config)
    selected["learned"] = None
    rows = [r for r in rows if r["case"]["candidate"]["controller"] != "learned"]
    result = _comparisons(config, selected, list(reversed(rows)))
    assert len(result) == 120
    assert sum(r["post_burn"]["status"] == "controller_withheld" for r in result) == 48
    available = [r for r in result if r["post_burn"]["status"] == "complete"]
    assert all(r["post_burn"]["interval"]["confidence"] == 1 - .05 / 120 for r in available)
    assert all(r["fault_window"]["interval"]["confidence"] == .95 for r in available)
    assert all(r["fault_window"]["winner"] is None for r in available)


@pytest.mark.parametrize("damage", ["saturation", "not_useful", "nonfinite", "failed", "missing"])
def test_no_win_when_one_seed_fails_winner_gate(config, damage):
    selected, rows = synthetic_rows(config)
    row = rows[0]
    if damage == "saturation":
        row["post_burn"]["saturation_fraction"] = .011
    elif damage == "not_useful":
        # Keep the apparent left advantage while making this seed unhelpful.
        for other in rows:
            if all(other["case"][key] == row["case"][key] for key in ("photons", "wind_scale", "fault", "seed")):
                other["post_burn"]["open_wfe_nm"] = 10.0
    elif damage == "nonfinite":
        row["post_burn"].update(all_outputs_finite=False, wfe_nm=None)
    elif damage == "failed":
        row.update(status="failed", post_burn=None, fault_window=None)
    else:
        rows.pop(0)
    result = _comparisons(config, selected, rows)[0]
    assert result["post_burn"]["winner"] is None
    if damage in ("failed", "nonfinite", "missing"):
        assert result["post_burn"]["interval"] is None


def test_failure_is_not_hidden_by_survivor_mean(config):
    selected, rows = synthetic_rows(config)
    rows[0].update(status="failed", post_burn=None, fault_window=None)
    first = _condition_rows(config, selected, rows)[0]
    assert first["recorded_trials"] == 10
    assert first["failed_trials"] == 1
    assert first["post_burn"]["finite_trials"] == 9
    assert first["post_burn"]["mean"] is None


def test_pairing_rejects_different_open_loop_metrics(config):
    selected, rows = synthetic_rows(config)
    rows[0]["post_burn"]["open_wfe_nm"] += 1
    with pytest.raises(ValueError, match="paired open-loop"):
        _comparisons(config, selected, rows)


@pytest.fixture
def raw_trial(tmp_path, config):
    config["loop"].update(frames=4, burn_in=1, fault_onset=2, fault_end=4)
    provenance = dict(source_commit="a" * 40, source_sha256="b" * 64, runtime_sha256="c" * 64)
    ledger = StudyLedger.create(tmp_path / "study", config, provenance)
    base = load_config(ROOT / config["base_config"])
    candidate = study_plan(config)["candidates"][0]
    case = dict(stage="evaluation", candidate=candidate, photons=config["conditions"]["photons"][0],
                wind_scale=config["conditions"]["wind_scales"][0], fault="none",
                seed=config["seeds"]["evaluation"][0],
                detector_seed=config["seeds"]["evaluation"][0] + config["seeds"]["noise_offset"])
    cfg = episode_config(base, config, "evaluation", case["photons"], case["wind_scale"], candidate)
    outputs = {key: np.zeros(4) for key in FULL_OUTPUT_KEYS}
    for key in ("alarm", "fallback", "saturated", "unavailable"):
        outputs[key] = np.zeros(4, dtype=bool)
    outputs.update(variance=np.ones(4), strehl=np.full(4, .8))
    opened = dict(variance=np.full(4, 4.0), strehl=np.full(4, .5), saturated=np.zeros(4, dtype=bool))
    prefix = f"evidence/evaluation/trials/{_case_id(case)}"
    raw = save_arrays(ledger, prefix + ".npz", outputs)
    shared = {}
    for kind, identity, arrays in (
            ("atmospheres", dict(stage="evaluation", seed=case["seed"], wind_scale=case["wind_scale"]),
             {"phase": np.zeros((4, cfg["n"], cfg["n"]))}),
            ("open_loop", {k: v for k, v in case.items() if k != "candidate"}, opened)):
        path = f"evidence/evaluation/{kind}/{_case_id(identity)}"
        ref = save_arrays(ledger, path + ".npz", arrays)
        meta = save_metadata(ledger, path + ".json", dict(identity=identity, raw=ref))
        shared[kind] = [ref, meta]
    summaries = paired_metrics(outputs, opened, cfg)
    detail = dict(case=case, config=cfg, raw=raw, atmosphere=shared["atmospheres"],
                  open_loop=shared["open_loop"], model=None, **summaries)
    detail_ref = save_metadata(ledger, prefix + ".json", detail)
    row = dict(case=case, status="completed", error=None, metrics=summaries["metrics"],
               artifacts=[raw, detail_ref, *shared["atmospheres"], shared["open_loop"][1]],
               open_loop=shared["open_loop"][0])
    return ledger, row, {"models": {}}, base, detail


def test_reconstructs_both_windows_from_raw(raw_trial):
    ledger, row, training, base, detail = raw_trial
    result = _reconstruct_trial(ledger, row, training, base)
    assert result["post_burn"] == detail["metrics"]
    assert result["fault_window"] == detail["fault_window"]


@pytest.mark.parametrize("damage", ["metric", "config", "window", "model", "reference"])
def test_raw_reader_rejects_changed_scientific_metadata(raw_trial, damage):
    ledger, row, training, base, detail = raw_trial
    if damage == "metric":
        row["metrics"]["wfe_nm"] += 1
    elif damage == "config":
        detail["config"]["wavelength_m"] *= 2
    elif damage == "window":
        detail["fault_window_frames"][0] += 1
    elif damage == "model":
        detail["model"] = {"status": "ready"}
    else:
        row["open_loop"] = detail["atmosphere"][0]
    # Replace metadata only in this synthetic corruption fixture. Updating its
    # reference ensures the scientific reconstruction, not just a hash, rejects it.
    from noether_optics.controller_selection import _bytes
    path = ledger.root / f"evidence/evaluation/trials/{_case_id(row['case'])}.json"
    path.write_bytes(_bytes(detail))
    row["artifacts"][1] = ledger.artifact(path.relative_to(ledger.root).as_posix())
    with pytest.raises(ValueError):
        _reconstruct_trial(ledger, row, training, base)


@pytest.fixture
def withheld_run(tmp_path, config):
    source, runtime = {"commit": "a" * 40, "files": {}}, {"synthetic_fixture": True}
    provenance = {"source_commit": source["commit"], "source_sha256": _digest(_bytes(source)),
                  "runtime_sha256": _digest(_bytes(runtime))}
    ledger = StudyLedger.create(tmp_path / "withheld", config, provenance)
    base = load_config(ROOT / config["base_config"])
    refs = [save_metadata(ledger, f"evidence/provenance/{name}.json", value)
            for name, value in (("source", source), ("runtime", runtime), ("study", config), ("base", base))]
    models = {c["id"]: dict(status="failed", error="synthetic model failure", artifacts=[])
              for c in study_plan(config)["candidates"] if c["controller"] in ("predictive", "learned")}
    ledger.bind_training(models, refs)
    for case in validation_cases(config):
        ledger.record_validation(dict(case=case, status="failed", metrics=None,
                                      error="synthetic trial failure", artifacts=[], open_loop=None))
    ledger.freeze_selection()
    return ledger


def test_all_withheld_is_complete_report_without_invented_evaluation(withheld_run):
    report = build_evaluation_report(withheld_run.root)
    assert report["validation_records"] == 600
    assert report["evaluation_records"] == 0
    assert len(report["comparisons"]) == 120
    assert all(row["withheld"] for row in report["conditions"])
    assert all(row["post_burn"]["interval"] is None for row in report["comparisons"])
    assert not (withheld_run.root / "evaluation").exists()


@pytest.mark.parametrize("damage", ["missing_validation", "changed_selection", "unreleased_evaluation"])
def test_report_requires_complete_frozen_evidence(withheld_run, damage):
    if damage == "missing_validation":
        next((withheld_run.root / "validation").glob("*.json")).unlink()
    elif damage == "changed_selection":
        path = withheld_run.root / "selection.json"
        value = json.loads(path.read_bytes()); value["result"]["pairwise_family"] = 1
        path.write_bytes(_bytes(value))
    else:
        folder = withheld_run.root / "evaluation"; folder.mkdir()
        (folder / ('0' * 64 + '.json')).write_text('{}')
    with pytest.raises(ValueError):
        build_evaluation_report(withheld_run.root)


def cost_fixture(tmp_path):
    manifest = {"provenance": {"source_commit": "a" * 40}}
    start = {"sequence": 0, "kind": "attempt_start", "manifest_sha256": _digest(_bytes(manifest)),
             "provenance": manifest["provenance"], "stage": "training", "clock": "perf_counter_ns", "units": "ns"}
    operation = {"sequence": 1, "kind": "operation", "operation": "kernel_execution",
                 "elapsed_ns": 123, "status": "completed"}
    root = tmp_path / "costs"; root.mkdir()
    path = root / "attempt.jsonl"
    path.write_bytes(b''.join(json.dumps(row).encode() + b'\n' for row in (start, operation)))
    return manifest, path


def test_interrupted_cost_attempt_keeps_operations_and_unknown_total(tmp_path):
    manifest, path = cost_fixture(tmp_path)
    with path.open('ab') as stream:
        stream.write(b'{"sequence":2,"kind":"operation"')
    result = cost_summary(tmp_path, manifest)
    assert result["attempts"][0]["status"] == "incomplete"
    assert result["attempts"][0]["elapsed_ns"] is None
    assert result["attempts"][0]["truncated_final_write"] is True
    assert result["operations"][0]["elapsed_ns"] == 123


def test_nested_costs_are_separate_categories(tmp_path):
    manifest, path = cost_fixture(tmp_path)
    with path.open('ab') as stream:
        for row in [{"sequence": 2, "kind": "operation", "operation": "trial_inclusive", "elapsed_ns": 500, "status": "completed"},
                    {"sequence": 3, "kind": "attempt_end", "elapsed_ns": 900, "status": "completed"}]:
            stream.write(json.dumps(row).encode() + b'\n')
    result = cost_summary(tmp_path, manifest)
    assert {r["operation"]: r["elapsed_ns"] for r in result["operations"]} == {"kernel_execution": 123, "trial_inclusive": 500}
    assert result["attempts"][0]["elapsed_ns"] == 900


def test_cost_corruption_before_final_write_is_not_excused_as_interruption(tmp_path):
    manifest, path = cost_fixture(tmp_path)
    with path.open('ab') as stream:
        stream.write(b'broken\n')
    with pytest.raises(ValueError, match="Malformed cost record"):
        cost_summary(tmp_path, manifest)
