"""Descriptive, raw-verified reporting after frozen controller selection.

This reader never creates a controller, atmosphere or detector stream. Acquisition
provenance belongs to the retained run, so reporting does not require the original
machine. It requires the complete released evaluation cohort and reconstructs
both reporting windows from archived frame streams before making comparisons.
"""
from __future__ import annotations

from itertools import combinations, product
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
from pathlib import Path

from scipy.stats import t

from .controller_evidence import check_provenance_binding, load_arrays
from .controller_physics import FULL_OUTPUT_KEYS, episode_config, paired_metrics
from .controller_protocol import PARAMETERS, study_plan
from .controller_selection import (
    StudyLedger, _bytes, _case_id, _digest, _read, validation_cases,
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def paired_interval(differences, *, alpha):
    """Student-t interval over independent seed-level paired differences.

    Zero sample variance produces the point interval implied by this estimator;
    it does not establish certainty outside the sampled population. Failed or
    nonfinite pairs must be handled by the caller, never silently filtered here.
    """
    values = list(differences)
    _require(len(values) >= 2 and 0 < alpha < 1, "Invalid paired interval inputs")
    _require(all(type(x) in (int, float) and math.isfinite(x) for x in values),
             "Paired interval requires every finite pair")
    mean = math.fsum(values) / len(values)
    variance = math.fsum((x - mean) ** 2 for x in values) / (len(values) - 1)
    radius = float(t.ppf(1 - alpha / 2, len(values) - 1)) * math.sqrt(variance / len(values))
    return {"n": len(values), "mean_nm": mean, "low_nm": mean - radius,
            "high_nm": mean + radius, "confidence": 1 - alpha}


def _metrics_match(saved, rebuilt, max_saturation):
    """Allow only float roundoff when decoding the same archived frame streams."""
    _require(set(saved) == set(rebuilt), "Metric schema changed")
    for key, expected in rebuilt.items():
        actual = saved[key]
        if isinstance(expected, bool) or expected is None:
            _require(actual is expected, "Finite flag or null metric changed")
        else:
            _require(type(actual) in (int, float) and math.isfinite(actual)
                     and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12),
                     "Metric differs from retained raw arrays")
    if rebuilt["all_outputs_finite"]:
        def eligibility(metrics):
            return (metrics["saturation_fraction"] <= max_saturation,
                    metrics["open_wfe_nm"] > 0 and metrics["wfe_nm"] < metrics["open_wfe_nm"])
        _require(eligibility(saved) == eligibility(rebuilt),
                 "Metric roundoff crosses a declared eligibility gate")


def _reconstruct_trial(ledger, row, training, base):
    case = row["case"]
    if row["status"] == "failed":
        return {"case": case, "status": "failed", "error": row["error"],
                "post_burn": None, "fault_window": None}
    prefix = f"evidence/{case['stage']}/trials/{_case_id(case)}"
    refs = {ref["path"]: ref for ref in row["artifacts"]}
    _require(prefix + ".json" in refs and prefix + ".npz" in refs,
             "Trial lacks declared raw and metadata artifacts")
    detail = _read(ledger.root / (prefix + ".json"))
    cfg = episode_config(base, ledger.config, case["stage"], case["photons"],
                         case["wind_scale"], case["candidate"])
    _require(_bytes(detail["case"]) == _bytes(case)
             and _bytes(detail["config"]) == _bytes(cfg), "Trial configuration changed")
    _require(detail["raw"] == refs[prefix + ".npz"], "Trial raw identity changed")
    _require(detail["model"] == training["models"].get(case["candidate"]["id"]),
             "Trial model differs from frozen training")
    condition = {key: value for key, value in case.items() if key != "candidate"}
    atmosphere = {"stage": case["stage"], "seed": case["seed"], "wind_scale": case["wind_scale"]}
    for kind, identity, evidence in (("atmospheres", atmosphere, detail["atmosphere"]),
                                     ("open_loop", condition, detail["open_loop"])):
        shared = f"evidence/{case['stage']}/{kind}/{_case_id(identity)}"
        _require(len(evidence) == 2 and evidence[0]["path"] == shared + ".npz"
                 and evidence[1]["path"] == shared + ".json", "Shared evidence identity changed")
        ledger._verify_artifacts(evidence)
        _require(_read(ledger.root / (shared + ".json")) == {"identity": identity, "raw": evidence[0]},
                 "Shared evidence metadata changed")
        _require(evidence[1] in row["artifacts"], "Shared metadata missing from terminal record")
        if kind == "atmospheres":
            _require(evidence[0] in row["artifacts"], "Atmosphere missing from terminal record")
        else:
            _require(evidence[0] == row["open_loop"], "Paired open-loop reference changed")
    outputs = load_arrays(ledger, detail["raw"])
    _require(FULL_OUTPUT_KEYS.issubset(outputs), "Full trial outputs required for reporting")
    reference = load_arrays(ledger, row["open_loop"])
    rebuilt = paired_metrics(outputs, reference, cfg)
    threshold = ledger.config["selection"]["max_saturation_fraction"]
    _metrics_match(row["metrics"], rebuilt["metrics"], threshold)
    _metrics_match(detail["metrics"], rebuilt["metrics"], threshold)
    _require(detail["post_burn_frames"] == rebuilt["post_burn_frames"]
             and detail["fault_window_frames"] == rebuilt["fault_window_frames"],
             "Reporting interval changed")
    if rebuilt["fault_window"] is None:
        _require(detail["fault_window"] is None, "Empty fault window changed")
    else:
        _metrics_match(detail["fault_window"], rebuilt["fault_window"], threshold)
    return {"case": case, "status": "completed", "error": None,
            "post_burn": rebuilt["metrics"], "fault_window": rebuilt["fault_window"]}


def _records(ledger, stage, cases, training, binding, seal, base):
    expected = {_case_id(case): case for case in cases}
    paths = list((ledger.root / stage).glob("*.json"))
    _require({path.stem for path in paths} == set(expected), f"Incomplete or unexpected {stage} cohort")
    rows = []
    for path in sorted(paths):
        _require(path.is_file() and not path.is_symlink(), "Invalid terminal record file")
        saved = _read(path)
        _require(set(saved) == {binding, "record"} and saved[binding] == seal,
                 f"{stage} binding changed")
        row = saved["record"]
        _require(_bytes(row["case"]) == _bytes(expected[path.stem]), f"{stage} identity changed")
        ledger._check_record(row, training)
        rows.append(_reconstruct_trial(ledger, row, training, base))
    return rows


def _usable(row, window, max_saturation):
    m = row[window]
    return (row["status"] == "completed" and m is not None and m["all_outputs_finite"]
            and m["saturation_fraction"] <= max_saturation
            and m["open_wfe_nm"] > 0 and m["wfe_nm"] < m["open_wfe_nm"])


def _condition_rows(config, selected, evaluation):
    result = []
    for controller, photons, wind, fault in product(
            PARAMETERS, config["conditions"]["photons"], config["conditions"]["wind_scales"],
            config["conditions"]["evaluation_faults"]):
        rows = [row for row in evaluation if row["case"]["candidate"]["controller"] == controller
                and row["case"]["photons"] == photons and row["case"]["wind_scale"] == wind
                and row["case"]["fault"] == fault]
        windows = {}
        for window in ("post_burn", "fault_window"):
            finite = [row[window] for row in rows if row["status"] == "completed"
                      and row[window] is not None and row[window]["all_outputs_finite"]]
            complete_finite = len(finite) == len(config["seeds"]["evaluation"])
            # A mean over only surviving seeds could hide a failure. Retain rows,
            # but provide an aggregate only for the full finite declared cohort.
            windows[window] = {
                "finite_trials": len(finite),
                "useful_trials": sum(_usable(row, window, config["selection"]["max_saturation_fraction"])
                                     for row in rows),
                "mean": {key: math.fsum(m[key] for m in finite) / len(finite)
                         for key in ("wfe_nm", "open_wfe_nm", "strehl", "saturation_fraction")}
                        if complete_finite else None,
            }
        result.append({"controller": controller, "candidate": selected[controller],
                       "photons": photons, "wind_scale": wind, "fault": fault,
                       "withheld": selected[controller] is None, "recorded_trials": len(rows),
                       "failed_trials": sum(row["status"] == "failed" for row in rows), **windows})
    return result


def _comparisons(config, selected, evaluation):
    family = study_plan(config)["predeclared_pairwise_family"]
    alpha = config["evaluation"]["familywise_alpha"]
    lookup = {(_case_id({key: value for key, value in row["case"].items() if key != "candidate"}),
               row["case"]["candidate"]["controller"]): row for row in evaluation}
    results = []
    for (left, right), photons, wind, fault in product(
            combinations(PARAMETERS, 2), config["conditions"]["photons"],
            config["conditions"]["wind_scales"], config["conditions"]["evaluation_faults"]):
        pairs = []
        for seed in config["seeds"]["evaluation"]:
            identity = _case_id(dict(stage="evaluation", photons=photons, wind_scale=wind,
                                    fault=fault, seed=seed, detector_seed=seed + config["seeds"]["noise_offset"]))
            pairs.append((lookup.get((identity, left)), lookup.get((identity, right))))
        windows = {}
        for window in ("post_burn", "fault_window"):
            unavailable = any(selected[name] is None for name in (left, right))
            complete = not unavailable and all(
                a is not None and b is not None and a["status"] == b["status"] == "completed"
                and a[window] is not None and b[window] is not None
                and a[window]["all_outputs_finite"] and b[window]["all_outputs_finite"] for a, b in pairs)
            if not complete:
                windows[window] = {"status": "controller_withheld" if unavailable else "failed_or_nonfinite_pairs",
                                   "interval": None, "winner": None}
                continue
            _require(all(a[window]["open_wfe_nm"] == b[window]["open_wfe_nm"] for a, b in pairs),
                     "Evaluation controllers do not share paired open-loop metrics")
            differences = [a[window]["wfe_nm"] - b[window]["wfe_nm"] for a, b in pairs]
            # The predeclared 120-comparison family applies to the primary window.
            # Fault-window intervals are explicitly exploratory, with no win claim.
            adjusted = window == "post_burn"
            interval = paired_interval(differences, alpha=alpha / family if adjusted else alpha)
            winner = None
            if adjusted:
                if interval["high_nm"] < 0 and all(_usable(a, window, config["selection"]["max_saturation_fraction"])
                                                    for a, _ in pairs):
                    winner = left
                elif interval["low_nm"] > 0 and all(_usable(b, window, config["selection"]["max_saturation_fraction"])
                                                     for _, b in pairs):
                    winner = right
            windows[window] = {"status": "complete", "interval": interval, "winner": winner,
                               "inference": "bonferroni_simultaneous" if adjusted else "exploratory_pointwise",
                               "family_size": family if adjusted else None,
                               "paired_differences_nm": differences}
        results.append({"left": left, "right": right, "difference": "left_minus_right",
                        "photons": photons, "wind_scale": wind, "fault": fault, **windows})
    _require(len(results) == family, "Comparison family changed")
    return results


def cost_summary(root, manifest):
    """Retain incomplete cost attempts and keep nested operation totals separate."""
    attempts, operations = [], {}
    for path in sorted((Path(root) / "costs").glob("*.jsonl")):
        _require(path.is_file() and not path.is_symlink(), "Invalid cost record file")
        raw = path.read_bytes()
        lines = raw.splitlines(keepends=True)
        rows, truncated = [], False
        for index, line in enumerate(lines):
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                _require(index == len(lines) - 1 and not line.endswith(b"\n"),
                         "Malformed cost record before final interrupted write")
                truncated = True
                break
            _require(type(row) is dict and type(row.get("sequence")) is int and row["sequence"] == index,
                     "Cost sequence changed")
            rows.append(row)
        _require(rows and rows[0].get("kind") == "attempt_start", "Cost attempt lacks admission")
        first = rows[0]
        _require(first.get("manifest_sha256") == _digest(_bytes(manifest))
                 and first.get("provenance") == manifest["provenance"]
                 and first.get("clock") == "perf_counter_ns" and first.get("units") == "ns",
                 "Cost attempt provenance or clock changed")
        _require(first.get("stage") in ("training", "validation", "evaluation"), "Invalid cost stage")
        end = None
        for index, row in enumerate(rows[1:], 1):
            _require(type(row.get("elapsed_ns")) is int and row["elapsed_ns"] >= 0,
                     "Invalid elapsed cost")
            _require(row.get("status") in ("completed", "interrupted"), "Invalid operation status")
            if row.get("kind") == "attempt_end":
                _require(index == len(rows) - 1 and not truncated, "Cost records follow attempt end")
                end = row
            else:
                _require(row.get("kind") == "operation" and isinstance(row.get("operation"), str),
                         "Unknown cost record")
                key = (first["stage"], row["operation"], row["status"])
                aggregate = operations.setdefault(key, {"count": 0, "elapsed_ns": 0})
                aggregate["count"] += 1
                aggregate["elapsed_ns"] += row["elapsed_ns"]
        attempts.append({"file": path.name, "sha256": hashlib.sha256(raw).hexdigest(),
                         "stage": first["stage"], "status": end["status"] if end else "incomplete",
                         "elapsed_ns": end["elapsed_ns"] if end else None,
                         "retained_records": len(rows), "truncated_final_write": truncated})
    return {"measurement_status": "retained_attempts" if attempts else "not_recorded", "attempts": attempts,
            "operations": [{"stage": key[0], "operation": key[1], "status": key[2], **value}
                           for key, value in sorted(operations.items())],
            "scope": "Inclusive operations overlap nested kernels; do not add their totals. Incomplete attempts are not zero-cost observations. These are study costs, not frame latency or deadlines."}


@contextmanager
def _costs_closed(root):
    path = root / ".cost-acquisition.lock"
    if not path.exists():
        yield
        return
    _require(not path.is_symlink(), "Invalid cost acquisition lock")
    with path.open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def build_evaluation_report(root: Path) -> dict:
    """Reconstruct a complete descriptive report without executing new trials."""
    root = Path(root).resolve()
    manifest = _read(root / "manifest.json")
    ledger = StudyLedger(root, manifest["config"], manifest["provenance"])
    with ledger._locked(), _costs_closed(root):
        training = ledger._training()
        base = _read(root / "evidence/provenance/base.json")
        check_provenance_binding(ledger, base, training["evidence"])
        selection, cases = ledger._gate()
        validation = _records(ledger, "validation", validation_cases(ledger.config), training,
                              "training_sha256", _digest(_bytes(training)), base)
        evaluation = _records(ledger, "evaluation", cases, training,
                              "selection_sha256", _digest(_bytes(selection)), base)
        chosen = selection["result"]["selected"]
        report = {
            "schema_version": 1, "study_id": ledger.config["study_id"],
            "status": "complete_descriptive_report", "acquisition_provenance": ledger.provenance,
            "manifest_sha256": _digest(_bytes(ledger.manifest)),
            "training_sha256": _digest(_bytes(training)), "selection_sha256": _digest(_bytes(selection)),
            "validation_records": len(validation), "evaluation_records": len(evaluation),
            "selection": selection["result"], "post_burn_frames": [ledger.config["loop"]["burn_in"], ledger.config["loop"]["frames"]],
            "fault_window_frames": [ledger.config["loop"]["fault_onset"], ledger.config["loop"]["fault_end"]],
            "familywise_alpha": ledger.config["evaluation"]["familywise_alpha"],
            "primary_pairwise_family": study_plan(ledger.config)["predeclared_pairwise_family"],
            "conditions": _condition_rows(ledger.config, chosen, evaluation),
            "comparisons": _comparisons(ledger.config, chosen, evaluation),
            "validation_trials": validation, "evaluation_trials": evaluation,
            "costs": cost_summary(root, manifest),
            "limitations": [
                "Simulation evidence; no hardware or observatory reliability claim.",
                "Student-t coverage assumes independent seed-level differences and its sampling model.",
                "Primary simultaneous intervals cover the predeclared pairwise family, not all reported statistics.",
                "Fault-window intervals are exploratory; no separate superiority verdict is assigned.",
                "Single learned initialization, finite training budget and Gaussian-to-Poisson training mismatch.",
            ],
        }
        _bytes(report)  # Refuse any accidental nonfinite JSON scalar.
        return report
