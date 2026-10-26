"""Bounded training coordination and the single aggregate freeze.

    admitted run -> bounded pilot acquisition -> every declared fitted candidate
        -> verified terminal model/evidence records -> immutable training.json

The numerical work remains in the pilot, predictor and learned-fit modules. This
coordinator adds deterministic ordering, chunk budgets, restart discovery and the
all-candidates barrier. It does not access validation/evaluation cohorts, collect
timings, expose a CLI or weaken a terminal failure into a retry.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl

from .controller_physics import training_cases
from .controller_pilots import acquire_pilots, fit_predictor_record, load_learned_inputs
from .controller_protocol import study_plan
from .controller_selection import _case_id, _read
from .controller_training_journal import fit_learned_record, load_learned_record


def run_training_chunk(ledger, base, shared_evidence, *, pilot_limit, model_limit):
    """Advance pilots and fitted candidates within explicit positive chunk limits.

    Existing terminal successes and failures are verified and skipped. The final
    training bundle is published only when every predictive and learned candidate
    has a terminal record. Re-entering a bound stage verifies and reports it without
    invoking calibration, simulation, fitting or optimizer construction.
    """
    ledger._check_limit(pilot_limit)
    ledger._check_limit(model_limit)
    candidates = _fitted_candidates(ledger)
    with _coordinator_lock(ledger):
        if (ledger.root / "training.json").exists():
            with ledger._locked():
                bundle = ledger._training()
            return _summary(ledger, bundle["models"], 0, 0, True)

        pilot = acquire_pilots(ledger, base, shared_evidence, pilot_limit)
        if not pilot["complete"]:
            return _summary(ledger, {}, pilot["new_records"], 0, False)

        terminal = _existing_records(ledger, base, shared_evidence, candidates)
        learned_inputs = None
        written = 0
        for candidate in candidates:
            if candidate["id"] in terminal:
                continue
            if candidate["controller"] == "predictive":
                result = fit_predictor_record(ledger, base, candidate, shared_evidence)
            else:
                if learned_inputs is None:
                    learned_inputs = load_learned_inputs(ledger, base, shared_evidence)
                cal, episodes = learned_inputs
                result = fit_learned_record(
                    ledger, base, candidate, cal, episodes, shared_evidence)
            terminal[candidate["id"]] = result
            written += 1
            if written == model_limit:
                break

        if len(terminal) == len(candidates):
            models = {candidate["id"]: terminal[candidate["id"]]["model"] for candidate in candidates}
            refs = [ref for candidate in candidates for ref in terminal[candidate["id"]]["evidence"]]
            refs.extend(ledger.artifact(
                f"evidence/training/{candidate['id']}/complete.json") for candidate in candidates)
            unique = {ref["path"]: ref for ref in refs}
            ledger.bind_training(models, [unique[path] for path in sorted(unique)])
            bundle = ledger._training()
            return _summary(ledger, bundle["models"], pilot["new_records"], written, True)
        return _summary(
            ledger, {key: row["model"] for key, row in terminal.items()},
            pilot["new_records"], written, False)


@contextmanager
def _coordinator_lock(ledger):
    """Serialize coordinators while candidate functions use the ledger lock."""
    with (ledger.root / ".training-stage.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _fitted_candidates(ledger):
    return [candidate for candidate in study_plan(ledger.config)["candidates"]
            if candidate["controller"] in ("predictive", "learned")]


def _existing_records(ledger, base, shared_evidence, candidates):
    terminal = {}
    for candidate in candidates:
        path = ledger.root / f"evidence/training/{candidate['id']}/complete.json"
        if not path.exists():
            continue
        result = (fit_predictor_record(ledger, base, candidate, shared_evidence)
                  if candidate["controller"] == "predictive"
                  else load_learned_record(ledger, base, candidate, shared_evidence))
        if result is None:
            raise ValueError("Declared terminal model record is missing")
        terminal[candidate["id"]] = result
    return terminal


def _pilot_counts(ledger):
    cases = training_cases(ledger.config)
    records = []
    root = ledger.root / "evidence/training/pilots/records"
    for case in cases:
        path = root / f"{_case_id(case)}.json"
        if path.exists():
            records.append(_read(path))
    return len(records), sum(row.get("status") == "failed" for row in records), len(cases)


def _summary(ledger, models, new_pilots, new_models, bound):
    pilot_records, failed_pilots, planned_pilots = _pilot_counts(ledger)
    planned_models = len(_fitted_candidates(ledger))
    return {
        "new_pilot_records": new_pilots,
        "pilot_records": pilot_records,
        "planned_pilot_records": planned_pilots,
        "failed_pilot_records": failed_pilots,
        "new_model_records": new_models,
        "terminal_model_records": len(models),
        "planned_model_records": planned_models,
        "failed_model_records": sum(row["status"] == "failed" for row in models.values()),
        "complete": bound,
        "training_bound": bound,
    }
