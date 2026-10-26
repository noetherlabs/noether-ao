"""Exercise the complete study command sequence on reduced nonreserved inputs.

This smoke completes training and validation, freezes selection, and persists one
held-out evaluation trial.  It does not complete the evaluation cohort, collect
timings, or support any controller-performance claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from noether_optics.artifacts import load_config
from noether_optics.controller_protocol import load_study, study_plan

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path("configs/controller_runner_smoke_v1.yaml")
RESERVED_CONFIG = ROOT / "configs/controller_study_v1.yaml"
COMPONENT_CONFIG = ROOT / "configs/controller_physics_smoke.yaml"
EVALUATION_LIMIT = 1


def _json_bytes(value) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _write_json(path: Path, value) -> None:
    path.write_bytes(_json_bytes(value))


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _streams(study: dict) -> set[int]:
    seeds = study["seeds"]
    roots = {seed for role in ("training", "validation", "evaluation") for seed in seeds[role]}
    return roots | {seed + seeds["noise_offset"] for seed in roots} | {seeds["initialization"]}


def seed_audit(study: dict) -> dict:
    """Prove smoke atmosphere, detector and initialization streams are isolated."""
    reserved = load_study(RESERVED_CONFIG)
    component = load_config(COMPONENT_CONFIG)
    component_roots = {seed for role in ("training", "validation", "evaluation")
                       for seed in component[f"{role}_seeds"]}
    component_streams = (component_roots
                         | {seed + study["seeds"]["noise_offset"] for seed in component_roots}
                         | {component["initialization_seed"]})
    smoke_streams = _streams(study)
    return {
        "smoke_streams": sorted(smoke_streams),
        "reserved_overlap": sorted(smoke_streams & _streams(reserved)),
        "component_overlap": sorted(smoke_streams & component_streams),
        "separate": not (smoke_streams & _streams(reserved) or smoke_streams & component_streams),
    }


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _invoke(arguments: list[str], environment: dict[str, str]) -> dict:
    print("Running:", "noether-optics", *arguments, flush=True)
    completed = subprocess.run(
        [sys.executable, "-m", "noether_optics.cli", *arguments], cwd=ROOT,
        env=environment, text=True, capture_output=True)
    if completed.returncode:
        if completed.stdout:
            print(completed.stdout, file=sys.stderr)
        if completed.stderr:
            print(completed.stderr, file=sys.stderr)
        raise RuntimeError(
            f"Study smoke command failed with status {completed.returncode}: {arguments[0]}")
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Study smoke command did not emit one JSON result: {arguments[0]}"
        ) from error
    if not isinstance(result, dict):
        raise ValueError("Study smoke command result must be an object")
    return {"arguments": arguments, "result": result}


def _protected_files(root: Path) -> dict[str, str]:
    paths = [root / "commands.json", *sorted((root / "run").rglob("*"))]
    return {path.relative_to(root).as_posix(): _digest(path) for path in paths if path.is_file()}


def _verify_protected(root: Path) -> dict[str, str]:
    protected = json.loads((root / "protected_sha256.json").read_text())
    if not isinstance(protected, dict) or not protected:
        raise ValueError("Missing protected runner-smoke inventory")
    for name, expected in protected.items():
        path = root / name
        if not path.is_file() or _digest(path) != expected:
            raise ValueError(f"Protected runner-smoke evidence changed: {name}")
    current = _protected_files(root)
    if current != protected:
        raise ValueError("Runner-smoke evidence inventory changed")
    return protected


def _derive(root: Path) -> dict:
    protected = _verify_protected(root)
    commands = json.loads((root / "commands.json").read_text())
    run = root / "run"
    manifest = json.loads((run / "manifest.json").read_text())
    study = manifest["config"]
    plan = study_plan(study)
    training = commands["training"]["result"]
    validation = commands["validation"]["result"]
    selection = commands["selection"]["result"]
    evaluation = commands["evaluation"]["result"]
    selected = {name: candidate for name, candidate in selection["result"]["selected"].items()
                if candidate is not None}
    withheld = sorted(set(selection["result"]["selected"]) - set(selected))
    expected_evaluation = (len(selected) * len(study["conditions"]["photons"])
                           * len(study["conditions"]["wind_scales"])
                           * len(study["conditions"]["evaluation_faults"])
                           * len(study["seeds"]["evaluation"]))
    source = json.loads((run / "evidence/provenance/source.json").read_text())
    seed_check = commands["seed_audit"]
    gates = {
        "nonreserved_seed_streams": seed_check["separate"],
        "resource_preflight_admitted": commands["resources"]["result"]["admitted"],
        "training_complete": training["complete"] and training["training_bound"]
            and training["pilot_records"] == training["planned_pilot_records"]
            and training["terminal_model_records"] == training["planned_model_records"],
        "validation_complete": validation["complete"]
            and validation["total_records"] == validation["planned_records"]
            == plan["planned_validation_trials"],
        "selection_frozen": selection["result"]["validation_records"]
            == plan["planned_validation_trials"] and bool(selected),
        "selection_reentry_exact": commands["selection_recheck"]["result"] == selection,
        "held_out_evaluation_entered": evaluation["new_records"] == EVALUATION_LIMIT
            and evaluation["total_records"] == EVALUATION_LIMIT
            and evaluation["planned_records"] == expected_evaluation > 0,
        "durable_stage_counts": len(list((run / "validation").glob("*.json")))
            == plan["planned_validation_trials"]
            and len(list((run / "evaluation").glob("*.json"))) == EVALUATION_LIMIT,
        "source_provenance_bound": source["commit"] == manifest["provenance"]["source_commit"],
    }
    return {
        "schema_version": 1,
        "kind": "controller_runner_smoke",
        "study_id": study["study_id"],
        "source_commit": source["commit"],
        "pilot_records": training["pilot_records"],
        "terminal_model_records": training["terminal_model_records"],
        "validation_records": validation["total_records"],
        "selected_controllers": sorted(selected),
        "withheld_controllers": withheld,
        "evaluation_records": evaluation["total_records"],
        "planned_evaluation_records_for_selection": evaluation["planned_records"],
        "protected_files": len(protected),
        "gates": gates,
        "all_pass": all(gates.values()),
        "scope": ("End-to-end command and evidence smoke with one held-out trial; "
                  "not a complete evaluation cohort, timing result or controller claim."),
    }


def _report(summary: dict) -> str:
    gate_rows = "\n".join(
        f"| `{name}` | {'pass' if passed else 'fail'} |"
        for name, passed in summary["gates"].items()
    )
    selected = ", ".join(summary["selected_controllers"]) or "none"
    withheld = ", ".join(summary["withheld_controllers"]) or "none"
    evaluation_progress = (
        f"{summary['evaluation_records']} of "
        f"{summary['planned_evaluation_records_for_selection']}"
    )
    return f"""# Controller runner smoke

This retained run exercises the provenance-bound training, validation, selection
and evaluation commands on reduced, nonreserved inputs. It is implementation
evidence, not a controller comparison or timing result.

| Field | Value |
| --- | --- |
| Source commit | `{summary['source_commit']}` |
| Training pilots | {summary['pilot_records']} |
| Terminal fitted models | {summary['terminal_model_records']} |
| Validation records | {summary['validation_records']} |
| Selected controllers | {selected} |
| Withheld controllers | {withheld} |
| Held-out records acquired | {evaluation_progress} |
| Protected files | {summary['protected_files']} |

## Gates

| Gate | Result |
| --- | --- |
{gate_rows}

The held-out stage intentionally stops after one durable record. The smoke does
not estimate controller performance, complete a statistical cohort, compare
resource costs or alter the frozen `controller_study_v1` protocol.
"""


def rebuild(root: Path) -> dict:
    root = Path(root).resolve()
    summary = _derive(root)
    _write_json(root / "summary.json", summary)
    (root / "README.md").write_text(_report(summary))
    return summary


def run(config_path: Path, root: Path) -> dict:
    if os.environ.get("CI"):
        raise RuntimeError("Run physical smoke artifacts locally, outside CI")
    if subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT, text=True).strip():
        raise RuntimeError("Commit tracked source changes before the full-runner smoke")
    config_path = Path(config_path)
    absolute_config = config_path if config_path.is_absolute() else ROOT / config_path
    study = load_study(absolute_config)
    audit = seed_audit(study)
    if not audit["separate"]:
        raise ValueError("Runner-smoke streams overlap reserved or component-smoke inputs")
    plan = study_plan(study)
    root = Path(root).resolve()
    results_root = (ROOT / "results").resolve()
    if not root.is_relative_to(results_root):
        raise ValueError("Runner-smoke output must stay under results/")
    run_root = root / "run"
    environment = dict(os.environ, JAX_ENABLE_X64="1", JAX_PLATFORMS="cpu", MPLBACKEND="Agg")
    environment.pop("CI", None)
    config_argument = _relative(absolute_config)
    run_argument = _relative(run_root)
    fitted_models = sum(row["controller"] in ("predictive", "learned")
                        for row in plan["candidates"])
    resources = _invoke(["study-resources", "--config", config_argument,
                         "--target", run_argument], environment)
    root.mkdir(parents=True, exist_ok=False)
    commands = {
        "seed_audit": audit,
        "resources": resources,
        "training": _invoke(["study-train", "--config", config_argument, "--out", run_argument,
                             "--create", "--pilot-limit", str(plan["planned_training_episodes"]),
                             "--model-limit", str(fitted_models)], environment),
        "validation": _invoke(["study-validate", "--config", config_argument, "--out", run_argument,
                               "--limit", str(plan["planned_validation_trials"])], environment),
        "selection": _invoke(["study-select", "--config", config_argument,
                              "--out", run_argument], environment),
        "evaluation": _invoke(["study-evaluate", "--config", config_argument, "--out", run_argument,
                               "--limit", str(EVALUATION_LIMIT)], environment),
    }
    commands["selection_recheck"] = _invoke(
        ["study-select", "--config", config_argument, "--out", run_argument], environment)
    _write_json(root / "commands.json", commands)
    _write_json(root / "protected_sha256.json", _protected_files(root))
    return rebuild(root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--rebuild", type=Path)
    args = parser.parse_args()
    if (args.out is None) == (args.rebuild is None):
        parser.error("Choose exactly one of --out or --rebuild")
    summary = rebuild(args.rebuild) if args.rebuild else run(args.config, args.out)
    print(json.dumps(summary, indent=2, allow_nan=False))
    if not summary["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
