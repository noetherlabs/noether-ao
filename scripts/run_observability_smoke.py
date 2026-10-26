"""Exercise the retained observability training and validation path physically.

The smoke uses reduced, nonreserved inputs. It freezes a training threshold and a
validation decision, but it never acquires evaluation data, records timings or
supports a detector-performance claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from noether_optics.observability_protocol import (
    PRIOR_ROOTS,
    SMOKE_STUDY_ID,
    load_protocol,
    protocol_plan,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path("configs/optical_gain_observability_runner_smoke_v1.yaml")
RESERVED_CONFIG = ROOT / "configs/optical_gain_observability_v1.yaml"


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


def _streams(protocol: dict) -> set[int]:
    roots = {
        seed
        for role in ("training", "validation", "evaluation")
        for seed in protocol["seeds"][role]
    }
    offset = protocol["seeds"]["noise_offset"]
    return roots | {seed + offset for seed in roots}


def seed_audit(protocol: dict) -> dict:
    """Prove smoke streams are distinct from frozen P2 and earlier experiments."""
    reserved = load_protocol(RESERVED_CONFIG, ROOT)
    offset = protocol["seeds"]["noise_offset"]
    prior = set(PRIOR_ROOTS) | {seed + offset for seed in PRIOR_ROOTS}
    smoke = _streams(protocol)
    reserved_overlap = sorted(smoke & _streams(reserved))
    prior_overlap = sorted(smoke & prior)
    return {
        "smoke_streams": sorted(smoke),
        "reserved_v1_overlap": reserved_overlap,
        "prior_experiment_overlap": prior_overlap,
        "separate": not reserved_overlap and not prior_overlap,
    }


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _invoke(arguments: list[str], environment: dict[str, str]) -> dict:
    print("Running:", "noether-optics", *arguments, flush=True)
    completed = subprocess.run(
        [sys.executable, "-m", "noether_optics.cli", *arguments],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
    )
    if completed.returncode:
        if completed.stdout:
            print(completed.stdout, file=sys.stderr)
        if completed.stderr:
            print(completed.stderr, file=sys.stderr)
        raise RuntimeError(
            "Observability smoke command failed with status "
            f"{completed.returncode}: {arguments[0]}"
        )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ValueError(
            "Observability smoke command did not emit one JSON result: "
            f"{arguments[0]}"
        ) from error
    if not isinstance(result, dict):
        raise ValueError("Observability smoke command result must be an object")
    return {"arguments": arguments, "result": result}


def _protected_files(root: Path) -> dict[str, str]:
    paths = [root / "commands.json", *sorted((root / "run").rglob("*"))]
    protected = {}
    for path in paths:
        if path.is_symlink():
            raise ValueError("Observability-smoke evidence cannot contain symlinks")
        if path.is_file():
            protected[path.relative_to(root).as_posix()] = _digest(path)
    return protected


def _verify_protected(root: Path) -> dict[str, str]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Observability-smoke root must be a regular directory")
    inventory = root / "protected_sha256.json"
    if inventory.is_symlink() or not inventory.is_file():
        raise ValueError("Missing protected observability-smoke inventory")
    data = inventory.read_bytes()
    protected = json.loads(data)
    if not isinstance(protected, dict) or not protected:
        raise ValueError("Missing protected observability-smoke inventory")
    if data != _json_bytes(protected):
        raise ValueError("Protected observability-smoke inventory is not canonical")
    for name, expected in protected.items():
        path = root / name
        if path.is_symlink() or not path.is_file() or _digest(path) != expected:
            raise ValueError(f"Protected observability-smoke evidence changed: {name}")
    current = _protected_files(root)
    if current != protected:
        raise ValueError("Observability-smoke evidence inventory changed")
    return protected


def _paired_controller_error(decision: dict) -> list[dict]:
    records = decision.get("validation_records")
    if not isinstance(records, list):
        raise ValueError("Smoke decision is missing validation records")
    grouped = {}
    for row in records:
        if row.get("status") != "completed" or not isinstance(row.get("case"), dict):
            raise ValueError("Smoke controller-error pairs require completed records")
        case = row["case"]
        key = (
            case["regime_id"],
            case["atmosphere_seed"],
            case["detector_seed"],
        )
        grouped.setdefault(key, {})[case["condition"]] = row
    pairs = []
    for key in sorted(grouped):
        conditions = grouped[key]
        if set(conditions) != {"clean", "gain_035"}:
            raise ValueError("Smoke controller-error records are not paired")
        clean = conditions["clean"]["metrics"]["retained_trial_metrics"]
        gain = conditions["gain_035"]["metrics"]["retained_trial_metrics"]
        pairs.append(
            {
                "regime_id": key[0],
                "atmosphere_seed": key[1],
                "detector_seed": key[2],
                "wfe_nm_delta": gain["wfe_nm"] - clean["wfe_nm"],
                "strehl_delta": gain["strehl"] - clean["strehl"],
                "clean_saturation_fraction": clean["saturation_fraction"],
                "gain_saturation_fraction": gain["saturation_fraction"],
            }
        )
    return pairs


def _reported_metrics(decision: dict, pairs: list[dict]) -> bool:
    regimes = decision.get("regimes")
    if not isinstance(regimes, list) or len(regimes) != 1:
        return False
    row = regimes[0]
    clean = row.get("clean", {})
    gain = row.get("gain_035", {})
    integrity = row.get("integrity", {})
    return (
        {
            "alarm_runs",
            "raw_crossings",
            "raw_crossing_frames",
            "raw_crossing_fraction",
            "alarm_frames",
            "alarm_eligible_frames",
            "alarm_frame_fraction",
        }.issubset(clean)
        and {
            "pre_onset_alarm_runs",
            "boundary_straddling_alarm_runs",
            "detected_runs",
            "missed_runs",
            "detected_only_delay_frames",
        }.issubset(gain)
        and {
            "all_completed_outputs_finite",
            "fallback_frames",
            "unavailable_frames",
            "saturated_frames",
            "saturation_fraction",
        }.issubset(integrity)
        and len(pairs) == 2
    )


def _derive(root: Path) -> dict:
    protected = _verify_protected(root)
    command_path = root / "commands.json"
    command_bytes = command_path.read_bytes()
    commands = json.loads(command_bytes)
    if command_bytes != _json_bytes(commands):
        raise ValueError("Observability-smoke command record is not canonical")
    run = root / "run"
    manifest = json.loads((run / "manifest.json").read_text())
    protocol = manifest["protocol"]
    if protocol["study_id"] != SMOKE_STUDY_ID:
        raise ValueError("Protected run is not the observability runner smoke")
    plan = protocol_plan(protocol)
    training = commands["training"]["result"]
    threshold = commands["threshold"]["result"]
    validation = commands["validation"]["result"]
    decision = commands["decision"]["result"]
    source = json.loads((run / "evidence/provenance/source.json").read_text())
    pairs = _paired_controller_error(decision["result"])
    threshold_path = run / threshold["artifact"]["path"]
    decision_path = run / decision["artifact"]["path"]
    gates = {
        "nonreserved_seed_streams": commands["seed_audit"]["separate"],
        "resource_preflight_admitted": commands["resources"]["result"]["admitted"],
        "training_complete": training["complete"]
        and training["total_records"] == training["planned_records"]
        == plan["planned_training_trials"],
        "threshold_frozen": len(threshold["result"]["training_records"])
        == plan["planned_training_trials"]
        and len(threshold["result"]["thresholds"]) == len(plan["regimes"]),
        "threshold_artifact_bound": threshold["artifact"]["path"] == "thresholds.json"
        and _digest(threshold_path) == threshold["artifact"]["sha256"],
        "threshold_reentry_exact": commands["threshold_recheck"]["result"]
        == threshold,
        "validation_complete": validation["complete"]
        and validation["total_records"] == validation["planned_records"]
        == plan["planned_validation_trials"],
        "validation_decision_frozen": len(decision["result"]["validation_records"])
        == plan["planned_validation_trials"],
        "validation_artifact_bound": decision["artifact"]["path"]
        == "validation_release.json"
        and _digest(decision_path) == decision["artifact"]["sha256"],
        "validation_reentry_exact": commands["decision_recheck"]["result"]
        == decision,
        "declared_metrics_reported": _reported_metrics(decision["result"], pairs),
        "durable_stage_counts": len(list((run / "training").glob("*.json")))
        == plan["planned_training_trials"]
        and len(list((run / "validation").glob("*.json")))
        == plan["planned_validation_trials"],
        "source_provenance_bound": source["commit"]
        == manifest["provenance"]["source_commit"],
        "evaluation_not_acquired": not (run / "evaluation").exists(),
    }
    regime = decision["result"]["regimes"][0]
    return {
        "schema_version": 1,
        "kind": "optical_gain_observability_runner_smoke",
        "study_id": protocol["study_id"],
        "source_commit": source["commit"],
        "training_records": training["total_records"],
        "validation_records": validation["total_records"],
        "threshold_sha256": threshold["artifact"]["sha256"],
        "validation_decision_sha256": decision["artifact"]["sha256"],
        "validation_status": decision["result"]["status"],
        "observed": {
            "clean": regime["clean"],
            "gain_035": regime["gain_035"],
            "paired_controller_error": pairs,
            "integrity": regime["integrity"],
        },
        "protected_files": len(protected),
        "gates": gates,
        "all_pass": all(gates.values()),
        "scope": (
            "Reduced nonreserved physical runner and evidence smoke; not a detector "
            "validation, evaluation release, timing result or safety claim."
        ),
    }


def _value(value) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return json.dumps(value, sort_keys=True)


def _report(summary: dict) -> str:
    gate_rows = "\n".join(
        f"| `{name}` | {'pass' if passed else 'fail'} |"
        for name, passed in summary["gates"].items()
    )
    observed = []
    for group, values in summary["observed"].items():
        rows = values.items() if isinstance(values, dict) else [("pairs", values)]
        observed.extend(
            f"| `{group}.{name}` | {_value(value)} |" for name, value in rows
        )
    observed_rows = "\n".join(observed)
    return f"""# Optical-gain observability runner smoke

This retained run exercises the provenance-bound physical training, threshold,
validation and decision paths on reduced, nonreserved inputs. The observed values
below are implementation-smoke outcomes, not detector validation or a safety claim.

| Field | Value |
| --- | --- |
| Source commit | `{summary['source_commit']}` |
| Training records | {summary['training_records']} |
| Validation records | {summary['validation_records']} |
| Validation artifact status | `{summary['validation_status']}` |
| Protected files | {summary['protected_files']} |

## Gates

| Gate | Result |
| --- | --- |
{gate_rows}

## Retained smoke observations

| Quantity | Value |
| --- | ---: |
{observed_rows}

No evaluation trial was acquired. The smoke records no timing, does not estimate a
false-alarm or detection rate, and does not alter or authorize the frozen
`optical_gain_observability_v1` cohort.
"""


def rebuild(root: Path) -> dict:
    root = Path(root).resolve()
    summary = _derive(root)
    _write_json(root / "summary.json", summary)
    (root / "README.md").write_text(_report(summary))
    return summary


def run(config_path: Path, root: Path) -> dict:
    if os.environ.get("CI"):
        raise RuntimeError("Run physical observability smoke artifacts locally, outside CI")
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        text=True,
    ).strip():
        raise RuntimeError("Commit tracked source changes before the observability smoke")
    config_path = Path(config_path)
    absolute_config = config_path if config_path.is_absolute() else ROOT / config_path
    protocol = load_protocol(absolute_config, ROOT)
    if protocol["study_id"] != SMOKE_STUDY_ID:
        raise ValueError("Physical smoke requires the dedicated observability smoke protocol")
    audit = seed_audit(protocol)
    if not audit["separate"]:
        raise ValueError("Observability-smoke streams overlap protected experiment inputs")
    plan = protocol_plan(protocol)
    root = Path(root).resolve()
    results_root = (ROOT / "results").resolve()
    if not root.is_relative_to(results_root):
        raise ValueError("Observability-smoke output must stay under results/")
    run_root = root / "run"
    environment = dict(
        os.environ,
        JAX_ENABLE_X64="1",
        JAX_PLATFORMS="cpu",
        MPLBACKEND="Agg",
    )
    environment.pop("CI", None)
    config_argument = _relative(absolute_config)
    run_argument = _relative(run_root)
    resources = _invoke(
        ["observability-resources", "--config", config_argument, "--target", run_argument],
        environment,
    )
    root.mkdir(parents=True, exist_ok=False)
    commands = {
        "seed_audit": audit,
        "resources": resources,
        "training": _invoke(
            [
                "observability-train",
                "--config",
                config_argument,
                "--out",
                run_argument,
                "--create",
                "--limit",
                str(plan["planned_training_trials"]),
            ],
            environment,
        ),
        "threshold": _invoke(
            ["observability-freeze-training", "--config", config_argument, "--out", run_argument],
            environment,
        ),
    }
    commands["threshold_recheck"] = _invoke(
        ["observability-freeze-training", "--config", config_argument, "--out", run_argument],
        environment,
    )
    commands["validation"] = _invoke(
        [
            "observability-validate",
            "--config",
            config_argument,
            "--out",
            run_argument,
            "--limit",
            str(plan["planned_validation_trials"]),
        ],
        environment,
    )
    commands["decision"] = _invoke(
        ["observability-freeze-validation", "--config", config_argument, "--out", run_argument],
        environment,
    )
    commands["decision_recheck"] = _invoke(
        ["observability-freeze-validation", "--config", config_argument, "--out", run_argument],
        environment,
    )
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
