"""Config-driven experiment command line."""
import argparse
import json
import os
from pathlib import Path


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main() -> None:
    """Dispatch one experiment; enable double precision before importing JAX."""
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    os.environ.setdefault("MPLBACKEND", "Agg")
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="experiment", required=True)
    for name in ("latency", "validate", "faults", "fidelity", "mixed"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True)
        command.add_argument("--out", required=True)
        if name == "latency":
            command.add_argument(
                "--resume", action="store_true",
                help="Resume only unfinished latency points on an identical runtime")
    training = commands.add_parser("study-train")
    training.add_argument("--config", required=True)
    training.add_argument("--out", required=True)
    training.add_argument("--create", action="store_true",
                          help="Create a new study run; omit when resuming the same bound run")
    training.add_argument("--pilot-limit", type=_positive_integer, required=True,
                          help="Maximum new pilot records")
    training.add_argument("--model-limit", type=_positive_integer, required=True,
                          help="Maximum new terminal model records")
    validation = commands.add_parser("study-validate")
    validation.add_argument("--config", required=True)
    validation.add_argument("--out", required=True)
    validation.add_argument("--limit", type=_positive_integer, required=True,
                            help="Maximum new validation records")
    selection = commands.add_parser("study-select")
    selection.add_argument("--config", required=True)
    selection.add_argument("--out", required=True)
    evaluation = commands.add_parser("study-evaluate")
    evaluation.add_argument("--config", required=True)
    evaluation.add_argument("--out", required=True)
    evaluation.add_argument("--limit", type=_positive_integer, required=True,
                            help="Maximum new evaluation records")
    resources = commands.add_parser("study-resources")
    resources.add_argument("--config", required=True)
    resources.add_argument("--target", required=True,
                           help="Planned study directory; it is inspected but not created")
    observability_resources = commands.add_parser("observability-resources")
    observability_resources.add_argument("--config", required=True)
    observability_resources.add_argument(
        "--target", required=True,
        help="Planned observability directory; it is inspected but not created")
    observability_training = commands.add_parser("observability-train")
    observability_training.add_argument("--config", required=True)
    observability_training.add_argument("--out", required=True)
    observability_training.add_argument(
        "--create", action="store_true",
        help="Create a new observability run; omit when resuming the same bound run")
    observability_training.add_argument(
        "--limit", type=_positive_integer, required=True,
        help="Maximum new training records")
    observability_freeze_training = commands.add_parser(
        "observability-freeze-training")
    observability_freeze_training.add_argument("--config", required=True)
    observability_freeze_training.add_argument("--out", required=True)
    observability_validation = commands.add_parser("observability-validate")
    observability_validation.add_argument("--config", required=True)
    observability_validation.add_argument("--out", required=True)
    observability_validation.add_argument(
        "--limit", type=_positive_integer, required=True,
        help="Maximum new validation records")
    observability_freeze_validation = commands.add_parser(
        "observability-freeze-validation")
    observability_freeze_validation.add_argument("--config", required=True)
    observability_freeze_validation.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.experiment == "observability-resources":
        from .observability_resources import load_resource_inputs, resource_preflight
        protocol, base = load_resource_inputs(Path(args.config))
        result = resource_preflight(protocol, base, Path(args.target))
        print(json.dumps(result, sort_keys=True))
        if not result["admitted"]:
            raise SystemExit(2)
        return
    if args.experiment == "observability-train":
        from .observability_resources import load_resource_inputs, resource_preflight
        from .observability_trials import open_observability_run, train_trials
        if args.create:
            protocol, planned_base = load_resource_inputs(Path(args.config))
            preflight = resource_preflight(protocol, planned_base, Path(args.out))
            if not preflight["admitted"]:
                print(json.dumps(preflight, sort_keys=True))
                raise SystemExit(2)
        ledger, base, _ = open_observability_run(
            Path(args.out), Path(args.config), create=args.create)
        result = train_trials(ledger, base, args.limit)
        print(json.dumps(result, sort_keys=True))
        return
    if args.experiment == "observability-freeze-training":
        from .observability_trials import open_observability_run
        ledger, _, _ = open_observability_run(Path(args.out), Path(args.config))
        result, artifact = ledger.freeze_training()
        print(json.dumps({"artifact": artifact, "result": result}, sort_keys=True))
        return
    if args.experiment == "observability-validate":
        from .observability_trials import open_observability_run, validate_trials
        ledger, base, _ = open_observability_run(Path(args.out), Path(args.config))
        result = validate_trials(ledger, base, args.limit)
        print(json.dumps(result, sort_keys=True))
        return
    if args.experiment == "observability-freeze-validation":
        from .observability_trials import open_observability_run
        ledger, _, _ = open_observability_run(Path(args.out), Path(args.config))
        result, artifact = ledger.freeze_validation()
        print(json.dumps({"artifact": artifact, "result": result}, sort_keys=True))
        return
    if args.experiment == "study-resources":
        from .controller_resources import load_resource_inputs, resource_preflight
        study, base = load_resource_inputs(Path(args.config))
        result = resource_preflight(study, base, Path(args.target))
        print(json.dumps(result, sort_keys=True))
        if not result["admitted"]:
            raise SystemExit(2)
        return
    if args.experiment == "study-train":
        from .controller_evidence import open_study_run
        from .controller_resources import load_resource_inputs, resource_preflight
        from .controller_training_stage import run_training_chunk
        open_options = {"create": args.create}
        if args.create:
            study, planned_base = load_resource_inputs(Path(args.config))
            preflight = resource_preflight(study, planned_base, Path(args.out))
            if not preflight["admitted"]:
                print(json.dumps(preflight, sort_keys=True))
                raise SystemExit(2)
            open_options["min_free_bytes"] = preflight["budget"]["required_free_bytes"]
        ledger, base, evidence = open_study_run(
            Path(args.out), Path(args.config), **open_options)
        result = run_training_chunk(
            ledger, base, evidence, pilot_limit=args.pilot_limit, model_limit=args.model_limit)
        print(json.dumps(result, sort_keys=True))
        return
    if args.experiment == "study-validate":
        from .controller_evidence import open_study_run
        from .controller_trials import validate_trials
        ledger, base, _ = open_study_run(Path(args.out), Path(args.config))
        result = validate_trials(ledger, base, args.limit)
        print(json.dumps(result, sort_keys=True))
        return
    if args.experiment == "study-select":
        from .controller_evidence import open_study_run
        ledger, _, _ = open_study_run(Path(args.out), Path(args.config))
        result = ledger.freeze_selection()
        print(json.dumps(result, sort_keys=True))
        return
    if args.experiment == "study-evaluate":
        from .controller_evidence import open_study_run
        from .controller_trials import evaluate_trials
        ledger, base, _ = open_study_run(Path(args.out), Path(args.config))
        result = evaluate_trials(ledger, base, args.limit)
        print(json.dumps(result, sort_keys=True))
        return
    from .artifacts import load_config
    if args.experiment == "latency":
        from .latency import run
    else:
        from .experiments import runners
        run = getattr(runners, args.experiment)
    config = load_config(args.config)
    if args.experiment != "latency":
        from .config import validate_simulation
        validate_simulation(config)
    if args.experiment == "latency":
        run(config, args.out, resume=args.resume)
    else:
        run(config, args.out)


if __name__ == "__main__":
    main()
