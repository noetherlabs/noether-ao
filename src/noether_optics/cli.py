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
    args = parser.parse_args()
    if args.experiment == "study-train":
        from .controller_evidence import open_study_run
        from .controller_training_stage import run_training_chunk
        ledger, base, evidence = open_study_run(
            Path(args.out), Path(args.config), create=args.create)
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
