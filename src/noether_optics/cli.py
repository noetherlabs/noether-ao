"""Config-driven experiment command line."""
import argparse
import os


def main() -> None:
    """Dispatch one experiment; enable double precision before importing JAX."""
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    os.environ.setdefault("MPLBACKEND", "Agg")
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=["latency", "validate", "faults", "fidelity", "mixed"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--resume", action="store_true", help="Resume only unfinished latency points on an identical runtime")
    args = parser.parse_args()
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
    if args.resume and args.experiment != "latency":
        parser.error("--resume applies to latency only")
    if args.experiment == "latency":
        run(config, args.out, resume=args.resume)
    else:
        run(config, args.out)


if __name__ == "__main__":
    main()
