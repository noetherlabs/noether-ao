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
    args = parser.parse_args()
    from .artifacts import load_config
    if args.experiment == "latency":
        from .latency import run
    else:
        from .experiments import runners
        run = runners[args.experiment]
    run(load_config(args.config), args.out)


if __name__ == "__main__":
    main()
