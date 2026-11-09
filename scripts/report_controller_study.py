"""Reconstruct a completed controller study without acquiring or selecting trials."""
from __future__ import annotations

import argparse
from pathlib import Path

from noether_optics.controller_evaluation import build_evaluation_report
from noether_optics.controller_evidence import _source, _runtime
from noether_optics.controller_selection import _bytes, _digest, _publish

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run, destination = args.run.resolve(), args.out.resolve()
    if destination.is_relative_to(run):
        raise ValueError("Write the derived report outside the retained acquisition run")
    if destination.exists():
        raise FileExistsError("Refusing to replace an existing report")
    source = _source(ROOT)
    runtime = _runtime()
    report = build_evaluation_report(run)
    if _source(ROOT) != source:
        raise ValueError("Analysis source changed during reconstruction")
    report["analysis"] = {"source": source, "source_sha256": _digest(_bytes(source)),
                          "runtime": runtime, "runtime_sha256": _digest(_bytes(runtime))}
    _publish(destination, report)
    print(_digest(_bytes(report)))


if __name__ == "__main__":
    main()
