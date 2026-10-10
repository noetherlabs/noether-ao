"""Config loading and auditable, non-overwriting experiment artifacts."""
from __future__ import annotations

import hashlib
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

import numpy as np
import psutil
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Read an experiment mapping; reject non-mappings."""
    value = yaml.safe_load(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("Experiment config must be a mapping")
    return value


def write_json(path: Path, value: Any) -> None:
    """Write strict JSON atomically, rejecting non-finite numerical values."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def fingerprint() -> dict[str, Any]:
    """Capture hardware, runtime and exact installed package versions."""
    import jax
    cpu = platform.processor()
    if sys.platform == "darwin":
        cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    data = {
        "platform": platform.platform(), "cpu": cpu,
        "logical_cores": os.cpu_count(), "physical_cores": psutil.cpu_count(logical=False),
        "memory_bytes": psutil.virtual_memory().total,
        "python": sys.version, "devices": [str(d) for d in jax.devices()],
        "jax_backend": jax.default_backend(), "x64": bool(jax.config.jax_enable_x64),
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "packages": dict(sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions())),
    }
    data["id"] = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]
    return data


def start_run(config: dict[str, Any], destination: str | Path) -> Path:
    """Create a fresh run with config, fingerprint, source hash and commit."""
    path = Path(destination)
    path.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, text=True)
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted((root / "src").rglob("*.py"))}
    write_json(path / "config.json", config)
    write_json(path / "machine.json", fingerprint())
    write_json(path / "provenance.json", {
        "git_commit": sha, "tracked_dirty": bool(dirty), "source_sha256": hashes,
        "utc": datetime.now(timezone.utc).isoformat(), "argv": sys.argv,
        "lock_sha256": hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest(),
    })
    return path


def mean_se(values: list[float]) -> tuple[float, float]:
    """Return across-seed mean and standard error (not across-frame error)."""
    a = np.asarray(values, dtype=float)
    return float(a.mean()), float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else 0.0


def write_table(path: Path, rows: list[dict]) -> None:
    """Write CSV and Markdown from the same numerical rows."""
    if not rows:
        return
    keys = list(rows[0])
    with path.with_suffix('.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys, lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    def cell(v):
        return f'{v:.6g}' if isinstance(v,float) else str(v)
    path.with_suffix('.md').write_text('| ' + ' | '.join(keys) + ' |\n| ' +
        ' | '.join('---' for _ in keys) + ' |\n' + '\n'.join(
        '| ' + ' | '.join(cell(row[k]) for k in keys) + ' |' for row in rows) + '\n')
