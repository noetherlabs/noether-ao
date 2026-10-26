"""Synchronized latency distributions; no timing in correctness tests or CI."""
from __future__ import annotations

import gc
import os
from pathlib import Path
import subprocess
import sys
import time

import jax
import numpy as np

from .artifacts import start_run, write_json
from .optics import propagation, relative_l2


def statistics(seconds: np.ndarray) -> dict[str, float]:
    """Summarize all samples using NumPy linear interpolated quantiles."""
    x = np.asarray(seconds)
    if x.ndim != 1 or not len(x) or not np.all(np.isfinite(x)) or np.any(x < 0):
        raise ValueError("Expected a nonempty finite nonnegative timing vector")
    return {**{f"p{q:g}_ms": float(np.percentile(x, q) * 1000) for q in (50, 90, 99, 99.9)},
            "max_ms": float(x.max() * 1000),
            "miss_1ms": float(np.mean(x > .001)), "miss_0.5ms": float(np.mean(x > .0005))}


def measure(fn, opd: np.ndarray, wavelengths: np.ndarray, device,
            repeats: int, warmup: int, host_path: bool) -> tuple[np.ndarray, float]:
    """Time each synchronized call, optionally including fresh host/device copies."""
    x, wl = jax.device_put(opd, device), jax.device_put(wavelengths, device)
    jax.block_until_ready((x, wl))
    for _ in range(warmup):
        jax.block_until_ready(fn(x, wl))
    timings = np.empty(repeats, dtype=np.float64)
    cpu_start, wall_start = time.process_time(), time.perf_counter()
    for i in range(repeats):
        start = time.perf_counter_ns()
        if host_path:
            # may_alias=False and output copy make transfer semantics explicit on CPU.
            x = jax.device_put(opd, device, may_alias=False)
            wl = jax.device_put(wavelengths, device, may_alias=False)
        result = fn(x, wl)
        jax.block_until_ready(result)
        if host_path:
            np.array(jax.device_get(result), copy=True)
        timings[i] = (time.perf_counter_ns() - start) * 1e-9
    ratio = (time.process_time() - cpu_start) / (time.perf_counter() - wall_start)
    return timings, ratio


def run_mode(config: dict, out: Path, mode: str) -> None:
    """Run one backend/affinity mode in its own process, before importing XLA."""
    device = jax.devices("gpu" if mode == "gpu" else "cpu")[0]
    rows = []
    for n in config["sizes"]:
        for nw in config["wavelength_counts"]:
            for dtype in config["precisions"]:
                key = f"{mode}_n{n}_w{nw}_{dtype}"
                rng = np.random.default_rng(config["seed"])
                opd = (rng.normal(size=(n, n)) * config["opd_rms_m"]).astype(dtype)
                wavelengths = np.linspace(*config["wavelength_range_m"], nw).astype(dtype)
                x, wl = jax.device_put(opd, device), jax.device_put(wavelengths, device)
                functions, compiles, results = {}, {}, {}
                for engine in ("dlux", "jax"):
                    start = time.perf_counter()
                    functions[engine] = jax.jit(propagation(n, dtype, engine)).lower(x, wl).compile()
                    compiles[engine] = time.perf_counter() - start
                    results[engine] = np.asarray(jax.block_until_ready(functions[engine](x, wl)))
                error = relative_l2(results["dlux"], results["jax"])
                expected_dtype = "complex64" if dtype == "float32" else "complex128"
                passed = error <= config["tolerance"][dtype] and all(str(v.dtype) == expected_dtype for v in results.values())
                if not passed:
                    write_json(out / f"{key}_failure.json", {"relative_l2": error, "dtype": {k:str(v.dtype) for k,v in results.items()}, "status": "accuracy_fail"})
                    raise ValueError(f"Accuracy/dtype gate failed at {key}")
                raw = {}
                for engine in ("dlux", "jax"):
                    for path in ("device", "host_roundtrip"):
                        times, ratio = measure(functions[engine], opd, wavelengths, device,
                                               config["repeats"], config["warmup"], path != "device")
                        raw[f"{engine}_{path}_seconds"] = times
                        row = {"mode": mode, "n": n, "wavelengths": nw, "precision": dtype,
                               "engine": engine, "path": path, "samples": len(times),
                               "compile_s": compiles[engine], "relative_l2": error,
                               "cpu_wall_ratio": ratio, "raw": f"{key}.npz", **statistics(times)}
                        rows.append(row)
                np.savez_compressed(out / f"{key}.npz", **raw, opd=opd, wavelengths=wavelengths)
                write_json(out / f"{mode}_summary.json", rows)
                print(f"{key}: passed; saved all timing samples", flush=True)
                del functions, results
                jax.clear_caches()
                gc.collect()


def run(config: dict, destination: str) -> None:
    """Run all supported modes, with explicit skip records for unsupported modes."""
    if config["repeats"] < 10000:
        raise ValueError("Published latency runs require at least 10000 evaluations per point")
    if os.environ.get("CI"):
        raise RuntimeError("Timing runs are forbidden in CI")
    out = start_run(config, destination)
    skips = []
    for mode in config["modes"]:
        if mode == "cpu_one" and not hasattr(os, "sched_setaffinity"):
            skips.append({"mode": mode, "reason": "OS has no enforceable sched_setaffinity; thread flags are not affinity"})
            continue
        if mode == "gpu" and not any(d.platform == "gpu" for d in jax.devices()):
            skips.append({"mode": mode, "reason": "No GPU device exposed by installed JAX backend"})
            continue
        bootstrap = "import os,json; from pathlib import Path; "
        if mode == "cpu_one":
            bootstrap += "os.sched_setaffinity(0,{min(os.sched_getaffinity(0))}); "
        bootstrap += "from noether_optics.latency import run_mode; run_mode(json.loads(Path(__import__('sys').argv[1]).read_text()),Path(__import__('sys').argv[2]),__import__('sys').argv[3])"
        subprocess.run([sys.executable, "-c", bootstrap, str(out / "config.json"), str(out), mode], check=True)
    write_json(out / "skips.json", skips)
    render(out)


def render(out: Path) -> None:
    """Regenerate tables and plot exclusively from saved timing summaries."""
    import json
    import matplotlib.pyplot as plt
    rows = [r for p in sorted(out.glob("*_summary.json")) for r in json.loads(p.read_text())]
    write_json(out / "summary.json", rows)
    columns = ["mode", "n", "wavelengths", "precision", "engine", "path", "p50_ms", "p90_ms", "p99_ms", "p99.9_ms", "max_ms", "miss_1ms", "miss_0.5ms", "compile_s"]
    lines = ["# Measured latency", "", "All units ms except compile_s and miss fractions. Host roundtrip includes forced copies.", "", "| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        lines.append("| " + " | ".join(f"{row[c]:.6g}" if isinstance(row[c], float) else str(row[c]) for c in columns) + " |")
    (out / "table.md").write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, path in zip(axes, ("device", "host_roundtrip")):
        for dtype in ("float32", "float64"):
            for nw in (1, 3, 5):
                selected = [r for r in rows if r["engine"] == "dlux" and r["path"] == path and r["mode"] == "cpu_all" and r["precision"] == dtype and r["wavelengths"] == nw]
                ax.plot([r["n"] for r in selected], [r["p99_ms"] for r in selected], ".-", label=f"{dtype}, {nw} wavelengths")
        ax.axhline(1, color="black", linestyle=":")
        ax.set(xlabel="Pupil samples per side", ylabel="p99 latency (ms)", yscale="log", title=path)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "latency.png", dpi=160)
    plt.close(fig)
