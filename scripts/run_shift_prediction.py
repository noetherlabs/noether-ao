"""Run the shift-prediction benchmark (docs/shift_prediction.md) and write its report."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from noether_optics import shift_prediction as sp

NAMES = ["no change", "global linear", "full-sensor linear", "nonlinear (random features)", sp.UPPER]
LABELS = {sp.UPPER: "more-data global linear (400k simulated frames)"}
BANDS = {"conformal": "empirical training quantile"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ms(values) -> tuple[float, float]:
    a = np.asarray(values, float)
    return float(a.mean()), float(a.std(ddof=1))


def _figures(bench: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), dpi=150, sharey=True)
    styles = [("k", ":"), ("C1", "-"), ("C2", "-"), ("C3", "--"), ("0.6", "-")]
    for ax, (rname, r) in zip(axes, bench.items()):
        for name, (color, ls) in zip(NAMES, styles):
            ax.plot(sp.HORIZONS, [_ms(r["skill"][name][str(h)])[0] for h in sp.HORIZONS],
                    color=color, ls=ls, label=LABELS.get(name, name))
        ax.set_xscale("log")
        ax.set_ylim(-0.2, 1.02)
        ax.axhline(0, color="0.7", lw=0.8)
        ax.set_title(rname, fontsize=9)
        ax.set_xlabel("frames ahead (ms at 1 kHz)")
    axes[0].set_ylabel("variance explained")
    axes[-1].legend(fontsize=7, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "skill.png")
    plt.close(fig)

    imp = bench["one layer, 10 m/s"]["importance_h20_seed0"]
    fig2, ax2 = plt.subplots(figsize=(3.8, 3.4), dpi=150)
    im = ax2.imshow(np.array(imp["subapertures"]), cmap="Greys", origin="lower")
    ax2.set_xlabel("across the wind")
    ax2.set_ylabel("along the wind (enters at top)")
    ax2.set_title("Weight on each subaperture, 20 frames ahead", fontsize=8)
    fig2.colorbar(im, ax=ax2, shrink=0.8)
    fig2.tight_layout()
    fig2.savefig(out / "importance.png")
    plt.close(fig2)


def _report(bench: dict, posthoc: dict, env: dict) -> str:
    L = ["# Shift prediction and range forecasting: exploratory results (v1)", "",
         "Simulation only. Five evaluation seeds per regime; mean ± sample standard deviation. "
         "Skill is the fraction of target variance explained (1 is perfect, 0 is the mean). "
         "The public history does not establish independent preregistration. "
         "Methods and interpretation: [study documentation](../../docs/shift_prediction.md).", "",
         "![Skill against horizon](skill.png)", ""]
    claims = {"E1": [], "E2": [], "E3": [], "E4": []}
    shuffle = []
    for rname, r in bench.items():
        sk = r["skill"]
        L += [f"## {rname}", "", "| Frames ahead | " + " | ".join(LABELS.get(n, n) for n in NAMES) + " |", "|" + " --- |" * (len(NAMES) + 1)]
        for h in sp.HORIZONS:
            cells = [f"{_ms(sk[n][str(h)])[0]:+.3f} ± {_ms(sk[n][str(h)])[1]:.3f}" for n in NAMES]
            L.append(f"| {h} | " + " | ".join(cells) + " |")
        L += ["", "Paired differences against global linear (pointwise 95% t-interval over five seeds; "
              "no adjustment for multiple comparisons):", ""]
        for h in (10, 20):
            for n, tag in (("full-sensor linear", "E3"), ("nonlinear (random features)", "E2")):
                m, lo, hi = sp.paired(sk[n][str(h)], sk["global linear"][str(h)])
                L.append(f"- {n}, {h} frames: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
                claims[tag].append(lo)
        gap = _ms(np.subtract(sk[sp.UPPER]["10"], sk["global linear"]["10"]))[0]
        L.append(f"- More-data reference minus finite-data global linear, 10 frames: {gap:+.3f}")
        claims["E1"].append(max(_ms(sk[n]["1000"])[0] for n in NAMES[:4]))
        shuffle.append((rname, _ms(sk[sp.SHUFFLED]["10"])[0]))
        L += ["", "Range forecast for the next 10,000 frames, drifting seeing (nominal marginal coverage 95.4%). "
              "Coverage counts overlapping forecast-window/time pairs, not simultaneous coverage of a complete path. "
              "Uncertainty below is sample standard deviation across seeds:", "",
              "| Band | Coverage | Half-width (arcsec) |", "| --- | --- | --- |"]
        for band in ("naive", "twin", "conformal"):
            cov = _ms([x[band]["coverage"] for x in r["range"]])
            wid = _ms([x[band]["half_width_arcsec"] for x in r["range"]])
            L.append(f"| {BANDS.get(band, band)} | {100 * cov[0]:.1f}% ± {100 * cov[1]:.1f} | {wid[0]:.2f} ± {wid[1]:.2f} |")
            if band == "conformal":
                claims["E4"].append(cov[0])
        speeds = ", ".join(f"{w['speed_m_s']:.1f}" if w["speed_m_s"] else "n/a" for w in r["wind"])
        true = ", ".join(str(v) for v in r["true_speeds_m_s"])
        L += ["", f"Single correlation-peak speed: {speeds} m/s (simulated layer speeds: {true} m/s). "
              "This estimator does not resolve separate layer velocities.", ""]

    L += ["## Execution checks", ""]
    for rname, v in shuffle:
        L.append(f"- Shuffled targets, {rname}: skill {v:+.3f} (limit 0.02): {'pass' if v <= 0.02 else 'FAIL'}")
    L += ["- Training/test separation is checked at each horizon during acquisition; "
          "this report does not independently reconstruct the simulated frame streams.", "",
          "## Exploratory comparisons", "",
          f"- E1, tested model means at 1,000 frames {'all stay' if max(claims['E1']) <= 0.05 else 'do not all stay'} "
          f"below 0.05 skill (best {max(claims['E1']):+.3f}). This is not a universal prediction limit.",
          f"- E2, random-feature comparisons with a positive paired interval: {sum(lo > 0 for lo in claims['E2'])}/{len(claims['E2'])}. "
          "Lack of detected superiority does not establish equivalence or rule out other nonlinear models.",
          f"- E3, full-sensor comparisons with a positive paired interval: {sum(lo > 0 for lo in claims['E3'])}/{len(claims['E3'])}. "
          "This includes extra sensing information and different histories; see the post-hoc check.",
          f"- E4, empirical-band mean coverage within 1 point of 95.4% in every regime: {'yes' if all(abs(c - 0.954) <= 0.01 for c in claims['E4']) else 'no'} "
          f"({', '.join(f'{100 * c:.1f}%' for c in claims['E4'])})", "",
          "## Post-hoc shared-noise check", "",
          "In v1 the global tilt and the subapertures carry independent noise, so the full-sensor model sees "
          "independently noisy spatial information. Here the global input is the mean of the measured subapertures and the "
          "spatial model uses subapertures only. This removes the independent global-noise channel, but the "
          "subaperture average is not the circular-pupil Z-tilt target operator.", "",
          "| Regime | " + " | ".join(f"Gain at {h} frames" for h in posthoc[next(iter(posthoc))]) + " |",
          "|" + " --- |" * (1 + len(posthoc[next(iter(posthoc))]))]
    for rname, rows in posthoc.items():
        cells = []
        for h, v in rows.items():
            m, lo, hi = sp.paired(v["subapertures only"], v["global from subapertures"])
            cells.append(f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        L.append(f"| {rname} | " + " | ".join(cells) + " |")
    L += ["", "These comparisons do not isolate advection prediction from spatial estimation, "
          "measurement geometry or the different history lengths.", "",
          "![Subaperture weights](importance.png)", "",
          "## Limitations", "",
          "- Frozen flow with at most two layers; no vibration, tracking drift, thermal effects or "
          "correlation-tracker nonlinearity.",
          "- The square 4 x 4 grid samples phase slightly outside the circular pupil in its corners.",
          "- The empirical training quantile has no finite-sample conformal guarantee for these "
          "dependent, overlapping windows under drift. The legacy JSON key is `conformal`.",
          "- Global models use 48 lags spanning 1,000 frames; spatial models use 20 recent lags. "
          "Inner tuning folds overlap in history and are not embargoed. Random-feature scaling uses "
          "all outer-training rows before inner selection. Outer test histories remain separated.",
          "- The more-data global-linear reference is not a mathematical upper bound; "
          "its historical JSON key is retained for compatibility.",
          "- None of this is a statement about an instrument until it is repeated on recorded frames.", "",
          f"Environment: Python {env['python']}, NumPy {env['numpy']}, Matplotlib {env['matplotlib']}. "
          "File digests are in [manifest.json](manifest.json).", ""]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="fresh output directory; existing evidence is never overwritten")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        ap.error("--out must name a new directory; existing evidence is preserved")
    out.mkdir(parents=True, exist_ok=False)
    bench = sp.run_benchmark()
    posthoc = sp.run_posthoc()
    env = {"python": platform.python_version(), "numpy": np.__version__, "matplotlib": matplotlib.__version__}
    (out / "results.json").write_text(json.dumps(bench, indent=1))
    (out / "posthoc.json").write_text(json.dumps(posthoc, indent=1))
    _figures(bench, out)
    (out / "README.md").write_text(_report(bench, posthoc, env))
    sources = [Path(sp.__file__), Path(__file__)]
    manifest = {
        "protocol": "v1",
        "environment": env,
        "sources": {p.name: _sha256(p) for p in sources},
        "outputs": {p.name: _sha256(p) for p in sorted(out.iterdir()) if p.name != "manifest.json"},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
