"""Plot every paired realization after verifying the retained numerical evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from verify import verify


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.root.resolve()
    verify(root)
    report = json.loads((root / "assessment.json").read_text())
    trials = report["individual_trials"]
    colors = {"continue_P": "#236397", "switch_P_to_I": "#c45b27"}
    labels = {"continue_P": "Continue predictor", "switch_P_to_I": "Switch to integrator"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none",
                         "svg.hashsalt": "noether-scheduled-fallback-v1"})
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.3), layout="constrained")
    fig.suptitle("Scheduled fallback under optical-gain drift\n"
                 "Six paired atmospheric realizations · 24 simulated trials", fontsize=15)
    for column, (condition, title) in enumerate([
        ("nominal", "Nominal sensor gain"),
        ("optical_gain_drift", "Sensor gain ramp: 1 → 0.35"),
    ]):
        ax = axes[0, column]
        for action in colors:
            group = [row for row in trials if row["case"]["condition"] == condition
                     and row["case"]["candidate_id"] == action]
            group.sort(key=lambda row: row["case"]["root_seed"])
            values = np.asarray([row["physical"]["variance"] for row in group])
            times = np.arange(values.shape[1])
            for value in values:
                ax.plot(times[32:], value[32:], color=colors[action], alpha=.20, lw=.8)
            ax.plot(times[32:], values.mean(axis=0)[32:], color=colors[action],
                    lw=2, label=labels[action])
        ax.axvspan(82, 128, color="#c8c8c8", alpha=.16, zorder=-1)
        ax.axvline(82, color="#777777", ls="--", lw=1)
        if condition == "optical_gain_drift":
            ax.axvline(64, color="#777777", ls=":", lw=1)
            ax.text(64.8, 26, "Drift begins", fontsize=8, va="top")
        ax.set(title=title, xlabel="Simulation time (ms)",
               ylabel="Piston-removed phase variance (rad²)",
               xlim=(32, 128), ylim=(.055, 40), yscale="log")
        ax.grid(axis="y", color="#e5e5e5", lw=.7)
        ax.legend(frameon=False, loc="upper left", fontsize=8.5)

        ax = axes[1, column]
        summary = report["primary_regimes"][condition]
        values = np.asarray(summary["values"])
        x = np.arange(1, 7)
        ax.axhline(0, color="#777777", ls="--", lw=1)
        ax.vlines(x, 0, values, color="#c6c6c6", lw=1)
        ax.scatter(x, values, color="#303d50", s=32, zorder=3)
        low, high = summary["interval"]
        mean = summary["mean"]
        ax.errorbar(7.5, mean, yerr=[[mean-low], [high-mean]], fmt="D",
                    color="#675083", capsize=5, ms=6, lw=1.6)
        ax.set_xticks([*x, 7.5], [f"R{i}" for i in x] + ["Mean\n+ interval"])
        ax.set(xlim=(.4, 8.2), ylabel="Normalized integrated-variance change",
               title="Switch − continue; lower favors switching")
        ax.grid(axis="y", color="#e5e5e5", lw=.7)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        if condition == "nominal":
            ax.set_ylim(-.0038, .0010)
        else:
            ax.set_ylim(-3.7, .6)
    fig.supxlabel("Faint traces: individual realizations; bold: arithmetic mean. "
                  "Shading: fixed scoring window, 82–127 ms.\n"
                  "Intervals: predeclared Student-t, df = 5, Bonferroni correction for two conditions. "
                  "Both include zero.", fontsize=9)
    fig.savefig(root / "comparison.png", dpi=180)
    fig.savefig(root / "comparison.svg", metadata={"Date": None})
    svg = root / "comparison.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


if __name__ == "__main__":
    main()
