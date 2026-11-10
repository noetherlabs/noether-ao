"""Plot all conditions for two controller contrasts from the frozen report."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_bytes())
    if report["status"] != "complete_descriptive_report" or report["primary_pairwise_family"] != 120:
        raise ValueError("Expected the complete frozen 120-comparison study")
    if args.out.exists():
        raise FileExistsError("Refusing to replace a figure")
    fig, axes = plt.subplots(1, 2, figsize=(10, 6.8), sharey=True, layout="constrained")
    for ax, target, title in zip(axes, ("predictive", "learned"),
                                 ("Predictor vs integrator", "Learned mixture vs integrator"), strict=True):
        rows = [r for r in report["comparisons"] if r["left"] == "integrator" and r["right"] == target]
        if len(rows) != 12 or any(r["post_burn"]["status"] != "complete" for r in rows):
            raise ValueError("Expected every declared condition")
        intervals = [r["post_burn"]["interval"] for r in rows]
        # The report stores integrator minus target; reverse it for an error-cost axis.
        means = np.array([-r["mean_nm"] for r in intervals])
        low = np.array([-r["high_nm"] for r in intervals])
        high = np.array([-r["low_nm"] for r in intervals])
        for i, (mean, lo, hi) in enumerate(zip(means, low, high, strict=True)):
            color = "#075985" if hi < 0 else "#64748b"
            ax.errorbar(mean, i, xerr=[[mean-lo], [hi-mean]], fmt="o", color=color,
                        capsize=3, markersize=5, linewidth=1.5)
        labels = [f"{r['photons']:,.0f} photons · {r['wind_scale']:g}× wind · "
                  + {"none": "clean", "optical_gain": "gain drift", "vibration": "vibration"}[r["fault"]]
                  for r in rows]
        ax.set_yticks(range(12), labels, fontsize=9)
        ax.axvline(0, color="#111827", linewidth=.9)
        for y in (2.5, 5.5, 8.5):
            ax.axhline(y, color="#e2e8f0", linewidth=.8)
        ax.set_title(title, fontsize=12, loc="left", pad=12)
        ax.set_xlabel("Target − integrator residual WFE (nm)\n← lower optical error", fontsize=10)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", alpha=.15)
        ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    fig.suptitle("Prediction Boundary · held-out simulation\n"
                 "10 paired seeds per condition · intervals adjusted over all 120 primary comparisons",
                 fontsize=13)
    fig.savefig(args.out, dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
