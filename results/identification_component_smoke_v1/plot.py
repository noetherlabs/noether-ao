"""Plot retained development measurements after numerical reconstruction."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from verify import verify


def plot(directory):
    verify(directory)
    evidence = json.loads((directory / 'evidence.json').read_text())
    colors = {'nominal': '#657489', 'ols': '#bf7333', 'iv': '#267d78'}
    labels = {'nominal': 'Nominal', 'ols': 'OLS', 'iv': 'IV'}
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.dpi': 180})
    figure, axes = plt.subplots(1, 3, figsize=(12.8, 4.3), layout='constrained')
    for offset, method in zip((-.22, 0, .22), colors):
        rows = [row for row in evidence['models'] if row['model']['method'] == method]
        axes[0].bar(np.arange(2)+offset, [row['response_relative_error'] for row in rows],
                    width=.2, color=colors[method], label=labels[method])
    axes[0].set(xticks=[0, 1], xticklabels=['White', 'Correlated'],
                ylabel='Relative Frobenius error', xlabel='Added sensor noise',
                title='A  Calibration response')
    axes[0].legend(frameon=False, fontsize=9)
    conditions = [(0., False), (.8, False), (.8, True)]
    for index, (rho, drift) in enumerate(conditions):
        rows = [row for row in evidence['forecasts'] if row['rho'] == rho and row['drift'] == drift]
        window = 'settled_drift' if drift else 'post_burn'
        for offset, method in zip((-.22, 0, .22), colors):
            values = [row['predictions'][method]['mse_by_window'][window] for row in rows]
            axes[1].scatter(index+offset+np.array([-.035, .035]), values,
                            color=colors[method], s=24, zorder=3)
            axes[1].plot([index+offset-.075, index+offset+.075], [np.mean(values)]*2,
                         color=colors[method], linewidth=2)
        costs = [row['paired_variance_change_rad2'] for row in rows]
        axes[2].scatter(index+np.array([-.04, .04]), costs, color='#3b4655', s=24, zorder=3)
        axes[2].plot([index-.10, index+.10], [np.mean(costs)]*2, color='#3b4655', linewidth=2)
    names = ['White', 'Correlated', 'Correlated\n+ gain drift']
    axes[1].set(xticks=range(3), xticklabels=names, ylim=(0, None),
                ylabel='Noisy residual forecast MSE (rad²)', title='B  One-step sensor prediction')
    axes[2].set(xticks=range(3), xticklabels=names, ylim=(0, None),
                ylabel='Added residual phase variance (rad²)', title='C  Cost of probing')
    for axis in axes:
        axis.grid(axis='y', alpha=.2)
        axis.set_axisbelow(True)
    figure.suptitle('Closed-loop identification · development measurements', fontsize=14, fontweight='bold')
    figure.supxlabel('Two forecast episodes per condition · points: episodes; bars in B/C: arithmetic means · no confidence intervals', fontsize=9)
    figure.savefig(directory / 'measurements.png')
    plt.close(figure)


if __name__ == '__main__':
    plot(Path(__file__).resolve().parent)
