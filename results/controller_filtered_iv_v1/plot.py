"""Render retained development measurements; perform no acquisition or fitting."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

DIRECTORY = Path(__file__).resolve().parent


def plot():
    evidence = json.loads((DIRECTORY/'evidence.json').read_text())
    methods = ('nominal', 'ols', 'iv', 'controller_filtered_iv')
    labels = ('Nominal', 'OLS', 'Raw-probe IV', 'Filtered-probe IV')
    colors = ('#455a64', '#978c7c', '#336b87', '#b75236')
    conditions = ((0., False, 'post_burn'), (.8, False, 'post_burn'),
                  (.8, True, 'settled_drift'))
    response = {(row['rho'], row['model']['method']): row['response_relative_error']
                for row in evidence['baseline_models']}
    response.update({(row['rho'], 'controller_filtered_iv'): row['response_relative_error']
                     for row in evidence['models']})
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5))
    for index, (method, label, color) in enumerate(zip(methods, labels, colors)):
        offset = (index-1.5)*.13
        axes[0].scatter(np.arange(2)+offset,
                        [response[rho, method] for rho in (0., .8)],
                        label=label, color=color, s=42, zorder=3)
        values = []
        for rho, drift, window in conditions:
            rows = [row for row in evidence['forecasts']
                    if row['rho'] == rho and row['drift'] == drift]
            values.append(np.mean([row['predictions'][method]['mse_by_window'][window]
                                   for row in rows]))
        axes[1].scatter(np.arange(3)+offset, values, color=color, s=42, zorder=3)
    axes[0].set(title='A  Response estimation', ylabel='Relative Frobenius error',
                xticks=[0, 1], xticklabels=['White noise', 'Correlated noise'],
                ylim=(-.05, 1.15), xlim=(-.4, 1.4))
    axes[1].set(title='B  Sensor forecasting', ylabel='Mean squared error (rad²; log scale)',
                xticks=[0, 1, 2], xticklabels=['White noise', 'Correlated noise', 'Settled drift'],
                yscale='log', ylim=(.0008, .025), xlim=(-.4, 2.4))
    for axis in axes:
        axis.grid(axis='y', alpha=.2)
        axis.set_axisbelow(True)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center',
               ncol=4, frameon=False, bbox_to_anchor=(.5, 1.02))
    fig.text(.5, .015, 'Development recordings · 2 training and 2 forecast episodes per condition · no new trajectories',
             ha='center', fontsize=9, color='#555555')
    fig.tight_layout(rect=(0, .06, 1, .92))
    fig.savefig(DIRECTORY/'comparison.png', dpi=180, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    plot()
