"""Plot the complete receiver-calibration development comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from noether_optics.receiver_calibration import summarize_root_utilities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    data = json.loads(args.summary.read_bytes())
    if data['schema'] != 'noether-receiver-calibration-development-result/1':
        raise ValueError('Expected the receiver-calibration development result')
    records = [{key: cell[key] for key in ('root_id', 'condition_id', 'charged_mean_delta')}
               for cell in data['cells']]
    stats = summarize_root_utilities(records, root_ids=data['roots'],
                                    condition_ids=data['conditions'], alpha=.05)
    if stats != data['statistics'] or stats['status'] != 'complete':
        raise ValueError('Published statistics do not reproduce from the complete cell grid')
    if len(records) != 32 or any(cell['fit_status'] != 'withheld' or cell['service_gain'] != 0
                                 for cell in data['cells']):
        raise ValueError('This figure describes the all-withheld development result')
    labels = {'nominal-low_flux': 'Nominal response · 1,000 photons',
              'nominal-high_flux': 'Nominal response · 1,000,000 photons',
              'coupled-low_flux': 'Coupled mismatch · 1,000 photons',
              'coupled-high_flux': 'Coupled mismatch · 1,000,000 photons'}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'svg.fonttype': 'none', 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.spines.left': False})
    figure, axis = plt.subplots(figsize=(8.6, 4.5))
    for y, condition in enumerate(data['conditions']):
        values = [100*cell['charged_mean_delta'] for cell in data['cells']
                  if cell['condition_id'] == condition]
        offsets = [(i-3.5)*.028 for i in range(8)]
        axis.scatter(values, [y+offset for offset in offsets], color='#aeb5bc',
                     s=20, zorder=2, label='Independent root' if y == 0 else None)
        row = stats['condition_statistics'][condition]['statistics']
        mean = 100*row['mean']
        low, high = [100*value for value in row['student_t_interval']]
        axis.errorbar(mean, y, xerr=[[mean-low], [high-mean]], fmt='o',
                      color='#253c50', markersize=5, capsize=4, linewidth=1.6,
                      label='Mean and simultaneous t intervals' if y == 0 else None)
    axis.axvline(0, color='#76818a', linewidth=1, linestyle='--')
    axis.set_yticks(range(4), [labels[value] for value in data['conditions']])
    axis.invert_yaxis()
    axis.tick_params(axis='y', length=0, pad=12)
    axis.set_xlim(right=.025)
    axis.set_xlabel('Coupling difference (percentage points; 896 charged frames)', labelpad=12)
    axis.grid(axis='x', color='#eceff1', linewidth=.7)
    axis.set_axisbelow(True)
    axis.legend(loc='upper left', bbox_to_anchor=(0, -.24), frameon=False, fontsize=9)
    figure.suptitle('Probing cost without an admitted correction', x=.03, ha='left',
                    fontsize=15, fontweight='medium', y=.96)
    figure.text(.03, .88, '32 of 32 updates withheld. Service uses the unchanged calibration.',
                color='#515b64', fontsize=10)
    figure.text(.03, .035, 'Synthetic phase-only plant · Eight root blocks · Bonferroni-adjusted intervals across four conditions',
                color='#515b64', fontsize=8.5)
    figure.subplots_adjust(left=.42, right=.97, top=.78, bottom=.31)
    args.out.mkdir(parents=True, exist_ok=False)
    for extension in ('png', 'svg'):
        figure.savefig(args.out / f'coupling-cost.{extension}', dpi=200, facecolor='white')
    svg = args.out / 'coupling-cost.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    plt.close(figure)


if __name__ == '__main__':
    main()
