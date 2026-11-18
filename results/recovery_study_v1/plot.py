"""Render the fixed primary contrasts from a complete raw-reconstructed report.

This consumer neither recalculates intervals nor selects conditions. It accepts
only all 24 declared contrasts from the complete 64-root evidence audit.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math


def read_complete_report(path):
    report = json.loads(path.read_bytes())
    if report['status'] != 'complete_raw_reconstructed_report':
        raise ValueError('Complete raw reconstruction required')
    statistics = report['statistics']
    spec = statistics['protocol']
    if (spec['seeds'] != list(range(2026111101, 2026111165))
            or spec['expected_terminal_records'] != 2304
            or statistics['family_size'] != 24
            or len(report['root_audits']) != 64):
        raise ValueError('Unexpected recovery cohort')
    rows = statistics['contrasts']
    keys = [(flux, wind, fault, arm) for flux in (1000., 1000000.)
            for wind in (.5, 2.) for fault in ('none', 'optical_gain', 'vibration')
            for arm in ('issued_state', 'command_matched')]
    keyed = {(r['photons'], r['wind_scale'], r['fault'], r['arm']): r for r in rows}
    if len(rows) != 24 or set(keyed) != set(keys):
        raise ValueError('All declared contrasts required, without duplicates')
    for row in rows:
        if row['roots'] != 64:
            raise ValueError('Incomplete root denominator')
        if row['status'] == 'withheld_nonfinite_pair':
            if row['mean'] is not None or row['interval'] is not None:
                raise ValueError('Withheld contrast has numerical inference')
        elif row['status'] in ('finite', 'zero_observed_variance'):
            if not math.isfinite(row['mean']):
                raise ValueError('Nonfinite point estimate')
            ci = row['interval']
            if ci is not None and not (math.isfinite(ci['lower'])
                    and math.isfinite(ci['upper'])
                    and ci['lower'] <= row['mean'] <= ci['upper']):
                raise ValueError('Malformed retained confidence interval')
        else:
            raise ValueError('Unknown statistical status')
    return report, keyed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report, rows = read_complete_report(args.report)
    if args.out.exists():
        raise ValueError('Use a fresh figure directory')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    conditions = [(flux, wind, fault) for flux in (1000., 1000000.)
                  for wind in (.5, 2.) for fault in ('none', 'optical_gain', 'vibration')]
    faults = {'none': 'Clean', 'optical_gain': 'Gain drift', 'vibration': 'Vibration'}
    labels = [f"{'Low' if flux == 1000. else 'High'} flux / "
              f"{'slow' if wind == .5 else 'fast'} / {faults[fault]}"
              for flux, wind, fault in conditions]
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'none',
                         'svg.hashsalt': 'noether-recovery-study-v1'})
    args.out.mkdir(parents=True)
    for name, limits in (('primary-contrasts', None),
                         ('primary-contrasts-detail', (-.006, .04))):
        fig, axes = plt.subplots(1, 2, figsize=(12.4, 7.5), sharex=True, sharey=True)
        for ax, arm, title in zip(axes, ('issued_state', 'command_matched'),
                                 ('Issued-state transfer', 'Command-matched transfer')):
            ax.axvline(0., color='#777777', linewidth=.8, linestyle='--')
            for index, condition in enumerate(conditions):
                row = rows[(*condition, arm)]
                if row['mean'] is None:
                    ax.text(.02, index, 'Withheld: nonfinite pair',
                            transform=ax.get_yaxis_transform(), va='center', fontsize=9)
                    continue
                ci = row['interval']
                color = {'increased_residual': '#a34130', 'decreased_residual': '#216880',
                         'unresolved': '#555555'}[row['direction']]
                if ci is not None:
                    ax.plot([ci['lower'], ci['upper']], [index, index], color=color, linewidth=1.7)
                ax.plot(row['mean'], index, 'o', color=color, markersize=4)
                if limits is not None and ci is not None:
                    for bound, marker, endpoint in ((limits[0], '<', ci['lower']),
                                                    (limits[1], '>', ci['upper'])):
                        if endpoint < limits[0] or endpoint > limits[1]:
                            ax.plot(bound, index, marker=marker, color=color,
                                    markersize=6, clip_on=False)
                    if not limits[0] <= row['mean'] <= limits[1]:
                        ax.text(.5, index - .19, 'Mean outside detail; see full scale',
                                transform=ax.get_yaxis_transform(), ha='center',
                                fontsize=7.5, color=color)
            for y in (2.5, 5.5, 8.5):
                ax.axhline(y, color='#dddddd', linewidth=.6)
            ax.set_title(title, loc='left', fontsize=12, pad=13)
            ax.set_xlabel('Normalized excess residual phase variance', labelpad=10)
            ax.tick_params(axis='y', length=0)
            ax.spines['left'].set_visible(False)
        if limits is not None:
            axes[0].set_xlim(*limits)
        axes[0].set_yticks(range(12), labels)
        axes[0].invert_yaxis()
        fig.suptitle('Scheduled fallback versus continued prediction'
                     + (' — detail near zero' if limits else ''), x=.025,
                     ha='left', fontsize=17)
        fig.text(.025, .925, '64 paired simulation roots per condition; 2,304 retained controller trials',
                 fontsize=11, color='#555555')
        note = ('Detail crops the wide gain-drift interval; arrows mark continuation beyond the axis. '
                'See the full-scale figure.\n' if limits else '')
        fig.text(.025, .025,
                 note + 'Positive values indicate higher residual variance. Retained Student-t intervals '
                 'use the frozen 24-contrast Bonferroni adjustment.\n'
                 'No practical margin or hardware benefit is established.',
                 fontsize=9, color='#555555')
        fig.subplots_adjust(left=.245, right=.975, bottom=.17, top=.855, wspace=.14)
        for suffix in ('png', 'svg'):
            path = args.out / f'{name}.{suffix}'
            fig.savefig(path, dpi=180, facecolor='white',
                        metadata={'Date': None} if suffix == 'svg' else None)
            if suffix == 'svg':
                path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
        plt.close(fig)
    provenance = {'report_sha256': hashlib.sha256(args.report.read_bytes()).hexdigest(),
                  'figure_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'manifest_sha256': report['manifest_sha256'],
                  'acquisition_sha256': report['acquisition_sha256'],
                  'interpretation': 'Presentation of all retained primary contrasts; no new inference.'}
    (args.out / 'provenance.json').write_text(json.dumps(provenance, indent=2, sort_keys=True)+'\n')


if __name__ == '__main__':
    main()
