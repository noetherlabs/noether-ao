"""Populate report tables and the unsent summary from preserved result values."""
from __future__ import annotations
from collections import Counter
import json
from pathlib import Path
import re


def read(name: str) -> object:
    """Load a repository-relative result JSON file."""
    return json.loads(Path(name).read_text())


def markdown(headers: list[str], rows: list[list]) -> str:
    """Format measured values consistently without changing stored precision."""
    def cell(value: object) -> str:
        if isinstance(value,float): return f'{value:.6g}'
        return str(value)
    return '\n'.join(['| '+' | '.join(headers)+' |',
        '| '+' | '.join('---' for _ in headers)+' |']+
        ['| '+' | '.join(cell(v) for v in row)+' |' for row in rows])+'\n'


def section(path: str, name: str, content: str) -> None:
    """Replace one explicit generated section, preserving authored prose."""
    target=Path(path);text=target.read_text()
    pattern=f'<!-- BEGIN {name} -->.*?<!-- END {name} -->'
    replacement=f'<!-- BEGIN {name} -->\n\n{content.strip()}\n\n<!-- END {name} -->'
    updated,count=re.subn(pattern,lambda _:replacement,text,flags=re.S)
    if count!=1: raise ValueError(f'Expected one {name} marker in {path}')
    target.write_text(updated)


def main() -> None:
    """Generate all headline and claim tables from session-backed raw summaries."""
    latency=read('results/m1/summary.json');fidelity=read('results/m5/summary.json')
    mixed=read('results/m6/summary.json');faults=read('results/m4/summary.json')
    cfg=read('results/m3/config.json');m5cfg=read('results/m5/config.json')
    def latency_row(n: int, nw: int, precision: str='float32', path: str='device') -> dict:
        return next(r for r in latency if (r['n'],r['wavelengths'],r['precision'],r['path'],r['engine'],r['mode'])==(n,nw,precision,path,'dlux','cpu_all'))
    small=latency_row(32,1);large=latency_row(256,5)
    fluxes=m5cfg['fidelity']['fluxes']
    equivalent=[f for f in fluxes if all(r['equivalent'] for r in fidelity if r['photons']==f)]
    threshold=[]
    for flux in fluxes:
        if flux not in equivalent: break
        threshold.append(flux)
    c2=('Useful equivalence at all tested levels: '+', '.join(f'{x:g}' for x in equivalent)+' photons/frame.' if equivalent else 'No tested flux gave useful equivalence across all fidelity levels.')
    c2+=' '+(f'At the tested points, equivalence holds from {min(fluxes):g} through {max(threshold):g} photons/frame.' if threshold else 'No supported low-flux threshold was found; lower fidelity often performed better under this fixed estimator.')
    learned=[r for r in mixed if r['controller']=='learned']
    both=sum(r['integrator_verdict']=='wins' and r['forward_verdict']=='wins' for r in learned)
    loses=sum(r['integrator_verdict']=='loses' or r['forward_verdict']=='loses' for r in learned)
    c3=f'Learned weights beat both integrator and forward in {both}/{len(learned)} conditions and lose to at least one in {loses}/{len(learned)}; remaining comparisons include uncertainty.'
    head=markdown(['Claim','Measured verdict'],[
        ['C1',f'Conditional support. N=32, one wavelength, float32: p99 {small["p99_ms"]:.6g} ms; N=256, five wavelengths: p99 {large["p99_ms"]:.6g} ms. CPU component only.'],
        ['C2',c2],['C3',c3],
        ['C4','Partial detection/recovery, no general safety result. Optical-gain drift missed; vibration fallback worsened WFE; clean false alarms occurred.']])
    section('README.md','HEADLINES',head);section('docs/results.md','HEADLINES',head)
    chosen=[latency_row(n,1) for n in (32,64,96,128,192,256)]+[large,latency_row(32,1,path='host_roundtrip')]
    counts=sum(r['samples'] for r in latency)
    text=f'{len(latency)} engine/path/shape rows contain {counts:,} synchronized evaluations. Maximum complex-field relative L2 error was {max(r["relative_l2"] for r in latency):.6g}. Selected dLux CPU rows follow; all use float32.\n\n'
    text+=markdown(['N','Wavelengths','Path','p50 ms','p99 ms','Maximum ms','Miss >1 ms, fraction'],[[r['n'],r['wavelengths'],r['path'],r['p50_ms'],r['p99_ms'],r['max_ms'],r['miss_1ms']] for r in chosen])
    text+='\nFingerprint: `'+read('results/m1/machine.json')['id']+'`.\n'
    section('docs/results.md','LATENCY',text)
    import csv
    physics=list(csv.DictReader(Path('results/m3/predictive_improvement.csv').open()))
    slopes=read('results/m3/fitting_slopes.json')
    psfs=list(csv.DictReader(Path('results/m3/psf.csv').open()))
    text=f"Both independent PSF checks passed; maximum normalized intensity relative L2 was {max(float(r['relative_l2']) for r in psfs):.6g}. The fitting spacing exponent was {slopes['d_exponent']:.6g}; the r0 exponent was {slopes['r0_exponent']:.6g}. All saved physical gates passed.\n\n"
    text+=markdown(['Sensor','Integrator nm','Predictive nm','WFE reduction %','Predictive − integrator nm','Paired 95% interval nm'],[[r['sensor'],float(r['integrator_nm']),float(r['predictive_nm']),float(r['reduction_percent']),float(r['paired_delta_nm']),f"[{float(r['ci95_low_nm']):.6g}, {float(r['ci95_high_nm']):.6g}]"] for r in physics])
    text+=f"\nThe nominal runs use {cfg['modes']} modes, {1/cfg['dt_s']:g} Hz, {cfg['delay_frames']} frames of delay, {cfg['frames']} frames per trial and {cfg['burn_in']} burn-in frames, with {len(cfg['seeds'])} evaluation seeds. These are configuration inputs.\n"
    section('docs/results.md','PHYSICS',text)
    text=c2+'\n\n'
    text+=markdown(['Level','Pupil N','Wavelengths','Modes'],[[l['name'],l['n'],len(l['wavelengths_m']),l['modes']] for l in m5cfg['fidelity']['levels']])+'\n'
    text+=markdown(['Photons/frame','Low WFE nm ± SE','Medium WFE nm ± SE','High WFE nm ± SE','All useful/equivalent'],[[flux]+[next(f"{r['wfe_nm']:.6g} ± {r['seed_se_nm']:.6g}" for r in fidelity if r['photons']==flux and r['level']==level) for level in ['low','medium','high']]+[flux in equivalent] for flux in fluxes])
    text+=f"\nEquivalence margin: ±{100*m5cfg['fidelity']['equivalence_fraction']:g}% of the high-fidelity mean (declared input); {len(m5cfg['seeds'])} paired seeds per cell. The disk interruption preserved the completed trials and recovered missing trials using verified original source. See `results/m5/interruption.json` and `recovery_complete.json`.\n"
    section('docs/results.md','FIDELITY',text)
    text=c3+'\n\n'
    stats=[]
    for controller in ('fixed','learned'):
        rows=[r for r in mixed if r['controller']==controller]
        for baseline in ('integrator','forward','predictive'):
            count=Counter(r[baseline+'_verdict'] for r in rows)
            stats.append([controller,baseline,count['wins'],count['loses'],count['inside_noise']])
    text+=markdown(['Mixture','Compared with','Wins','Loses','Inside noise'],stats)+'\n'
    text+=markdown(['Photons/frame','Wind multiplier','Fault','Learned WFE nm ± SE','vs integrator','vs forward','vs predictive'],[[r['flux'],r['wind'],r['fault'],f"{r['wfe_nm']:.6g} ± {r['seed_se_nm']:.6g}",r['integrator_verdict'],r['forward_verdict'],r['predictive_verdict']] for r in learned])
    training=read('results/m6/training.json');m6cfg=read('results/m6/config.json')
    import numpy as np
    with np.load('results/m6/raw/training.npz') as data:
        first=float(data['history'][0,1]);parameters=sum(data[f'final_{i}'].size for i in range(4))
    text+=f"\nThe {parameters}-parameter network took {m6cfg['mixed']['training_steps']} configured training steps. Saved training loss changed from {first:.6g} to {training['final_loss']:.6g} rad². This training loss is not held-out performance. Exact training conditions are in [training.json](../results/m6/training.json).\n"
    section('docs/results.md','MIXED',text)
    clean=read('results/m4/clean_false_alarms.json')
    text=f"The clean frame false-alarm rate was {100*clean['frame_false_alarm_rate']:.6g}% over {clean['evaluated_clean_frames']} frames; the clean run false-alarm rate was {100*clean['run_false_alarm_rate']:.6g}%. The frozen score threshold was {clean['threshold']:.6g}.\n\n"
    text+=markdown(['Fault','Detected without earlier alarm / runs','Delay ms, detected only','No fallback WFE nm','Fallback WFE nm','Paired Δ 95% interval nm'],[[r['fault'],f"{r['detected_without_prior_alarm']}/{r['total_runs']}",r['delay_ms_detected_only'] if r['delay_ms_detected_only'] is not None else 'none detected',r['wfe_without_nm'],r['wfe_fallback_nm'],f"[{r['ci95_low_nm']:.6g}, {r['ci95_high_nm']:.6g}]"] for r in faults])
    section('docs/results.md','FAULTS',text)
    dragrace=list(csv.DictReader(Path('results/m2/table.csv').open()))
    selected=[r for r in dragrace if r['status']=='ok' and int(r['n']) in (128,256)]
    section('contributions/drafts/dragrace_pr.md','DRAGRACE',markdown(['Adapter','Pupil grid','Median ms','p99 ms','Observed max ms'],[[r['adapter'],int(r['n']),float(r['median_ms']),float(r['p99_ms']),float(r['max_ms'])] for r in selected]))
    pyramid=next(r for r in physics if r['sensor']=='pyramid')
    summary=f'''# Simulation results and questions for Louis

Unsent draft for human review. This is a small civil-astronomy prototype, not an
instrument deployment or an institutional collaboration.

The dLux CPU FFT sweep is configuration dependent: at N=32, one wavelength and
float32, p99 was {small['p99_ms']:.6g} ms; at N=256 and five wavelengths it was
{large['p99_ms']:.6g} ms. These are synchronized single-plane component measurements.
There is no GPU result or full-loop deadline claim. The separate Dragrace MFT
contribution adds a small-pupil case and stored p99/maximum columns; it remains unsubmitted upstream.

The delayed pyramid simulation passed independent HCIPy open-loop PSF, fitting-law,
delay-growth and nominal stability checks. Its predictor reduced mean WFE by
{float(pyramid['reduction_percent']):.6g}% relative to the integrator in the configured
bright-star regime. This does not reproduce an unspecified instrument configuration.

{c2}

{c3}

The assurance result is mixed. Gain drift escaped detection; fallback under
vibration made WFE worse. Clean false alarms occurred in
{100*clean['run_false_alarm_rate']:.6g}% of the short held-out runs. Missing/late-output
fallback helped, but those faults are synthetic availability flags. No general
safety guarantee follows. Full numbers, raw data, plots and reproduction commands
are linked from [the results report](../../docs/results.md).

Three questions:

1. Which pupil sampling, detector binning and wavelength quadrature would make this
   ideal pyramid comparison useful for a MagAO-X-like configuration?
2. Which telemetry or optical-gain estimator should accompany a frame-residual
   detector, given that decreasing optical gain escaped this upper-tail score?
3. Which tuned classical/predictive baseline and weighting inputs would make the
   mixed-controller comparison persuasive before increasing the training horizon?

Could you suggest a public or synthetic MagAO-X-like test configuration: pupil and
bandpass, DM basis, pyramid geometry/modulation, detector noise/flux, delays, and
turbulence profile? A configuration that others may redistribute would let this
repository test a relevant case without implying access to private instrument data.
'''
    Path('contributions/drafts/summary_for_louis.md').write_text(summary)


if __name__=='__main__': main()
