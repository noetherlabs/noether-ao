"""Auditable simulation experiments. All metrics are regenerated from raw arrays."""
from __future__ import annotations

import gc
import json
from pathlib import Path

import dLux as dl
import hcipy as hp
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

from ..artifacts import start_run, write_json, mean_se, write_table as table
from ..atmosphere import phase_sequence
from ..optics import aperture, relative_l2
from ..simulation import (calibration, modal_basis, predict_fit, loop, summarize,
                          CONTROLLERS, FAULTS, paired_interval, initialize_weights)


def predictor_for(cfg: dict, cal: dict) -> np.ndarray:
    """Fit only training-seed open-loop modal telemetry, without test data."""
    telemetry = []
    for seed in cfg['training_seeds']:
        phases = phase_sequence(cfg, seed)
        telemetry.append(np.asarray(phases).reshape(len(phases),-1) @ np.asarray(cal['projection']).T)
    # Fit individual sequences before averaging coefficients; never bridge seeds.
    return np.mean([predict_fit(a,cfg['delay_frames'],cfg['predictor_ridge']) for a in telemetry],axis=0)


class Recorder:
    """Save raw outputs immediately, retaining per-trial configs and inputs."""
    def __init__(self, path: Path):
        self.path = path
        (path/'raw').mkdir()
        self.index = []

    def save(self, cfg: dict, output: dict, labels: dict, full: bool = False) -> dict:
        """Persist a trial before calculating its reported metrics."""
        name = f'raw/trial_{len(self.index):04d}.npz'
        arrays = {k:np.asarray(v) for k,v in output.items() if full or k not in ('measurement','prediction','estimate','command','applied')}
        arrays['all_finite_full_outputs']=np.asarray(all(np.isfinite(np.asarray(v)).all() for v in output.values()))
        np.savez_compressed(self.path/name, **arrays)
        record = {'file':name, 'config':cfg, 'labels':labels}
        self.index.append(record)
        write_json(self.path/'index.json',self.index)
        return labels | summarize(arrays,cfg)


def compiled(cfg: dict, cal: dict, controller: str, predictor: np.ndarray,
             **kwargs) -> object:
    """Compile one shape/config, varying phase, seed and optional network weights."""
    return jax.jit(lambda phases,seed,params=None: loop(cfg,cal,phases,seed,controller,
                    jnp.asarray(predictor),params=params,**kwargs))


def open_metrics(phases: np.ndarray, cfg: dict) -> dict:
    """Reference uncorrected variance and exact on-axis Strehl."""
    mask=np.asarray(aperture(cfg['n']))
    phases=phases*mask
    phases-=np.sum(phases,axis=(-2,-1))[:,None,None]/mask.sum()*mask
    variance=np.sum(phases**2,axis=(-2,-1))/mask.sum()
    strehl=np.abs(np.sum(mask*np.exp(1j*phases),axis=(-2,-1))/mask.sum())**2
    return {'variance':variance,'strehl':strehl,'saturated':np.zeros(len(phases),dtype=bool)}


def validate(cfg: dict, destination: str) -> Path:
    """Independent optical check, fitting law and closed-loop physical checks."""
    path=start_run(cfg,destination); rec=Recorder(path); gates={}
    n=cfg['n']; wavelength=cfg['wavelength_m']; diameter=cfg['diameter_m']
    pupil=hp.make_pupil_grid(n,diameter)
    focal=hp.make_uniform_grid([2*n,2*n],[n*wavelength/diameter]*2)
    prop=hp.FraunhoferPropagator(pupil,focal)
    rng=np.random.default_rng(cfg['seed'])
    psf_rows=[]
    for label,phase in [('flat',np.zeros((n,n))),('aberrated',rng.normal(size=(n,n))*.2)]:
        field=np.asarray(aperture(n))*np.exp(1j*phase)
        hp_psf=np.asarray(prop(hp.Wavefront(hp.Field(field.ravel(),pupil),wavelength)).intensity).reshape(2*n,2*n)
        wave=dl.Wavefront(wavelength=wavelength,npixels=n,diameter=diameter).set(phasor=jnp.asarray(field))
        dl_psf=np.asarray(wave.propagate(npixels=2*n,pixel_scale=wavelength/(2*diameter)).psf)
        hp_psf/=hp_psf.sum(); dl_psf=dl_psf/dl_psf.sum()
        error=float(relative_l2(jnp.asarray(dl_psf),jnp.asarray(hp_psf)))
        psf_rows.append({'case':label,'relative_l2':error,'pass':error<=cfg['validation']['psf_tolerance']})
        np.savez_compressed(path/f'raw/psf_{label}.npz',hcipy=hp_psf,dlux=dl_psf,phase=phase)
    table(path/'psf',psf_rows); gates['hcipy_psf']=all(r['pass'] for r in psf_rows)
    print('Independent PSF comparison:',psf_rows,flush=True)
    fit_cfg=cfg|{'n':cfg['validation']['fitting_n'],'frames':cfg['validation']['fitting_frames']}
    counts=cfg['validation']['fitting_modes']; basis=modal_basis(fit_cfg['n'],max(counts)); mask=np.asarray(aperture(fit_cfg['n']))
    values=[]
    for seed in cfg['seeds']:
        phase=np.asarray(phase_sequence(fit_cfg,seed))*mask
        phase-=phase.sum(axis=(-2,-1))[:,None,None]/mask.sum()*mask
        coeff=phase.reshape(len(phase),-1) @ basis.reshape(len(basis),-1).T/mask.sum()
        for count in counts:
            residual=phase-np.einsum('tm,mij->tij',coeff[:,:count],basis[:count])
            v=(residual**2).sum(axis=(-2,-1))/mask.sum()
            values.append({'seed':seed,'modes':count,'d_m':diameter/np.sqrt(count),'variance_rad2':float(v.mean())})
        np.savez_compressed(path/f'raw/fitting_{seed}.npz',phase=phase,coefficients=coeff)
    table(path/'fitting',values)
    mean_var=[np.mean([r['variance_rad2'] for r in values if r['modes']==m]) for m in counts]
    slope=float(np.polyfit(np.log(diameter/np.sqrt(counts)),np.log(mean_var),1)[0])
    r0vals=cfg['validation']['r0_values']; r0vars=[]
    for r0 in r0vals:
        p=np.asarray(phase_sequence(fit_cfg|{'r0_m':r0},cfg['seed']))*mask
        p-=p.sum(axis=(-2,-1))[:,None,None]/mask.sum()*mask
        coeff=p.reshape(len(p),-1) @ basis[:counts[0]].reshape(counts[0],-1).T/mask.sum()
        residual=p-np.einsum('tm,mij->tij',coeff,basis[:counts[0]])
        per_frame=np.sum(residual**2,axis=(-2,-1))/mask.sum()
        np.savez_compressed(path/f'raw/r0_{r0}.npz',variance=per_frame)
        r0vars.append(float(np.mean(per_frame)))
    np.savez_compressed(path/'raw/fitting_laws.npz',modes=counts,variance=mean_var,r0=r0vals,r0variance=r0vars)
    r0slope=float(np.polyfit(np.log(r0vals),np.log(r0vars),1)[0])
    gates['fitting_d']=abs(slope-5/3)<=cfg['validation']['fitting_slope_tolerance']
    gates['fitting_r0']=abs(r0slope+5/3)<=cfg['validation']['r0_slope_tolerance']
    write_json(path/'fitting_slopes.json',{'d_exponent':slope,'r0_exponent':r0slope})
    print('Fitting law:',slope,r0slope,flush=True)
    rows=[]
    for sensor in ['linear','pyramid']:
        scfg=cfg|{'sensor':sensor}; cal=calibration(scfg)
        phases={seed:phase_sequence(scfg,seed) for seed in cfg['seeds']}
        for seed,p in phases.items():
            rows.append(rec.save(scfg,open_metrics(np.asarray(p),scfg),{'sensor':sensor,'controller':'open','delay':0,'seed':seed}))
        for delay in cfg['validation']['delays']:
            dc=scfg|{'delay_frames':delay}; pred=predictor_for(dc,cal)
            np.savez_compressed(path/f'raw/predictor_{sensor}_{delay}.npz',coefficients=pred)
            controllers=['integrator','predictive','forward'] if delay==cfg['delay_frames'] else ['integrator']
            for controller in controllers:
                fn=compiled(dc,cal,controller,pred)
                for seed,p in phases.items():
                    out=jax.block_until_ready(fn(p,seed))
                    row=rec.save(dc,out,{'sensor':sensor,'controller':controller,'delay':delay,'seed':seed})
                    rows.append(row)
                print(sensor,delay,controller,round(rows[-1]['wfe_nm'],4),flush=True)
        del phases,cal,fn; jax.clear_caches(); gc.collect()
    table(path/'trials',rows)
    for sensor in ['linear','pyramid']:
        means={d:np.mean([r['residual_variance_rad2'] for r in rows if r['sensor']==sensor and r['controller']=='integrator' and r['delay']==d]) for d in cfg['validation']['delays']}
        delays=sorted(means); slope_delay=np.polyfit(delays,[means[d] for d in delays],1)[0]
        gates[f'{sensor}_delay']=bool(means[delays[-1]]>means[delays[0]] and slope_delay>0)
        closed=[r for r in rows if r['sensor']==sensor and r['controller']=='integrator' and r['delay']==cfg['delay_frames']]
        open_wfe=np.mean([r['wfe_nm'] for r in rows if r['sensor']==sensor and r['controller']=='open'])
        gates[f'{sensor}_stable']=bool(all(r['all_outputs_finite'] and np.isfinite(r['wfe_nm']) and 0<r['strehl']<=1+1e-12 and r['saturation_fraction']==0 for r in closed) and np.mean([r['wfe_nm'] for r in closed])<open_wfe)
    write_json(path/'gates.json',gates); render(path,'validate')
    return path


def faults(cfg: dict, destination: str) -> Path:
    """Calibrate on clean training seeds, then test every configured fault."""
    path=start_run(cfg,destination); rec=Recorder(path); cal=calibration(cfg)
    pred=predictor_for(cfg,cal); fn=compiled(cfg,cal,'forward',pred)
    scores=[]
    for seed in cfg['training_seeds']:
        out=jax.block_until_ready(fn(phase_sequence(cfg,seed),seed))
        np.savez_compressed(path/f'raw/calibration_{seed}.npz',score=np.asarray(out['score']))
        scores.extend(np.asarray(out['score'])[cfg['burn_in']:].tolist())
    threshold=float(np.quantile(scores,cfg['detector']['calibration_quantile']))
    write_json(path/'detector.json',{'threshold':threshold,'training_seeds':cfg['training_seeds'],'quantile':cfg['detector']['calibration_quantile']})
    np.savez_compressed(path/'raw/predictor.npz',coefficients=pred)
    phases={s:phase_sequence(cfg,s) for s in cfg['seeds']}
    for fault in FAULTS:
        for fallback in [False,True]:
            fn=compiled(cfg,cal,'forward',pred,threshold=threshold,fallback=fallback,fault=fault)
            for seed,p in phases.items():
                out=jax.block_until_ready(fn(p,seed))
                rec.save(cfg,out,{'fault':fault,'fallback_enabled':fallback,'seed':seed},full=seed==cfg['seeds'][0])
        print('Fault completed:',fault,flush=True)
        jax.clear_caches(); gc.collect()
    render(path,'faults'); return path


def fidelity(cfg: dict, destination: str) -> Path:
    """Vary only the twin against a common higher-resolution plant and noise seeds."""
    path=start_run(cfg,destination); rec=Recorder(path)
    m1=Path(cfg['fidelity']['m1_result'])
    measurements=json.loads((m1/'summary.json').read_text())
    latency_rows=[]
    for level in cfg['fidelity']['levels']:
        matches=[r for r in measurements if r['engine']=='dlux' and r['path']=='device' and r['mode']=='cpu_all' and r['precision']=='float64' and r['n']==level['n'] and r['wavelengths']==len(level['wavelengths_m'])]
        if len(matches)!=1:
            raise ValueError('Each fidelity needs exactly one matching M1 component result')
        r=matches[0]
        latency_rows.append({'level':level['name'],'n':level['n'],'wavelengths':len(level['wavelengths_m']),'p50_ms':r['p50_ms'],'p99_ms':r['p99_ms'],'max_ms':r['max_ms'],'miss_1ms':r['miss_1ms'],'source_raw':str(m1/r['raw'])})
    table(path/'component_latency',latency_rows)
    write_json(path/'m1_machine.json',json.loads((m1/'machine.json').read_text()))
    fc=cfg|cfg['fidelity']['truth']
    phases_by_seed={seed:phase_sequence(fc,seed) for seed in cfg['seeds']}
    for seed,phases in phases_by_seed.items():
        np.savez_compressed(path/f'raw/atmosphere_{seed}.npz',phase=np.asarray(phases))
    for level in cfg['fidelity']['levels']:
        print('Calibrating fidelity:',level['name'],flush=True)
        cal=calibration(fc,level); pred=predictor_for(fc,cal)
        for flux in cfg['fidelity']['fluxes']:
            c=fc|{'photons':flux}; fn=compiled(c,cal,'forward',pred)
            for seed in cfg['seeds']:
                phases=phases_by_seed[seed]
                out=jax.block_until_ready(fn(phases,seed))
                out['open_variance']=open_metrics(np.asarray(phases),c)['variance']
                rec.save(c,out,{'level':level['name'],'flux':flux,'seed':seed})
            print('Fidelity completed:',level['name'],flux,flush=True)
        del cal,fn; jax.clear_caches(); gc.collect()
    render(path,'fidelity'); return path


def mixed(cfg: dict, destination: str) -> Path:
    """Train a shared weighting network through a short loop; evaluate held-out seeds."""
    if set(cfg['seeds'])&set(cfg['training_seeds']):
        raise ValueError('Training and evaluation seeds overlap')
    path=start_run(cfg,destination); rec=Recorder(path); cal=calibration(cfg)
    pred=predictor_for(cfg,cal); mc=cfg['mixed']
    tc=cfg|{'frames':mc['training_frames'],'burn_in':8}
    training=[]
    for i,seed in enumerate(cfg['training_seeds']):
        c=tc|{'photons':mc['fluxes'][i%len(mc['fluxes'])],'wind_scale':mc['winds'][i%len(mc['winds'])]}
        training.append((c,phase_sequence(c,seed),seed))
    params=initialize_weights(cfg['seed']); initial=params
    def objective(parameters):
        losses=[]
        for c,p,seed in training:
            outputs=loop(c,cal,p,seed,'learned',jnp.asarray(pred),params=parameters,differentiable_noise=True)
            losses.append(jnp.mean(outputs['variance'][c['burn_in']:]))
        return jnp.mean(jnp.stack(losses))
    optimizer=optax.adam(mc['learning_rate']); state=optimizer.init(params)
    @jax.jit
    def step(parameters,state):
        loss,gradient=jax.value_and_grad(objective)(parameters)
        updates,state=optimizer.update(gradient,state,parameters)
        return optax.apply_updates(parameters,updates),state,loss,optax.global_norm(gradient)
    history=[]
    for i in range(mc['training_steps']):
        params,state,loss,norm=jax.block_until_ready(step(params,state))
        history.append([i,float(loss),float(norm)])
        if not np.isfinite(history[-1][1:]).all():
            raise RuntimeError('Nonfinite training objective or gradients')
        print('Training',i,float(loss),flush=True)
    np.savez_compressed(path/'raw/training.npz',history=history,**{f'initial_{i}':np.asarray(v) for i,v in enumerate(initial)},**{f'final_{i}':np.asarray(v) for i,v in enumerate(params)})
    write_json(path/'training.json',{'training_seeds':cfg['training_seeds'],'evaluation_seeds':cfg['seeds'],'final_loss':float(objective(params)),'noise':'Gaussian reparameterization in training; exact Poisson in evaluation'})
    jax.clear_caches(); gc.collect()
    for flux in mc['fluxes']:
        for wind in mc['winds']:
            c=cfg|{'photons':flux,'wind_scale':wind}; phases={s:phase_sequence(c,s) for s in cfg['seeds']}
            for fault in mc['faults']:
                for controller in CONTROLLERS:
                    fn=compiled(c,cal,controller,pred,fault=fault)
                    for seed,p in phases.items():
                        out=jax.block_until_ready(fn(p,seed,params if controller=='learned' else None))
                        rec.save(c,out,{'flux':flux,'wind':wind,'fault':fault,'controller':controller,'seed':seed})
                    print('Mixed completed:',flux,wind,fault,controller,flush=True)
                jax.clear_caches(); gc.collect()
    render(path,'mixed'); return path


def validation_tables(path: Path, rows: list[dict], cfg: dict) -> None:
    """Regenerate physical validation details from the saved per-seed metrics."""
    delay_rows=[]; improvement=[]
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for sensor in ['linear','pyramid']:
        cell=[r for r in rows if r['sensor']==sensor]
        for delay in cfg['validation']['delays']:
            group=[r for r in cell if r['controller']=='integrator' and r['delay']==delay]
            wfe,se=mean_se([r['wfe_nm'] for r in group])
            delay_rows.append({'sensor':sensor,'delay_frames':delay,'wfe_nm':wfe,'seed_se_nm':se,
                'variance_rad2':float(np.mean([r['residual_variance_rad2'] for r in group])),
                'strehl':float(np.mean([r['strehl'] for r in group]))})
        a=sorted([r for r in cell if r['controller']=='integrator' and r['delay']==cfg['delay_frames']],key=lambda r:r['seed'])
        b=sorted([r for r in cell if r['controller']=='predictive'],key=lambda r:r['seed'])
        av=[r['wfe_nm'] for r in a];bv=[r['wfe_nm'] for r in b]
        delta,lo,hi=paired_interval(bv,av)
        improvement.append({'sensor':sensor,'integrator_nm':float(np.mean(av)),'predictive_nm':float(np.mean(bv)),
            'reduction_percent':float(100*(1-np.mean(bv)/np.mean(av))),
            'paired_delta_nm':delta,'ci95_low_nm':lo,'ci95_high_nm':hi})
        group=[r for r in delay_rows if r['sensor']==sensor]
        axes[1].errorbar([r['delay_frames'] for r in group],[r['wfe_nm'] for r in group],yerr=[r['seed_se_nm'] for r in group],marker='o',label=sensor)
    with np.load(path/'raw/fitting_laws.npz') as data:
        d=cfg['diameter_m']/np.sqrt(data['modes'])
        axes[0].loglog(d,data['variance'],'o-',label='measured')
        axes[0].loglog(d,data['variance'][0]*(d/d[0])**(5/3),'--',label='5/3 slope, anchored')
    axes[0].set(xlabel='Effective spacing D / sqrt(modes), m',ylabel='Fitting variance, rad²');axes[0].legend()
    axes[1].set(xlabel='Command delay, frames',ylabel='Residual WFE, nm (seed SE)');axes[1].legend()
    fig.savefig(path/'physical_checks.png',dpi=160);plt.close(fig)
    table(path/'delay',delay_rows);table(path/'predictive_improvement',improvement)


def render(path: Path, experiment: str) -> None:
    """Rebuild reported controller metrics, tables and plots from saved raw arrays."""
    cfg=json.loads((path/'config.json').read_text())
    records=json.loads((path/'index.json').read_text())
    rows=[]
    for record in records:
        with np.load(path/record['file']) as data:
            row=record['labels']|summarize(data,record['config'])
            if 'open_variance' in data:
                row['open_wfe_nm']=float(np.sqrt(data['open_variance'][record['config']['burn_in']:].mean())*record['config']['wavelength_m']/(2*np.pi)*1e9)
            rows.append(row)
    table(path/'trials',rows)
    fig,ax=plt.subplots(figsize=(8,4.8),layout='constrained')
    output=[]
    if experiment=='validate':
        validation_tables(path,rows,cfg)
        for sensor in ['linear','pyramid']:
            for controller in ['open','integrator','predictive','forward']:
                selected=[r['wfe_nm'] for r in rows if r['sensor']==sensor and r['controller']==controller and (r['delay']==cfg['delay_frames'] or controller=='open')]
                mean,se=mean_se(selected)
                output.append({'sensor':sensor,'controller':controller,'wfe_nm':mean,'seed_se_nm':se})
        x=np.arange(len(output)); ax.bar(x,[r['wfe_nm'] for r in output],yerr=[r['seed_se_nm'] for r in output])
        ax.set_xticks(x,[r['sensor']+'\n'+r['controller'] for r in output],rotation=35,ha='right')
    elif experiment=='fidelity':
        levels=[l['name'] for l in cfg['fidelity']['levels']]
        for level in levels:
            group=[]
            for flux in cfg['fidelity']['fluxes']:
                selected=sorted([r for r in rows if r['level']==level and r['flux']==flux],key=lambda r:r['seed'])
                high=sorted([r for r in rows if r['level']==levels[-1] and r['flux']==flux],key=lambda r:r['seed'])
                a=[r['wfe_nm'] for r in selected]; b=[r['wfe_nm'] for r in high]
                mean,se=mean_se(a); delta,lo,hi=paired_interval(a,b)
                band=cfg['fidelity']['equivalence_fraction']*np.mean(b)
                # Saturated controllers cannot establish useful equivalence.
                stable=all(r['saturation_fraction']==0 and r['wfe_nm']<r['open_wfe_nm'] for r in selected+high)
                row={'level':level,'photons':flux,'wfe_nm':mean,'seed_se_nm':se,'delta_vs_high_nm':delta,'ci95_low_nm':lo,'ci95_high_nm':hi,'equivalent':bool(lo>=-band and hi<=band and stable),'unsaturated':stable}
                output.append(row); group.append(row)
            ax.errorbar([r['photons'] for r in group],[r['wfe_nm'] for r in group],yerr=[r['seed_se_nm'] for r in group],marker='o',label=level)
        ax.set_xscale('log'); ax.set_xlabel('Photons per frame'); ax.legend()
    elif experiment=='faults':
        rocfig,rocax=plt.subplots(figsize=(6,5),layout='constrained')
        clean=[]
        for record in records:
            if record['labels']['fault']=='none' and not record['labels']['fallback_enabled']:
                clean.extend(np.load(path/record['file'])['score'][cfg['burn_in']:].tolist())
        roc_data={}
        for fault in FAULTS:
            no=[r['wfe_nm'] for r in rows if r['fault']==fault and not r['fallback_enabled']]
            yes=[r['wfe_nm'] for r in rows if r['fault']==fault and r['fallback_enabled']]
            delays=[]; detections=0; postalarms=0; alarms=[]; fault_scores=[]; pretrigger=[]; availability=[]
            for record in records:
                if record['labels']['fault']!=fault or record['labels']['fallback_enabled']: continue
                with np.load(path/record['file']) as data:
                    start=cfg['faults']['onset']; stop=min(cfg['faults']['end'],len(data['score']))
                    active=np.flatnonzero(data['alarm'][start:stop])
                    if len(active):
                        postalarms+=1
                        if not np.any(data['alarm'][cfg['burn_in']:start]) and fault!='none':
                            detections+=1; delays.append(float(active[0]*cfg['dt_s']*1000))
                    alarms.extend(data['alarm'][cfg['burn_in']:start].tolist())
                    pretrigger.append(bool(np.any(data['alarm'][cfg['burn_in']:start])))
                    fault_scores.extend(data['score'][start:stop].tolist())
                    availability.extend(data['unavailable'][start:stop].tolist())
            delta,lo,hi=paired_interval(yes,no)
            output.append({'fault':fault,'detected_without_prior_alarm':detections,'post_onset_alarm_runs':postalarms,'total_runs':len(no),'delay_ms_detected_only':float(np.mean(delays)) if delays else None,'pre_onset_frame_fpr':float(np.mean(alarms)),'pre_onset_run_fpr':float(np.mean(pretrigger)),'wfe_without_nm':float(np.mean(no)),'wfe_fallback_nm':float(np.mean(yes)),'fallback_delta_nm':delta,'ci95_low_nm':lo,'ci95_high_nm':hi})
            thresholds=np.r_[np.inf,np.quantile(np.r_[clean,fault_scores],np.linspace(1,0,101)),-np.inf]
            fpr=np.array([np.mean(np.asarray(clean)>t) for t in thresholds]); score_tpr=np.array([np.mean(np.asarray(fault_scores)>t) for t in thresholds])
            tpr=np.array([np.mean((np.asarray(fault_scores)>t)|np.asarray(availability)) for t in thresholds])
            roc_data[fault+'_fpr']=fpr; roc_data[fault+'_tpr']=tpr; roc_data[fault+'_score_only_tpr']=score_tpr
            rocax.plot(fpr,tpr,label=fault)
        np.savez_compressed(path/'raw/roc.npz',**roc_data)
        rocax.plot([0,1],[0,1],':',color='gray'); rocax.set(xlabel='Clean frame false-positive rate',ylabel='Post-onset score OR availability true-positive rate'); rocax.legend(fontsize=8); rocfig.savefig(path/'roc.png',dpi=160); plt.close(rocfig)
        x=np.arange(len(output)); ax.bar(x-.18,[r['wfe_without_nm'] for r in output],.36,label='forward'); ax.bar(x+.18,[r['wfe_fallback_nm'] for r in output],.36,label='fallback'); ax.set_xticks(x,[r['fault'] for r in output],rotation=35,ha='right'); ax.legend()
    elif experiment=='mixed':
        for flux in cfg['mixed']['fluxes']:
            for wind in cfg['mixed']['winds']:
                for fault in cfg['mixed']['faults']:
                    cell=[r for r in rows if r['flux']==flux and r['wind']==wind and r['fault']==fault]
                    values={c:[r['wfe_nm'] for r in sorted(cell,key=lambda r:r['seed']) if r['controller']==c] for c in CONTROLLERS}
                    for controller in CONTROLLERS:
                        mean,se=mean_se(values[controller]); row={'flux':flux,'wind':wind,'fault':fault,'controller':controller,'wfe_nm':mean,'seed_se_nm':se}
                        for baseline in ['integrator','forward']:
                            delta,lo,hi=paired_interval(values[controller],values[baseline])
                            row.update({baseline+'_delta_nm':delta,baseline+'_ci95_low':lo,baseline+'_ci95_high':hi,baseline+'_verdict':'wins' if hi<0 else 'loses' if lo>0 else 'inside_noise'})
                        output.append(row)
        for controller in CONTROLLERS:
            group=[r for r in output if r['controller']==controller]
            ax.errorbar(np.arange(len(group)),[r['wfe_nm'] for r in group],yerr=[r['seed_se_nm'] for r in group],marker='o',label=controller)
        ax.set_xlabel('Condition index (table order)'); ax.legend(fontsize=8)
    else:
        raise ValueError(experiment)
    ax.set_ylabel('Residual WFE (nm RMS; bars = across-seed SE)')
    ax.set_title(experiment+' — '+json.loads((path/'machine.json').read_text())['id'])
    fig.savefig(path/'result.png',dpi=160); plt.close(fig)
    table(path/'table',output); write_json(path/'summary.json',output)


def run(experiment: str, config: dict, destination: str) -> Path:
    """Dispatch a named experiment without implicit parameter changes."""
    return {'validate':validate,'faults':faults,'fidelity':fidelity,'mixed':mixed}[experiment](config,destination)
