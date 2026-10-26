"""Run every installed adapter locally, preserving upstream correctness gates."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

os.environ.setdefault('JAX_ENABLE_X64','1')
os.environ.setdefault('MPLBACKEND','Agg')

from noether_optics.artifacts import start_run,write_json


def render_results(path: Path, repo: Path) -> None:
    """Rebuild the report, table and figure from existing worker results."""
    details=json.loads((path/'config.json').read_text())
    sys.path.insert(0,str(repo/'src'))
    from dragrace.report import aggregate,render_text
    rows=aggregate(path/'raw')
    for row in rows:
        row['path']=Path(row['path']).relative_to(path).as_posix()
    (path/'report.txt').write_text(render_text(rows))
    write_json(path/'aggregated.json',rows)
    from noether_optics.artifacts import write_table as table
    table(path/'table',[{'adapter':r['adapter'],'n':r['scan_value'],'status':r['status'],
        'median_ms':r['median_s']*1000 if r['median_s'] is not None else None,
        'p99_ms':r['p99_s']*1000 if r['p99_s'] is not None else None,
        'max_ms':r['max_s']*1000 if r['max_s'] is not None else None,
        'relative_l2':r['rel_l2'],'reason':r['reason']} for r in rows])
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,5),layout='constrained')
    for adapter in details['adapters']:
        selection=[r for r in rows if r['adapter']==adapter and r['status']=='ok' and not r['traced']]
        if selection:
            ax.plot([r['scan_value'] for r in selection],[r['p99_s']*1000 for r in selection],'.-',label=adapter)
    ax.set(xlabel='Pupil samples per side (fixed focal grid)',ylabel='p99 propagation time (ms)',yscale='log',title='Local Dragrace AO MFT scan')
    ax.legend();fig.savefig(path/'tails.png',dpi=160);plt.close(fig)


def main() -> None:
    """Run the exact AO case with strict backend checks in separate workers."""
    p=argparse.ArgumentParser()
    p.add_argument('--repo',type=Path,default=Path('../Diffraction-Dragrace'))
    p.add_argument('--config',type=Path,default=Path('configs/dragrace_cpu_all.yaml'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--render-only',action='store_true')
    args=p.parse_args(); repo=args.repo.resolve(); config=args.config.resolve()
    if args.render_only:
        render_results(args.out.resolve(),repo)
        return
    if os.environ.get('CI'):
        raise RuntimeError('No benchmark timings in CI')
    case=repo/'cases/pupil_to_focus/mft_ao_scan.yaml'
    details={'seed':17,'randomness':'None: deterministic upstream circular pupil case',
             'upstream_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
             'case_sha256':hashlib.sha256(case.read_bytes()).hexdigest(),
             'case_text':case.read_text(),'config_text':config.read_text(),
             'adapters':['numpy_baseline','dlux','abcdlux','hcipy','poppy','lentil','prysm','proper']}
    path=start_run(details,args.out).resolve()
    shutil.copyfile(case,path/'mft_ao_scan.yaml'); shutil.copyfile(config,path/config.name)
    env=os.environ.copy();env['PYTHONPATH']=str(repo/'src')
    states=[]
    for adapter in details['adapters']:
        output=path/'raw'/adapter
        output.mkdir(parents=True)
        command=[sys.executable,'-m','dragrace.worker','--case',str(case),'--config',str(config),'--adapter',adapter,'--mode','timing','--out',str(output)]
        with (output/'worker.log').open('w') as log:
            result=subprocess.run(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
        states.append({'adapter':adapter,'returncode':result.returncode})
        write_json(path/'workers.json',states)
        print(adapter,result.returncode,flush=True)
    render_results(path,repo)



if __name__=='__main__': main()
