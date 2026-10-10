"""Run locked-checkout correctness, table reproduction and experiment entry points."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault('JAX_ENABLE_X64','1')
os.environ.setdefault('MPLBACKEND','Agg')


def main() -> None:
    """Keep logs, raw smoke outputs and explicit full-versus-smoke verification scope."""
    from noether_optics.artifacts import start_run,write_json
    p=argparse.ArgumentParser();p.add_argument('--dragrace',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    dragrace=args.dragrace.resolve();py=sys.executable
    path=start_run({'purpose':'Fresh-checkout verification; smoke data excluded from scientific claims',
        'validation':'M3 full configuration; M4-M6 shorter functional configuration; M1 selected point with full sample count; M2 full AO case'},args.out).resolve()
    commands=[('noether-tests',[py,'-m','pytest','-q'],{}),
        ('dragrace-tests',[py,'-m','pytest','-q',str(dragrace/'tests')],{'PYTHONPATH':str(dragrace/'src')}),
        ('source-audit',[py,'scripts/audit_provenance.py','--out',str(path/'source-snapshots.json')],{}),
        ('tables',[py,'scripts/rebuild_all.py','--dragrace',str(dragrace),'--report',str(path/'table-reproduction.json')],{}),
        ('m1',[py,'-m','noether_optics.cli','latency','--config','configs/latency_smoke.yaml','--out',str(path/'smoke/m1')],{}),
        ('m2',[py,'scripts/run_dragrace.py','--repo',str(dragrace),'--out',str(path/'smoke/m2')],{}),
        ('m3',[py,'-m','noether_optics.cli','validate','--config','configs/simulation.yaml','--out',str(path/'full/m3')],{})]
    for label,name in [('m4','faults'),('m5','fidelity'),('m6','mixed')]:
        commands.append((label,[py,'-m','noether_optics.cli',name,'--config','configs/reproduction_smoke.yaml','--out',str(path/'smoke'/label)],{}))
    states=[]
    for label,command,extra in commands:
        print('Verifying',label,flush=True)
        working_directory=dragrace if label=='dragrace-tests' else Path.cwd()
        with (path/(label+'.log')).open('w') as log:
            result=subprocess.run(command,cwd=working_directory,env=os.environ|extra,stdout=log,stderr=subprocess.STDOUT)
        state={'check':label,'command':command,'cwd':str(working_directory),'returncode':result.returncode};states.append(state)
        write_json(path/'checks.json',states)
        if result.returncode: raise SystemExit(f'Failed {label}; inspect saved log')
    full_gates=json.loads((path/'full/m3/gates.json').read_text())
    smoke_gates={n:json.loads((path/f'smoke/{n}/gates.json').read_text()) for n in ('m4','m5','m6')}
    workers=json.loads((path/'smoke/m2/workers.json').read_text())
    all_pass=all(full_gates.values()) and all(all(g.values()) for g in smoke_gates.values()) and all(w['returncode']==0 for w in workers)
    write_json(path/'verification.json',{'all_commands_passed':True,'full_m3_gates':full_gates,
        'smoke_completeness_gates':smoke_gates,'m2_workers':workers,'all_pass':all_pass,
        'scope':'Tables rebuilt from original raw data; M3 rerun fully; other simulation entry points exercised with labeled shorter horizons. New timing equality is not required.'})
    if not all_pass: raise SystemExit('Verification gate failure; see verification.json')
    print('Fresh-checkout verification passed.',flush=True)


if __name__=='__main__': main()
