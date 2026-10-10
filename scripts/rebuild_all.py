"""Regenerate published numerical tables from saved results, then compare values."""
from __future__ import annotations
import argparse
import csv
import hashlib
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault('JAX_ENABLE_X64','1')
os.environ.setdefault('MPLBACKEND','Agg')


def compare_csv(before: str, after: str) -> bool:
    """Compare labels exactly and numerical cells at strict relative tolerance."""
    import io
    import numpy as np
    a=list(csv.reader(io.StringIO(before)));b=list(csv.reader(io.StringIO(after)))
    if len(a)!=len(b): return False
    for ar,br in zip(a,b):
        if len(ar)!=len(br): return False
        for x,y in zip(ar,br):
            if x==y: continue
            try:
                if not np.isclose(float(x),float(y),rtol=1e-12,atol=1e-14): return False
            except ValueError: return False
    return True


def main() -> None:
    """Rebuild every completed milestone without running a benchmark in CI."""
    p=argparse.ArgumentParser()
    p.add_argument('--dragrace',type=Path,default=Path('../Diffraction-Dragrace'))
    p.add_argument('--report',type=Path,required=True)
    args=p.parse_args()
    files=[path for n in range(1,7) for path in (Path('results')/f'm{n}').glob('*') if path.suffix in ('.csv','.md')]
    files += [Path(p) for p in ('README.md','docs/results.md','outbox/summary_for_louis.md','outbox/dragrace_pr.md')]
    before={str(p):p.read_text() for p in files}
    protected=[p for n in range(1,7) for p in (Path('results')/f'm{n}').rglob('*') if p.is_file() and (p.name in ('config.json','machine.json','provenance.json','index.json') or p.suffix=='.npz' or '/raw/' in p.as_posix()) and p.name!='roc.npz']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    commands=[]
    for name,directory in [('latency','m1'),('validate','m3'),('faults','m4'),('fidelity','m5'),('mixed','m6')]:
        command=[sys.executable,'-m','noether_optics.experiments.rebuild',name,'results/'+directory]
        subprocess.run(command,check=True);commands.append(command)
    command=[sys.executable,'scripts/run_dragrace.py','--repo',str(args.dragrace.resolve()),'--out','results/m2','--render-only']
    subprocess.run(command,check=True);commands.append(command)
    command=[sys.executable,'scripts/write_report.py']
    subprocess.run(command,check=True);commands.append(command)
    compared={}
    for filename,old in before.items():
        new=Path(filename).read_text()
        compared[filename]=compare_csv(old,new) if filename.endswith('.csv') else old==new
    unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    from noether_optics.artifacts import write_json,fingerprint
    args.report.parent.mkdir(parents=True,exist_ok=True)
    write_json(args.report,{'tables':compared,'raw_inputs_unchanged':unchanged,
                           'commands':commands,'machine':fingerprint(),
                           'all_pass':all(compared.values()) and unchanged})
    if not all(compared.values()) or not unchanged:
        raise SystemExit('Reproduction difference: inspect the report')
    print(f'Reproduced {len(compared)} table files; raw inputs unchanged.')


if __name__=='__main__': main()
