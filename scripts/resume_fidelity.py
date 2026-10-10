"""Recover an interrupted fidelity run using its exact recorded source commit."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

os.environ['JAX_ENABLE_X64']='1'
os.environ['MPLBACKEND']='Agg'


def main() -> None:
    """Validate existing inputs and append only absent configured trials."""
    parser=argparse.ArgumentParser()
    parser.add_argument('directory',type=Path)
    args=parser.parse_args(); path=args.directory
    cfg=json.loads((path/'config.json').read_text())
    provenance=json.loads((path/'provenance.json').read_text())
    records=json.loads((path/'index.json').read_text())
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (path/'raw').glob('*.npz')}
    with tempfile.TemporaryDirectory(prefix='noether-fidelity-source-') as temporary:
        for filename,digest in provenance['source_sha256'].items():
            data=subprocess.check_output(['git','show',provenance['git_commit']+':'+filename])
            if hashlib.sha256(data).hexdigest()!=digest:
                raise RuntimeError('Recorded source differs from committed source: '+filename)
            target=Path(temporary)/filename;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        sys.path.insert(0,str(Path(temporary)/'src'))
        import jax
        import numpy as np
        from noether_optics.artifacts import fingerprint,write_json
        from noether_optics.experiments.runners import Recorder,calibration,predictor_for,compiled,open_metrics,render
        machine=fingerprint()
        if machine!=json.loads((path/'machine.json').read_text()):
            raise RuntimeError('Recovery requires an identical runtime fingerprint')
        for record in records:
            with np.load(path/record['file']) as data:
                if not all(np.isfinite(data[k]).all() for k in data.files):
                    raise RuntimeError('Invalid saved trial: '+record['file'])
        done={(r['labels']['level'],r['labels']['flux'],r['labels']['seed']) for r in records}
        if len(done)!=len(records): raise RuntimeError('Duplicate saved trials')
        write_json(path/'interruption.json',{'reason':'ENOSPC while writing trial_0071.npz; first 71 indexed trials complete',
            'completed_trials':len(records),'utc':datetime.now(timezone.utc).isoformat(),
            'source_commit':provenance['git_commit'],'source_hashes_verified':True,'machine':machine,
            'preserved_raw_sha256':hashes})
        rec=Recorder.__new__(Recorder);rec.path=path;rec.index=records
        fc=cfg|cfg['fidelity']['truth']
        for level in cfg['fidelity']['levels']:
            missing=[(flux,seed) for flux in cfg['fidelity']['fluxes'] for seed in cfg['seeds'] if (level['name'],flux,seed) not in done]
            if not missing: continue
            cal=calibration(fc,level);pred=predictor_for(fc,cal)
            for flux in dict.fromkeys(flux for flux,seed in missing):
                c=fc|{'photons':flux};fn=compiled(c,cal,'forward',pred)
                for f,seed in missing:
                    if f!=flux: continue
                    with np.load(path/f'raw/atmosphere_{seed}.npz') as data: phases=data['phase']
                    out=jax.block_until_ready(fn(phases,seed))
                    out['open_variance']=open_metrics(phases,c)['variance']
                    rec.save(c,out,{'level':level['name'],'flux':flux,'seed':seed})
                    print('Recovered',level['name'],flux,seed,flush=True)
        if not all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items()):
            raise RuntimeError('A preserved raw file changed')
        render(path,'fidelity')
        write_json(path/'recovery_complete.json',{'trials':len(rec.index),'original_raw_unchanged':True,
            'utc':datetime.now(timezone.utc).isoformat()})


if __name__=='__main__': main()
