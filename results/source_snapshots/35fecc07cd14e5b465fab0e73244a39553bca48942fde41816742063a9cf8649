"""Rebuild every published table from versioned raw files without rerunning timing."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path


def main() -> None:
    """Regenerate one saved experiment's tables and figures."""
    os.environ.setdefault('JAX_ENABLE_X64','1')
    os.environ.setdefault('MPLBACKEND','Agg')
    parser=argparse.ArgumentParser()
    parser.add_argument('experiment',choices=['latency','validate','faults','fidelity','mixed'])
    parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    if args.experiment=='latency':
        from ..latency import render
        render(args.directory)
    else:
        from .runners import render,table
        render(args.directory,args.experiment)
        if args.experiment=='validate':
            import numpy as np
            cfg=json.loads((args.directory/'config.json').read_text())
            psfs=[]
            for name in ['flat','aberrated']:
                with np.load(args.directory/f'raw/psf_{name}.npz') as data:
                    error=float(np.linalg.norm(data['dlux']-data['hcipy'])/np.linalg.norm(data['hcipy']))
                    psfs.append({'case':name,'relative_l2':error,'pass':error<=cfg['validation']['psf_tolerance']})
            table(args.directory/'psf',psfs)
            from ..simulation import modal_basis
            from ..optics import aperture
            n=cfg['validation']['fitting_n']; counts=cfg['validation']['fitting_modes']; basis=modal_basis(n,max(counts)); mask=np.asarray(aperture(n))
            rows=[]
            for seed in cfg['seeds']:
                with np.load(args.directory/f'raw/fitting_{seed}.npz') as data:
                    for count in counts:
                        residual=data['phase']-np.einsum('tm,mij->tij',data['coefficients'][:,:count],basis[:count])
                        rows.append({'seed':seed,'modes':count,'d_m':cfg['diameter_m']/np.sqrt(count),'variance_rad2':float(np.mean(np.sum(residual**2,axis=(-2,-1))/mask.sum()))})
            table(args.directory/'fitting',rows)


if __name__=='__main__':
    main()
