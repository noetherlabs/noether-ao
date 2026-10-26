"""Locate exact acquisition source and lock snapshots by content hash."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def blob(commit: str, name: str) -> bytes | None:
    """Read a committed blob without treating an absent historical path as data."""
    result=subprocess.run(['git','show',commit+':'+name],capture_output=True)
    return result.stdout if result.returncode==0 else None


def main() -> None:
    """Record exact matches, including files untracked at a run's starting commit."""
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();reports={};unresolved=[];unresolved_executed=[]
    for n in range(1,7):
        p=Path(f'results/m{n}/provenance.json');provenance=json.loads(p.read_text())
        start=provenance['git_commit'];matches={}
        targets=provenance['source_sha256']|{'uv.lock':provenance['lock_sha256']}
        for name,digest in targets.items():
            snapshot=Path('results/source_snapshots')/digest
            archived=snapshot.is_file() and hashlib.sha256(snapshot.read_bytes()).hexdigest()==digest
            revisions=[] if archived else [start]+subprocess.check_output(['git','log','--format=%H','--',name],text=True).splitlines()
            match=None
            for revision in dict.fromkeys(revisions):
                data=blob(revision,name)
                if data is not None and hashlib.sha256(data).hexdigest()==digest:
                    match=revision;break
            matches[name]={'sha256':digest,'matching_commit':match,'in_recorded_start_commit':match==start,
                           'source_snapshot':str(snapshot) if archived else None}
            if match is None and not archived:
                unresolved.append(f'm{n}: {name}')
                # M2's committed script imports only artifacts from this package;
                # the AO implementation was an unused, untracked source inventory.
                if n!=2 or name in ('src/noether_optics/artifacts.py','src/noether_optics/__init__.py','uv.lock'):
                    unresolved_executed.append(f'm{n}: {name}')
        reports[f'm{n}']={'recorded_start_commit':start,'files':matches}
    report={'runs':reports,'all_source_and_lock_snapshots_recoverable':not unresolved,'unresolved':unresolved,
        'executed_source_and_lock_snapshots_recoverable':not unresolved_executed,
        'unresolved_executed':unresolved_executed,
        'note':'A later matching commit identifies a hashed working-tree snapshot; it does not rewrite original provenance. M2 imported only artifacts from this package; its inventory also hashed unused, then-untracked AO drafts. Two of those unused snapshots cannot be recovered.'}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    if unresolved_executed: raise SystemExit('Unresolved executed snapshots: '+str(unresolved_executed))
    print('Executed source and lock snapshots are recoverable; unresolved unused inventory:',unresolved)


if __name__=='__main__': main()
