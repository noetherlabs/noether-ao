"""Acquire the frozen recovery cohort in immutable, bounded root packets.

    collection manifest -> next root -> raw packet -> hash-verified completion
                                     interrupted -> retain and refuse overwrite

Resume verifies completed roots and dispatches only the next unattempted root.
This is acquisition bookkeeping, not a physical audit or scientific verdict.
Run under an external wall/memory/storage supervisor with exclusive compute.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import re

from . import recovery_development as acquisition
from .controller_evidence import _hash_file, capture_inputs
from .controller_selection import _bytes, _publish
from .recovery_study import protocol


def read_json(path):
    return json.loads(Path(path).read_bytes())


def verify_packet(root):
    """Verify an exact immutable inventory before any consumer loads arrays."""
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('Require a regular recovery packet directory')
    paths = list(root.rglob('*'))
    if any(p.is_symlink() or not (p.is_file() or p.is_dir()) for p in paths):
        raise ValueError('Recovery packet contains a nonregular artifact')
    complete_path = root / 'complete.json'
    if not complete_path.is_file():
        raise ValueError('Incomplete root retained; automatic overwrite or retry is forbidden')
    digest = _hash_file(complete_path)
    complete = read_json(complete_path)
    if set(complete) != {'status', 'files'} or not isinstance(complete['files'], dict):
        raise ValueError('Malformed recovery packet inventory')
    names = {p.relative_to(root).as_posix() for p in paths if p.is_file()}
    if names != set(complete['files']) | {'complete.json'} or 'complete.json' in complete['files']:
        raise ValueError('Recovery packet inventory differs')
    for name, expected in complete['files'].items():
        if _hash_file(root / name) != expected:
            raise ValueError('Recovery artifact changed: ' + name)
    if _hash_file(complete_path) != digest:
        raise ValueError('Recovery completion changed during verification')
    return complete, digest


def root_indices(spec, seed):
    return [i for i, case in enumerate(acquisition.cases(spec)) if case['seed'] == seed]


def verify_root(root, collection, collection_sha256, seed):
    complete, digest = verify_packet(root)
    spec = collection['protocol']
    indices = root_indices(spec, seed)
    manifest = read_json(Path(root) / 'manifest.json')
    expected = {key: collection[key] for key in ('protocol', 'source', 'runtime', 'base')}
    expected.update(case_indices=indices, collection=collection_sha256)
    if manifest != expected:
        raise ValueError('Recovery root manifest differs from its frozen collection')
    report = read_json(Path(root) / 'report.json')
    schedule = acquisition.cases(spec)
    if ([row['case'] for row in report['cases']] != [schedule[i] for i in indices]
            or report['schema'] != spec['schema'] or report['smoke'] is not spec.get('smoke', False)
            or report['terminal_records'] != 3 * len(indices)
            or report['expected_terminal_records'] != 3 * len(indices)):
        raise ValueError('Recovery root schedule or terminal count differs')
    if any(set(row['arms']) != set(acquisition.ARMS)
           or set(row['contrasts']) != set(acquisition.ARMS[1:]) for row in report['cases']):
        raise ValueError('Recovery root is missing a declared arm or contrast')
    status = ('complete_finite_study_shard' if all(
        contrast['status'] == 'finite_pair'
        for row in report['cases'] for contrast in row['contrasts'].values())
        else 'complete_study_shard_with_withheld_outcomes')
    if report['status'] != status or complete['status'] != status:
        raise ValueError('Recovery root completion status differs')
    return {'seed': seed, 'complete_sha256': digest}, report


def collection_manifest(controller_run, exclusion_sha256):
    """Bind the externally reviewed exclusion inventory, not a freshness proof.

    The external admission must verify that inventory and its scope before
    dispatch. Recording its digest alone cannot establish historical exclusion.
    """
    if not isinstance(exclusion_sha256, str) or not re.fullmatch('[0-9a-f]{64}', exclusion_sha256):
        raise ValueError('Require the SHA256 of the reviewed stream exclusion inventory')
    acquisition.load_training(controller_run)
    inputs = capture_inputs(acquisition.ROOT / 'configs/controller_study_v1.yaml', acquisition.ROOT)
    for name in ('base', 'study'):
        if hashlib.sha256(_bytes(inputs[name])).hexdigest() != acquisition.INPUTS[f'evidence/provenance/{name}.json']:
            raise ValueError('Recovery configuration differs from frozen training inputs')
    return {'schema': 'noether-recovery-acquisition/1', 'protocol': protocol(),
            'source': inputs['source'], 'runtime': inputs['runtime'], 'base': inputs['base'],
            'exclusion_inventory_sha256': exclusion_sha256,
            'resume_policy': 'verify_complete_roots; retain_and_refuse_incomplete_roots'}


def run_collection(output, controller_run, *, exclusion_sha256, max_roots=1):
    """Acquire up to ``max_roots`` next roots; never replace attempted evidence."""
    if type(max_roots) is not int or not 1 <= max_roots <= len(protocol()['seeds']):
        raise ValueError('max_roots must be a positive integer within the frozen cohort')
    raw_output = Path(output).absolute()
    controller_run = Path(controller_run).resolve()
    if raw_output.is_symlink() or raw_output.resolve().is_relative_to(controller_run):
        raise ValueError('Use a separate regular recovery collection directory')
    output = raw_output.resolve()
    manifest = collection_manifest(controller_run, exclusion_sha256)
    output.mkdir(exist_ok=True)
    lock_path = output / '.lock'
    if lock_path.is_symlink():
        raise ValueError('Collection lock must be a regular file')
    with lock_path.open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = output / 'manifest.json'
        if not manifest_path.exists():
            if set(p.name for p in output.iterdir()) != {'.lock'}:
                raise ValueError('Uninitialized collection contains unbound files')
            _publish(manifest_path, manifest)
            (output / 'roots').mkdir()
        if manifest_path.is_symlink() or read_json(manifest_path) != manifest:
            raise ValueError('Frozen collection source, runtime, inputs or exclusion inventory differs')
        if any(p.is_symlink() for p in output.iterdir()):
            raise ValueError('Symlinked recovery collection member')
        if not (output / 'roots').is_dir() or set(p.name for p in output.iterdir()) - {
                '.lock', 'manifest.json', 'roots', 'acquisition.json'}:
            raise ValueError('Unexpected recovery collection member')
        manifest_sha = _hash_file(manifest_path)
        seeds = manifest['protocol']['seeds']
        names = {p.name for p in (output / 'roots').iterdir()}
        if names != {str(seed) for seed in seeds[:len(names)]}:
            raise ValueError('Recovery roots must be an exact prefix of the declared order')
        records = []
        for seed in seeds[:len(names)]:
            record, _ = verify_root(output / 'roots' / str(seed), manifest, manifest_sha, seed)
            records.append(record)
        if (output / 'acquisition.json').exists() and len(records) != len(seeds):
            raise ValueError('Premature collection completion record')
        start = len(records)
        for seed in seeds[start:start + max_roots]:
            packet = output / 'roots' / str(seed)
            acquisition._run_protocol(packet, controller_run, manifest['protocol'],
                                      case_indices=root_indices(manifest['protocol'], seed),
                                      binding=manifest_sha)
            record, _ = verify_root(packet, manifest, manifest_sha, seed)
            records.append(record)
        if collection_manifest(controller_run, exclusion_sha256) != manifest:
            raise ValueError('Collection identity changed during acquisition')
        if _hash_file(manifest_path) != manifest_sha:
            raise ValueError('Collection manifest changed during acquisition')
        result = {'schema': manifest['schema'], 'status': 'incomplete_acquisition',
                  'manifest_sha256': manifest_sha, 'completed_roots': len(records),
                  'expected_roots': len(seeds), 'root_packets': records,
                  'interpretation': 'Acquisition only; independent raw reconstruction and statistics pending.'}
        if len(records) == len(seeds):
            result['status'] = 'complete_acquisition'
            path = output / 'acquisition.json'
            if path.exists():
                if read_json(path) != result:
                    raise ValueError('Collection completion differs from retained root packets')
            else:
                _publish(path, result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controller-run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--exclusion-sha256', required=True)
    parser.add_argument('--max-roots', type=int, default=1)
    args = parser.parse_args()
    result = run_collection(args.out, args.controller_run, exclusion_sha256=args.exclusion_sha256,
                            max_roots=args.max_roots)
    print(json.dumps({key: result[key] for key in ('status', 'completed_roots', 'expected_roots')}))


if __name__ == '__main__':
    main()
