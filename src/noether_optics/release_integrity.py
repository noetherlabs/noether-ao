"""Streaming integrity checks for a pinned Prediction Boundary download.

This verifies released bytes, not scientific conclusions or a hostile filesystem.
No optical model, random stream, or model fitting is invoked.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat

_CHUNK_BYTES = 1024 * 1024
_MAX_MANIFEST_BYTES = 10 * 1024 * 1024
_MAX_FILES = 10000


def _regular_path(root: Path, name: str) -> Path:
    if not isinstance(name, str):
        raise ValueError('Manifest paths must be strings')
    relative = PurePosixPath(name)
    if (not name or str(relative) != name or relative.is_absolute()
            or '..' in relative.parts or '\\' in name):
        raise ValueError('Manifest path is not a canonical relative path')
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError('Use a regular-file local download, not cache symlinks: ' + name)
    if not stat.S_ISREG(current.stat().st_mode):
        raise ValueError('Expected a regular release file: ' + name)
    return current


def _identity(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _sha256(path: Path) -> tuple[str, int]:
    before = path.stat()
    value = hashlib.sha256()
    count = 0
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(_CHUNK_BYTES), b''):
            value.update(chunk)
            count += len(chunk)
            if count > before.st_size:
                raise ValueError('Release file grew during verification: ' + path.name)
    if _identity(before) != _identity(path.stat()) or count != before.st_size:
        raise ValueError('Release file changed during verification: ' + path.name)
    return value.hexdigest(), count


def verify_release(root: Path, *, manifest_sha256: str, manifest_path: Path | None = None) -> dict:
    """Check every declared original file against an externally pinned manifest.

    Hub download metadata, dataset documentation and empty execution locks are
    outside the original-data manifest and are not part of this verification.
    The function does not write to the download or reconstruct study metrics.
    A separate manifest can verify an original acquisition directory unchanged.
    """
    if (not isinstance(manifest_sha256, str)
            or re.fullmatch('[0-9a-f]{64}', manifest_sha256) is None):
        raise ValueError('Supply the published SHA-256 of the release manifest')
    root = Path(root).resolve(strict=True)
    if manifest_path is None:
        manifest_path = _regular_path(root, 'release-manifest.json')
    else:
        manifest_path = Path(manifest_path)
        manifest_path = _regular_path(manifest_path.parent.resolve(strict=True), manifest_path.name)
    before = manifest_path.stat()
    if before.st_size > _MAX_MANIFEST_BYTES:
        raise ValueError('Release manifest exceeds the 10 MiB metadata bound')
    with manifest_path.open('rb') as stream:
        payload = stream.read(_MAX_MANIFEST_BYTES + 1)
    if (_identity(before) != _identity(manifest_path.stat())
            or len(payload) > _MAX_MANIFEST_BYTES
            or hashlib.sha256(payload).hexdigest() != manifest_sha256):
        raise ValueError('Release manifest differs from its published digest')
    manifest = json.loads(payload)
    if (not isinstance(manifest, dict)
            or manifest.get('schema') != 'noether-prediction-boundary-release/1'
            or type(manifest.get('total_bytes')) is not int
            or manifest['total_bytes'] < 0):
        raise ValueError('Invalid Prediction Boundary release manifest')
    rows = manifest.get('files')
    if not isinstance(rows, list) or not 0 < len(rows) <= _MAX_FILES:
        raise ValueError('Manifest must declare between 1 and 10,000 files')
    names = set()
    expected_bytes = 0
    # Validate the whole declaration before opening a potentially large archive.
    for row in rows:
        if (not isinstance(row, dict) or set(row) != {'path', 'bytes', 'sha256'}
                or not isinstance(row['path'], str)
                or type(row['bytes']) is not int or row['bytes'] < 0
                or not isinstance(row['sha256'], str)
                or re.fullmatch('[0-9a-f]{64}', row['sha256']) is None):
            raise ValueError('Invalid release file declaration')
        if row['path'] in names or row['path'] == 'release-manifest.json':
            raise ValueError('Duplicate or self-referential release path')
        names.add(row['path'])
        expected_bytes += row['bytes']
        path = _regular_path(root, row['path'])
        if path.stat().st_size != row['bytes']:
            raise ValueError('Release file size differs: ' + row['path'])
    if expected_bytes != manifest['total_bytes']:
        raise ValueError('Manifest total does not equal the declared file sizes')
    observed_bytes = 0
    identities = {}
    for row in rows:
        path = _regular_path(root, row['path'])
        identities[row['path']] = _identity(path.stat())
        digest, size = _sha256(path)
        if digest != row['sha256'] or size != row['bytes']:
            raise ValueError('Release file content differs: ' + row['path'])
        observed_bytes += size
    for name, identity in identities.items():
        if _identity(_regular_path(root, name).stat()) != identity:
            raise ValueError('Release file changed during the full scan: ' + name)
    if _sha256(manifest_path)[0] != manifest_sha256:
        raise ValueError('Release manifest changed during verification')
    return {'status': 'verified_release_bytes', 'manifest_sha256': manifest_sha256,
            'files': len(rows), 'bytes': observed_bytes,
            'scope': 'Original files declared in the pinned manifest; no scientific reanalysis.'}
