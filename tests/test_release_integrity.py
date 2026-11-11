"""Download corruption and path-boundary regressions; no research acquisition."""
import hashlib
import json

import pytest

from noether_optics.release_integrity import verify_release


@pytest.fixture
def release(tmp_path):
    root = tmp_path / 'download'
    (root / 'evidence').mkdir(parents=True)
    payload = b'finite synthetic fixture bytes'
    (root / 'evidence/raw.npz').write_bytes(payload)
    manifest = {'schema': 'noether-prediction-boundary-release/1', 'total_bytes': len(payload),
                'files': [{'path': 'evidence/raw.npz', 'bytes': len(payload),
                           'sha256': hashlib.sha256(payload).hexdigest()}]}
    return root, manifest


def pin(root, manifest):
    payload = json.dumps(manifest).encode()
    (root / 'release-manifest.json').write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def test_regular_download_verifies_without_touching_data_or_hub_metadata(release):
    root, manifest = release
    (root / '.cache/huggingface').mkdir(parents=True)
    (root / '.cache/huggingface/download').write_bytes(b'download state')
    expected = pin(root, manifest)
    before = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
    result = verify_release(root, manifest_sha256=expected)
    assert result['status'] == 'verified_release_bytes'
    assert result['files'] == 1 and result['bytes'] == manifest['total_bytes']
    assert before == {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_self_consistent_changed_manifest_cannot_replace_external_pin(release):
    root, manifest = release
    expected = pin(root, manifest)
    manifest['total_bytes'] += 1
    pin(root, manifest)
    with pytest.raises(ValueError, match='published digest'):
        verify_release(root, manifest_sha256=expected)


def test_separate_manifest_checks_acquisition_without_adding_files(release, tmp_path):
    root, manifest = release
    expected = pin(root, manifest)
    separate = tmp_path / 'release-manifest.json'
    (root / 'release-manifest.json').rename(separate)
    before = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
    result = verify_release(root, manifest_sha256=expected, manifest_path=separate)
    assert result['files'] == 1
    assert before == {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_same_size_corruption_is_detected(release):
    root, manifest = release
    expected = pin(root, manifest)
    path = root / 'evidence/raw.npz'
    path.write_bytes(b'x' * path.stat().st_size)
    with pytest.raises(ValueError, match='content differs'):
        verify_release(root, manifest_sha256=expected)


@pytest.mark.parametrize('kind', ['file', 'directory'])
def test_cached_symlinks_cannot_escape_local_download(release, tmp_path, kind):
    root, manifest = release
    expected = pin(root, manifest)
    if kind == 'file':
        path = root / 'evidence/raw.npz'
        blob = tmp_path / 'blob'
        path.rename(blob)
        path.symlink_to(blob)
    else:
        path = root / 'evidence'
        outside = tmp_path / 'outside'
        path.rename(outside)
        path.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match='local download'):
        verify_release(root, manifest_sha256=expected)


@pytest.mark.parametrize('name', ['../outside', '/outside', 'a/../outside', './raw', 'a\\raw'])
def test_manifest_path_traversal_is_rejected(release, name):
    root, manifest = release
    manifest['files'][0]['path'] = name
    with pytest.raises(ValueError, match='relative path'):
        verify_release(root, manifest_sha256=pin(root, manifest))


def test_duplicate_files_cannot_inflate_complete_inventory(release):
    root, manifest = release
    manifest['files'] *= 2
    manifest['total_bytes'] *= 2
    with pytest.raises(ValueError, match='Duplicate'):
        verify_release(root, manifest_sha256=pin(root, manifest))


@pytest.mark.parametrize('value', [True, -1, 1.5, '29'])
def test_file_sizes_require_nonnegative_integer(release, value):
    root, manifest = release
    manifest['files'][0]['bytes'] = value
    with pytest.raises(ValueError, match='declaration'):
        verify_release(root, manifest_sha256=pin(root, manifest))


def test_missing_file_does_not_count_as_verified(release):
    root, manifest = release
    expected = pin(root, manifest)
    (root / 'evidence/raw.npz').unlink()
    with pytest.raises(FileNotFoundError):
        verify_release(root, manifest_sha256=expected)


def test_changed_size_is_rejected_before_content_scan(release):
    root, manifest = release
    expected = pin(root, manifest)
    (root / 'evidence/raw.npz').write_bytes(b'truncated')
    with pytest.raises(ValueError, match='size differs'):
        verify_release(root, manifest_sha256=expected)


def test_inconsistent_total_is_rejected(release):
    root, manifest = release
    manifest['total_bytes'] += 1
    with pytest.raises(ValueError, match='Manifest total'):
        verify_release(root, manifest_sha256=pin(root, manifest))


def test_mutation_after_file_hash_cannot_pass_full_scan(release, monkeypatch):
    from noether_optics import release_integrity
    root, manifest = release
    expected = pin(root, manifest)
    original = release_integrity._sha256
    def change_after_hash(path):
        result = original(path)
        if path.name == 'raw.npz':
            path.write_bytes(b'x' * path.stat().st_size)
        return result
    monkeypatch.setattr(release_integrity, '_sha256', change_after_hash)
    with pytest.raises(ValueError, match='full scan'):
        verify_release(root, manifest_sha256=expected)
