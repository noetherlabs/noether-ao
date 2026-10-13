"""Temporary Git repositories and synthetic arrays; no study simulation."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from noether_optics import controller_evidence as evidence
from noether_optics.controller_protocol import load_study
from noether_optics.controller_selection import _bytes, _digest

ROOT = Path(__file__).resolve().parents[1]


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, stderr=subprocess.STDOUT)


@pytest.fixture
def repository(tmp_path, monkeypatch):
    root = tmp_path / "source"
    root.mkdir()
    for directory in ("src/sample", "scripts", "configs"):
        (root / directory).mkdir(parents=True)
    (root / "src/sample/__init__.py").write_text('"""Synthetic source fixture."""\n')
    (root / "scripts/run.py").write_text("# Synthetic driver; never executes physics.\n")
    (root / "pyproject.toml").write_text('[project]\nname = "fixture-only"\nversion = "0"\n')
    (root / "uv.lock").write_text("version = 1\n")
    (root / ".gitignore").write_text("__pycache__/\nignored.py\n")
    (root / "configs/simulation.yaml").write_bytes((ROOT / "configs/simulation.yaml").read_bytes())
    study = load_study(ROOT / "configs/controller_study_v1.yaml")
    study["study_id"] = "evidence_fixture_only"
    for role, start in (("training", 71001), ("validation", 72001), ("evaluation", 73001)):
        study["seeds"][role] = list(range(start, start + len(study["seeds"][role])))
    study["seeds"]["initialization"] = 74001
    (root / "configs/controller_study_v1.yaml").write_text(yaml.safe_dump(study))
    git(root, "init", "-q")
    git(root, "config", "user.name", "Evidence fixture")
    git(root, "config", "user.email", "fixture@example.invalid")
    git(root, "add", ".")
    git(root, "commit", "-qm", "Synthetic test inputs")
    monkeypatch.setattr(evidence, "ROOT", root)
    monkeypatch.setattr(evidence, "_runtime", lambda: {"machine": {"x64": True, "packages": {"fixture": "1"}},
                                                     "environment": {}, "jax_options": {}})
    return root


def capture(repository):
    return evidence.capture_inputs(Path("configs/controller_study_v1.yaml"), repository)


def test_actual_git_inventory_and_deterministic_binding(repository):
    actual = capture(repository)
    assert actual == capture(repository)
    files = actual["source"]["files"]
    assert set(files) == {"src/sample/__init__.py", "scripts/run.py", "configs/simulation.yaml",
                          "configs/controller_study_v1.yaml", "pyproject.toml", "uv.lock"}
    for path, digest in files.items():
        assert digest == hashlib.sha256((repository / path).read_bytes()).hexdigest()
    provenance = actual["provenance"]
    assert provenance["source_commit"] == git(repository, "rev-parse", "HEAD").decode().strip()
    assert provenance["source_sha256"] == _digest(_bytes(actual["source"]))
    assert provenance["runtime_sha256"] == _digest(_bytes(actual["runtime"]))
    assert actual["study"]["study_id"] == "evidence_fixture_only"
    assert actual["base"]["sensor"] == "pyramid"
    # Results and ordinary interpreter caches are not execution inputs.
    (repository / "results").mkdir()
    (repository / "results/raw.bin").write_bytes(b"synthetic")
    cache = repository / "src/sample/__pycache__"
    cache.mkdir()
    (cache / "module.pyc").write_bytes(b"cache fixture")
    assert capture(repository) == actual


@pytest.mark.parametrize("damage", ["dirty", "untracked", "ignored", "symlink", "missing", "pin"])
def test_bad_execution_inputs_reject(repository, damage):
    if damage == "dirty": (repository / "scripts/run.py").write_text("changed\n")
    elif damage == "untracked": (repository / "src/sample/new.py").write_text("# new\n")
    elif damage == "ignored": (repository / "src/sample/ignored.py").write_text("# ignored\n")
    elif damage == "symlink": (repository / "src/sample/link.py").symlink_to(repository / "scripts/run.py")
    elif damage == "missing": (repository / "uv.lock").unlink()
    else:
        (repository / "configs/simulation.yaml").write_text("sensor: changed\n")
        git(repository, "add", ".")
        git(repository, "commit", "-qm", "Change pinned input without updating pin")
    with pytest.raises(ValueError):
        capture(repository)


def test_source_changed_during_runtime_capture_rejected(repository, monkeypatch):
    def drift():
        (repository / "scripts/run.py").write_text("# changed during capture\n")
        return {"machine": {"x64": True}}
    monkeypatch.setattr(evidence, "_runtime", drift)
    with pytest.raises(ValueError, match="tracked changes|changed during"):
        capture(repository)


def test_external_study_and_base_rejected(repository, tmp_path):
    outside = tmp_path / "outside.yaml"
    outside.write_text("private: synthetic\n")
    with pytest.raises(ValueError, match="Study must be"):
        evidence.capture_inputs(outside, repository)
    study_path = repository / "configs/controller_study_v1.yaml"
    study = yaml.safe_load(study_path.read_text())
    study["base_config"] = str(outside)
    study_path.write_text(yaml.safe_dump(study))
    git(repository, "add", ".")
    git(repository, "commit", "-qm", "Invalid external base")
    with pytest.raises(ValueError, match="Base must be"):
        capture(repository)


def test_x64_required(repository, monkeypatch):
    monkeypatch.setattr(evidence, "_runtime", lambda: {"machine": {"x64": False}})
    with pytest.raises(ValueError, match="x64"):
        capture(repository)


@pytest.fixture
def opened(repository, tmp_path):
    return evidence.open_study_run(tmp_path / "run", create=True)


def test_create_resume_and_provenance_files_are_bound(opened):
    ledger, base, refs = opened
    before = {ref["path"]: (ledger.root / ref["path"]).read_bytes() for ref in refs}
    resumed, second_base, second_refs = evidence.open_study_run(ledger.root)
    assert resumed.manifest == ledger.manifest and second_base == base and second_refs == refs
    assert {ref["path"]: (ledger.root / ref["path"]).read_bytes() for ref in refs} == before
    source = next(ref for ref in refs if ref["path"].endswith("source.json"))
    runtime = next(ref for ref in refs if ref["path"].endswith("runtime.json"))
    assert source["sha256"] == ledger.provenance["source_sha256"]
    assert runtime["sha256"] == ledger.provenance["runtime_sha256"]
    with pytest.raises(FileExistsError):
        evidence.open_study_run(ledger.root, create=True)
    # Existing manifest and bytes survive the rejected new-run request.
    assert resumed.manifest == json.loads((ledger.root / "manifest.json").read_text())


@pytest.mark.parametrize("damage", ["source_commit", "runtime", "saved_inventory"])
def test_resume_rejects_changed_bindings(opened, repository, monkeypatch, damage):
    ledger, _, refs = opened
    if damage == "source_commit":
        git(repository, "commit", "--allow-empty", "-qm", "Different exact source commit")
    elif damage == "runtime":
        monkeypatch.setattr(evidence, "_runtime", lambda: {"machine": {"x64": True, "packages": {"fixture": "2"}}})
    else:
        (ledger.root / refs[0]["path"]).write_bytes(b"changed source inventory")
    with pytest.raises(ValueError, match="provenance changed|replace existing"):
        evidence.open_study_run(ledger.root)


def test_interrupted_provenance_publication_resumes_without_replacement(repository, tmp_path, monkeypatch):
    root = tmp_path / "interrupted"
    real = evidence.save_metadata
    def interrupted(ledger, path, value):
        if path.endswith("runtime.json"):
            raise OSError("simulated publication interruption")
        return real(ledger, path, value)
    monkeypatch.setattr(evidence, "save_metadata", interrupted)
    with pytest.raises(OSError):
        evidence.open_study_run(root, create=True)
    original = (root / "evidence/provenance/source.json").read_bytes()
    monkeypatch.setattr(evidence, "save_metadata", real)
    ledger, _, refs = evidence.open_study_run(root)
    assert len(refs) == 4 and (root / "evidence/provenance/source.json").read_bytes() == original
    assert ledger.root == root


def test_disk_budget_and_partial_root_do_not_destroy_evidence(repository, tmp_path, monkeypatch):
    root = tmp_path / "run"
    monkeypatch.setattr(evidence.shutil, "disk_usage", lambda p: SimpleNamespace(free=1))
    with pytest.raises(RuntimeError, match="free disk"):
        evidence.open_study_run(root, create=True)
    assert not root.exists()
    with pytest.raises(ValueError, match="positive"):
        evidence.open_study_run(root, create=True, min_free_bytes=0)
    root.mkdir()
    (root / "partial.bin").write_bytes(b"preserve")
    with pytest.raises(FileNotFoundError):
        evidence.open_study_run(root, min_free_bytes=1)
    assert (root / "partial.bin").read_bytes() == b"preserve"


def test_arrays_preserve_dtype_shape_failures_and_exact_retry(opened):
    ledger, _, _ = opened
    arrays = {"failed": np.array([np.nan, np.inf, -np.inf], dtype=np.float64),
              "scalar": np.array(True), "matrix": np.arange(12, dtype=np.int32).reshape(3, 4)[:, ::2],
              "complex": np.array([1 + 2j], dtype=np.complex128)}
    ref = evidence.save_arrays(ledger, "evidence/raw/trial.npz", arrays)
    path = ledger.root / ref["path"]
    original = path.read_bytes()
    assert evidence.save_arrays(ledger, ref["path"], dict(reversed(list(arrays.items())))) == ref
    assert path.read_bytes() == original
    loaded = evidence.load_arrays(ledger, ref)
    for key, value in arrays.items():
        assert loaded[key].shape == value.shape and loaded[key].dtype == value.dtype
        np.testing.assert_array_equal(loaded[key], value)
    # Also safe inside an evaluation callback's existing lock.
    with ledger._locked():
        assert evidence.save_arrays(ledger, ref["path"], arrays) == ref
        metadata = evidence.save_metadata(ledger, "evidence/raw/trial.json", {"raw": ref, "failed": True})
    assert ledger.artifact(metadata["path"]) == metadata


@pytest.mark.parametrize("damage", ["value", "shape", "dtype", "key"])
def test_changed_array_cannot_replace_saved_bytes(opened, damage):
    ledger, _, _ = opened
    arrays = {"data": np.arange(4, dtype=np.float64)}
    ref = evidence.save_arrays(ledger, "evidence/data.npz", arrays)
    original = (ledger.root / ref["path"]).read_bytes()
    if damage == "value": arrays["data"][0] = 20
    elif damage == "shape": arrays["data"] = arrays["data"].reshape(2, 2)
    elif damage == "dtype": arrays["data"] = arrays["data"].astype(np.float32)
    else: arrays = {"renamed": arrays["data"]}
    with pytest.raises(ValueError, match="replace existing"):
        evidence.save_arrays(ledger, ref["path"], arrays)
    assert (ledger.root / ref["path"]).read_bytes() == original
    assert not list((ledger.root / "evidence").glob(".pending-*"))


@pytest.mark.parametrize("path", ["/tmp/absolute.npz", "evidence/../escape.npz", "raw/data.npz",
                                  "evidence//alias.npz", "evidence/data.json"])
def test_unsafe_array_paths_reject(opened, path):
    with pytest.raises(ValueError, match="canonical"):
        evidence.save_arrays(opened[0], path, {"data": np.ones(1)})


@pytest.mark.parametrize("arrays", [{}, {"../evil": np.ones(1)}, {"object": np.array([{}], dtype=object)},
                                    {"text": np.array(["text"])}])
def test_non_numeric_or_unsafe_keys_reject_before_file_creation(opened, arrays):
    ledger = opened[0]
    with pytest.raises(ValueError):
        evidence.save_arrays(ledger, "evidence/rejected/data.npz", arrays)
    assert not (ledger.root / "evidence/rejected").exists()


def test_symlinks_and_nonfinite_metadata_rejected(opened, tmp_path):
    ledger = opened[0]
    (ledger.root / "evidence/link").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        evidence.save_arrays(ledger, "evidence/link/escape.npz", {"x": np.ones(1)})
    (ledger.root / "evidence/inside").mkdir()
    (ledger.root / "evidence/alias").symlink_to(ledger.root / "evidence/inside", target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        evidence.save_metadata(ledger, "evidence/alias/a.json", {"x": 1})
    with pytest.raises(ValueError):
        evidence.save_metadata(ledger, "evidence/nonfinite.json", {"x": float("nan")})
    assert not (ledger.root / "evidence/nonfinite.json").exists()


@pytest.mark.parametrize("failure", ["write", "link"])
def test_failed_array_publication_leaves_no_completed_artifact(opened, monkeypatch, failure):
    ledger = opened[0]
    retained = evidence.save_arrays(ledger, "evidence/retained.npz", {"x": np.arange(3.)})
    def fail(*args, **kwargs): raise OSError("simulated disk interruption")
    if failure == "write": monkeypatch.setattr(evidence.np.lib.format, "write_array", fail)
    else: monkeypatch.setattr(evidence.os, "link", fail)
    with pytest.raises(OSError, match="interruption"):
        evidence.save_arrays(ledger, "evidence/incomplete.npz", {"x": np.arange(2.)})
    assert not (ledger.root / "evidence/incomplete.npz").exists()
    assert not list((ledger.root / "evidence").glob(".pending-*"))
    assert ledger.artifact(retained["path"]) == retained


@pytest.mark.parametrize("same", [True, False])
def test_concurrent_publication_has_one_complete_winner(opened, same):
    ledger = opened[0]
    def publish(value):
        try:
            return evidence.save_arrays(ledger, "evidence/race.npz", {"x": np.full(1000, value)})
        except ValueError as error:
            return error
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(publish, [1, 1 if same else 2]))
    successes = [result for result in results if isinstance(result, dict)]
    assert len(successes) == (2 if same else 1)
    if same: assert results[0] == results[1]
    else: assert "replace existing" in str(next(r for r in results if isinstance(r, ValueError)))
    saved = evidence.load_arrays(ledger, successes[0])["x"]
    assert (saved == 1).all() or (saved == 2).all()
    assert not list((ledger.root / "evidence").glob(".pending-*"))


def test_saved_hash_drift_and_pending_file_do_not_become_valid_artifacts(opened):
    ledger = opened[0]
    pending = ledger.root / "evidence/.pending-crash"
    pending.write_bytes(b"interrupted process fixture")
    ref = evidence.save_arrays(ledger, "evidence/data.npz", {"x": np.arange(3.)})
    assert pending.read_bytes() == b"interrupted process fixture"
    with (ledger.root / ref["path"]).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="hash or identity"):
        evidence.load_arrays(ledger, ref)


@pytest.mark.parametrize("flag", ["--assume-unchanged", "--skip-worktree"])
def test_hidden_git_changes_still_reject_actual_source_bytes(repository, flag):
    git(repository, "update-index", flag, "scripts/run.py")
    (repository / "scripts/run.py").write_text("# hidden from ordinary git status\n")
    assert git(repository, "status", "--porcelain", "--untracked-files=no") == b""
    with pytest.raises(ValueError, match="bytes differ"):
        capture(repository)


def test_loader_rejects_pickle_arrays_even_with_matching_artifact_hash(opened):
    ledger = opened[0]
    path = ledger.root / "evidence/object.npz"
    np.savez(path, object=np.array([{}], dtype=object))
    ref = ledger.artifact("evidence/object.npz")
    with pytest.raises(ValueError, match="Object arrays"):
        evidence.load_arrays(ledger, ref)
