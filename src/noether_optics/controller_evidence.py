"""Fresh study provenance and exclusive numeric artifacts; no scientific execution.

    clean checkout + pinned inputs + runtime -> ledger -> immutable provenance
    numeric arrays -> deterministic NPZ -> flush/link -> ledger artifact reference

Publication is safe inside a ledger-held callback: it never reacquires that lock.
Hashes detect accidental drift, not a hostile writer with access to the run.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import zipfile
import zlib

import numpy as np

from .artifacts import fingerprint, load_config
from .controller_protocol import load_study
from .controller_selection import StudyLedger, _bytes, _digest, _publish, _read

ROOT = Path(__file__).resolve().parents[2]
EXECUTION_PATHS = ("src", "scripts", "configs", "pyproject.toml", "uv.lock")
RUNTIME_ENV = ("JAX_PLATFORMS", "JAX_ENABLE_X64", "JAX_DEFAULT_MATMUL_PRECISION",
               "XLA_FLAGS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "TF_NUM_INTRAOP_THREADS",
               "TF_NUM_INTEROP_THREADS")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(repository: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=repository)


def _source(repository: Path) -> dict:
    if Path(os.fsdecode(_git(repository, "rev-parse", "--show-toplevel")).strip()).resolve() != repository:
        raise ValueError("Require the source repository root")
    if _git(repository, "status", "--porcelain", "--untracked-files=no").strip():
        raise ValueError("Commit tracked changes before opening a scientific study")
    commit = os.fsdecode(_git(repository, "rev-parse", "HEAD")).strip()
    objects = {}
    for entry in _git(repository, "ls-tree", "-r", "-z", commit, "--", *EXECUTION_PATHS).split(b"\0"):
        if entry:
            metadata, name = entry.split(b"\t", 1)
            objects[os.fsdecode(name)] = metadata.split()[2].decode()
    names = set(objects)
    object_format = os.fsdecode(_git(repository, "rev-parse", "--show-object-format")).strip()
    for required in ("pyproject.toml", "uv.lock"):
        if required not in names:
            raise ValueError(f"Missing tracked execution input: {required}")
    # Inspect ignored files too; an ignored Python module can still affect imports.
    actual = {"pyproject.toml", "uv.lock"}
    for directory in EXECUTION_PATHS[:3]:
        folder = repository / directory
        if not folder.is_dir() or folder.is_symlink():
            raise ValueError(f"Missing or symlinked execution directory: {directory}")
        for path in folder.rglob("*"):
            relative = path.relative_to(repository)
            if "__pycache__" in relative.parts or path.name == ".DS_Store":
                continue
            if path.is_symlink():
                raise ValueError(f"Symlinked execution input: {relative}")
            if path.is_file():
                actual.add(relative.as_posix())
    if actual != names:
        raise ValueError("Untracked, ignored or missing execution inputs")
    files = {}
    for name in sorted(names):
        path = repository / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Invalid execution input: {name}")
        digest = hashlib.sha256()
        blob = hashlib.new(object_format)
        blob.update(f"blob {path.stat().st_size}\0".encode())
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
                blob.update(block)
        # Status alone can miss assume-unchanged/skip-worktree files. Bind actual
        # bytes to HEAD as well as recording the independent SHA256 inventory.
        if blob.hexdigest() != objects[name]:
            raise ValueError(f"Execution bytes differ from the committed source: {name}")
        files[name] = digest.hexdigest()
    return {"schema_version": 1, "commit": commit, "files": files}


def _runtime() -> dict:
    import jax
    return {"machine": fingerprint(), "zlib_runtime": zlib.ZLIB_RUNTIME_VERSION,
            "jax_options": {key: getattr(jax.config, key) for key in
                            ("jax_enable_x64", "jax_disable_jit", "jax_default_matmul_precision",
                             "jax_default_prng_impl", "jax_numpy_dtype_promotion", "jax_numpy_rank_promotion")},
            "environment": {key: os.environ.get(key) for key in RUNTIME_ENV}}


def capture_inputs(study_path: Path, repository: Path | None = None) -> dict:
    """Capture actual deterministic bindings, including a second source check.

    Installed package versions/devices/options are recorded; this is not a hash
    inventory of every installed binary. No atmosphere, timing or cohort is run.
    """
    repository = Path(repository or ROOT).resolve()
    source = _source(repository)
    path = Path(study_path)
    path = path if path.is_absolute() else repository / path
    path = path.resolve()
    if not path.is_relative_to(repository) or path.relative_to(repository).as_posix() not in source["files"]:
        raise ValueError("Study must be a tracked execution input")
    preliminary = load_config(path)
    base_path = (repository / preliminary["base_config"]).resolve()
    if not base_path.is_relative_to(repository) or base_path.relative_to(repository).as_posix() not in source["files"]:
        raise ValueError("Base must be a tracked execution input")
    study = load_study(path, repository)
    base = load_config(base_path)
    runtime = _runtime()
    if not runtime["machine"]["x64"]:
        raise ValueError("The study requires JAX x64")
    if _source(repository) != source:
        raise ValueError("Source changed during provenance capture")
    return {"study": study, "base": base, "source": source, "runtime": runtime,
            "provenance": {"source_commit": source["commit"],
                           "source_sha256": _digest(_bytes(source)),
                           "runtime_sha256": _digest(_bytes(runtime))}}


def _destination(ledger: StudyLedger, relative: str, suffix: str) -> Path:
    path = Path(relative)
    if (path.is_absolute() or path.as_posix() != relative or ".." in path.parts
            or len(path.parts) < 2 or path.parts[0] != "evidence" or path.suffix != suffix):
        raise ValueError(f"Require a canonical evidence/ path ending in {suffix}")
    target = ledger.root / path
    # Reject internal as well as external symlinks; never publish through aliases.
    for current in (target, *target.parents):
        if current == ledger.root:
            break
        if current.is_symlink():
            raise ValueError("Evidence publication cannot follow symlinks")
    if not target.resolve().is_relative_to(ledger.root / "evidence"):
        raise ValueError("Evidence path escapes the run")
    return target


def save_metadata(ledger: StudyLedger, relative: str, value) -> dict:
    """Publish strict, canonical JSON exclusively and return its ledger reference."""
    path = _destination(ledger, relative, ".json")
    _publish(path, value)
    return ledger.artifact(relative)


def open_study_run(root: Path, study_path: Path = Path("configs/controller_study_v1.yaml"),
                   *, create: bool = False, min_free_bytes: int = 512 * 1024**2):
    """Create/resume against the actual loaded package's checkout and runtime.

    Returns (ledger, base, shared provenance references). Pass those references
    to bind_training with the pilot/training evidence. Missing provenance files
    after an interrupted creation can be republished only from identical bindings.
    A changed saved file is never repaired by replacement. A partial directory
    without a manifest needs inspection; it is not silently deleted or reused.
    """
    if type(min_free_bytes) is not int or min_free_bytes < 1:
        raise ValueError("Require a positive free-space budget")
    root = Path(root).resolve()
    ancestor = root
    while not ancestor.exists():
        ancestor = ancestor.parent
    if shutil.disk_usage(ancestor).free < min_free_bytes:
        raise RuntimeError("Insufficient free disk for the declared study budget")
    inputs = capture_inputs(study_path, ROOT)
    factory = StudyLedger.create if create else StudyLedger
    ledger = factory(root, inputs["study"], inputs["provenance"])
    with ledger._locked():
        refs = [save_metadata(ledger, f"evidence/provenance/{name}.json", inputs[name])
                for name in ("source", "runtime", "study", "base")]
    return ledger, inputs["base"], refs


def _numeric_arrays(arrays: dict) -> dict:
    if not isinstance(arrays, dict) or not arrays:
        raise ValueError("Require a nonempty mapping of numeric arrays")
    result = {}
    for key, value in arrays.items():
        if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", key):
            raise ValueError("Array keys must be simple identifiers")
        array = np.asarray(value)
        if array.dtype.kind not in "biufc":
            raise ValueError("Only numeric/boolean arrays without pickle are supported")
        # Preserve scalar shape, dtype and nonfinite failures; detach mutable input.
        result[key] = np.array(array, copy=True, order="C")
    return result


def save_arrays(ledger: StudyLedger, relative: str, arrays: dict) -> dict:
    """Stream a deterministic NPZ, flush, then publish without replacement.

    Archive order, timestamps and compression are fixed for byte-identical retries
    under the bound runtime. Interrupted .pending-* files never count as artifacts.
    No ledger lock is acquired: this may be called inside evaluate_chunk's lock.
    """
    path = _destination(ledger, relative, ".npz")
    values = _numeric_arrays(arrays)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "w+b") as stream:
            with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for key in sorted(values):
                    entry = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                    entry.compress_type = zipfile.ZIP_DEFLATED
                    entry.external_attr = 0o600 << 16
                    with archive.open(entry, "w", force_zip64=True) as member:
                        np.lib.format.write_array(member, values[key], allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.stat().st_size != Path(temporary).stat().st_size or _hash_file(path) != _hash_file(Path(temporary)):
                raise ValueError(f"Refusing to replace existing evidence: {relative}") from None
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return ledger.artifact(relative)


def load_arrays(ledger: StudyLedger, reference: dict) -> dict:
    """Verify the artifact hash before decoding numeric arrays without pickle."""
    path = _destination(ledger, reference["path"], ".npz")
    ledger._verify_artifacts([reference])
    with np.load(path, allow_pickle=False) as archive:
        if len(archive.files) != len(set(archive.files)):
            raise ValueError("Duplicate array keys in evidence")
        return _numeric_arrays({key: archive[key] for key in archive.files})


def check_provenance_binding(ledger, base, references):
    """Check persisted provenance/configuration before admitted numerical work."""
    required = {f"evidence/provenance/{name}.json" for name in ("source", "runtime", "study", "base")}
    refs = {ref["path"]: ref for ref in references}
    if not required.issubset(refs):
        raise ValueError("Training must bind all source/runtime/study/base evidence")
    for path in required:
        ledger._verify_artifacts([refs[path]])
    prefix = "evidence/provenance/"
    if (refs[prefix + "source.json"]["sha256"] != ledger.provenance["source_sha256"]
            or refs[prefix + "runtime.json"]["sha256"] != ledger.provenance["runtime_sha256"]):
        raise ValueError("Stored provenance differs from the ledger binding")
    source = _read(ledger.root / (prefix + "source.json"))
    if source["commit"] != ledger.provenance["source_commit"]:
        raise ValueError("Stored source commit differs from the ledger binding")
    if (_bytes(_read(ledger.root / (prefix + "study.json"))) != _bytes(ledger.config)
            or _bytes(_read(ledger.root / (prefix + "base.json"))) != _bytes(base)):
        raise ValueError("Study configuration differs from frozen evidence")
