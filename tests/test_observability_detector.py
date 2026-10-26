from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from noether_optics.observability_detector import (
    _build_threshold_bundle,
    causal_mean,
    detector_trace,
    freeze_thresholds,
    training_cases,
)
from noether_optics.observability_protocol import load_protocol


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/optical_gain_observability_v1.yaml"
PROVENANCE = {
    "source_commit": "a" * 40,
    "source_sha256": "b" * 64,
    "runtime_sha256": "c" * 64,
}


@pytest.fixture
def protocol() -> dict:
    return load_protocol(CONFIG)


@pytest.fixture
def protocol_sha256() -> str:
    return hashlib.sha256(CONFIG.read_bytes()).hexdigest()


def _records(protocol: dict) -> list[dict]:
    records = []
    for index, case in enumerate(training_cases(protocol)):
        regime = int(case["regime_id"].split("-")[-1])
        seed_offset = case["atmosphere_seed"] - protocol["seeds"]["training"][0]
        score = 5.0 + regime + seed_offset / 100 + np.arange(512, dtype=np.float64) / 1000
        records.append(
            {
                "case": case,
                "status": "completed",
                "artifact": {
                    "path": f"evidence/training/trial-{index:03d}.npz",
                    "sha256": hashlib.sha256(f"synthetic raw {index}".encode()).hexdigest(),
                },
                "score": score,
                "error": None,
            }
        )
    return records


def _materialize(root: Path, records: list[dict]) -> None:
    directory = root / "evidence/training"
    directory.mkdir(parents=True)
    for record in records:
        path = root / record["artifact"]["path"]
        np.savez(path, score=record["score"])
        record["artifact"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()


def test_causal_detector_alignment_is_strict_and_has_no_future_samples(protocol) -> None:
    score = np.full(512, 10.0)
    score[150:175] = 0.0
    trace = detector_trace(protocol, score, 5.0)
    np.testing.assert_array_equal(trace["statistic_frame_id"], np.arange(131, 512))
    np.testing.assert_array_equal(trace["alarm_frame_id"], np.arange(138, 512))
    assert trace["lower_statistic"].shape == trace["raw_crossing"].shape == (381,)
    assert trace["p2_alarm"].shape == (374,)
    first_crossing = int(np.flatnonzero(trace["raw_crossing"])[0])
    first_alarm = int(np.flatnonzero(trace["p2_alarm"])[0])
    assert trace["alarm_frame_id"][first_alarm] == trace["statistic_frame_id"][first_crossing + 7]

    changed = score.copy()
    changed[200:] = 1000
    original_frames, original = causal_mean(score, 128, 4)
    changed_frames, altered = causal_mean(changed, 128, 4)
    np.testing.assert_array_equal(original_frames, changed_frames)
    np.testing.assert_array_equal(original[original_frames < 200], altered[changed_frames < 200])

    equality = detector_trace(protocol, np.full(512, 5.0), 5.0)
    assert not equality["raw_crossing"].any()


@pytest.mark.parametrize(
    "damage",
    ["shape", "nonfinite", "negative", "complex", "boolean", "threshold", "threshold_boolean"],
)
def test_detector_rejects_invalid_scores_and_thresholds(protocol, damage) -> None:
    score = np.ones(512)
    threshold: float | bool = 1.0
    if damage == "shape":
        score = score[:-1]
    elif damage == "nonfinite":
        score[0] = np.nan
    elif damage == "negative":
        score[0] = -1
    elif damage == "complex":
        score = score.astype(complex)
    elif damage == "boolean":
        score = score.astype(bool)
    elif damage == "threshold":
        threshold = float("nan")
    else:
        threshold = True
    with pytest.raises(ValueError, match="Score|Threshold"):
        detector_trace(protocol, score, threshold)


def test_training_cases_are_complete_clean_and_use_only_training_streams(protocol) -> None:
    cases = training_cases(protocol)
    assert len(cases) == 64
    assert {case["role"] for case in cases} == {"training"}
    assert {case["condition"] for case in cases} == {"clean"}
    assert {case["regime_id"] for case in cases} == {
        "regime-00", "regime-01", "regime-02", "regime-03"
    }
    assert {case["atmosphere_seed"] for case in cases} == set(protocol["seeds"]["training"])
    assert not {case["atmosphere_seed"] for case in cases} & set(
        protocol["seeds"]["validation"] + protocol["seeds"]["evaluation"]
    )


def test_complete_training_freezes_deterministic_per_regime_thresholds(
    protocol, protocol_sha256
) -> None:
    records = _records(protocol)
    bundle = _build_threshold_bundle(
        protocol, protocol_sha256, PROVENANCE, list(reversed(records))
    )
    assert bundle == _build_threshold_bundle(protocol, protocol_sha256, PROVENANCE, records)
    assert bundle["status"] == "frozen_before_validation"
    assert bundle["calculation_dtype"] == "float64"
    assert bundle["validation_released"] is False
    assert len(bundle["training_records"]) == 64
    assert {len(row["id"]) for row in bundle["training_records"]} == {64}
    assert len(bundle["thresholds"]) == 4
    assert bundle["bindings"] == {
        "protocol_sha256": protocol_sha256,
        "base_config_sha256": protocol["base_config_sha256"],
        **PROVENANCE,
    }

    by_case = {json.dumps(row["case"], sort_keys=True): row for row in records}
    for threshold in bundle["thresholds"]:
        scores = [
            by_case[json.dumps(case, sort_keys=True)]["score"]
            for case in training_cases(protocol)
            if case["regime_id"] == threshold["regime"]["id"]
        ]
        windows = np.concatenate(
            [
                np.array(
                    [score[frame - 3 : frame + 1].mean() for frame in range(131, 512)]
                )
                for score in scores
            ]
        )
        expected = np.quantile(windows, 0.005, method="linear")
        assert threshold["threshold"] == expected
        assert threshold["calibration_records"] == 16
        assert threshold["calibration_windows"] == 6096
        assert len(threshold["training_record_ids"]) == 16


@pytest.mark.parametrize(
    "damage",
    ["missing", "duplicate", "foreign", "failed", "shared_artifact", "bad_path", "bad_hash"],
)
def test_threshold_freeze_rejects_incomplete_or_unbound_records(
    protocol, protocol_sha256, damage
) -> None:
    records = _records(protocol)
    if damage == "missing":
        records.pop()
    elif damage == "duplicate":
        records[-1] = deepcopy(records[0])
    elif damage == "foreign":
        records[0]["case"]["atmosphere_seed"] += 1_000_000
    elif damage == "failed":
        records[0].update(status="failed", score=None, error="synthetic numerical failure")
    elif damage == "shared_artifact":
        records[1]["artifact"] = records[0]["artifact"]
    elif damage == "bad_path":
        records[0]["artifact"]["path"] = "../outside.npz"
    else:
        records[0]["artifact"]["sha256"] = "not-a-digest"
    with pytest.raises(ValueError):
        _build_threshold_bundle(protocol, protocol_sha256, PROVENANCE, records)


@pytest.mark.parametrize("damage", ["shape", "nonfinite", "negative", "schema", "error"])
def test_threshold_freeze_rejects_invalid_completed_scores(
    protocol, protocol_sha256, damage
) -> None:
    records = _records(protocol)
    if damage == "shape":
        records[0]["score"] = records[0]["score"][:-1]
    elif damage == "nonfinite":
        records[0]["score"][0] = np.inf
    elif damage == "negative":
        records[0]["score"][0] = -1
    elif damage == "schema":
        records[0]["extra"] = "unexpected"
    else:
        records[0]["error"] = "completed with an impossible error"
    with pytest.raises(ValueError):
        _build_threshold_bundle(protocol, protocol_sha256, PROVENANCE, records)


def test_threshold_freeze_rejects_a_malformed_case(protocol, protocol_sha256) -> None:
    records = _records(protocol)
    records[0]["case"] = None
    with pytest.raises(ValueError, match="case schema"):
        _build_threshold_bundle(protocol, protocol_sha256, PROVENANCE, records)


def test_threshold_bindings_must_be_exact(protocol, protocol_sha256) -> None:
    records = _records(protocol)
    with pytest.raises(ValueError, match="protocol SHA256"):
        _build_threshold_bundle(protocol, "short", PROVENANCE, records)
    provenance = dict(PROVENANCE, runtime_sha256="short")
    with pytest.raises(ValueError, match="runtime_sha256"):
        _build_threshold_bundle(protocol, protocol_sha256, provenance, records)


def test_threshold_publication_is_exclusive_and_idempotent(protocol, tmp_path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    records = _records(protocol)
    _materialize(root, records)
    bundle, reference = freeze_thresholds(root, CONFIG, PROVENANCE, records)
    original = (root / "thresholds.json").read_bytes()
    assert bundle["bindings"]["protocol_sha256"] == hashlib.sha256(
        CONFIG.read_bytes()
    ).hexdigest()
    assert reference == {
        "path": "thresholds.json",
        "sha256": hashlib.sha256(original).hexdigest(),
    }
    assert json.loads(original) == bundle
    assert freeze_thresholds(root, CONFIG, PROVENANCE, records) == (
        bundle, reference
    )

    changed_provenance = dict(PROVENANCE, source_commit="d" * 40)
    with pytest.raises(ValueError, match="replace existing evidence"):
        freeze_thresholds(root, CONFIG, changed_provenance, records)
    assert (root / "thresholds.json").read_bytes() == original


@pytest.mark.parametrize(
    "damage", ["hash", "score", "missing_score", "pickle", "symlink"]
)
def test_threshold_publication_verifies_raw_score_artifacts(
    protocol, tmp_path, damage
) -> None:
    root = tmp_path / "run"
    root.mkdir()
    records = _records(protocol)
    _materialize(root, records)
    path = root / records[0]["artifact"]["path"]
    if damage == "hash":
        path.write_bytes(path.read_bytes() + b"drift")
    elif damage == "score":
        records[0]["score"] = records[0]["score"] + 1
    elif damage == "missing_score":
        np.savez(path, variance=np.ones(512))
        records[0]["artifact"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    elif damage == "pickle":
        np.savez(path, score=np.array([object()], dtype=object))
        records[0]["artifact"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    else:
        target = tmp_path / "elsewhere.npz"
        target.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(target)
    with pytest.raises(ValueError):
        freeze_thresholds(root, CONFIG, PROVENANCE, records)
    assert not (root / "thresholds.json").exists()


def test_threshold_publication_rejects_missing_or_symlinked_roots(
    protocol, tmp_path
) -> None:
    records = _records(protocol)
    missing = tmp_path / "missing"
    with pytest.raises(ValueError, match="existing non-symlink"):
        freeze_thresholds(missing, CONFIG, PROVENANCE, records)
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="existing non-symlink"):
        freeze_thresholds(alias, CONFIG, PROVENANCE, records)

    root = tmp_path / "run"
    root.mkdir()
    _materialize(root, records)
    target = tmp_path / "existing-threshold.json"
    target.write_text("protected\n")
    (root / "thresholds.json").symlink_to(target)
    with pytest.raises(ValueError, match="cannot replace a symlink"):
        freeze_thresholds(root, CONFIG, PROVENANCE, records)
    assert target.read_text() == "protected\n"
