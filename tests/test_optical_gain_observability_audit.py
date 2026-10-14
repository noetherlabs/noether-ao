from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_optical_gain_observability", ROOT / "scripts/audit_optical_gain_observability.py"
)
assert SPEC and SPEC.loader
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def test_causal_mean_uses_only_full_post_burn_in_windows() -> None:
    frames, values = AUDIT.causal_mean(np.arange(8, dtype=float), burn_in=2, window=3)
    np.testing.assert_array_equal(frames, [4, 5, 6, 7])
    np.testing.assert_allclose(values, [3, 4, 5, 6])
    with pytest.raises(ValueError, match="finite vector"):
        AUDIT.causal_mean(np.array([1.0, np.nan]), burn_in=0, window=1)
    with pytest.raises(ValueError, match="too short"):
        AUDIT.causal_mean(np.arange(3, dtype=float), burn_in=2, window=2)


def test_persistence_marks_streak_endpoints_and_rejects_invalid_inputs() -> None:
    crossings = np.array([False, True, True, True, False, True, True, True, True])
    np.testing.assert_array_equal(
        AUDIT.persistent_alarm(crossings, 3),
        [False, False, False, True, False, False, False, True, True],
    )
    assert AUDIT.maximum_true_run(crossings) == 4
    with pytest.raises(ValueError, match="boolean vector"):
        AUDIT.persistent_alarm(np.array([0, 1]), 2)
    with pytest.raises(ValueError, match="positive integer"):
        AUDIT.persistent_alarm(crossings, 0)
    with pytest.raises(ValueError, match="too short"):
        AUDIT.persistent_alarm(crossings, len(crossings) + 1)


def test_committed_audit_rebuilds_from_protected_m4_inputs() -> None:
    expected = json.loads(
        (ROOT / "results/optical_gain_observability_audit_v1/audit.json").read_text()
    )
    actual = AUDIT.build_audit(ROOT / "results/m4")
    assert actual == expected
    assert AUDIT._serialized(actual) == (
        ROOT / "results/optical_gain_observability_audit_v1/audit.json"
    ).read_text()
    candidate = actual["exploratory_lower_tail_candidate"]
    gain = next(row for row in candidate["fault_screen"] if row["fault"] == "optical_gain")
    assert gain["detected_runs"] == gain["runs"] == 5
    assert gain["pre_onset_alarm_runs"] == 0
    persistence = candidate["development_persistence_screen"]
    persisted_gain = next(
        row for row in persistence["fault_screen"] if row["fault"] == "optical_gain"
    )
    assert persistence["clean_evaluation"]["alarm_runs"] == 0
    assert persisted_gain["detected_runs"] == persisted_gain["runs"] == 5
    assert persisted_gain["pre_onset_alarm_runs"] == 0
    assert persisted_gain["detection_delay_frames"] == [17, 16, 15, 13, 15]
    assert actual["scope"]["eligible_for_future_p2_evaluation"] is False
