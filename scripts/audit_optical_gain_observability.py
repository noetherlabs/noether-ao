"""Rebuild a read-only optical-gain observability audit from retained M4 data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


WINDOW_FRAMES = 4
LOWER_QUANTILE = 0.005
SCALE_QUANTILES = (0.1, 0.5, 0.9)


def _json(path: Path) -> Any:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _within(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise ValueError(f"Artifact escapes source directory: {relative}")
    return path


def causal_mean(score: np.ndarray, burn_in: int, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Return frame identities and full causal windows after burn-in."""
    values = np.asarray(score, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Score must be a finite vector")
    if not isinstance(burn_in, int) or not isinstance(window, int) or burn_in < 0 or window < 1:
        raise ValueError("Burn-in and window must be nonnegative/positive integers")
    if burn_in + window > len(values):
        raise ValueError("Score is too short for a full post-burn-in window")
    statistic = np.convolve(values[burn_in:], np.ones(window) / window, mode="valid")
    frames = np.arange(burn_in + window - 1, len(values), dtype=int)
    return frames, statistic


def _score(path: Path, frames: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        if "score" not in data.files:
            raise ValueError(f"Missing score array: {path}")
        score = np.asarray(data["score"], dtype=float)
    if score.shape != (frames,) or not np.isfinite(score).all():
        raise ValueError(f"Invalid score array: {path}")
    return score


def _scale_summary(path: Path, start: int, stop: int) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        required = {"measurement", "prediction", "score", "variance"}
        if not required.issubset(data.files):
            raise ValueError(f"Full WFS arrays are missing from {path}")
        measurement = np.asarray(data["measurement"], dtype=float)
        prediction = np.asarray(data["prediction"], dtype=float)
        score = np.asarray(data["score"], dtype=float)
        variance = np.asarray(data["variance"], dtype=float)
    if measurement.shape != prediction.shape or measurement.ndim != 2:
        raise ValueError(f"Mismatched WFS arrays in {path}")
    if score.shape != (len(measurement),) or variance.shape != (len(measurement),):
        raise ValueError(f"Mismatched scalar outputs in {path}")
    if not (0 <= start < stop <= len(measurement)):
        raise ValueError(f"Invalid response-scale interval for {path}")
    if not all(np.isfinite(value).all() for value in (measurement, prediction, score, variance)):
        raise ValueError(f"Nonfinite full outputs in {path}")
    reference = prediction[0]
    expected = prediction - reference
    observed = measurement - reference
    denominator = np.sum(expected * expected, axis=1)
    numerator = np.sum(expected * observed, axis=1)
    scale = np.divide(
        numerator,
        denominator,
        out=np.full_like(denominator, np.nan),
        where=denominator > np.finfo(float).eps,
    )
    selected = scale[start:stop]
    selected = selected[np.isfinite(selected)]
    if len(selected) != stop - start:
        raise ValueError(f"Undefined response scale in requested interval for {path}")
    q10, median, q90 = np.quantile(selected, SCALE_QUANTILES)
    return {
        "frames": [start, stop],
        "response_scale_q10": float(q10),
        "response_scale_median": float(median),
        "response_scale_q90": float(q90),
        "score_median": float(np.median(score[start:stop])),
        "residual_variance_mean_rad2": float(np.mean(variance[start:stop])),
    }


def build_audit(source: Path) -> dict[str, Any]:
    """Derive the exploratory result without running the optical simulator."""
    source = source.resolve()
    fixed_names = ("config.json", "detector.json", "index.json", "summary.json", "provenance.json")
    config, detector, index, summary, provenance = (_json(source / name) for name in fixed_names)
    if detector["training_seeds"] != config["training_seeds"]:
        raise ValueError("Detector and configuration training seeds differ")
    frames = config["frames"]
    burn_in = config["burn_in"]
    onset = config["faults"]["onset"]
    end = min(config["faults"]["end"], frames)
    settled = onset + config["faults"]["drift_frames"]
    if not (0 <= burn_in < onset < settled < end <= frames):
        raise ValueError("M4 frame intervals are not ordered")

    used = {source / name for name in fixed_names}
    training_statistics: list[float] = []
    training_scores: list[float] = []
    for seed in detector["training_seeds"]:
        path = source / "raw" / f"calibration_{seed}.npz"
        used.add(path)
        score = _score(path, frames)
        training_scores.extend(score[burn_in:].tolist())
        _, statistic = causal_mean(score, burn_in, WINDOW_FRAMES)
        training_statistics.extend(statistic.tolist())
    lower_threshold = float(np.quantile(training_statistics, LOWER_QUANTILE))
    rebuilt_upper_threshold = float(np.quantile(training_scores, detector["quantile"]))
    if not np.isclose(rebuilt_upper_threshold, detector["threshold"], rtol=0, atol=1e-12):
        raise ValueError("Stored upper threshold does not reproduce from calibration scores")

    labels_seen: set[tuple[str, bool, int]] = set()
    records: dict[str, list[tuple[int, Path]]] = {}
    full_records: dict[tuple[str, int], Path] = {}
    for entry in index:
        if entry.get("config") != config:
            raise ValueError(f"Record configuration differs: {entry.get('file')}")
        labels = entry["labels"]
        identity = (labels["fault"], labels["fallback_enabled"], labels["seed"])
        if identity in labels_seen:
            raise ValueError(f"Duplicate M4 identity: {identity}")
        labels_seen.add(identity)
        path = _within(source, entry["file"])
        if labels["fallback_enabled"]:
            continue
        used.add(path)
        records.setdefault(labels["fault"], []).append((labels["seed"], path))
        if labels["seed"] == config["seeds"][0]:
            full_records[(labels["fault"], labels["seed"])] = path

    expected_seeds = sorted(config["seeds"])
    for fault, fault_records in records.items():
        if sorted(seed for seed, _ in fault_records) != expected_seeds:
            raise ValueError(f"Incomplete no-fallback cohort for {fault}")

    fault_screen: list[dict[str, Any]] = []
    clean_flags: list[bool] = []
    clean_run_alarms = 0
    for fault, fault_records in records.items():
        pre_onset = 0
        delays: list[int] = []
        for _, path in sorted(fault_records):
            score = _score(path, frames)
            frame_ids, statistic = causal_mean(score, burn_in, WINDOW_FRAMES)
            alarms = statistic < lower_threshold
            if fault == "none":
                clean_flags.extend(alarms.tolist())
                clean_run_alarms += int(np.any(alarms))
                continue
            pre_onset += int(np.any(alarms[frame_ids < onset]))
            detected = frame_ids[(frame_ids >= onset) & (frame_ids < end) & alarms]
            if len(detected):
                delays.append(int(detected[0] - onset))
        if fault != "none":
            fault_screen.append({
                "fault": fault,
                "runs": len(fault_records),
                "pre_onset_alarm_runs": pre_onset,
                "detected_runs": len(delays),
                "detection_delay_frames": delays,
            })

    clean_records = records.get("none", [])
    if not clean_records or not clean_flags:
        raise ValueError("No clean evaluation records")
    summary_by_fault = {row["fault"]: row for row in summary}
    if len(summary_by_fault) != len(summary):
        raise ValueError("Duplicate fault summary")
    legacy_gain = summary_by_fault["optical_gain"]
    first_seed = config["seeds"][0]
    clean_path = full_records[("none", first_seed)]
    gain_path = full_records[("optical_gain", first_seed)]
    clean_pre = _scale_summary(clean_path, burn_in, onset)
    clean_settled = _scale_summary(clean_path, settled, end)
    gain_pre = _scale_summary(gain_path, burn_in, onset)
    gain_settled = _scale_summary(gain_path, settled, end)

    manifest = {str(path.relative_to(source)): _sha256(path) for path in sorted(used)}
    manifest_text = "".join(f"{digest}  {name}\n" for name, digest in manifest.items())
    return {
        "schema_version": 1,
        "kind": "legacy_optical_gain_observability_audit",
        "source": {
            "directory": "results/m4",
            "git_commit": provenance["git_commit"],
            "input_files": len(manifest),
            "input_manifest_sha256": hashlib.sha256(manifest_text.encode()).hexdigest(),
            "sha256": manifest,
        },
        "scope": {
            "status": "exploratory_existing_evidence_only",
            "legacy_m4_evaluation_reclassified_as_p2_development": True,
            "eligible_for_future_p2_evaluation": False,
            "new_simulation_run": False,
            "new_seed_access": False,
            "timing_measurement": False,
            "control_action_evaluated": False,
        },
        "legacy_upper_tail_detector": {
            "calibration_quantile": detector["quantile"],
            "threshold": detector["threshold"],
            "rebuilt_threshold": rebuilt_upper_threshold,
            "threshold_reproduces": True,
            "optical_gain_detected_without_prior_alarm": legacy_gain["detected_without_prior_alarm"],
            "optical_gain_runs": legacy_gain["total_runs"],
        },
        "exploratory_lower_tail_candidate": {
            "statistic": "causal rolling mean of the existing whitened residual score",
            "tail": "lower",
            "window_frames": WINDOW_FRAMES,
            "calibration_quantile": LOWER_QUANTILE,
            "numpy_quantile_method": "linear",
            "calibration_windows": len(training_statistics),
            "threshold": lower_threshold,
            "clean_evaluation": {
                "runs": len(clean_records),
                "alarm_runs": clean_run_alarms,
                "evaluated_windows": len(clean_flags),
                "alarm_windows": int(np.count_nonzero(clean_flags)),
                "window_false_alarm_fraction": float(np.mean(clean_flags)),
            },
            "fault_screen": fault_screen,
        },
        "passive_response_scale": {
            "definition": "dot(prediction-reference, measurement-reference) / squared_norm(prediction-reference)",
            "reference": "prediction[0], fixed by the saved loop's zero initial prediction and command",
            "full_frame_seed": first_seed,
            "quantiles": list(SCALE_QUANTILES),
            "clean": {"pre_fault_interval": clean_pre, "settled_interval": clean_settled},
            "optical_gain": {"pre_fault_interval": gain_pre, "settled_interval": gain_settled},
            "settled_response_median_relative_change": (
                gain_settled["response_scale_median"] / clean_settled["response_scale_median"] - 1
            ),
            "settled_score_median_ratio": (
                gain_settled["score_median"] / clean_settled["score_median"]
            ),
            "settled_residual_variance_ratio": (
                gain_settled["residual_variance_mean_rad2"]
                / clean_settled["residual_variance_mean_rad2"]
            ),
        },
        "interpretation": [
            "The original upper-tail detector missed optical-gain drift in every retained M4 run.",
            "In this legacy development screen, a four-frame lower-tail score detected all five optical-gain runs with no pre-onset alarm in those runs, while one of five clean runs alarmed.",
            "The one seed with saved full WFS frames shows why passive scale regression is insufficient: its settled median stayed near one while the paired clean-to-gain residual variance increased substantially.",
            "These observations select a candidate for fresh predeclared training and validation; they do not validate a detector, recovery action, instrument behavior or safety claim.",
        ],
        "next_gate": (
            "Predeclare disjoint P2 training, validation and evaluation streams; calibrate the lower tail only on training, "
            "freeze all thresholds after validation, and keep fallback disabled while observability is assessed."
        ),
        "primary_sources": [
            {
                "title": "A telescope-ready approach for modal compensation of pyramid wavefront sensor optical gain",
                "url": "https://doi.org/10.1051/0004-6361/201935847",
                "relevance": "Independent modal dithers can identify optical gain and update a reconstructor.",
            },
            {
                "title": "Pyramid wavefront sensor optical gains compensation using a convolutional model",
                "url": "https://arxiv.org/abs/2006.08294",
                "relevance": "Residual-wavefront statistics and AO telemetry can support non-invasive gain estimation.",
            },
            {
                "title": "The focal-plane assisted pyramid wavefront sensor",
                "url": "https://arxiv.org/abs/2103.02297",
                "relevance": "An independent focal-plane channel can track optical gain frame by frame.",
            },
        ],
    }


def _serialized(value: Any) -> str:
    return json.dumps(value, indent=2, allow_nan=False) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("results/m4"))
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--out", type=Path)
    destination.add_argument("--check", type=Path)
    args = parser.parse_args()
    report = build_audit(args.source)
    if args.check:
        if args.check.read_text() != _serialized(report):
            raise SystemExit(f"Audit drift: {args.check}")
        print(f"Optical-gain observability audit reproduces: {args.check}")
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(_serialized(report))
    temporary.replace(args.out)
    print(f"Wrote optical-gain observability audit: {args.out}")


if __name__ == "__main__":
    main()
