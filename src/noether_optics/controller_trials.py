"""Durable trial adapters invoked only through the study ledger's stage gates.

The public entry points construct a lazy adapter. Calibration, models and phases
are first accessed inside its admitted callback. Training orchestration, resource
cost measurement and complete study commands remain separate responsibilities.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import shutil

from .controller_evidence import load_arrays, save_arrays, save_metadata
from .controller_selection import _bytes, _case_id, _read


def validate_trials(ledger, base: dict, limit: int) -> dict:
    return ledger.validate_chunk(_TrialExecutor(ledger, base), limit)


def evaluate_trials(ledger, base: dict, limit: int) -> dict:
    return ledger.evaluate_chunk(_TrialExecutor(ledger, base), limit)


class _TrialExecutor:
    def __init__(self, ledger, base):
        self.ledger = ledger
        self.base = deepcopy(base)
        self.calibration = None
        self.kernels = {}

    def _check_provenance(self, training):
        required = {f"evidence/provenance/{name}.json" for name in ("source", "runtime", "study", "base")}
        refs = {ref["path"]: ref for ref in training["evidence"]}
        if not required.issubset(refs):
            raise ValueError("Training must bind all source/runtime/study/base evidence")
        for path in required:
            self.ledger._verify_artifacts([refs[path]])
        prefix = "evidence/provenance/"
        if (refs[prefix + "source.json"]["sha256"] != self.ledger.provenance["source_sha256"]
                or refs[prefix + "runtime.json"]["sha256"] != self.ledger.provenance["runtime_sha256"]):
            raise ValueError("Stored provenance differs from the ledger binding")
        source = _read(self.ledger.root / (prefix + "source.json"))
        if source["commit"] != self.ledger.provenance["source_commit"]:
            raise ValueError("Stored source commit differs from the ledger binding")
        if (_bytes(_read(self.ledger.root / (prefix + "study.json"))) != _bytes(self.ledger.config)
                or _bytes(_read(self.ledger.root / (prefix + "base.json"))) != _bytes(self.base)):
            raise ValueError("Trial configuration differs from frozen evidence")

    def _shared(self, kind, identity, build):
        relative = f"evidence/{identity['stage']}/{kind}/{_case_id(identity)}"
        metadata_path = self.ledger.root / (relative + ".json")
        if metadata_path.exists():
            saved = _read(metadata_path)
            if (set(saved) != {"identity", "raw"} or _bytes(saved["identity"]) != _bytes(identity)
                    or saved["raw"]["path"] != relative + ".npz"):
                raise ValueError("Shared physical evidence identity changed")
            arrays = load_arrays(self.ledger, saved["raw"])
            refs = [saved["raw"], self.ledger.artifact(relative + ".json")]
        else:
            arrays = build()
            raw = save_arrays(self.ledger, relative + ".npz", arrays)
            metadata = save_metadata(self.ledger, relative + ".json", {"identity": identity, "raw": raw})
            refs = [raw, metadata]
        return arrays, refs

    def __call__(self, case, training):
        self._check_provenance(training)
        candidate = case["candidate"]
        model = training["models"].get(candidate["id"])
        if model is not None and model["status"] == "failed":
            return {"case": case, "status": "failed", "metrics": None,
                    "artifacts": model["artifacts"], "open_loop": None,
                    "error": "Training failed: " + model["error"]}

        # These imports and all numerical construction are downstream of the gate.
        import numpy as np
        from .atmosphere import phase_sequence
        from .controller_physics import controller_kernel, episode_config, paired_metrics, paired_open_loop
        from .simulation import calibration

        cfg = episode_config(self.base, self.ledger.config, case["stage"],
                             case["photons"], case["wind_scale"], candidate)
        # Full observations/predictions dominate raw volume. This is a per-trial
        # conservative floor, not a claim that the entire study fits on disk.
        estimate = cfg["frames"] * (cfg["n"]**2 + 2 * cfg["detector_n"]**2 + 6 * cfg["modes"] + 16) * 8
        if shutil.disk_usage(self.ledger.root).free < max(512 * 1024**2, 2 * estimate):
            raise RuntimeError("Insufficient free disk before trial execution")
        predictor, params = None, None
        if candidate["controller"] in ("predictive", "learned"):
            if model is None:
                raise ValueError("Missing frozen model")
            refs = [ref for ref in model["artifacts"] if Path(ref["path"]).name == "parameters.npz"]
            if len(refs) != 1:
                raise ValueError("Fitted models require exactly one parameters.npz artifact")
            frozen = load_arrays(self.ledger, refs[0])
            if candidate["controller"] == "predictive":
                if set(frozen) != {"coefficients"}:
                    raise ValueError("Invalid frozen predictor schema")
                predictor = frozen["coefficients"]
                if predictor.shape != (cfg["modes"], 2) or not np.isfinite(predictor).all():
                    raise ValueError("Invalid frozen predictor values")
            else:
                shapes = ((3, 8), (8,), (8, 1), (1,))
                if set(frozen) != {f"parameter_{i}" for i in range(4)}:
                    raise ValueError("Invalid frozen learned schema")
                params = tuple(frozen[f"parameter_{i}"] for i in range(4))
                if any(p.shape != shape or not np.isfinite(p).all() for p, shape in zip(params, shapes)):
                    raise ValueError("Invalid frozen learned values")
        if self.calibration is None:
            self.calibration = calibration(self.base)
        cal = self.calibration
        atmosphere_id = {"stage": case["stage"], "seed": case["seed"], "wind_scale": case["wind_scale"]}
        phase, phase_refs = self._shared("atmospheres", atmosphere_id,
                                        lambda: {"phase": phase_sequence(cfg, case["seed"])})
        if set(phase) != {"phase"} or phase["phase"].shape != (cfg["frames"], cfg["n"], cfg["n"]):
            raise ValueError("Invalid shared atmosphere shape")
        condition = {key: value for key, value in case.items() if key != "candidate"}
        reference, reference_refs = self._shared("open_loop", condition,
            lambda: paired_open_loop(phase["phase"], cfg, cal, case["fault"]))
        kernel_key = (candidate["id"], case["stage"], case["photons"], case["wind_scale"], case["fault"])
        if kernel_key not in self.kernels:
            self.kernels[kernel_key] = controller_kernel(cfg, cal, candidate["controller"], case["fault"], predictor)
        outputs = self.kernels[kernel_key](phase["phase"], case["seed"], params)
        relative = f"evidence/{case['stage']}/trials/{_case_id(case)}"
        raw = save_arrays(self.ledger, relative + ".npz", {key: np.asarray(value) for key, value in outputs.items()})
        summaries = paired_metrics(outputs, reference, cfg)
        detail = save_metadata(self.ledger, relative + ".json", {
            "case": case, "config": cfg, "raw": raw, "atmosphere": phase_refs,
            "open_loop": reference_refs, "model": model, **summaries})
        # Nonfinite physical outputs remain completed raw acquisitions with a false
        # finite flag/null metrics. The selection gate rejects them; no data vanish.
        return {"case": case, "status": "completed", "metrics": summaries["metrics"],
                "artifacts": [raw, detail, *phase_refs, reference_refs[1]],
                "open_loop": reference_refs[0], "error": None}
