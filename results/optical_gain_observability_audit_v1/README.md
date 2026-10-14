# Optical-gain observability audit

This is a read-only, exploratory reanalysis of the retained M4 fault evidence. It
does not rerun the simulator, access a new seed, test a recovery action or establish
detector performance. The M4 evaluation cohort is now development evidence for P2
and is ineligible for any future P2 held-out claim.

## Finding

The original detector alarms only on a large whitened frame-residual score. It
detected optical-gain drift in **0/5** retained runs. Under the simplified M4 fault,
the settled score moves in the other direction: in the one seed with complete WFS
frames, its median falls from **6.84190** in the paired clean run to **2.72924** under
gain drift.

A causal four-frame mean with a separately calibrated lower threshold is therefore a
plausible candidate. Using the original three clean calibration seeds and a 0.005
lower quantile, the exploratory threshold is **5.49682**. Applied to the old M4
evaluation cohort, it gives:

| Check | Exploratory result |
| --- | --- |
| Optical-gain detections | 5/5 |
| Pre-onset alarms in those runs | 0/5 |
| Detection delays | 6, 8, 8, 9 and 10 frames |
| Clean alarm runs | 1/5 |
| Clean alarm windows | 5/585 (0.8547%) |
| Other simulated faults crossing the lower threshold | 0/30 runs |

These numbers select a candidate; they are not validation results. The window and
tail were chosen after inspecting the old cohort. Both must be frozen before a new
validation cohort, and the final threshold must be calibrated only from fresh P2
training evidence.

## Why passive response regression is insufficient

For the single seed with saved full measurement and prediction frames, the audit
also regresses the observed WFS departure from its flat reference against the model
prediction. Once the injected drift has settled at a response multiplier of 0.35,
the regression median changes by only **−1.83%**, while paired residual phase
variance rises by a factor of **2.99**. Closed-loop compensation makes the passive
response scale look almost nominal while true error grows.

That is a one-seed diagnostic, not a general identifiability proof. It does rule out
treating this simple passive regression as adequate evidence. Primary literature
offers stronger independent excitation or sensing routes if the non-invasive
lower-tail candidate does not validate:

- [Deo et al. (2019)](https://doi.org/10.1051/0004-6361/201935847) estimate modal
  optical gains with small DM dithers and update the command matrix.
- [Chambouleyron et al. (2020)](https://arxiv.org/abs/2006.08294) use a convolutional
  model and residual-wavefront statistics derived from AO telemetry.
- [Chambouleyron et al. (2021)](https://arxiv.org/abs/2103.02297) add a focal-plane
  channel for frame-by-frame gain tracking.

The current ideal sensor has no physical modulation mechanism or independent focal
plane path. Implementing either route would require a separate, predeclared model and
cannot be inferred from this audit.

## Reproduction

[`audit.json`](audit.json) records all 48 input hashes, the exact statistic, every
fault-screen result and the source commit of the original M4 run.

```sh
.venv/bin/python scripts/audit_optical_gain_observability.py \
  --source results/m4 \
  --check results/optical_gain_observability_audit_v1/audit.json
.venv/bin/python -m pytest -q tests/test_optical_gain_observability_audit.py
```

The next gate is a written P2 protocol with disjoint training, validation and
evaluation streams, explicit run-level false-alarm criteria and fallback disabled.
No new fault outcome should be acquired until that protocol is frozen.
