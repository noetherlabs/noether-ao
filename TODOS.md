# Next work

M0–M7 are complete; their negative findings remain part of the baseline. Handle
new CI failures and actionable owner feedback before the research tasks below.
Complete one bounded task per autonomous run. Do not create speculative busywork
if all actionable tasks are done or require new evidence unavailable locally.

## P1 — Execute the predeclared controller study

The [protocol](docs/controller_study_v1.md), [inputs](configs/controller_study_v1.yaml)
and [unexecuted plan](plans/controller_study_v1.json) are predeclared. First implement
separate training/selection/evaluation stages with a frozen-selection barrier,
measured pseudo-open-loop predictor telemetry, complete trial-identity checks and
resumable raw recording. Verify on separately labeled smoke seeds without using
the reserved study cohorts. Commit that runner before the full study.

Then run the fixed candidate budgets in bounded resumable chunks. Include every
predeclared regime, failed candidates, paired uncertainty and resource costs. Do
not use old M3–M6 outcomes or reserved evaluation outcomes to retune candidates,
training budgets or gates. If no candidate is eligible, retain the failed selection
and withhold that controller's evaluation as the protocol requires.

Acceptance: provenance-backed comparison after equal validation effort; negative
or uncertain results are valid. Keep Gaussian-training/Poisson-evaluation limits
explicit and do not silently replace the original claim tables.

## Completed — Fair comparison protocol and seed checks

Five controllers have six meaningful candidates each. The predictor uses horizon
and ridge, since its existing command ignores gain. Training, validation and
evaluation cohorts and initialization are separated, including derived detector
keys and legacy exclusions. The base simulation configuration is hash-pinned.
The deterministic plan records 600 validation trials and at most 600 evaluation
trials as budgets only; no training or controller outcomes have been generated.

## P2 — Optical-gain drift observability

M4 misses gain drift and produces harmful vibration fallback. After independent
sensor validation, investigate a separately calibrated observable or detector on
new training/validation cohorts. Freeze thresholds before new held-out fault runs.

Acceptance: clean-run and frame false alarms, pre-onset alarms, missed detections,
detected-only delay, and paired controller error all reported. Useful detection
does not establish a safe fallback or a general safety guarantee.

## Completed — Independent ideal-pyramid validation

The [predeclared protocol](docs/pyramid_validation.md) passed on grids 32/64/96:
36 intensity cases, 30 derivative cases and 24 deliberate-error controls. The
reference uses HCIPy matrix integrals and a physical thin surface. See the
[saved report](results/pyramid_validation_v1/report.md), raw arrays and provenance.
All nine generated report files reproduce byte-for-byte; protected raw inputs
are unchanged. The core suite has 42 passing tests.

This validates the matched finite-grid ideal sensor contract. Native HCIPy
defaults, focal-support convergence and instrument validation remain outside the
claim. No M0–M7 production model, stored evidence or scientific verdict changed.

## Completed publication preparation

- Preserved milestone commits and documented the history; delivery is commits only.
- Added pinned correctness-only CI and regression checks for its timing boundary.
- Made the Dragrace reconstruction command independent of the checkout name.
