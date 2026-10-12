# Next work

M0–M7 are complete; their negative findings remain part of the baseline. Handle
new CI failures and actionable owner feedback before the research tasks below.
Complete one bounded task per autonomous run. Do not create speculative busywork
if all actionable tasks are done or require new evidence unavailable locally.

## P1 — Fair controller validation protocol

The present controls use fixed gains, short runs and ideal predictor telemetry.
Specify a bounded gain/horizon search on new training and validation seeds, with
an equal search budget per controller and fresh held-out seeds reserved in advance.
First commit the protocol/configuration and seed-split checks; then implement and
run it in a separate commit. Include all regimes selected before evaluation, failure
rates, paired uncertainty and compute cost. No tuning based on old M3–M6 outcomes.

Acceptance: provenance-backed comparison after equal validation effort; negative
or uncertain results are valid. Keep Gaussian-training/Poisson-evaluation limits
explicit and do not silently replace the original claim tables.

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
