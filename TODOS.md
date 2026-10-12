# Next work

M0–M7 are complete; their negative findings remain part of the baseline. Handle
new CI failures and actionable review feedback before the research tasks below.
Complete one bounded task per autonomous run. Do not create speculative busywork
if all actionable tasks are done or require new evidence unavailable locally.

## P1 — Independent pyramid sensor validation

The current HCIPy check covers the open-loop PSF, while plant and twin share sensor
code. Build an independent ideal, unmodulated pyramid reference after inspecting
the pinned HCIPy API. First document sampling, phase sign, pupil layout, centering,
wavelength and normalization conventions; then compare flat and single-mode
responses and small-signal derivatives at multiple grid sizes.

Acceptance: predeclared tolerances and deterministic tests that can catch sign,
scale and layout errors, plus a saved comparison report. If conventions cannot be
matched, retain the discrepancy and narrow the claim. Do not change baseline M3
results or assert instrument validation.

## P1 — Fair controller validation protocol

The present controls use fixed gains, short runs and ideal predictor telemetry.
Specify a bounded gain/horizon search on new training and validation seeds, with
an equal search budget per controller and fresh held-out seeds reserved in advance.
First land the protocol/configuration and seed-split checks; then implement and
run it in a separate PR. Include all regimes selected before evaluation, failure
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

## Completed publication preparation

- Preserved milestone review boundaries and documented the PR dependency order.
- Added pinned correctness-only CI and regression checks for its timing boundary.
- Made the Dragrace reconstruction command independent of the checkout name.
