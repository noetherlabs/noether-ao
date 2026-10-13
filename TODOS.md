# Next work

M0–M7 are complete; their negative findings remain part of the baseline. Handle
new CI failures and actionable owner feedback before the research tasks below.
Complete one bounded task per autonomous run. Do not create speculative busywork
if all actionable tasks are done or require new evidence unavailable locally.

## P1 — Execute the predeclared controller study

The [protocol](docs/controller_study_v1.md), [inputs](configs/controller_study_v1.yaml)
and [unexecuted plan](plans/controller_study_v1.json) are predeclared. The
[execution ledger and selection barrier](docs/controller_study_execution.md) now
check complete trial identities, training bindings, immutable evidence and resumed
evaluation chunks. Numerical helpers now provide measured pseudo-open-loop pilot
telemetry, separate per-episode predictor fits, exact-noise controller kernels
and paired raw-array scoring. Learned fitting now implements common initialization,
normalized equal-episode gradients, fixed Adam budgets and failure records. Next
connect the adapter and separate stage commands: durable pilot, training and trial
recording through the ledger. Fresh source/runtime capture and exclusive numeric
artifact storage are implemented; bind their returned provenance references with
the training evidence. Add durable mid-fit recording/resumption before long fits.
The separate component smoke has passed. The adapter must enter evaluation through
the ledger's lazy callback, after the gate.
Verify on separately labeled physical smoke seeds without using reserved cohorts.
Commit the complete runner before the full study. Ledger fixtures are not physical
smoke evidence or study outcomes.

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

## Completed — Selection barrier and immutable study ledger

The ledger requires every declared validation identity, applies all candidate
gates with a deterministic tie-break, withholds failed controllers and binds
evaluation to frozen source/configuration/training/validation evidence. Chunks
resume without replacing or retrying saved outcomes, including failures. Synthetic
tests cover blocked backend calls, interrupted writes, drift and concurrency.
The actual scientific adapter and reserved-cohort runs remain pending above.

## Completed — Numerical components and physical component smoke

Configuration overrides preserve physical delay; pilot fitting uses complete,
separate measured episodes; paired open-loop scoring includes vibration. Kernels
use exact Poisson/read noise and no fallback. The local suite passes 148 tests.
The [predeclared component smoke](results/controller_physics_smoke_v1/README.md)
passes all nine gates: 12 pilots, six predictor fits and 15 controller/fault paths.
Both reports rebuild exactly with 53 protected files unchanged. It uses a smaller
model, fresh seeds and an explicitly untrained network; the full stage-runner and
controller study remain pending.

## Completed — Fixed-budget learned fitting helper

Each declared learned candidate starts from the common initialization. Each Adam
step averages independently normalized gradients from the complete training cohort
at the same parameters. Histories retain every executed episode, gradient and
parameter checkpoint, with a final objective evaluated after the last update.
Numerical failures retain partial evidence and return no usable model. Correctness
tests use analytic objectives and a reduced optical case on separate seeds; they
are not full study training or held-out performance results. Durable integration
and the full runner smoke remain in P1.

## Completed — Source/runtime binding and immutable raw storage

The study opener captures the actual checkout, pinned study/base and fresh runtime
on create/resume. Source bytes must match their committed Git objects, including
files hidden from ordinary status, and untracked execution files are rejected.
Canonical source/runtime evidence hashes match the ledger bindings. Numeric NPZ
files preserve shapes, dtypes and nonfinite failures, permit exact retries and
reject changed replacements. Tests cover interrupted writes, competing writers,
path aliases and changed evidence. No scientific cohort is acquired by this layer.
The full runner and durable per-stage records remain in P1.

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
