# Next work

M0–M7 are complete; their negative findings remain part of the baseline. Handle
new CI failures and actionable owner feedback before the research tasks below.
Complete one bounded task per autonomous run. Do not create speculative busywork
if all actionable tasks are done or require new evidence unavailable locally.

## P1 — Resolve public-release owner decisions

The concrete [release-readiness record](../release_readiness.md) is complete for
source commit `fa8a8b8`. Full-history credential scanning found zero findings; all
1,372 tracked NPZ archives were readable and pickle-free; two wheel builds matched;
the pinned environment and 370-test correctness run passed. The README is embedded
in package metadata, pytest is development-only, and the exact upstream
Diffraction-Dragrace MIT notice is retained. GitHub currently reports the repository
as public, but no project license has been chosen.

Remaining work is owner-gated: confirm whether the current public visibility is
intentional; choose a project license and publication-history
strategy; decide how public artifacts handle 35 protected files containing local
provenance paths; define package metadata for the tested Git/URL dependency pins;
name a monitored security-reporting route; and review the three unsent Dragrace
drafts. Implement those decisions, verify a standards-only clean install, then
repeat the audit before any tag or visibility change. Do not rewrite measured files
or private Git history merely to make the release appear clean.

## P1 — Execute the predeclared controller study

The [protocol](../controller_study_v1.md), [inputs](../../configs/controller_study_v1.yaml)
and [unexecuted plan](../../plans/controller_study_v1.json) are predeclared. The
[execution ledger and selection barrier](../controller_study_execution.md) now
check complete trial identities, training bindings, immutable evidence and resumed
evaluation chunks. Numerical helpers now provide measured pseudo-open-loop pilot
telemetry, separate per-episode predictor fits, exact-noise controller kernels
and paired raw-array scoring. Learned fitting now implements common initialization,
normalized equal-episode gradients, fixed Adam budgets and failure records. Bounded
training coordination, the complete aggregate binding and its `study-train` command
are implemented. A resume-only `study-validate` command now advances positive chunks
through the lazy ledger gate. A resume-only `study-select` command now audits and
freezes complete validation. A bounded `study-evaluate` command now enters held-out
chunks through the frozen-selection gate. Durable validation and evaluation trial
adapters record full raw outputs through ledger chunks.
Fresh source/runtime capture and exclusive numeric artifact storage are implemented;
bind their returned provenance references with the training evidence. The learned
fit journal now preserves episode losses/gradients, replays saved episodes through
the unchanged optimizer and returns terminal model/history references. The training
coordinator integrates those references with durable pilot/predictor records and
freezes one immutable bundle after every fitted candidate terminates. Static
whole-run disk sizing is complete: 42.167 GB of numeric payload yields a conservative
105.033 GB free-space threshold with zero compression credit. The current development
filesystem fails that gate; use a qualifying volume rather than reducing evidence.
The separate component smoke and committed full-runner smoke have passed. The latter
retains complete smoke training and validation, frozen selection and exactly one
held-out record on distinct 61001/62001/63001 seed families. Its protected evidence
exercises the real stage commands and lazy evaluation gate. Ledger fixtures remain
separate from this physical smoke evidence. A qualifying full-study volume is now
the remaining operational precondition for reserved acquisition.

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
The [predeclared component smoke](../../results/controller_physics_smoke_v1/README.md)
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
The evaluation command is recorded below. Resource sizing and the full-runner smoke
remain in P1.

## Completed — Durable validation and evaluation trials

Validation chunks now audit old records and skip saved successes/failures under the
same lock/publication rules as evaluation. A lazy physics adapter loads models and
constructs arrays only within admitted callbacks. It saves shared atmospheres,
fault-matched open loops, full outputs and both metric windows before publishing
each record. Model/provenance/schema checks and per-trial disk refusal precede new
physics. Correctness tests use reduced separate cohorts and synthetic frozen
models; no reserved study or full-runner smoke result is claimed.

## Completed — Durable learned-fit episode journal

Each episode's loss, full-output finite flag and gradient is saved with its exact
parameter/input identity. Resumption verifies the evidence and rebuilds Adam from
cached gradients, running only missing episodes. Terminal fits return immutable
model/history references, including ineligible numerical failures. Interruption
diagnostics are retained and unexpected exceptions propagate. Correctness checks
compare interrupted and uninterrupted fits and exercise a reduced optical resume
on separate seeds. Aggregate training orchestration is now completed below; stage
commands are now completed below, while a full-runner smoke remains in P1. This
is not reserved training or a new controller-performance result.

## Completed — Durable pilot and predictor records

Pilot acquisition audits prior records, skips saved successes/failures and appends
bounded exact-noise integrator episodes under the ledger lock. It saves calibration,
shared phases, full observations/commands and measured telemetry. Predictor fitting
requires the complete cohort, audits telemetry against the saved measurements and
fits each burned-in episode separately. Terminal models retain coefficients and
histories, or raw numerical failures without usable parameters. These records join
the aggregate training bundle through the completed coordinator. The evaluation
command is recorded below; the resource report and full-runner smoke remain P1 work.
No reserved scientific cohort was acquired.

## Completed — Training coordination and aggregate binding

The bounded coordinator acquires only the requested number of missing pilots, then
processes predictive and learned candidates in the predeclared order. Existing
terminal successes and failures are reverified and skipped. A separate stage lock
prevents competing coordinators while candidate operations retain their ledger-lock
checks. Only a complete twelve-model collection can publish `training.json`; the
bundle contains deduplicated, sorted evidence plus every terminal record.

Completed learned records now verify their persisted manifest, candidate, provenance,
calibration, episode identities and artifact hashes without recomputing open-loop
normalization or constructing an optimizer. Correctness checks cover bounded resume,
aggregate publication, retained predictor failures, evidence tampering, invalid
limits and competing workers on separate 101001/102001/103001 fixtures. Later-stage
commands are completed below; full-run resource sizing and the end-to-end physical
smoke remain P1.

## Completed — Bounded training command

`noether-optics study-train` creates or resumes a provenance-bound run and delegates
one explicit pilot/model chunk to the training coordinator. Both limits must be
positive before the run is opened. `--create` is required only for the first call;
omitting it resumes the identically bound run. The command prints a deterministic
JSON progress record and cannot accept the legacy latency `--resume` flag.

Command-boundary checks replace the opener and numerical coordinator, proving that
parsed paths, creation state and chunk limits cross the boundary exactly and that
invalid or legacy-only option combinations cause no run access. This is runner
wiring, not reserved acquisition. The validation, selection and evaluation commands
are recorded below; resource sizing and the physical full-runner smoke remain P1.

## Completed — Bounded validation command

`noether-optics study-validate` resumes a provenance-bound study and delegates one
explicit positive chunk to the existing lazy validation adapter. It cannot create a
run. The ledger therefore requires the complete frozen training bundle and rejects
post-selection validation before the adapter can build calibration, load models or
generate phases. One sorted JSON object reports durable progress.

Command-boundary checks replace both the opener and validation entry point, proving
exact path/limit forwarding and zero run access after missing, nonpositive or
training-only options. The complete suite passes 338 tests. This wiring acquires no
reserved cohort, physical array, result or timing. The selection command is recorded
below; the evaluation command is also recorded below. Resource sizing and the
separately seeded full-runner smoke remain P1 work.

## Completed — Bounded selection command

`noether-optics study-select` resumes a provenance-bound study and calls the ledger's
existing atomic selection barrier. Complete validation and every bound artifact are
rechecked before the deterministic chosen/withheld controller set is published.
There is no physics callback, chunk size or creation mode, and exact re-entry is
idempotent. One sorted JSON object reports the selection and binding hashes.

Four command-boundary checks prove exact opener/freeze dispatch and zero run access
for creation, validation-limit or model-limit options. The complete suite passes
342 tests. This wiring acquires no reserved cohort, physical array, result or timing.
The evaluation command is recorded below. Resource sizing and the separately seeded
full-runner smoke remain P1 work.

## Completed — Bounded evaluation command

`noether-optics study-evaluate` resumes a provenance-bound study and delegates one
explicit positive chunk to the existing lazy evaluation adapter. The ledger
recomputes and verifies frozen selection plus every upstream binding before releasing
chosen-controller held-out identities. Withheld controllers receive no cases. Saved
successes and failures remain terminal and each record binds the selection hash.

Five command-boundary checks prove exact opener/adapter dispatch and zero run access
after missing, nonpositive or creation options. The complete suite passes 347 tests.
This wiring acquires no reserved cohort, physical array, result or timing. The four
stage commands are complete; resource/disk sizing and the separately seeded full-runner
smoke remain P1 work before reserved acquisition.

## P2 — Optical-gain drift observability

M4 misses gain drift and produces harmful vibration fallback. After independent
sensor validation, investigate a separately calibrated observable or detector on
new training/validation cohorts. Freeze thresholds before new held-out fault runs.

The [legacy M4 observability audit](../../results/optical_gain_observability_audit_v1/README.md)
is complete. It preserves 48 input hashes and acquires no new simulation evidence.
The original upper-tail detector found 0/5 gain-drift runs. A development-only
four-frame lower-tail candidate found 5/5 with 6–10 frame delays, no pre-onset alarms
in those runs, and one clean alarm run out of five. Passive response regression stayed
near nominal in the one full-frame seed while residual variance rose by 2.99 times.
The old cohort is now ineligible for P2 evaluation.

The [P2 v1 protocol](../optical_gain_observability_v1.md) is frozen and its training
stage is in progress. It
assigns new nonoverlapping seed streams; training-only, per-regime calibration; a
validation-only release; run-, frame- and raw-crossing gates; target misses and
detected-only delay; paired clean/fault controller error; and no detector action or
fallback. It predeclares modal dither with paired science controls as the first v2
escalation. Twenty-four of 64 training-clean terminal records are retained; no new threshold,
validation decision or evaluation outcome has been generated.

The causal transform and immutable training threshold freeze are implemented. The
freeze requires all 64 declared training-clean identities, verifies each score against
its hash-bound raw NPZ, derives the protocol-file hash, binds source/runtime identities
and refuses to replace an existing threshold. Its tests are synthetic and create no
study result.

The one-way validation barrier is also implemented. It rederives the threshold file
from retained training artifacts, requires all 240 validation identities, verifies
stored detector traces, retains terminal failures, applies every gate per regime and
publishes release or withholding without replacement. Its tests are synthetic and
open no declared seed.

The protocol-bound stage ledger is implemented. It advances training and validation
in deterministic positive chunks under one process lock, publishes canonical terminal
records exclusively, rechecks all saved bindings and artifacts before skipping them,
retains numerical failures and retries only interrupted callbacks with no terminal
record. Threshold freeze and validation release remain separate one-way actions.

Static storage admission is implemented. The outcome-free command derives 2.341 GB of
numeric payload and a 10.191 GB free-space requirement from all declared array shapes,
1,504 maximum trials, 1,568 numeric archives and 32 full-WFS records, with zero
compression credit. The current development target was refused at 7.995 GB free and
was not created.

The full run later passed the same independent gate with 16,472,510,464 bytes free and
was created from clean source commit `ae62a92`. Four bounded calls retain all 16
declared seed pairs in `regime-00` and the first eight in `regime-01`, for 24 terminal
completions with hash-bound raw arrays and metadata. The run contains 83,544,120 bytes
at this boundary. A later read-only preflight found 5,300,817,920 free bytes against
the unchanged 10,190,604,096-byte requirement and refused continuation before a fifth
acquisition call. The 24 retained records remain unchanged. No experiment timing was
collected.

The physical training/validation adapter and complete retention path are implemented.
It applies the exact forward-controller, noise and condition mapping; normalizes all
declared array shapes and dtypes; retains post-burn-in WFE, Strehl and saturation in
hash-bound metadata; and keeps produced numerical failures. Training raw streams stay
immutable while threshold-derived traces are added as separate artifacts at freeze.

The strict reduced runner has passed all 14 implementation gates from clean committed
source with protected, byte-reproducible evidence and no evaluation access. Its
unfavorable detector outcome is retained and cannot tune v1.

The sealed evaluation and final-report lifecycle is implemented. Evaluation requires
a previously published favorable validation release, reconstructs the complete
threshold and validation chain before exposing identities, and binds every terminal
record to those immutable hashes. The complete cohort can publish one descriptive
report with exact run intervals, descriptive frame fractions, paired Bonferroni WFE
intervals, explicit failure denominators and no second gate. Synthetic smoke fixtures
cover resumption, interruption, retained failures, evidence drift and report
idempotence without opening a declared cohort.

Next recover at least 4,889,786,176 bytes on the target filesystem, rerun the read-only
preflight and resume the remaining 40 training records from the source-pinned checkout
only after admission. Freeze the threshold only after all 64 terminal records and
preserve any negative threshold, validation or evaluation result.
Clean-run and frame false alarms,
pre-onset alarms, misses, detected-only delay and paired controller error must all be
reported. Useful detection does not establish safe fallback or a general safety claim.

## Completed — Independent ideal-pyramid validation

The [predeclared protocol](../pyramid_validation.md) passed on grids 32/64/96:
36 intensity cases, 30 derivative cases and 24 deliberate-error controls. The
reference uses HCIPy matrix integrals and a physical thin surface. See the
[saved report](../../results/pyramid_validation_v1/report.md), raw arrays and provenance.
All nine generated report files reproduce byte-for-byte; protected raw inputs
are unchanged. The core suite has 42 passing tests.

This validates the matched finite-grid ideal sensor contract. Native HCIPy
defaults, focal-support convergence and instrument validation remain outside the
claim. No M0–M7 production model, stored evidence or scientific verdict changed.

## Completed publication preparation

- Preserved milestone commits and documented the history; delivery is commits only.
- Added pinned correctness-only CI and regression checks for its timing boundary.
- Made the Dragrace reconstruction command independent of the checkout name.
