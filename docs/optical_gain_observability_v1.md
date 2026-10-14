# Optical-gain observability protocol v1

## Status and question

This is a frozen, unexecuted P2 protocol. The machine-readable
[configuration](../configs/optical_gain_observability_v1.yaml) is bound to the base
simulation by SHA256; the generated [plan](../plans/optical_gain_observability_v1.json)
contains seed identities and budgets, with every outcome, threshold and release field
set to false. The causal detector, physical training/validation adapter, complete
retention checks, resumable record ledger, training-calibration publisher and
validation-release barrier are implemented. Their tests use synthetic arrays and
mocked physics boundaries and do not create a study threshold or decision.

The question is narrow: **can a training-calibrated lower-tail transform of the
existing whitened residual score detect the simplified scalar optical-gain ramp under
fresh conditions without unacceptable clean alarms?** The detector observes a fixed
forward controller. It cannot change a command, latch fallback or claim recovery.

The retained M4 cohort is development evidence only. Its
[read-only audit](../results/optical_gain_observability_audit_v1/README.md) selected
the candidate below after outcomes were visible, so no M4 seed is eligible for a P2
validation or evaluation claim.

## Frozen candidate

For each regime, let `s[f]` be the existing whitened residual score at frame `f`.
After the 128-frame burn-in, the first statistic is the causal four-frame arithmetic
mean ending at frame 131:

```text
m[f] = mean(s[f-3], s[f-2], s[f-1], s[f])
```

The numerical threshold is the NumPy `linear` 0.005 quantile of all training-clean
`m[f]` values in that regime. A raw crossing uses strict comparison `m[f] < threshold`.
An alarm frame is an endpoint with eight consecutive raw crossings. The first possible
alarm is therefore frame 138. A run alarms if any eligible alarm frame exists. To
attribute an alarm to an injected condition, all eight crossing endpoints in that
streak must be at or after fault onset; a nearly complete pre-onset streak cannot earn
a detection at the boundary.

The tail, averaging window, quantile, strict comparison, regime-specific calibration
and persistence are frozen. There is no union with the legacy upper-tail detector.
Thresholds come from fresh training-clean records only and become immutable before
validation. Validation cannot recalibrate them.

Eight-window persistence is a post hoc development choice. In the old M4 evidence,
the only clean excursion lasted five windows while all five gain-fault runs contained
streaks of at least 33 windows. Persistence removed the old clean alert and completed
gain alarms 13–17 frames after onset. That result motivates the rule; it does not
estimate fresh performance.

## Plant and cohorts

The protocol inherits the base ideal-pyramid plant, then fixes 512 frames, a 128-frame
burn-in, fault onset at frame 256 and a 32-frame gain ramp. The tested controller is
`forward` at gain 0.3 and leak 0.999. It is not refit. Every trial uses exact Poisson
plus read noise, with fallback and detector-to-command coupling disabled.

| Role | Roots per regime | Conditions | Planned trials |
| --- | ---: | --- | ---: |
| Training | 16 | clean | 64 |
| Validation | 30 | clean, terminal gain 0.35 | 240 |
| Evaluation | 60 | clean, gain 0.35, gain 0.65, vibration, misregistration | 1,200 |
| **Maximum** |  |  | **1,504** |

The four regimes are the Cartesian product of 1,000 and 1,000,000 photons with wind
scales 0.5 and 2.0. Within a role and regime, every condition reuses the same atmosphere
and detector roots to form paired comparisons. The same roots are common random
numbers across regimes. Training, validation and evaluation streams are otherwise
disjoint, including the detector key offset. The validator excludes all recorded
baseline, latency, reserved-controller and smoke roots and their derived detector
streams.

The gain conditions linearly move the sensor response to multipliers 0.35 or 0.65 over
32 frames. Vibration retains the base 0.7 rad, 40 Hz injection; misregistration retains
the two-pixel shift. These are simulation faults, not instrument-calibrated envelopes.

## Alarm accounting

- A clean alarm run contains any persistent alarm from frame 138 through frame 511.
- A pre-onset alarm run contains an alarm completion before frame 256.
- A target detection requires no pre-onset alarm, a streak beginning at or after frame
  256 and its first completion in `[263, 512)`. Every other target run is a miss.
- A persistent event that straddles frame 256 is reported separately as a boundary
  alarm and never credited as a target detection or nuisance response.
- Detection delay is `first_alarm_frame - 256` and is summarized only for detected
  runs. The miss count is always shown beside it.
- Nuisance conditions report alarmed and quiet runs, rather than borrowing the target
  language of hits and misses.
- Raw crossing, persistent alarm-frame and run fractions use their explicit eligible
  denominators. Frame values are temporally correlated and receive no binomial
  confidence interval.

## Validation release gate

Each of the four regimes must independently satisfy every row. One failure seals the
entire evaluation cohort and remains a retained negative result.

| Gate per regime | Frozen criterion |
| --- | ---: |
| Clean alarm runs | 0 / 30 |
| Clean raw crossing fraction | at most 0.01 |
| Clean persistent alarm-frame fraction | at most 0.001 |
| Gain-0.35 pre-onset alarm runs | 0 / 30 |
| Gain-0.35 qualifying detections | at least 29 / 30 |
| Gain-0.35 misses | at most 1 / 30 |
| Detected-only completion delay | every value at most 32 frames |
| Full-run integrity | all finite; no fallback, unavailable frame or saturation |

The clean run gate is decisive; the persistent frame gate is retained as a separate
accounting and consistency boundary. The raw crossing gate checks whether the
training-calibrated tail transfers before persistence suppresses excursions.

These are engineering screening thresholds. If zero of 30 validation-clean runs alarm,
the one-sided 95% Clopper–Pearson upper bound is still 9.50%. Passing does not establish
an operational false-alarm rate or safety guarantee.

## Evaluation and inference

Evaluation is opened once, only after a complete immutable validation decision passes
all regimes. It cannot alter the threshold, persistence or release criteria. Evaluation
is descriptive; it has no second pass/fail gate.

Run-level clean, target and nuisance proportions receive two-sided 95% exact
Clopper–Pearson intervals. With zero alarms among 60 clean runs in one regime, the
one-sided 95% upper bound is 4.87%. Alarm-frame and raw-crossing fractions remain
descriptive because adjacent windows share samples and atmosphere dynamics.

Every nonclean evaluation condition is paired with its clean record at the same
regime and seed. Over the declared fault window `[256, 512)`, the primary
controller-error quantity is `WFE_fault - WFE_clean` in nanometres. Paired Student-t
intervals use familywise alpha 0.05 with Bonferroni
correction over 16 predeclared cells: four nonclean conditions by four regimes. Each
interval therefore has confidence 99.6875%. Strehl deltas are secondary descriptive
values. Detection does not change either trajectory, so these comparisons measure
fault consequence under observation, not recovery benefit.

## Evidence and failure handling

Every trial retains the score, lower statistic, raw crossing, P2 alarm, residual
variance, Strehl, saturation, fallback, model-availability, command and
applied-command arrays. Explicit frame-index arrays bind the 381 statistic endpoints
(131–511) and 374 persistence-eligible alarm endpoints (138–511); startup frames are
not represented by fabricated zeros or NaNs. Per-run WFE, Strehl and saturation
metrics are retained separately. Measurement and prediction frames are retained for
the first seed of every role/regime/condition cell. Failed numerical outcomes stay in
the cohort. Storage planning receives no compression credit.

Training has one unavoidable ordering boundary: its regime threshold depends on all
16 clean records, so crossing and persistence arrays cannot be truthfully written by
an individual pre-threshold callback. Each training callback therefore publishes an
immutable raw archive containing the eight threshold-independent streams plus its
metric metadata. The one-way training freeze recalibrates the threshold, writes a
second five-array detector archive for each record, verifies both hashes and binds
both references into schema-2 `thresholds.json` before that file is published.
Validation receives an already-frozen threshold and writes all 13 arrays in one raw
archive. This split changes archive count, not the declared numeric payload.

The implemented training boundary accepts only all 64 declared clean records. It
normalizes arithmetic to float64, verifies each supplied score against the `score`
array and SHA256 of its retained NPZ, derives the protocol SHA256 from the checked
source file, records source/runtime bindings, and publishes `thresholds.json`
exclusively. Failed, missing, duplicated, foreign, nonfinite, path-unsafe or
artifact-mismatched inputs cannot publish a threshold; an identical retry is
idempotent and a changed retry cannot replace the file.

The validation boundary first reconstructs that threshold file from all bound
training artifacts. It then requires all 240 declared validation identities, verifies
every completed raw NPZ and stored detector trace against the shared kernel, and
applies every criterion above independently in each regime. Numerical failures remain
in the immutable decision and force withholding. `validation_release.json` records
either global release or withholding; identical retries are idempotent and no changed
decision can replace it.

The stage ledger binds the full protocol and its SHA256 to source/runtime provenance,
holds one nonblocking process lock across each callback and terminal publication, and
schedules cases in declared order. At the start of a resumed chunk it revalidates the
complete semantic contract; after each admitted callback it rechecks the immutable
record and artifact bytes without repeatedly decoding earlier large arrays. An
interruption without a terminal record is retried; published outcomes are never
replaced. Threshold and release files close their stages permanently.

The read-only storage preflight derives each NPZ member from this frozen contract. It
budgets 1,504 maximum trials as 1,568 numeric archives: one raw archive per trial plus
64 threshold-derived training detector archives. Measurement and prediction images
are included for the first seed of each of 32 role/regime/condition cells. The numeric payload is
2,341,413,792 bytes; a no-compression-credit archive envelope, metadata allowance,
temporary-write headroom and operating reserve require 10,190,604,096 free bytes. The
maximum assumes validation releases all 1,200 evaluation trials. A failed or withheld
run can use less space but cannot weaken admission.

```sh
.venv/bin/noether-optics observability-resources \
  --config configs/optical_gain_observability_v1.yaml \
  --target runs/optical_gain_observability_v1
```

The command imports no JAX or simulator module, creates no target and exits 2 after
printing its JSON report when capacity is insufficient. The first live check found
7,994,904,576 bytes free and refused admission. This is a filesystem observation, not
an execution cost or outcome.

Before acquisition, the implementation must still pass the committed
`optical_gain_observability_runner_smoke_v1` protocol through the physical callbacks
on a filesystem that passes its 5,370,101,856-byte admission gate. The reduced smoke
uses two training and two paired validation roots on a 24-frame plant, never enters
evaluation, and does not require a favorable detector outcome. Unit tests and mocked
callback checks are not execution evidence, and no P2 threshold, validation decision
or outcome exists yet.

The physical callback entry points enforce the full storage envelope before every
chunk. The check adds current free bytes to filesystem blocks already allocated by
the same immutable run, preserving the original admission budget as evidence grows
while detecting unrelated consumption. Per-trial temporary headroom is checked again
inside the admitted callback before JAX or the simulator constructs numerical work.
New-run opening performs storage refusal first, then captures clean committed source,
JAX/runtime options, the checked protocol and pinned base as four canonical provenance
artifacts. Their hashes must reproduce the ledger binding before every physical trial.

## Predeclared escalation

If v1 fails validation or yields inadequate held-out behavior, v1 ends with that
result. The first escalation is **modal dither** in a new
`optical_gain_observability_v2` protocol with wholly new training, validation and
evaluation streams. That protocol must pair each dithered trial with a dither-free
science control and report WFE, Strehl and command RMS impact. V1 held-out outcomes
may not be used to choose a different escalation branch.

This route follows the independent-excitation approach described by
[Deo et al. (2019)](https://doi.org/10.1051/0004-6361/201935847). An independent
focal-plane channel, as studied by
[Chambouleyron et al. (2021)](https://arxiv.org/abs/2103.02297), changes the sensing
contract more substantially and requires a separate later protocol. Neither route is
implemented or evaluated here.

## Rebuild the plan

This command validates the base hash and seed separation, then writes with exclusive
creation. It imports no JAX module and does not simulate, calibrate or inspect a cohort.

```sh
.venv/bin/python scripts/plan_optical_gain_observability.py \
  --config configs/optical_gain_observability_v1.yaml \
  --out runs/optical-gain-observability-plan.json
```

The committed plan must match the rebuilt JSON apart from its chosen output path.
