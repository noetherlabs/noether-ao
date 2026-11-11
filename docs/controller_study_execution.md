# Controller-study execution

The [controller protocol](controller_study_v1.md) separates training, validation,
selection and held-out evaluation. `StudyLedger` enforces those transitions and
binds each record to its configuration, source, runtime and input artifacts.
The retained [runner check](../results/controller_runner_smoke_v1/README.md)
exercises the complete sequence on reduced, separately seeded inputs. It does not
constitute the full controller study.

## Running a study

Check storage before creating a run:

```sh
.venv/bin/noether-optics study-resources \
  --config configs/controller_study_v1.yaml --target runs/controller-study-v1
```

The frozen configuration requires 105,033,365,984 free bytes, including conservative
archive and temporary-write allowances. The check creates no run and gives no
credit for compression. The complete study must also pass the separate runner
check before acquisition on reserved cohorts.

Training advances in bounded chunks:

```sh
.venv/bin/noether-optics study-train \
  --config configs/controller_study_v1.yaml --out runs/controller-study-v1 \
  --create --pilot-limit 1 --model-limit 1 --measure-costs
```

Use `--create` only for a new directory; omit it on subsequent calls. Limits bound
newly recorded work. The coordinator acquires the common pilot cohort, fits every
predictive and learned candidate, and freezes training only after all candidates
have terminal records. Failed models remain failures.

After training is complete:

```sh
.venv/bin/noether-optics study-validate \
  --config configs/controller_study_v1.yaml --out runs/controller-study-v1 --limit 1 --measure-costs
.venv/bin/noether-optics study-select \
  --config configs/controller_study_v1.yaml --out runs/controller-study-v1
.venv/bin/noether-optics study-evaluate \
  --config configs/controller_study_v1.yaml --out runs/controller-study-v1 --limit 1 --measure-costs
```

Repeat validation chunks until the declared cohort is complete before selecting.
Selection rechecks every candidate, paired reference and eligibility gate. It uses
the protocol's tie order and withholds controllers with no eligible candidate.
Evaluation constructs numerical inputs only after the selection checks pass.

## Measurement and fitting contracts

Predictor training reconstructs pseudo-open-loop estimates from measured frames,
applied commands and calibration. Each episode is fitted after its own burn-in;
coefficients are averaged with equal episode weights. Simulator phase truth is
not a predictor measurement.

Learned candidates share initialization and the declared fixed Adam budget.
Each update averages gradients evaluated at the same parameters across the full
training cohort. Training uses Gaussian pathwise noise; validation and evaluation
use exact Poisson/read noise. This model mismatch remains an experimental limit.
Nonfinite fitting outcomes retain their arrays and cannot supply a fallback model.

Controlled and open-loop trials share atmosphere and physical disturbances,
including vibration. Optical-gain drift changes sensing, not the reference phase.
Metrics retain full-output finite flags as well as post-burn and fault-window
summaries. JSON nulls with a failed finite flag represent nonfinite metrics.

## Evidence and recovery

Source capture verifies tracked execution inputs under `src/`, `scripts/` and
`configs/`, together with the package metadata and dependency lock. Their bytes
must match committed objects. Resumption requires the original source commit,
configuration and runtime; an artifact-only commit does not silently replace the
acquisition identity.

Numeric archives preserve shapes, dtypes, complex values and nonfinite outcomes,
without pickle or object arrays. Metadata uses canonical JSON and SHA256 bindings.
Raw artifacts are published before records, through flushed temporary files and
exclusive links. Identical retries are accepted; changed evidence is rejected.
References must stay beneath the run's evidence directory.

Locks serialize cooperating writers. Resumption audits existing successes and
failures before skipping them. An implementation or storage exception interrupts
the chunk; it is not relabelled as a failed scientific trial. Earlier records and
unfinished files remain available for diagnosis. The learned-fit journal replays
saved gradients through the same optimizer and verifies parameter bytes before
computing missing episodes.

These integrity checks detect accidental drift and cooperating-process races.
They do not protect against a writer who can replace the entire run.

## Reduced checks

The component check exercises measured reconstruction, prediction, delay and
fault-matched scoring with untrained learned weights. The runner check exercises
training, validation, selection and one held-out trial. Neither supplies a
statistical controller comparison. Use fresh output directories for acquisition:

```sh
.venv/bin/python scripts/smoke_controller_physics.py --out runs/controller-components
.venv/bin/python scripts/run_controller_study_smoke.py --out results/controller-runner --measure-costs
```

Reconstruct saved reports without acquiring new trials:

```sh
.venv/bin/python scripts/smoke_controller_physics.py --rebuild results/controller_physics_smoke_v1
.venv/bin/python scripts/run_controller_study_smoke.py --rebuild results/controller_runner_smoke_v1
```

## Resource costs

Use `--measure-costs` for every full-study acquisition chunk. It is rejected in CI
before opening a study. Run without other compute on the host. The per-run lock
excludes cooperating measured writers; it cannot exclude unrelated applications.

Each invocation writes an exclusive `costs/*.jsonl` file bound to the run manifest,
source and runtime. It records input placement, lowering/compilation and
synchronized execution of the controller or gradient kernel separately. Kernel
executables are reused across seeds and parameter values with the same shapes,
dtypes, weak-type signature and input devices. Compilation records describe the
observed cache state, not guaranteed cold compilation. Controller output
conversion is recorded separately. A fresh process retains its own compilation
costs.

Pilot, fitted-model and trial records also retain inclusive elapsed spans. These
include their nested kernels and the relevant preparation, checks and persistence;
do not add enclosing and nested costs. Predictor numerical fitting has its own
span. Learned fitting includes optimizer bookkeeping, gradient-journal replay and
host conversion; primitive dispatch/compilation outside the explicitly lowered
kernels is not separated from that enclosing span. Invocation cost additionally
includes stage audits and accounting writes; admission precedes that span.

Every completed operation is flushed independently. Unexpected exceptions retain
an interrupted attempt; a process killed before the end record leaves an incomplete
attempt, never a zero-cost observation. A declared predictor numerical failure
completes its measured operation and remains a failed scientific outcome. Reused
terminal records and replayed gradients do not create new kernel executions.
Retain interrupted and resumed attempts together, rather than reporting only the
successful final chunk.

These are study resource costs, not per-frame latency distributions or deadline
guarantees. The simulated loop frequency and storage estimate imply neither.

## Completed-cohort reporting

For the released cohort, first [download and verify the raw evidence](release_integrity.md).
This reuses the completed acquisition; it does not require rerunning training or
evaluation.

After every released evaluation case has a terminal record, reconstruct the
descriptive report from retained arrays:

```sh
.venv/bin/python scripts/report_controller_study.py \
  --run runs/controller-study-v1 --out runs/controller-study-v1-report.json
```

The report rechecks frozen selection and provenance, rejects missing or unexpected
trial identities, and reconstructs post-burn and fault-window metrics from raw
arrays. It does not acquire trials, change selection, or refit a model. The
analysis checkout must be clean and committed; its source and runtime are recorded
separately from the acquisition provenance. Write the derived report outside the
retained run. Existing reports are never replaced.

Primary paired comparisons retain the complete predeclared family, including
withheld controllers. Simultaneous Student-t intervals use seed-level differences;
failed or nonfinite pairs receive no interval. A superiority verdict also requires
every trial of the proposed winner to satisfy the usefulness and saturation gates.
Fault-window intervals are exploratory pointwise intervals and do not receive an
additional superiority verdict. A condition with a failed seed has no aggregate
mean computed from only its surviving trials. All individual records remain in
the report.

Cost summaries retain incomplete attempts with unknown total elapsed time and keep
nested kernel and inclusive operation costs separate. Missing cost measurements
are labelled as not recorded. These summaries do not establish a control-loop
latency or a hardware deadline.

The completed [Prediction Boundary report](../results/controller_study_v1/README.md)
contains 600 validation and 600 evaluation records. Its separate NumPy audit
reconstructs finite-grid residual variance, coherent pupil Strehl and zero-DM
references from saved phase screens and applied commands, and checks command
delay and modal clipping:

```sh
.venv/bin/python scripts/audit_controller_physical_metrics.py \
  --run runs/controller-study-v1 --report runs/controller-study-v1-report.json \
  --out runs/controller-study-v1-physical-audit.json
```

This audit requires the complete retained acquisition, not only the published
summary files. It uses the saved pupil and basis; it does not independently
reproduce the atmospheric statistics, sensor, controller or hardware. Tolerances
are fixed in the script. A mismatch withholds the audit result; it is not repaired
by changing retained arrays. The raw-verified report remains the authority for
cohort identity, model selection, statistical reconstruction and failure handling.
