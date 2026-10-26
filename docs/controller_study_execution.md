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
  --create --pilot-limit 1 --model-limit 1
```

Use `--create` only for a new directory; omit it on subsequent calls. Limits bound
newly recorded work. The coordinator acquires the common pilot cohort, fits every
predictive and learned candidate, and freezes training only after all candidates
have terminal records. Failed models remain failures.

After training is complete:

```sh
.venv/bin/noether-optics study-validate \
  --config configs/controller_study_v1.yaml --out runs/controller-study-v1 --limit 1
.venv/bin/noether-optics study-select \
  --config configs/controller_study_v1.yaml --out runs/controller-study-v1
.venv/bin/noether-optics study-evaluate \
  --config configs/controller_study_v1.yaml --out runs/controller-study-v1 --limit 1
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
.venv/bin/python scripts/run_controller_study_smoke.py --out runs/controller-runner
```

Reconstruct saved reports without acquiring new trials:

```sh
.venv/bin/python scripts/smoke_controller_physics.py --rebuild results/controller_physics_smoke_v1
.venv/bin/python scripts/run_controller_study_smoke.py --rebuild results/controller_runner_smoke_v1
```

Timing acquisition runs outside CI. Compilation and execution costs require
separate synchronized measurements; they are not inferred from the simulated
loop frequency or storage estimates.
