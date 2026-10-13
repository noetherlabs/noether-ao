# Controller-study execution safeguards

This is implementation infrastructure for the [predeclared study](controller_study_v1.md),
not a controller result or a finished scientific runner. The selection and record
ledger is implemented. Numerical helpers now connect the unchanged simulator to
measured pilot telemetry, per-episode predictor fits and paired raw-array scoring.
A learned fitting helper now implements fixed-budget, equal-episode gradient
averaging. Source/runtime capture and immutable numeric artifact storage now
connect to the ledger. Durable validation/evaluation callbacks now execute bounded
trial chunks. Training orchestration, complete stage commands and ledger-backed
physical smoke remain pending before any reserved cohort is used. The frozen YAML,
plan and original M0–M7 evidence are unchanged.

## State and evidence contract

    create(config, fresh provenance)
           |
    save pilot/history/parameter artifacts in evidence/
           |
    bind_training(all predictor + learned candidates, evidence)
           |
    validate_chunk(lazy adapter, limit) -> record exact case + raw/paired references
           |          incomplete / duplicate / changed binding -> reject
           v
    freeze_selection(all declared cases, all rejected candidates retained)
           |          no eligible candidate -> withhold that controller
           v
    evaluate_chunk(adapter, limit)
           |          no selection / changed evidence -> reject before adapter
           |          existing valid records -> skip, including recorded failures
           |          adapter exception -> propagate; earlier records stay intact
           v
    append each outcome bound to selection hash; resume with same provenance

`noether_optics.controller_selection.StudyLedger` owns these transitions. Its
constructor verifies a previously created run against the current configuration
and caller-supplied provenance. `create` requires a new directory. Provenance has
three required fields: the full source commit, source-inventory SHA256 and
runtime-fingerprint SHA256. The adapter must calculate them from the actual
checkout and environment, retain those inventories under `evidence/`, and call
the constructor with freshly calculated values on every resumed process. A
syntactically valid digest is not proof that its caller measured the right inputs.
For now even the source commit must match on resume; an artifact-only commit does
not silently change the recorded source. Use the pinned source checkout.

Every fitted predictor and learned candidate has either immutable parameter
references or an explicit training failure. Shared training evidence is mandatory.
Training failure can only produce failed validation records. A failed record has
an error and no invented metrics; a completed record has raw references, an
open-loop reference, finite-status information, WFE, paired open-loop WFE,
saturation fraction and Strehl. Unrepresentable nonfinite metrics use JSON `null`
with `all_outputs_finite=false`; they cannot become eligible. Failures and
completed trials are counted separately.

Validation records bind the training-bundle hash. Their identities include stage,
full candidate parameters, photon regime, wind, fault, atmosphere seed and detector
seed. Selection requires exactly the declared identities; extra or missing cases
cannot substitute for full coverage. All candidates in a regime/seed must share
the same paired open-loop artifact and WFE. A single failed, nonfinite, saturated
or non-useful trial rejects that candidate. Eligible scores use an equal-weight
mean of trial-level WFE/open-WFE ratios. Exact ties follow declared candidate
order, irrespective of file or input ordering. Withheld controllers do not enter
evaluation, and the predeclared comparison-family denominator stays 120.

The frozen selection binds configuration, provenance, training and validation.
Before evaluation identities are released or any backend callback is invoked,
the ledger recomputes selection and rechecks all referenced file hashes. Each
saved evaluation record binds the selection hash. Resumption audits every existing
evaluation record before doing new work; it skips recorded failures as well as
successful records. All-withheld runs have zero planned evaluation cases and are
complete as a ledger operation, **not** a successful scientific comparison.

Records use strict canonical JSON and deterministic, full-length SHA256 case IDs.
Writes publish an already-flushed temporary file with an exclusive hard link;
existing evidence can only be retried byte-identically. A process lock prevents
cooperating writers from advancing stages at the same time and is held throughout
each evaluation chunk. A second writer fails promptly. A callback exception or
publication error propagates instead of being relabeled as a scientific outcome.
Previously published records survive; leftover `.pending-*` files never count as
finished trials. The code targets the existing macOS/Linux environments.

References must stay inside the run's `evidence/` directory. Missing, changed,
duplicate and escaping references are rejected. Files are hashed in bounded
blocks. Hashes and locks detect accidental drift and cooperating-process races;
they are not an access-control sandbox against a writer able to replace the whole
run. Full audits intentionally reread artifacts. The later adapter must budget
that I/O separately from compilation, training and trial execution costs; the
ledger itself collects no timings.

## Remaining adapter responsibilities

The ledger verifies record identity, selection rules and file integrity. It does
not inspect numerical arrays, derive metrics, verify model shapes, generate
atmospheres or certify that a controller used only permitted measurements. The
adapter must enforce those semantics and retain enough raw evidence to audit them:

- Open the run with `controller_evidence.open_study_run` to capture fresh
  source/runtime inventories, verify pinned inputs and require a clean checkout.
  Pass an adequate disk budget and retain the returned provenance references in
  the training binding. Check remaining space during large stages too.
- Save shared atmospheres, exact pilot observations and measured pseudo-open-loop
  estimates. Fit separate sequences, never bridge seeds or use phase truth as
  predictor telemetry. Preserve parameter arrays and training histories.
- Call the learned fitting helper for all six declared candidates, persist every
  returned history/checkpoint and numerical failure, and bind their immutable
  artifacts to the ledger. Freeze training before validation.
- Derive summaries from retained per-frame arrays and full-output finite flags.
  Verify frame counts and matching open-loop seeds/disturbances; vibration must
  be included in both controlled and paired open-loop truth. Keep full and fault
  intervals. References alone do not establish these physical properties.
- Expose separate training, validation/selection and evaluation commands. Pass a
  lazy evaluation callback to `evaluate_chunk`; never generate evaluation arrays
  or preload its models before the gate. Write raw artifacts atomically and
  exclusively before publishing their ledger records. Preserve interrupted raw
  files and resume identical inputs; never silently overwrite or discard them.
- Distinguish an interrupted adapter from a scientifically failed trial. Return
  explicit failed records for actual execution failures; retain diagnostics.
  Unexpected implementation exceptions stop the chunk for review.
- Test on separate, labeled physical smoke cohorts. Commit the complete runner
  before using any reserved study seed. Then run bounded study chunks and produce
  the declared simultaneous paired intervals without excluding failed pairs.

## Numerical helpers and component smoke

`controller_physics` provides reusable numerical components; it does not itself
execute a reserved cohort or replace the ledger gate. Episode configuration copies
the pinned plant and applies declared stage, regime and candidate overrides. It
replaces legacy seed labels with the actual study cohort and keeps the two-frame
physical delay distinct from prediction horizon. Training has no active fault.

    retained measurement + applied command + calibration
      -> measured pseudo-open-loop estimates (no phase-truth input)
      -> complete training cohort, burn each episode separately
      -> one fit per episode -> equally averaged predictor coefficients

    frozen controller inputs -> exact Poisson/read-noise kernel, no fallback
    atmosphere + same physical vibration -> paired zero-DM reference
      -> full-output finite checks -> post-burn and fault-window raw summaries

The pilot reconstruction ignores the simulator's estimate field and computes
the result from observations and applied commands. Fitting rejects missing,
duplicate, foreign, short or nonfinite episodes and returns every episode's
coefficients alongside the mean. No temporal fit crosses an episode boundary.
The kernel requires explicit predictive coefficients or the frozen width-eight
network, checks array shapes and seed range, and uses exact Poisson/read noise.
Gaussian learned fitting is implemented separately in `controller_training`, as
described below; the trial kernel continues to use exact Poisson/read noise.

Open-loop scoring includes the same vibration phase as the controlled loop;
optical-gain drift changes sensing only. Raw metrics preserve a full-output finite
flag when large streams are omitted, so a finite post-burn average cannot hide
a nonfinite early frame. Nonfinite reported metrics become JSON nulls with a
failed finite flag. The ledger remains responsible for reference identity and
artifact integrity; the durable trial adapter now connects both layers.

The [component smoke configuration](../configs/controller_physics_smoke.yaml)
predeclares smaller 16-pixel/eight-mode, 24-frame checks on separate 51001/52001/
53001 cohorts and initialization 54001. It derives and hashes an explicit smoke
base, rejects overlap with reserved roots or derived noise streams, and refuses
artifact acquisition in CI or from a dirty tracked source checkout. It acquires
no timings. The planned checks use 12 pilot episodes, six predictor fits and
five controller paths under three faults. Only the first candidate of each
controller is exercised; no selection is performed. Learned weights are the
**untrained initialization**, so this cannot establish a training result or fair
controller comparison.

Predeclared smoke gates are 1e-12 maximum absolute error for pilot reconstruction,
predictor rebuild and zero-DM variance/Strehl agreement; deliberately omitting
vibration must cause variance error above 1e-8. Also require all controller/fault
paths to be present, finite and free of fallback. Raw atmospheres, calibration,
pilot streams, per-episode coefficients, untrained weights and full trial streams
are saved before the verdict. A protected-file inventory guards rebuilds.

```sh
# Commit source first. The output directory must be new.
.venv/bin/python scripts/smoke_controller_physics.py --out results/controller_physics_smoke_v1
# Rebuild summaries from saved raw evidence without simulating.
.venv/bin/python scripts/smoke_controller_physics.py --rebuild results/controller_physics_smoke_v1
```

Unit tests additionally verify exact noise-path agreement with the unchanged
simulator, future-phase causality, command delay, same-disturbance zero-DM
agreement, independent episode fits, strict raw shape/finite checks, reserved-seed
rejection and CI refusal. This is component wiring verification; the eventual
complete stage runner still needs its own independent smoke cohort.

## Durable validation/evaluation trial adapters

`StudyLedger.validate_chunk` now applies the same locked audit, callback and
publication path as `evaluate_chunk`. Before doing new work, it checks the training
binding and every existing validation record, including failures. It skips saved
records, enforces a positive integer limit and rejects work after selection is
frozen. Both stages pass detached case/training dictionaries, recheck immutable
inputs after callbacks, and publish each result immediately. Exceptions stop the
chunk; earlier records remain. Evaluation also checks that its selection file did
not change during a callback.

`controller_trials.validate_trials(ledger, base, limit)` and `evaluate_trials`
construct a lazy adapter and enter those ledger methods. Construction copies only
the base mapping. Calibration, frozen model loading and atmosphere generation
occur inside the admitted callback, after the relevant gate. Callers must first
open the run with fresh provenance. Training evidence must bind the four captured
source/runtime/study/base JSON files, whose hashes/configuration are checked again
before numerical work. The adapter does not generate training data or fit models.

Ready predictive/learned models need exactly one `parameters.npz` reference among
their immutable model artifacts. Predictive files contain only `coefficients`
with shape `(modes, 2)`; learned files contain `parameter_0` through `parameter_3`
with the fixed width-eight shapes. Values must be finite. Other model artifacts
can retain histories. Failed training yields an explicit failed trial record,
including its diagnostic and retained artifacts, before calibration or atmosphere
access. No substitute/default model is used.

    admitted case -> frozen provenance -> disk/model checks -> lazy calibration
        -> shared atmosphere by stage/wind/root
        -> shared zero-DM reference by full condition, including fault
        -> exact Poisson/read-noise loop, no fallback
        -> full raw output + post-burn/fault metrics -> ledger record

Shared atmospheric and reference arrays each have an immutable identity sidecar
and verified raw-file hash. Candidates in the same condition reuse the same open
reference; photon/candidate changes reuse the same wind/root atmosphere. Each
record references full controller outputs, both shared inputs and their metadata,
plus execution configuration and both metric windows. Vibration affects both the
controlled and reference truth. Nonfinite outputs remain completed acquisitions
with retained raw failures and a false finite flag/null metrics as appropriate;
selection rejects them. Unexpected implementation or storage exceptions propagate.

If interruption leaves raw data before its metadata or ledger record, an identical
retry can complete publication; changed data cannot replace it. Resumption audits
old atmosphere/model/output references before any new callback. Calibrations and
compiled kernels are reused within one chunk. A conservative per-trial free-space
floor is checked before physics, but this does not guarantee space for the entire
study. The adapter currently retains full streams. Full-run disk sizing and
resource-cost reporting must be completed before reserved-cohort acquisition.

Tests cover both ledger transitions and reduced physical paths on independent
81001/82001/83001 cohorts with initialization 84001. Frozen model arrays and some
prior records are explicitly synthetic fixtures; they are not trained study models
or scientific validation outcomes. Physical checks include fault-matched zero-DM
agreement, full raw metric reconstruction, shared references, nonfinite retention,
raw-before-metadata interruption and loading learned parameters from saved files.
No stage CLI, complete training orchestrator, timing measurement, end-to-end smoke
artifact or reserved-study result is supplied by this bounded increment.

## Source/runtime capture and immutable numeric evidence

`controller_evidence.open_study_run` creates or reopens a ledger using the actual
loaded package's checkout. It captures the exact commit, all tracked execution
inputs under `src/`, `scripts/` and `configs/`, plus `pyproject.toml` and `uv.lock`.
Tracked edits, untracked/ignored execution files and source symlinks are rejected.
Standard `__pycache__` files and `.DS_Store` are excluded. Each source file's actual
bytes must also match its committed Git blob, because assume-unchanged and
skip-worktree flags can conceal edits from ordinary status. A second source
inventory after runtime capture rejects intervening changes. Large result trees
are excluded from execution hashes and retain their own evidence inventories.

The study and its SHA256-pinned base must both be tracked execution inputs.
Runtime evidence uses the existing machine/package/device fingerprint plus
selected effective JAX settings, relevant numerical environment flags and the
zlib runtime version. JAX x64 is required. It records installed versions/options,
not hashes of every installed binary. Source/runtime bindings are timestamp-free,
canonical JSON with full SHA256 hashes; they are deterministic within the same
checkout/runtime. This is a process-entry check, not continuous source monitoring.

    source + pinned study/base + runtime -> fresh bindings -> ledger create/resume
        -> immutable source/runtime/study/base JSON -> shared evidence references
    numeric arrays -> deterministic NPZ -> flush -> exclusive publication
        -> checked artifact reference -> future training/trial record

The opener returns `(ledger, base, references)`. The future runner must include
those references alongside pilot/training evidence when it binds training. It
checks a positive free-space budget (default 512 MiB for initialization); full
stages must budget their actual expected output size. Resumption requires the
same exact source commit, input configuration and runtime, including after an
artifact-only commit. Use the pinned source checkout. Missing provenance files
from an interrupted initialization can be republished from identical bindings;
changed files are never overwritten. A partial root without a manifest is retained
for inspection rather than silently deleted or treated as a complete run.

`save_arrays` writes a numeric NPZ archive with sorted entry names, fixed ZIP
timestamps and compression method, preserving scalar shape, dtype, complex values
and raw NaN/Inf failure evidence. It excludes object/string arrays and pickle.
`save_metadata` reuses the ledger's strict JSON publisher; nonfinite JSON is
rejected. Both publish exclusively after flushing. Identical retries preserve the
existing file, while changed values, keys, shapes, dtypes or metadata fail.
Publication accepts only canonical paths beneath `evidence/`, rejects symlink
aliases, and never takes a second ledger lock inside an evaluation callback.
`load_arrays` verifies the saved artifact hash before decoding without pickle.

Caught write/publication errors leave no completed artifact and preserve existing
files. Pending files left by an interrupted process are not automatically consumed
or removed. Concurrent identical publications converge on one file; differing
writers get one winner and an explicit replacement error. Hash checks and exclusive
publication guard cooperating processes and accidental drift; they are not an
access-control boundary against a hostile writer able to replace the entire run.

Tests use temporary Git repositories and synthetic arrays with separate metadata
cohorts. They cover hidden Git edits, source/runtime drift, pinned-base mismatch,
partial initialization, disk refusal, exact retries, numeric failure fidelity,
write/link interruptions, concurrent writers, symlink/path rejection and pickle
rejection. This layer performs no simulation, training, timing or reserved-cohort
acquisition. Trial bindings now use these writers. Stage commands, per-fit checkpoints and
model orchestration still need integration and full-runner smoke checks.

## Learned fitting and its evidence contract

`controller_training.fit_learned_controller` fits one declared learned candidate
using the complete training cohort. Its input episodes contain only an exact case
identity and the full real phase array. Missing, duplicate, foreign, nonfinite or
malformed episodes are rejected before initialization or gradient execution. The
caller must verify the pinned base and supply matching calibration and atmospheres;
the helper neither generates cohorts nor independently authenticates their origin.

    all training episodes -> their own clean post-burn open-loop variances
        -> common width-eight initialization for each candidate
        -> sequential Gaussian optical loss/gradient at the same parameters
        -> equal mean across episodes -> one Adam update -> checkpoint
        -> fixed step budget -> final objective at the final parameters

Each loss is post-burn phase variance divided by that episode's post-burn open-loop
variance. The reference is calculated from the same input phases. A zero or
nonfinite denominator rejects the input; it is never replaced by an epsilon.
Candidate gain/leak settings apply to training; the fixed physical delay remains
unchanged. The existing simulator controls measurement access, and truth appears
only in the plant and loss. Training uses its explicit Gaussian pathwise branch,
clean conditions and no fallback. The protocol still requires exact Poisson/read
noise for validation/evaluation.

The helper executes the declared number of Adam steps with the declared learning
rate and one common initialization per candidate. Every episode's derivative is
evaluated at the same current parameters before a single averaged update. It does
not update between episodes, concatenate temporal sequences, stop early on finite
loss, or restart. Sequential gradients keep only one reverse-mode episode tape
live; phase inputs and small per-episode/parameter histories remain in memory.
Kernels are reused across seeds within each regime. No timings are acquired.

The returned record contains case order, open-loop denominators, initial and
updated parameter checkpoints, per-episode losses/full-output finite flags and
gradients, per-step mean loss/gradient/norm, and final-parameter loss/gradients.
Final scoring performs no further optimizer update. A numerical failure stops
immediately, retains completed and partial records (including failed values), and
returns `status="failed"` with `parameters=null`. Nonfinite optimizer state is
retained with the failed update. A prior finite checkpoint is not a usable fallback
model. Only a fully finite fit returns `status="ready"` and final parameters.

Invalid inputs and unexpected implementation/backend exceptions raise instead of
being mislabeled scientific failures. This numerical helper is not a durable
checkpoint writer; the future stage runner must preserve raw evidence and manage
interruption/resumption. In-memory records can contain nonfinite arrays on failure,
so persist them as raw arrays with strict JSON metadata/diagnostics, not JSON NaN.
The ledger still needs immutable model/history references for every candidate.

Correctness tests use independent 61001/62001/63001 cohorts and initialization
64001, with reduced arrays. An analytic multi-episode objective is checked against
an independent NumPy Adam calculation for all twenty updates; all six candidates
are checked for equal initialization and fixed budgets. The optical kernel is
checked against the unchanged Gaussian loop and a central finite difference. A
one-step, twelve-episode reduced physical fit verifies finite, changed weights
and complete histories. These are software checks, not saved physical smoke
outcomes, trained study models or evidence of held-out improvement.

Additional failures cover held-out cases blocked before backend access, malformed shapes,
coerced identities, nonfinite early outputs, gradient/mean overflow, failed
optimizer state/parameters, final scoring, and unexpected backend errors.

## Original ledger engineering review and tests

The bounded change reuses `controller_protocol` for candidate enumeration and all
predeclared constraints. It adds one ledger class and pure selection code, leaving
the baseline simulation and generic artifact helpers untouched. Their existing
replace-in-place metadata writer does not provide this study's exclusive record
publication, so the ledger uses its own small publisher. There is no UI, new
dependency, timing acquisition or live study in this increment.

The tests in `tests/test_controller_selection.py` use explicitly synthetic bytes,
metrics and a separate 31001/32001/33001 cohort with initialization 34001. They
exercise the complete lifecycle without importing optical or random-number code:

| Path / failure | Required behavior |
| --- | --- |
| Exact ties; reversed input order | Same declared first candidate |
| Absolute vs relative WFE ranking | Select the declared relative score |
| Missing, duplicate, foreign or coerced case | No frozen selection |
| Any failed/nonfinite/saturated/non-useful trial | Reject candidate and retain reason |
| Changed paired open-loop reference | Reject inconsistent comparison |
| Missing training or selection | No later stage/backend call |
| Source, runtime, config, model or raw-data drift | No resumed evaluation |
| Training binding or saved selection changed | Reject before backend call |
| One/all controllers withheld | Exclude their evaluation; retain family size |
| Exact retry, chunk interruption, completed resume | Preserve bytes; no duplicated trials |
| Backend changes identity or frozen model | Reject publication |
| Disk failure during atomic publication | No partial completed record |
| Concurrent writer, escaping path or symlink | Reject without side effects |

Run the correctness checks with the repository's usual command. These fixtures
are not physical smoke results, training evidence, resource-cost measurements or
support for any controller claim.

```sh
CI=true JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q
```
