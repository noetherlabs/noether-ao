# Development status

All required available-platform gates passed. Negative scientific findings remain
negative. The owner subsequently authorized publication to the private
`noetherlabs/noether-ao` repository, then requested commits only on main. Upstream
Dragrace drafts and correspondence remain unsent. See [milestone history](../history.md)
and [next work](TODOS.md).

| Milestone | Gate | Evidence and reason |
| --- | --- | --- |
| M0 | pass | Pinned environment, opened-source notes and design in docs/research_notes.md. |
| M1 | pass on available CPU; platform skips | 144 rows, 10000 samples each, 1440000 calls; maximum complex-field relative L2 error 0. GPU and enforced single-core affinity unavailable. |
| M2 | pass | Clean four-file Dragrace change d194dab; 140 tests passed, two affinity skips; seven adapters pass all six sizes. PROPER installed but MFT unsupported. Both drafts exist. |
| M3 | pass | HCIPy PSFs, fitting laws, delay response and stable integrator gates pass. Full fresh-clone rerun reproduces all six numerical CSV tables. |
| M4 | pass for coverage | All 80 fault/switch/seed trials finite; delays, false alarms, ROC and paired errors saved. This is not a safety gate. |
| M5 | pass for coverage | All 75 trials finite; five seeds per cell, raw data, uncertainty and plot saved. No useful-equivalence threshold found. |
| M6 | pass for coverage | All 300 held-out trials finite; disjoint seeds, finite nonzero gradients and changed weights. All comparisons retained. |
| M7 | pass | Fresh clones install from lock; both suites pass; 33 report/table files regenerate with unchanged raw inputs; every experiment entry point succeeds. |

## Claim results

- **C1: configuration-dependent support.** Float32 dLux CPU device-path p99 is
  0.0592937 ms at N=32 with one wavelength and 6.38013 ms at N=256 with five.
  A small median is not a deadline guarantee. No GPU or full-loop timing claim.
- **C2: proposed noise-equivalence threshold not established.** No flux passes the
  predeclared useful equivalence criterion across all three fidelities. Lower
  fidelity often performs better with this fixed estimator; conditioning, modes,
  sampling and photon noise are confounded.
- **C3: learned advantage not established.** Paired intervals show learned weights
  beating both integrator and forward in 0/12 conditions and losing to at least
  one in 5/12. Training loss reduction is not held-out improvement.
- **C4: partial detection/recovery, no general safety result.** Optical-gain drift
  is missed. Vibration fallback worsens WFE. One of five clean runs alarms, with
  one alarm frame out of 600 evaluated clean frames. Availability fallback helps
  in its synthetic test; pupil-shift fallback does not restore nominal correction.

The nominal pyramid predictor reduces mean WFE by 26.372725760970482 percent.
This is a new configured result, not exact replication of an unspecified earlier
setup. It does not generalize to every noise/fault condition in M6.

See [the claim report](../results.md) for numbers, plots and caveats, and
[the README](../../README.md) for installation and reproduction commands.

## Reproduction evidence

[Fresh-checkout verification](../../results/verification/fresh-checkout/verification.json)
records every command, configuration, fingerprint and gate. The fresh suites have
26 Noether passes and 140 Dragrace passes, with two platform skips. Noether's two
fresh-import warnings are upstream zodiax docstring escape deprecations.

M3 was rerun with the full scientific configuration; every original numerical CSV
agrees at the declared strict comparison tolerance. M2 reran its full scan, with
the same accuracy/support outcomes. M1 uses a selected-point smoke with the full
sample count. M4–M6 use shorter functional configurations. These smoke outputs
are separate from the claim tables. All raw verification outputs are retained.

The M1 interruption and M5 disk-space recovery are recorded beside their raw data.
M5 appended only four missing trials from the exact original source and identical
runtime; the first 71 raw trial files were unchanged. The source-inventory audit
also records two unrecoverable, unused AO draft snapshots from M2; its executed
driver/artifact code and numerical dependencies are recoverable.

## Drafts needing human review

- [Dragrace issue](../../contributions/drafts/dragrace_issue.md)
- [Dragrace PR description](../../contributions/drafts/dragrace_pr.md)
- [Summary and three questions for Louis](../../contributions/drafts/summary_for_louis.md)

## Three largest correctness risks

1. Ideal plant/twin optics and a finite periodic atmosphere. Independent open-loop
   PSF and matched finite-grid sensor agreement do not validate a real instrument
   or establish convergence as the focal support grows.
2. Short runs and few seeds, ideal predictor telemetry, fixed baseline gains and
   Gaussian training versus Poisson evaluation noise limit fairness/generalization.
3. Component timings and synthetic availability faults do not establish real-time
   or physical safety. Missed faults, harmful fallback and false alarms are observed.

## Private publication follow-up

The complete original history is published on main in
`noetherlabs/noether-ao`; setup commits add portable instructions, correctness-only
CI and the follow-up queue. Eight milestone PRs were opened before the owner
changed the preference to commits only; they are historical review records.
Current local publication checks pass: 29 Noether
tests, 140 Dragrace tests with two platform skips, and 33 reproduced table/report
files with raw inputs unchanged. Evidence is in `results/publication`.

The app has an active hourly follow-up for this task, configured to address
checks and then one bounded item from TODOS.md per run. It publishes focused
commits to main, creates no PRs, and reports meaningful progress or changed blockers.

## Independent ideal-pyramid validation

The [saved comparison](../../results/pyramid_validation_v1/report.md) passes all
predeclared gates: 36 pixel cases, 30 derivative cases and 24 deliberate-error
controls on grids 32/64/96 with one or three wavelengths. Maximum intensity
relative L2 error is 1.6942335379634155e-14; maximum derivative relative L2 error
is 9.360885028878882e-11. Maximum absolute unit-power error is
9.769962616701378e-15. The smallest deliberate error is 0.031291468420221256,
above the required 1e-4 rejection threshold.

The independent implementation uses HCIPy matrix Fourier integrals, physical
surface phase and pixel powers. No alignment, scale or output normalization was
fitted. Native HCIPy default centering/filter/parity differences are explicit in
[the protocol](../pyramid_validation.md). This is a finite-grid ideal-model
validation, not an instrument or safety result.

The clean-source run points to commit `56ed79f`; its frozen configuration matches
the earlier protocol commit `e11d3bb`. Verification reproduced nine report files
byte-for-byte with all nine protected data/provenance files unchanged. The core
suite now passes 42 tests. Original M0–M7 runtime code, raw data and claims are
unchanged.

## Fair controller-comparison protocol

The [new protocol](../controller_study_v1.md) is predeclared with a hash-pinned
base configuration, fresh training/validation/evaluation seeds, meaningful equal
candidate budgets, frozen selection and simultaneous paired intervals. Its
[generated plan](../../plans/controller_study_v1.json) is explicitly unexecuted: all
trial counts are planned budgets, not observed results.

Validation rejects direct/derived seed collisions, legacy reuse, uint32 overflow,
duplicate or unequal search candidates, an inert predictor-gain search, changed
base settings and changes to evaluation noise/refitting. No production controller,
baseline raw artifact or C1–C4 verdict changed. The next task is staged runner
implementation and smoke verification before generating any reserved-cohort data.
The core suite passes 71 tests, including 29 protocol checks. The generated plan
reproduces byte-for-byte, and its existing-file guard preserves a frozen output.

## Controller-study execution safeguards

The [study ledger](../controller_study_execution.md) now implements deterministic
selection, complete validation-identity checks, model and open-loop bindings,
immutable records and a gate that verifies frozen evidence before any evaluation
callback runs. Recorded failures are retained and skipped on resume; interrupted
chunks preserve earlier records. Controllers without eligible candidates are
withheld without changing the predeclared comparison family.

The local core suite passes **117 tests**, including **46 new ledger tests**.
They cover changed source/runtime/configuration, altered artifacts, incomplete or
duplicate trials, failed training, nonfinite metrics, interruption, failed atomic
publication, concurrent writers, path escapes and completed-run resumption.
All new test records are synthetic fixtures on separate seeds, not physical
smoke data or scientific outcomes. No reserved study cohort was simulated, no
timings were acquired, and the base model, frozen protocol/plan and original
evidence are unchanged.

The complete scientific runner is still pending: actual provenance capture,
measured pilot telemetry, training, array-derived metrics and stage commands must
be connected to this ledger and physically smoke-tested before the full study.

## Numerical controller-study components

The new numerical helpers implement measured pilot telemetry, complete-cohort
per-episode predictor fits, exact Poisson/read-noise controller kernels and paired
post-burn/fault-window metrics. The open-loop reference includes the same vibration
as the controlled plant. The baseline simulation is unchanged.

The local suite passes **148 tests**, including **31 new numerical/runner-guard
tests**. Separate tests cover measured-only inputs, independent fitting windows,
causality, noise-path equivalence, zero-DM fault pairing, malformed/raw nonfinite
data, reserved-stream exclusion and CI refusal.

The [component smoke](../../results/controller_physics_smoke_v1/README.md) passes all nine
predeclared gates from clean source `dc16e2f`. It retains 12 pilot episodes, six
predictor fits, 15 controller/fault trials and three zero-DM checks. Maximum pilot
reconstruction error is 9.43689570931383e-16; maximum zero-DM variance/Strehl error
is 1.2212453270876722e-15, both below the 1e-12 gate. Omitting vibration produces
variance error 1.0852514490041327, exceeding the 1e-8 rejection threshold. All
trial outputs are finite and fallback is absent; usefulness is not a smoke gate.

Both generated reports rebuild byte-for-byte with all 53 protected files unchanged.
All 48 raw NPZ files, actual configuration, machine/source records and untrained
network weights are saved. The smaller model, first-candidate-only trials and
untrained learned-controller weights limit this to component wiring evidence.
Actual study provenance and ledger-backed stage commands are still pending; the
learned fitting increment is recorded below. No reserved-cohort outcomes or new
timing measurements were generated.

## Fixed-budget learned fitting

`controller_training.fit_learned_controller` now checks complete training inputs,
starts every candidate from the same declared initialization, normalizes each
post-burn episode by its own clean open-loop variance, and averages all episode
gradients at one parameter state before each Adam update. It retains every
executed episode's losses/gradients and all parameter checkpoints, including
partial numerical failures. Failed fits return no usable model; final scoring
uses the actual parameters after the last update.

The local correctness command `CI=true JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q`
passes **175 tests**, including **27 new fitting checks**. Independent NumPy Adam
arithmetic verifies all twenty updates on analytic inputs. All six candidate paths
have the same initialization and budget. A reduced optical gradient agrees with
a central finite difference; a one-step twelve-episode optical fit has finite,
changed weights and complete histories. Failure tests cover missing/foreign data,
invalid normalization, nonfinite initialization/outputs/gradients/updates/final
scoring, and unexpected backend errors.

These are correctness checks using separate 61001/62001/63001 cohorts and
initialization 64001. No reserved study cohort was simulated, no new scientific
smoke or performance outcome was acquired, and no timing measurements were made.
Original source, frozen protocol/plan and retained evidence are unchanged. The
full adapter still needs real provenance, durable training/trial recording and
separate stage commands, followed by a distinct end-to-end physical smoke before
the full controller study.


## Study provenance and raw evidence storage

`controller_evidence` now captures the actual source commit and execution-file
hashes, checks study/base pins, records the fresh runtime, and creates or resumes
an identically bound ledger. It rejects dirty/untracked execution inputs and
compares actual source bytes with committed Git objects even when ordinary Git
status hides changes. Source/runtime/configuration evidence is published as
immutable metadata and returned for the future training binding.

Raw numeric NPZ publication preserves shapes, dtypes and nonfinite failures. Exact
retries leave the existing file intact; changed replacements fail. Verified loads
reject hash drift and pickle. Tests cover interrupted initialization/publication,
competing writers, path/symlink aliases and disk refusal. The full local correctness
command `CI=true JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q` passes **214 tests**,
including **39 new evidence-storage checks**.

These checks use temporary Git repositories and synthetic arrays. No reserved
cohort, physical experiment, training sweep or timing was run, and all baseline
source/evidence and frozen study inputs remain unchanged. The next task is to
connect durable pilot/model/trial records and separate stage commands, including
mid-fit interruption handling, then verify a distinct end-to-end physical smoke.


## Durable validation and evaluation trials

Validation now runs through audited, resumable ledger chunks. Existing successes
and failures are checked and skipped, and each new record is published immediately.
The shared chunk implementation preserves the frozen evaluation gate and rechecks
selection after a callback. A lazy adapter connects exact-noise controller kernels
to immutable atmosphere/open-loop/raw-output files and both metric windows.
It requires bound provenance and valid frozen parameter files before numerical
work; failed training yields an explicit failed record without running physics.

The full local correctness command `CI=true JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q`
passed **240 tests**, including **26 new stage/adapter checks**. A subsequent focused
regression check also passed for selection mutation during an evaluation callback;
it confirms that no evaluation record is published. Checks include raw metric
reconstruction, shared references across candidates, learned weights loaded from
saved files, fault-matched zero-DM agreement, nonfinite retention, interrupted raw
publication, malformed models, evidence drift and disk refusal.

Physical wiring checks use reduced 81001/82001/83001 cohorts and synthetic frozen
model fixtures. They are not study training, scientific validation results or a
full-runner smoke artifact. No reserved study data or timings were acquired.
The original simulator, frozen protocol/plan and saved evidence are unchanged.
The durable pilot, model and training coordination components are recorded below.
Separate stage commands and resource reporting remain, followed by a distinct
full-runner smoke before the reserved comparison. Full-stream retention also requires
full-run disk sizing.

## Durable learned-fit journal

Learned fitting now saves each completed episode's loss, full-output finite flag
and gradient with exact parameter/input bindings. After interruption, it verifies
the saved evidence and replays gradients through the existing Adam loop to recover
optimizer state, computing only missing episodes. Terminal successes and numerical
failures retain model/history references; failed fits never expose usable final
parameters or receive a retry. Unexpected exceptions retain earlier files and
best-effort interruption diagnostics before propagating.

One ledger lock spans each fit. Training/selection bindings or existing validation
records prohibit fitting, and bindings are rechecked before final publication.
Source/runtime/configuration evidence is required. Shared phases must agree across
photon regimes and candidates; changed calibration, phase, parameter or episode
bytes reject resumption. Raw-before-metadata interruptions permit only identical
republication. The input adapter still relies on the caller to supply physically
matching calibration and atmospheres from the pinned source.

The full local command `CI=true JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q`
passes **280 tests**, including **39 new journal checks**. Analytic interruptions
within both optimizer steps and final scoring reproduce the uninterrupted fit's
parameter/history bytes. A reduced optical fit resumes after its Adam update,
retaining its earlier files and producing finite, changed weights. Tests also
cover retained numerical/optimizer failures, frozen stages, input/evidence drift,
lock exclusion and interrupted raw/terminal publication.

Checks use separate 91001/92001/93001 cohorts and initialization 94001. They are
software correctness evidence, not reserved training, scientific comparison or a
full-runner smoke. No timing measurements were acquired. The original simulator,
frozen study inputs/plan and baseline evidence are unchanged. Durable pilots,
predictor fits and aggregate training bindings are now recorded below. Stage commands,
resource/disk reporting and complete-runner verification remain.

## Maintained technical README

The README now introduces the research question and product objective, maps the
implemented architecture to source, separates baseline findings from later
validation, and records the current controller-study components and remaining
runner work. It includes the learned-fit journal and durable trial adapters,
corrects the stale ideal-pyramid validation limit, and retains reproduction paths.
README upkeep is now required by repository instructions and the existing active
hourly automation.

Documentation checks verified 40 local links/anchors, syntax for six shell blocks,
the live experiment CLI choices and eight generated report sections against saved
data. The generated C1–C4 headlines remain byte-identical. No source, dependency,
protocol or raw result changed; this update acquired no new scientific evidence.

## Durable pilot and predictor records

Pilot acquisition now publishes audited, resumable episode records with calibration,
shared atmosphere files, complete raw outputs and reconstructed measured telemetry.
Each predictor fits the complete declared cohort, retains per-episode coefficients,
and returns a terminal model/evidence record. Failed pilots or numerical fits
remain terminal failures; interruptions preserve prior files and diagnostics.
Learned fits reuse the same atmosphere files without replacing their bytes.

The full local correctness suite passed **318 tests** in 311.91 seconds, including
**38 new pilot/predictor checks**. Tests cover interrupted publication, exact resume,
finite and nonfinite fits, shared learned-fit inputs, integrity/frozen-stage guards,
disk refusal, and all six predictors on a reduced physical pilot cohort. Separate
101001/102001/103001 cohorts and initialization 104001 keep these correctness checks
apart from reserved scientific data. No new timings or reserved outcomes were
acquired. Training coordination and aggregate binding are now recorded below;
stage commands, resource/disk reporting and a distinct full-runner smoke remain.


## Reader-facing repository structure

The README now leads with the project purpose, quickstart, system diagram,
measured evidence and implementation state. Architecture and reproduction detail
have dedicated guides; ROADMAP.md defines controller-study, mismatch-detection,
instrument-interface and operational-evidence milestones. It states that public
release and licensing remain pending rather than presenting a private repository
as open source.

AGENTS.md is untracked and ignored locally. Development logs now live under
`docs/development`; unsubmitted upstream drafts live under `contributions/drafts`.
Current links and report-generator paths follow those moves. The Python package,
CLI and local checkout keep their compatible names. The remote contains no extra
noether-optics wrapper folder. The existing hourly automation now reads the new
paths and keeps agent/scheduler instructions out of tracked source.

The prior implementation increment passed all 318 local tests. This presentation
increment rebuilt all 33 baseline reports/tables with protected raw evidence
unchanged; the [rebuild report](../../results/verification/repository-layout-20261014/rebuild.json)
is retained. Local documentation links, shell-block syntax, generated headlines
and the live CLI choices were checked. Numerical source, frozen protocols,
dependency pins and baseline outcomes are unchanged by this reorganization.


The documented first physical-validation command completed successfully, with all
seven saved physical gates passing. This is the reduced functional configuration,
not a new scientific comparison; its check record is saved alongside the rebuild
report. Final documentation checks cover 129 local links, 13 shell blocks and eight
generated sections. The release-readiness audit is now the next bounded preparatory
task; no visibility or licensing change occurred.

## Training coordination and aggregate binding

`controller_training_stage.run_training_chunk` now advances the open training stage
under separate pilot and model limits. It waits for the full pilot cohort, processes
all predictive and learned candidates in their predeclared order, rechecks terminal
successes and failures on resume, and publishes `training.json` only after all twelve
fitted candidates have immutable model/evidence records. The aggregate evidence is
deduplicated and sorted and includes each terminal record.

A separate nonblocking coordinator lock prevents two workers from claiming the same
chunk while pilot and model operations keep their existing ledger locks. Learned
inputs come from the saved pilot atmospheres; a rebuilt callable calibration must
match every persisted numeric member exactly. Completed learned fits now use a
hash-only verification path, so routine resumption does not rerun optical open-loop
normalization or construct an optimizer. Bound-stage re-entry performs no numerical
work. Predictor failures remain explicit failed models in the aggregate bundle.

Focused correctness checks pass for chunked end-to-end aggregation, retained failed
pilots/predictors, changed terminal artifacts, invalid limits, competing workers and
the bound-stage no-execution path. They use the existing separate correctness seeds,
not the reserved study cohorts. The complete local correctness suite passes **326
tests** in 362.72 seconds, including **8 new coordinator checks**. No performance
timing or scientific outcome was acquired. The bounded training command is recorded
below. Validation, selection and evaluation commands are also recorded below;
whole-run resource/disk sizing and a distinct physical runner smoke remain before
the reserved controller study.

## Bounded training command

The CLI now exposes `study-train` as its own argparse subcommand. It opens or resumes
the ledger through fresh source/runtime capture, then delegates exactly one positive
pilot/model chunk to `run_training_chunk`. `--create` is first-call only; a resume
omits it and must reproduce the bound inputs. One sorted JSON object reports committed
progress. The command never reaches validation, selection or evaluation.

Dedicated subparsers keep training-only options out of the legacy experiment help
and reject invalid limits or cross-command flags before configuration/run access.
Seven command-boundary checks replace both the opener and coordinator; they confirm
path conversion, creation/resume state, integer limits, JSON output and zero access
after invalid input. The complete local correctness suite passes **333 tests** in
291.82 seconds. No study directory, physical array, reserved seed or timing was
created. The validation and selection commands are recorded below; the evaluation
command is also recorded below. Resource sizing and the runner smoke remain.

## Bounded validation command

The CLI now exposes `study-validate` as a separate resume-only subcommand. It opens
the existing run through fresh source/runtime capture, forwards one explicit positive
record limit to `controller_trials.validate_trials`, and prints one sorted JSON
progress object. The underlying trial executor remains lazy: the ledger checks the
complete training binding and rejects a frozen selection before any calibration,
model or phase construction.

The command has no creation path. Missing or nonpositive limits and training-only
`--create` fail during argument admission before run access. Twelve focused CLI
checks replace the opener and validation entry point. The complete local correctness
suite passes **338 tests** in **98.56 seconds**. No study directory, physical array,
reserved seed, controller result or timing sample was created. The selection command
is recorded below; the evaluation command is also recorded below. Full-run resource
sizing and a separately seeded runner smoke remain.

## Bounded selection command

The CLI now exposes `study-select` as a resume-only subcommand. It opens the bound
run through fresh source/runtime capture and delegates directly to
`StudyLedger.freeze_selection`. That transaction audits complete validation,
training and artifact bindings, applies the frozen eligibility/tie rules, and
exclusively publishes the selected and withheld controller set. Exact re-entry is
idempotent. One sorted JSON object reports the complete selection and its hashes.

The command has no creation, chunk or numerical callback path. `--create`, `--limit`
and model-stage flags fail during argument admission before run access. Four new
selection checks bring the focused CLI file to **16 passing tests**. The complete
local correctness suite passes **342 tests** in **94.85 seconds**. No study directory,
physical array, reserved seed, controller result or timing sample was created. The
evaluation command is recorded below. Full-run resource sizing and a separately
seeded runner smoke remain.

## Bounded evaluation command

The CLI now exposes `study-evaluate` as a resume-only, positive-chunk subcommand. It
opens the bound run through fresh source/runtime capture and delegates to
`controller_trials.evaluate_trials`. The lazy adapter enters
`StudyLedger.evaluate_chunk`, which revalidates frozen selection, validation,
training and artifact bindings before releasing chosen-controller identities or
constructing calibration, models and phases. Each result binds the selection hash;
withheld controllers receive no held-out cases.

The command cannot create or refreeze a run. Missing or nonpositive limits and
training-only `--create` fail during argument admission before run access. Five new
evaluation checks bring the focused CLI file to **21 passing tests**. The complete
local correctness suite passes **347 tests** in **90.79 seconds**. No persistent
study run, physical array, reserved seed, controller result or timing sample was
created. The four stage commands are complete; full-run resource/disk sizing and a
separately seeded end-to-end runner smoke remain before reserved acquisition.

## Whole-run storage preflight

The read-only `study-resources` command now turns the frozen case counts and retained
array schemas into an auditable disk envelope. It counts 3,069 numeric archives and
42,166,977,776 logical numeric bytes. With zero compression credit, per-member ZIP
allowance, metadata/filesystem space, one pending-write allowance and an operating
reserve, the admission threshold is 105,033,365,984 free bytes.

Sixteen new resource and command-boundary checks cover exact frozen totals, schema
drift, invalid plants, nearest-ancestor filesystem checks, equality admission,
insufficient capacity, file targets and argument isolation. The focused resource/CLI
suite passes **37 tests**; the complete local correctness suite passes **363 tests in
92.05 seconds**. A live read-only check against `runs/controller_study_v1`
found about **5.5 GB free** and returned the specified status 2. It created no
target, physical array, reserved outcome or timing sample. The disk-sizing design is
complete and creation cannot bypass it. The runner smoke below has since passed;
full-study capacity remains the operational blocker for reserved acquisition.

## Passed full-runner smoke

`controller_runner_smoke_v1.yaml` now fixes new training, validation, evaluation and
initialization streams with no reserved-study or component-smoke overlap. Its reduced
plant uses N=8, four modes, an 8-pixel detector and eight frames. Five controllers
receive one candidate each; the unchanged plan still requires 12 pilots and 100
validation trials before selection. The smoke stops after one held-out evaluation
record, so it tests the complete command/evidence path without acquiring a comparison
cohort.

`run_controller_study_smoke.py` invokes the actual resource, training, validation,
selection and evaluation commands in fresh processes, rechecks selection exactly,
protects every retained run file and rebuilds a gate report from those bytes. CI and
dirty-source runs are rejected. Seven new pre-acquisition checks cover the fixed plan,
stream overlap, CI refusal and changed/missing/additional evidence. The complete local
correctness suite passed 370 tests in 89.80 seconds, and GitHub Correctness passed for
the runner commit.

The retained run then passed all nine gates from source commit `37f6bfc`. It saved
12 pilot records, two terminal fitted models, all 100 validation records, an exactly
repeatable frozen selection and one of 600 possible held-out records. All five reduced
candidates were eligible and none of the training or validation records failed. The
488-file protected inventory verifies on rebuild, while the derived summary and report
reproduce byte for byte. No reserved seed or timing sample was acquired, and this
single held-out record supports no controller comparison. The remaining reserved-study
blocker is the unchanged 105.033 GB storage-admission threshold.

## Public-release readiness audit

The release audit is complete for source commit `fa8a8b8`; the repository remains
private and unlicensed. Before the audit, package metadata was hardened so the wheel
includes the technical README, keeps pytest outside runtime requirements and retains
the exact upstream Diffraction-Dragrace MIT notice. The full local suite passed 370
tests in 92.32 seconds, and GitHub Correctness run
[`38509422393`](https://github.com/noetherlabs/noether-ao/actions/runs/38509422393)
passed from the same commit.

Gitleaks 8.30.1 reported zero findings across 38 commits and 2,394 Git objects. A
read-only archive inspection opened 1,372 tracked NPZ files containing 10,937 arrays
and found no string, object or unreadable array. Two clean wheel builds produced the
same SHA256, and an isolated no-dependency installation exposed all ten CLI commands.
The machine-readable evidence and scoped direct-license inventory are retained under
`results/verification/release-readiness-20261014/`.

The audit verdict is **not ready for public release**. No project license exists;
historical commits retain former `AGENTS.md` and `outbox/` paths; 35 protected result
files contain local provenance paths; and standard wheel metadata does not reproduce
the Git/URL dependency sources used by the locked environment. A monitored security
route and human review of three unsent drafts also remain. No visibility, license,
history, evidence file, reserved cohort or timing sample was changed.

## Legacy optical-gain observability audit

The first P2 increment reuses the protected M4 fault run as development evidence and
does not call the simulator. A deterministic audit binds 48 source files by SHA256,
rebuilds from commit `5fc820d`, and marks every M4 evaluation seed ineligible for a
future P2 held-out claim. Two focused tests cover causal-window indexing, invalid
arrays and byte-for-byte reconstruction of the committed JSON result. The complete
local correctness suite passes **372 tests in 96.30 seconds**.

The original 0.995 upper-tail detector found optical-gain drift in 0/5 runs. An
exploratory causal four-frame mean, calibrated at the 0.005 lower quantile on the
three original clean training seeds, found 5/5 old gain runs after 6–10 frames and
had no pre-onset alarms in those runs. It still alarmed in 1/5 clean runs and 5/585
clean windows. None of the 30 retained non-target fault runs crossed the lower
threshold.

The single seed with complete WFS frames also rejects a tempting passive estimator.
Once the injected response multiplier reaches 0.35, a direct measured-versus-predicted
response regression changes by only −1.83%, while paired residual variance rises by
2.99 times. This is a one-seed identifiability warning, not a general result. No
detector has been validated, no fallback was activated, and no new seed, timing or
reserved evidence was acquired. The next step is a frozen P2 protocol with fresh
training, validation and evaluation streams.

## Frozen optical-gain observability protocol v1

The P2 v1 protocol is now machine-readable, bound to the unchanged base simulation
hash and represented by an exclusive-write generated plan. Static validation rejects
direct and derived seed collisions, missing prior exclusions, base-config drift,
candidate changes, relaxed release gates, altered inference and post-outcome escalation
selection. The plan declares 64 training trials, 240 validation trials and at most
1,200 evaluation trials across four photon/wind regimes. It explicitly records that
no threshold, outcome, validation decision or evaluation release exists.

The old audit now records the post hoc persistence choice exactly. Eight consecutive
lower crossings remove the one legacy clean excursion and retain all five legacy gain
detections, with persistence-complete delays of 13–17 frames. Those data remain
development-only. The frozen v1 numerical threshold will be calibrated separately per
regime from fresh training-clean data; evaluation stays sealed unless every validation
regime passes.

No simulator, fresh seed, timing path, fallback or controller action was used in this
increment. The next boundary is implementation plus a separate nonreserved runner
smoke and storage preflight before any declared cohort is opened. The focused audit
and protocol suite passes **34 tests**; the complete local correctness suite passes
**404 tests in 89.44 seconds**.

## Optical-gain detector and training threshold freeze

The v1 causal kernel now computes the declared four-frame post-burn-in statistic,
strict lower-tail crossings and eight-crossing persistence with explicit frame
identities. The legacy M4 audit imports the same kernel and still reconstructs its
committed JSON byte for byte.

The threshold publisher enumerates the 64 declared training-clean cases, orders them
independently of submission order and calibrates one NumPy `linear` 0.005 quantile per
regime over 6,096 windows. Before exclusive publication it verifies the exact protocol
file, all raw NPZ hashes and score arrays, and complete case identities, then records
validated source/runtime identity fields. Synthetic negative checks cover incomplete,
duplicate, foreign, failed, nonfinite, path-unsafe, hash-drifted, pickle-bearing and
score-mismatched evidence, as well as idempotent and changed publication attempts.

No simulator, declared seed, numerical threshold, validation decision, held-out
outcome, timing path or controller action was used. The next implementation boundary
is the validation-release barrier plus deterministic retention/resumption and storage
admission before a distinct nonreserved runner smoke. The focused detector, protocol
and audit suite passes **65 tests**; the complete local correctness suite passes
**435 tests in 92.97 seconds**.

## Optical-gain validation release barrier

The v1 validation barrier now enumerates all 240 frozen validation identities and
rebuilds `thresholds.json` from its 64 hash-bound training artifacts before reading a
validation decision. Completed records must retain every gate-relevant frame array;
their statistic frames, float64 means, crossings, alarm frames and persistence flags
must reproduce exactly from the shared detector kernel.

All clean false-alarm, severe-gain detection, pre-onset, boundary, delay, finite,
fallback, availability and saturation criteria are applied independently to each of
the four regimes. A terminal numerical failure remains in the record and produces an
immutable withheld decision. A passing complete cohort publishes the same one-way
artifact with evaluation released. Identical retries preserve bytes; changed retries
cannot replace either decision.

The 21 focused checks use synthetic arrays and cover every criterion, cohort pairing,
onset-straddling streaks, terminal failure, raw and training hash drift, altered
detector traces, idempotence and changed-decision refusal. No simulator, declared
seed, P2 threshold, validation decision, held-out outcome, timing path or controller
action was used. Deterministic runner resumption, complete retention, storage
admission and a distinct nonreserved smoke remain next. The complete local
correctness suite passes **456 tests in 96.01 seconds**.
