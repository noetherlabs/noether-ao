# Decisions

- Build isolated sibling repositories inside the writable project workspace:
  noether-optics and Diffraction-Dragrace. Do not modify prior Noether work.
- Sole engineer, as requested. No subagents. Record assumptions instead of asking
  questions. The autonomous request supersedes interactive gstack review prompts.
- Civil astronomy only. This is a simulation and measurement project, not a claim
  of operational assurance or collaboration with any institution.
- Configuration values, mathematical constants, version identifiers and acceptance
  tolerances are inputs, not measured results. All reported performance values must
  be generated in this session and linked to saved raw data and provenance.
- Use Python 3.11 and a uv lock. Keep source checkouts for source/API inspection.
- A single-plane FFT with explicit dynamic OPD is the initial latency contract.
  It is not the fixed focal-grid MFT contract in Dragrace. Report them separately.
  Multiwavelength FFT planes have wavelength-dependent angular grids; do not sum
  them into a broadband detector without resampling.
- macOS has no Linux sched_setaffinity. Never substitute environment thread flags
  for affinity. Unsupported execution modes get explicit skip records.
- The gstack scripts at the legacy path are absent, but the installed review skill
  and checklist are readable. Use their engineering/test/review workflow without
  installing or publishing unrelated changes.
- Gate tolerances and experiment configurations must be written before runs.
  Failed gates remain failures; fixes may correct implementation, never move a gate.

## Simulation assumptions (before control experiments)

The plant is a finite periodic von Karman screen with two independent frozen layers.
No subharmonics, scintillation, telescope obscuration, dispersion, coronagraph,
non-common-path aberrations, or hardware dynamics are claimed. A sampled,
reference-grid-orthonormal Zernike modal DM defines radians RMS, with a command FIFO.
The ideal pyramid uses dLux forward/inverse FFTs, a four-face phase wedge,
flux-conserving detector binning and a fixed detector extent. This is not a
MagAO-X configuration. A stuck actuator is represented by a localized loss of
DM response, because this first model has modal rather than physical actuators.

The linear predictor learns two taps per mode on ideal open-loop telemetry from
training seeds only. This is optimistic calibration. The forward estimator uses
fixed-preconditioner gradient steps; it is not an exact Gauss-Newton solve.
The compiled control/optical path supports gradients. Training uses a Gaussian
shot/read approximation so pathwise derivatives exist; evaluation uses exact
Poisson sampling and Gaussian read noise. Discrete fault decisions are not
claimed differentiable. Fixed and learned weights blend candidate commands
using the actual issued command state. Fallback latches until the run ends.

Detector thresholds are empirical clean-training quantiles. No statistical
independence, formal safety proof, or out-of-distribution guarantee is assumed.
ROC curves include the optical score OR the explicit missing/late availability
rule, matching the deployed decision logic. Score-only ROC values are also saved;
synthetic availability flags are not evidence of optical fault discrimination.
Fidelity uses a common higher-resolution plant and a shared detector. M1 latency
is a component cost only; a pyramid and its derivatives cost additional work.
Across-seed error bars are standard errors. Paired confidence intervals are
Student-t intervals over the fixed held-out seeds, without multiple-test correction.

## Interrupted timing process

The original M1 process disappeared without a completion record. Completed raw
points are retained without filtering. Resumption validates saved points and runs
only unfinished points, using the identical timed region and experiment config.
The continuation fingerprint and interruption record are saved beside the raw data.

The pyramid misregistration injection translates the complete complex entrance
pupil inside the padded support, including its amplitude edge. It is not merely a
shift of phase samples under an unchanged pupil. The optical-gain fault ramps from
the nominal response to the configured gain over a configured drift interval.

Unavailable twin outputs are discarded. Without fallback, the advanced controller
holds its last command; with fallback, the independently calibrated integrator
can issue a fresh command from the current WFS frame. Synthetic availability flags
model missing/deadline-late outputs; they do not measure real scheduler failures.

## Dragrace adapter environment

The current POPPY release rejected Python 3.11 during dependency resolution.
Use the newest compatible POPPY release instead, pinned in the optional Dragrace
extra. PROPER installs from its official archive and prysm from the inspected
source commit. No core numerical dependency was unpinned or silently upgraded.

M4 retains every frame's score, alarm, fallback state, variance and Strehl for
all seeds. Full predicted/observed WFS frames are also retained for the first
held-out seed of every fault/switch condition. This bounds disk use while keeping
all raw values needed to regenerate the published detector and WFE statistics.

A unit test caught NumPy's batched-vector solve shape convention in the predictor.
The solve now supplies an explicit singleton right-hand-side dimension; the
predictor alignment test remains unchanged.

The brief was resubmitted unchanged when this task resumed. Previously measured
results retain their original UTC timestamps and fingerprints. M3 had completed
successfully; it was reviewed and committed before starting M4. New measurements
receive new provenance rather than being relabeled as part of the earlier run.

Fault WFE comparisons use the same post-burn-in interval for both switch states,
including the pre-fault portion of that interval. Detection delays use the configured
fault interval and exclude runs already alarmed before onset. Clean false-alarm
rates use the entire clean post-burn-in interval; pre-onset rates are also reported.
Frame ROC points are correlated in time and are not independent event trials.
No detector threshold was retuned after seeing held-out faults.

Fresh-checkout verification has two distinct checks. It rebuilds every published
table from its preserved raw data, and exercises all experiment commands with
explicit shorter smoke configurations. Smoke outputs are excluded from C1–C4
claim tables. The latency smoke retains the minimum evaluation count. The full
scientific configurations and their original gates remain unchanged.

Before the mixed-controller run, review replaced the average modal noise estimate
with the per-mode diagonal of the calibrated reconstructed noise covariance.
This affects only the fixed and learned mixture features; the completed standalone
controller experiments do not use that quantity in their issued commands.

M5 exhausted available disk space while saving its final four trials. Recovery
verified every source hash against the original commit, required an identical
runtime fingerprint, reused preserved atmosphere arrays and appended only missing
trials. All pre-existing raw files retained their hashes. This was an artifact
write failure, not a failed physical gate. No results were selected or discarded.
The non-monotonic fidelity result is retained: fixed estimator steps and penalties
are not re-tuned after seeing evaluation data. It does not isolate photon noise
from controller conditioning, mode count and sampling.

The provenance audit found that M2's broad source inventory included then-untracked,
unused AO drafts. Two draft snapshots were modified before their first commit and
cannot be recovered by hash. M2's committed driver imports only the committed
artifact helper from this package; the unresolved files did not execute in M2.
Keep this inventory limitation explicit and retain the original provenance.

M6 retains the predeclared training budget and all held-out outcomes, including
losses. No evaluation-driven re-training or controller tuning was performed.
Before fresh-checkout reproduction, report-only review made the fidelity axis
logarithmic, labeled the combined useful/unsaturated criterion explicitly, and
standardized generated CSV line endings. Scientific arrays and gates did not change.

M7 used independent local Git clones of both repositories and a new environment
installed from the lock. Cached distribution files were hard-linked to conserve
disk space; source checkouts and editable installation paths were separate.
Verification rebuilt all published report/table files without changing raw inputs,
reran M3 with the full configuration, reran the full Dragrace scan, and exercised
M1/M4/M5/M6 with the explicitly labeled smoke configurations. M3 numerical CSVs
matched the original run within the strict reproduction tolerance. Fresh smoke
outputs are retained separately and never substituted into the scientific claims.

## Private repository publication and continuation (2026-10-12)

The owner explicitly authorized pushing all completed work to the private
`noetherlabs/noether-ao` repository and opening progressive PRs. This supersedes
the earlier no-push/no-PR restriction only for that repository. It does not
authorize an upstream Dragrace submission or outbound personal messages.

Preserve the eleven original commit hashes: raw provenance refers to them.
Bootstrap main at the original planning commit and open eight stacked milestone
PRs at existing boundaries, followed by a repository-setup PR. These are review
slices of work already completed locally, not claims of earlier GitHub review.
Keep them open for review. Merging should preserve original commits rather than
squashing or rebasing measured history. Subsequent work builds on the latest
unmerged stack head until its prerequisites are merged.

The repository name is noether-ao; retain the existing Python distribution,
imports, command and pinned version to avoid changing recorded environments.
The setup change does not publish a new package release. CI runs correctness
tests only, on CPU, with timing guards enabled; saved-data report rendering is
allowed because it does not collect measurements. Timing runs remain local and
isolated. Preserve the scientific gates, limits and all negative findings.

For autonomous continuation, use a four-hour task follow-up cadence and one
bounded task from TODOS.md per run. Prioritize test failures and actionable PR
feedback, then independent sensor validation and fair controller comparisons.
Record assumptions rather than requesting routine implementation choices.

### Superseding delivery instruction: commits only

During publication the owner explicitly changed the instruction to "no pr,
commits only" and permitted a force push if necessary. Publish the complete tested
history directly on main and close the eight PRs already opened. Do not create
further PRs. The main update is a fast-forward, so no force push is necessary.
Keep the original measured commits and progressive commit history intact. This
supersedes the stacked-PR and no-direct-main instructions immediately above.

The command-line OAuth credential cannot upload workflow changes. The connected
GitHub app has the required access and created the pinned correctness workflow
through the contents API. Its commit was merged without changing the tested tree.
Use that authorized app for workflow changes; ordinary commits use Git push.

## Hourly continuation and independent sensor validation

The owner requested immediate work and changed the follow-up interval to one
hour. Update the existing task automation rather than creating another. Continue
with commits only on main, retaining original provenance and negative findings.

The next bounded task uses HCIPy matrix Fourier transforms and a physical thin
surface as an independent implementation of the existing ideal sensor contract.
Native HCIPy pyramid defaults differ in centering, focal support and output
parity; do not silently fit those differences away. Document them and predeclare
gates before running comparisons. No production sensor or controller change is
planned. Gstack engineering review is performed autonomously under the owner's
no-questions instruction: the minimal reference/runner/test design is selected,
with the test-flow diagram and failure modes in docs/pyramid_validation.md.

The full independent sensor comparison passed every predeclared gate without
changing any tolerance or production code. Store the six raw grid/spectrum arrays
in results/pyramid_validation_v1 with source/library hashes and clean-commit
provenance. Both detector pixels and small-signal derivatives agree with the
HCIPy matrix reference; sign, geometry, orientation and scale controls are rejected.
The scope remains the matched finite-grid ideal optical contract. Do not promote
this numerical agreement into native-default, convergence, instrument, controller
fairness or safe-recovery claims. Rebuilding all generated reports reproduced
their exact bytes and preserved protected inputs. Next freeze a fair controller
protocol before generating any new training/validation/evaluation outcomes.

## Predeclared fair-controller study v1

The current predictive command ignores config.gain, so a common gain grid would
give it duplicate candidates. Give each controller six actual candidate settings:
gain/leak for integrator, forward, fixed and learned; horizon/ridge for predictive.
Equal validation/search opportunities are not equal wall-clock compute budgets;
report actual costs separately when the study is run.

Reserve fresh disjoint training, validation, evaluation and initialization seeds.
Audit the existing loop's seed+100000 detector derivation as well as atmosphere
roots, including cross-role collisions, overflow and all recorded baseline seeds.
The original ideal-telemetry predictor helper is unsuitable for this protocol:
future fitting must use noisy pseudo-open-loop pilot estimates from training only.
Learned fitting retains its explicitly labeled Gaussian approximation; validation
and evaluation use exact Poisson/read noise. Train each candidate for the same
fixed number of steps, freeze one global choice per controller on clean validation,
and never refit after selection or add candidates based on outcomes.

Use declared endpoint fluxes and slow/fast winds, with predeclared held-out drift
and vibration conditions. Hold physical delay and estimator budgets fixed. Require
each validation trial to be useful, finite and within the saturation limit; if no
candidate qualifies, record failure and withhold evaluation for that controller.
Use a fixed simultaneous comparison family instead of selecting favorable nominal
95% intervals from many conditions. These are protocol choices, not findings.

Gstack's small-change engineering/review workflow is applied autonomously: reuse
the existing simulation in a later runner, add only protocol validation/planning
now, and keep the test-flow/failure map in docs/controller_study_v1.md. User delivery
instructions override main-branch/PR steps: review, test and commit directly to
main without creating a PR. This increment generates no new scientific outcomes.

## Controller-study stage barrier (2026-10-12)

Split runner implementation into reviewable increments: first the immutable
ledger and selection/evaluation barrier, then the scientific adapter and physical
smoke verification. This keeps reserved seeds untouched while lifecycle checks are
tested. The full study, budgets and scientific gates are unchanged. Use gstack's
small-change planning, two-pass review and shipping checks under the existing
autonomous/direct-main instructions. The test map is in
docs/controller_study_execution.md; no PR, version bump or new dependency is needed.

Reuse controller_protocol for candidate identities and fixed rules. Keep the
baseline simulation and raw evidence untouched. Use exclusive atomic metadata
publication and one process lock, because generic replace-in-place writers can
silently alter a resumed study. Every validation record binds its original
training bundle; frozen selection binds the entire validation set. Candidate
comparisons must share paired open-loop references, and recorded failures never
disappear or get retried as if unobserved. Unexpected adapter exceptions stop a
chunk; scientifically failed trials must be explicitly recorded by the adapter.

The ledger validates supplied source/runtime fingerprints but cannot independently
prove their capture or the physical meaning of raw arrays. The future adapter must
capture real inventories, derive metrics from retained arrays, enforce measured
telemetry, audit frame counts/disturbance pairing and remain lazy until the gate.
Record this boundary explicitly rather than calling synthetic fixtures a physical
smoke run. Resume currently requires the exact source commit as well as source,
runtime and configuration hashes. Use a pinned source checkout when committing
output artifacts separately. Full integrity rereads favor correctness; budget
their I/O separately from scientific resource costs when building the adapter.

## Numerical adapter and component smoke (2026-10-12)

Implement the numerical boundary before the long-lived stage runner: reuse the
unchanged simulator, reconstruct pilot telemetry from observations plus applied
commands, and fit each burned-in training episode separately before averaging.
Never use the old ideal-telemetry predictor helper. Add a paired zero-DM reference
with the same physical vibration and score retained arrays in both required
windows. Full-output finite evidence remains mandatory when streams are omitted.

Use the gstack small-change engineering/review/shipping workflow under the existing
one-engineer, autonomous direct-main instruction. Test flow is documented in
docs/controller_study_execution.md. Preserve baseline source, protocol and data.
This increment supplies functions and a bounded component smoke, not a finished
study runner. Defer optimizer/training checkpoints, real study provenance and
ledger stage integration to the next increment rather than bypassing those gates.

Predeclare a 16-pixel/eight-mode 24-frame smoke with fresh roots and explicit
derived-stream collision checks. Test all controller/fault code paths using the
first declared candidate; the network is explicitly untrained. Save raw inputs,
calibration, predictor episodes/coefficients, outputs and source/runtime metadata.
Require 1e-12 absolute reconstruction/rebuild/zero-DM agreement and reject a clean
reference substituted for vibration at a 1e-8 variance-error threshold. These are
wiring checks, not controller acceptance or superiority thresholds. Commit source
before running, retain any failures, rebuild from protected raw files, and never
generate reserved study cohorts or timings in this increment.

The clean-source component smoke passed every predeclared gate. Retain its complete
raw record and negative control under results/controller_physics_smoke_v1. Rebuilt
summaries match byte-for-byte, with all protected inputs unchanged. Treat the
untrained-network paths only as execution checks; do not infer learned improvement
or tune the reserved study from these smoke outcomes. Next integrate learned
training and real provenance with the ledger, then use a distinct end-to-end smoke
cohort before full-study execution.

## Learned fitting helper (2026-10-13)

Complete the next numerical boundary as one bounded increment: each learned
candidate starts from the declared common width-eight initialization, computes
an equal mean of all twelve episode gradients at the same parameters, and takes
one Adam update per declared step. Normalize each post-burn phase variance by
its own clean open-loop mean variance. Reject zero/nonfinite denominators rather
than introducing an undeclared epsilon. Use the existing Gaussian training path;
truth remains confined to the simulator/loss. Evaluation stays exact Poisson/read.

Retain per-episode losses and gradients, parameter checkpoints, final objective
and explicit partial numerical failures in the returned record. Never silently
return the last finite model after a failed update. Invalid cohort inputs and
unexpected implementation errors raise; the future adapter must persist records
and distinguish those errors from a scientifically failed fit. This helper does
not provide durable resumption or authorize reserved-cohort acquisition.

Apply gstack small-change planning/review/shipping autonomously. Reuse existing
simulation, protocol, configuration and reference helpers, with one new numerical
module and tests; no baseline model, protocol, dependency or version change.
The test map is saved under the project's gstack directory. Use independent test
seeds and a reduced optical correctness case; do not claim a trained study model,
new physical smoke result or improvement before the complete runner is committed.

The bounded implementation passes all 175 local correctness tests (27 new).
Review confirmed one averaged update per full cohort, raw failure retention,
no usable model after failure, and final scoring after the last update. The
existing ledger already accepts ready/failed model statuses; the future adapter
must publish the returned arrays and bind their references, not pass the in-memory
record directly as ledger metadata. Original simulator, protocol and evidence
remain untouched. No reserved cohort or timing acquisition occurred.

## Study evidence storage and provenance (2026-10-13)

Before connecting the numerical helpers to stage commands, implement fresh source
and runtime capture plus immutable raw-array storage. Reuse the existing machine
fingerprint, pinned-study loader and StudyLedger; keep the baseline artifact and
simulation helpers unchanged. Bind the exact commit and canonical inventories on
create/resume. Inventory tracked execution inputs under src/, scripts/ and configs/
plus pyproject.toml and uv.lock, and reject dirty or untracked execution inputs,
including ignored Python/config files. Exclude the large historical result tree
from execution hashes; preserve its existing provenance separately.

Use deterministic numeric NPZ entries, flush before exclusive publication, and
allow only byte-identical retries. Preserve NaN/Inf in raw arrays while rejecting
objects/pickle and nonfinite JSON metadata. Raw publication must work inside the
ledger's locked evaluation callback without acquiring a second lock. Evidence
references become part of the future training/trial bindings; fresh inventory
capture is required at every process entry, not a claim of continuous monitoring.

Apply the same autonomous gstack planning, two-pass review and direct-main shipping
workflow. Test with temporary repositories and synthetic arrays, including failed
publication and concurrent writers. Do not generate reserved cohort arrays or new
scientific outcomes. Stage commands and a distinct full-runner smoke remain next.

The initial 36 storage checks passed. Review identified that ordinary Git status
can conceal modified execution files marked assume-unchanged or skip-worktree;
source capture now verifies every actual file against its committed Git blob as
well as retaining an independent SHA256 inventory. Added regression checks for
both flags and for a hash-matching archive that contains an object/pickle array.
The final local suite passes 214 tests (39 new) without scientific acquisition.
Runtime evidence also records zlib's version for deterministic archive retries.
The generic runtime inventory does not claim to hash every installed binary.

## Durable validation and evaluation trials (2026-10-13)

Connect the existing numerical helpers and exclusive evidence writer to bounded
ledger chunks. Add validation resumption with the same audit/skip/publication rules
as evaluation; factor their common locked execution path to avoid divergent stage
behavior. Construct the physics adapter lazily: no calibration, models or atmosphere
arrays before the ledger admits a callback. Failed training creates explicit failed
validation records without running physics. Unexpected implementation or I/O errors
stop the chunk and preserve previously published records and raw files.

Use one immutable parameters.npz per ready fitted model: predictive coefficients,
or learned parameter_0 through parameter_3. Other model artifacts retain histories.
Save shared atmosphere arrays by stage/wind/root, shared paired references by full
condition including physical fault, full controller streams and both metric windows.
Bind all raw/metadata references into each trial. Partial raw files are retried only
byte-identically; no replacement or dropped failure is allowed. Check free disk
before new physics; full-run sizing remains required before reserved acquisition.

Use gstack small-change planning, two-pass review and direct-main shipping under
the autonomous one-engineer instruction. Test small physical paths on separate
cohorts with explicitly synthetic frozen models. This is trial-stage integration,
not a complete runner or new controller result. Training orchestration, durable
mid-fit handling, CLI/resource reporting and a distinct full-runner smoke remain.

The focused ledger/adapter suite passed 68 tests, then the full suite passed 240
with additional learned-model loading/schema checks. A final focused regression
passed for selection mutation during an evaluation callback; its guard prevents
publishing a trial against a changed selection. The increment has 27 new checks
in total. No physical gate or model formula changed, no reserved cohort ran, and
no timing was acquired. The adapter requires the provenance references returned
by the study opener in the training binding, rather than trusting supplied base
settings or artifact names alone. Missing stage orchestration remains explicit.

## Durable learned-fit journal (2026-10-14)

Persist each completed learned episode evaluation before advancing. Resume by
feeding saved losses/gradients back through the existing fixed-budget fitting
loop, rebuilding Adam's state from its original initialization. This avoids both
repeating completed physical episodes and serializing executable optimizer state.
The protocol's same-parameter episode averaging, noise path, learning rate and
number of updates remain unchanged. Parameter/input identity checks must reject
incompatible replay, including changed atmosphere bytes or calibration arrays.

Store shared input phases, numerical calibration, step parameters, episode losses,
full-output finite flags and gradients with exclusive publication. Terminal model
and history references retain finite successes and nonfinite failures. Unexpected
exceptions leave interruption diagnostics and propagate; recorded scientific
failures are not retried. Hold the ledger lock throughout, and refuse refitting
once training or selection is bound. Reuse the trial adapter's provenance checks.

Use the autonomous gstack planning/review/shipping workflow. Verify analytic
interrupted-versus-uninterrupted fitting and a reduced optical case on separate
correctness seeds; do not generate reserved cohorts, timings or study outcomes.
Pilot orchestration, aggregate training/CLI commands, resource reporting and the
full-runner smoke remain separate work in P1.

The journal suite passes 39 checks and the full local correctness suite passes
280 tests. Interrupted analytic fits reproduce complete history/parameter bytes;
a reduced optical fit resumes after one Adam update on fresh correctness seeds.
Review added run/stage rechecks before terminal model publication, with regression
tests for mutation during fitting. No numerical gate, baseline plant or reserved
study input changed. A raw file without its identity metadata is not a committed
episode; resumption recomputes it and must reproduce its bytes. Interruption
logging is best effort if storage itself fails or the process is terminated.

## README as the maintained technical entry point (2026-10-14)

The owner requested a more rigorous research/founder voice and continuing README
updates. Interpret the Leiden/PhD/YC references as an editorial standard: technical
depth, a clear product thesis, evidence, reproducibility and restrained confidence.
They are not authorization to invent qualifications, institutional relationships,
funding, adoption, performance, an open-source license or public availability.

Restructure the README around the research question, implemented architecture,
preserved results, current study status and runnable entry points. Correct the
stale sensor-validation limit to include the completed independent ideal-pyramid
comparison. Keep the generated C1–C4 block byte-identical and retain reproduction
commands. Add README upkeep to repository instructions and the existing hourly
automation; update factual changes in the same increment rather than accumulating
another documentation backlog. This is documentation work, with no numerical or
scientific protocol change and no new acquisition.

The documentation check validated 40 local links/anchors and six shell blocks,
confirmed the live CLI choices, and reproduced all eight generated report sections
in memory from saved data without writing artifacts. The README headline block
is unchanged from the prior commit. Full scientific or numerical reruns are not
needed for this prose-only change; the regular remote correctness workflow still
runs after publication.

## Durable pilot and predictor records (2026-10-14)

Implement bounded pilot acquisition and terminal predictor records as the next P1
training stage. Reuse the existing exact-noise integrator, measured reconstruction,
per-episode predictor fits and exclusive evidence publisher. Audit saved outcomes
before new physics, retain full pilot streams and calibration, and share atmosphere
paths with learned fitting. Require the complete declared pilot cohort before any
predictor fit; a failed pilot makes its shared training data unusable for every
predictor. Record these failures rather than dropping episodes or changing seeds.

Expose known numerical reconstruction/fit failures through a typed exception that
retains their arrays, including partial coefficients on a failed linear solve.
Unexpected implementation or I/O errors propagate with best-effort diagnostics.
One ledger lock covers each public operation, with frozen-training/provenance
checks before work and before publication. The per-episode disk floor is not a
full-run capacity guarantee. The complete orchestrator, aggregate training freeze,
stage commands, resource costs and full-runner smoke remain P1 work; no reserved
cohort is acquired by this implementation increment. Use separate correctness
cohorts and the autonomous gstack planning/review/direct-main shipping workflow.

Validation: all 318 local correctness tests passed in 311.91 seconds; the 38 new
pilot/predictor tests also passed separately. The two-pass pre-landing review found
no additional blocking issues. Tests exercise actual reduced optical pilots and all
six predictors, interruption and drift rejection, numerical-failure retention and
shared atmosphere integration with the learned journal. They do not establish a
controller-performance result. Baseline raw evidence, simulator formulas and frozen
study configuration/plan are unchanged.


## Reader-facing repository structure (2026-10-14)

The owner requested a professional research/product repository with local agent
instructions removed from the visible source tree. Untrack AGENTS.md while
retaining an ignored local copy for autonomous work; do not rewrite historical
commits. Move the four development logs to docs/development and unsent upstream
drafts to contributions/drafts. Keep scientific artifacts and their original
provenance unchanged, and update current report paths and relative links.

The local checkout name noether-optics is not a nested directory in the remote.
Retain the Python package, distribution and CLI names for compatibility with pinned
environments and old commands; a fresh clone is already named noether-ao. Avoid a
cosmetic rename that invalidates virtual-environment paths or historical examples.

Reorganize README into a concise technical entry point, with detailed architecture
and reproduction guides and a deployment roadmap containing explicit evidence
gates. The owner's references to research institutions and venture success are
quality goals, not credentials or outcomes to claim. Stars, customers and deployed
performance are not invented. Private visibility remains unchanged and no license
is selected on the owner's behalf; public-release preparation is documented.


Validation: all 33 baseline reports/tables reconstructed with protected raw inputs
unchanged; generated C1–C4 headlines remain byte-identical. Verified 129 local
links, syntax of 13 shell blocks, eight generated sections and live CLI choices.
The README physical-validation example completed with all seven gates passing on
the existing reduced configuration. Check records are retained under
results/verification/repository-layout-20261014. The review corrected relative links
and the guide's working-directory wording after relocation; no blocking findings
remain. Queue a license/history/package audit before any public-release decision.

## Training coordination and aggregate freeze (2026-10-14)

Add one bounded coordinator rather than duplicating numerical training logic. It
must use the durable pilot, predictor and learned-fit entry points, preserve the
predeclared candidate order, count only new terminal model records against a chunk,
and leave `training.json` absent until all six predictors and all six learned
candidates have immutable success or failure records. Aggregate evidence includes
the terminal records and is deduplicated and sorted before the ledger freeze.

Use a separate nonblocking stage lock because each delegated public operation owns
the existing ledger lock for its complete transaction. This serializes coordinators
without weakening or nesting candidate locks. A bound-stage retry verifies the
frozen aggregate and performs no numerical work. Completed learned candidates use
persisted manifest/candidate/provenance/calibration/episode bindings and artifact
hashes; they do not recompute open-loop normalization merely to confirm completion.

Restore learned-fit callables by rebuilding the pinned calibration once when a
missing learned candidate is reached. Require exact shape, dtype and value agreement
for every stored numeric calibration member before fitting, and load phases only
from the audited pilot cohort. Common-input corruption remains an interruption for
inspection; candidate numerical failures remain terminal and eligible for aggregate
binding. This increment does not add stage commands, resource timings, reserved
cohort execution or a controller claim.

Validation: the focused pilot/journal/coordinator suite passes 85 checks, and the
complete local correctness suite passes 326 tests in 362.72 seconds. Eight new
coordinator checks cover the bounded barrier and its negative paths. No reserved
seed, benchmark timing or controller outcome was generated.

## Bounded training CLI (2026-10-14)

Expose the settled coordinator as `noether-optics study-train`, not as another
numerical implementation. Route creation/resumption only through `open_study_run`
and require explicit positive pilot and model limits before opening it. `--create`
selects a new directory; omission means an exact-bound resume. Emit the coordinator's
summary as sorted JSON so shell automation does not need to inspect internal files.

Use argparse subcommands rather than global conditional flags. This preserves the
existing command form while keeping `--create`, `--pilot-limit` and `--model-limit`
out of legacy experiments. Invalid limits and cross-command options must fail at
argument admission, before configuration loading, source capture or filesystem
mutation. Do not expose a free-space override: the opener's existing minimum remains
in force, while full-run sizing is still a separate pre-acquisition requirement.

The command is implementation wiring only. The later-stage commands are recorded
below. Do not execute the reserved study until resource/disk reporting and the
separately seeded full-runner smoke are committed and verified.

Validation: seven focused CLI tests pass, followed by all 333 local correctness
tests in 291.82 seconds. No study opener or numerical backend runs in the new tests,
and no reserved evidence or performance timing is acquired.

## Bounded validation CLI (2026-10-14)

Expose validation as a dedicated `noether-optics study-validate` subcommand that
can only resume an existing bound run. Require one positive `--limit` during argument
admission, open through `open_study_run`, and delegate to `validate_trials`. Emit the
ledger summary as sorted JSON for deterministic orchestration. Do not add `--create`:
the training command owns first-run creation and must freeze training before the
ledger admits validation.

Preserve lazy execution by passing the existing `_TrialExecutor` only through
`StudyLedger.validate_chunk`. This keeps training, provenance and selection checks
ahead of calibration, model and phase construction. The CLI adds no alternate trial
path, selection side effect, evaluation access, resource override or timing capture.

Validation: twelve focused CLI checks pass, followed by all 338 local correctness
tests in 98.56 seconds. The tests replace run opening and validation execution; no
reserved evidence, physical output or controller-performance result is acquired.
The selection command is recorded below; the evaluation command, full-run sizing
and the independent runner smoke remain preconditions for the reserved study.

## Bounded selection CLI (2026-10-14)

Expose the existing immutable selection transaction as `noether-optics study-select`.
The command can only resume a provenance-bound run and delegates directly to
`StudyLedger.freeze_selection`. It accepts no `--create` or chunk limit because
selection is a single deterministic barrier: every declared validation record must
already exist and pass integrity checks before `selection.json` can be published.

Keep candidate eligibility, rejection reasons, withholding and exact-tie ordering
inside the ledger's pretested pure selection path. The CLI must not load optical
physics, manufacture missing outcomes, alter validation, or release evaluation
identities. An identical retry remains an exact idempotent publication. Print the
full selection and binding hashes as sorted JSON for inspectable orchestration.

Validation: four new selection checks bring the focused CLI file to 16 passing tests,
followed by all 342 local correctness tests in 94.85 seconds. Invalid cross-stage
options are rejected before the mocked opener, and the valid path freezes the mocked
ledger exactly once. No reserved evidence, numerical output or controller result is
acquired. The evaluation command is recorded below; full-run sizing and the
independent runner smoke remain preconditions for the reserved study.

## Bounded evaluation CLI (2026-10-14)

Expose the existing held-out trial adapter as `noether-optics study-evaluate` with
one required positive `--limit`. The command can only resume a provenance-bound run;
it opens through `open_study_run`, delegates to `evaluate_trials`, and emits the
ledger's progress summary as sorted JSON. It has no creation or selection side effect.

Preserve evaluation laziness by entering only through `StudyLedger.evaluate_chunk`.
That gate must verify and recompute the frozen selection before releasing evaluation
identities or allowing calibration, model, atmosphere or noise construction. Chosen
controllers alone receive held-out cases; withheld controllers remain absent. Saved
successes and failures stay terminal and every record remains bound to the selection
hash. The CLI adds no alternate physics, resource override or timing capture.

Validation: five new evaluation checks bring the focused CLI file to 21 passing
tests, followed by all 347 local correctness tests in 90.79 seconds. Invalid input
is rejected before the mocked opener, and the valid path forwards the exact bound
ledger, base configuration and chunk size once. No reserved evidence, numerical
output or controller result is acquired. The four stage commands are now complete;
full-run sizing and the independent runner smoke remain preconditions for the
reserved study.

## Static whole-run storage admission (2026-10-14)

Size the complete frozen study from planned identities and persisted array schemas,
without sampling compression, running the simulator or opening a ledger. Count the
maximum storage path: every fitted model succeeds and all five controllers enter
evaluation. Preserve full trial outputs as designed; do not reduce scientific
evidence to fit the current workstation.

Report logical numeric bytes separately from the conservative admission envelope.
Assign zero credit to DEFLATE. For each NPZ, budget twice its numeric payload plus
4 KiB per member; add one eighth for JSON references and filesystem allocation with
a 512 MiB floor, one largest-artifact/512 MiB pending-write allowance, and a ten
percent/4 GiB operating reserve. These are declared engineering envelopes, not
measured compression ratios or compute costs.

Expose the calculation as read-only `study-resources --config ... --target ...`.
Print the report even on refusal and return status 2 when free bytes are below the
105,033,365,984-byte frozen requirement. Enforce the same report in
`study-train --create` and pass its threshold to the run opener for an immediate
second disk check. Do not expose a bypass flag. The current
development volume is insufficient, so reserved execution remains blocked while
the separate physical runner smoke proceeds on nonreserved seeds.

## End-to-end runner-smoke boundary (2026-10-14)

Use a new tracked smoke study rather than weakening the reserved protocol or treating
ledger fixtures as physical evidence. Keep the existing minimum cohort rules, but
reduce spatial dimensions, frames and candidate count. Assign distinct 61001, 62001,
63001 and 64001 seed families and explicitly exclude both reserved-study and previous
component-smoke streams, including derived detector keys.

Run the production command sequence in separate processes from clean committed
source. Complete all 12 training pilots, both fitted models and 100 validation trials
before freezing selection. Require at least one eligible controller, then retain one
held-out evaluation record. One record exercises the final lazy gate and immutable
publication without pretending to be a complete evaluation cohort. Do not time the
run, require favorable rankings or alter `controller_study_v1` after observing smoke
outcomes.

Protect the run and command results by SHA256 and rebuild a concise report. The gate
passes only with isolated streams, admitted storage, complete upstream stages, exact
selection re-entry, a physical evaluation record, expected stage-file counts and
matching source provenance. Commit the runner and inputs before acquiring evidence.

## End-to-end runner-smoke execution (2026-10-14)

Execute only after source commit `37f6bfc` passes remote correctness. The first
read-only admission check refused at 5.142 GB free, below the fixed 5.420 GB smoke
floor, and created no output. Clear only the regenerable UV package cache; do not
change the protocol or evidence envelope. Re-run the same check and proceed after it
reports 5.585 GB free and sufficient capacity.

Retain the resulting `controller_runner_smoke_v1` evidence after all nine gates pass:
12 pilots, two terminal fitted models, 100 validation records, frozen selection,
exact selection re-entry and one held-out record. Protect 488 command/run files and
require the derived report and summary to rebuild byte for byte. Record that all five
reduced candidates were eligible, without interpreting their scores as comparative
evidence. The run accessed no reserved stream, completed no evaluation cohort and
recorded no timing. Full-study storage admission remains unchanged and unsatisfied.

## Public-release audit boundary (2026-10-14)

Audit the source, complete Git history, retained numerical archives, direct
dependency terms and built wheel before making any public-release claim. Record
aggregate privacy findings without copying credential candidates or private values
into the report. Anchor package inspection to committed source and retain enough
machine-readable evidence to repeat the decision.

Treat a clean secret scan and deterministic wheel as necessary evidence, not release
authorization. The project owner must choose the project license, publication-history
strategy, treatment of workstation paths, dependency distribution contract, monitored
security route and disposition of unsubmitted drafts. Do not infer those choices from
the request for a professional repository.

Do not rewrite protected scientific artifacts to remove provenance paths because that
would invalidate recorded hashes. Do not rewrite private Git history merely because
former agent instructions and draft locations are historical. Implement explicit
owner choices in a separate bounded change, repeat clean installation and audit checks,
and change visibility only when directly authorized.

## Reclassify legacy M4 evidence for P2 development (2026-10-14)

Use the retained M4 fault cohort only to diagnose why the original detector missed
the simplified optical-gain ramp and to select a candidate statistic. Preserve every
original C4 result and hash, but permanently exclude those seeds from the new P2
held-out claim. Post hoc success on M4 cannot validate the candidate.

Screen the existing whitened residual in both directions before adding new physical
machinery. The legacy upper-tail score decreases under the gain fault; therefore carry
a four-frame lower-tail mean into protocol design as the first passive candidate. Its
0.005 quantile is recalibrated only from fresh training data. Do not transfer the
legacy numerical threshold into P2, and do not treat 5/5 old detections as an
acceptance result.

Keep fallback disabled while observability is measured. Report clean frame and run
false alarms, pre-onset alarms, misses, detected-only delay and paired controller
error before testing any intervention. A modal dither changes the plant, and an
auxiliary focal-plane estimator changes the sensing contract. If either is to follow
the passive candidate, freeze that branch and its science-impact control before
opening new evaluation outcomes.

## Freeze passive optical-gain observability v1 (2026-10-14)

Use four photon/wind regimes and new 310001, 320001 and 330001 seed families for
training, validation and evaluation. Pair conditions within each regime and exclude
all earlier scientific roots plus their detector-key derivatives. Calibrate a strict
0.005 lower threshold separately in each regime from 16 training-clean roots. Freeze
the causal four-frame mean and eight consecutive crossing rule before validation;
never carry the legacy numerical threshold forward.

Keep evaluation sealed unless all four regimes pass the complete validation gate.
The severe gain ramp must produce at least 29/30 qualifying detections with no
pre-onset alarm and persistence completed within the 32-frame drift. Clean validation
requires zero alarm runs, raw crossing fraction at most 0.01 and persistent alarm-frame
fraction at most 0.001. Treat these as engineering screens: zero clean alarms in 30
runs still permits a 9.50% one-sided 95% binomial upper bound.

The detector remains observational. Retain paired clean/fault controller errors, but
do not alter commands or test fallback. If v1 fails or is inadequate, the next route
is a new modal-dither protocol with fresh cohorts and a dither-free paired science
control. Independent focal-plane sensing remains a separate later contract; v1
held-out outcomes may not choose between those branches.

## Bind the v1 training threshold to raw evidence (2026-10-14)

Use one shared causal detector kernel for both the legacy read-only audit and the v1
runner. Perform the frozen mean in float64, preserve explicit statistic/alarm frame
identities, compare strictly below the per-regime threshold and expose persistence
only after all eight crossings exist.

Publish a threshold only from the exact 64-record training-clean cohort. Verify the
protocol from its source bytes and each supplied score from the SHA256-bound raw NPZ;
record the normalized score digest, raw reference, source identity and runtime
identity in deterministic case order. Any missing, foreign, duplicate, failed,
nonfinite or mismatched record prevents publication. Use exclusive creation so an
identical retry is harmless and a changed retry cannot replace the first freeze.

Treat the threshold builder and its synthetic tests as implementation evidence only.
They do not authorize cohort acquisition and do not create a P2 numerical threshold.

## Make validation release a one-way evidence decision (2026-10-14)

Release evaluation only from the exact 240 validation identities after reconstructing
the frozen thresholds from their 64 retained training artifacts. Recompute every
completed detector trace from the raw score and reject altered frame identities,
statistics, crossings or alarms. Apply the frozen clean, gain, delay and integrity
criteria independently in all four regimes; one failed criterion withholds the whole
evaluation cohort.

Treat numerical failure as a terminal validation outcome. Keep its identity and error
in the decision, count the run as missed or incomplete where applicable, and publish a
withheld result. An invalid completed artifact is an evidence error and cannot publish
a decision. Publish `validation_release.json` exclusively so a matching retry is
idempotent and a later favorable record cannot replace an earlier negative decision.

This barrier does not acquire validation or expose evaluation identities. Runner
resumption, full retention, storage admission and separately seeded smoke evidence
remain required before any declared P2 cohort is opened.

## Resume P2 stage records without replay (2026-10-14)

Use one run manifest to bind the complete observability protocol, its source-file
SHA256 and the supplied source/runtime provenance. Hold one nonblocking process lock
across each numerical callback and terminal record publication. Enumerate training and
validation cases only from the frozen protocol and write one canonical, exclusive JSON
record per identity in that declared order.

On every resume, audit all existing stage records before invoking a callback. A saved
success or numerical failure is terminal and may be skipped only after its manifest,
threshold where applicable, identity, raw path and artifact hash revalidate. An
interruption with no terminal JSON is retried and any already-published raw artifact
remains subject to exclusive byte identity. Reject symlinked record or artifact paths,
shared raw paths and changed evidence rather than repairing or replacing them.

Keep stage advancement separate from its freeze. Training may publish a threshold only
after all 64 terminal records exist and all completed successfully. Validation may
publish release or withholding only after all 240 terminal records exist; a numerical
failure remains evidence and yields withholding. Neither callback path exposes an
evaluation identity. Physical acquisition, complete retention, storage admission and
the distinct nonreserved smoke remain prerequisites to declared P2 execution.

## Size the complete P2 retention contract before physics (2026-10-14)

Derive storage only from the frozen protocol, its exact base-configuration hash and
explicit array shapes. Treat score, detector statistics and frame identities, physical
metrics, integrity flags and 100-mode command/applied streams as mandatory in every
trial archive. Add 64 by 64 measurement and prediction images only for the first seed
of every role/regime/condition cell, producing 32 full-WFS archives. Keep per-run WFE,
Strehl and saturation scalars inside the metadata envelope.

Size the worst case in which validation releases all 1,200 evaluation trials. Across
64 training, 240 validation and 1,200 evaluation trials, the 1,504 archives contain
2,341,413,792 bytes of numeric payload. Reuse the controller study's conservative
envelope: twice raw payload plus 4 KiB per NPZ member, one-eighth metadata allowance,
temporary-write headroom, at least 4 GiB operating reserve and zero compression credit.
The resulting admission threshold is 10,190,604,096 free bytes. Failed or withheld
trials may reduce use but never reduce the up-front requirement.

Expose this as the read-only `observability-resources` command. It must import neither
JAX nor the simulator, create no target and return exit status 2 after printing the
report when space is insufficient. The first live check measured 7,994,904,576 bytes
free, refused admission and left the intended target absent. This creates no timing,
threshold, validation decision or P2 outcome. Physical retention and a distinct smoke
remain required before declared acquisition.

## Split pre-threshold training evidence from derived detector traces (2026-10-14)

Do not fabricate training crossings before the regime threshold exists and do not
replace a raw archive after calibration. A physical training callback retains the
threshold-independent score, variance, Strehl, integrity flags and command streams,
plus measurement/prediction images under the existing first-seed policy. It binds
post-burn-in WFE, Strehl and saturation scalars in canonical metadata. Only after all
64 clean records complete does the freeze derive each threshold and publish a separate
five-array detector trace for every training record. Schema-2 `thresholds.json` binds
the raw, metadata and detector hashes as one calibration chain. Validation, which
already receives the frozen threshold, keeps all 13 frame arrays in one archive.

Treat a produced nonfinite or negative physical stream as a terminal numerical
failure with its raw archive and failure metadata retained. Treat a malformed identity,
shape, dtype, callback contract or changed artifact as an execution/evidence error and
do not publish a terminal outcome. Revalidate every saved record semantically when a
chunk opens; while the process lock remains held, use byte/hash snapshots after each
new callback so large earlier WFS arrays are not decoded quadratically.

The split adds 64 numeric archives but duplicates no declared array. The maximum
study is therefore 1,504 trials represented by 1,568 numeric archives. Numeric payload
and the conservative admission result remain exactly 2,341,413,792 and
10,190,604,096 bytes. A separate nonreserved physical smoke remains mandatory before
any declared P2 seed is opened.

Require that same whole-study capacity at every public physical chunk boundary. Count
filesystem blocks already allocated under the immutable run together with current free
space, so retained evidence does not consume its own admission credit while unrelated
disk use can still stop acquisition. Keep a second per-trial headroom check immediately
before numerical construction.

For a new physical run, perform the static refusal before importing JAX or capturing
runtime state. Once admitted, require a clean committed execution tree and retain
canonical source, runtime, protocol and base JSON below `evidence/provenance/`. Bind
their digests into the ledger manifest and recheck the stored content before every
numerical callback. Keep the bare ledger constructor as a low-level synthetic-test
boundary; the physical entry points require the provenance artifacts. Load the base
YAML through the same hash-checking byte path during ledger construction, provenance
capture and validation release so no check-then-reread window can substitute data.

## Observability runner-smoke boundary (2026-10-14)

Use a dedicated tracked protocol on the existing reduced 8-pixel, four-mode,
8-pixel-detector plant. Keep the production physical adapter, ledger, threshold and
validation-decision code unchanged while reducing the loop to 24 frames and one
photon/wind regime. Assign new 710001, 720001 and 730001 seed families; reserve both
their derived detector streams and every frozen P2 v1 stream.

Complete two clean training trials, freeze and exactly re-enter the threshold, then
complete two paired clean/gain validation roots and freeze and exactly re-enter the
decision. Retain every required frame array, metric, provenance object and terminal
record. Require the resource envelope, durable stage counts, declared report fields,
protected-file hashes and source binding to pass. A favorable detector result is not
a smoke gate: retain either release or withholding and report it only as a reduced
implementation outcome.

Do not acquire any smoke evaluation trial. A smoke validation artifact cannot open,
alter or authorize the frozen `optical_gain_observability_v1` evaluation cohort. Run
only from clean committed source, outside CI and without timing. The static reduced
envelope requires 5,370,101,856 free bytes; admission remains a real gate even though
the numeric payload is only 237,616 bytes.

## Observability runner-smoke execution (2026-10-14)

Execute from source commit `a997753` only after its local 518-test suite and remote
Correctness workflow pass. The first read-only preflight refused at 2,387,918,848 free
bytes and created no run. Remove only regenerable uv, npm, Homebrew and Bun caches;
the unchanged gate then admits at 5,519,986,688 free bytes.

Retain the complete two-record training cohort, one immutable threshold, four paired
validation records and one immutable validation decision. All 14 implementation gates
pass, including separate streams, storage admission, complete stage counts, exact
threshold and decision re-entry, artifact hashes, source provenance, declared metrics
and zero evaluation access. Protect 29 command/run files; both derived files reproduce
byte for byte.

Preserve the unfavorable smoke outcome without tuning: one of two clean runs alarmed
with 8/34 raw crossings and 1/20 persistent alarm frames; neither gain-0.35 run earned
a qualifying detection, one had a boundary-straddling alarm, and the decision is
`evaluation_withheld`. All outputs are finite with zero fallback, unavailable or
saturated frames. Paired gain-minus-clean WFE deltas are 0.6758110717 and
0.0076200481 nm; Strehl deltas are -0.0037580778 and -0.0000616811. These six reduced
physical trials establish runner integrity only. They do not validate the detector,
authorize v1, select an escalation or provide timing evidence.
