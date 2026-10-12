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
