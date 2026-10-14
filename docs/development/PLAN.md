# Plan

Reader entry points: [README](../../README.md), [documentation](../README.md) and
[deployment roadmap](../../ROADMAP.md).

Current position: M0–M7 complete. All required available-platform gates passed; negative claim results retained.

Independent ideal-pyramid validation passed under its predeclared
[protocol](../pyramid_validation.md). The fair controller-validation
[protocol](../controller_study_v1.md), frozen inputs and seed/search checks are
now prepared. The [execution ledger](../controller_study_execution.md) implements
complete validation checks, immutable records and the frozen-selection barrier.
Numerical helpers now supply measured training telemetry, predictor fits and
paired physical scoring. Their separate [component smoke](../../results/controller_physics_smoke_v1/README.md)
passes all predeclared gates with reproducible raw evidence. Learned fitting now
implements complete-episode gradient averaging and fixed Adam budgets with retained
checkpoints and numerical failures. Fresh source/runtime capture and exclusive
raw-array storage now bind evidence to the ledger. Durable validation/evaluation
trial adapters now save raw outputs and paired references through resumable ledger
chunks. Learned fits now journal each episode's loss/gradient and replay saved
episodes through Adam after interruption; terminal successes and failures are
retained. Pilot acquisition now runs in audited chunks and predictor fits return
terminal coefficient/history records from complete measured cohorts. A bounded
coordinator now schedules every fitted candidate and freezes the aggregate training
bundle only after all terminal evidence verifies. A provenance-bound `study-train`
command exposes positive pilot/model chunks and JSON progress. A separate
`study-validate` command now admits positive chunks through the existing lazy ledger
gate. A resume-only `study-select` command audits and freezes the complete validation
verdict. `study-evaluate` now enters positive held-out chunks only through that frozen
selection gate. A static whole-run preflight now derives 42.167 GB of numeric payload
and a conservative 105.033 GB free-space requirement without accessing any cohort.
The committed full runner passed all nine gates on the distinct
61001/62001/63001 smoke streams and retained one held-out record after complete
training, validation and selection. It is an execution gate, not a comparison
cohort. The current development filesystem still fails the full-study capacity
gate; provision a qualifying target before any reserved outcome is acquired.
The full repository and passing correctness CI are already on main. Delivery
remains commits only; the follow-up cadence is now hourly.
Each meaningful increment also updates README.md when its technical description,
reproduction instructions, evidence or current study status changes.
Original M0–M7 evidence stays immutable; later experiments receive new run IDs.

The unexecuted P2 optical-gain protocol now has a causal detector, complete-cohort
threshold and validation barriers, and a protocol-bound ledger for deterministic
training/validation record resumption. Its physical adapter now applies the frozen
forward-controller and fault mapping, retains every declared array and post-burn-in
metric, and preserves threshold-dependent training traces without rewriting raw
acquisitions. The static storage gate derives 2.341 GB of numeric payload and a
conservative 10.191 GB free-space requirement; the current development target fails
that gate. The next bounded work is a distinct nonreserved end-to-end smoke on an
admitted filesystem. No declared P2 seed is opened before that gate passes.

| Milestone | Required gate | State |
|---|---|---|
| M0 | Opened-source notes and design for every experiment | pass |
| M1 | dLux/JAX agreement; full specified timing sweep and raw samples | pass on available CPU; affinity/GPU skipped |
| M2 | Small local Dragrace branch; existing/new tests pass; drafts | pass |
| M3 | HCIPy PSF, fitting scaling, delay response, stable integrator | pass |
| M4 | Every injected fault; delay, false alarm, ROC, fallback comparison | pass; C4 partial |
| M5 | Fidelity/flux table and plot with at least five seeds | pass; no equivalence threshold |
| M6 | Held-out controller comparison; fixed and trained modal weights | pass; C3 not established |
| M7 | Reproduction commands, claims, plots, limits and private drafts | pass; fresh clone verified |

For each milestone: read source, specify gate, implement, pytest, seeded run,
store provenance, review full diff, commit locally. Gates are never relaxed.
After three distinct unsuccessful attempts, record failure and proceed.

## Architecture and test map

    YAML -> validated config -> immutable seeded arrays -> JAX optical plant
                                      |                        |
                                      v                        v
                                 provenance               delayed WFS frame
                                      |                        |
                               raw NPZ/JSON <- metrics <- controller -> command FIFO
                                                            |
                                      noise-normalized residual -> latched fallback

The remaining P2 pre-acquisition action is to execute the committed reduced
observability runner protocol from clean source, preserve its complete protected
evidence and prove byte-exact report reconstruction. It uses only nonreserved
710001/720001/730001 families and acquires no evaluation trial. A favorable smoke
detector outcome is not required and cannot authorize the v1 cohort.

Tests: independent Fourier/HCIPy agreement; sensor energy and derivative checks;
screen spectral scaling; delay FIFO impulse; controller stability; causal detector;
fault routing; held-out split; timing synchronization, quantiles and artifact checks.
Timing runs run alone. Training and physics validation run after timing ends.

## What already exists

Reuse dLux Wavefront propagation, JAX autodiff/scan, HCIPy independent propagation,
and Dragrace case loading, stored distributions, correctness and backend gates.
Keep Dragrace changes to the scan, report, tests and supporting documentation.

## Not in scope

Hardware deployment, on-sky validation, public publication, safety certification,
institutional collaboration claims, and copying private correspondence.

## Review decisions

The requested scope is substantial and deliberately retained. Keep a small functional
simulation core separate from experiment drivers. Public artifacts identify synthetic
plant assumptions. Validate physical gates separately from numerical software tests.
Do not turn an unavailable platform or failed physical validation into a pass.

## Current implementation review

The delayed plant, ideal pyramid, sensor calibration, causal detector, fidelity
sweep and differentiable mixture are prepared. M3 physical gates passed; all held-out controller comparisons are recorded.
The review checks explicitly distinguish pre-existing false alarms from new fault
detections, failed control from useful fidelity equivalence, component propagation
latency from a full control deadline, and training from held-out evaluation seeds.
