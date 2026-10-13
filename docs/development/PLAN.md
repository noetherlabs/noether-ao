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
bundle only after all terminal evidence verifies. Next add stage commands, complete
resource/disk sizing, and verify a full runner smoke before reserved outcomes.
The full repository and passing correctness CI are already on main. Delivery
remains commits only; the follow-up cadence is now hourly.
Each meaningful increment also updates README.md when its technical description,
reproduction instructions, evidence or current study status changes.
Original M0–M7 evidence stays immutable; later experiments receive new run IDs.

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
