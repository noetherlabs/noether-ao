# Plan

Current position: M1 complete; M2 adapter installation, tests and scan next.

| Milestone | Required gate | State |
|---|---|---|
| M0 | Opened-source notes and design for every experiment | pass |
| M1 | dLux/JAX agreement; full specified timing sweep and raw samples | pass on available CPU; affinity/GPU skipped |
| M2 | Small local Dragrace branch; existing/new tests pass; drafts | in progress |
| M3 | HCIPy PSF, fitting scaling, delay response, stable integrator | pending |
| M4 | Every injected fault; delay, false alarm, ROC, fallback comparison | pending |
| M5 | Fidelity/flux table and plot with at least five seeds | pending |
| M6 | Held-out controller comparison; fixed and trained modal weights | pending |
| M7 | Reproduction commands, claims, plots, limits and private drafts | pending |

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
sweep and differentiable mixture are prepared. Their physical gates remain pending.
The review checks explicitly distinguish pre-existing false alarms from new fault
detections, failed control from useful fidelity equivalence, component propagation
latency from a full control deadline, and training from held-out evaluation seeds.
