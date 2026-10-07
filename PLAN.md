# Plan

Current position: M0 research/design complete; M1 correctness tests passed; full latency run next.

| Milestone | Required gate | State |
|---|---|---|
| M0 | Opened-source notes and design for every experiment | pass |
| M1 | dLux/JAX agreement; full specified timing sweep and raw samples | in progress |
| M2 | Small local Dragrace branch; existing/new tests pass; drafts | pending |
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
