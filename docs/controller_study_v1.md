# Fair controller-validation protocol v1

This is a predeclared plan, not a controller result. Commit this protocol and
its checked configuration before implementing the runner or generating training,
validation or evaluation outcomes. The original M0–M7 data and claims stay intact.

## Scope and frozen inputs

Compare all five existing controllers: integrator, predictive, forward, fixed
mixture and learned mixture. Inherit the exact plant, sensor, noise, command limits,
two-frame physical delay, estimator settings and turbulence from the SHA256-pinned
`configs/simulation.yaml`. Use `configs/controller_study_v1.yaml` for overrides.
No setting is selected by searching old M3–M6 outcome tables.

The fixed regimes are the Cartesian product of 1,000 and 1,000,000 photons/frame
with wind multipliers 0.5 and 2.0. These declared endpoint/stress regimes span dim
and bright sensing and slow and fast flow; they are not predictions of a favorable
result. Validation and evaluation use 512 frames with 128 burn-in frames at the
existing 1 ms simulated timestep. Training uses 256 frames with 64 burn-in frames.
The longer runs still do not establish long-duration observatory behavior.

Evaluate clean, optical-gain-drift and vibration conditions after selection. Faults
start at frame 256 and end at 512; remaining fault settings stay pinned to the base
configuration. No fallback is enabled. This isolates control performance from the
separate detector/fallback question. Both the full post-burn-in interval and the
fault interval must be reported; the former is the primary comparison interval.

## Separate stochastic cohorts

The configuration reserves three training roots (21001–21003), five validation
roots (22001–22005), ten evaluation roots (23001–23010) and initialization root
24001. These are explicit inputs, not measured values. Baseline recorded seeds
are excluded. Check both atmosphere roots and sensor roots (`seed + 100000`),
including uint32 overflow, rather than merely checking the three root lists.
Initialization may be reused across candidate networks to give them the same
starting weights; it must not reuse any atmosphere or detector stream.

Within a cohort/condition, every candidate uses the same atmosphere arrays and
noise keys. Photon observations can differ when feedback changes intensity.
Never concatenate separate seeds when fitting temporal predictors. The validation
runner must not load or generate evaluation arrays. Freeze selected candidate
IDs, coefficients/weights, the complete configuration and source hashes before a
separate evaluation command can access evaluation seeds. No refit after selection.

## Equal search opportunity

Each controller gets six distinct candidates and the same 20 validation trials per
candidate: four clean regimes times five validation seeds. The declared grids are:

| Controller | Candidate axes |
| --- | --- |
| Integrator, forward, fixed, learned | Gains 0.1/0.3/0.5 × leaks 0.99/0.999 |
| Predictive | Horizons 1/2/3 frames × predictor ridge 0.001/0.01 |

The existing predictive command ignores the gain setting; sweeping it would create
duplicate experiments. Its six settings instead affect the actual fitted predictor.
Prediction horizon is a fitting target, not a change to physical command delay.
Forward-estimator step size/count, fixed-weight formula, network width, reconstructor
and clipping limits are held constant. Candidate order is deterministic and is the
predeclared exact-tie breaker. One candidate per controller is selected globally
across clean regimes, never separately after seeing each evaluation condition.

Equal candidate/validation budgets do not mean equal compute expense or equally
expressive algorithms. Record synchronized compilation, fitting/training and trial
elapsed costs separately, outside CI and without concurrent compute. These are
study resource costs, not per-frame latency distributions or deadline guarantees.

## Training access

Generate one common pilot integrator dataset with gain 0.3 and leak 0.999 on all
12 training root/regime combinations, using exact Poisson/read noise. Fit each
predictor candidate on measured pseudo-open-loop estimates (`applied command +
reconstructed residual`), not ideal phase coefficients. Fit each sequence after
burn-in independently, then average its coefficients with equal episode weights.
The existing `predictor_for` helper uses ideal telemetry and is not an acceptable
runner for this protocol without a new measured-telemetry path.

Train a separate learned network for each of its six candidates, from the same
initial weights, using exactly 20 Adam steps at learning rate 0.02. Every step
averages loss/gradients equally over the same 12 training episodes; process episodes
sequentially if needed to bound memory. The loss is post-burn-in phase variance
divided by that episode's open-loop variance. Simulator truth is available only
for this training loss and later scoring, never as a controller measurement.

Keep the existing Gaussian pathwise noise approximation for network training and
exact Poisson plus read noise for all validation/evaluation. This mismatch, the
single initialization and finite training budget remain explicit limitations.
No extra restarts, longer training or candidate additions after validation outcomes.
Stop and record nonfinite loss/gradients; do not conceal a failed candidate.

## Selection and held-out reporting

A candidate is eligible only if every validation trial is finite, its command
saturation fraction is at most 0.01, and its WFE is below the paired open-loop
WFE in every trial. Minimize mean `WFE / open-loop WFE` with equal trial weights
among eligible candidates. Retain every rejected candidate and reason. If none
is eligible, record validation failure and withhold that controller's evaluation;
do not replace it with a default or relax the gate. No superiority claim against
a withheld controller is possible.

Maximum budgets are 600 validation trials and 600 selected-controller evaluation
trials, excluding shared open-loop/pilot records. These are planned counts only.
The paired open-loop reference uses the same atmosphere and physical disturbance,
including injected vibration, with zero DM command; a clean open loop must not be
substituted for a faulted one. Optical-gain drift changes sensing, not true phase.
Report every held-out regime and seed, WFE, Strehl, nonfinite failures, saturation,
and whether control beats its own open loop. Never discard failing seeds to form
a favorable confidence interval. Failed or missing pairs get no numerical
superiority verdict, and finite but unsuccessful trials remain in the tables.

Use paired seed-level WFE differences within each evaluation condition. The
predeclared family contains all ten controller pairs across twelve conditions:
120 comparisons. Provide Bonferroni simultaneous Student-t intervals using family
alpha 0.05 and retain that denominator even if some controllers are withheld.
Ordinary 95% intervals may be labeled exploratory. A win requires the simultaneous
interval to exclude zero in the favorable direction and all trials of the claimed
winner to pass the usefulness/saturation gates. Do not pool frames as independent
samples, claim universal dominance from selected conditions, or replace C1–C4.

## Implementation sequence and review

    protocol + pinned base -> seed/parameter/budget validation -> plan only
                                      |
                            future separate runner
                                      v
                     training -> clean validation -> frozen selection
                                                            |
                                     sealed evaluation <----+
                                            |
                                full retained outcomes + report

This increment adds one pure protocol module, one planning command and tests.
Tests must reject direct and derived seed collisions, reused legacy streams,
overflow, duplicates, an ignored predictive-gain axis, unequal search budgets,
baseline configuration drift and evaluation-noise/refit changes. It generates no
scientific outcomes and changes no runtime controller behavior.

Next implement staged/resumable execution, training-telemetry plumbing, complete
trial identity checks and an evaluation command that requires a frozen selection.
Use unseen, separately labeled smoke seeds for implementation checks; never use
reserved study seeds to debug. Write raw trials immediately, retain full pilot
estimates and learned-training histories, and retain per-frame metrics for every
trial. Preserve shared atmospheres once per wind/root and sufficient representative
sensor/command streams to audit causality. Check free disk before running; an
interruption must preserve existing hashes and resume only identical source/config.

## Reproduce the plan

```sh
.venv/bin/python scripts/plan_controller_study.py --out runs/controller-study-plan.json
```

The output is marked `not_executed`, records input hashes, enumerates candidate
identities and seed streams, and computes planned budgets. An existing output
file is never overwritten. CI may check this plan; it must not run the study or
collect benchmark timings. The historical model and independent sensor checks
remain separate prerequisites rather than proof of controller superiority.
