# Architecture

The simulated plant combines frozen-flow turbulence, a modal deformable mirror,
an ideal unmodulated pyramid sensor, photon/read noise and a delayed command queue.
Controllers include a classical integrator, a temporal predictor, forward-model
estimation, and fixed or learned mixtures of forward and inverse commands.

| Layer | Responsibility | Implementation |
| --- | --- | --- |
| Optical plant | Propagation, atmosphere, sensor response and delayed closed-loop dynamics | [optics](../src/noether_optics/optics.py), [atmosphere](../src/noether_optics/atmosphere.py), [simulation](../src/noether_optics/simulation.py) |
| Numerical experiments | Latency, physical gates, model fidelity, controller mixtures and fault response | [latency](../src/noether_optics/latency.py), [experiments](../src/noether_optics/experiments) |
| Study physics | Measured pilot telemetry, predictor fitting, exact-noise trials and fault-matched open-loop references | [controller_physics](../src/noether_optics/controller_physics.py) |
| Learned fitting | Common initialization, normalized episode losses and fixed-budget Adam updates | [controller_training](../src/noether_optics/controller_training.py) |
| Pilot data and predictors | Bounded pilot acquisition, measured telemetry audits and terminal predictor records | [controller_pilots](../src/noether_optics/controller_pilots.py) |
| Training barrier | `study-train` dispatch, deterministic candidate scheduling and complete aggregate model/evidence freeze | [CLI](../src/noether_optics/cli.py), [training stage](../src/noether_optics/controller_training_stage.py) |
| Evidence and recovery | Source/runtime bindings, immutable numeric artifacts, resumable fits and trial records | [evidence](../src/noether_optics/controller_evidence.py), [fit journal](../src/noether_optics/controller_training_journal.py), [trial adapters](../src/noether_optics/controller_trials.py) |
| Validation barrier | `study-validate` dispatch, complete-training admission and positive bounded trial chunks | [CLI](../src/noether_optics/cli.py), [trial adapters](../src/noether_optics/controller_trials.py), [study ledger](../src/noether_optics/controller_selection.py) |
| Selection | `study-select` dispatch, complete validation audit, eligibility checks and frozen models before held-out evaluation | [CLI](../src/noether_optics/cli.py), [study ledger](../src/noether_optics/controller_selection.py) |
| Evaluation barrier | `study-evaluate` dispatch, frozen-selection admission and positive bounded held-out chunks | [CLI](../src/noether_optics/cli.py), [trial adapters](../src/noether_optics/controller_trials.py), [study ledger](../src/noether_optics/controller_selection.py) |
| Resource preflight | Static array-schema accounting, shared conservative archive envelope and read-only target-filesystem admission | [controller sizing](../src/noether_optics/controller_resources.py), [observability sizing](../src/noether_optics/observability_resources.py), [envelope](../src/noether_optics/resource_envelope.py), [CLI](../src/noether_optics/cli.py) |
| Optical-gain evidence gates | Protocol-bound stage resumption, causal detector, complete-cohort threshold freeze and one-way validation release | [stage ledger](../src/noether_optics/observability_run.py), [detector and calibration](../src/noether_optics/observability_detector.py), [validation release](../src/noether_optics/observability_validation.py), [frozen protocol](optical_gain_observability_v1.md) |

The distinction between software correctness, physical agreement and scientific
performance is part of the design. A passing test checks an implementation contract.
A physical gate checks a declared model property. A controller claim requires
retained trials and an explicit comparison with uncertainty.

The storage preflight stays outside the numerical dependency graph:

    frozen plan/protocol + pinned base -> planned cases + saved-array schemas
                                           -> retained evidence envelope
                                           -> temporary headroom + reserve
                                           -> target free-space admission

It imports no simulator, generates no atmosphere or detector stream and assigns no
benefit to data-dependent compression. Elapsed compilation, fitting and trial costs
remain measurements for an isolated run; they are not inferred by this estimator.

## Model boundary

- The plant is a small unobstructed pupil, modal DM, ideal unmodulated pyramid and
  finite periodic frozen-flow atmosphere. Independent checks cover sampled PSFs
  and the matched finite-grid pyramid response. Focal-support convergence and
  instrument validation remain outside that evidence.
- Baseline control results use short runs and a small held-out seed set. The
  baseline predictor uses ideal training telemetry and classical gains are fixed.
  The new study addresses these comparison limits; its outcomes remain pending.
- Learned weights use Gaussian pathwise training noise and exact Poisson/read
  noise at evaluation. This mismatch and the finite training budget remain limits.
- The detector has no proven safe set or recovery guarantee. Fault coverage and
  useful fallback in some cases do not establish safety in general.
- CPU latency distributions include desktop scheduling effects. No GPU, on-sky,
  hardware acquisition or hard real-time result is claimed.
