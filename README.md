# noether-ao

**Differentiable adaptive optics for model-based control and runtime assurance.**

`noether-ao` is Noether Labs' research software for testing how an optical forward
model can support control and fault detection in an autonomous observatory. It
combines dLux/JAX propagation, a delayed adaptive-optics (AO) simulation, controller
comparisons and experiments that retain the evidence behind each conclusion.

The objective is a controller whose predictions, commands and failures can be
examined and reproduced. The current implementation is a simulation and benchmark
framework; observatory deployment remains a research objective.

[Results](#measured-evidence) · [Quickstart](#quickstart) ·
[Architecture](#system-architecture) · [Controller study](#current-controller-study) ·
[Reproduction](#reproduce-the-published-tables) · [Research queue](TODOS.md)

## Research question

When does a differentiable optical model earn its place inside a feedback loop?
The answer depends on more than propagation speed. A useful model must extract
information from noisy wavefront-sensor (WFS) observations, account for delayed
commands, improve correction within compute constraints, and expose mismatches
without making recovery worse.

This repository makes those questions testable: synchronized latency distributions,
paired controller trials, independent optical checks and fault-injection experiments.
For runtime assurance, detecting a mismatch is only part of the question: the
resulting control action must also help. The fault studies therefore report both
detection and paired fallback performance, including harmful recovery decisions.

## System architecture

The simulated plant combines frozen-flow turbulence, a modal deformable mirror,
an ideal unmodulated pyramid sensor, photon/read noise and a delayed command queue.
Controllers include a classical integrator, a temporal predictor, forward-model
estimation, and fixed or learned mixtures of forward and inverse commands.

| Layer | Responsibility | Implementation |
| --- | --- | --- |
| Optical plant | Propagation, atmosphere, sensor response and delayed closed-loop dynamics | [optics](src/noether_optics/optics.py), [atmosphere](src/noether_optics/atmosphere.py), [simulation](src/noether_optics/simulation.py) |
| Numerical experiments | Latency, physical gates, model fidelity, controller mixtures and fault response | [latency](src/noether_optics/latency.py), [experiments](src/noether_optics/experiments/) |
| Study physics | Measured pilot telemetry, predictor fitting, exact-noise trials and fault-matched open-loop references | [controller_physics](src/noether_optics/controller_physics.py) |
| Learned fitting | Common initialization, normalized episode losses and fixed-budget Adam updates | [controller_training](src/noether_optics/controller_training.py) |
| Evidence and recovery | Source/runtime bindings, immutable numeric artifacts, resumable fits and trial records | [evidence](src/noether_optics/controller_evidence.py), [fit journal](src/noether_optics/controller_training_journal.py), [trial adapters](src/noether_optics/controller_trials.py) |
| Selection | Complete validation, eligibility checks and frozen models before held-out evaluation | [study ledger](src/noether_optics/controller_selection.py) |

The distinction between software correctness, physical agreement and scientific
performance is part of the design. A passing test checks an implementation contract.
A physical gate checks a declared model property. A controller claim requires
retained trials and an explicit comparison with uncertainty.

## Measured evidence

The completed v0 experiments address propagation latency (C1), model fidelity
under photon noise (C2), learned controller mixtures (C3), and fault detection with
fallback (C4). The following table is generated from the preserved baseline runs.

<!-- BEGIN HEADLINES -->

| Claim | Measured verdict |
| --- | --- |
| C1 | Conditional support. N=32, one wavelength, float32: p99 0.0592937 ms; N=256, five wavelengths: p99 6.38013 ms. CPU component only. |
| C2 | No tested flux gave useful equivalence across all fidelity levels. No supported low-flux threshold was found; lower fidelity often performed better under this fixed estimator. |
| C3 | Learned weights beat both integrator and forward in 0/12 conditions and lose to at least one in 5/12; remaining comparisons include uncertainty. |
| C4 | Partial detection/recovery, no general safety result. Optical-gain drift missed; vibration fallback worsened WFE; clean false alarms occurred. |

<!-- END HEADLINES -->

The [result report](docs/results.md) contains the experiment definitions, paired
intervals, raw-data links and limits for each verdict. [Physical acceptance gates](docs/acceptance.md)
were declared before acquisition. Configuration values and planned trial counts
are inputs; reported measurements come from saved runs with source and machine
provenance. Negative findings remain part of the release.

Subsequent validation adds two separate kinds of evidence:

- **Independent ideal-pyramid agreement.** The [saved comparison](results/pyramid_validation_v1/report.md)
  passes 36 intensity cases, 30 derivative cases and 24 deliberate-error controls
  against an HCIPy matrix-integral reference. The [protocol](docs/pyramid_validation.md)
  covers matched finite-grid sensor conventions, rather than an instrument model.
- **Controller component wiring.** The [saved smoke run](results/controller_physics_smoke_v1/README.md)
  passes all nine declared gates; reports rebuild byte-for-byte with protected raw
  evidence unchanged. It uses a reduced plant and an untrained network, so it
  establishes component behavior without establishing learned-control performance.

## Current controller study

The next comparison addresses limitations of the baseline: fixed classical gains,
ideal predictor training telemetry and a small held-out cohort. Its
[predeclared protocol](docs/controller_study_v1.md) gives each controller six
meaningful candidates, separates training/validation/evaluation streams, uses
measured predictor telemetry and specifies simultaneous paired intervals. The
[configuration](configs/controller_study_v1.yaml) and [trial plan](plans/controller_study_v1.json)
are frozen inputs. Reserved study outcomes have not yet been acquired.

| Component | Current state |
| --- | --- |
| Protocol and selection | Implemented: equal candidate budgets, complete validation, eligibility gates, deterministic selection and no refit after selection |
| Numerical training | Implemented: measured pilot reconstruction, separate episode predictor fits and normalized learned fitting from common initialization |
| Provenance and storage | Implemented: fresh source/runtime capture, pinned inputs, immutable numeric artifacts and verified loading |
| Interrupted learned fits | Implemented: saved episode losses/gradients replay through Adam; completed evaluations are reused and numerical failures remain terminal |
| Validation/evaluation trials | Implemented: bounded ledger chunks, shared atmospheres, fault-matched references, full raw outputs and retained failed outcomes |
| Complete study runner | Pending: pilot/model orchestration, aggregate training binding, stage commands, resource reporting and full-run disk sizing |
| Scientific comparison | Pending: a distinct full-runner smoke, then acquisition on reserved cohorts and the predeclared analysis |

The [execution contract](docs/controller_study_execution.md) describes the boundary
between these components. Validation must finish before selection is frozen;
held-out evaluation is admitted only after the source, configuration, models and
prior evidence pass integrity checks. Interrupted work must preserve earlier
records. A failed candidate cannot be replaced with a favorable restart.

See [STATUS.md](STATUS.md) for recorded checks and [TODOS.md](TODOS.md) for the
next bounded research tasks. [Correctness CI](https://github.com/noetherlabs/noether-ao/actions/workflows/correctness.yml)
runs software checks without acquiring benchmark timings.

## Quickstart

Use Python 3.11 and uv. Repository access currently requires an authorized GitHub
account. From a fresh clone:

```sh
git clone https://github.com/noetherlabs/noether-ao.git
cd noether-ao
uv sync --frozen --python 3.11
CI=true JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q
```

The lock pins dependency versions and Git sources. Install the optional
cross-library benchmark dependencies when reproducing the Dragrace contribution:

```sh
uv sync --frozen --extra dragrace --python 3.11
```

The repository is named `noether-ao`; the package and command retain
`noether_optics` / `noether-optics` for compatibility with recorded environments.
Set `JAX_ENABLE_X64=1` before importing JAX when using the numerical modules
directly; the experiment CLI sets it itself.

Saved timing runs used an Apple M1 CPU. GPU execution and enforced single-core
affinity were unavailable on that host and are recorded as skipped. Thread
environment flags are not treated as enforced affinity.

## Run baseline experiments

Run these from this directory. Each output directory must be new. The full
configuration is [simulation.yaml](configs/simulation.yaml); seeds, noise,
atmosphere, delays, fault settings and acceptance tolerances are explicit.

```sh
# M1: synchronized timing; run alone, outside CI
.venv/bin/noether-optics latency --config configs/latency.yaml --out runs/m1
# M3: independent physical checks and three closed-loop controllers
.venv/bin/noether-optics validate --config configs/simulation.yaml --out runs/m3
# M4: detector, all faults, and paired fallback comparisons
.venv/bin/noether-optics faults --config configs/simulation.yaml --out runs/m4
# M5: common plant, three twin fidelities, five photon levels
.venv/bin/noether-optics fidelity --config configs/simulation.yaml --out runs/m5
# M6: train modal weights, then evaluate all controllers on held-out seeds
.venv/bin/noether-optics mixed --config configs/simulation.yaml --out runs/m6
```

M5 references the preserved M1 component measurements at `results/m1`. To use
new timings, copy the simulation configuration and change `fidelity.m1_result`
to `runs/m1`. These are single FFT propagation costs, not whole pyramid sensor,
gradient, controller or hardware-loop timings. The optical loop has a simulated
millisecond timestep; that does not establish millisecond wall-clock execution.

These entry points reproduce the v0 experiment designs. The predeclared controller
study above still requires its own complete stage commands.

## Local Dragrace contribution

The separate local branch adds one AO MFT scan and appends p99 and observed
maximum to existing timing reports. It keeps the original physics and accuracy
gate. The patch is [contributions/dragrace.patch](contributions/dragrace.patch).
To reconstruct the branch in a new sibling directory:

```sh
git clone https://github.com/Jashcraf/Diffraction-Dragrace ../Diffraction-Dragrace
git -C ../Diffraction-Dragrace checkout -b codex/ao-tail-latency 1bf53802950c5f76905d3dee0ba73b75147401bf
git -C ../Diffraction-Dragrace apply "$PWD/contributions/dragrace.patch"
NOETHER_PYTHON="$PWD/.venv/bin/python"
(cd ../Diffraction-Dragrace && JAX_ENABLE_X64=1 PYTHONPATH=src "$NOETHER_PYTHON" -m pytest -q)
.venv/bin/python scripts/run_dragrace.py --repo ../Diffraction-Dragrace --out runs/m2
```

This runs every installed adapter. An installed adapter may still report an
unsupported case. Timing never runs in CI. No `reviewed_by` field was filled.
The issue and PR descriptions in [outbox](outbox/) are unsent upstream drafts.
This private repository archives the patch; it does not submit it upstream.

## Reproduce the published tables

In a fresh local clone, install from the lock and reconstruct the Dragrace sibling
as above. Then run:

```sh
.venv/bin/python scripts/rebuild_all.py --dragrace ../Diffraction-Dragrace --report results/verification/rebuild.json
```

This recomputes tables and figures from the preserved raw arrays and timing
distributions, checks every published numerical table, and verifies that raw
inputs did not change. It does not claim that new timing samples will equal old
ones. Full experiments are rerun with the commands above.

For shorter functional checks of the experiment entry points, substitute
`configs/reproduction_smoke.yaml` for the simulation configuration and
`configs/latency_smoke.yaml` for the timing configuration, using fresh output
directories. These smoke results are excluded from scientific claim tables.
The timing smoke still takes the required number of samples at its selected point.
The verification report records exactly which checks ran in the fresh checkout.

To run that verification sequence yourself (including fresh measurements):

```sh
.venv/bin/python scripts/verify_checkout.py --dragrace ../Diffraction-Dragrace --out runs/verification
```

It reruns the full physical validation and the full Dragrace AO case; other
entry-point checks use the labeled shorter configurations. Run it outside CI,
with no concurrent compute work, because it includes timing.

The completed [fresh-checkout verification](results/verification/fresh-checkout/verification.json)
passed. Its [table report](results/verification/fresh-checkout/table-reproduction.json)
and [independent result review](results/verification/fresh-checkout/independent-result-review.json)
record unchanged raw inputs and numerical agreement of the full physical rerun.

## Limits

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

## Development and evidence policy

Developed by Noether Labs for civil-astronomy research. Contributions should make
an experiment easier to reproduce, a comparison more informative, or a failure
easier to diagnose. New experiments use new output directories; baseline data,
source identities and negative findings stay intact. Changes to capabilities,
commands, evidence or limitations update this README in the same increment.

[CONTRIBUTING.md](CONTRIBUTING.md) defines the validation and delivery workflow.
The [decision log](DECISIONS.md) records assumptions and interruptions;
[research notes](docs/research_notes.md) document sources and experimental design.
The [plan](PLAN.md), [milestone history](docs/history.md) and
[changelog](CHANGELOG.md) distinguish completed work from the research ahead.
