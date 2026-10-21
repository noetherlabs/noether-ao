# Noether AO

**Differentiable adaptive optics and reproducible control experiments.**

Noether AO is research software for studying optical control under sensor noise,
command delay and model mismatch. Built on dLux and JAX, it combines an optical
plant, classical and learned controllers, and reproducible experiment protocols.

[Quickstart](#quickstart) · [Results](#measured-evidence) ·
[Documentation](docs/README.md) · [Architecture](docs/architecture.md) ·
[Contributing](CONTRIBUTING.md)

## Overview

- **Optical simulation:** differentiable propagation, frozen-flow turbulence, a
  modal deformable mirror and an ideal pyramid wavefront sensor.
- **Control:** integral, predictive and forward-model controllers, with fixed and
  learned mixtures evaluated in the same delayed plant.
- **Fault experiments:** sensor and controller faults, residual-based detection,
  and paired comparisons of continued control and fallback.
- **Reproducibility:** explicit configurations, disjoint study cohorts, retained
  raw measurements and deterministic reconstruction of reported results.

## Quickstart

Use Python 3.11 and [uv](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/noetherlabs/noether-ao.git
cd noether-ao
uv sync --frozen --python 3.11
```

Run the reduced validation example:

```sh
.venv/bin/noether-optics validate \
  --config configs/reproduction_smoke.yaml \
  --out runs/first-validation
```

This writes physical checks to `gates.json`, controller metrics to `table.md`,
and a comparison figure to `result.png`. Use a new output directory for each run.
The reduced example checks functionality; it does not reproduce the full studies.

Run the correctness tests:

```sh
CI=true JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q
```

The [reproduction guide](docs/reproduction.md) covers complete experiments,
optional dependencies, resource requirements and reconstruction from saved data.

## Measured evidence

The completed v0 experiments examine propagation latency (C1), optical-model
fidelity (C2), learned controller mixtures (C3), and detection with fallback (C4).
The following summary is generated from retained results.

<!-- BEGIN HEADLINES -->

| Claim | Measured verdict |
| --- | --- |
| C1 | Conditional support. N=32, one wavelength, float32: p99 0.0592937 ms; N=256, five wavelengths: p99 6.38013 ms. CPU component only. |
| C2 | No tested flux gave useful equivalence across all fidelity levels. No supported low-flux threshold was found; lower fidelity often performed better under this fixed estimator. |
| C3 | Learned weights beat both integrator and forward in 0/12 conditions and lose to at least one in 5/12; remaining comparisons include uncertainty. |
| C4 | Partial detection/recovery, no general safety result. Optical-gain drift missed; vibration fallback worsened WFE; clean false alarms occurred. |

<!-- END HEADLINES -->

See the [results report](docs/results.md) for methods, paired uncertainty and raw
evidence. Latency measurements describe CPU propagation components on an Apple M1,
not a complete control-loop deadline.

## Research studies

| Study | Scope and status | Documentation |
| --- | --- | --- |
| Pyramid sensor validation | Finite-grid intensity and derivative agreement with an independent HCIPy reference. | [Protocol](docs/pyramid_validation.md) · [Report](results/pyramid_validation_v1/report.md) |
| Controller comparison | Frozen training, selection and evaluation protocol. Full comparison pending; reduced execution checks completed. | [Protocol](docs/controller_study_v1.md) · [Runner checks](results/controller_runner_smoke_v1/README.md) |
| Optical-gain observability | Frozen detection study. Training is incomplete; detector performance is not validated. | [Protocol](docs/optical_gain_observability_v1.md) · [Exploratory audit](results/optical_gain_observability_audit_v1/README.md) |

Implementation checks and scientific results are reported separately. The
[controller component checks](results/controller_physics_smoke_v1/README.md) and
[observability runner checks](results/optical_gain_observability_runner_smoke_v1/README.md)
verify execution on reduced configurations; they do not establish controller
superiority or detector performance.

## Documentation

- [Architecture](docs/architecture.md) — optical model, controllers and experiment boundaries.
- [Reproduction](docs/reproduction.md) — installation, commands and resource planning.
- [Results](docs/results.md) — measured findings, uncertainty and limitations.
- [Documentation index](docs/README.md) — protocols and reference material.
- [Roadmap](ROADMAP.md) — planned research and instrument integration.

```text
src/noether_optics/   Optical models, controllers and experiment runners
configs/             Simulation settings and frozen study configurations
plans/               Declared experiment plans
tests/               Numerical, physical and integrity checks
results/             Retained measurements, reports and provenance
scripts/             Reproduction and reporting tools
docs/                Methods, protocols and architecture
```

The Python package is `noether_optics`; the command-line interface is
`noether-optics`.

## Scope and limitations

The experiments use a simulated optical plant. They do not establish on-sky
performance, instrument validation or hard real-time operation. The current
results do not demonstrate a general advantage for learned control or guarantee
that fallback improves performance. Model assumptions and comparison limits are
documented in the [architecture guide](docs/architecture.md#model-boundary).

## Contributing and licence

Developed by Noether Labs. See [Contributing](CONTRIBUTING.md) for development and
validation requirements, and the [changelog](CHANGELOG.md) for changes.

This repository does not currently include an open-source licence. See the
[release status](ROADMAP.md#public-release) for the outstanding requirements.
