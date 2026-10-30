# Noether AO

Differentiable adaptive-optics simulation and controller comparisons in JAX.

Noether AO models wavefront propagation, frozen-flow turbulence, deformable
mirrors and pyramid wavefront sensing. It compares classical, predictive and
learned controllers under shared delays, sensor noise and injected faults.
Independent optical references use HCIPy and SPECULA.

## Installation

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/noetherlabs/noether-ao.git
cd noether-ao
uv sync --frozen --python 3.11
.venv/bin/noether-optics validate --config configs/reproduction_smoke.yaml --out runs/validation
```

This reduced example checks the experiment pipeline. Full experiment commands
and optional dependencies are in the [reproduction guide](docs/reproduction.md).

## Experiments

| Experiment | Methods | Results |
| --- | --- | --- |
| Pyramid optics and derivatives | [Optical model](docs/pyramid_validation.md) | [HCIPy comparison](results/pyramid_validation_v1/report.md) |
| Wavefront-sensor calibration | [Calibration protocol](docs/wfs_calibration.md) | [HCIPy/SPECULA comparison](results/wfs_calibration_v1/README.md) |
| Closed-loop response identification | [Estimators](docs/identification.md) | [Identification](results/identification_component_smoke_v1/README.md), [filtered instruments](results/controller_filtered_iv_v1/README.md) |
| Scheduled controller handover | [Measurement design](docs/scheduled_fallback.md) | [Paired trials](results/scheduled_fallback_v1/README.md) |
| Optical-gain detection | [Frozen protocol](docs/optical_gain_observability_v1.md) | [240-run validation: evaluation withheld](results/optical_gain_observability_v1/README.md) |
| Control, noise and faults | [Physical criteria](docs/acceptance.md) | [Baseline comparisons](docs/results.md) |

## Baseline findings

<!-- BEGIN HEADLINES -->

| Claim | Measured verdict |
| --- | --- |
| C1 | Conditional support. N=32, one wavelength, float32: p99 0.0592937 ms; N=256, five wavelengths: p99 6.38013 ms. CPU component only. |
| C2 | No tested flux gave useful equivalence across all fidelity levels. No supported low-flux threshold was found; lower fidelity often performed better under this fixed estimator. |
| C3 | Learned weights beat both integrator and forward in 0/12 conditions and lose to at least one in 5/12; remaining comparisons include uncertainty. |
| C4 | Partial detection/recovery, no general safety result. Optical-gain drift missed; vibration fallback worsened WFE; clean false alarms occurred. |

<!-- END HEADLINES -->

These are simulation and CPU component measurements. They do not establish
instrument performance, full sensor-to-actuator latency or safe fallback.
The [results report](docs/results.md) gives methods, uncertainty and limitations.

## Documentation

- [Architecture](docs/architecture.md)
- [Methods and study protocols](docs/README.md)
- [Reproducing results](docs/reproduction.md)
- [Development and tests](CONTRIBUTING.md)

The source is publicly available. No open-source license has been granted for
project-owned code; third-party notices retain their stated terms.
