# Noether AO

**Differentiable adaptive optics / reproducible closed-loop control.**

JAX and dLux models of wavefront propagation, frozen-flow turbulence, deformable mirrors and pyramid wavefront sensing. Classical, predictive and learned controllers run against shared delays, sensor noise and fault conditions.

[Results](docs/results.md) · [Architecture](docs/architecture.md) · [Reproduction](docs/reproduction.md) · [Research roadmap](ROADMAP.md)

## Experiments

- **Optical physics:** independently check sampled PSFs and pyramid-sensor intensities and derivatives against HCIPy.
- **Sensor calibration:** compare independent response matrices, held-out reconstruction and calibration transfer with HCIPy and SPECULA, retaining native-model disagreement.
- **Control:** compare integrator, temporal predictor, forward-model and learned-mixture policies with separated training, selection and evaluation data.
- **Degradation:** inject optical-gain drift, vibration, sensor faults and control delays; measure both detection and the cost of fallback.

Current investigations extend the measured-telemetry pathway toward controller-aware system identification, calibrated observation noise and instrument replay. These directions are not released instrument integrations.

## Measured evidence

The initial C1–C4 experiments retain raw measurements, configurations and non-winning results.

<!-- BEGIN HEADLINES -->

| Claim | Measured verdict |
| --- | --- |
| C1 | Conditional support. N=32, one wavelength, float32: p99 0.0592937 ms; N=256, five wavelengths: p99 6.38013 ms. CPU component only. |
| C2 | No tested flux gave useful equivalence across all fidelity levels. No supported low-flux threshold was found; lower fidelity often performed better under this fixed estimator. |
| C3 | Learned weights beat both integrator and forward in 0/12 conditions and lose to at least one in 5/12; remaining comparisons include uncertainty. |
| C4 | Partial detection/recovery, no general safety result. Optical-gain drift missed; vibration fallback worsened WFE; clean false alarms occurred. |

<!-- END HEADLINES -->

The CPU component timings are not full sensor-to-actuator latency measurements. Methods and paired uncertainty are in the [results report](docs/results.md).

## Active studies

| Study | Record |
| --- | --- |
| Pyramid wavefront sensing | [Protocol](docs/pyramid_validation.md) · [Independent comparison](results/pyramid_validation_v1/report.md) |
| Wavefront-sensor calibration | [Protocol](docs/wfs_calibration.md) · [HCIPy/SPECULA comparison](results/wfs_calibration_v1/README.md) |
| Frozen controller comparison | [Protocol](docs/controller_study_v1.md) · [Runner checks](results/controller_runner_smoke_v1/README.md) |
| Optical-gain observability | [Protocol](docs/optical_gain_observability_v1.md) · [Audit](results/optical_gain_observability_audit_v1/README.md) |
| Scheduled controller handover | [Methods](docs/scheduled_fallback.md) · [Paired simulation results](results/scheduled_fallback_v1/README.md) |
| Closed-loop response identification | [Methods](docs/identification.md) · [Development measurements](results/identification_component_smoke_v1/README.md) · [Instrument comparison](results/controller_filtered_iv_v1/README.md) |

The full controller study and optical-gain validation remain in progress. [ROADMAP.md](ROADMAP.md) tracks acceptance and resource gates.

## Run

Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/noetherlabs/noether-ao.git
cd noether-ao
uv sync --frozen --python 3.11
.venv/bin/noether-optics validate --config configs/reproduction_smoke.yaml --out runs/first-validation
CI=true JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q
```

The reduced validation is a functional check, not a reproduction of the full experimental campaigns. See the [reproduction guide](docs/reproduction.md) for complete runs and provenance.

## Release

Current research results use simulated optical systems; instrument qualification and hard real-time performance require separate measurements. The repository is publicly visible but project-owned code is not yet licensed for redistribution. [Release requirements](ROADMAP.md#public-release).
