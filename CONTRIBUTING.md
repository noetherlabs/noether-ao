# Development

Use Python 3.11 and the locked environment:

```sh
uv sync --frozen --python 3.11
CI=true JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q
```

For direct numerical-module use, set `JAX_ENABLE_X64=1` before importing JAX.
The experiment CLI sets it itself. Optional comparison dependencies are described
in the [reproduction guide](docs/reproduction.md).

## Numerical changes

Include a focused regression for a corrected behavior. Changes to optical
sampling, units, normalization, noise or delay require an independent numerical
check at the affected boundary. Unit tests alone do not establish physical
agreement or controller performance.

Use new output directories for new experiments. Declare configurations, seeds
and comparison criteria before collecting results. Keep training, validation and
evaluation data separate; retain failures and negative outcomes. Report timing
measurements outside CI and distinguish simulated time from wall-clock time.

Rebuild affected reports from saved data and check that numerical inputs remain
unchanged. Describe what changed, why, and which checks ran.

## Distribution

Project-owned code does not yet have an open-source license or external
contribution agreement. Public availability alone does not grant redistribution
rights. The retained Diffraction-Dragrace patch includes its upstream MIT notice.
