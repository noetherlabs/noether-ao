# Contributing

Noether AO accepts changes that improve the reliability of an experiment, the
fairness of a comparison, or the diagnosis of a failure. Start with the
[architecture](docs/architecture.md) and [roadmap](ROADMAP.md). The repository is
publicly visible but does not yet include an open-source licence. A declared
licence and contribution policy remain prerequisites for accepting external
contributions. See the [release status](ROADMAP.md#public-release).

## Development setup

Use Python 3.11 and uv from the repository root:

```sh
uv sync --frozen --python 3.11
CI=true JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q
```

The lock pins packages and Git dependencies. For direct numerical-module use, set
`JAX_ENABLE_X64=1` before importing JAX; the experiment CLI sets it itself.
The [reproduction guide](docs/reproduction.md) covers optional cross-library
dependencies and complete experiment commands.

## Validation

Include a focused check for changed numerical behavior, failure handling or an
integrity boundary. Run the affected tests during development and the core suite
before integration. GitHub Actions runs the core correctness suite. It does not
run timing acquisition, scientific sweeps or optional Dragrace adapter validation.

For scientific changes, declare the physical acceptance criteria before acquiring
results. Passing unit tests does not establish physical agreement or controller
performance. Report platform skips and failed checks explicitly.

## Research evidence

- Preserve baseline raw data, provenance, configurations and measured commit hashes.
- Use new output directories, explicit configurations and fixed seeds for new runs.
- Keep training, validation and evaluation cohorts disjoint; never tune on results
  already used for an evaluation claim.
- Retain numerical failures, negative findings and all declared comparisons.
- Run `study-resources` against the intended volume before a full controller study;
  do not reduce retained evidence or assume compression to pass the capacity gate.
- Distinguish simulated time from measured execution time. Collect timings outside
  CI, with no concurrent compute workload.
- Rebuild affected reports from saved arrays and verify that protected inputs are
  unchanged. Check citations against the sources they support.

Keep changes small enough to review. Describe the concrete problem, resulting
behavior, checks actually run and remaining limitations. Fetch the current main
branch before integration and preserve other contributors' work and measured
history.

## Documentation

Keep the README focused on the project, installation, stable capabilities and a
concise summary of measured findings. Put experimental methods, acquisition
progress, resource accounting and stage-by-stage execution details in the relevant
protocol or report. Link to those documents rather than copying their contents
into the README.

Preserve the generated `HEADLINES` section and verify its reconstruction. Check
local links and shell examples. Keep implementation notes and task coordination
out of public-facing documentation.

Keep local agent instructions and scheduler settings out of tracked source.
Private correspondence must never enter Git. The [Dragrace patch](contributions/dragrace.patch)
and its [drafts](contributions/drafts/) are unsubmitted upstream work and do not
imply maintainer or institutional approval.
