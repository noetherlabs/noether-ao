# Contributing

This private repository contains a reproducible civil-astronomy research prototype.
Use the [PR series](docs/pr-series.md) to find the latest complete branch. Keep
changes focused, describe the problem and resulting behavior, and include the
checks actually run. Use `codex/` branches and explicit PR dependencies.

## Correctness

```sh
uv sync --frozen --python 3.11
CI=true JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q
```

GitHub Actions uses the same core suite. It does not run experiment timing,
training sweeps, full scientific acceptance runs or the optional Dragrace adapters.
Install `--extra dragrace` and follow README for local cross-library validation.
Rendering existing samples is permitted in CI; acquiring timing samples is not.

## Evidence

Preserve committed raw data, provenance, configuration and original commit hashes.
Run changed experiments in a new output directory with an explicit configuration,
fixed seeds and source/machine records. Keep training, validation and evaluation
seeds disjoint. Declare acceptance tolerances before looking at new held-out
results, and preserve all failed gates, losses and platform skips.

Separate unit-test success, physical acceptance, scientific claims and wall-clock
performance. A simulated 1 kHz loop is not a measured 1 kHz controller. Do not run
timings concurrently with other compute work. Regenerate affected tables from raw
data and retain the comparison report. Never tune on an evaluation cohort already
used to make a claim.

Update PLAN.md, STATUS.md, DECISIONS.md and TODOS.md when the work changes their
facts. Citation URLs must be opened and checked before use. Private correspondence
must never enter Git. Dragrace submission drafts remain in outbox until separately
authorized; a PR here does not confer upstream or institutional approval.

## Review and integration

Review stacked PRs in order, retarget the next PR to main after its predecessor is
merged, and preserve measured commit history with merge commits. Do not squash or
rebase the historical milestone commits. Do not delete prerequisite branches while
other open PRs still target them. New automated work opens or updates PRs; it does
not automatically merge them.
