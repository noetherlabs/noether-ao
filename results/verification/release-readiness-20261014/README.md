# Release-readiness evidence

This directory retains the machine-readable public-release audit for source commit
`fa8a8b8cf7f5f4bf35ff4b002119e05984592a00`.

[`audit.json`](audit.json) records aggregate results from the complete-history
credential scan, retained NPZ inspection, dependency and notice inventory, locked
environment check, deterministic wheel build, isolated command-surface inspection and
correctness run. It intentionally contains no secret values, credential candidates,
personal addresses or absolute workstation paths.

The interpreted verdict and required owner decisions are documented in
[`docs/release_readiness.md`](../../../docs/release_readiness.md). The verdict is not
ready for public release. This record grants no license, changes no repository
visibility and does not authorize publication.
