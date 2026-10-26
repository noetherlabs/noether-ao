# Controller physics component smoke

All nine predeclared wiring gates passed from clean source
`dc16e2f171de47eb1ec45d634fb0c63d3c771178`. This is a smaller 16-pixel/eight-mode,
24-frame component check. It is not the reserved controller study, a fair
comparison, learned training, a timing measurement or a superiority result.

The run stores 12 noisy pilot episodes spanning the separate three-seed/four-regime
training cohort, six predictor candidates fitted per episode before averaging,
15 controller/fault trials and three zero-DM reference checks. Every controller
uses its first declared candidate. The learned controller uses **untrained initial
weights**, retained explicitly in the raw data.

| Check | Observed value | Predeclared gate |
| --- | --- | --- |
| Maximum measured-pilot reconstruction error | 9.43689570931383e-16 | <= 1e-12 absolute |
| Maximum predictor rebuild error | 0.0 | <= 1e-12 absolute |
| Maximum zero-DM variance/Strehl error | 1.2212453270876722e-15 | <= 1e-12 absolute |
| Variance error when vibration is omitted | 1.0852514490041327 | > 1e-8; deliberate error rejected |

All 15 controller/fault outputs are finite and no fallback occurs. Coverage gates
also pass. Finite output is not a useful-control or performance gate. Per-condition
WFE, paired open-loop WFE, saturation and Strehl remain in [trials.json](trials.json)
for both post-burn and fault windows; they are not selected-controller estimates.

[summary.json](summary.json) contains every gate and measured check.
[index.json](index.json) identifies the retained raw arrays and their seeds,
configurations and candidate parameters. Six shared training atmospheres are saved
once per wind/root, separately from the smoke validation atmosphere. Calibration,
full observation/command streams, per-episode coefficients and untrained network
parameters are retained in 48 NPZ files. The [configuration](config.json),
[derived smoke base](derived-smoke-base.yaml), [machine](machine.json) and
[source provenance](provenance.json) identify the exact run.

[verification.json](verification.json) records a subsequent rebuild: both generated
JSON reports are byte-identical and all 53 files in the
[protected inventory](protected_sha256.json) are unchanged. Reproduce with:

```sh
.venv/bin/python scripts/smoke_controller_physics.py --rebuild results/controller_physics_smoke_v1
```

The complete study runner still needs learned fitting, real study provenance,
durable stage integration with the ledger and a separate end-to-end smoke before
using any reserved study seeds. No original C1–C4 claim or baseline evidence changes.
