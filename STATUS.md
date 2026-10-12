# Final status

All required available-platform gates passed. Negative scientific findings remain
negative. The owner subsequently authorized publication to the private
`noetherlabs/noether-ao` repository, then requested commits only on main. Upstream
Dragrace drafts and correspondence remain unsent. See [milestone history](docs/history.md)
and [next work](TODOS.md).

| Milestone | Gate | Evidence and reason |
| --- | --- | --- |
| M0 | pass | Pinned environment, opened-source notes and design in docs/research_notes.md. |
| M1 | pass on available CPU; platform skips | 144 rows, 10000 samples each, 1440000 calls; maximum complex-field relative L2 error 0. GPU and enforced single-core affinity unavailable. |
| M2 | pass | Clean four-file Dragrace change d194dab; 140 tests passed, two affinity skips; seven adapters pass all six sizes. PROPER installed but MFT unsupported. Both drafts exist. |
| M3 | pass | HCIPy PSFs, fitting laws, delay response and stable integrator gates pass. Full fresh-clone rerun reproduces all six numerical CSV tables. |
| M4 | pass for coverage | All 80 fault/switch/seed trials finite; delays, false alarms, ROC and paired errors saved. This is not a safety gate. |
| M5 | pass for coverage | All 75 trials finite; five seeds per cell, raw data, uncertainty and plot saved. No useful-equivalence threshold found. |
| M6 | pass for coverage | All 300 held-out trials finite; disjoint seeds, finite nonzero gradients and changed weights. All comparisons retained. |
| M7 | pass | Fresh clones install from lock; both suites pass; 33 report/table files regenerate with unchanged raw inputs; every experiment entry point succeeds. |

## Claim results

- **C1: configuration-dependent support.** Float32 dLux CPU device-path p99 is
  0.0592937 ms at N=32 with one wavelength and 6.38013 ms at N=256 with five.
  A small median is not a deadline guarantee. No GPU or full-loop timing claim.
- **C2: proposed noise-equivalence threshold not established.** No flux passes the
  predeclared useful equivalence criterion across all three fidelities. Lower
  fidelity often performs better with this fixed estimator; conditioning, modes,
  sampling and photon noise are confounded.
- **C3: learned advantage not established.** Paired intervals show learned weights
  beating both integrator and forward in 0/12 conditions and losing to at least
  one in 5/12. Training loss reduction is not held-out improvement.
- **C4: partial detection/recovery, no general safety result.** Optical-gain drift
  is missed. Vibration fallback worsens WFE. One of five clean runs alarms, with
  one alarm frame out of 600 evaluated clean frames. Availability fallback helps
  in its synthetic test; pupil-shift fallback does not restore nominal correction.

The nominal pyramid predictor reduces mean WFE by 26.372725760970482 percent.
This is a new configured result, not exact replication of an unspecified earlier
setup. It does not generalize to every noise/fault condition in M6.

See [the claim report](docs/results.md) for numbers, plots and caveats, and
[the README](README.md) for installation and reproduction commands.

## Reproduction evidence

[Fresh-checkout verification](results/verification/fresh-checkout/verification.json)
records every command, configuration, fingerprint and gate. The fresh suites have
26 Noether passes and 140 Dragrace passes, with two platform skips. Noether's two
fresh-import warnings are upstream zodiax docstring escape deprecations.

M3 was rerun with the full scientific configuration; every original numerical CSV
agrees at the declared strict comparison tolerance. M2 reran its full scan, with
the same accuracy/support outcomes. M1 uses a selected-point smoke with the full
sample count. M4–M6 use shorter functional configurations. These smoke outputs
are separate from the claim tables. All raw verification outputs are retained.

The M1 interruption and M5 disk-space recovery are recorded beside their raw data.
M5 appended only four missing trials from the exact original source and identical
runtime; the first 71 raw trial files were unchanged. The source-inventory audit
also records two unrecoverable, unused AO draft snapshots from M2; its executed
driver/artifact code and numerical dependencies are recoverable.

## Drafts needing human review

- [Dragrace issue](outbox/dragrace_issue.md)
- [Dragrace PR description](outbox/dragrace_pr.md)
- [Summary and three questions for Louis](outbox/summary_for_louis.md)

## Three largest correctness risks

1. Ideal plant/twin optics and a finite periodic atmosphere. Independent open-loop
   PSF and matched finite-grid sensor agreement do not validate a real instrument
   or establish convergence as the focal support grows.
2. Short runs and few seeds, ideal predictor telemetry, fixed baseline gains and
   Gaussian training versus Poisson evaluation noise limit fairness/generalization.
3. Component timings and synthetic availability faults do not establish real-time
   or physical safety. Missed faults, harmful fallback and false alarms are observed.

## Private publication follow-up

The complete original history is published on main in
`noetherlabs/noether-ao`; setup commits add portable instructions, correctness-only
CI and the follow-up queue. Eight milestone PRs were opened before the owner
changed the preference to commits only; they are historical review records.
Current local publication checks pass: 29 Noether
tests, 140 Dragrace tests with two platform skips, and 33 reproduced table/report
files with raw inputs unchanged. Evidence is in `results/publication`.

The app has an active hourly follow-up for this task, configured to address
checks and then one bounded item from TODOS.md per run. It publishes focused
commits to main, creates no PRs, and reports meaningful progress or changed blockers.

## Independent ideal-pyramid validation

The [saved comparison](results/pyramid_validation_v1/report.md) passes all
predeclared gates: 36 pixel cases, 30 derivative cases and 24 deliberate-error
controls on grids 32/64/96 with one or three wavelengths. Maximum intensity
relative L2 error is 1.6942335379634155e-14; maximum derivative relative L2 error
is 9.360885028878882e-11. Maximum absolute unit-power error is
9.769962616701378e-15. The smallest deliberate error is 0.031291468420221256,
above the required 1e-4 rejection threshold.

The independent implementation uses HCIPy matrix Fourier integrals, physical
surface phase and pixel powers. No alignment, scale or output normalization was
fitted. Native HCIPy default centering/filter/parity differences are explicit in
[the protocol](docs/pyramid_validation.md). This is a finite-grid ideal-model
validation, not an instrument or safety result.

The clean-source run points to commit `56ed79f`; its frozen configuration matches
the earlier protocol commit `e11d3bb`. Verification reproduced nine report files
byte-for-byte with all nine protected data/provenance files unchanged. The core
suite now passes 42 tests. Original M0–M7 runtime code, raw data and claims are
unchanged.

## Fair controller-comparison protocol

The [new protocol](docs/controller_study_v1.md) is predeclared with a hash-pinned
base configuration, fresh training/validation/evaluation seeds, meaningful equal
candidate budgets, frozen selection and simultaneous paired intervals. Its
[generated plan](plans/controller_study_v1.json) is explicitly unexecuted: all
trial counts are planned budgets, not observed results.

Validation rejects direct/derived seed collisions, legacy reuse, uint32 overflow,
duplicate or unequal search candidates, an inert predictor-gain search, changed
base settings and changes to evaluation noise/refitting. No production controller,
baseline raw artifact or C1–C4 verdict changed. The next task is staged runner
implementation and smoke verification before generating any reserved-cohort data.
The core suite passes 71 tests, including 29 protocol checks. The generated plan
reproduces byte-for-byte, and its existing-file guard preserves a frozen output.

## Controller-study execution safeguards

The [study ledger](docs/controller_study_execution.md) now implements deterministic
selection, complete validation-identity checks, model and open-loop bindings,
immutable records and a gate that verifies frozen evidence before any evaluation
callback runs. Recorded failures are retained and skipped on resume; interrupted
chunks preserve earlier records. Controllers without eligible candidates are
withheld without changing the predeclared comparison family.

The local core suite passes **117 tests**, including **46 new ledger tests**.
They cover changed source/runtime/configuration, altered artifacts, incomplete or
duplicate trials, failed training, nonfinite metrics, interruption, failed atomic
publication, concurrent writers, path escapes and completed-run resumption.
All new test records are synthetic fixtures on separate seeds, not physical
smoke data or scientific outcomes. No reserved study cohort was simulated, no
timings were acquired, and the base model, frozen protocol/plan and original
evidence are unchanged.

The complete scientific runner is still pending: actual provenance capture,
measured pilot telemetry, training, array-derived metrics and stage commands must
be connected to this ledger and physically smoke-tested before the full study.

## Numerical controller-study components

The new numerical helpers implement measured pilot telemetry, complete-cohort
per-episode predictor fits, exact Poisson/read-noise controller kernels and paired
post-burn/fault-window metrics. The open-loop reference includes the same vibration
as the controlled plant. The baseline simulation is unchanged.

The local suite passes **148 tests**, including **31 new numerical/runner-guard
tests**. Separate tests cover measured-only inputs, independent fitting windows,
causality, noise-path equivalence, zero-DM fault pairing, malformed/raw nonfinite
data, reserved-stream exclusion and CI refusal. A component smoke command and its
inputs/gates are prepared; run it from this committed source before recording any
physical smoke result. Learned fitting, actual study provenance and ledger-backed
stage commands are still pending. No reserved-cohort outcomes have been generated.
