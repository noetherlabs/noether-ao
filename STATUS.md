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

1. Shared ideal plant/twin sensor code and a finite periodic atmosphere. Independent
   open-loop PSF agreement does not validate a real pyramid or instrument.
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

The next increment is independent ideal-pyramid validation. Its coordinate
contract and gates are predeclared in `docs/pyramid_validation.md` and
`configs/pyramid_validation.yaml`; no comparison results are claimed yet.
