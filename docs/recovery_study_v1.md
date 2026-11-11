# Scheduled fallback study

This prospective study measures the effect of switching from a fixed predictive
controller to a fixed leaky integrator. Its operational question is whether a
fallback improves the subsequent trajectory under the observed regime. It does
not evaluate a fault detector or claim that the fallback is safe.

**Status:** statistical contract and reduction implementation. The 64-root cohort
has not been acquired. Acquisition requires a scoped historical-stream exclusion
inventory, a frozen runner, bounded execution and independent raw reconstruction.

## Design

The [development plant and handovers](recovery_development.md) are unchanged:
512 frames, a scheduled event at frame 256, two-frame command delay, 1 ms sample
period, a 32-sample pupil and 100 controlled modes. The predictor coefficients and
integrator gain 0.5/leak 0.99 remain fixed from the previous controller study.
No recovery-specific fitting or retuning is allowed.

Sixty-four distinct atmosphere roots are crossed with 1,000 and 1,000,000 photons,
wind multipliers 0.5 and 2, and clean, optical-gain drift and vibration conditions.
Every case retains continued prediction, issued-state transfer and command-matched
transfer: 768 cases and 2,304 trials. All arms share each atmosphere and detector
root. Detector keys need not produce identical Poisson draws after trajectories
diverge. Conditions reuse roots; they are not independent additional samples.

The command-matched arm is the ordinary transfer one frame later for this fixed
controller. It is a timing control, not a new compensator. The event is scheduled
even in clean episodes, which measure the cost of unnecessary intervention.

The exact schedule, source input hashes and reporting rules are returned by
[`recovery_study.protocol`](../src/noether_optics/recovery_study.py). Development
roots and detector offsets are excluded. This explicit separation does not
replace the historical-stream inventory required before acquisition.

## Primary estimand

For root *s* and each fixed condition/transfer, define

```text
D_s = sum(t=256..511, variance_transfer[t] - variance_continued[t])
      / sum(t=256..511, variance_zero_DM[t]).
```

The denominator uses the same atmospheric trajectory and injected disturbance.
Positive values mean the intervention increased residual phase variance. The
common horizon includes the command queue delay; the interval does not move to
the observed recovery time. The estimand is the mean of these per-root ratios,
not a ratio of pooled sums.

The primary family contains 24 contrasts: two transfers × two fluxes × two winds
× three conditions. Each mean uses a two-sided Student-t interval with 63 degrees
of freedom and marginal alpha 0.05/24. Bonferroni does not require independence
between contrasts. Coverage is exact for independent normal root differences;
otherwise the Student-t procedure is an approximation. Frames are never treated
as independent replicates. Zero observed variance yields a point estimate with
no interval or directional conclusion.

An interval wholly above or below zero is reported as increased or decreased
residual error. No instrument-specific practical margin is declared, so neither
label is a commissioning acceptance decision. WFE in nanometres, pupil Strehl,
peak error, command changes, saturation and censored reference-return outcomes
are secondary descriptive results. A pupil Strehl metric is not a measured
science-camera contrast or fibre-coupling efficiency.

## Size and failure policy

The four-root development cohort is used only for planning. Under a marginal
normal-model upper 95% bound on each pilot standard deviation, the largest
conditional simultaneous-interval half-width is approximately 0.0662 at 32 roots,
0.0524 at 48 and 0.0447 at 64. Sixty-four roots were chosen to target less than
0.05 on this dimensionless planning scale. This is not a customer tolerance,
simultaneous variance bound, power guarantee or promise of achieved precision;
four pilot roots provide weak variance information.

The planning inputs are the complete 144-trial development report
`c738804f202b53a44422fe90e3b5926ffaf776eee7cb1f81f9ed0c21e173698b`
and its raw reconstruction. The calculation uses
`upper_sd = pilot_sd * sqrt(3 / chi2.ppf(0.05, 3))`, followed by
`t.ppf(1 - 0.05 / 48, n - 1) * upper_sd / sqrt(n)` for each of the 24
contrasts. Raw development archives are retained locally and are not included
in this repository; the hash alone is not a downloadable reproduction package.

The cohort size is fixed before evaluation. An uncertain result does not trigger
additional roots, alternative endpoints or threshold tuning. Produced nonfinite
arrays remain attached to terminal failures. A nonfinite pair withholds the
entire affected contrast, with all 64 identities retained; failed roots are not
dropped. Missing or duplicate identities prevent a complete statistical report.
An interruption leaves an incomplete attempt and cannot count as evidence of
success. The interval family remains 24 even when a contrast is withheld.

The statistical reducer accepts reconstructed paired outcomes. It does not
authenticate those inputs, verify physical artifacts, or prove that acquisition
occurred. A release must bind the frozen protocol, source, environment, complete
raw cohort and independent reconstruction report separately.
