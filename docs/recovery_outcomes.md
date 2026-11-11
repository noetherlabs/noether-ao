# Measuring the cost of a controller switch

`noether_optics.recovery_metrics.paired_recovery_outcomes` compares complete
continuation and intervention histories over one fixed event horizon. It measures
whether switching improved the simulated optical residual, how large the command
steps became, and whether an apparent recovery persisted.

This is an analysis primitive. It does not define a cohort, detect a fault, choose
a response, or establish controller safety. The prospective recovery comparison
has not run. The [development pilot](recovery_development.md) acquires complete
paired histories for resource and variability assessment before that freeze.
See [controller handover](controller_handover.md) for the implemented
transfer mechanics and their one-frame-delay equivalence.

## Fixed-horizon cost

For event onset `k0`, exclusive horizon end `k1`, frame period `dt`, and spatial
pupil phase variance `V`, the signed paired cost is

```text
excess = dt * sum(V_intervention[k] - V_continue[k], k=k0,...,k1-1)
normalized_excess = excess / (dt * sum(V_zero_DM[k], k=k0,...,k1-1))
```

Positive values mean the intervention was worse. Both arms use the same horizon
and strictly positive zero-DM denominator. Frames before the delayed command can
affect the plant remain included. Moving the reporting window to favor a later
switch changes the question and is not supported.

Each arm also reports WFE, mean coherent pupil Strehl, saturation fraction, peak
variance, and per-mode command magnitude, step, and step divided by frame period.
The step entering the event window is included. Requested commands are reported
only when their complete stream is present; a clipped command is never substituted
for an unavailable request. Units are rad and rad/s in the simulator's modal basis.
These are not actuator stroke or slew violations: that requires a calibrated
actuator mapping and hardware limits.

## Sustained reference return

The caller must supply a threshold multiplier and a consecutive-frame count.
The common threshold is the multiplier times the continuation arm's mean variance
between burn-in and event onset. The first qualifying run begins no earlier than
the declared first possible intervention effect. The report separates the run's
start from its confirmation frame, when all required frames have been observed.

A qualifying run ending on the last observed frame counts. A shorter terminal
run is right-censored, with no invented recovery time. The report also retains
the fraction of subsequent frames above the threshold, so a later relapse is
visible. If no subsequent frames exist, that fraction is unavailable. This
reference-return rule is a simulation definition, not an instrument acceptance
criterion. Fix its parameters before inspecting evaluation outcomes.

## Evidence requirements

The reducer requires every standard physical output stream, checks modal command
bounds and the complete FIFO relation, verifies requested-command clipping and
saturation when that stream is available, and requires exact variance, Strehl and
applied-command equality before the declared first possible effect. Produced
nonfinite streams withhold all paired outcomes; finite survivors are not averaged
in their place. A zero normalization denominator or numerical overflow also
withholds outcomes. Inputs are never altered.

The caller must separately verify source, atmosphere, calibration, predictor,
detector-key and artifact identities. Equal prefixes do not prove these identities.
Matched detector keys also do not imply identical Poisson samples once the two
closed loops produce different intensities.

No interval is computed from frames. A cohort analysis must use independent
disturbance roots as replicates, retain terminal failures, and declare its
multiplicity and fixed sample size before evaluating that cohort.
