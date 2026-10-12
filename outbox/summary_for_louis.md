# Simulation results and questions for Louis

Unsent draft for human review. This is a small civil-astronomy prototype, not an
instrument deployment or an institutional collaboration.

The dLux CPU FFT sweep is configuration dependent: at N=32, one wavelength and
float32, p99 was 0.0592937 ms; at N=256 and five wavelengths it was
6.38013 ms. These are synchronized single-plane component measurements.
There is no GPU result or full-loop deadline claim. The separate Dragrace MFT
contribution adds a small-pupil case and stored p99/maximum columns; it remains unsubmitted upstream.

The delayed pyramid simulation passed independent HCIPy open-loop PSF, fitting-law,
delay-growth and nominal stability checks. Its predictor reduced mean WFE by
26.3727% relative to the integrator in the configured
bright-star regime. This does not reproduce an unspecified instrument configuration.

No tested flux gave useful equivalence across all fidelity levels. No supported low-flux threshold was found; lower fidelity often performed better under this fixed estimator.

Learned weights beat both integrator and forward in 0/12 conditions and lose to at least one in 5/12; remaining comparisons include uncertainty.

The assurance result is mixed. Gain drift escaped detection; fallback under
vibration made WFE worse. Clean false alarms occurred in
20% of the short held-out runs. Missing/late-output
fallback helped, but those faults are synthetic availability flags. No general
safety guarantee follows. Full numbers, raw data, plots and reproduction commands
are linked from [the results report](../docs/results.md).

Three questions:

1. Which pupil sampling, detector binning and wavelength quadrature would make this
   ideal pyramid comparison useful for a MagAO-X-like configuration?
2. Which telemetry or optical-gain estimator should accompany a frame-residual
   detector, given that decreasing optical gain escaped this upper-tail score?
3. Which tuned classical/predictive baseline and weighting inputs would make the
   mixed-controller comparison persuasive before increasing the training horizon?

Could you suggest a public or synthetic MagAO-X-like test configuration: pupil and
bandpass, DM basis, pyramid geometry/modulation, detector noise/flux, delays, and
turbulence profile? A configuration that others may redistribute would let this
repository test a relevant case without implying access to private instrument data.
