# Acceptance gates declared before simulation runs

These are acceptance inputs, not observations. They will not be changed to make a
failed run pass. Diagnostic attempts must save their own outputs.

- M1 complex-field relative L2: at most 1e-5 in float32 and 1e-11 in float64.
  Realized dtype must match. At least 10000 samples for each available point/path.
- M3 same sampled-pupil, same-grid HCIPy PSF: normalized intensity relative L2
  at most 1e-8 in float64. Test an aberrated pupil as well as a flat pupil.
- M3 fitting: variance versus effective actuator spacing D/sqrt(number of modes)
  log-log slope within 0.4 of 5/3, on a mode-count sweep. Independently vary r0;
  variance versus 1/r0 exponent within 0.1 of 5/3. The r0 test alone is insufficient.
- M3 delay: held-out ensemble mean residual variance must increase from the shortest
  to longest delay, with positive fitted slope. Report intermediate non-monotonicity.
- M3 stability: every output finite; integrator residual RMS below open-loop RMS;
  measured on-axis Strehl strictly between zero and one, with no saturated commands
  in the nominal bright-star validation run. Stage A and stage B are separate gates.
- M4 completeness: every configured fault produces raw scores, switch states, delay
  or explicit missed status, false-alarm metric and paired fallback/no-fallback error.
  A passing completeness gate is not a positive result for C4.
- M5 completeness: every fidelity/flux cell has at least five held-out seeds and
  saved per-seed values. A negligible difference requires a paired 95% interval
  wholly inside plus/minus five percent of the high-fidelity mean; otherwise report
  inconclusive or meaningful difference. A high-error common failure is not success.
- M6 completeness: training seeds and held-out seeds are disjoint; all controllers
  and both mixtures run all cells. Wins/losses use paired intervals versus each
  standalone controller; overlapping intervals are not proof of equivalence.
- M7 reproduce: install from the lock in a fresh checkout, run all correctness tests,
  regenerate tables/plots from raw data, and run the documented experiment entry
  points. Exact timing equality is neither expected nor claimed across runs/hosts.

## P2 optical-gain observability release gate

The frozen [P2 protocol](optical_gain_observability_v1.md) calibrates one numerical
threshold per regime from training-clean data, then makes it immutable. Evaluation
remains sealed unless every regime has 0/30 clean alarm runs, clean raw crossing
fraction at most 0.01, clean alarm-frame fraction at most 0.001, 0/30 gain pre-onset
alarms, at least 29/30 gain detections, at most one miss, every detected delay at most
32 frames, finite outputs, no fallback or unavailable frames, and zero saturation.

Failure is retained and evaluation is withheld. Passing is an engineering screen,
not an instrument or safety qualification. The detector is observational and cannot
support a recovery claim. The release calculation and exclusive artifact are
implemented with synthetic tests. A locked, protocol-bound ledger also preserves and
revalidates terminal training/validation records across interrupted chunks. No
declared P2 threshold or validation result exists.
