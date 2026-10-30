# Optical-gain observability: validation result

The frozen residual-score detector failed its validation criterion in all four
tested regimes. The complete study contains 64 clean training runs and 240
validation runs. Evaluation was withheld; no evaluation outcome was acquired.

## Result

Each regime contains 30 clean and 30 paired gain-fault runs. The injected sensor
response ramps to 0.35 of its nominal value over 32 frames. Detections require an
entire persistent crossing sequence after fault onset, with no pre-onset alarm.

| Photons | Wind scale | Clean alarm runs | Gain detections | Misses | Detected-only delay (frames) | Failed criterion |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1,000 | 0.5 | 0/30 | 0/30 | 30/30 | Not defined | Detection and delay |
| 1,000 | 2.0 | 0/30 | 0/30 | 30/30 | Not defined | Detection and delay |
| 1,000,000 | 0.5 | 4/30 | 30/30 | 0/30 | 10–24 | Clean alarms |
| 1,000,000 | 2.0 | 0/30 | 30/30 | 0/30 | 13–148 | Delay |

The predeclared gate requires **every** regime to have zero clean alarm runs,
at least 29/30 gain detections, and every detected delay at most 32 frames.
All runs completed with finite outputs and without fallback, unavailable frames
or saturation. Clean raw-crossing fractions and persistent alarm-frame fractions
passed their separate limits in all regimes. The negative decision therefore
reflects detector performance rather than numerical failure.

## Method

The [frozen protocol](../../docs/optical_gain_observability_v1.md) specifies a
four-frame causal mean of the whitened residual score, a regime-specific lower
0.005 training quantile, and eight consecutive crossings for an alarm. Thresholds
were frozen from all 64 training runs before validation. The forward controller
remained fixed; the detector could neither change commands nor trigger recovery.

Trials use 512 frames, 128-frame burn-in, fault onset at frame 256, Poisson and
read noise, and fresh training/validation seed streams. Clean and fault cases
share atmosphere and detector roots within each regime. Roots are also shared
across regimes, so the four rows are not independent populations.

These are validation-screening results, not an operational false-alarm guarantee.
Zero alarms among 30 runs still permits a one-sided 95% exact binomial upper bound
of approximately 9.5%. Delay ranges exclude missed runs; the miss counts must be
read alongside them. Correlated frame counts are not independent trials.

## Interpretation

The development-selected residual statistic did not transfer adequately across
the declared photon and wind conditions. Low-photon detection failed completely;
high-photon detection traded clean alarms or long delays against sensitivity.
These observations reject this frozen candidate for the declared release gate.
They do not identify the physical cause of those failures, establish that gain is
unobservable with other measurements, or evaluate a recovery controller.

The evaluation barrier remains closed. Any revised detector requires a separate
protocol and fresh cohorts; these validation outcomes are development evidence
for that future study.

## Evidence

- [Validation decision](validation_release.json): all 240 terminal records,
  per-regime criteria, counts, delays and raw-artifact hashes.
- [Training thresholds](thresholds.json): all 64 training references, immutable
  thresholds and detector-trace hashes.
- [Machine-readable protocol](../../configs/optical_gain_observability_v1.yaml).

The decision was reconstructed from the complete retained training and validation
arrays using the acquisition source. Reconstruction reproduced the frozen decision
byte for byte. This compact packet contains the decision and threshold artifacts;
it does **not** bundle the raw NPZ arrays or acquisition source snapshot. It supports
inspection of the reported counts and gates, not independent replay of the physics.

| Binding | SHA-256 or acquisition identifier |
| --- | --- |
| Acquisition source commit | `ae62a92a644c50c8e08d58ac07bf4e41d8d9c8ee` |
| Source content | `be6b932f40cb6f7340e2018c09815091b91f8404626ce906a32975a962439995` |
| Runtime | `030b8324cdaf09bfcf6e3c4bc04fd35bf643271c1e26053ddde2cfd5814a7719` |
| Protocol | `2b387e8108fb801e6594cbb4531e9e2951f3f245279382ed65e399ce88243221` |
| Threshold artifact | `98a25c5b31509b40fc339d909fbd631f8fd387a022649aca2d72390ee92dc854` |
| Validation artifact | `d023dfc3a71b23582e00f240f32adc8dbad441bf9e7484a77026c9ff0085fd27` |

Acquisition identifiers refer to the measured source and runtime, independently
of the documentation revision.
