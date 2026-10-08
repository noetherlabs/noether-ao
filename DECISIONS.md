# Decisions

- Build isolated sibling repositories inside the writable project workspace:
  noether-optics and Diffraction-Dragrace. Do not modify prior Noether work.
- Sole engineer, as requested. No subagents. Record assumptions instead of asking
  questions. The autonomous request supersedes interactive gstack review prompts.
- Civil astronomy only. This is a simulation and measurement project, not a claim
  of operational assurance or collaboration with any institution.
- Configuration values, mathematical constants, version identifiers and acceptance
  tolerances are inputs, not measured results. All reported performance values must
  be generated in this session and linked to saved raw data and provenance.
- Use Python 3.11 and a uv lock. Keep source checkouts for source/API inspection.
- A single-plane FFT with explicit dynamic OPD is the initial latency contract.
  It is not the fixed focal-grid MFT contract in Dragrace. Report them separately.
  Multiwavelength FFT planes have wavelength-dependent angular grids; do not sum
  them into a broadband detector without resampling.
- macOS has no Linux sched_setaffinity. Never substitute environment thread flags
  for affinity. Unsupported execution modes get explicit skip records.
- The gstack scripts at the legacy path are absent, but the installed review skill
  and checklist are readable. Use their engineering/test/review workflow without
  installing or publishing unrelated changes.
- Gate tolerances and experiment configurations must be written before runs.
  Failed gates remain failures; fixes may correct implementation, never move a gate.

## Simulation assumptions (before control experiments)

The plant is a finite periodic von Karman screen with two independent frozen layers.
No subharmonics, scintillation, telescope obscuration, dispersion, coronagraph,
non-common-path aberrations, or hardware dynamics are claimed. A sampled,
reference-grid-orthonormal Zernike modal DM defines radians RMS, with a command FIFO.
The ideal pyramid uses dLux forward/inverse FFTs, a four-face phase wedge,
flux-conserving detector binning and a fixed detector extent. This is not a
MagAO-X configuration. A stuck actuator is represented by a localized loss of
DM response, because this first model has modal rather than physical actuators.

The linear predictor learns two taps per mode on ideal open-loop telemetry from
training seeds only. This is optimistic calibration. The forward estimator uses
fixed-preconditioner gradient steps; it is not an exact Gauss-Newton solve.
The compiled control/optical path supports gradients. Training uses a Gaussian
shot/read approximation so pathwise derivatives exist; evaluation uses exact
Poisson sampling and Gaussian read noise. Discrete fault decisions are not
claimed differentiable. Fixed and learned weights blend candidate commands
using the actual issued command state. Fallback latches until the run ends.

Detector thresholds are empirical clean-training quantiles. No statistical
independence, formal safety proof, or out-of-distribution guarantee is assumed.
ROC curves concern the optical score. Missing/late outputs additionally trigger
an explicit availability rule, which is reported separately from score ROC.
Fidelity uses a common higher-resolution plant and a shared detector. M1 latency
is a component cost only; a pyramid and its derivatives cost additional work.
Across-seed error bars are standard errors. Paired confidence intervals are
Student-t intervals over the fixed held-out seeds, without multiple-test correction.

## Interrupted timing process

The original M1 process disappeared without a completion record. Completed raw
points are retained without filtering. Resumption validates saved points and runs
only unfinished points, using the identical timed region and experiment config.
The continuation fingerprint and interruption record are saved beside the raw data.

The pyramid misregistration injection translates the complete complex entrance
pupil inside the padded support, including its amplitude edge. It is not merely a
shift of phase samples under an unchanged pupil. The optical-gain fault ramps from
the nominal response to the configured gain over a configured drift interval.

Unavailable twin outputs are discarded. Without fallback, the advanced controller
holds its last command; with fallback, the independently calibrated integrator
can issue a fresh command from the current WFS frame. Synthetic availability flags
model missing/deadline-late outputs; they do not measure real scheduler failures.
