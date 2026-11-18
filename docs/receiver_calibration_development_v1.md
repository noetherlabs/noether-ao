# Calibration utility at a Gaussian receiver

This development study asks whether estimating a stale sensor response improves
subsequent fiber coupling enough to repay the probes used to estimate it. The
endpoint includes probing, washout and service startup. A good response fit
alone is not a positive result.

The [configuration](../configs/receiver_calibration_development_v1.json) fixes
the comparison before acquisition. The
[single-trial runner](../scripts/run_receiver_calibration_development.py)
records one root and one condition; the
[independent reader](../scripts/audit_receiver_calibration.py) reconstructs its
physical residual and utility from retained arrays. This document specifies an
experiment, not a measured improvement or a hardware qualification.

## Comparison

The plant is a 16-mode linear sensor and delayed integrator on a 32 × 32 pupil
grid. The aperture diameter is 1 m; phase coefficients use a 750 nm reference.
Frames are 1 ms apart, with a two-frame command delay. Integrator gain, leak and
command limit remain fixed at 0.3, 0.999 and ±5 rad per mode. The atmospheric
parameters and nominal calibration are fixed by the configuration.

Eight independent random-stream roots each cover four conditions:

| Sensor response | Sensor photons per frame |
| --- | ---: |
| Nominal | 1,000 |
| Nominal | 1,000,000 |
| Coupled mismatch | 1,000 |
| Coupled mismatch | 1,000,000 |

The nominal response transform is the identity. The mismatch is a tridiagonal
raw-channel transform with diagonal entries increasing linearly from 0.55 to
0.9, upper diagonal 0.08 and lower diagonal −0.04. It acts on the physical
sensor response before detector noise; the controller initially retains its
nominal reconstructor. This is a static modal calibration error, not a
time-varying drift or a nonlinear pyramid-sensor model.

Conditions within a root share the declared atmosphere, detector and probe
stream identities. Paired arms within a condition use identical phase and
detector keys. Commissioning and service have distinct streams. The four
conditions are correlated comparisons, not four independent replicates.

## Timing and information available to the fit

All intervals below use zero-based, half-open frame indices.

| Recording | Frames | Role |
| --- | --- | --- |
| Commissioning | `[0, 128)` | Common unprobed warm-up |
| Commissioning | `[128, 256)` | Independent ±0.05 rad requested probes in each mode |
| Commissioning | `[256, 384)` | No new probes; retain washout and delayed arrivals |
| Service | `[0, 640)` | Compare nominal and frozen corrected reconstructors |

Probes are added to feedback before command clipping. The regressor is the
actual applied command. The instrument is the requested probe shifted by the
declared two-frame delay; it is not an estimate of the probe's physical
contribution after clipping. The response fit uses exactly commissioning rows
`[130, 258)`, including the final delayed probe arrivals.

Let rows of $Y$ be measured residuals reconstructed with the nominal
calibration, rows of $X$ the applied commands, and rows of $Z$ the delayed
probe requests. After subtracting column means, the instrumental-variable fit
solves

$$
Q_Z^\mathsf{T}X\widehat B
\simeq -Q_Z^\mathsf{T}Y,
$$

where $Q_Z$ spans the centered instrument columns. The fit receives only
commissioning measurements, issued and applied commands, requested probes and
the pre-existing nominal calibration. It does not receive true phase, the
physical mismatch transform, service observations or receiver coupling.

The rank tolerance is $10^{-10}$; minimum canonical correlation is 0.05.
The estimated response must be invertible under the declared rank test. The
corrected reconstructor must be finite, and its spectral norm may not exceed
three times the nominal norm. Rank, relevance and gain checks can withhold the correction.
They do not establish instrument exogeneity or closed-loop stability. See
[response identification](identification.md) for the estimator assumptions.

The fit and its SHA256 binding are serialized and read back before the service
phase sequence is generated. The service controller uses that frozen matrix.
If correction is withheld, service retains the nominal reconstructor and
reference. The nominal service recording is reused explicitly, giving zero
paired service gain while retaining the full commissioning probe cost.

Both service arms start with zero commands and a zero delay queue. They do not
inherit their commissioning terminal states. This measures calibration
transfer between independent recordings, including service startup; it does
not demonstrate a continuous online handover or determine how long a recovered
calibration remains useful.

## Receiver endpoint and charged cost

The [passive receiver](receiver.md) reconstructs each spatial residual from the
original phase, modal basis and **applied** command. It converts that residual
to OPD, then evaluates overlap with an ideal centered Gaussian receiver at
1550 nm, focal length 4 m and focal-plane field radius 4.4 μm. Coupling
$\eta\in[0,1]$ is a fraction of incident pupil power. It is not inferred from
Strehl or scalar wavefront error.

For a root-condition pair, let superscripts $0$ and $p$ denote unprobed and
probed commissioning; $0$ and $c$ denote nominal and corrected service.
Define the signed commissioning cost and service gain as

$$
C=\sum_{t=128}^{383}\left(\eta^0_{\mathrm{comm},t}
                         -\eta^p_{\mathrm{comm},t}\right),\qquad
G=\sum_{t=0}^{639}\left(\eta^c_{\mathrm{service},t}
                       -\eta^0_{\mathrm{service},t}\right).
$$

The primary per-condition utility is $u=(G-C)/896$: all 256 commissioning
frames from the first probe onward plus all 640 service frames. No negative
cost, negative gain or unfavorable finite outcome is clipped or discarded.
The first 128 commissioning frames must agree exactly between arms, so the
same numerator also supports a full-recording mean $(G-C)/1024$. Multiplying
the numerator by 0.001 s gives a coupling-time difference, not energy or bits.

For example, a hypothetical commissioning loss of 10 coupling-frames and
service gain of 5 gives $u=-5/896$, despite the positive service gain. A
withheld fit with the same commissioning loss gives $u=-10/896$. These are
arithmetic illustrations, not experimental results.

The model excludes received-power noise, scintillation, polarization,
transmission losses, pointing acquisition and modem behavior. Sensor photon
counts do not supply a downlink power or throughput budget. Any descriptive
low-coupling threshold is separate from the primary endpoint and is not a
demonstrated link-failure threshold.

## Root-level uncertainty and incomplete outcomes

Average the four utilities within each root, then average the eight root
means. The primary Student-t interval uses these eight values, seven degrees
of freedom and two-sided $\alpha=0.05$. It is a model-dependent small-sample
summary; frames and conditions are not treated as independent observations.
A bounded-mean Hoeffding interval is also reported, assuming independent root
blocks with utilities in $[-1,1]$. With eight roots it may be very wide.

Each condition has an eight-root descriptive interval with
$\alpha=0.05/4$, applying Bonferroni correction across the four conditions.
There is no frame-wise independent-sample confidence interval.

Every one of the 32 declared root-condition identities requires a terminal
record. A withheld but finite outcome has its measured utility. A failed or
unavailable outcome has `charged_mean_delta: null` and bounds `[-1, 1]`.
Missing cells contribute those bounds to the equally weighted population
mean; they are not replaced by zeros or removed from the denominator. The
primary point estimate and interval are withheld when any cell is unknown.
A condition receives statistics only if all its eight outcomes are known.
Duplicate, absent or unexpected identities are structural errors.

## Running and checking one trial

Install the pinned optional receiver dependency with
`uv sync --locked --extra receiver`. Use a clean committed checkout matching
the installed AO source. Prepare a separately reviewed seed audit for the
configuration's exact namespace and role derivation; hashing a file does not
by itself establish that its streams are disjoint from earlier experiments.

The following command assumes `CONFIG_SHA256`, `SEED_AUDIT` and
`SEED_AUDIT_SHA256` already identify those reviewed inputs, and `TRIAL_OUTPUT`
is a fresh directory outside the checkout:

```bash
uv run --locked --extra receiver python scripts/run_receiver_calibration_development.py \
  --repository . \
  --config configs/receiver_calibration_development_v1.json \
  --config-sha256 "$CONFIG_SHA256" \
  --seed-audit "$SEED_AUDIT" \
  --seed-audit-sha256 "$SEED_AUDIT_SHA256" \
  --root r00 --condition nominal-low_flux \
  --out "$TRIAL_OUTPUT"
```

The runner records source and runtime identities, calibration, both phase
sequences, requested probes, all original loop fields, the frozen fit and
receiver traces. It checks the zero-initialized FIFO, command clipping,
original schemas and nonmutation. Bounds are 16 MiB of logical scientific
arrays and 32 MiB of retained data per root-condition. The configured study
budget is 1 GiB. Caller-side resource admission and complete-cohort accounting
are separate from this one-trial command.

Once the terminal record's digest is captured as `RESULT_SHA256`, independently
check the packet and write the audit outside it:

```bash
uv run --locked --extra receiver python scripts/audit_receiver_calibration.py \
  "$TRIAL_OUTPUT" \
  --config configs/receiver_calibration_development_v1.json \
  --config-sha256 "$CONFIG_SHA256" \
  --seed-audit "$SEED_AUDIT" \
  --seed-audit-sha256 "$SEED_AUDIT_SHA256" \
  --result-sha256 "$RESULT_SHA256" \
  --output "$AUDIT_OUTPUT"
```

Existing trial destinations are refused. On failure, produced bounded numeric
artifacts and a terminal failure manifest are retained; oversized or unsafe
arrays are refused before serialization. Failed packets can establish what
was retained and why execution stopped, not a reconstructed scientific
outcome. Neither a passed software test nor a completed packet establishes
positive utility. That requires the declared comparison and its full outcome
accounting.
