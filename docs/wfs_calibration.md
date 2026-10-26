# Independent pyramid-sensor calibration comparison

This experiment extends the [ideal-pyramid validation](pyramid_validation.md)
from individual images to interaction matrices, independently calibrated
reconstructors, held-out mixed wavefronts and calibration transfer. It does not
change the production sensor, controllers or frozen observability experiments.

Completed evidence: [findings](../results/wfs_calibration_v1/README.md),
[full report](../results/wfs_calibration_v1/report.md) and
[reconstruction verification](../results/wfs_calibration_v1/verification.json).

## Declared comparison

Four optical paths receive identical phase arrays in radians at 750 nm:

- **Noether:** production dLux/JAX sensor; automatic-differentiation calibration.
- **HCIPy matched:** the existing independent matrix-integral/thin-surface
  reference; central push-pull calibration, with no production imports.
- **HCIPy native:** `PyramidWavefrontSensorOptics`, including its circular focal
  stop, interpixel focal centering and two forward propagations.
- **SPECULA native:** `ModulatedPyramid`, zero modulation, CPU double precision,
  including its native focal stop, centering, interpolation and flux accounting.

The last two paths are **native-model comparisons**, not asserted equivalents of
Noether's square-band, inverse-return model. Pixel order is unchanged; no fitted
flip, sign, shift, gain or normalization may be applied to improve agreement.
Report their disagreement and calibration-transfer errors even when large.

The pupil diameter is one metre, sampled at pixel centres. Opposing pupil centres
are separated by 1.2 diameters. The detector covers four diameters in 64 pixels.
HCIPy native uses focal sampling q=4 and radius n/2 lambda/D; SPECULA uses the
corresponding Nyquist field of view, fft_res=4 and pup_diam=16 detector pixels.
Input power is one; detector outputs retain transmission losses. SPECULA receives
OPD in nm (`phase * wavelength_m * 1e9 / (2*pi)`) and incident flux density
`1 / illuminated_area`; its native output is not renormalized by this adapter.

## Inputs, metrics and acceptance

`configs/wfs_calibration.yaml` freezes grids, seeds and tolerances before the
retained run. Five independently generated polynomial modes are piston-free and
individually unit RMS. Their sampled Gram matrix is retained: combinations and
errors use **actual pupil RMS**, not an assumption of exact orthogonality.
Push-pull probes of 1e-4 and 5e-5 radians test derivative convergence. Sixteen
seeded mixed directions are evaluated at 0.001, 0.05 and 0.25 radians RMS;
none is used to fit a calibration or choose a tolerance.

Each backend uses its own zero image and response matrix, with the same fixed
SVD cutoff. Report singular values, rank, conditioning, per-case phase errors,
and every ordered backend-to-backend calibration transfer. Matrix-only transfer
subtracts the target's own zero image; full transfer subtracts the calibration
source's zero image. Both are explicit, because a zero-point mismatch can dominate.

Required gates: finite nonnegative images, positive transmitted power no greater
than unity (roundoff allowance 1e-10), full five-mode rank, derivative convergence,
piston invariance, and <=1% relative phase RMS error at 0.001 rad for each backend.
Noether/HCIPy-matched additionally require the declared zero-image and interaction
agreement and <=1% small-signal calibration transfer. Native cross-model
transfer has no pass gate. Larger-amplitude errors are reported without retuning.
Deliberately reversing calibration sign and using an nm-as-radians scale must
produce >10% small-signal relative phase error for every backend.

At the flat wavefront, propagate independent Poisson shot noise and Gaussian
read noise through each fixed reconstructor. Retain all 512 modal noise samples
and compare empirical covariance to `R diag(p/N + read_noise_e^2/N^2) R.T`.
This is a diagnostic of the declared independent-pixel noise model, not measured
instrument noise or a controller ranking. No latency acquisition is performed.

```text
frozen configuration -> common pupil modes / held-out phases
                     -> each independent optical backend
                        -> zero + +/- pokes -> fixed reconstructor
                        -> held-out images -> self and cross reconstruction
                        -> flat image -> analytic / sampled noise covariance
retained arrays + hashes -> report rebuilt without optical simulations
```

Tests cover known linear sensors, nonorthogonal phase metrics, missing/invalid
data, sign/unit negative controls, independent optical smoke checks and artifact
tampering. Acquisition refuses an existing destination and a reserve below
512 MiB. This bounded comparison is separate from the larger P2 storage gate.

## Reproduction

Use the pinned Noether environment (`uv sync --frozen --python 3.11`). SPECULA is
an optional isolated environment; install the source revision recorded in the
saved provenance with its dependencies. Its Python is passed explicitly:

The retained run used HCIPy 0.7.1 and SPECULA commit
`d37bbcc48db2387a1b021eb9744f046b733ca159` (Python 3.12.0, NumPy 2.5.4,
SciPy 1.18.1). This SPECULA checkout includes the separate, unmerged
[Shack–Hartmann sampling fix #836](https://github.com/ArcetriAdaptiveOptics/SPECULA/pull/836);
the pyramid code is unchanged by that patch. An isolated source installation is:

```sh
uv venv --python 3.12 .venv-specula
uv pip install --python .venv-specula/bin/python \
  'git+https://github.com/salma2vec/SPECULA.git@d37bbcc48db2387a1b021eb9744f046b733ca159' \
  'numpy==2.5.4' 'scipy==1.18.1'
```

All installed dependency versions are retained in the backend JSON files; the
command above pins the source and numerical kernels, not the entire transitive
environment. Reproduction should use the recorded versions. SPECULA remains
optional for the core test suite, but the comparison runner requires all backends.

```sh
mkdir -p runs
PYTHONPATH=src JAX_ENABLE_X64=1 .venv/bin/python scripts/compare_wfs_calibration.py \
  --out runs/wfs-calibration-new --specula-python .venv-specula/bin/python
PYTHONPATH=src .venv/bin/python scripts/compare_wfs_calibration.py \
  --out results/wfs_calibration_v1 --render-only
```

The runner and backend source hashes, installed package versions, raw arrays and
configuration accompany the report. Native APIs: [HCIPy pyramid source](https://github.com/ehpor/hcipy/blob/master/hcipy/wavefront_sensing/pyramid.py)
and [SPECULA pyramid source](https://github.com/ArcetriAdaptiveOptics/SPECULA/blob/main/specula/processing_objects/modulated_pyramid.py).
The result establishes only the stated finite-grid simulated calibration
behavior. It does not establish an instrument model, on-sky validity, modulation
agreement, closed-loop performance or practical calibration time savings.
