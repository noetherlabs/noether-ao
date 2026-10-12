# Independent ideal-pyramid validation protocol

Completed evidence: [comparison report](../results/pyramid_validation_v1/report.md)
and [reproduction checks](../results/pyramid_validation_v1/verification.json).

This protocol is committed before evaluating the comparisons. The existing
plant/twin implementation and M0–M7 data remain unchanged. The acceptance settings
in `configs/pyramid_validation.yaml` are fixed numerical tolerances, not results.

## Scope and conventions

Validate the discretized ideal unmodulated sensor using HCIPy 0.7.1 matrix Fourier
quadrature and a physical thin surface, independently of dLux/JAX FFTs. Reuse the
production sensor only on the test side, and the artifact/provenance helper for
recording. Do not add another simulation/controller architecture or retune gains.

- Pupil diameter and reference focal length are one metre. Input samples are
  interpixel-centred with pitch `1/n`; a hard circular aperture has diameter one.
- Incident phase is in radians at 750 nm, with positive complex phase. At each
  wavelength it scales by reference wavelength divided by actual wavelength.
- The focal plane spans the square Nyquist band, sampled at `wavelength/4`.
  The even grid includes zero and is centred half a focal pixel below zero.
  No circular spatial filter or fitted centering correction is introduced.
- The nondispersive surface has refractive index 1.5 and sag
  `-0.6 * (abs(x) + abs(y)) / (1.5 - 1)` at unit focal length. Each pupil centre
  shifts by 0.6 diameters per axis, giving 1.2 diameters between opposing centres.
- The return transform uses the inverse sign and an interpixel-centred output
  grid spanning four diameters. There is no post-hoc flip, transpose or phase fit.
- Normalize incident physical power to one. Preserve absolute pixel powers at the
  detector, sum within detector bins, and average equal-weight wavelengths.
  Never normalize the output to conceal lost flux.

HCIPy's native `PyramidWavefrontSensorOptics` is not a drop-in numerical reference
for this contract: its default focal grid is interpixel-centred, it uses a circular
focal filter and forward propagation on both legs. Its separation parameter is
the distance between pupil centres. The reference below deliberately constructs
the same stated optical contract with independent HCIPy primitives. Agreement
would establish that contract, not agreement with native defaults, convergence to
an infinite focal plane, a real instrument, or a modulated/dispersive pyramid.

## Gates and failure handling

Compare grids 32, 64 and 96 on the common 128-pixel detector, first at 750 nm and
then over 650/750/850 nm. Compare flat phase and 0.05-radian RMS tilt-x, tilt-y,
defocus, oblique astigmatism and coma-x. Construct analytic inputs independently,
remove their sampled piston and normalize their sampled pupil RMS.

Require relative L2 intensity error at most 1e-10 and absolute unit-power error
at most 1e-12. At each nonflat phase, compare the production JAX directional
derivative with a central difference of the HCIPy reference using step 1e-5 rad;
require relative L2 error at most 1e-6. Retain all pixels and derivatives.

For each grid/spectrum, also use a fixed asymmetric combination of modes and
deliberately reverse phase sign, increase separation by 10%, transpose the output,
or increase detector power by 10%. Require each wrong result to differ from the
reference by more than 1e-4 relative L2. These negative controls demonstrate that
the test can reject sign, separation, orientation and scale defects.

Invalid shapes, nonfinite phases, invalid sampling and nonpositive wavelengths
must raise clear errors. A failed physical gate is recorded with its full arrays
and prevents a success exit; do not widen tolerances or edit old measurements.
Any correction to the reference must be justified from the source/coordinate
contract and recorded separately from a production-model correction.

## Engineering review and test map

    fixed configuration -> independent analytic phase patterns
                           |                           |
                           v                           v
                       dLux/JAX                   HCIPy matrices
                           |                           |
                       JAX JVP                    finite difference
                           +----------+----------------+
                                      v
                         pixel/power/derivative gates
                                      |
                  raw arrays + source/library hashes + report

Use one reference module, one runner, a configuration and deterministic tests.
The small correctness tests exercise two grids, mixed wavelengths and binning;
the saved run covers the full protocol. Invalid-input tests and the negative
controls cover silent sign, scale, shape and layout errors. Explicit matrix
transforms avoid sharing the production FFT route; peak grids are only 384 square.
No timing samples or controller sweeps belong in this increment.

## Sources inspected

The installed 0.7.1 sources for `PyramidWavefrontSensorOptics`, `SurfaceApodizer`,
`FraunhoferPropagator`, `MatrixFourierTransform`, `make_uniform_grid` and wavefront
pixel power were inspected directly. Record their file hashes with the run. The
online development docs confirm the API concepts, but the lock and installed
source govern the version used here:

- [HCIPy pyramid API](https://docs.hcipy.org/dev/api/hcipy.wavefront_sensing.PyramidWavefrontSensorOptics.html)
- [HCIPy Fraunhofer API](https://docs.hcipy.org/dev/api/hcipy.propagation.FraunhoferPropagator.html)

## Reproduction

From the pinned repository environment, use a new output directory:

```sh
.venv/bin/python scripts/validate_pyramid.py --out runs/pyramid-validation
```

The runner saves every comparison array, the full configuration, machine details,
the source commit and hashes of the runner and inspected HCIPy files. It exits
unsuccessfully if any fixed gate fails. To rebuild the report from saved arrays:

```sh
.venv/bin/python scripts/validate_pyramid.py --out runs/pyramid-validation --render-only
```

The quick correctness suite includes `tests/test_pyramid_reference.py`. Neither
the tests nor this validation runner collect benchmark timing samples.
