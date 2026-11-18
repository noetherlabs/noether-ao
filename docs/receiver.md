# Passive Gaussian receiver observation

`noether_optics.receiver.evaluate_receiver` evaluates single-mode coupling from
the spatial residual of the existing modal AO simulation. It runs **after** the
control loop. It neither changes the controller nor supplies a measured sensor
signal. The overlap is computed by the shared `noether-physics` package.

The returned value is a fraction of incident pupil power coupled into an ideal,
centered Gaussian mode. It does not include receiver-power detector noise, absolute incident
flux, throughput, scintillation, polarization or modem behavior. A change in
this value is not a measured link or delivered-bit improvement.

## Units and timing

The atmosphere and modal commands use radians at `config['wavelength_m']`.
For frame $t$, the adapter reconstructs the same piston-free phase used by the
loop:

$$
r_t = M\left(\phi_t-\sum_j u^{\mathrm{applied}}_{tj}B_j\right),\qquad
\widetilde r_t=r_t-M\frac{\sum r_t}{\sum M},\qquad
d_t=\frac{\lambda_{\mathrm{ref}}}{2\pi}\widetilde r_t.
$$

Here $M$ is the binary loop pupil and $d_t$ is OPD in metres. There is no extra
factor of two: these modal commands already represent compensating phase.
A reflective physical mirror instead adds twice its surface displacement;
that conversion belongs in its own command-to-OPD map.

The receiver wavelength may differ from the phase reference wavelength. Both
are explicit arguments. Using a new reference wavelength to represent the
**same physical OPD** must preserve coupling at a fixed receiver wavelength.
Changing the receiver wavelength generally changes coupling.

The loop applies `outputs['command'][t - delay_frames]` at frame `t`, with an
initial queue of zeros. The adapter verifies that exact relationship before
using `outputs['applied']`. It then compares the reconstructed spatial variance
with `outputs['variance']`, using `rtol=5e-12` and `atol=5e-14` in radians
squared for float64 reduction-order differences. It does not infer coupling
from the scalar variance or Strehl ratio.

## Installation

From this repository, install the pinned optional receiver dependency:

```bash
uv sync --locked --extra receiver
```

The extra references an immutable commit of `noether-physics` in
[Noether Link Audit](https://github.com/noetherlabs/noether-link-audit/tree/d4259edb2573f03c743f3fb55e792dff6e2da04d/packages/noether_physics).
It keeps the AO project's existing numerical dependency versions. The receiver
conformance suite is required in CI; the base simulator remains usable without
this extra.

## Example

Given `config`, `cal`, `phases` and the corresponding complete `loop` output:

```python
from noether_optics.receiver import evaluate_receiver

receiver = evaluate_receiver(
    config,
    cal,
    phases,
    outputs,
    phase_reference_wavelength_m=config["wavelength_m"],
    receiver_wavelength_m=1.55e-6,
    focal_length_m=0.1,
    mode_radius_m=5e-5,
    source_id="development-run-001",
    fault="none",
)

coupling = receiver["coupling"]
frame_index = receiver["frame_index"]
time_s = receiver["time_s"]
```

The receiver geometry in this example is illustrative. `mode_radius_m` is the
Gaussian **field** radius at which amplitude falls to $1/e$ in the focal plane,
equivalently the $1/e^2$ intensity radius. Pupil pitch is
`config['diameter_m'] / config['n']`.
Arrays use `[frame, y, x]`; both pupil axes are interpixel-centered and increase
along `receiver['pupil_axis_m']`. Frame times are `frame_index * config['dt_s']`
relative to the beginning of the simulation, not instrument timestamps.

By default the pupil amplitude is the loop's binary mask. Supply
`pupil_amplitude` for fixed nonnegative illumination or obscuration inside that
mask. This changes receiver-side weighting only; it does not recalibrate the
wavefront sensor or rerun the atmosphere and controller. A common-path pupil
change would require a new, self-consistent AO model. Uniform amplitude scaling
leaves the power fraction unchanged. The
receiving Gaussian retains its full-plane normalization; truncating the pupil
does not renormalize the receiving mode.

## Checks and limits

- Input phase, basis, issued/applied commands and variance must be finite
  float64 arrays with complete episode shapes. The phase reference must match
  the loop configuration. Nonfinite loop outputs are rejected.
- `fault` is required. Only `none` and `optical_gain` are supported: optical-gain
  drift changes sensor response, so the original spatial phase and applied
  command still define the physical residual. Other faults are rejected even
  when their configured active window is empty.
- No residual image cube is retained. The result contains scalar coupling,
  frame/time arrays and receiver/source metadata. All original arrays and loop
  outputs remain untouched.
- `source_id` is supplied by the caller. FIFO and variance checks detect
  inconsistent timing or scale; they do not authenticate historical arrays.
  In particular, variance cannot distinguish different spatial fields with
  the same energy. The caller must preserve the original phase, calibration,
  configuration and output pairing in the run's provenance.

The conformance tests use an exact finite-cell interference formula, two phase
reference wavelengths, a nonzero-aberration command derivative, and the
physical reflective-surface factor of two. An asymmetric spatial field checks
array orientation directly; a centered Gaussian scalar overlap alone cannot
detect every transpose or reflection. Small executions of the existing loop
check command-arrival timing, future-input causality and nonmutation for both
supported faults. These are interface checks, not a controller comparison or
hardware validation.

Keep this evaluator trace separate from controller inputs. A future measured
receiver-power stream would need its own noise, calibration and availability
contract. Its existence would not reveal the true spatial phase to the
controller.
