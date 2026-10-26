# Closed-loop response identification

The identification adapter estimates a static command-to-observation response
from measured modal telemetry. It then compares nominal, ordinary least-squares
(OLS) and instrumental-variable (IV) calibration within the same one-step sensor
prediction pipeline. This is a component implementation; a complete held-out
comparison and instrument validation are separate experiments.

## Observation and collection model

For row-vector samples within one stationary calibration interval,

$$
y_t = d_t - x_t B + n_t,\qquad
x_t = c_{t-\ell},\qquad z_t = p_{t-\ell}.
$$

Here $y$ is the calibrated residual observation, $x$ is the actual delayed
post-clipping modal command, $p$ is an independently generated probe request,
and $B$ is the effective command response. The instrument $z$ is a delayed
request, not a separately measured probe contribution after clipping. Modal
commands are radians at the reference wavelength, not mirror displacement.

The existing integrator retains its collection law:

$$
f_t = \lambda c_{t-1} + k R(m_t-m_0),\qquad
c_t = \operatorname{clip}(f_t+p_t).
$$

Consequently, a probe enters the next controller state through the issued
command. The opt-in loop records `feedback_requested`, `probe_requested`,
`combined_requested`, post-clipping `command` and delayed `applied` separately.
It also records the injected measurement perturbation. Existing non-instrumented
calls retain their original arithmetic path and output keys.

The adapter accepts the full linear modal sensor, an integrator collection
controller and either no fault or the existing optical-contrast-gain fault.
Stuck-actuator distortion is excluded because a nominal modal command does not
fully describe the distorted physical surface. The reference response used by
the evaluator is obtained from masked, piston-removed basis responses through
the actual plant and reconstructor. It includes calibration regularization.

## Response estimation

OLS fits the centered equation directly. IV first projects centered commands
and observations onto an orthonormal basis $Q_Z$ for the instruments:

$$
\widehat B_{\mathrm{OLS}} = -\operatorname{lstsq}(X,Y),\qquad
\widehat B_{\mathrm{IV}} = -\operatorname{lstsq}(Q_Z^\top X,Q_Z^\top Y).
$$

The required population assumptions are independent external excitation,
$\mathbb E[Z^\top(D+N)]=0$, and full-rank $\mathbb E[Z^\top X]$.
Feedback can violate the OLS regressor/noise independence assumption. IV is an
established approach to this problem; it does not establish exogeneity merely
by fitting the data. See [Gilson and Van den Hof (2005)](https://doi.org/10.1016/j.automatica.2004.09.016).

The implementation uses SVD and least-squares solves, without normal-equation
inverses or a sample-sized projection matrix. It retains ranks, singular values
and canonical correlations. The minimum canonical correlation is a declared
descriptive relevance gate, not an iid first-stage F test or confidence level.
Rank deficiency and weak instruments withhold the fit. Saturated samples remain
in the fit; selectively dropping them would alter the experimental population.

The estimated response combines WFS sensitivity, calibration and DM response.
Those factors cannot generally be separated without an independent anchor.
Likewise, DM probes do not instrument atmospheric dynamics after exact command
cancellation. This implementation is not Internal Controller Identification
(ICI), the distinct nonlinear parameterization introduced by
[Meroi et al. (2026)](https://arxiv.org/html/2610.27970v1).

## Controller-filtered instruments

The reference instrument replays the external probe through the known nominal
collection law, without accessing actual measurements, commands or faults:

$$
x_t^p=c_{t-\ell}^p,\qquad
f_t^p=\lambda c_{t-1}^p-kx_t^pB_0,\qquad
c_t^p=\operatorname{clip}(f_t^p+p_t),\qquad z_t=x_t^p.
$$

Here $B_0=(R_0H_0)^\top$ comes from the declared reconstructor and interaction
matrix. It includes their regularization and is distinct from both the
evaluator's physical finite-grid response and the $I$ baseline below. All
reference parameters must be fixed before collection. The deterministic replay
preserves external-probe independence; it does not establish instrument
relevance, optimal variance or validity under an incorrectly specified
observation equation. Model-based instrument construction is established
[closed-loop IV practice](https://doi.org/10.1016/j.automatica.2004.09.016).

Replay starts with the same zero command/FIFO and includes the complete burn-in.
The state follows the clipped issued reference command. The integrator does not
clip its residual estimate. Actual plant saturation and a response estimated
from the recording are never inputs to this replay. The raw and filtered
instruments use the same response solver, predictor, relevance cutoff and
training episodes.

The retained development recordings can be reanalysed without acquiring another
trajectory:

```sh
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .venv/bin/python scripts/compare_reference_iv.py --out runs/reference-iv
```

The command requires clean committed source and verifies the original artifacts
and their independent reconstruction. Its completion manifest distinguishes
acquisition source from analysis source and binds all consumed files. Its
256 MiB free-space check covers only this small derived output; it does not
change a full-study resource gate. The comparison remains exploratory because
these development recordings have already been examined.

The [retained comparison](../results/controller_filtered_iv_v1/README.md) found
higher response and forecast error with this reference filter than with raw
delayed probes, despite both candidates passing the relevance cutoff.

## Synthetic mismatch and nominal-response restoration

For a declared raw-sensor column transform $S$, the linear adapter replaces the
complete clean response by $m_0+S(h(\phi-\mathrm{DM})-m_0)$ before adding noise.
It transforms both atmosphere and DM contributions. Physical phase, basis and
wavefront-error computation remain unchanged. This defines a synthetic coupled
modal-response error, not a model of nonlinear pyramid optical gain.

With a nominal column Jacobian $H_0$ and fixed reconstructor $R_0$, an estimated
row response $\widehat B$ defines

$$
R_{\mathrm{new}}=B_0^\top\operatorname{solve}(\widehat B^\top,R_0),\qquad
B_{\mathrm{new}}=B_{\mathrm{true}}\widehat B^{-1}B_0.
$$

An exact response estimate restores the nominal transfer; supplying $B_0$
preserves the original reconstructor. This policy does not recompute a Tikhonov
penalty in new sensor coordinates. It never substitutes a fitted intercept for
the detector reference. Singular estimates and corrections exceeding an
explicit reconstructor operator-norm budget are withheld. These numerical
checks do not prove closed-loop stability. Physical outcome evaluation still
needs independent unprobed trajectories and retained failures.

For $\widehat B=sB_0$, the correction reduces to $R_0/s$. Under the qualified
integrator this is equivalent to changing its gain from $k$ to $k/s$, including
clipping and command memory. Scalar calibration recovery therefore cannot alone
establish superiority over gain tuning. Adaptive modal gain control is already
an established AO approach; see [Deo et al. (2021)](https://arxiv.org/abs/2103.09921).
The present adapter does not implement CLOSE. Deployment is qualified only for
the integrator; forward-model controllers and detector scores need a coherent
updated observation model before using the corrected reconstructor.

## Common prediction endpoint

For each response estimate, reconstruct a disturbance-coordinate sequence
$q_t = y_t\widehat B^{-1}+x_t$ and fit the existing per-mode two-tap predictor.
The nominal baseline sets $\widehat B=I$, matching the existing measured
pseudo-open-loop reconstruction. Response matrices and predictor coefficients
are fitted separately per training episode and averaged with equal episode
weight. No transition crosses an episode boundary. Any withheld training fit
withholds the candidate; opposing response estimates cannot cancel into a
numerically invertible near-zero calibration.

At a forecast origin **after frame $t$ command issue**, the predicted next
residual is

$$
\widehat y_{t+1|t} =
\left(a_1\odot q_t+a_2\odot q_{t-1}-c_{t+1-\ell}\right)\widehat B.
$$

For delay $\ell\ge1$, the command is already known. Future recorded residuals
and future command requests are not predictors. Early targets whose commands
belong to an unspecified initial queue are omitted. Synchronization with applied
commands is essential to [pseudo-open-loop reconstruction](https://arxiv.org/abs/1903.12124).

Response error and noisy sensor-prediction error are different endpoints.
An incorrect response can still predict observations under one collection
controller by exploiting correlated noise. Neither metric establishes that
deploying an alternative controller would improve wavefront error.

## Component checks

The retained [development measurements](../results/identification_component_smoke_v1/README.md)
include raw trajectories and an independent NumPy reconstruction. They show
lower response error for IV than OLS, higher forecast error for both than the
nominal calibration, and a positive probing cost in every paired recording.

```sh
CI=true JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .venv/bin/python -m pytest -q tests/test_identification.py
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .venv/bin/python scripts/check_identification.py --out runs/identification-smoke
```

The runner requires clean committed source, a new output directory and 512 MiB
free space. Its [fixed configuration](../configs/identification_smoke.yaml) uses
only development identities. Training and forecast episodes have distinct
atmosphere/detector roots; probe and additive-noise identities are separate from
both. The same exogenous draws pair probed and unprobed collection episodes.

The additive AR(1) process starts at stationary variance and obeys
$\operatorname{Cov}(n_i,n_j)=\sigma^2\rho^{|i-j|}$. Changing $\rho$ preserves
its declared marginal variance. This perturbation is added to the existing
white sensor noise; the resulting process is a mixture. No whitening, filtering
or decimation is performed implicitly.

The smoke retains raw simulator arrays, phase truth in evaluator artifacts,
all fitted models or withholding reasons, forecasts, noisy forecast losses,
probe RMS, saturation and paired variance changes. Drift testing freezes the
fit and reports pre-drift and settled-drift windows separately. It measures
transfer to changed calibration, not adaptation. A final manifest binds source,
configuration, runtime and artifact digests. An interrupted run has no completion
manifest and is not silently overwritten or resumed as a different experiment.

These checks do not select a winner. A research comparison still needs a frozen
complete cohort, validation-only choices, independent evaluation trajectories,
episode-level uncertainty and all failures. The existing controller and P2
observability studies keep their protocols, reserved identities and storage gates.
