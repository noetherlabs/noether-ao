# Scheduled fallback: experiment and measurement contract

This experiment measures the finite-horizon cost of a prescribed handover from
a selected temporal predictor to a selected leaky integrator. The two policies
operate on a shared pyramid-sensor observation path. The switch is independent
of observed fault evidence.

The [result and retained arrays](../results/scheduled_fallback_v1/README.md)
contain all six atmospheric realizations, both sensor conditions and both
actions. The public package reconstructs measurements from those arrays; it
does not reproduce the acquisition or the preceding policy-selection campaign.

## Design

The six root seeds are 282001001–282001006. Each root defines a shared
128-frame phase sequence. Four trials use that sequence: nominal and drifting
sensor response, each with continued prediction or scheduled fallback. Root
realizations are the units of statistical replication.

The fitted predictor coefficients and integrator parameters are held fixed.
The predictor uses ridge 0.01; the integrator uses gain 0.5 and leak 0.999.
The selected parameter arrays and their digests are included. Historical
selection digests are provenance references, not a public reconstruction of
the training and selection campaign.

The plant uses a 1 m pupil, Fried parameter 0.4 m, eight DM modes, a 16 × 16
pupil grid, a 32 × 32 pyramid-sensor detector and wavelength 750 nm. Each frame
has a nominal photon budget of 100,000 and read noise 0.3 electrons. The
simulation timestep is 1 ms, the command delay is two frames, and each modal
command is clipped to ±5 rad. These sampling and command bounds are simulation
parameters, not instrument specifications.

For the perturbed condition, the noiseless sensor image is

$$I_t = I_0 + g_t\,[I(r_t)-I_0],$$

where $r_t$ is the piston-removed closed-loop residual phase,
$I_0$ is the flat-wavefront image and $g_t$ decreases linearly from 1 at
frame 64 to 0.35 at frame 96, then stays at 0.35. Camera noise is drawn from the
resulting image. Before the perturbation, the unmodified nominal image is used
exactly. This construction changes sensor contrast while preserving the
underlying atmospheric input and the definition of the physical residual
metric. The resulting closed-loop residual trajectories can change.

At frame 80, the switching arm selects the integrator candidate using the last
issued command and current reconstructed residual:
$u_t=\operatorname{clip}(0.999u_{t-1}+0.5\hat r_t)$.
It retains the existing command history and two-frame FIFO delay queue; the
last issued command can differ from the currently applied command. The first
changed command is applied at frame 82. Shared exogenous seeds do not imply
identical camera samples after the closed-loop optical states diverge.
The predictor continues to run in shadow after the switch, and its invalidity
still invalidates the whole trial. The experiment therefore does not establish
recovery from a failed primary implementation.

## Primary outcome

Let $\phi_{r,t}$ be the retained atmospheric phase, $B$ the retained modal
basis, and $u^a_{r,t}$ the actually applied command in arm $a$. The verifier
reconstructs the residual $\phi_{r,t}-Bu^a_{r,t}$, removes pupil piston and
computes its pupil-averaged squared phase $V^a_{r,t}$ in rad².

The same calculation with zero DM command gives $V^0_{r,t}$. For each root and
condition, the primary contrast is

$$\Delta_r =
\frac{\sum_{t=82}^{127}\Delta t\,[V^{\mathrm{switch}}_{r,t}
       -V^{\mathrm{continue}}_{r,t}]}
     {\sum_{t=82}^{127}\Delta t\,V^0_{r,t}}.$$

The denominator is strictly positive and shared by the paired arms. The
contrast is not normalized by the continuing controller and is not a percentage
reduction. Negative values favor the scheduled switch.

Each condition reports all six contrasts and their unweighted mean. The
predeclared interval is $\bar\Delta\pm t_{5,0.9875}s_\Delta/\sqrt{6}$:
a two-sided 97.5% interval per condition, giving a Bonferroni family error
allocation of 5% across the two conditions under the stated Student-t
assumptions. No frame-level confidence interval or post-hoc acceptance test is
used. A nonfinite or invalid trial withholds its whole pair and any complete
cohort mean or interval; no replacement seed or survivor average is permitted.

## Secondary outcomes and evidence checks

The same normalized contrast over frames 64–127, peak residual variance,
saturated-frame counts and command changes are descriptive secondary outcomes.
Reference return means the first eight consecutive frames below 1.1 times the
continuing arm's mean variance over frames 32–63. The search covers frames
82–127. Failure to observe such a visit is right-censored; an early visit does
not establish sustained recovery.

The public verifier checks exact causal prefixes between paired arms and
between nominal and perturbed conditions, reconstructs phase variance, WFE and
coherent pupil Strehl, and checks delayed/clipped command consistency. It
compares reconstructed results with every retained trial and summary. Integrity
hashes bind exported bytes; they do not authenticate historical acquisition
times, validate the optical simulator against hardware, or supply a stability
or safety certificate.
