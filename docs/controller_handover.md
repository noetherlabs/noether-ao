# Scheduled controller handover

`scheduled_handover_kernel` executes a predictive-to-integrator transition in the
existing delayed optical loop. It provides an explicit boundary and command
trace for experiments that separate recovery actions from fault detection. It
does not define a detector, acquire a scientific cohort or establish beneficial
recovery.

```python
from noether_optics.recovery import scheduled_handover_kernel

execute = scheduled_handover_kernel(
    config, calibration, frozen_predictor,
    frame=event_frame, mode="issued_state", fault="optical_gain",
)
outputs = execute(phases, atmosphere_seed)
```

The configuration supplies the target integrator's gain and leak. The predictor
and configuration are copied when the kernel is constructed. Calibration is the
same immutable calibration used by the existing simulator. Phase truth is an
input to the physical plant and evaluation; the controllers receive sensor
measurements, calibrated parameters and their own histories.

## Two transition conventions

Let $p_k$ be the outgoing predictor request, $e_k$ the reconstructed residual,
$u_{k-1}$ the last **issued** command, and $S$ the configured modal clip. The
integrator request is $i_k=\ell u_{k-1}+g e_k$.

| Mode | At the event frame $k_f$ | Subsequent frames |
| --- | --- | --- |
| `issued_state` | Issue $S(i_{k_f})$. | Apply the same integrator recurrence. |
| `command_matched` | Issue the outgoing $S(p_{k_f})$. | Apply the integrator recurrence using that issued command as memory. |

The second convention can be obtained by setting the integrator state to
$(S(p_{k_f})-g e_{k_f})/\ell$ at the boundary. The implementation uses the exact
output identity rather than subtracting and adding nearly cancelling values.
It matches the outgoing command at that frame; it need not equal the previous
issued command and therefore does not imply zero slew.

**For this one-state integrator, command matching is exactly an issued-state
transfer one frame later.** Both produce the same physical trajectory with
the same inputs. This equivalence is tested, including clipping, delayed
commands and state-dependent detector noise. It is not a new controller or
an independent transient-compensation mechanism. Controllers with additional
observer or anti-windup states need a separate matching rule.

Neither convention resets the FIFO. With delay $d$, an ordinary transfer can
first affect the plant at $k_f+d$; the command-matched transfer can first affect
it at $k_f+1+d$. The applied command at the boundary can differ from the last
issued command. Keeping that distinction is essential when measuring transients.

## Retained outputs

Both modes retain the existing physical outputs and add:

| Field | Meaning |
| --- | --- |
| `handover_triggered` | The scheduled event has occurred, including the boundary frame. |
| `handover_target_active` | The integrator request is selected on this frame. |
| `outgoing_requested` | The predictor request computed on the realized trajectory, before clipping. |
| `handover_requested` | The selected request before clipping. |

After trajectories diverge, `outgoing_requested` is a shadow calculation on the
intervened history. It is **not** the command from a counterfactual continuation
run. A paired continue arm is needed to measure intervention cost.

The existing `fallback` field remains false: it records detector-driven fallback,
which this API excludes. Simultaneous detector fallback, external probing,
measurement perturbations and differentiable training noise are rejected by the
scheduled path. The default loop and output schema remain unchanged when no
handover is requested.

The kernel rejects incomplete or nonfinite atmosphere inputs and invalid frozen
coefficients. Nonfinite outputs produced during execution remain in the returned
streams for the study runner to retain and classify. The predictor continues in
shadow after transfer; this interface does not demonstrate recovery from a
failed primary implementation. Terminal record, retry and cohort rules belong
to a separately declared study protocol.

## Interpretation

An event supplied at the known fault onset is an oracle experimental control,
not a deployable detection method. A fault-free run with the same event measures
a prescribed false intervention. Neither yields a false-alarm rate by itself.
Command continuity also does not establish reduced optical error, stability,
physical actuator safety or useful recovery. Measure paired finite-horizon
[residual cost, command changes and sustained recovery](recovery_outcomes.md),
alongside absolute performance.
The modal clip is a simulation bound; physical stroke and slew limits require
an instrument-specific actuator mapping.

The earlier [scheduled fallback study](scheduled_fallback.md) retains its own
source, parameters and evidence. This API does not change or regenerate it.

Controller switching has established AO prior art, including
[Raynaud et al. (2016)](https://doi.org/10.1117/12.2234266). Broader bumpless-transfer
theory addresses plant-output transients, not only one matched command; see
[Zaccarian and Teel (2005)](https://doi.org/10.1016/j.automatica.2005.02.003).
Those results do not supply stability guarantees for this optical simulation.
