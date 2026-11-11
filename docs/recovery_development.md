# Recovery development pilot

This pilot compares continued prediction with two scheduled transfers to the
validation-selected leaky integrator. It measures resource use and paired
variability before a separate recovery evaluation is frozen. It contains no
held-out cohort or detector and cannot establish a recovery performance claim.

## Physical comparison

The plant is the same 32-sample, 100-mode, unmodulated pyramid-sensor simulation
used in [Prediction Boundary](../results/controller_study_v1/README.md): 750 nm,
1 m aperture, 1 ms frames, two-frame command delay and a ±5 rad modal command
bound. Each episode has 512 frames, with burn-in ending at 128 and the event
horizon spanning frames 256–511.

Four development roots are crossed with two photon counts (1,000 and 1,000,000),
two wind multipliers (0.5 and 2), and three conditions: clean, optical-gain drift
and vibration. Every condition retains all three arms, for 144 trials. The clean
condition applies the same scheduled event as a false-intervention control.

The frozen `predictive-01` coefficients and `integrator-04` gain 0.5/leak 0.99
are reused without fitting or recovery-specific tuning. Calibration callables
are reconstructed, and every numeric calibration array must exactly match its
retained training counterpart before acquisition begins. Phase histories and
detector roots are shared across arms. Identical keys do not imply identical
Poisson samples after the feedback trajectories diverge.

The oracle supplies the intervention frame only. Issued-state transfer preserves
the existing command queue. Command matching preserves the predictor request on
the event frame, then transfers; for this controller it is exactly the ordinary
transfer one frame later. See [handover mechanics](controller_handover.md).

## Outcomes and costs

Both intervention arms use the same event horizon and zero-DM denominator for
integrated excess phase variance. Positive cost means intervention was worse.
[Outcome reductions](recovery_outcomes.md) also retain absolute WFE, pupil
Strehl, peak error, modal command changes and saturation. A reference return
requires 32 consecutive frames at or below 1.1 times the pre-event mean variance.
Its start, confirmation, right censoring and subsequent relapse are separate.

The first call to each kernel includes JIT compilation. Subsequent calls are
marked warm. Per-call timing includes synchronization and host conversion;
archive writes are outside that interval. Inclusive acquisition time includes
calibration, phase generation, compilation, execution and artifact writes.
These measurements are resource costs, not frame latency or a telescope deadline.

## Running a development attempt

The command requires the retained controller acquisition, including the exact
base configuration, study, training, selection, predictor and calibration hashes
declared in the module. Current base/study definitions must match those bytes.
The public controller report alone does not contain all required raw artifacts.
Use a clean committed checkout, the locked CPU environment, and external process
supervision that bounds wall time, memory and artifact volume. The runner also
requires 105,033,365,984 free bytes at admission and between cases.

```bash
python -m noether_optics.recovery_development \
  --controller-run /path/to/controller-acquisition \
  --out /path/to/new-development-smoke --smoke
```

The smoke runs one clean, low-flux, slow-wind triplet using the first development
root. Omitting `--smoke` runs the complete declared development schedule. Smoke
data are not independent additions to that four-root schedule. A distinct
prospective cohort still needs an exclusion inventory, precision calculation,
fixed size, inference family and failure policy before evaluation.

Each attempt uses a fresh output directory. An immutable intent precedes every
controller call; raw arrays precede the terminal record. Produced nonfinite
arrays remain attached and withhold paired outcomes. Unexpected interruptions
leave a stopped attempt and cannot be resumed automatically. The completion
record is written only after source, training inputs and previously published
artifact hashes are rechecked. It binds all retained files, including full
measurement and prediction histories. Hashes detect changes; they do not prove
instrument authenticity or provide a hostile-writer security boundary.
