# Reproducing the experiments

Install the locked environment using the [quickstart](../README.md#quickstart).
All commands below run from the repository root. Use new output directories for
new acquisitions and keep timing runs isolated.

## Environment and optional adapters

The core installation is `uv sync --frozen --python 3.11`. Install the optional
cross-library dependencies before reconstructing the Dragrace contribution:

```sh
uv sync --frozen --extra dragrace --python 3.11
```

Set `JAX_ENABLE_X64=1` before importing JAX when using numerical modules directly;
the experiment CLI sets it itself. Saved timings used an Apple M1 CPU. GPU and
enforced single-core affinity were unavailable on that host and are recorded as
skipped. Thread environment flags are not treated as enforced affinity.

## Run baseline experiments

Run these from the repository root. Each output directory must be new. The full
configuration is [simulation.yaml](../configs/simulation.yaml); seeds, noise,
atmosphere, delays, fault settings and acceptance tolerances are explicit.

```sh
# M1: synchronized timing; run alone, outside CI
.venv/bin/noether-optics latency --config configs/latency.yaml --out runs/m1
# M3: independent physical checks and three closed-loop controllers
.venv/bin/noether-optics validate --config configs/simulation.yaml --out runs/m3
# M4: detector, all faults, and paired fallback comparisons
.venv/bin/noether-optics faults --config configs/simulation.yaml --out runs/m4
# M5: common plant, three twin fidelities, five photon levels
.venv/bin/noether-optics fidelity --config configs/simulation.yaml --out runs/m5
# M6: train modal weights, then evaluate all controllers on held-out seeds
.venv/bin/noether-optics mixed --config configs/simulation.yaml --out runs/m6
```

M5 references the preserved M1 component measurements at `results/m1`. To use
new timings, copy the simulation configuration and change `fidelity.m1_result`
to `runs/m1`. These are single FFT propagation costs, not whole pyramid sensor,
gradient, controller or hardware-loop timings. The optical loop has a simulated
millisecond timestep; that does not establish millisecond wall-clock execution.

These entry points reproduce the v0 experiment designs. The [predeclared controller
study](controller_study_v1.md) has committed training, validation, selection and
evaluation commands, documented in its [execution contract](controller_study_execution.md).
Do not execute the reserved configuration until the target passes the storage
preflight and the separately seeded runner smoke is complete.

Check the destination filesystem before creating a controller-study run:

```sh
.venv/bin/noether-optics study-resources \
  --config configs/controller_study_v1.yaml \
  --target runs/controller-study-v1
```

The command is read-only and prints one JSON report. For the frozen v1 plan it
derives 42,166,977,776 bytes of numeric payload and requires 105,033,365,984 free
bytes after conservative archive, metadata, temporary-write and operating envelopes.
It credits no NPZ compression and exits 2 when the target filesystem is too small.
This is storage planning, not a timing result or permission to use reserved cohorts.

Check the frozen optical-gain observability retention envelope separately:

```sh
.venv/bin/noether-optics observability-resources \
  --config configs/optical_gain_observability_v1.yaml \
  --target runs/optical_gain_observability_v1
```

This read-only command budgets 2,341,413,792 bytes of numeric arrays across the
maximum 1,504 trials and 1,568 numeric archives, then requires 10,190,604,096 free
bytes. It imports no JAX or simulator module, credits no compression, creates no target
and exits 2 on refusal. Passing this gate still does not authorize a P2 cohort; the
distinct observability runner smoke must pass first.

## Run the end-to-end controller smoke

From a clean committed checkout, run the complete command sequence on the reduced,
nonreserved smoke configuration:

```sh
.venv/bin/python scripts/run_controller_study_smoke.py \
  --out results/controller_runner_smoke_v1
.venv/bin/python scripts/run_controller_study_smoke.py \
  --rebuild results/controller_runner_smoke_v1
```

The runner completes all training and validation records, freezes selection, then
persists exactly one held-out evaluation record through the real stage commands. It
runs locally outside CI and records no elapsed time. The fixed 61001/62001/63001
seed families and their detector streams are disjoint from both the reserved study
and the earlier component smoke. This verifies orchestration and durable evidence;
it does not complete an evaluation cohort or support a controller comparison.
The committed reference execution is retained in
[`results/controller_runner_smoke_v1`](../results/controller_runner_smoke_v1/README.md);
its rebuild verifies 488 protected files and reproduces the derived report exactly.

## Local Dragrace contribution

The separate local branch adds one AO MFT scan and appends p99 and observed
maximum to existing timing reports. It keeps the original physics and accuracy
gate. The patch is [contributions/dragrace.patch](../contributions/dragrace.patch).
To reconstruct the branch in a new sibling directory:

```sh
git clone https://github.com/Jashcraf/Diffraction-Dragrace ../Diffraction-Dragrace
git -C ../Diffraction-Dragrace checkout -b codex/ao-tail-latency 1bf53802950c5f76905d3dee0ba73b75147401bf
git -C ../Diffraction-Dragrace apply "$PWD/contributions/dragrace.patch"
NOETHER_PYTHON="$PWD/.venv/bin/python"
(cd ../Diffraction-Dragrace && JAX_ENABLE_X64=1 PYTHONPATH=src "$NOETHER_PYTHON" -m pytest -q)
.venv/bin/python scripts/run_dragrace.py --repo ../Diffraction-Dragrace --out runs/m2
```

This runs every installed adapter. An installed adapter may still report an
unsupported case. Timing never runs in CI. No `reviewed_by` field was filled.
The issue and PR descriptions in [contribution drafts](../contributions/drafts/) are unsent upstream drafts.
This private repository archives the patch; it does not submit it upstream.

## Reproduce the published tables

In a fresh local clone, install from the lock and reconstruct the Dragrace sibling
as above. Then run:

```sh
.venv/bin/python scripts/rebuild_all.py --dragrace ../Diffraction-Dragrace --report results/verification/rebuild.json
```

This recomputes tables and figures from the preserved raw arrays and timing
distributions, checks every published numerical table, and verifies that raw
inputs did not change. It does not claim that new timing samples will equal old
ones. Full experiments are rerun with the commands above.

For shorter functional checks of the experiment entry points, substitute
`configs/reproduction_smoke.yaml` for the simulation configuration and
`configs/latency_smoke.yaml` for the timing configuration, using fresh output
directories. These smoke results are excluded from scientific claim tables.
The timing smoke still takes the required number of samples at its selected point.
The verification report records exactly which checks ran in the fresh checkout.

To run that verification sequence yourself (including fresh measurements):

```sh
.venv/bin/python scripts/verify_checkout.py --dragrace ../Diffraction-Dragrace --out runs/verification
```

It reruns the full physical validation and the full Dragrace AO case; other
entry-point checks use the labeled shorter configurations. Run it outside CI,
with no concurrent compute work, because it includes timing.

The completed [fresh-checkout verification](../results/verification/fresh-checkout/verification.json)
passed. Its [table report](../results/verification/fresh-checkout/table-reproduction.json)
and [independent result review](../results/verification/fresh-checkout/independent-result-review.json)
record unchanged raw inputs and numerical agreement of the full physical rerun.
