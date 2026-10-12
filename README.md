# noether-ao

Config-driven experiments for civil-astronomy adaptive optics: synchronized dLux
propagation timings, a differentiable delayed AO simulation, empirical fault
detection, and fixed or learned blends of classical and forward-model commands.
This is a reproducible prototype. It is not a deployed observatory controller.

Repository: [noetherlabs/noether-ao](https://github.com/noetherlabs/noether-ao).
The Python package and command remain `noether-optics` / `noether_optics` so the
recorded v0 environments and reproduction commands keep their original names.
See [the milestone history](docs/history.md) and [next work](TODOS.md).

## Results

<!-- BEGIN HEADLINES -->

| Claim | Measured verdict |
| --- | --- |
| C1 | Conditional support. N=32, one wavelength, float32: p99 0.0592937 ms; N=256, five wavelengths: p99 6.38013 ms. CPU component only. |
| C2 | No tested flux gave useful equivalence across all fidelity levels. No supported low-flux threshold was found; lower fidelity often performed better under this fixed estimator. |
| C3 | Learned weights beat both integrator and forward in 0/12 conditions and lose to at least one in 5/12; remaining comparisons include uncertainty. |
| C4 | Partial detection/recovery, no general safety result. Optical-gain drift missed; vibration fallback worsened WFE; clean false alarms occurred. |

<!-- END HEADLINES -->

See [the result report](docs/results.md), [physical acceptance gates](docs/acceptance.md),
[research and source notes](docs/research_notes.md), and [milestone status](STATUS.md).
Every result directory contains its configuration, raw data, source commit/hashes,
machine fingerprint, tables and plots. Quantities in configuration files are inputs.
Measurements in reports are calculated from saved runs.

## Install

Use Python 3.11 and uv. Clone the private repository with an authorized GitHub
account, then install from its root:

```sh
git clone https://github.com/noetherlabs/noether-ao.git
cd noether-ao
uv sync --frozen --extra dragrace --python 3.11
.venv/bin/python -m pytest -q
```

The lock includes exact transitive versions and source commits. The optional
`dragrace` extra installs the other propagation libraries used in the contribution.
Omit that extra only if you do not need the cross-library benchmark. The saved
runs used the CPU backend on an Apple M1; no JAX GPU device was available.
On macOS, single-core affinity is explicitly skipped because the required API is
unavailable. Environment thread flags are not treated as enforced affinity.

## Run experiments

Run these from this directory. Each output directory must be new. The full
configuration is [simulation.yaml](configs/simulation.yaml); seeds, noise,
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

The simulation core is importable from `noether_optics.atmosphere`,
`noether_optics.optics` and `noether_optics.simulation`. Set `JAX_ENABLE_X64=1`
before importing JAX when calling these modules directly; the CLI sets it itself.

## Local Dragrace contribution

The separate local branch adds one AO MFT scan and appends p99 and observed
maximum to existing timing reports. It keeps the original physics and accuracy
gate. The patch is [contributions/dragrace.patch](contributions/dragrace.patch).
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
The issue and PR descriptions in [outbox](outbox/) are unsent upstream drafts.
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

The completed [fresh-checkout verification](results/verification/fresh-checkout/verification.json)
passed. Its [table report](results/verification/fresh-checkout/table-reproduction.json)
and [independent result review](results/verification/fresh-checkout/independent-result-review.json)
record unchanged raw inputs and numerical agreement of the full physical rerun.

## Limits

- The plant is a small unobstructed pupil, modal DM, ideal unmodulated pyramid and
  finite periodic frozen-flow atmosphere. HCIPy independently checks the open-loop
  PSF; it does not validate the whole sensor or instrument.
- Control results use short runs and a small held-out seed set. The predictor has
  ideal training telemetry. Classical gains are fixed, not exhaustively tuned.
  Learned weights use Gaussian training noise and Poisson evaluation noise.
- The detector has no proven safe set or recovery guarantee. Fault coverage and
  useful fallback in some cases do not establish safety in general.
- CPU latency distributions include desktop scheduling effects. No GPU, on-sky,
  hardware acquisition or hard real-time result is claimed.

The [decision log](DECISIONS.md) records modeling assumptions and interruptions.
Publication to this private repository is authorized. External correspondence and
the upstream Dragrace submission remain unsent. [Contributor instructions](CONTRIBUTING.md)
describe the checks and evidence required for subsequent commits.
