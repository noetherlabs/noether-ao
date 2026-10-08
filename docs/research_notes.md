# Research notes

Opened during this session. Literature results are context, not measurements of
this project. No published performance numbers are copied into our result tables.

## Literature

- [Nousiainen et al., model-based AO reinforcement learning](https://arxiv.org/html/2104.13685v1).
  Opened full HTML and abstract. A learned dynamics model supports policy improvement
  in a delayed AO loop. Calibration changes and continual learning matter. Our small
  weighting network is not an implementation or reproduction of PO4AO.
- [Landman et al., model-free AO reinforcement learning](https://arxiv.org/html/2108.11332v1).
  Opened full HTML and abstract. Recurrent policies use measurement history for
  prediction and vibration rejection. This motivates wind/fault holdouts; a gain
  learned on one frozen-flow realization is not evidence of generalization.
- [Guyon and Males, EOF prediction](https://arxiv.org/html/1707.00570v1).
  Opened full HTML and abstract. History vectors and regularized linear least squares
  predict future wavefronts. Use a causal autoregressive baseline, with training
  turbulence distinct from evaluation turbulence. Do not expose future truth to control.
- [Haffert et al., subspace predictive AO](https://arxiv.org/html/2103.07566v1).
  Opened full HTML and abstract. The controller learns from residuals and command
  changes. This is a stronger baseline than the simple predictor implemented here;
  no claim of reproducing DDSPC is warranted.
- [Males et al., MagAO-X instrument](https://openresearch-repository.anu.edu.au/items/5d8f28b5-9955-4bd0-91ac-2e65841ceda6).
  Opened primary repository abstract. Describes a visible-light extreme-AO instrument
  with a pyramid sensor and high actuator count. Our unobstructed, small modal plant
  is not a MagAO-X digital twin. Telescope-specific geometry remains unknown.
- [Por et al., HCIPy](https://docs.hcipy.org/dev/index.html).
  Opened official documentation and installed propagation/grid source. Modular
  scalar optics and atmospheric layers provide an independent PSF reference.
  The linked author-poster fetch failed because of its size; not treated as read.
- [Chambouleyron et al., pyramid optical gains](https://arxiv.org/html/2006.08294v1).
  Opened full HTML and abstract. Pyramid response changes with residual phase;
  calibrating only at a flat wavefront can bias the interaction matrix. Include
  optical-gain drift and report nonlinear sensor limitations explicitly.
- [Sha et al., Simplex architecture overview](https://www.sei.cmu.edu/library/a-software-architecture-for-dependable-and-evolvable-industrial-computing-systems/).
  Opened primary technical-report page. Separate advanced control, a dependable
  controller, and decision logic. A discrepancy detector alone does not prove a
  safe invariant set or recoverability. This project's fallback is empirical.
- [Desdoigts et al., dLux I](https://arxiv.org/html/2406.08703v1).
  Opened full HTML and abstract. Joint differentiable calibration of detector and
  optical parameters motivates explicit forward models and derivative checks.
  Offline calibration success is not an online deadline measurement.
- [Desdoigts et al., dLux II](https://arxiv.org/abs/2406.08704).
  Opened abstract. Fisher-information objectives support optical design through
  differentiable propagation. It does not validate our controller or noise threshold.
- [Desdoigts et al., Amigo](https://arxiv.org/html/2510.09806v1).
  Opened full HTML and abstract. End-to-end interferometer calibration includes
  detector effects and learned components. It concerns a different instrument and
  inference task; no latency or control claim is transferred here.

## Source and API inspection

- [Dragrace](https://github.com/Jashcraf/Diffraction-Dragrace): cloned locally;
  README, docs text, all case/config mappings, and dLux adapter source inspected.
  File hashes are in source_manifest.json. Cases separate physics from backends;
  scan points each have accuracy gates; timing includes synchronization; stored
  distributions permit tail reporting without changing the measurement path.
  The README forbids CI timing. Leave reviewed_by empty.
- Dragrace docs/conventions.md: interpixel versus pixel centering and SI conversions
  are physical comparison inputs. Never fit away a spatial sampling mismatch.
- docs/methodology.md: documented API contracts, setup separation, affinity and
  realized CPU/wall ratios. The checked-out config comments contain stale thread
  flag language; the worker implementation is authoritative.
- docs/flop_model.md: arithmetic, transcendental evaluations and byte traffic are
  distinct measures. No derived FLOP estimates are presented as measured speed.
- docs/gradient_board.md: real-parameter gradients need finite-difference checks;
  copying intermediate complex cotangents across libraries can conjugate twice.
- docs/phase_retrieval_board.md: ambiguity, initialization and stopping criteria
  can change what timing means. Not part of our small forward-board contribution.
- docs/parsing_results.md: retain status, contract and machine grouping; scan timing
  is nested under each point; absent distributions must not become invented tails.
- docs/adding_an_adapter.md and example_idiomatic.py: time the documented path and
  synchronize inside the clock. We add a case/report feature, not an adapter.
- [dLux documentation](https://louisdesdoigts.github.io/dLux/latest/) and installed
  dLux source: Wavefront stores phasor; set returns an updated immutable pytree;
  add_opd accepts metres. propagate_FFT applies a centered, unitary-scaled transform.
  Its default padding is overridden explicitly. propagate uses legacy MFT;
  newer method names can be unimplemented. Do not infer availability from names.
- [abcdLux](https://github.com/LouisDesdoigts/abcdLux): README, overview, and installed
  fraunhofer/mft source read. Cached separable kernels have explicit input/output
  coordinates and throughput scaling. It is developing code, not evidence of a
  speed improvement on this host. Locked to the inspected commit.
- [zodiax](https://github.com/LouisDesdoigts/zodiax): installed Base/set/multiply
  implementation read. Functional updates use Equinox tree replacement, so dynamic
  OPD arrays can enter compiled dLux functions without mutable hidden state.

## Design for M1–M6

M1 uses unit-power circular pupils, seeded dynamic OPD, separate wavelength FFT
planes, explicit precision, ahead-of-time compilation and warmup. Every clock
includes block_until_ready. Compare complete complex fields before timing. Record
all samples, tails, deadline misses, CPU/wall, compile time and forced-copy path.
Unavailable affinity/GPU modes are skips, never performance claims.

M2 copies the existing MFT scan physics and changes only its size/execution scope.
Add p99/max derived from distributions without replacing existing report columns.
Keep machine/contract/backend/accuracy exclusions. Test scalar and scan reporting,
missing distributions and all existing tests. Run installed adapters locally.

M3 uses two von Karman frozen-flow layers, a piston-free orthonormal modal DM,
a linear calibration stage and a dLux unmodulated pyramid stage. A command FIFO
implements delay. Integrator, linear predictor and fixed-iteration inverse fit
share observations. Validate PSF independently with HCIPy, fitting-error scaling,
delay response, bounded loop state and diffraction-based Strehl.

M4 predicts measurements causally from past estimates and queued commands. Compare
observed and predicted frames in noise units. Calibrate the threshold on clean
training seeds only; latch fallback on held-out anomalies or missing/late output.
Run every fault both with and without the switch; record failures to detect and
cases where fallback worsens error. No formal safety claim.

M5 changes array sampling, spectral quadrature and retained modes on a fixed truth
plant. Rebin predicted detector flux onto the same detector grid. Sweep flux over
four decades and use independent seeds. Use paired differences, uncertainty and
a predeclared equivalence margin, not a non-significant p-value, to say negligible.
M1 FFT costs are a component-cost reference, not full pyramid/controller latency.

M6 blends integrator and forward commands per mode. A fixed SNR/residual rule is
the baseline for a small learned network trained through short differentiable
rollouts. Training uses an explicitly labelled Gaussian shot-noise surrogate;
evaluation uses Poisson counts. Hold out turbulence/noise seeds and compare across
flux, wind and injected faults. Report paired uncertainty and every losing cell.

### HCIPy publication (opened after the documentation)

[Por et al., HCIPy](https://ehpor.github.io/assets/pdf/Por-2018-HCIPy.pdf).
The author-hosted paper describes Field/Grid abstractions and modular optical
propagation and AO components. It supplies an independent implementation for
our pupil/focal intensity check; using its propagator is not validation of our
entire atmosphere, pyramid, detector or controller.

### Adapter installation sources

[POPPY installation](https://poppy-optics.readthedocs.io/en/latest/installation.html),
[prysm source](https://github.com/brandondube/prysm),
[lentil source](https://github.com/andykee/lentil), and
[PROPER releases](https://sourceforge.net/projects/proper-library/files/) were opened.
POPPY and lentil use package releases; the Dragrace prysm adapter needs the source
API rather than the older PyPI release. PROPER uses its official source archive.
Installation success and case support are separate outcomes; unsupported adapters
must retain that status and are not assigned a timing.
