# Add an AO MFT scan and append empirical tail latency

The existing pupil scan starts above some small-grid AO regimes, and a median
alone hides deadline-tail behavior. This change adds `mft_ao_scan` with unchanged
MFT physics and appends p99/max from `timing.device_compute`. Missing or invalid
sample arrays display missing tails. No measured column is replaced. No adapter
verification label is changed.

Validation: 140 tests passed; the two CPU-affinity tests are skipped on macOS.
The first invocation lacked the required JAX double-precision setting; the
successful invocation used `JAX_ENABLE_X64=1` before import. Logs are retained.
The contribution is local commit `d194dab` on `codex/ao-tail-latency`.

The local scan has 1000 evaluations per size after 10 warmups. Seven adapters
passed every size's existing accuracy gate. PROPER installed but returned
unsupported because it has no matching matrix-DFT path. Timings are from this
session on a workstation, outside CI; maxima are observations, not bounds.

Selected results follow. [All sizes and raw-result locations](../../results/m2/table.md)
are retained with [the original report](../../results/m2/report.txt).

<!-- BEGIN DRAGRACE -->

| Adapter | Pupil grid | Median ms | p99 ms | Observed max ms |
| --- | --- | --- | --- | --- |
| abcdlux | 128 | 0.349646 | 0.557959 | 1.15008 |
| abcdlux | 256 | 0.997709 | 2.46371 | 4.35896 |
| dlux | 128 | 0.857146 | 2.473 | 4.81971 |
| dlux | 256 | 2.11644 | 4.37263 | 5.38879 |
| hcipy | 128 | 0.282063 | 0.460667 | 1.32004 |
| hcipy | 256 | 0.721917 | 1.06354 | 1.39796 |
| lentil | 128 | 1.02169 | 1.28637 | 1.5915 |
| lentil | 256 | 2.97146 | 9.09121 | 21.2918 |
| numpy_baseline | 128 | 0.270521 | 0.371917 | 0.437125 |
| numpy_baseline | 256 | 0.717249 | 1.20675 | 1.4605 |
| poppy | 128 | 2.20371 | 3.40679 | 5.01679 |
| poppy | 256 | 3.5589 | 12.1754 | 34.9458 |
| prysm | 128 | 0.248708 | 0.349333 | 0.689958 |
| prysm | 256 | 0.703875 | 0.986167 | 1.42683 |

<!-- END DRAGRACE -->

Machine: Apple M1; macOS-14.6-arm64-arm-64bit; 8 available
logical cores; Python 3.11.15; JAX 0.10.2.
No GPU was exposed. CPU affinity cannot be enforced on this OS. NumPy and SciPy
build metadata identify Accelerate, while threadpoolctl cannot inspect its thread
pool; actual CPU/wall ratios and upstream nonselectable-axis warnings remain in
the raw results. The upstream machine ID is `sha256:d4ec86edc24f9711`; the extended
runtime fingerprint is `95faefff5c4ffe15` in [machine.json](../../results/m2/machine.json).

These are idiomatic library API costs on this configuration. The NumPy baseline
is a kernel reference. They do not measure a full AO loop or compare FFT and MFT
as interchangeable operations. The report uses the existing index percentile
convention; the independent Noether harness uses interpolated percentiles.

Human review is required before publishing this draft. No public action was taken.
