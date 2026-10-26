# Independent WFS calibration comparison

All required gates passed: **True**.

Four independently propagated/calibrated optical paths; native-model transfer is diagnostic, not an equivalence claim. See the [frozen protocol](../../docs/wfs_calibration.md).

![Calibration accuracy and transfer](comparison.png)

| Pupil | Backend | Power | Condition | Max error at 0.001 rad | Gates |
| --- | --- | --- | --- | --- | --- |
| 32 | noether | 1 | 1.50445 | 0.000252983 | True |
| 32 | hcipy_matched | 1 | 1.50445 | 0.000252981 | True |
| 32 | hcipy_native | 0.99632728 | 1.44439 | 0.000348193 | True |
| 32 | specula_native | 0.99632728 | 1.44439 | 0.000347262 | True |
| 64 | noether | 1 | 1.46679 | 0.000253179 | True |
| 64 | hcipy_matched | 1 | 1.46679 | 0.000253177 | True |
| 64 | hcipy_native | 0.99797348 | 1.41315 | 0.000347391 | True |
| 64 | specula_native | 0.99797348 | 1.41315 | 0.000346494 | True |

Errors are relative pupil RMS, not percentages. All per-case errors, singular values, gate verdicts and noise covariances are retained in [summary.json](summary.json).

## Calibration transfer

Each row applies the source reconstructor to target images. Both subtracting the target zero image (matrix only) and using the source zero image (full) are shown. No fitted alignment or scaling is applied.

| Pupil | Source → target | Interaction relative L2 | Small-signal matrix-only error | Small-signal full error |
| --- | --- | --- | --- | --- |
| 32 | noether → hcipy_matched | 5.98811e-09 | 0.000252983 | 0.000252983 |
| 32 | noether → hcipy_native | 1.55648 | 2.00775 | 69.4654 |
| 32 | noether → specula_native | 2.04955 | 2.07154 | 71.1705 |
| 32 | hcipy_matched → noether | 5.98811e-09 | 0.000252981 | 0.000252981 |
| 32 | hcipy_matched → hcipy_native | 1.55648 | 2.00775 | 69.4654 |
| 32 | hcipy_matched → specula_native | 2.04955 | 2.07154 | 71.1705 |
| 32 | hcipy_native → noether | 1.47538 | 1.90211 | 66.2426 |
| 32 | hcipy_native → hcipy_matched | 1.47538 | 1.90211 | 66.2426 |
| 32 | hcipy_native → specula_native | 1.29605 | 1.8707 | 1.8707 |
| 32 | specula_native → noether | 1.94275 | 1.95393 | 67.8488 |
| 32 | specula_native → hcipy_matched | 1.94275 | 1.95393 | 67.8488 |
| 32 | specula_native → hcipy_native | 1.29605 | 1.87079 | 1.87079 |
| 64 | noether → hcipy_matched | 5.8582e-09 | 0.000253179 | 0.000253179 |
| 64 | noether → hcipy_native | 1.55768 | 2.01117 | 72.0294 |
| 64 | noether → specula_native | 2.05192 | 2.07197 | 73.7352 |
| 64 | hcipy_matched → noether | 5.8582e-09 | 0.000253177 | 0.000253177 |
| 64 | hcipy_matched → hcipy_native | 1.55768 | 2.01117 | 72.0294 |
| 64 | hcipy_matched → specula_native | 2.05192 | 2.07197 | 73.7352 |
| 64 | hcipy_native → noether | 1.47327 | 1.89969 | 68.87 |
| 64 | hcipy_native → hcipy_matched | 1.47327 | 1.89969 | 68.87 |
| 64 | hcipy_native → specula_native | 1.29645 | 1.87188 | 1.87188 |
| 64 | specula_native → noether | 1.94072 | 1.95161 | 70.4749 |
| 64 | specula_native → hcipy_matched | 1.94072 | 1.95161 | 70.4749 |
| 64 | specula_native → hcipy_native | 1.29645 | 1.87196 | 1.87196 |

## Nonlinearity and noise

| Pupil | Backend | Phase RMS (rad) | Max relative reconstruction error |
| --- | --- | --- | --- |
| 32 | noether | 0.001 | 0.000252983 |
| 32 | noether | 0.05 | 0.0137315 |
| 32 | noether | 0.25 | 0.0900697 |
| 32 | hcipy_matched | 0.001 | 0.000252981 |
| 32 | hcipy_matched | 0.05 | 0.0137315 |
| 32 | hcipy_matched | 0.25 | 0.0900697 |
| 32 | hcipy_native | 0.001 | 0.000348193 |
| 32 | hcipy_native | 0.05 | 0.0185531 |
| 32 | hcipy_native | 0.25 | 0.11382 |
| 32 | specula_native | 0.001 | 0.000347262 |
| 32 | specula_native | 0.05 | 0.0162235 |
| 32 | specula_native | 0.25 | 0.0976274 |
| 64 | noether | 0.001 | 0.000253179 |
| 64 | noether | 0.05 | 0.0137035 |
| 64 | noether | 0.25 | 0.0890817 |
| 64 | hcipy_matched | 0.001 | 0.000253177 |
| 64 | hcipy_matched | 0.05 | 0.0137035 |
| 64 | hcipy_matched | 0.25 | 0.0890817 |
| 64 | hcipy_native | 0.001 | 0.000347391 |
| 64 | hcipy_native | 0.05 | 0.0184713 |
| 64 | hcipy_native | 0.25 | 0.112629 |
| 64 | specula_native | 0.001 | 0.000346494 |
| 64 | specula_native | 0.05 | 0.0162262 |
| 64 | specula_native | 0.25 | 0.096644 |

Noise ratios compare sampled to analytic modal variances at the flat wavefront (512 independent detector realizations, 100000 incident photons, 1 electron read noise). Finite-sample deviations from one are diagnostic; this is not instrument noise.

| Pupil | Backend | Minimum variance ratio | Maximum variance ratio |
| --- | --- | --- | --- |
| 32 | noether | 0.876891 | 1.16699 |
| 32 | hcipy_matched | 0.876891 | 1.16699 |
| 32 | hcipy_native | 0.968847 | 1.1086 |
| 32 | specula_native | 0.968847 | 1.1086 |
| 64 | noether | 0.90761 | 1.02667 |
| 64 | hcipy_matched | 0.90761 | 1.02667 |
| 64 | hcipy_native | 0.964424 | 1.07393 |
| 64 | specula_native | 0.964424 | 1.07393 |

## Scope

HCIPy matched uses matrix Fourier integrals and a physical thin surface; Noether uses dLux/JAX FFTs and automatic derivatives. HCIPy native and SPECULA native retain their focal stops, centering and propagation conventions. Native disagreement is not evidence of a library bug. No controller, on-sky, modulation, runtime or hardware-validity claim follows from this run.

[Configuration](config.json) · [Noether provenance](provenance.json) · [File hashes](manifest.json). Per-backend geometry, package versions and library source hashes accompany each raw archive in `raw/`. All 48 held-out images per backend and grid, both push-pull step sizes and modal noise draws are retained.
