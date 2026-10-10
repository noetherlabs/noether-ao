# Status

M0: pass. Opened-source notes, design and pinned environment are recorded.
M1: pass on the available CPU configuration. GPU and enforced single-core affinity skipped.
All 144 engine/path/shape rows contain 10000 saved evaluations, for 1440000 measured calls. Maximum complex-field relative L2 error: 0.0.
Raw results, machine fingerprint, interruption record, tests, table and plot: `results/m1/`.
C1: configuration-dependent support; no full control-loop deadline guarantee.

M2: pass. Clean local branch d194dab, 140 tests passed and two platform affinity skips. Seven adapters passed all six scan sizes; PROPER installed but correctly reports unsupported MFT. Both drafts exist in outbox. Raw results and fingerprints: results/m2.
M3: pass. Independent HCIPy PSFs, fitting scaling, delay growth and nominal stability all passed. Pyramid predictor reduction: 26.372725760970482 percent in mean WFE; this is not an exact replication of an unspecified earlier setup. Raw data: results/m3.
M4: pass for experimental coverage; C4 is partial and does not establish safety. All 80 fault/switch/seed trials are finite. Optical-gain drift was missed, vibration fallback worsened WFE, and one of five clean runs triggered a false alarm (one of 600 evaluated frames). Raw results, score/availability ROC and paired differences: results/m4.
M5: pass for coverage. All 75 trials are finite and all cells have five held-out seeds. No low-flux equivalence threshold passed the predeclared useful-equivalence criterion. Lower fidelities often improved WFE under the fixed estimator; model resolution, mode count and estimator conditioning are confounded. The last four trials were recovered after ENOSPC using the verified original source commit and identical fingerprint; the first 71 raw trials were unchanged. M6: in progress. M7: pending.
All work is local. No publication, messages or collaboration claims.
