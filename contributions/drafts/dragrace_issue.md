# Proposal: small-pupil AO MFT scan and stored tail-latency columns

Would you welcome a small contribution that adds `mft_ao_scan` and appends p99
and observed maximum to the timing report? The scan keeps the existing MFT
physics and focal sampling, while covering the smaller pupil grids used in AO
component experiments. Tail columns are calculated from stored samples; the
existing columns, correctness gates and measurement path stay intact.

A local branch and passing tests are ready for review. The scan ran on the local
CPU for every installed adapter. PROPER correctly reports that this MFT case is
unsupported. This is a proposal draft; no issue or pull request has been opened.
