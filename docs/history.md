# Milestone commit history

Main contains the complete work. The owner's latest instruction is commits only:
future improvements are validated, committed and pushed directly to main without
opening PRs. Original measured commit hashes remain reachable for reproduction.

| Stage | Scope | Preserved commit |
| --- | --- | --- |
| Planning | Initial experiment plan and decision log | 999bd9c |
| M0 and harness | Research, pinned core, synchronized latency harness | 2a76d9e |
| M1 | Raw CPU sweep and interruption evidence | ce469cb |
| M2 | Dragrace adapters, portable patch, results and upstream drafts | 5ba5266 |
| M3 | AO implementation and physical validation; prepares M4–M6 drivers | 5fc820d |
| M4 | Fault coverage and limits of fallback | 9d65625 |
| M5 | Fidelity study and source-verified disk recovery | 810b14e |
| M6 | Mixture results, negative findings and claim report | af09c43 |
| M7 | Fresh-checkout reproduction evidence | e11be4c |

See the [commit history](https://github.com/noetherlabs/noether-ao/commits/main).
The setup commits add portable instructions, correctness-only CI and the autonomous
work queue. Historical validation records do not imply GitHub Actions ran on each
earlier commit. Timing data came from local isolated runs; CI never remeasures it.

Eight milestone PRs were initially opened under the earlier delivery instruction.
Their creation record is retained in `results/publication/pr-series.json`; the
owner subsequently requested closing them and publishing commits only. No upstream
Dragrace PR or personal message was sent.
