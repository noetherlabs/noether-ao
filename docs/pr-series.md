# Progressive PR review order

The milestone branches preserve the original local commit hashes and measurements.
These PRs expose the completed work in reviewable stages; they do not imply prior
GitHub review. Main begins at planning commit `999bd9c` while the PRs remain open.
The latest complete branch is `codex/repository-setup`.

| Order | Head branch | Base branch | Scope | Preserved tip |
| --- | --- | --- | --- | --- |
| [1](https://github.com/noetherlabs/noether-ao/pull/1) | codex/01-latency-harness | main | Research, pinned core, synchronized harness | 2a76d9e |
| [2](https://github.com/noetherlabs/noether-ao/pull/2) | codex/02-cpu-latency-results | codex/01-latency-harness | M1 raw CPU sweep and interruption evidence | ce469cb |
| [3](https://github.com/noetherlabs/noether-ao/pull/3) | codex/03-dragrace-contribution | codex/02-cpu-latency-results | M2 adapters, portable patch, results and upstream drafts | 5ba5266 |
| [4](https://github.com/noetherlabs/noether-ao/pull/4) | codex/04-ao-physics | codex/03-dragrace-contribution | AO implementation and M3 validation; prepares M4–M6 drivers | 5fc820d |
| [5](https://github.com/noetherlabs/noether-ao/pull/5) | codex/05-fault-assurance | codex/04-ao-physics | M4 fault coverage and limits of fallback | 9d65625 |
| [6](https://github.com/noetherlabs/noether-ao/pull/6) | codex/06-fidelity-study | codex/05-fault-assurance | M5 fidelity study and source-verified disk recovery | 810b14e |
| [7](https://github.com/noetherlabs/noether-ao/pull/7) | codex/07-learned-mixtures | codex/06-fidelity-study | M6 mixture results, negative findings and claim report | af09c43 |
| [8](https://github.com/noetherlabs/noether-ao/pull/8) | codex/08-reproduction | codex/07-learned-mixtures | Fresh-checkout M7 reproduction evidence | e11be4c |
| 9 | codex/repository-setup | codex/08-reproduction | Private repository setup, CI and autonomous work queue | New work |

Review the [pull requests](https://github.com/noetherlabs/noether-ao/pulls) in this
order. Merge with merge commits to preserve provenance, then retarget the next PR
to main. Keep prerequisite branches while dependent PRs remain open. New work
should branch from the latest complete head until this series is integrated.

Historical milestone PRs contain their original validation records. The setup PR
adds CI for the current full tree; a passing current check is not a claim that
GitHub Actions ran on each historical milestone. Timing data came from local
isolated runs and will never be remeasured by CI.
