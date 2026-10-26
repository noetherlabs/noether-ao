# Controller runner smoke

This retained run exercises the provenance-bound training, validation, selection
and evaluation commands on reduced, nonreserved inputs. It is implementation
evidence, not a controller comparison or timing result.

| Field | Value |
| --- | --- |
| Source commit | `37f6bfc6896741a0eea809f07d98113db4759d35` |
| Training pilots | 12 |
| Terminal fitted models | 2 |
| Validation records | 100 |
| Selected controllers | fixed, forward, integrator, learned, predictive |
| Withheld controllers | none |
| Held-out records acquired | 1 of 600 |
| Protected files | 488 |

## Gates

| Gate | Result |
| --- | --- |
| `nonreserved_seed_streams` | pass |
| `resource_preflight_admitted` | pass |
| `training_complete` | pass |
| `validation_complete` | pass |
| `selection_frozen` | pass |
| `selection_reentry_exact` | pass |
| `held_out_evaluation_entered` | pass |
| `durable_stage_counts` | pass |
| `source_provenance_bound` | pass |

The held-out stage intentionally stops after one durable record. The smoke does
not estimate controller performance, complete a statistical cohort, compare
resource costs or alter the frozen `controller_study_v1` protocol.
