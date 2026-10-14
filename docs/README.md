# Documentation

Choose a route through the project:

| Goal | Start here |
| --- | --- |
| Understand the system and its assumptions | [Architecture](architecture.md) |
| Run experiments or rebuild saved results | [Reproduction](reproduction.md) |
| Inspect measured findings and raw evidence | [Results](results.md) |
| Check predeclared physical criteria | [Acceptance gates](acceptance.md) |
| Assess independent sensor validation | [Pyramid protocol](pyramid_validation.md) |
| Inspect the optical-gain observability diagnosis | [Exploratory audit](../results/optical_gain_observability_audit_v1/README.md) |
| Review the frozen optical-gain study | [Observability protocol](optical_gain_observability_v1.md) |
| Understand the next controller comparison | [Study protocol](controller_study_v1.md) |
| Work on durable execution, resource admission and recovery | [Execution contract](controller_study_execution.md) |
| Assess public-release readiness and unresolved decisions | [Release audit](release_readiness.md) |
| Follow deployment and release milestones | [Roadmap](../ROADMAP.md) |
| Make a change | [Contributing](../CONTRIBUTING.md) |

## Research record

[Research notes](research_notes.md) collect opened sources and experimental design.
[Milestone history](history.md) preserves the original measured commit identities;
[source inventory](source_manifest.json) records the historical source audit.

The [development plan](development/PLAN.md), [status](development/STATUS.md),
[decisions](development/DECISIONS.md) and [task queue](development/TODOS.md) form a
chronological engineering record. Dated entries describe what was known at each
increment; later entries and the roadmap describe the current position.

The [Dragrace patch](../contributions/dragrace.patch) can be reconstructed using
the reproduction guide. Its [draft descriptions](../contributions/drafts/) remain
unsubmitted upstream.
