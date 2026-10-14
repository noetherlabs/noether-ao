# Roadmap

The objective is a reusable path from optical simulation to a qualified control
change: define the plant, compare controllers, retain the evidence, then validate
an instrument integration. Deployment is an engineering milestone with its own
acceptance evidence. It is not implied by a successful simulation.

## 1. Complete the reproducible controller comparison

**In progress.** The optical plant, training components, durable evidence storage,
selection barrier, trial adapters, training coordination, bounded stage commands and
whole-run storage preflight are implemented. The frozen plan retains 42.167 GB of
logical numeric payload and requires a conservative 105.033 GB free-space envelope
with no compression credit. The complete runner has passed nine gates on separate
smoke seeds with protected evidence. The remaining pre-acquisition work is to
provision a filesystem that passes the full-study capacity gate.

Deliverable: one reproducible comparison under the [frozen study protocol](docs/controller_study_v1.md),
with all eligible controllers, paired uncertainty, failed candidates and resource
costs retained. A null or negative advantage remains a valid outcome. Acquisition
on reserved cohorts starts only after both the completed smoke and the storage gate
are satisfied; the latter remains blocked on the current development volume.

## 2. Make mismatch detection useful for control

**In progress.** The baseline misses optical-gain drift and shows harmful vibration
fallback. A read-only [legacy-evidence audit](results/optical_gain_observability_audit_v1/README.md)
shows that the upper-tail score moves downward under the simplified gain fault. The
[v1 observability protocol](docs/optical_gain_observability_v1.md) now freezes a
training-calibrated lower-tail candidate, eight-window persistence, fresh disjoint
cohorts and validation-only release gates. The plan is unexecuted: no fresh threshold
or outcome exists. Next implement immutable calibration/release artifacts and pass a
separately seeded runner smoke before accessing a declared cohort.

Deliverable: a report of clean false alarms, missed faults, detection delay and paired
fault-versus-clean controller error. Any later recovery experiment must also report
the consequence of its control action. See the [research queue](docs/development/TODOS.md).

## 3. Qualify an instrument interface

**Planned; not implemented.** Select a concrete instrument or laboratory fixture
before specifying hardware adapters. Define coordinate conventions, units,
calibration provenance, detector timestamps, actuator limits, command age and
missing-frame behavior against that interface.

Deliverable: recorded-telemetry replay and hardware-in-the-loop evidence covering
calibration drift, delayed/missing data and command bounds. Measure the complete
sensor-to-command path on the target machine, including transfer and scheduling.
The deadline, acceptable tail behavior and recovery criteria must come from the
target instrument; the current CPU propagation benchmark cannot supply them.

## 4. Demonstrate operational value

**Planned.** First qualify the interface in a controlled setting. Any subsequent
instrument trial needs its own authorization, operating envelope and acceptance
protocol.

Deliverable: a comparison against the instrument's established controller, with
wavefront/image-quality metrics, availability, calibration effort, compute cost
and intervention requirements. Those measurements determine practical value;
this repository currently makes no deployed-performance or customer claim.

## Public release

The repository remains private and unlicensed. The
[release-readiness audit](docs/release_readiness.md) completed the credential,
retained-array, direct-license and package-construction reviews against commit
`fa8a8b8`. It found no credential or unreadable-array blocker, retained the upstream
Diffraction-Dragrace notice, and reproduced the same wheel hash twice. Preparing the
source does not make it an open-source release.

Release now requires explicit owner decisions and their implementation:

- Select and approve a license for project-owned code and artifacts.
- Choose whether a public release preserves, reviews and squashes, or separates the
  existing private Git history.
- Accept the 35 retained provenance files containing workstation paths or publish
  labeled derived copies without changing the protected originals.
- Align published dependency metadata with the Git/URL sources used by the tested
  environment, then verify a standards-only installation.
- Name monitored issue and security-reporting routes and review the three unsent
  Diffraction-Dragrace drafts.
- Repeat the release audit, publish a tagged version and authorize visibility only
  after those gates pass.

The [development plan](docs/development/PLAN.md) records implementation scope;
the [status](docs/development/STATUS.md) records checks actually completed.
