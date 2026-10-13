# Roadmap

The objective is a reusable path from optical simulation to a qualified control
change: define the plant, compare controllers, retain the evidence, then validate
an instrument integration. Deployment is an engineering milestone with its own
acceptance evidence. It is not implied by a successful simulation.

## 1. Complete the reproducible controller comparison

**In progress.** The optical plant, training components, durable evidence storage,
selection barrier and trial adapters are implemented. The remaining work connects
training coordination and stage commands, establishes a full-run resource/disk
budget, and verifies the complete runner on separate smoke seeds.

Deliverable: one reproducible comparison under the [frozen study protocol](docs/controller_study_v1.md),
with all eligible controllers, paired uncertainty, failed candidates and resource
costs retained. A null or negative advantage remains a valid outcome. Acquisition
on reserved cohorts starts only after the complete runner passes its separate smoke.

## 2. Make mismatch detection useful for control

**Queued.** The baseline misses optical-gain drift and shows harmful vibration
fallback. Investigate observability and calibration on fresh training/validation
cohorts; freeze thresholds before held-out fault trials.

Deliverable: a report of clean false alarms, missed faults, detection delay and the
paired wavefront effect of recovery. A detector result must include the consequence
of its control action. See the [research queue](docs/development/TODOS.md).

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

The repository remains private and unlicensed. Preparing its source and
presentation does not make it an open-source release. Release work includes:

- Select and approve a license for project-owned code and artifacts; inventory
  third-party licenses and retain required notices for dependencies and patches.
- Review the complete history and retained artifacts for redistribution rights,
  private information and credentials before changing visibility.
- Verify the documented installation and examples from a clean checkout, publish
  a tagged version, and state supported environments and known limitations.
- Publish contribution, issue and security-reporting routes that are actually
  monitored, then authorize the visibility change.

The [development plan](docs/development/PLAN.md) records implementation scope;
the [status](docs/development/STATUS.md) records checks actually completed.
