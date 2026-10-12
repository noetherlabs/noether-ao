# Controller-study execution safeguards

This is implementation infrastructure for the [predeclared study](controller_study_v1.md),
not a controller result or a finished scientific runner. The selection and record
ledger is implemented. The simulation adapter, actual provenance capture,
training telemetry, study CLI and physical smoke run remain to be built before
any reserved cohort is used. The frozen YAML, plan and original M0–M7 evidence
are unchanged.

## State and evidence contract

    create(config, fresh provenance)
           |
    save pilot/history/parameter artifacts in evidence/
           |
    bind_training(all predictor + learned candidates, evidence)
           |
    record_validation(exact case, outcome, raw + paired open-loop references)
           |          incomplete / duplicate / changed binding -> reject
           v
    freeze_selection(all declared cases, all rejected candidates retained)
           |          no eligible candidate -> withhold that controller
           v
    evaluate_chunk(adapter, limit)
           |          no selection / changed evidence -> reject before adapter
           |          existing valid records -> skip, including recorded failures
           |          adapter exception -> propagate; earlier records stay intact
           v
    append each outcome bound to selection hash; resume with same provenance

`noether_optics.controller_selection.StudyLedger` owns these transitions. Its
constructor verifies a previously created run against the current configuration
and caller-supplied provenance. `create` requires a new directory. Provenance has
three required fields: the full source commit, source-inventory SHA256 and
runtime-fingerprint SHA256. The adapter must calculate them from the actual
checkout and environment, retain those inventories under `evidence/`, and call
the constructor with freshly calculated values on every resumed process. A
syntactically valid digest is not proof that its caller measured the right inputs.
For now even the source commit must match on resume; an artifact-only commit does
not silently change the recorded source. Use the pinned source checkout.

Every fitted predictor and learned candidate has either immutable parameter
references or an explicit training failure. Shared training evidence is mandatory.
Training failure can only produce failed validation records. A failed record has
an error and no invented metrics; a completed record has raw references, an
open-loop reference, finite-status information, WFE, paired open-loop WFE,
saturation fraction and Strehl. Unrepresentable nonfinite metrics use JSON `null`
with `all_outputs_finite=false`; they cannot become eligible. Failures and
completed trials are counted separately.

Validation records bind the training-bundle hash. Their identities include stage,
full candidate parameters, photon regime, wind, fault, atmosphere seed and detector
seed. Selection requires exactly the declared identities; extra or missing cases
cannot substitute for full coverage. All candidates in a regime/seed must share
the same paired open-loop artifact and WFE. A single failed, nonfinite, saturated
or non-useful trial rejects that candidate. Eligible scores use an equal-weight
mean of trial-level WFE/open-WFE ratios. Exact ties follow declared candidate
order, irrespective of file or input ordering. Withheld controllers do not enter
evaluation, and the predeclared comparison-family denominator stays 120.

The frozen selection binds configuration, provenance, training and validation.
Before evaluation identities are released or any backend callback is invoked,
the ledger recomputes selection and rechecks all referenced file hashes. Each
saved evaluation record binds the selection hash. Resumption audits every existing
evaluation record before doing new work; it skips recorded failures as well as
successful records. All-withheld runs have zero planned evaluation cases and are
complete as a ledger operation, **not** a successful scientific comparison.

Records use strict canonical JSON and deterministic, full-length SHA256 case IDs.
Writes publish an already-flushed temporary file with an exclusive hard link;
existing evidence can only be retried byte-identically. A process lock prevents
cooperating writers from advancing stages at the same time and is held throughout
each evaluation chunk. A second writer fails promptly. A callback exception or
publication error propagates instead of being relabeled as a scientific outcome.
Previously published records survive; leftover `.pending-*` files never count as
finished trials. The code targets the existing macOS/Linux environments.

References must stay inside the run's `evidence/` directory. Missing, changed,
duplicate and escaping references are rejected. Files are hashed in bounded
blocks. Hashes and locks detect accidental drift and cooperating-process races;
they are not an access-control sandbox against a writer able to replace the whole
run. Full audits intentionally reread artifacts. The later adapter must budget
that I/O separately from compilation, training and trial execution costs; the
ledger itself collects no timings.

## Remaining adapter responsibilities

The ledger verifies record identity, selection rules and file integrity. It does
not inspect numerical arrays, derive metrics, verify model shapes, generate
atmospheres or certify that a controller used only permitted measurements. The
adapter must enforce those semantics and retain enough raw evidence to audit them:

- Build actual source/runtime inventories, check the pinned base configuration,
  require a clean source checkout, and check free disk before a scientific run.
- Save shared atmospheres, exact pilot observations and measured pseudo-open-loop
  estimates. Fit separate sequences, never bridge seeds or use phase truth as
  predictor telemetry. Preserve parameter arrays and training histories.
- Implement the predeclared six learned fits and fixed optimizer budgets. Record
  all training failures. Freeze training before validation.
- Derive summaries from retained per-frame arrays and full-output finite flags.
  Verify frame counts and matching open-loop seeds/disturbances; vibration must
  be included in both controlled and paired open-loop truth. Keep full and fault
  intervals. References alone do not establish these physical properties.
- Expose separate training, validation/selection and evaluation commands. Pass a
  lazy evaluation callback to `evaluate_chunk`; never generate evaluation arrays
  or preload its models before the gate. Write raw artifacts atomically and
  exclusively before publishing their ledger records. Preserve interrupted raw
  files and resume identical inputs; never silently overwrite or discard them.
- Distinguish an interrupted adapter from a scientifically failed trial. Return
  explicit failed records for actual execution failures; retain diagnostics.
  Unexpected implementation exceptions stop the chunk for review.
- Test on separate, labeled physical smoke cohorts. Commit the complete runner
  before using any reserved study seed. Then run bounded study chunks and produce
  the declared simultaneous paired intervals without excluding failed pairs.

## Engineering review and tests

The bounded change reuses `controller_protocol` for candidate enumeration and all
predeclared constraints. It adds one ledger class and pure selection code, leaving
the baseline simulation and generic artifact helpers untouched. Their existing
replace-in-place metadata writer does not provide this study's exclusive record
publication, so the ledger uses its own small publisher. There is no UI, new
dependency, timing acquisition or live study in this increment.

The tests in `tests/test_controller_selection.py` use explicitly synthetic bytes,
metrics and a separate 31001/32001/33001 cohort with initialization 34001. They
exercise the complete lifecycle without importing optical or random-number code:

| Path / failure | Required behavior |
| --- | --- |
| Exact ties; reversed input order | Same declared first candidate |
| Absolute vs relative WFE ranking | Select the declared relative score |
| Missing, duplicate, foreign or coerced case | No frozen selection |
| Any failed/nonfinite/saturated/non-useful trial | Reject candidate and retain reason |
| Changed paired open-loop reference | Reject inconsistent comparison |
| Missing training or selection | No later stage/backend call |
| Source, runtime, config, model or raw-data drift | No resumed evaluation |
| Training binding or saved selection changed | Reject before backend call |
| One/all controllers withheld | Exclude their evaluation; retain family size |
| Exact retry, chunk interruption, completed resume | Preserve bytes; no duplicated trials |
| Backend changes identity or frozen model | Reject publication |
| Disk failure during atomic publication | No partial completed record |
| Concurrent writer, escaping path or symlink | Reject without side effects |

Run the correctness checks with the repository's usual command. These fixtures
are not physical smoke results, training evidence, resource-cost measurements or
support for any controller claim.

```sh
CI=true JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q
```
