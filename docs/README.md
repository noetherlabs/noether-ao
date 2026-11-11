# Documentation

## Using the software

- [Architecture and model assumptions](architecture.md)
- [Installation, experiments and report reconstruction](reproduction.md)
- [Prediction Boundary data download and integrity](release_integrity.md)
- [Numerical findings and limitations](results.md)
- [Development and tests](../CONTRIBUTING.md)

## Optical models and calibration

- [Physical acceptance criteria](acceptance.md)
- [Independent pyramid validation](pyramid_validation.md)
- [HCIPy and SPECULA calibration comparisons](wfs_calibration.md)
- [Closed-loop response identification](identification.md)
- [Scheduled controller handover](scheduled_fallback.md)

## Study protocols

- [Controller comparison](controller_study_v1.md)
- [Execution and evidence contracts](controller_study_execution.md)
- [Optical-gain observability](optical_gain_observability_v1.md)
- [Shift prediction and range forecasting](shift_prediction.md)

The controller study completed [600 held-out evaluation trials](../results/controller_study_v1/README.md)
after frozen training and validation. Optical-gain observability
completed its [validation study](../results/optical_gain_observability_v1/README.md)
and withheld evaluation after failing its frozen gate. Reduced checks and
exploratory results are labelled separately from complete studies.

[References](research_notes.md) describe the literature and numerical conventions.
