# Optical-gain observability runner smoke

This retained run exercises the provenance-bound physical training, threshold,
validation and decision paths on reduced, nonreserved inputs. The observed values
below are implementation-smoke outcomes, not detector validation or a safety claim.

| Field | Value |
| --- | --- |
| Source commit | `a997753109061484ca7bd768f4484e923f88a327` |
| Training records | 2 |
| Validation records | 4 |
| Validation artifact status | `evaluation_withheld` |
| Protected files | 29 |

## Gates

| Gate | Result |
| --- | --- |
| `nonreserved_seed_streams` | pass |
| `resource_preflight_admitted` | pass |
| `training_complete` | pass |
| `threshold_frozen` | pass |
| `threshold_artifact_bound` | pass |
| `threshold_reentry_exact` | pass |
| `validation_complete` | pass |
| `validation_decision_frozen` | pass |
| `validation_artifact_bound` | pass |
| `validation_reentry_exact` | pass |
| `declared_metrics_reported` | pass |
| `durable_stage_counts` | pass |
| `source_provenance_bound` | pass |
| `evaluation_not_acquired` | pass |

## Retained smoke observations

| Quantity | Value |
| --- | ---: |
| `clean.alarm_eligible_frames` | 20 |
| `clean.alarm_frame_fraction` | 0.05 |
| `clean.alarm_frames` | 1 |
| `clean.alarm_runs` | 1 |
| `clean.raw_crossing_fraction` | 0.235294 |
| `clean.raw_crossing_frames` | 34 |
| `clean.raw_crossings` | 8 |
| `gain_035.boundary_straddling_alarm_runs` | 1 |
| `gain_035.detected_only_delay_frames` | [] |
| `gain_035.detected_runs` | 0 |
| `gain_035.missed_runs` | 2 |
| `gain_035.pre_onset_alarm_runs` | 0 |
| `paired_controller_error.pairs` | [{"atmosphere_seed": 720001, "clean_saturation_fraction": 0.0, "detector_seed": 820001, "gain_saturation_fraction": 0.0, "regime_id": "regime-00", "strehl_delta": -0.003758077840998353, "wfe_nm_delta": 0.6758110716583161}, {"atmosphere_seed": 720002, "clean_saturation_fraction": 0.0, "detector_seed": 820002, "gain_saturation_fraction": 0.0, "regime_id": "regime-00", "strehl_delta": -6.16811153697494e-05, "wfe_nm_delta": 0.007620048069583163}] |
| `integrity.all_completed_outputs_finite` | true |
| `integrity.fallback_frames` | 0 |
| `integrity.saturated_frames` | 0 |
| `integrity.saturation_fraction` | 0 |
| `integrity.unavailable_frames` | 0 |

No evaluation trial was acquired. The smoke records no timing, does not estimate a
false-alarm or detection rate, and does not alter or authorize the frozen
`optical_gain_observability_v1` cohort.
