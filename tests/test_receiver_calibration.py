"""Commissioning cost, measured-data boundaries and independent-root accounting."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
from math import pi, tan

import numpy as np
import pytest

import noether_optics.receiver_calibration as receiver_calibration
from noether_optics.receiver_calibration import (
    CalibrationWindows,
    fit_commissioning_response,
    paired_coupling_summary,
    summarize_root_utilities,
)


def tiny_windows():
    return CalibrationWindows(commissioning_frames=10, probe_start=2, probe_stop=6,
                              service_frames=4, delay_frames=2)


def measured_case(windows=None):
    """Four independent probe rows and a deterministic, nonsymmetric response."""
    windows = tiny_windows() if windows is None else windows
    signs = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
    probe = np.zeros((windows.commissioning_frames, 2), dtype=np.float64)
    duration = windows.probe_stop - windows.probe_start
    probe[windows.probe_start:windows.probe_stop] = .05 * np.tile(signs, (duration // 4, 1))
    issued = .5 * probe
    applied = np.zeros_like(issued)
    applied[windows.delay_frames:] = issued[:-windows.delay_frames]
    reference = np.array([5., -3.])
    response = np.array([[.6, .2], [-.1, .9]])
    measurement = reference - applied @ response + np.array([.13, -.21])
    # Finite unrelated observations outside the declared fit window expose a
    # mistaken requested-probe window or whole-record fit, without poisoning it.
    outside = np.ones(windows.commissioning_frames, dtype=bool)
    outside[windows.fit_start:windows.fit_stop] = False
    measurement[outside] += np.arange(windows.commissioning_frames)[outside, None] * [2., -3.]
    return dict(measurement=measurement, issued=issued, applied=applied,
                probe_requested=probe, nominal_reconstructor=np.eye(2),
                nominal_interaction=np.eye(2), detector_reference=reference,
                windows=windows), response


def coupling_case():
    nominal_commissioning = np.full(10, .5)
    probed_commissioning = nominal_commissioning.copy()
    probed_commissioning[[2, 3, 6, 7, 8]] = .25
    nominal_service = np.full(4, .5)
    corrected_service = np.array([0., .5, 1., 1.])
    return nominal_commissioning, probed_commissioning, nominal_service, corrected_service


def record(root, condition, value):
    return {"root_id": root, "condition_id": condition, "charged_mean_delta": value}


def test_default_windows_include_delayed_last_probe_and_all_service_frames():
    windows = CalibrationWindows()
    assert (windows.fit_start, windows.fit_stop) == (130, 258)
    assert windows.charged_frames == 896
    assert windows.total_frames == 1024
    with pytest.raises(FrozenInstanceError):
        windows.probe_stop = 255


@pytest.mark.parametrize("change", [
    {"probe_start": -1}, {"probe_start": True}, {"probe_stop": 2},
    {"probe_stop": 9}, {"commissioning_frames": 0}, {"service_frames": 0},
    {"delay_frames": 0}, {"delay_frames": 1.5}, {"delay_frames": True},
])
def test_invalid_or_unobserved_recording_windows_refuse(change):
    values = dict(commissioning_frames=10, probe_start=2, probe_stop=6,
                  service_frames=4, delay_frames=2)
    values.update(change)
    with pytest.raises(ValueError):
        CalibrationWindows(**values)


def test_fit_uses_exact_delayed_rows_and_original_detector_reference(monkeypatch):
    inputs, response = measured_case()
    before = {key: value.copy() for key, value in inputs.items() if isinstance(value, np.ndarray)}
    called = []
    existing_fit = receiver_calibration.fit_response

    def observe_fit(residual, applied, instrument, **options):
        called.append((residual.copy(), applied.copy(), instrument.copy(), options))
        return existing_fit(residual, applied, instrument, **options)

    monkeypatch.setattr(receiver_calibration, "fit_response", observe_fit)
    fitted = fit_commissioning_response(**inputs)
    assert fitted["status"] == "ready"
    assert fitted["fit"]["method"] == "iv"
    assert fitted["fit"]["diagnostics"]["frames"] == 4
    assert fitted["fit_rows"] == (4, 8)
    assert len(called) == 1
    residual, applied, instrument, options = called[0]
    np.testing.assert_array_equal(residual, inputs["measurement"][4:8] - inputs["detector_reference"])
    np.testing.assert_array_equal(applied, inputs["applied"][4:8])
    np.testing.assert_array_equal(instrument, inputs["probe_requested"][2:6])
    assert not np.array_equal(instrument, applied)
    assert options["method"] == "iv"
    np.testing.assert_allclose(fitted["response"], response, rtol=0, atol=2e-13)
    np.testing.assert_array_equal(fitted["detector_reference"], [5., -3.])
    assert not np.array_equal(fitted["detector_reference"], fitted["fit"]["intercept"])
    for key, value in before.items():
        np.testing.assert_array_equal(inputs[key], value)


def test_default_fit_keeps_all_128_delayed_probe_rows():
    inputs, response = measured_case(CalibrationWindows())
    fitted = fit_commissioning_response(**inputs)
    assert fitted["status"] == "ready"
    assert fitted["fit"]["diagnostics"]["frames"] == 128
    assert fitted["fit_rows"] == (130, 258)
    np.testing.assert_allclose(fitted["response"], response, rtol=0, atol=2e-13)


def test_finite_observations_outside_the_fit_window_cannot_change_the_fit():
    inputs, _ = measured_case()
    fitted = fit_commissioning_response(**inputs)
    altered = deepcopy(inputs)
    altered["measurement"][:4] -= 1000
    altered["measurement"][8:] += 2000
    changed = fit_commissioning_response(**altered)
    assert changed["status"] == fitted["status"]
    np.testing.assert_array_equal(changed["response"], fitted["response"])
    np.testing.assert_array_equal(changed["reconstructor"], fitted["reconstructor"])


def test_decision_snapshot_has_no_input_alias_and_read_only_matrices():
    inputs, _ = measured_case()
    fitted = fit_commissioning_response(**inputs)
    frozen = {key: fitted[key].copy() for key in ("response", "reconstructor", "detector_reference")}
    for value in inputs.values():
        if isinstance(value, np.ndarray):
            value[:] = 999
    for key, original in frozen.items():
        np.testing.assert_array_equal(fitted[key], original)
        assert not fitted[key].flags.writeable
        with pytest.raises(ValueError):
            fitted[key].flat[0] = 0


@pytest.mark.parametrize("forbidden", ["phase_truth", "true_transform", "service", "coupling"])
def test_fit_has_no_evaluator_or_future_service_argument(forbidden):
    inputs, _ = measured_case()
    inputs[forbidden] = np.full((100, 2), np.nan)
    with pytest.raises(TypeError):
        fit_commissioning_response(**inputs)


@pytest.mark.parametrize("damage", ["tail_nan", "tail_fifo", "early_probe", "late_probe",
                                     "short_measurement", "missing_measurement"])
def test_complete_commissioning_validation_precedes_window_selection(damage):
    inputs, _ = measured_case()
    if damage == "tail_nan":
        inputs["measurement"][9, 0] = np.nan
    elif damage == "tail_fifo":
        inputs["applied"][9, 0] = .1
    elif damage == "early_probe":
        inputs["probe_requested"][0, 0] = .1
    elif damage == "late_probe":
        inputs["probe_requested"][8, 0] = .1
    elif damage == "short_measurement":
        inputs["measurement"] = inputs["measurement"][:-1]
    else:
        del inputs["measurement"]
    with pytest.raises((ValueError, TypeError)):
        fit_commissioning_response(**inputs)


def test_withheld_instruments_freeze_the_nominal_reconstructor():
    inputs, _ = measured_case()
    inputs["probe_requested"][:] = 0
    fitted = fit_commissioning_response(**inputs)
    assert fitted["status"] == "withheld"
    assert fitted["response"] is None
    assert fitted["fit"]["reason"] == "rank_deficient_instruments"
    np.testing.assert_array_equal(fitted["reconstructor"], inputs["nominal_reconstructor"])
    assert not fitted["reconstructor"].flags.writeable
    inputs["nominal_reconstructor"][:] = 0
    np.testing.assert_array_equal(fitted["reconstructor"], np.eye(2))


def test_withheld_restoration_keeps_ready_fit_and_nominal_service_calibration():
    inputs, response = measured_case()
    fitted = fit_commissioning_response(**inputs, maximum_reconstructor_gain=.5)
    assert fitted["status"] == "withheld"
    assert fitted["fit"]["status"] == "ready"
    assert fitted["restoration"]["reason"] == "reconstructor_gain_budget"
    np.testing.assert_allclose(fitted["response"], response, rtol=0, atol=2e-13)
    np.testing.assert_array_equal(fitted["reconstructor"], inputs["nominal_reconstructor"])
    np.testing.assert_array_equal(fitted["detector_reference"], inputs["detector_reference"])


@pytest.mark.parametrize("field", ["measurement", "issued", "probe_requested", "detector_reference"])
@pytest.mark.parametrize("dtype", [np.float32, np.int64])
def test_fit_cannot_silently_change_recording_precision(field, dtype):
    inputs, _ = measured_case()
    inputs[field] = inputs[field].astype(dtype)
    with pytest.raises(ValueError, match="float64"):
        fit_commissioning_response(**inputs)


@pytest.mark.parametrize("options", [
    {"maximum_reconstructor_gain": 0}, {"maximum_reconstructor_gain": True},
    {"maximum_reconstructor_gain": np.inf}, {"minimum_relevance": -1},
    {"minimum_relevance": 1}, {"minimum_relevance": np.nan},
    {"minimum_relevance": True}, {"rcond": 0}, {"rcond": 1}, {"rcond": True},
])
def test_invalid_fit_limits_refuse_even_when_instruments_would_be_withheld(options):
    inputs, _ = measured_case()
    inputs["probe_requested"][:] = 0
    with pytest.raises(ValueError):
        fit_commissioning_response(**inputs, **options)


def test_probe_washout_and_service_startup_costs_are_charged_with_their_signs():
    traces = coupling_case()
    before = [value.copy() for value in traces]
    result = paired_coupling_summary(*traces, windows=tiny_windows(), dt_s=.002)
    # Five quarter-unit commissioning losses include late applied probes and
    # washout. Service starts with a half-unit loss, then gains two half-units.
    assert result["commissioning_cost"] == 1.25
    assert result["service_gain"] == .5
    assert result["charged_delta"] == -.75
    assert result["charged_mean_delta"] == -.0625
    assert result["total_mean_delta"] == pytest.approx(-.75 / 14)
    assert result["coupling_time_delta_s"] == -.0015
    assert result["charged_frames"] == 12
    assert result["total_frames"] == 14
    for value, original in zip(traces, before, strict=True):
        np.testing.assert_array_equal(value, original)


def test_withheld_service_correction_still_pays_commissioning_cost():
    nominal_commissioning, probed_commissioning, nominal_service, _ = coupling_case()
    result = paired_coupling_summary(nominal_commissioning, probed_commissioning,
                                    nominal_service, nominal_service.copy(), windows=tiny_windows())
    assert result["service_gain"] == 0
    assert result["commissioning_cost"] == 1.25
    assert result["charged_delta"] == -1.25
    assert result["charged_mean_delta"] == pytest.approx(-5 / 48)


def test_negative_commissioning_cost_is_not_clipped_to_zero():
    nominal, probed, service, _ = coupling_case()
    probed[2:] = .75
    result = paired_coupling_summary(nominal, probed, service, service.copy(), windows=tiny_windows())
    assert result["commissioning_cost"] == -2
    assert result["service_gain"] == 0
    assert result["charged_delta"] == 2


@pytest.mark.parametrize("direction", [-1, 1])
def test_finite_extreme_outcomes_reach_the_declared_utility_bounds(direction):
    nominal_level, candidate_level = (0., 1.) if direction == 1 else (1., 0.)
    nominal = np.full(10, nominal_level)
    candidate = np.full(10, candidate_level)
    candidate[:2] = nominal[:2]
    result = paired_coupling_summary(nominal, candidate, np.full(4, nominal_level),
                                    np.full(4, candidate_level), windows=tiny_windows())
    assert result["charged_mean_delta"] == direction
    assert result["total_mean_delta"] == pytest.approx(direction * 12 / 14)


@pytest.mark.parametrize("damage", ["nan", "infinity", "negative", "above_one", "short",
                                     "matrix", "warmup_mismatch"])
def test_invalid_or_unpaired_coupling_records_refuse(damage):
    traces = list(coupling_case())
    if damage == "nan":
        traces[1][-1] = np.nan
    elif damage == "infinity":
        traces[3][0] = np.inf
    elif damage == "negative":
        traces[3][0] = -.01
    elif damage == "above_one":
        traces[3][0] = 1.01
    elif damage == "short":
        traces[2] = traces[2][:-1]
    elif damage == "matrix":
        traces[2] = traces[2][:, None]
    else:
        traces[1][0] = .25
    with pytest.raises(ValueError):
        paired_coupling_summary(*traces, windows=tiny_windows())


@pytest.mark.parametrize("dt", [0., -1., np.nan, np.inf, True])
def test_invalid_sample_time_cannot_become_coupling_time(dt):
    with pytest.raises(ValueError):
        paired_coupling_summary(*coupling_case(), windows=tiny_windows(), dt_s=dt)


@pytest.mark.parametrize("dtype", [np.float32, np.int64, np.bool_])
def test_coupling_summary_requires_retained_float64_values(dtype):
    traces = list(coupling_case())
    traces[3] = traces[3].astype(dtype)
    with pytest.raises(ValueError, match="float64"):
        paired_coupling_summary(*traces, windows=tiny_windows())


def test_conditions_are_paired_inside_roots_not_counted_as_independent_samples():
    records = [record("r0", "c0", -.25), record("r0", "c1", -.25),
               record("r1", "c0", .25), record("r1", "c1", .25)]
    result = summarize_root_utilities(records, root_ids=["r0", "r1"], condition_ids=["c0", "c1"])
    assert result["status"] == "complete"
    assert result["n_roots"] == 2
    assert result["n_conditions"] == 2
    assert result["n_missing_cells"] == 0
    assert [row["mean_delta"] for row in result["root_values"]] == [-.25, .25]
    assert result["mean_delta"] == 0
    assert result["primary"]["n_roots"] == 2
    assert result["primary"]["degrees_of_freedom"] == 1
    assert result["primary"]["standard_error"] == .25
    # Student-t with one degree of freedom is Cauchy: the critical quantile
    # is cot(pi*alpha/2), independently of the implementation's t library.
    width = .25 / tan(pi * .05 / 2)
    np.testing.assert_allclose(result["primary"]["student_t_interval"], [-width, width], rtol=2e-11)
    np.testing.assert_array_equal(result["primary"]["hoeffding_interval"], [-1., 1.])
    for condition in ("c0", "c1"):
        statistics = result["condition_statistics"][condition]["statistics"]
        assert statistics["n_roots"] == 2
        assert statistics["alpha"] == .025
        adjusted = .25 / tan(pi * .025 / 2)
        np.testing.assert_allclose(statistics["student_t_interval"], [-adjusted, adjusted], rtol=2e-11)
    assert summarize_root_utilities(list(reversed(records)), root_ids=["r0", "r1"],
                                    condition_ids=["c0", "c1"]) == result


def test_eight_roots_and_four_conditions_use_seven_degrees_of_freedom():
    roots, conditions = [f"r{i}" for i in range(8)], [f"c{i}" for i in range(4)]
    records = [record(root, condition, (index - 3.5) / 8)
               for index, root in enumerate(roots) for condition in conditions]
    result = summarize_root_utilities(records, root_ids=roots, condition_ids=conditions)
    assert result["primary"]["n_roots"] == 8
    assert result["primary"]["degrees_of_freedom"] == 7
    assert result["mean_delta"] == 0
    for condition in conditions:
        statistics = result["condition_statistics"][condition]["statistics"]
        assert statistics["alpha"] == .05 / 4
        assert statistics["n_roots"] == 8
        assert statistics["degrees_of_freedom"] == 7


def test_benchmark_mean_does_not_hide_a_negative_condition():
    records = [record(root, condition, value) for root in ("r0", "r1")
               for condition, value in (("c0", .5), ("c1", -.25))]
    result = summarize_root_utilities(records, root_ids=["r0", "r1"], condition_ids=["c0", "c1"])
    assert result["mean_delta"] == .125
    assert result["condition_statistics"]["c1"]["mean_delta"] == -.25


def test_missing_outcome_retains_worst_best_bounds_without_complete_case_mean():
    records = [record("r0", "c0", .25), record("r0", "c1", .25),
               record("r1", "c0", .5), record("r1", "c1", None)]
    result = summarize_root_utilities(records, root_ids=["r0", "r1"], condition_ids=["c0", "c1"])
    assert result["status"] == "incomplete"
    assert result["n_roots"] == 2
    assert result["n_missing_cells"] == 1
    assert result["mean_delta"] is None
    assert result["primary"] is None
    np.testing.assert_array_equal(result["mean_bounds"], [0., .5])
    assert [row["mean_delta"] for row in result["root_values"]] == [.25, None]
    assert result["root_values"][1]["bounds"] == [-.25, .75]
    assert result["condition_statistics"]["c1"]["statistics"] is None
    assert result["condition_statistics"]["c0"]["statistics"]["n_roots"] == 2


def test_all_missing_outcomes_still_retain_the_declared_grid():
    records = [record(root, condition, None) for root in ("r0", "r1") for condition in ("c0", "c1")]
    result = summarize_root_utilities(records, root_ids=["r0", "r1"], condition_ids=["c0", "c1"])
    assert result["n_missing_cells"] == 4
    assert result["mean_delta"] is None
    assert result["primary"] is None
    assert result["mean_bounds"] == [-1., 1.]
    assert len(result["root_values"]) == 2


@pytest.mark.parametrize("damage", ["missing", "duplicate", "unknown_root", "unknown_condition",
                                     "nan", "infinity", "below_minus_one", "above_one", "bool", "truth_field"])
def test_aggregation_rejects_incomplete_identities_and_invalid_values(damage):
    records = [record("r0", "c0", -.25), record("r0", "c1", -.25),
               record("r1", "c0", .25), record("r1", "c1", .25)]
    if damage == "missing":
        records.pop()
    elif damage == "duplicate":
        records.append(records[0].copy())
    elif damage == "unknown_root":
        records[0]["root_id"] = "r2"
    elif damage == "unknown_condition":
        records[0]["condition_id"] = "c2"
    elif damage == "truth_field":
        records[0]["service_phase_truth"] = []
    else:
        records[0]["charged_mean_delta"] = {
            "nan": np.nan, "infinity": np.inf, "below_minus_one": -1.01, "above_one": 1.01,
            "bool": True,
        }[damage]
    with pytest.raises(ValueError):
        summarize_root_utilities(records, root_ids=["r0", "r1"], condition_ids=["c0", "c1"])


@pytest.mark.parametrize("alpha", [0., 1., -1., np.nan, np.inf, True])
def test_invalid_uncertainty_level_refuses(alpha):
    with pytest.raises(ValueError):
        summarize_root_utilities([record("r0", "c0", 0.)], root_ids=["r0"],
                                 condition_ids=["c0"], alpha=alpha)


@pytest.mark.parametrize("roots,conditions", [([], ["c0"]), (["r0"], []),
                                              (["r0", "r0"], ["c0"]),
                                              (["r0"], ["c0", "c0"]),
                                              ([" "], ["c0"]), ([1], ["c0"])])
def test_invalid_declared_identity_grids_refuse(roots, conditions):
    with pytest.raises(ValueError):
        summarize_root_utilities([], root_ids=roots, condition_ids=conditions)
