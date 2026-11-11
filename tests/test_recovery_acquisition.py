"""Injected persistence tests; no prospective atmosphere or noise generation."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from noether_optics import recovery_acquisition as driver
from noether_optics import recovery_development as pilot
from test_recovery_development import injected  # Shared injected plant, not a simulator.


@pytest.fixture
def collection(tmp_path, monkeypatch, injected):
    frozen, config, executed = injected
    spec = pilot.protocol(smoke=True)
    spec.update(schema='fixture-recovery/1', role='injected_test', seeds=[91, 92], smoke=False)
    monkeypatch.setattr(driver, 'protocol', lambda: deepcopy(spec))
    monkeypatch.setattr(driver, 'capture_inputs', pilot.capture_inputs)
    return tmp_path / 'collection', frozen, config, executed


def run(collection, **kwargs):
    output, frozen, _, _ = collection
    return driver.run_collection(output, frozen, exclusion_sha256='a' * 64, **kwargs)


def test_root_resume_preserves_completed_bytes_without_redispatch(collection):
    output, frozen, _, executed = collection
    first = run(collection)
    assert first['status'] == 'incomplete_acquisition'
    assert first['completed_roots'] == 1
    assert len(executed) == 3
    assert not (output / 'acquisition.json').exists()
    packet = output / 'roots/91'
    before = {p.relative_to(packet): p.read_bytes() for p in packet.rglob('*') if p.is_file()}
    final = run(collection)
    assert final['status'] == 'complete_acquisition'
    assert final['completed_roots'] == 2
    assert [row[1] for row in executed] == [91] * 3 + [92] * 3
    assert before == {p.relative_to(packet): p.read_bytes() for p in packet.rglob('*') if p.is_file()}
    assert run(collection) == final
    assert len(executed) == 6
    second_manifest = driver.read_json(output / 'roots/92/manifest.json')
    assert second_manifest['case_indices'] == [1]
    assert second_manifest['protocol']['seeds'] == [91, 92]
    assert (output / 'roots/92/evidence/trials/001-continued.json').is_file()


@pytest.mark.parametrize('change', ['raw', 'missing', 'extra', 'symlink', 'incomplete', 'gap'])
def test_mutated_or_incomplete_root_prevents_any_further_dispatch(collection, change):
    output, _, _, executed = collection
    run(collection)
    packet = output / 'roots/91'
    raw = packet / 'evidence/trials/000-continued.npz'
    if change == 'raw':
        raw.write_bytes(b'changed')
    elif change == 'missing':
        raw.unlink()
    elif change == 'extra':
        (packet / 'extra.json').write_text('{}')
    elif change == 'symlink':
        raw.unlink()
        raw.symlink_to(packet / 'evidence/calibration.npz')
    elif change == 'incomplete':
        (packet / 'complete.json').unlink()
    else:
        packet.rename(output / 'roots/92')
    with pytest.raises(ValueError):
        run(collection)
    assert len(executed) == 3
    assert not (output / 'acquisition.json').exists()


def test_interrupted_root_is_retained_and_not_resampled(collection, monkeypatch):
    from noether_optics import recovery
    output, _, _, executed = collection
    def interrupted(*args):
        raise KeyboardInterrupt()
    monkeypatch.setattr(recovery, 'scheduled_handover_kernel', lambda *args, **kwargs: interrupted)
    with pytest.raises(KeyboardInterrupt):
        run(collection)
    packet = output / 'roots/91'
    assert len(executed) == 1
    assert (packet / 'evidence/trials/000-continued.npz').is_file()
    assert (packet / 'evidence/trials/000-issued_state-interrupted.json').is_file()
    assert (packet / 'stopped.json').is_file()
    with pytest.raises(ValueError, match='Incomplete root retained'):
        run(collection)
    assert len(executed) == 1


@pytest.mark.parametrize('field', ['source', 'runtime', 'exclusion_inventory_sha256'])
def test_changed_collection_binding_is_refused_before_dispatch(collection, monkeypatch, field):
    _, _, _, executed = collection
    run(collection)
    original = driver.collection_manifest
    def changed(*args):
        value = original(*args)
        value[field] = {'changed': True}
        return value
    monkeypatch.setattr(driver, 'collection_manifest', changed)
    with pytest.raises(ValueError, match='Frozen collection'):
        run(collection)
    assert len(executed) == 3


@pytest.mark.parametrize('count', [0, -1, True, 1.0, 3])
def test_bounded_root_count_rejected_without_output(collection, count):
    with pytest.raises(ValueError, match='max_roots'):
        run(collection, max_roots=count)
    assert not collection[0].exists()
    assert not collection[3]


def test_nonfinite_terminal_is_retained_and_skipped_on_resume(collection, monkeypatch):
    from noether_optics import recovery
    original = recovery.scheduled_handover_kernel
    def factory(*args, **kwargs):
        execute = original(*args, **kwargs)
        def nonfinite(*args):
            result = execute(*args)
            result['measurement'][300, 0] = np.nan
            return result
        return nonfinite
    monkeypatch.setattr(recovery, 'scheduled_handover_kernel', factory)
    run(collection)
    report = driver.read_json(collection[0] / 'roots/91/report.json')
    assert report['status'] == 'complete_study_shard_with_withheld_outcomes'
    assert report['cases'][0]['contrasts']['issued_state']['status'] == 'nonfinite_outputs'
    run(collection)
    assert [row[1] for row in collection[3]] == [91] * 3 + [92] * 3


@pytest.fixture
def auditor(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / 'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('recovery_raw_audit', scripts / 'audit_recovery_study.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_raw_audit_rejects_plausible_but_physically_wrong_saved_variance(collection, auditor):
    output, frozen, _, _ = collection
    run(collection)
    manifest = driver.read_json(output / 'manifest.json')
    predictor, cal, _ = pilot.load_training(frozen)
    # This fixture deliberately stores variance=1 for zero physical phase.
    # Hash integrity alone passes; the independent optical equation must fail.
    with pytest.raises(ValueError, match='physical accounting differs'):
        auditor.reconstruct_root(output / 'roots/91', manifest,
            pilot._hash_file(output / 'manifest.json'), 91, {}, cal, predictor, {})


def test_incomplete_collection_cannot_generate_statistics(collection, auditor):
    run(collection)
    with pytest.raises(FileNotFoundError):
        auditor.audit(collection[0], collection[1])


def test_collection_lock_rejects_concurrent_writer(collection):
    import fcntl
    output = collection[0]
    output.mkdir()
    with (output / '.lock').open('a+b') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            run(collection)
    assert not collection[3]
    assert not (output / 'manifest.json').exists()


@pytest.fixture
def physical_collection(collection, monkeypatch):
    from noether_optics import atmosphere, controller_physics, recovery
    _, frozen, config, _ = collection
    config['faults'].update(vibration_rad=0., vibration_hz=1.)
    base_name = 'evidence/provenance/base.json'
    (frozen / base_name).write_bytes(pilot._bytes(config))
    pilot.INPUTS[base_name] = pilot._hash_file(frozen / base_name)
    # A constant zero-piston phase with unit variance and zero mirror command.
    # Its pupil-coherent peak is cos(1)^2, independently of the stored metrics.
    phase = np.tile(np.array([[1., -1.], [-1., 1.]])[None], (512, 1, 1))
    monkeypatch.setattr(atmosphere, 'phase_sequence', lambda cfg, seed: phase.copy())
    monkeypatch.setattr(controller_physics, 'paired_open_loop', lambda *args: {
        'variance': np.ones(512), 'strehl': np.full(512, np.cos(1.) ** 2),
        'saturated': np.zeros(512, bool)})
    original = controller_physics.controller_kernel
    def factory(*args, mode=None, **kwargs):
        execute = original(*args)
        def physical(*args):
            outputs = execute(*args)
            outputs['strehl'][:] = np.cos(1.) ** 2
            if mode is not None:
                outputs['handover_triggered'] = np.arange(512) >= 256
                outputs['handover_target_active'] = np.arange(512) >= 256 + (mode == 'command_matched')
                outputs['handover_requested'] = outputs['command'].copy()
            return outputs
        return physical
    monkeypatch.setattr(controller_physics, 'controller_kernel', factory)
    monkeypatch.setattr(recovery, 'scheduled_handover_kernel', lambda *args, **kwargs: factory(*args, mode=kwargs['mode']))
    return collection


def test_independent_physical_reconstruction_returns_exact_pairs(physical_collection, auditor):
    output, frozen, _, _ = physical_collection
    run(physical_collection)
    manifest = driver.read_json(output / 'manifest.json')
    predictor, cal, selection = pilot.load_training(frozen)
    records, audit = auditor.reconstruct_root(output / 'roots/91', manifest,
        pilot._hash_file(output / 'manifest.json'), 91, {}, cal, predictor, selection)
    assert records == [dict(seed=91, photons=1000., wind_scale=.5, fault='none',
                            arm=arm, status='finite_pair', value=0.) for arm in pilot.ARMS[1:]]
    assert audit['finite_arm_records'] == 3
    assert audit['nonfinite_arm_records'] == 0
    assert max(audit['max_absolute_error'].values()) < 1e-15


def test_audit_retains_nonfinite_pair_without_calling_it_a_physical_pass(physical_collection, auditor, monkeypatch):
    from noether_optics import recovery
    original = recovery.scheduled_handover_kernel
    def factory(*args, **kwargs):
        execute = original(*args, **kwargs)
        def nonfinite(*args):
            outputs = execute(*args)
            if kwargs['mode'] == 'issued_state':
                outputs['measurement'][300, 0] = np.nan
            return outputs
        return nonfinite
    monkeypatch.setattr(recovery, 'scheduled_handover_kernel', factory)
    output, frozen, _, _ = physical_collection
    run(physical_collection)
    manifest = driver.read_json(output / 'manifest.json')
    predictor, cal, selection = pilot.load_training(frozen)
    records, audit = auditor.reconstruct_root(output / 'roots/91', manifest,
        pilot._hash_file(output / 'manifest.json'), 91, {}, cal, predictor, selection)
    assert records[0]['status'] == 'nonfinite_pair' and records[0]['value'] is None
    assert records[1]['status'] == 'finite_pair'
    assert audit['finite_arm_records'] == 2 and audit['nonfinite_arm_records'] == 1
    assert audit['withheld_pairs'][0]['raw_status'] == 'nonfinite_outputs'


def test_nonfinite_reference_withholds_both_pairs(physical_collection, auditor, monkeypatch):
    from noether_optics import controller_physics
    original = controller_physics.paired_open_loop
    def nonfinite(*args):
        result = original(*args)
        result['variance'][300] = np.nan
        return result
    monkeypatch.setattr(controller_physics, 'paired_open_loop', nonfinite)
    output, frozen, _, _ = physical_collection
    run(physical_collection)
    manifest = driver.read_json(output / 'manifest.json')
    predictor, cal, selection = pilot.load_training(frozen)
    records, audit = auditor.reconstruct_root(output / 'roots/91', manifest,
        pilot._hash_file(output / 'manifest.json'), 91, {}, cal, predictor, selection)
    assert all(row['status'] == 'nonfinite_pair' and row['value'] is None for row in records)
    assert len(audit['nonfinite_reference_cases']) == 1
    assert audit['finite_arm_records'] == 3


def test_complete_injected_cohort_reconstructs_before_statistics(physical_collection, auditor, monkeypatch):
    from noether_optics.recovery_study import protocol
    # These are identity labels only. The injected constant phase and controller
    # call no RNG and acquire none of the prospective physical/noise streams.
    monkeypatch.setattr(driver, 'protocol', protocol)
    monkeypatch.setattr(auditor, 'capture_inputs', pilot.capture_inputs)
    output, frozen, _, executed = physical_collection
    result = run(physical_collection, max_roots=64)
    assert result['completed_roots'] == 64 and len(executed) == 2304
    report = auditor.audit(output, frozen)
    assert report['status'] == 'complete_raw_reconstructed_report'
    assert len(report['root_audits']) == 64
    contrasts = report['statistics']['contrasts']
    assert len(contrasts) == 24
    assert all(row['roots'] == 64 and row['mean'] == 0. and row['interval'] is None
               and row['status'] == 'zero_observed_variance' for row in contrasts)
