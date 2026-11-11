"""Development persistence and wiring checks; no real atmosphere acquisition."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from noether_optics import recovery_development as pilot


def test_development_schedule_has_paired_unique_roots_and_all_controls():
    spec = pilot.protocol()
    cases = pilot.cases(spec)
    assert len(cases) == 48
    assert len({tuple(row.items()) for row in cases}) == 48
    assert set(spec['seeds']).isdisjoint({seed + spec['noise_offset'] for seed in spec['seeds']})
    for seed in spec['seeds']:
        group = [row for row in cases if row['seed'] == seed]
        assert len(group) == 12
        assert {row['fault'] for row in group} == {'none', 'optical_gain', 'vibration'}
    assert len(pilot.cases(pilot.protocol(smoke=True))) == 1
    assert spec['role'] == 'development'


def test_terminal_nonfinite_raw_is_retained_after_durable_intent(tmp_path):
    run = pilot.DevelopmentRun(tmp_path / 'run', {})
    outputs = {'measurement': np.array([1., np.nan])}
    def execute(phases, seed):
        intent = json.loads((run.root / 'evidence/trials/fixture-intent.json').read_bytes())
        assert intent['seed'] == seed
        assert not (run.root / 'evidence/trials/fixture.json').exists()
        return outputs
    actual, ref = pilot.acquire_arm(run, 'fixture', {}, execute, None, 17, warm=False)
    assert np.isnan(actual['measurement'][-1])
    terminal = json.loads((run.root / ref['path']).read_bytes())
    assert terminal['status'] == 'nonfinite'
    assert pilot._hash_file(run.root / terminal['raw']['path']) == terminal['raw']['sha256']
    with np.load(run.root / terminal['raw']['path'], allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['measurement'], outputs['measurement'])


def test_interruption_cannot_be_counted_as_terminal_or_reopened(tmp_path):
    run = pilot.DevelopmentRun(tmp_path / 'run', {})
    def execute(phases, seed):
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        pilot.acquire_arm(run, 'fixture', {}, execute, None, 17, warm=False)
    assert (run.root / 'evidence/trials/fixture-intent.json').is_file()
    assert (run.root / 'evidence/trials/fixture-interrupted.json').is_file()
    assert not (run.root / 'evidence/trials/fixture.json').exists()
    with pytest.raises(FileExistsError):
        pilot.DevelopmentRun(run.root, {})


def test_frozen_input_hash_is_checked_before_archive_loading(tmp_path, monkeypatch):
    (tmp_path / 'training.json').write_text('modified')
    monkeypatch.setattr(pilot, 'INPUTS', {'training.json': '0'*64})
    with pytest.raises(ValueError, match='Frozen recovery input changed'):
        pilot.load_training(tmp_path)


def test_published_artifact_mutation_is_not_blessed_by_a_new_inventory(tmp_path):
    run = pilot.DevelopmentRun(tmp_path / 'run', {})
    pilot.save_arrays(run, 'evidence/raw.npz', {'value': np.array([1.])})
    run.verify()
    (run.root / 'evidence/raw.npz').write_bytes(b'changed')
    with pytest.raises(ValueError, match='artifact changed'):
        run.verify()


@pytest.fixture
def injected(tmp_path, monkeypatch):
    from noether_optics import atmosphere, controller_physics, recovery, simulation

    frozen = tmp_path / 'old-study'
    frozen.mkdir()
    (frozen / 'training.json').write_text('{}')
    monkeypatch.setattr(pilot.shutil, 'disk_usage', lambda path: SimpleNamespace(free=pilot.MIN_FREE_BYTES+1))
    config = dict(frames=512, modes=2, n=2, burn_in=128, faults=dict(onset=256, end=512),
                  delay_frames=2, command_clip_rad=5., dt_s=.001, wavelength_m=750e-9)
    (frozen / 'evidence/provenance').mkdir(parents=True)
    (frozen / 'evidence/provenance/base.json').write_bytes(pilot._bytes(config))
    (frozen / 'evidence/provenance/study.json').write_bytes(pilot._bytes({}))
    monkeypatch.setattr(pilot, 'INPUTS', {name: pilot._hash_file(frozen / name) for name in (
        'training.json', 'evidence/provenance/base.json', 'evidence/provenance/study.json')})
    source = {'commit': 'fixture', 'files': {}}
    cal = {'basis': np.zeros((2, 2, 2)), 'mask': np.ones((2, 2))}
    monkeypatch.setattr(pilot, 'load_training', lambda path: (np.ones((2, 2)), cal, {}))
    monkeypatch.setattr(pilot, 'capture_inputs', lambda *args: {
        'source': source, 'runtime': {'fixture': True}, 'study': {}, 'base': config})
    monkeypatch.setattr(pilot, '_source', lambda *args: source)
    monkeypatch.setattr(simulation, 'calibration', lambda base: cal)
    monkeypatch.setattr(atmosphere, 'phase_sequence', lambda cfg, seed: np.zeros((512, 2, 2)))
    monkeypatch.setattr(controller_physics, 'episode_config', lambda *args: dict(config))
    monkeypatch.setattr(controller_physics, 'paired_open_loop', lambda *args: {
        'variance': np.full(512, 2.), 'strehl': np.full(512, .2), 'saturated': np.zeros(512, bool)})
    executed = []
    def kernel(name):
        def execute(phases, seed):
            executed.append((name, seed, phases.copy()))
            out = {key: np.zeros(512, bool) for key in ('saturated', 'alarm', 'fallback', 'unavailable')}
            out.update(variance=np.ones(512), strehl=np.full(512, .5), score=np.zeros(512))
            for key in ('measurement', 'prediction', 'command', 'applied', 'estimate'):
                out[key] = np.zeros((512, 2))
            return out
        return execute
    monkeypatch.setattr(controller_physics, 'controller_kernel', lambda *args: kernel('continued'))
    monkeypatch.setattr(recovery, 'scheduled_handover_kernel', lambda *args, **kwargs: kernel(kwargs['mode']))
    return frozen, config, executed


def test_injected_smoke_retains_all_three_arms_and_binds_complete_artifacts(tmp_path, injected):
    frozen, config, executed = injected
    output = tmp_path / 'pilot'
    report = pilot.run_development(output, frozen, smoke=True)
    assert report['status'] == 'complete_finite_development'
    assert report['terminal_records'] == report['expected_terminal_records'] == 3
    assert [row[0] for row in executed] == ['continued', 'issued_state', 'command_matched']
    assert len({row[1] for row in executed}) == 1
    np.testing.assert_array_equal(executed[0][2], executed[2][2])
    contrasts = report['cases'][0]['contrasts']
    for arm, effect in [('issued_state', 258), ('command_matched', 259)]:
        assert contrasts[arm]['first_possible_effect_frame'] == effect
        assert contrasts[arm]['horizon_frames'] == [256, 512]
        assert contrasts[arm]['outcomes']['normalized_excess_variance'] == 0
    complete = json.loads((output / 'complete.json').read_bytes())
    assert {'manifest.json', 'report.json'} <= complete['files'].keys()
    for name, digest in complete['files'].items():
        assert pilot._hash_file(output / name) == digest
    assert not (output / 'stopped.json').exists()
    with pytest.raises(ValueError, match='fresh output'):
        pilot.run_development(output, frozen, smoke=True)


def test_input_calibration_mismatch_stops_before_any_controller(tmp_path, injected, monkeypatch):
    from noether_optics import simulation
    frozen, config, executed = injected
    monkeypatch.setattr(simulation, 'calibration', lambda cfg: {'wrong': np.zeros(2)})
    output = tmp_path / 'pilot'
    with pytest.raises(ValueError, match='calibration differs'):
        pilot.run_development(output, frozen, smoke=True)
    assert executed == []
    assert (output / 'evidence/calibration.npz').is_file()
    assert (output / 'stopped.json').is_file()
    assert not (output / 'complete.json').exists()


def test_source_change_withholds_completion_even_after_raw_trials(tmp_path, injected, monkeypatch):
    frozen, config, executed = injected
    monkeypatch.setattr(pilot, '_source', lambda *args: {'commit': 'changed'})
    output = tmp_path / 'pilot'
    with pytest.raises(ValueError, match='Source changed'):
        pilot.run_development(output, frozen, smoke=True)
    assert len(executed) == 3
    assert len(list((output / 'evidence/trials').glob('*.npz'))) == 3
    assert (output / 'stopped.json').is_file()
    assert not (output / 'report.json').exists()


def test_changed_plant_or_study_configuration_is_rejected_before_acquisition(tmp_path, injected, monkeypatch):
    frozen, config, executed = injected
    original = pilot.capture_inputs
    def changed(*args):
        inputs = original(*args)
        inputs['base'] = {**inputs['base'], 'dt_s': .002}
        return inputs
    monkeypatch.setattr(pilot, 'capture_inputs', changed)
    output = tmp_path / 'pilot'
    with pytest.raises(ValueError, match='configuration differs'):
        pilot.run_development(output, frozen, smoke=True)
    assert not output.exists()
    assert executed == []
