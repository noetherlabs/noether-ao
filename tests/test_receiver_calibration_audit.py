"""Independent packet reconstruction and bounded hostile-header refusals."""

from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys
import zipfile

import pytest

pytest.importorskip('noether_physics')

from noether_optics import receiver_calibration_runner as runner
from test_receiver_calibration_runner import (
    direct_coupling, file_sha, inventory, read_arrays, read_json, tiny_case,
)


SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'


@pytest.fixture
def auditor(monkeypatch):
    # Load the two repository audit scripts by their exact locations. Optical
    # dependencies still resolve from the normally installed environment.
    physical_spec = importlib.util.spec_from_file_location(
        'audit_controller_physical_metrics', SCRIPTS / 'audit_controller_physical_metrics.py')
    physical = importlib.util.module_from_spec(physical_spec)
    physical_spec.loader.exec_module(physical)
    monkeypatch.setitem(sys.modules, 'audit_controller_physical_metrics', physical)
    spec = importlib.util.spec_from_file_location(
        'receiver_packet_audit', SCRIPTS / 'audit_receiver_calibration.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def completed_packet(tiny_case, tmp_path):
    config, audit, options, captured = tiny_case
    packet = tmp_path / 'completed'
    result = runner.run_trial(config, audit, output=packet, **options)
    assert result['status'] == 'completed'
    return packet, config, audit, options, captured


def audit_options(case):
    packet, config, audit, options, _ = case
    return dict(config_path=config, seed_audit_path=audit,
                expected_result_sha256=file_sha(packet / 'result.json'),
                expected_config_sha256=options['expected_config_sha256'],
                expected_seed_audit_sha256=options['expected_seed_audit_sha256'])


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')


def bind_file(packet, result, name, *, arrays=None):
    path = packet / name
    result['files'][name] = {'sha256': file_sha(path), 'bytes': path.stat().st_size}
    if arrays is not None:
        result['files'][name]['arrays'] = arrays


def rebind_fit(packet, result):
    """Keep hashes self-consistent so semantic refusals cannot pass on a digest error."""
    fit = read_json(packet / 'fit.json')
    for key, name in (('inputs_sha256', 'inputs.json'),
                      ('calibration_sha256', 'calibration.npz'),
                      ('commissioning_probed_sha256', 'commissioning-probed.npz'),
                      ('probe_sha256', 'probe.npz')):
        fit[key] = result['files'][name]['sha256']
    write_json(packet / 'fit.json', fit)
    bind_file(packet, result, 'fit.json')
    freeze = read_json(packet / 'fit-freeze.json')
    freeze['fit_sha256'] = result['files']['fit.json']['sha256']
    write_json(packet / 'fit-freeze.json', freeze)
    bind_file(packet, result, 'fit-freeze.json')
    result['fit_sha256'] = freeze['fit_sha256']
    write_json(packet / 'result.json', result)


def test_completed_physical_packet_reconstructs_without_mutation(completed_packet, auditor):
    packet, _, _, _, captured = completed_packet
    before = inventory(packet)
    report = auditor.audit_trial(packet, **audit_options(completed_packet))
    assert inventory(packet) == before
    assert report['status'] == 'raw_receiver_utility_reconstructed'
    assert report['fit_status'] == 'withheld'
    assert report['result_sha256'] == before['result.json']
    assert len(report['max_absolute_errors']) == 12
    assert max(report['max_absolute_errors'].values()) < 1e-10

    protocol = captured['protocol']
    cal = read_arrays(packet / 'calibration.npz')
    traces = {}
    for arm in ('commissioning-nominal', 'commissioning-probed',
                'service-nominal', 'service-corrected'):
        phase = read_arrays(packet / (arm.split('-')[0] + '-phase.npz'))['phase']
        applied = read_arrays(packet / (arm + '.npz'))['applied']
        traces[arm] = direct_coupling(phase, applied, cal, protocol)
    start = protocol['windows']['probe_start']
    cost = math.fsum((traces['commissioning-nominal'] - traces['commissioning-probed'])[start:])
    gain = math.fsum(traces['service-corrected'] - traces['service-nominal'])
    assert cost > 0 and gain == 0
    assert report['utility']['commissioning_cost'] == pytest.approx(cost, abs=2e-10)
    assert report['utility']['charged_mean_delta'] == pytest.approx((gain-cost)/22, abs=2e-10)
    assert report['utility']['total_mean_delta'] == pytest.approx((gain-cost)/24, abs=2e-10)
    assert report['charged_mean_delta'] < 0


@pytest.mark.parametrize('mismatch', ['terminal', 'artifact', 'config', 'stream'])
def test_hash_or_external_stream_mismatch_refuses_before_loading_arrays(
    completed_packet, auditor, monkeypatch, mismatch,
):
    packet, _, _, _, _ = completed_packet
    options = audit_options(completed_packet)
    if mismatch == 'terminal':
        options['expected_result_sha256'] = '0'*64
    elif mismatch == 'artifact':
        with (packet / 'summary.json').open('ab') as stream:
            stream.write(b' ')
    elif mismatch == 'config':
        options['expected_config_sha256'] = '0'*64
    else:
        inputs = read_json(packet / 'inputs.json')
        inputs['streams']['probe'] ^= 1
        write_json(packet / 'inputs.json', inputs)
        result = read_json(packet / 'result.json')
        bind_file(packet, result, 'inputs.json')
        rebind_fit(packet, result)
        options['expected_result_sha256'] = file_sha(packet / 'result.json')

    def forbidden_load(*args, **kwargs):
        pytest.fail('Invalid external identities reached numerical archive loading')

    monkeypatch.setattr(auditor.np, 'load', forbidden_load)
    with pytest.raises(ValueError, match='identity|bytes differ|stream row differs'):
        auditor.audit_trial(packet, **options)


def test_completed_terminal_bounds_cannot_disagree_with_its_score(completed_packet, auditor):
    packet = completed_packet[0]
    result = read_json(packet / 'result.json')
    result['utility_bounds'] = [0., 0.]
    write_json(packet / 'result.json', result)
    with pytest.raises(ValueError, match='(?i)bound'):
        auditor.audit_trial(packet, **audit_options(completed_packet))


@pytest.mark.parametrize('attack', ['large_allocation', 'large_header', 'object_dtype'])
def test_small_npz_rejects_unsafe_npy_header_before_numpy_load(
    completed_packet, auditor, monkeypatch, attack,
):
    packet = completed_packet[0]
    shape = (2**40,) if attack == 'large_allocation' else (1,)
    descr = '|O' if attack == 'object_dtype' else '<f8'
    header = repr({'descr': descr, 'fortran_order': False, 'shape': shape}).encode('ascii')
    header += b' ' * (5000-len(header) if attack == 'large_header' else (-len(header)-13) % 64)
    header += b'\n'
    payload = b'' if attack == 'large_allocation' else b'\0'*8
    npy = b'\x93NUMPY\x02\x00' + struct.pack('<I', len(header)) + header + payload
    target = packet / 'calibration.npz'
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_STORED) as archive:
        archive.writestr('basis.npy', npy)
    assert target.stat().st_size < 8192
    result = read_json(packet / 'result.json')
    bind_file(packet, result, 'calibration.npz', arrays={
        'basis': {'shape': list(shape), 'dtype': descr, 'bytes': math.prod(shape)*8},
    })
    rebind_fit(packet, result)

    def forbidden_load(*args, **kwargs):
        pytest.fail('An unsafe NPY header reached np.load')

    monkeypatch.setattr(auditor.np, 'load', forbidden_load)
    expected = {'large_allocation': 'allocation|bound',
                'large_header': '(?i)header', 'object_dtype': 'dtype|shape'}[attack]
    with pytest.raises(ValueError, match=expected):
        auditor.audit_trial(packet, **audit_options(completed_packet))


def failed_packet(tiny_case, tmp_path, input_mode):
    """Failure-integrity fixtures contain no acquired optical arrays."""
    config, audit, options, captured = tiny_case
    packet = tmp_path / ('failed-' + input_mode)
    packet.mkdir()
    partial = packet / 'partial.npz'
    partial.write_bytes(b'a bounded interrupted archive, not valid optical data')
    result = {
        'schema': 'noether-receiver-calibration-trial/1', 'status': 'failed',
        'study_id': captured['protocol']['study_id'],
        'root_id': options['root_id'], 'condition_id': options['condition_id'],
        'stage': 'commissioning_phase', 'fit_status': None, 'fit_sha256': None,
        'charged_mean_delta': None, 'utility_bounds': [-1., 1.],
        'error': {'type': 'ValueError', 'message': 'Retained unit-fixture failure'},
        'logical_scientific_bytes': 0, 'files': {}, 'missing_artifacts': [],
    }
    bind_file(packet, result, 'partial.npz')
    if input_mode != 'absent':
        if input_mode == 'malformed':
            (packet / 'inputs.json').write_bytes(b'{"unterminated')
        else:
            write_json(packet / 'inputs.json', captured)
        bind_file(packet, result, 'inputs.json')
    write_json(packet / 'result.json', result)
    return packet, config, audit, options, captured


@pytest.mark.parametrize('input_mode', ['absent', 'malformed', 'valid'])
def test_failed_packet_verifies_retained_bytes_without_claiming_reconstruction(
    tiny_case, tmp_path, auditor, monkeypatch, input_mode,
):
    case = failed_packet(tiny_case, tmp_path, input_mode)
    packet = case[0]
    before = inventory(packet)
    monkeypatch.setattr(auditor.np, 'load', lambda *a, **k: pytest.fail('Failed payload was interpreted'))
    report = auditor.audit_trial(packet, **audit_options(case))
    assert report['status'] == 'retained_failed_trial'
    assert report['charged_mean_delta'] is None
    assert report['utility_bounds'] == [-1., 1.]
    assert report['scientific_payload_reconstructed'] is False
    assert report['input_binding_verified'] is (input_mode == 'valid')
    assert (report['input_binding_error'] is None) is (input_mode == 'valid')
    assert report['retained_files_verified'] == (1 if input_mode == 'absent' else 2)
    assert inventory(packet) == before


@pytest.mark.parametrize('damage', [None, 'string_entry', 'duplicate', 'unsafe_path', 'present_file'])
def test_failed_packet_missing_artifacts_are_explicit_original_identity_records(
    tiny_case, tmp_path, auditor, damage,
):
    case = failed_packet(tiny_case, tmp_path, 'valid')
    packet = case[0]
    result = read_json(packet / 'result.json')
    missing = {'file': 'missing-phase.npz', 'original_sha256': 'a'*64}
    if damage == 'string_entry':
        result['missing_artifacts'] = ['missing-phase.npz']
    elif damage == 'duplicate':
        result['missing_artifacts'] = [missing, deepcopy(missing)]
    else:
        if damage == 'unsafe_path':
            missing['file'] = '../missing-phase.npz'
        elif damage == 'present_file':
            missing['file'] = 'partial.npz'
        result['missing_artifacts'] = [missing]
    write_json(packet / 'result.json', result)
    if damage is not None:
        with pytest.raises(ValueError, match='missing-artifact'):
            auditor.audit_trial(packet, **audit_options(case))
    else:
        report = auditor.audit_trial(packet, **audit_options(case))
        assert report['missing_artifacts'] == [missing]
        assert report['scientific_payload_reconstructed'] is False


def test_failed_packet_cannot_replace_unknown_utility_by_zero(tiny_case, tmp_path, auditor):
    case = failed_packet(tiny_case, tmp_path, 'absent')
    result = read_json(case[0] / 'result.json')
    result['charged_mean_delta'] = 0.
    write_json(case[0] / 'result.json', result)
    with pytest.raises(ValueError, match='numerical success'):
        auditor.audit_trial(case[0], **audit_options(case))


@pytest.mark.parametrize('relative', ['audit.json', 'child/../audit.json'])
def test_cli_refuses_writing_inside_packet_before_reading_evidence(
    tmp_path, auditor, monkeypatch, relative,
):
    packet = tmp_path / 'packet'
    packet.mkdir()
    (packet / 'preserve').write_bytes(b'unchanged original evidence')
    before = inventory(packet)
    monkeypatch.setattr(auditor, 'audit_trial', lambda *a, **k: pytest.fail('CLI output guard ran too late'))
    monkeypatch.setattr(sys, 'argv', [
        'audit_receiver_calibration.py', str(packet), '--config', str(tmp_path/'config.json'),
        '--seed-audit', str(tmp_path/'seeds.json'), '--result-sha256', '0'*64,
        '--config-sha256', '0'*64, '--seed-audit-sha256', '0'*64,
        '--output', str(packet/relative),
    ])
    with pytest.raises(ValueError, match='outside the retained packet'):
        auditor.main()
    assert inventory(packet) == before
