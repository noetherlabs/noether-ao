"""Simulation cohorts must not reuse atmosphere or derived detector streams."""
from pathlib import Path

import pytest

from noether_optics.artifacts import load_config
from noether_optics.config import validate_simulation


@pytest.fixture
def config():
    return load_config(Path(__file__).parents[1] / 'configs/simulation.yaml')


@pytest.mark.parametrize('role', ['training_seeds', 'seeds'])
@pytest.mark.parametrize('seed', [True, 2**32 - 100000, 2**32 + 11])
def test_rejects_invalid_or_overflowing_stream_roots(config, role, seed):
    config[role][0] = seed
    with pytest.raises(ValueError, match='seed|stream'):
        validate_simulation(config)


def test_rejects_duplicate_training_seed(config):
    config['training_seeds'][1] = config['training_seeds'][0]
    with pytest.raises(ValueError, match='unique|distinct'):
        validate_simulation(config)


@pytest.mark.parametrize('role', ['training_seeds', 'seeds'])
def test_rejects_atmosphere_detector_stream_collision(config, role):
    # These distinct root IDs generate exactly the same PRNG key: one as a
    # detector stream, the other as an atmosphere stream.
    config[role][1] = config['training_seeds'][0] + 100000
    with pytest.raises(ValueError, match='overlap'):
        validate_simulation(config)


def test_rejects_initialization_stream_collision(config):
    config['seed'] = config['seeds'][0] + 100000
    with pytest.raises(ValueError, match='overlap'):
        validate_simulation(config)


def test_accepts_highest_nonoverflowing_root(config):
    config['training_seeds'][0] = 2**32 - 100001
    validate_simulation(config)


@pytest.mark.parametrize('seed', [True, -1, 2**32])
def test_rejects_invalid_initialization_seed(config, seed):
    config['seed'] = seed
    with pytest.raises(ValueError, match='seed'):
        validate_simulation(config)


def test_accepts_recorded_simulation_configuration(config):
    validate_simulation(config)
