"""Validate experiment assumptions before generating measured results."""
from __future__ import annotations


def validate_simulation(config: dict) -> None:
    """Reject configurations that violate the declared experiment contracts."""
    if not 100 <= config['modes'] <= 500:
        raise ValueError('Experiment DM must have 100 to 500 modes')
    if config['n'] < 16 or config['n'] % 2:
        raise ValueError('Use an even pupil grid with at least 16 samples')
    if config['frames'] <= config['burn_in'] or config['burn_in'] < 1:
        raise ValueError('A nonempty post-burn-in interval is required')
    if config['photons'] <= 0 or config['read_noise_e'] < 0:
        raise ValueError('Photon count must be positive and read noise nonnegative')
    if config['dt_s'] <= 0 or config['delay_frames'] < 1:
        raise ValueError('Positive time step and at least one frame of delay required')
    train=config['training_seeds']; test=config['seeds']
    # simulation.loop derives detector keys by adding 100000 to each root.
    # Distinct root IDs alone do not guarantee distinct random streams.
    if not train or not all(type(s) is int and 0 <= s < 2**32 - 100000
                            for s in train+test):
        raise ValueError('Explicit integer seeds with uint32 detector-stream headroom required')
    if len(set(test)) != len(test) or len(test)<5:
        raise ValueError('At least five distinct held-out seeds required')
    if len(set(train)) != len(train):
        raise ValueError('Training seeds must be unique')
    roots = train + test
    streams = roots + [seed + 100000 for seed in roots]
    if len(set(streams)) != len(streams):
        raise ValueError('Training or evaluation atmosphere/detector streams overlap')
    initial = config['seed']
    if type(initial) is not int or not 0 <= initial < 2**32:
        raise ValueError('Initialization seed must be an integer that fits uint32')
    if initial in streams:
        raise ValueError('Initialization seed overlaps an atmosphere or detector stream')
    levels=config['fidelity']['levels']; flux=config['fidelity']['fluxes']
    if len(levels)<3 or max(flux)/min(flux)<10000:
        raise ValueError('At least three fidelities and four decades of flux required')
    if len({l['name'] for l in levels}) != len(levels):
        raise ValueError('Fidelity names must be unique')
    if not 0 < config['detector']['calibration_quantile'] < 1:
        raise ValueError('Calibration quantile must be interior')
