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
    if len(set(test)) != len(test) or len(test)<5:
        raise ValueError('At least five distinct held-out seeds required')
    if set(train)&set(test):
        raise ValueError('Training and evaluation seeds overlap')
    if not train or not all(isinstance(s,int) and s>=0 for s in train+test):
        raise ValueError('Explicit nonnegative integer seeds required')
    levels=config['fidelity']['levels']; flux=config['fidelity']['fluxes']
    if len(levels)<3 or max(flux)/min(flux)<10000:
        raise ValueError('At least three fidelities and four decades of flux required')
    if len({l['name'] for l in levels}) != len(levels):
        raise ValueError('Fidelity names must be unique')
    if not 0 < config['detector']['calibration_quantile'] < 1:
        raise ValueError('Calibration quantile must be interior')
