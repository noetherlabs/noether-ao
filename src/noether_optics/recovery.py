"""Scheduled controller-transfer mechanics, without a detector or study runner.

The caller supplies a fixed event frame. This can be an oracle experimental
control, but does not make the event observable from physical measurements.
No scientific cohort, acceptance criterion or safety claim is defined here.
"""
from __future__ import annotations

from copy import deepcopy
from math import isfinite
from numbers import Real

import jax
import jax.numpy as jnp
import numpy as np


def validate_handover(config: dict, frames: int, frame: int, mode: str) -> None:
    """Validate static transfer settings before tracing the optical loop."""
    if type(frames) is not int or frames < 1:
        raise ValueError('Handover requires a positive episode length')
    if type(frame) is not int or not 0 <= frame < frames:
        raise ValueError('Handover frame must be an integer within the episode')
    if not isinstance(mode, str) or mode not in ('issued_state', 'command_matched'):
        raise ValueError('Unknown handover mode')
    delay = config['delay_frames']
    if type(delay) is not int or delay < 1:
        raise ValueError('Handover delay must be a positive integer')
    for name in ('gain', 'leak', 'command_clip_rad'):
        value = config[name]
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not isfinite(value):
            raise ValueError('Handover parameters must be finite real numbers')
    if config['gain'] < 0 or not 0 < config['leak'] <= 1 or config['command_clip_rad'] <= 0:
        raise ValueError('Handover requires gain >=0, leak in (0,1] and positive command clip')


def scheduled_handover_kernel(config: dict, calibration: dict, predictor,
                              *, frame: int, mode: str, fault: str = 'none'):
    """Return a fixed predictive-to-integrator kernel with boundary telemetry.

    Configuration and predictor arrays are copied. Calibration is the existing
    immutable simulator calibration. Returned nonfinite streams are retained for
    the caller's failure policy; they are never silently filtered or retried.
    """
    from .controller_costs import measured_jit
    from .simulation import loop

    cfg = deepcopy(config)
    validate_handover(cfg, cfg['frames'], frame, mode)
    if cfg['sensor'] not in ('linear', 'pyramid') or fault not in ('none', 'optical_gain', 'vibration'):
        raise ValueError('Unsupported scheduled-handover sensor or fault')
    coefficients = np.asarray(predictor)
    if (coefficients.dtype.kind not in 'fiu' or coefficients.shape != (cfg['modes'], 2)
            or not np.isfinite(coefficients).all()):
        raise ValueError('Handover requires finite frozen predictor coefficients')
    coefficients = jnp.asarray(np.array(coefficients, dtype=np.float64, copy=True))
    compiled = measured_jit(jax.jit(lambda phases, seed: loop(
        cfg, calibration, phases, seed, 'predictive', coefficients,
        fault=fault, handover_frame=frame, handover_mode=mode)), 'scheduled_handover',
        {'frame': frame, 'mode': mode, 'fault': fault})

    def execute(phases, seed):
        array = np.asarray(phases)
        if (array.shape != (cfg['frames'], cfg['n'], cfg['n'])
                or array.dtype.kind not in 'fiu' or not np.isfinite(array).all()):
            raise ValueError('Handover requires a complete finite atmosphere')
        if type(seed) is not int or not 0 <= seed < 2**32 - 100000:
            raise ValueError('Atmosphere/detector seed must fit uint32 with its noise offset')
        return jax.block_until_ready(compiled(phases, seed))

    return execute
