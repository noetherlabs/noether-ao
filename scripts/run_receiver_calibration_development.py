"""Run one admitted commissioning/service receiver trial; never overwrite it."""
import os

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('JAX_ENABLE_X64', '1')

from noether_optics.receiver_calibration_runner import main


if __name__ == '__main__':
    main()
