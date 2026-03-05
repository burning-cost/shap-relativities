"""
Synthetic datasets for testing and demonstration.

The motor dataset provides a synthetic UK personal lines motor portfolio with
a known data generating process. Use it to validate relativity extraction
against the true parameters.
"""

from .motor import TRUE_FREQ_PARAMS, TRUE_SEV_PARAMS, load_motor

__all__ = ["load_motor", "TRUE_FREQ_PARAMS", "TRUE_SEV_PARAMS"]
