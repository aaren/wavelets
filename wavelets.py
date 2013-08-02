from functools import wraps

import numpy as np
import scipy.signal


class Wavelets(object):
    """Container for various wavelet basis functions.

    To be admissible as a wavelet, a function must:

    - have zero mean
    - be localised in both time and frequency space

    These functions are a function of a dimensionless time
    parameter.
    """
    # morlet wavelet
    morlet = scipy.signal.morlet
    # ricker wavelet
    ricker = scipy.signal.ricker
    # aka Derivitive Of Gaussian order 2, mexican hat or marr
    dog2 = ricker


class WaveletAnalysis(object):
    def __init__(self, x, dt=1):
        """Arguments:
            x - 1 dimensional input signal
            dt - sample spacing
        """


