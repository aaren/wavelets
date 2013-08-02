from functools import wraps

import numpy as np
import scipy.signal


class Wavelets(object):
    """Container for various wavelet basis functions.

    Sx.y are references to section x.y in Terrence and Compo,
    A Practical Guide to Wavelet Analysis (BAMS, 1998)


    ### Wavelet function requirements (S3.b) ###

    To be admissible as a wavelet, a function must:

    - have zero mean
    - be localised in both time and frequency space

    These functions are a function of a dimensionless time
    parameter.

    ### Function selection considerations (S3.e) ###

    #### Complex / Real

    A *complex* wavelet function will return information about both
    amplitude and phase and is better adapted for capturing
    *osillatory behaviour*.

    A *real* wavelet function returns only a single component and
    can be used to isolate *peaks or discontinuities*.

    ### Width

    Define the width of a wavelet as the e-folding time of the
    wavelet amplitude.

    The resolution of the wavelet function is determined by the
    balance between the width in real and fourier space.

    A narrow function in time will have good time resolution but
    poor frequency resolution and vice versa.

    ### Shape

    The wavelet function should represent the type of features
    present in the time series.

    For time series with sharp jumps or steps, choose a boxcar-like
    function such as Harr; while for smoothly varying time series,
    choose something like a damped cosine.

    The choice of wavelet function is not critical if one is only
    qualitatively interested in the wavelet power spectrum.
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


