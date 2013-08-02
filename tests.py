from nose.tools import *
import numpy.testing as npt

import numpy as np
import scipy.signal

import wavelets
from wavelets import Wavelets
from wavelets import WaveletAnalysis


N = 1000
x = np.random.random(N)

wa = WaveletAnalysis(x)

def test_N():
    assert_equal(N, wa.N)

def test_compare_cwt():
    """Compare the output of Scipy's cwt (using direct convolution)
    and my cwt (using fft convolution).
    """
    cwt = scipy.signal.cwt
    fft_cwt = wavelets.fft_cwt

    data = np.random.random(2000)
    wave_anal = WaveletAnalysis(data, wavelet='ricker')
    widths = wave_anal.scales[::-1]

    morlet = scipy.signal.morlet

    cwt = cwt(data, morlet, widths)
    fft_cwt = fft_cwt(data, morlet, widths)

    npt.assert_array_almost_equal(cwt, fft_cwt, decimal=13)
