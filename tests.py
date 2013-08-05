from nose.tools import *
import numpy.testing as npt

import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

import wavelets
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
    wave_anal = WaveletAnalysis(data, wavelet=wavelets.ricker)
    widths = wave_anal.scales()[::-1]

    morlet = scipy.signal.morlet

    cwt = cwt(data, morlet, widths)
    fft_cwt = fft_cwt(data, morlet, widths)

    npt.assert_array_almost_equal(cwt, fft_cwt, decimal=13)

def compare_morlet(N=2000):
    """Compare scipy morlet with my morlet (same, but correct
    argument order).
    """
    data = np.random.random(N)
    wave_anal = WaveletAnalysis(data, wavelet='ricker')
    scales = wave_anal.scales[::-1]

    cwt = wavelets.fft_cwt
    cwt_sp = cwt(data, scipy.signal.morlet, scales)
    cwt_me = cwt(data, wavelets.morlet, scales)
    cwt_ri = cwt(data, scipy.signal.ricker, scales)

    t = np.indices(data.shape)
    T, S = np.meshgrid(t, scales)

    fig, ax = plt.subplots(nrows=3)

    ax[0].set_title('Scipy morlet')
    ax[0].contourf(T, S, cwt_sp, 100)

    ax[1].set_title('My morlet')
    ax[1].contourf(T, S, cwt_me, 100)

    ax[2].set_title('Scipy Ricker')
    ax[2].contourf(T, S, cwt_ri, 100)

    fig.tight_layout()

    return fig


def test_Cd():
    """default wavelet is morlet. Terrence and Compo calculate C_d
    for this of 0.776."""
    assert_almost_equal(wa.C_d, 0.776, places=2)


def test_var():
    """The wavelet transform conserves total energy, i.e. variance.

    The variance of the data should be the same as the variance of
    the wavelet.

    Check that they are within 10%
    """
    rdiff = 1 - wa.data_variance/ wa.wavelet_variance
    assert_less(rdiff, 0.1)


def test_reconstruction():
    """In principle one can reconstruct the input data from the
    wavelet transform.

    Check within 10%.
    """
    rdata = wa.reconstruction()
    npt.assert_array_almost_equal(wa.data, rdata, decimal=1)

    err = wa.data - rdata
    assert(np.abs(err.mean()) < 0.05)
    assert(err.std() < 0.05)
