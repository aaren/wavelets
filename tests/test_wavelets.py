from __future__ import division

from nose.tools import *
import numpy.testing as npt

import numpy as np
import scipy.signal
from scipy.io import wavfile
import matplotlib.pyplot as plt

import wavelets
from wavelets import WaveletAnalysis

__all__ = ['test_N', 'compare_cwt', 'compare_morlet', 'test_Cd',
           'test_var_time', 'test_var_freq', 'test_reconstruction_time',
           'test_reconstruction_freq', 'test_power_bias', 'test_plot_coi',
           ]


N = 1000
x = np.random.random(N)

wa = WaveletAnalysis(x)


def test_N():
    assert_equal(N, wa.N)


def compare_cwt():
    """Compare the output of Scipy's cwt (using direct convolution)
    and my cwt (using fft convolution).
    """
    cwt = scipy.signal.cwt
    fft_cwt = wavelets.cwt

    data = np.random.random(2000)
    wave_anal = WaveletAnalysis(data, wavelet=wavelets.Ricker())
    widths = wave_anal.scales[::-1]

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

    cwt = wavelets.cwt
    cwt_sp = cwt(data, scipy.signal.morlet, scales)
    cwt_me = cwt(data, wavelets.Morlet(), scales)
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


def test_var_time():
    """The wavelet transform conserves total energy, i.e. variance.

    The variance of the data should be the same as the variance of
    the wavelet.

    Check that they are within 10%% for the time representation.
    """
    rdiff = 1 - wa.data_variance / wa.wavelet_variance
    assert_less(rdiff, 0.1)


def test_var_freq():
    """The wavelet transform conserves total energy, i.e. variance.

    The variance of the data should be the same as the variance of
    the wavelet.

    Check that they are within 10%% for the frequency representation.
    """
    wa = WaveletAnalysis(x, compute_with_freq=True)
    rdiff = 1 - wa.data_variance / wa.wavelet_variance
    assert_less(rdiff, 0.1)


def test_reconstruction_time():
    """In principle one can reconstruct the input data from the
    wavelet transform.

    Check within 10% when computing with time representation of
    wavelet.
    """
    rdata = wa.reconstruction()
    npt.assert_array_almost_equal(wa.data, rdata, decimal=1)

    err = wa.data - rdata
    assert(np.abs(err.mean()) < 0.05)
    assert(err.std() < 0.05)


def test_reconstruction_freq():
    """In principle one can reconstruct the input data from the
    wavelet transform.

    Check within 10% when computing with frequency representation of
    wavelet.
    """
    wa = WaveletAnalysis(x, compute_with_freq=True)
    rdata = wa.reconstruction()
    npt.assert_array_almost_equal(wa.data, rdata, decimal=1)

    err = wa.data - rdata
    assert(np.abs(err.mean()) < 0.05)
    assert(err.std() < 0.05)


def test_power_bias():
    """See if the global wavelet spectrum is biased or not.

    Wavelet transform a signal of 3 distinct fourier frequencies.

    The power spectrum should contain peaks at the frequencies, all
    of which should be the same height.
    """
    dt = 0.1
    x = np.arange(5000) * dt

    T1 = 20 * dt
    T2 = 100 * dt
    T3 = 500 * dt

    w1 = 2 * np.pi / T1
    w2 = 2 * np.pi / T2
    w3 = 2 * np.pi / T3

    signal = np.cos(w1 * x) + np.cos(w2 * x) + np.cos(w3 * x)

    wa = WaveletAnalysis(signal, dt=dt,
                         wavelet=wavelets.Morlet(), unbias=False)

    power_biased = wa.global_wavelet_spectrum
    wa.unbias = True
    power = wa.global_wavelet_spectrum
    wa.mask_coi = True
    power_coi = wa.global_wavelet_spectrum

    freqs = wa.fourier_periods

    fig, ax = plt.subplots(nrows=2)

    ax_transform = ax[0]
    fig_info = (r"Wavelet transform of "
                r"$cos(2 \pi / {T1}) + cos(2 \pi / {T2}) + cos(2 \pi / {T3})$")
    ax_transform.set_title(fig_info.format(T1=T1, T2=T2, T3=T3))
    X, Y = np.meshgrid(wa.time, wa.fourier_periods)
    ax_transform.set_xlabel('time')
    ax_transform.set_ylabel('fourier period')
    ax_transform.set_ylim(10 * dt, 1000 * dt)
    ax_transform.set_yscale('log')
    ax_transform.contourf(X, Y, wa.wavelet_power, 100)

    # shade the region between the edge and coi
    C, S = wa.coi
    F = wa.fourier_period(S)
    f_max = F.max()
    ax_transform.fill_between(x=C, y1=F, y2=f_max, color='gray', alpha=0.3)

    ax_power = ax[1]
    ax_power.set_title('Global wavelet spectrum '
                       '(estimator for power spectrum)')
    ax_power.plot(freqs, power, 'k', label=r'unbiased all domain')
    ax_power.plot(freqs, power_coi, 'g', label=r'unbiased coi only')
    ax_power.set_xscale('log')
    ax_power.set_xlim(10 * dt, wa.time.max())
    ax_power.set_xlabel('fourier period')
    ax_power.set_ylabel(r'power / $\sigma^2$  (bias corrected)')

    ax_power_bi = ax_power.twinx()
    ax_power_bi.plot(freqs, power_biased, 'r', label='biased all domain')
    ax_power_bi.set_xlim(10 * dt, wa.time.max())
    ax_power_bi.set_ylabel(r'power / $\sigma^2$  (bias uncorrected)')
    ax_power_bi.set_yticklabels(ax_power_bi.get_yticks(), color='r')

    label = "T={0}"
    for T in (T1, T2, T3):
        ax_power.axvline(T)
        ax_power.annotate(label.format(T), (T, 1))

    ax_power.legend(fontsize='x-small', loc='lower right')
    ax_power_bi.legend(fontsize='x-small', loc='upper right')

    fig.tight_layout()
    fig.savefig('tests/test_power_bias.png')

    return fig


def test_plot_coi():
    """Can we plot the Cone of Influence?."""
    fig, ax = plt.subplots()

    ax.set_title('Wavelet power spectrum with Cone of Influence')

    t, s = wa.time, wa.scales

    # plot the wavelet power
    T, S = np.meshgrid(t, s)
    ax.contourf(T, S, wa.wavelet_power, 100)

    ax.set_yscale('log')
    ax.set_ylabel('scale')
    ax.set_xlabel('time')

    # TODO: make a second re scaled y axis without plotting something.
    ax_fourier = ax.twinx()
    f = wa.fourier_periods
    T, F = np.meshgrid(t, f)
    ax_fourier.contourf(T, F, wa.wavelet_power, 100)
    ax_fourier.set_yscale('log')
    ax_fourier.set_ylabel('fourier period')

    # shade the region between the edge and coi
    C, S = wa.coi
    S_max = wa.scales.max()
    ax_fourier.fill_between(x=C, y1=S, y2=S_max, color='gray', alpha=0.3)
    ax_fourier.set_xlim(0, t.max())

    fig.savefig('tests/test_coi.png')


def analyse_song():
    """Compute the wavelet transform of a song."""
    fs, song = wavfile.read('alarma.wav')

    # select first part of one channel
    stride = 1
    # time step is inverse sample rate * stride
    dt = stride / fs
    # number of seconds of song to analyse
    t_s = 1
    n_s = fs * t_s

    # sub sample song on a single channel
    sub_song = song[:n_s:stride, 0]

    wa = WaveletAnalysis(sub_song, dt=dt)

    fig, ax = plt.subplots()
    T, F = np.meshgrid(wa.time, wa.fourier_periods)
    freqs = 1 / F
    ax.contourf(T, freqs, wa.wavelet_power, 100)
    ax.set_yscale('log')

    ax.set_ylabel('frequency (Hz)')
    ax.set_xlabel('time (s)')

    ax.set_ylim(100, 10000)

    fig.savefig('alarma_wavelet.png')


def plot_random_data():
    """Used for the screenshot in the README."""
    f, a = plt.subplots(frameon=False)
    a.contourf(wa.wavelet_power, 256)
    a.set_ylim(30, 0)
    a.set_axis_off()
    f.set_dpi(100)
    f.set_size_inches(8, 4)
    f.savefig('random_data.png')


def plot_morlet():
    """
    TODO: make a pretty plot of the morlet for the README
    """
    morlet = wavelets.Morlet().time_rep
    s = 1
    T = np.linspace(-5 * s, 5 * s, 200)
    Y = morlet(T, s=s)

    f, a = plt.subplots()
    a.plot(T, Y, 'k')
    a.set_title('Morlet wavelet')
    a.set_xlabel('t / s')

    f.savefig('morlet.png')
