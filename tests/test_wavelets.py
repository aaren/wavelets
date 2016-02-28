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


def test_Morlet():
    """Compare against \Psi_0(0) in Table 2 of TC98.

    Value at frequency = 0 should be 0.
    """
    npt.assert_almost_equal(wavelets.Morlet()(0), np.pi ** -.25, 3)
    npt.assert_almost_equal(wavelets.Morlet().frequency(0), 0, 6)


def test_Paul():
    """Compare against \Psi_0(0) in Table 2 of TC98.

    Value at frequency = 0 should be 0.
    """
    npt.assert_almost_equal(wavelets.Paul(m=4)(0), 1.079, 3)
    npt.assert_almost_equal(wavelets.Paul(m=4).frequency(0), 0, 6)


def test_DOG():
    """Compare against \Psi_0(0) in Table 2 of TC98.

    Value at frequency = 0 should be 0.
    """
    npt.assert_almost_equal(wavelets.DOG(m=2)(0), 0.867, 3)
    npt.assert_almost_equal(wavelets.DOG(m=6)(0), 0.884, 3)
    npt.assert_almost_equal(wavelets.DOG(m=2).frequency(0), 0, 6)
    npt.assert_almost_equal(wavelets.DOG(m=6).frequency(0), 0, 6)


test_data = np.loadtxt('tests/nino3data.asc', skiprows=3)

nino_time = test_data[:, 0]
nino_dt = np.diff(nino_time).mean()
anomaly_sst = test_data[:, 2]

wa = WaveletAnalysis(anomaly_sst, time=nino_time, dt=nino_dt)


def test_N():
    assert_equal(anomaly_sst.size, wa.N)


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


def compare_time_freq(N=2000):
    """Make sure that time and frequency based computation give the
    same result.
    """
    assert(wa.frequency is True)
    wavelet_time = wa.wavelet_transform

    wa.frequency = False
    assert(wa.frequency is False)
    frequency = wa.wavelet_transform

    npt.assert_array_almost_equal(wavelet_time, frequency, decimal=13)


def test_Cd():
    """default wavelet is morlet. Terrence and Compo calculate C_d
    for this of 0.776."""
    assert_almost_equal(wa.C_d, 0.776, places=2)


def test_var_time():
    """The wavelet transform conserves total energy, i.e. variance.

    The variance of the data should be the same as the variance of
    the wavelet.

    Check that they are within 1%% for the time representation.

    N.B. the performance of this test does depend on the input data.
    If e.g. np.random.random is used for the input, the variance
    difference is larger.
    """
    rdiff = 1 - wa.data_variance / wa.wavelet_variance
    assert_less(rdiff, 0.01)


def test_var_freq():
    """The wavelet transform conserves total energy, i.e. variance.

    The variance of the data should be the same as the variance of
    the wavelet.

    Check that they are within 1%% for the frequency representation.

    N.B. the performance of this test does depend on the input data.
    If e.g. np.random.random is used for the input, the variance
    difference is larger.
    """
    wa = WaveletAnalysis(anomaly_sst, frequency=True)
    rdiff = 1 - wa.data_variance / wa.wavelet_variance
    assert_less(rdiff, 0.01)


def test_reconstruction_time():
    """In principle one can reconstruct the input data from the
    wavelet transform.

    Check within 10% when computing with time representation of
    wavelet.
    """
    rdata = wa.reconstruction()

    err = wa.data - rdata
    assert(np.abs(err.mean()) < 0.02)
    # what does this mean?
    # assert(err.std() < 0.05)


def test_reconstruction_freq():
    """In principle one can reconstruct the input data from the
    wavelet transform.

    Check within 10% when computing with frequency representation of
    wavelet.
    """
    wa = WaveletAnalysis(anomaly_sst, frequency=True)
    rdata = wa.reconstruction()

    err = wa.data - rdata
    assert(np.abs(err.mean()) < 0.02)
    # what does this mean?
    # assert(err.std() < 0.05)


def test_power_bias():
    """See if the global wavelet spectrum is biased or not.

    Wavelet transform a signal of 3 distinct Fourier frequencies.

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

    ax.set_title('Nino 3 SST wavelet power spectrum')

    t, s = wa.time, wa.scales

    # plot the wavelet power
    T, S = np.meshgrid(t, s)
    ax.contourf(T, S, wa.wavelet_power, 256)

    ax.set_ylabel('scale (years)')
    ax.set_xlabel('year')
    ax.set_yscale('log')
    ax.grid(True)

    # put the ticks at powers of 2 in the scale
    ticks = np.unique(2 ** np.floor(np.log2(s)))[1:]
    ax.yaxis.set_ticks(ticks)
    ax.yaxis.set_ticklabels(ticks.astype(str))
    ax.set_ylim(64, 0.5)

    # second y scale with equivalent Fourier periods to scales
    # except with the ticks at the powers of 2
    ax_fourier = ax.twinx()
    ax_fourier.set_yscale('log')
    # match the Fourier ticks to the scale ticks
    ax_fourier.set_yticks(ticks)
    ax_fourier.set_yticklabels(ticks.astype(str))
    ax_fourier.set_ylabel('fourier period (years)')
    fourier_lim = [wa.fourier_period(i) for i in ax.get_ylim()]
    ax_fourier.set_ylim(fourier_lim)

    # shade the region between the edge and coi
    C, S = wa.coi
    ax.fill_between(x=C, y1=S, y2=s.max(), color='gray', alpha=0.3)
    ax.set_xlim(t.min(), t.max())

    fig.savefig('tests/test_coi.png')

def test_fourier_frequencies():
    # Just some signal, no special meaning
    dt = .1
    x = np.arange(5000) * dt
    signal = np.cos(1. * x) + np.cos(2. * x) + np.cos(3. * x)

    wa = WaveletAnalysis(signal, dt=dt,
                         wavelet=wavelets.Morlet(), unbias=False)
    # Set frequencies and check if they match when retrieving them again
    frequencies = np.linspace(1., 100., 100)
    wa.fourier_frequencies = frequencies
    npt.assert_array_almost_equal(wa.fourier_frequencies, frequencies)
    # Check periods
    npt.assert_array_almost_equal(wa.fourier_periods, 1. / frequencies)

    # Set periods and re-check
    wa.fourier_periods = 1. / frequencies
    npt.assert_array_almost_equal(wa.fourier_frequencies, frequencies)
    npt.assert_array_almost_equal(wa.fourier_periods, 1. / frequencies)


def test_multi_dim():
    data = np.random.random((10, 100))
    wa = WaveletAnalysis(data, frequency=True)
    ns = len(wa.scales)
    assert(wa.wavelet_transform.shape == (ns, 10, 100))

    wan = WaveletAnalysis(data[0], frequency=True)
    assert(wan.wavelet_transform.shape == (ns, 100))

    npt.assert_array_almost_equal(wa.wavelet_transform[:, 0, :],
                                  wan.wavelet_transform[:, :],
                                  decimal=13)


def test_multi_dim_axis():
    data = np.random.random((10, 100))
    wa = WaveletAnalysis(data, frequency=True, axis=0)
    ns = len(wa.scales)
    print (wa.wavelet_transform.shape)
    print (ns)
    assert(wa.wavelet_transform.shape == (ns, 10, 100))

    wan = WaveletAnalysis(data[:, 0], frequency=True)
    print (wan.wavelet_transform.shape)
    assert(wan.wavelet_transform.shape == (ns, 10))

    npt.assert_array_almost_equal(wa.wavelet_transform[:, :, 0],
                                  wan.wavelet_transform[:, :],
                                  decimal=13)


def test_multi_dim_axis_nd():
    data = np.random.random((3, 4, 100, 5))
    wa = WaveletAnalysis(data, frequency=True, axis=2)
    ns = len(wa.scales)
    print (wa.wavelet_transform.shape)
    print (ns)
    assert(wa.wavelet_transform.shape == (ns, 3, 4, 100, 5))

    wan = WaveletAnalysis(data[0, 0, :, 0], frequency=True)
    print (wan.wavelet_transform.shape)
    assert(wan.wavelet_transform.shape == (ns, 100))

    npt.assert_array_almost_equal(wa.wavelet_transform[:, 0, 0, :, 0],
                                  wan.wavelet_transform[:, :],
                                  decimal=13)


def test_multi_dim_axis_nd_time():
    data = np.random.random((3, 4, 100, 5))
    wa = WaveletAnalysis(data, frequency=False, axis=2)
    ns = len(wa.scales)
    print (wa.wavelet_transform.shape)
    print (ns)
    assert(wa.wavelet_transform.shape == (ns, 3, 4, 100, 5))

    wan = WaveletAnalysis(data[0, 0, :, 0], frequency=False)
    print (wan.wavelet_transform.shape)
    assert(wan.wavelet_transform.shape == (ns, 100))

    npt.assert_array_almost_equal(wa.wavelet_transform[:, 0, 0, :, 0],
                                  wan.wavelet_transform[:, :],
                                  decimal=13)


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
    morlet = wavelets.Morlet().time
    s = 1
    T = np.linspace(-5 * s, 5 * s, 200)
    Y = morlet(T, s=s)

    f, a = plt.subplots()
    a.plot(T, Y, 'k')
    a.set_title('Morlet wavelet')
    a.set_xlabel('t / s')

    f.savefig('morlet.png')
