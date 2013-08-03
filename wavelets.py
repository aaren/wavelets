from __future__ import division

from functools import wraps

import numpy as np
import scipy
import scipy.signal
import scipy.optimize


def fft_cwt(data, wavelet, widths):
    """Continuous wavelet transform using the fourier transform
    convolution as used in Terrence and Compo.

    (as opposed to the direct convolution method used by
    scipy.signal.cwt)

    *This method is over 10x faster than the scipy default.*

    Performs a continuous wavelet transform on `data`,
    using the `wavelet` function. A CWT performs a convolution
    with `data` using the `wavelet` function, which is characterized
    by a width parameter and length parameter.

    Parameters
    ----------
    data : (N,) ndarray
        data on which to perform the transform.
    wavelet : function
        Wavelet function, which should take 2 arguments.
        The first argument is the number of points that the returned vector
        will have (len(wavelet(width,length)) == length).
        The second is a width parameter, defining the size of the wavelet
        (e.g. standard deviation of a gaussian). See `ricker`, which
        satisfies these requirements.
    widths : (M,) sequence
        Widths to use for transform.

    Returns
    -------
    cwt: (M, N) ndarray
        Will have shape of (len(data), len(widths)).

    """
    output = np.zeros((len(widths), len(data)))
    for ind, width in enumerate(widths):
        wavelet_data = wavelet(min(10 * width, len(data)), width)
        output[ind, :] = scipy.signal.fftconvolve(data, wavelet_data,
                                                            mode='same')
    return output

@wraps(scipy.signal.morlet)
def morlet(M, s=1.0, w0=5.0, complete=False):
    """Wrap scipy morlet function with arguments in the right order. """
    return scipy.signal.morlet(M, w0, s, complete)

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

    ### Equivalent Fourier period (S3.h) ###

    The peak wavelet response does not necessarily occur at 1 / s.

    If we wish to compare wavelet spectra at different scales with
    each other and with fourier modes, we need a common set of
    units.

    The equivalent fourier period is defined as where the wavelet
    power spectrum reaches its maximum and can be found analytically.
    """
    # morlet wavelet
    morlet = staticmethod(scipy.signal.morlet)
    # ricker wavelet
    ricker = staticmethod(scipy.signal.ricker)
    # aka Derivitive Of Gaussian order 2, mexican hat or marr
    dog2 = ricker

    # Fourier wavelengths
    def fourier_period_morlet(s, w0=5):
        """Equivalent fourier period of morlet"""
        return 4 * np.pi * s / (w0 + (2 + w0 ** 2) ** .5)

    def fourier_period_dog2(s):
        """Equivalent fourier period of ricker / dog2 / mexican hat."""
        return 2 * np.pi * s / (5 / 2) ** .5

    morlet.fourier_period = fourier_period_morlet
    ricker.fourier_period = fourier_period_dog2


class WaveletAnalysis(object):
    def __init__(self, x, dt=1, dj=0.125, wavelet='morlet'):
        """Arguments:
            x - 1 dimensional input signal
            dt - sample spacing
            dj - scale resolution
            wavelet - wavelet function to use
        """
        self.x = x
        self.N = len(x)
        self.dt = dt
        self.dj = dj
        self.wavelet = getattr(Wavelets, wavelet)
        # which continuous wavelet transform to use
        self.cwt = fft_cwt

    @property
    def fourier_period(self):
        """Return a function that calculates the equivalent fourier
        period as a function of scale.
        """
        return getattr(self.wavelet, 'fourier_period')

    @property
    def s0(self):
        """Find the smallest resolvable scale by finding where the
        equivalent fourier period is equal to 2 * dt. For a Morlet
        wavelet, this is roughly 1.
        """
        def f(s):
            return self.fourier_period(s) - 2 * self.dt
        return scipy.optimize.fsolve(f, 1)[0]

    @property
    def scales(self):
        """Form a set of scales to use in the wavelet transform.

        For non-orthogonal wavelet analysis, one can use an
        arbitrary set of scales.

        It is convenient to write the scales as fractional powers of
        two:

            s_j = s_0 * 2 ** (j * dj), j = 0, 1, ..., J

            J = (1 / dj) * log2(N * dt / s_0)

        s0 - smallest resolvable scale
        J - largest scale

        choose s0 so that the equivalent Fourier period is 2 * dt.

        The choice of dj depends on the width in spectral space of
        the wavelet function. For the morlet, dj=0.5 is the largest
        that still adequately samples scale. Smaller dj gives finer
        scale resolution.
        """
        # resolution
        dj = self.dj
        # smallest resolvable scale, chosen so that the equivalent
        # fourier period is approximately 2dt
        s0 = self.s0

        # Largest scale
        J = int((1 / dj) * np.log2(self.N * self.dt / s0))

        sj = s0 * 2 ** (dj * np.arange(0, J + 1))
        return sj

    def w(self, k):
        """Angular frequency as a function of wavenumber.

        See eq5 of TC.
        """
        res = 2 * np.pi * k / (self.N * self.dt)
        if k <= N / 2:
            return res
        elif k > N / 2:
            return -res

    @property
    def wavelet_transform(self):
        """Calculate the wavelet transform."""
        return self.cwt(self.x, self.wavelet, self.widths)

    def reconstruction(self):
        """Reconstruct the original signal from the wavelet
        transform. See S3.i.

        For non-orthogonal wavelet functions, it is possible to
        reconstruct the original time series using an arbitrary
        wavelet function. The simplest is to use a delta function.

        The reconstructed time series is found as the sum of the
        real part of the wavelet transform over all scales,

        x_n = (dj * dt^(1/2)) / (C_d * Y_0(0)) \
                * Sum_(j=0)^J { Re(W_n(s_j)) / s_j^(1/2) }

        where the factor C_d comes from the recontruction of a delta
        function from its wavelet transform using the wavelet
        function Y_0. This C_d is a constant for each wavelet
        function.
        """
        dj = self.dj
        dt = self.dt
        C_d = self.C_d
        # TODO: is wavelet centred properly?
        Y_0 = self.wavelet
        # TODO: write the wavelet transform
        W_n = self.wavelet_transform
        s = np.expand_dims(self.scales, 1)

        real_sum = np.sum(W_n.real / s ** .5, axis=0)
        x_n = real_sum * (dj * dt ** .5 / (C_d * Y_0(0)))
        return x_n

    def C_d(self):
        """Constant used in reconstruction of data from delta
        wavelet function. See self.reconstruction and S3.i.

        To derive C_d for a new wavelet function, first assume a
        time series with a delta function at time n=0, given by x_n
        = d_n0. This time series has a Fourier transform x_k = 1 /
        N, constant over k.

        Substituting x_k into eq4 at n=0, the wavelet transform
        becomes

            W_d(s) = (1 / N) Sum[k=0][N-1] { Y'*(s, w_k) }

        The reconstruction then gives

            C_d = (dj * dt^(1/2)) / Y_0(0) \
                    * Sum_(j=0)^J { Re(W_d(s_j)) / s_j^(1/2) }

        C_d is scale independent and a constant for each wavelet
        function.
        """
        dj = self.dj
        dt = self.dt
        C_d = 1
        # TODO: is wavelet centred properly?
        Y_0 = self.wavelet
        # TODO: write the wavelet transform
        W_d = self.wavelet_transform_delta
        s = np.expand_dims(self.scales, 1)

        real_sum = np.sum(W_d.real / s ** .5, axis=0)
        C_d = real_sum * (dj * dt ** .5 / (C_d * Y_0(0)))
        return C_d

    @property
    def wavelet_transform_delta(self):
        """Calculate the delta wavelet transform.

        Returns an array of the transform computed over the scales.
        """
        N = self.N
        # wavelet as function of (s, w_k)
        Y_ = self.wavelet_freq
        k = np.arange(N)
        s = self.scales
        K, S = np.meshgrid(k, s)

        # compute Y_ over all s, w_k and sum over k
        W_d = (1 / N) * np.sum(Y_(S, self.w_k(K)), axis=0)

        return W_d







# TODO: cone of influence
# TODO: does scipy cwt pad the array with zeroes up to next power of two in length?
# uses signaltools.convolve, which uses correlate
# for modes other than 'valid', correlate zero pads but only to
# match the arrays. correlate calls sigtools._correlateND which is a
# C object

# It seems that scipy.cwt is using the convolution method of
# calculating the wavelet transform, rather than the fourier
# transform method used in Terrence and Compo

# TODO: reconstruction (S3.i)
# TODO: derive C_d for given wavelet
# TODO: derive Y'(s,w) for given Y(t) (Y is wavelet)

# TODO: scipy morlet implementation is incorrect, need to swap
# arguments around for feeding to cwt (make it like ricker)
