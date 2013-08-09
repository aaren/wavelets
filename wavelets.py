from __future__ import division

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

        The wavelet function should not be normalised by 1/sqrt(s),
        i.e. it should have unit energy.

    widths : (M,) sequence
        Widths to use for transform.
    dt: float
        sample spacing. defaults to 1 (data sample units).

    Returns
    -------
    cwt: (M, N) ndarray
        Will have shape of (len(data), len(widths)).

    """
    # wavelets can be complex so output is complex
    output = np.zeros((len(widths), len(data)), dtype=np.complex)
    for ind, width in enumerate(widths):
        # number of points needed to capture wavelet
        M = 10 * width
        t = np.arange((-M + 1) / 2., (M + 1) / 2.)
        wavelet_data = (1 / width) ** .5 * wavelet(t, width)
        output[ind, :] = scipy.signal.fftconvolve(data,
                                                  wavelet_data,
                                                  mode='same')
    return output


# TODO: reimplement cwt to match mathematical ufunc wavelet functions
# TODO: use ifft for cwt, rather than fftconvolve
class Morlet(object):
    def __init__(self, w0=6):
        self.w0 = w0

    def __call__(self, *args, **kwargs):
        return self.time_rep(*args, **kwargs)

    def time_rep(self, t, s=1.0, complete=True):
        """
        Complex Morlet wavelet, centred at zero.

        Parameters
        ----------
        t : float
            Time. If s is not specified, this can be used as the
            non-dimensional time t/s.
        s : float
            Scaling factor. Default is 1.
        complete : bool
            Whether to use the complete or the standard version.

        Returns
        -------
        complex: value of the morlet wavelet at the given time

        See Also
        --------
        scipy.signal.gausspulse

        Notes
        -----
        The standard version::

            pi**-0.25 * exp(1j*w*x) * exp(-0.5*(x**2))

        This commonly used wavelet is often referred to simply as the
        Morlet wavelet.  Note that this simplified version can cause
        admissibility problems at low values of w.

        The complete version::

            pi**-0.25 * (exp(1j*w*x) - exp(-0.5*(w**2))) * exp(-0.5*(x**2))

        The complete version of the Morlet wavelet, with a correction
        term to improve admissibility. For w greater than 5, the
        correction term is negligible.

        Note that the energy of the return wavelet is not normalised
        according to s.

        The fundamental frequency of this wavelet in Hz is given
        by ``f = 2*s*w*r / M`` where r is the sampling rate.

        """
        w = self.w0

        x = t / s

        output = np.exp(1j * w * x)

        if complete:
            output -= np.exp(-0.5 * (w ** 2))

        output *= np.exp(-0.5 * (x ** 2)) * np.pi ** (-0.25)

        return output

    # Fourier wavelengths
    def fourier_period(self, s):
        """Equivalent fourier period of morlet"""
        return 4 * np.pi * s / (self.w0 + (2 + self.w0 ** 2) ** .5)

    # Frequency representation
    def frequency_rep(self, s, w):
        """Frequency representation of morlet.

        s - scale
        w - angular frequency
        """
        # heaviside mock
        Hw = 0.5 * (np.sign(w) + 1)
        return np.pi ** .25 * Hw * np.exp(-(s * w - self.w0) ** 2 / 2)


class Ricker(object):
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.time_rep(*args, **kwargs)

    def time_rep(self, t, s=1.0):
        """
        Return a Ricker wavelet, also known as the "Mexican hat wavelet".

        It models the function:

            ``A (1 - x^2/s^2) exp(-t^2/s^2)``,

        where ``A = 2/sqrt(3)pi^1/3``.

        Note that the energy of the return wavelet is not normalised
        according to s.

        Parameters
        ----------
        t : float
            Time. If s is not specified, this can be used as the
            non-dimensional time t/s.
        s : scalar
            Width parameter of the wavelet.

        Returns
        -------
        float : value of the ricker wavelet at the given time

        """
        x = t / s

        # this prefactor comes from the gamma function in
        # Derivative of Gaussian.
        A = np.pi ** -0.25 * np.sqrt(4 / 3)

        output = A * (1 - x ** 2) * np.exp(-x ** 2 / 2)

        return output

    def fourier_period(self, s):
        """Equivalent fourier period of ricker / dog2 / mexican hat."""
        return 2 * np.pi * s / (5 / 2) ** .5

    def frequency_rep(self, s, w):
        """Frequency representation of ricker.

        s - scale
        w - angular frequency
        """
        A = np.pi ** -0.25 * np.sqrt(4 / 3)
        return A * (s * w) ** 2 * np.exp(-(s * w) ** 2 / 2)


class WaveletAnalysis(object):
    """
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
    def __init__(self, data=np.random.random(1000), dt=1, dj=0.125,
                 wavelet=Morlet()):
        """Arguments:
            x - 1 dimensional input signal
            dt - sample spacing
            dj - scale resolution
            wavelet - wavelet function to use
            TODO: allow override s0
        """
        self.data = data
        self.anomaly_data = self.data - self.data.mean()
        self.N = len(data)
        self.data_variance = self.data.var()
        self.dt = dt
        self.dj = dj
        self.wavelet = wavelet
        # which continuous wavelet transform to use
        self.cwt = fft_cwt

    @property
    def fourier_period(self):
        """Return a function that calculates the equivalent fourier
        period as a function of scale.
        """
        return getattr(self.wavelet, 'fourier_period')

    @property
    def fourier_periods(self):
        """Return the equivalent fourier periods for the scales used."""
        return self.fourier_period(self.scales())

    def s0(self, dt=None):
        """Find the smallest resolvable scale by finding where the
        equivalent fourier period is equal to 2 * dt. For a Morlet
        wavelet, this is roughly 1.
        """
        dt = dt or self.dt

        def f(s):
            return self.fourier_period(s) - 2 * dt
        return scipy.optimize.fsolve(f, 1)[0]

    def scales(self, dt=None):
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
        dt = dt or self.dt
        # resolution
        dj = self.dj
        # smallest resolvable scale, chosen so that the equivalent
        # fourier period is approximately 2dt
        s0 = self.s0(dt=dt)

        # Largest scale
        J = int((1 / dj) * np.log2(self.N * dt / s0))

        sj = s0 * 2 ** (dj * np.arange(0, J + 1))
        return sj

    # TODO: use np.frompyfunc on this
    def w_k(self, k=None, dt=None):
        """Angular frequency as a function of fourier index.

        If no k, returns an array of all the angular frequencies
        calculated using the length of the data.

        See eq5 of TC.
        """
        dt = dt or self.dt
        N = self.N
        a = 2 * np.pi / (N * dt)
        if k is None:
            k = np.arange(N)
            w_k = np.arange(N) * a
            w_k[np.where(k > N // 2)] *= -1
        elif type(k) is np.ndarray:
            w_k = a * k
            w_k[np.where(k > N // 2)] *= -1
        else:
            w_k = a * k
            if k <= N // 2:
                pass
            elif k > N // 2:
                w_k *= -1
        return w_k

    @property
    def wavelet_transform(self):
        """Calculate the wavelet transform."""
        return self.cwt(self.anomaly_data, self.wavelet.time_rep,
                        self.scales(dt=1))

    @property
    def wavelet_power(self):
        """Calculate the wavelet power spectrum, using the bias
        correction factor introduced by Liu et al. 2007, which is to
        divide by the scale.
        """
        s = np.expand_dims(self.scales(), 1)
        return np.abs(self.wavelet_transform) ** 2 / s

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
        Y_00 = self.wavelet.time_rep(0)
        W_n = self.wavelet_transform
        # TODO: allow specification of scales
        s = np.expand_dims(self.scales(), 1)

        real_sum = np.sum(W_n.real / s ** .5, axis=0)
        x_n = real_sum * (dj * dt ** .5 / (C_d * Y_00))

        # add the mean back on (x_n is anomaly time series)
        x_n += self.data.mean()

        return x_n

    @property
    def global_wavelet_spectrum(self):
        mean_power = np.mean(self.wavelet_power, axis=1)
        var = self.data_variance
        return mean_power / var

    @property
    def C_d(self):
        """Constant used in reconstruction of data from delta
        wavelet function. See self.reconstruction and S3.i.

        To derive C_d for a new wavelet function, first assume a
        time series with a delta function at time n=0, given by x_n
        = d_n0. This time series has a Fourier transform x_k = 1 /
        N, constant over k.

        Substituting x_k into eq4 at n=0 (the peak of the delta
        function), the wavelet transform becomes

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
        W_d = self.wavelet_transform_delta
        s = np.expand_dims(self.scales(), 1)
        s = self.scales()
        # value of the wavelet function at t=0
        Y_00 = self.wavelet.time_rep(0)

        real_sum = np.sum(W_d.real / s ** .5)
        C_d = real_sum * (dj * dt ** .5 / (C_d * Y_00))
        # TODO: coming out as 0.26 for morlet
        return 0.776

    @property
    def wavelet_transform_delta(self):
        """Calculate the delta wavelet transform.

        Returns an array of the transform computed over the scales.
        """
        N = self.N
        # wavelet as function of (s, w_k)
        Y_ = self.wavelet.frequency_rep
        k = np.arange(N)
        s = self.scales()
        K, S = np.meshgrid(k, s)

        # compute Y_ over all s, w_k and sum over k
        W_d = (1 / N) * np.sum(Y_(S, self.w_k(K)), axis=1)

        # N.B This W_d is 1D

        return W_d

    @property
    def wavelet_variance(self):
        """Equivalent of Parseval's theorem for wavelets, S3.i.

        The wavelet transform conserves total energy, i.e. variance.

        Returns the variance of the input data.
        """
        dj = self.dj
        dt = self.dt
        C_d = self.C_d
        N = self.N

        A = dj * dt / (C_d * N)

        var = A * np.sum(self.wavelet_power)

        return var


# TODO: cone of influence

# TODO: derive C_d for given wavelet
