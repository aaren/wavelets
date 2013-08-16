from __future__ import division

import numpy as np
import scipy
import scipy.signal
import scipy.optimize

__all__ = ['cwt', 'Morlet', 'Ricker', 'WaveletAnalysis']


def cwt(data, wavelet=None, widths=None, dt=1, wavelet_freq=False):
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
        Wavelet function in either time or frequency space, which
        should take 2 arguments. If the wavelet is frequency based,
        wavelet_freq must be set to True.

        The first parameter is time or frequency.

        The second is a width parameter, defining the size of the wavelet
        (e.g. standard deviation of a gaussian).

        The wavelet function, Y, should be such that
        Int[-inf][inf](|Y|^2) = 1

        It is then multiplied here by a normalisation factor,
        which gives it unit energy.

        In the time domain, the normalisation factor is

            (s / dt)

        In the frequency domain, the normalisation factor is

            (2 * pi * dt / s) ^ (1/2),

    widths : (M,) sequence
        Widths to use for transform.

    dt: float
        sample spacing. defaults to 1 (data sample units).

    wavelet_freq: boolean. Whether the wavelet function is one of
                  time or frequency. Default, False, is for a time
                  representation of the wavelet function.

    Returns
    -------
    cwt: (M, N) ndarray
        Will have shape of (len(data), len(widths)).

    """
    if widths is None:
        raise UserWarning('Have to specify some widths (scales)')

    if not wavelet:
        raise UserWarning('Have to specify a wavelet function')

    N = data.size
    # wavelets can be complex so output is complex
    output = np.zeros((len(widths), N), dtype=np.complex)

    if wavelet_freq:
        # compute in frequency
        # next highest power of two for padding
        pN = int(2 ** np.ceil(np.log2(N)))
        # N.B. padding in fft adds zeros to the *end* of the array,
        # not equally either end.
        fft_data = scipy.fft(data, n=pN)
        # frequencies
        w_k = np.fft.fftfreq(pN, d=dt) * 2 * np.pi
        for ind, width in enumerate(widths):
            # sample wavelet and normalise
            norm = (2 * np.pi * width / dt) ** .5
            wavelet_data = norm * wavelet(w_k, width)
            out = scipy.ifft(fft_data * wavelet_data.conj(), n=pN)
            # remove zero padding
            output[ind, :] = out[:N]

    elif not wavelet_freq:
        # compute in time
        for ind, width in enumerate(widths):
            # number of points needed to capture wavelet
            M = 10 * width / dt
            # times to use, centred at zero
            t = np.arange((-M + 1) / 2., (M + 1) / 2.) * dt
            # sample wavelet and normalise
            norm = (dt / width) ** .5
            wavelet_data = norm * wavelet(t, width)
            output[ind, :] = scipy.signal.fftconvolve(data,
                                                      wavelet_data,
                                                      mode='same')

    return output


class Morlet(object):
    def __init__(self, w0=6):
        """w0 is the nondimensional frequency constant. If this is
        set too low then the wavelet does not sample very well: a
        value over 5 should be ok, Terrence and Compo set it to 6.
        """
        self.w0 = w0
        if w0 == 6:
            # value of C_d from TC98
            self.C_d = 0.776

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
    def frequency_rep(self, w, s=1.0):
        """Frequency representation of morlet.

        s - scale
        w - angular frequency
        """
        x = w * s
        # heaviside mock
        Hw = 0.5 * (np.sign(x) + 1)
        return np.pi ** -.25 * Hw * np.exp((-(x - self.w0) ** 2) / 2)

    def coi(self, s):
        """The e folding time for the autocorrelation of wavelet
        power at each scale, i.e. the timescale over which an edge
        effect decays by a factor of 1/e^2.

        This can be worked out analytically by solving

            |Y_0(T)|^2 / |Y_0(0)|^2 = 1 / e^2
        """
        return 2 ** .5 * s


class Ricker(object):
    def __init__(self):
        # value of C_d from TC98
        self.C_d = 3.541

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

    def frequency_rep(self, w, s=1.0):
        """Frequency representation of ricker.

        s - scale
        w - angular frequency
        """
        A = np.pi ** -0.25 * np.sqrt(4 / 3)
        return A * (s * w) ** 2 * np.exp(-(s * w) ** 2 / 2)

    def coi(self, s):
        """The e folding time for the autocorrelation of wavelet
        power at each scale, i.e. the timescale over which an edge
        effect decays by a factor of 1/e^2.

        This can be worked out analytically by solving

            |Y_0(T)|^2 / |Y_0(0)|^2 = 1 / e^2
        """
        return 2 ** .5 * s


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
                 wavelet=Morlet(), unbias=False, mask_coi=False,
                 compute_with_freq=False):
        """Arguments:
            x - 1 dimensional input signal
            dt - sample spacing
            dj - scale resolution
            wavelet - wavelet class to use, must have an attribute
                      `time_rep`, giving a wavelet function that takes (t, s)
                      as arguments and, if compute_with_freq is True, an
                      attribute `frequency_rep`, giving a wavelet function
                      that takes (w, s) as arguments.
            unbias - whether to unbias the power spectrum, as in Liu
                     et al. 2007 (default False)
            compute_with_freq - default False, compute the cwt using
                                a frequency representation
            mask_coi - disregard wavelet power outside the cone of
                       influence when computing global wavelet spectrum
                       (default False)
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
        self.cwt = cwt
        self.compute_with_freq = compute_with_freq
        self.unbias = unbias
        self.mask_coi = mask_coi

    @property
    def time(self):
        """Return array of times corresponding to data points."""
        return np.indices(self.data.shape).squeeze() * self.dt

    @property
    def fourier_period(self):
        """Return a function that calculates the equivalent fourier
        period as a function of scale.
        """
        return getattr(self.wavelet, 'fourier_period')

    @property
    def fourier_periods(self):
        """Return the equivalent fourier periods for the scales used."""
        return self.fourier_period(self.scales)

    @property
    def s0(self):
        """Find the smallest resolvable scale by finding where the
        equivalent fourier period is equal to 2 * dt. For a Morlet
        wavelet, this is roughly 1.
        """
        dt = self.dt

        def f(s):
            return self.fourier_period(s) - 2 * dt
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
        dt = self.dt
        # resolution
        dj = self.dj
        # smallest resolvable scale, chosen so that the equivalent
        # fourier period is approximately 2dt
        s0 = self.s0

        # Largest scale
        J = int((1 / dj) * np.log2(self.N * dt / s0))

        sj = s0 * 2 ** (dj * np.arange(0, J + 1))
        return sj

    # TODO: use np.frompyfunc on this
    # TODO: can we just replace it with fftfreqs?
    def w_k(self, k=None):
        """Angular frequency as a function of fourier index.

        If no k, returns an array of all the angular frequencies
        calculated using the length of the data.

        See eq5 of TC.
        """
        dt = self.dt
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
        widths = self.scales

        if self.compute_with_freq:
            wavelet = self.wavelet.frequency_rep
        else:
            wavelet = self.wavelet.time_rep

        return self.cwt(self.anomaly_data,
                        wavelet=wavelet,
                        widths=widths,
                        dt=self.dt,
                        wavelet_freq=self.compute_with_freq)

    @property
    def wavelet_power(self):
        """Calculate the wavelet power spectrum, optionally using
        the bias correction factor introduced by Liu et al. 2007,
        which is to divide by the scale.
        """
        s = np.expand_dims(self.scales, 1)
        if self.unbias:
            return np.abs(self.wavelet_transform) ** 2 / s
        elif not self.unbias:
            return np.abs(self.wavelet_transform) ** 2

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
        s = np.expand_dims(self.scales, 1)

        real_sum = np.sum(W_n.real / s ** .5, axis=0)
        x_n = real_sum * (dj * dt ** .5 / (C_d * Y_00))

        # add the mean back on (x_n is anomaly time series)
        x_n += self.data.mean()

        return x_n

    @property
    def global_wavelet_spectrum(self):
        if not self.mask_coi:
            mean_power = np.mean(self.wavelet_power, axis=1)
        elif self.mask_coi:
            mean_power = self.coi_mean(self.wavelet_power, axis=1)
        var = self.data_variance
        return mean_power / var

    def coi_mean(self, arr, axis=1):
        """Calculate a mean, but only over times within the cone of
        influence.

        Implement so can replace np.mean(wavelet_power, axis=1)
        """
        # TODO: consider applying upstream, inside wavelet_power
        coi = self.wavelet.coi
        s = self.scales
        t = self.time
        T, S = np.meshgrid(t, s)
        inside_coi = (coi(S) < T) & (T < (T.max() - coi(S)))
        mask_power = np.ma.masked_where(~inside_coi, self.wavelet_power)
        mask_mean = np.mean(mask_power, axis=axis)
        return mask_mean

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
        if self.wavelet.C_d is not None:
            return self.wavelet.C_d
        dj = self.dj
        dt = self.dt
        C_d = 1
        W_d = self.wavelet_transform_delta
        s = np.expand_dims(self.scales, 1)
        s = self.scales
        # value of the wavelet function at t=0
        Y_00 = self.wavelet.time_rep(0)

        real_sum = np.sum(W_d.real / s ** .5)
        C_d = real_sum * (dj * dt ** .5 / (C_d * Y_00))
        # TODO: coming out as 0.26 for morlet
        return C_d

    @property
    def wavelet_transform_delta(self):
        """Calculate the delta wavelet transform.

        Returns an array of the transform computed over the scales.
        """
        N = self.N
        # wavelet as function of (s, w_k)
        Y_ = self.wavelet.frequency_rep
        k = np.arange(N)
        s = self.scales
        K, S = np.meshgrid(k, s)

        # compute Y_ over all s, w_k and sum over k
        W_d = (1 / N) * np.sum(Y_(self.w_k(K), S), axis=1)

        # N.B This W_d is 1D

        return W_d

    @property
    def wavelet_variance(self):
        """Equivalent of Parseval's theorem for wavelets, S3.i.

        The wavelet transform conserves total energy, i.e. variance.

        Returns the variance of the input data.
        """
        # TODO: mask coi for calculation of wavelet_variance
        # is this possible? how does it change the factors?
        dj = self.dj
        dt = self.dt
        C_d = self.C_d
        N = self.N
        s = np.expand_dims(self.scales, 1)

        A = dj * dt / (C_d * N)

        var = A * np.sum(np.abs(self.wavelet_transform) ** 2 / s)

        return var

    @property
    def coi(self):
        """The Cone of Influence is the region near the edges of the
        input signal in which edge effects may be important.

        Return a tuple (T, S) that describes the edge of the cone
        of influence as a single line in (time, scale).
        """
        Tmin = self.time.min()
        Tmax = self.time.max()
        T = Tmax - Tmin
        s = self.scales
        c1 = Tmin + self.wavelet.coi(s)
        c2 = Tmax - c1

        C = np.hstack((c1[np.where(c1 < T / 2)], c2[np.where(c2 > T / 2)]))
        S = np.hstack((s[np.where(c1 < T / 2)], s[np.where(c2 > T / 2)]))

        # sort w.r.t time
        iC = C.argsort()
        sC = C[iC]
        sS = S[iC]

        return sC, sS


# TODO: derive C_d for given wavelet
