# -*- coding: utf-8 -*-
"""
:mod:`pambox.periph.inner` regroups processes of the inner.
"""
from __future__ import division, print_function, absolute_import

import numpy as np
from numpy import pi, exp, sin, cos, sqrt, abs, ones
from numpy.core.umath import pi
import scipy as sp
import scipy.signal as ss
try:
    _ = np.use_fastnumpy
    from numpy.fft import fft, ifft, rfft, irfft
except AttributeError:
    from scipy.fftpack import fft, ifft
    from numpy.fft import rfft, irfft


CENTER_F = np.asarray([63, 80, 100, 125, 160, 200, 250, 315, 400, 500,
                       630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000,
                       5000, 6300, 8000])
FS = np.asarray([22050.])


def erbbw(fc):
    """Bandwith or an ERB.

    Parameters
    ----------
    fc : ndarray
        Center frequency, or center frequencies, of the filter.

    Returns
    -------
    ndarray
        Equivalent rectangular bandwidth of the filter(s).
    """
    # In Hz, according to Glasberg and Moore (1990)
    return 24.7 + fc / 9.265


def lowpass_env_filtering(x, cutoff=150., n=1, fs=22050):
    """Low-pass filters a signal using a Butterworth filter.

    Parameters
    ----------
    x : ndarray
    cutoff : float, optional
        Cut-off frequency of the low-pass filter, in Hz. The default is 150 Hz.
    n : int, optional
        Order of the low-pass filter. The default is 1.
    fs : float, optional
        Sampling frequency of the signal to filter. The default is 22050 Hz.

    Returns
    -------
    ndarray
        Low-pass filtered signal.

    """

    b, a = sp.signal.butter(N=n, Wn=cutoff * 2. / fs, btype='lowpass')
    return sp.signal.lfilter(b, a, x)


class GammatoneFilterbank():
    """Gammatone Filterbank

    Parameters
    ----------
    cf : array_like
        Center frequencies of the filterbank.
    fs : float
        Sampling frequency of the signals to filter.
    b : float
        beta of the gammatone filters. The default is `b` = 1.019.
    order : int
        Order. The default value is 1.
    q : float
        Q-value of the ERB. The default value is 9.26449.
    min_bw : float
        Minimum bandwidth of an ERB.

    References
    ----------

    """

    def __init__(self, cf, fs, b=1.019, order=1, q=9.26449,
                 min_bw=24.7):
        try:
            len(cf)
        except TypeError:
            cf = [cf]
        cf = np.asarray(cf)
        self.fs = fs
        t = 1 / self.fs
        self.b, self.erb_order, self.EarQ, self.min_bw = b, order, q, min_bw
        erb = ((cf / q) ** order + min_bw ** order) ** (
            1 / order)

        b = b * 2 * pi * erb

        a0 = t
        a2 = 0
        b0 = 1
        b1 = -2 * cos(2 * cf * pi * t) / exp(b * t)
        b2 = exp(-2 * b * t)

        a11 = -(2 * t * cos(2 * cf * pi * t) / exp(b * t) + 2 * sqrt(
            3 + 2 ** 1.5) * t * sin(2 * cf * pi * t) / exp(b * t)) / 2
        a12 = -(2 * t * cos(2 * cf * pi * t) / exp(b * t) - 2 * sqrt(
            3 + 2 ** 1.5) * t * sin(2 * cf * pi * t) / exp(b * t)) / 2
        a13 = -(2 * t * cos(2 * cf * pi * t) / exp(b * t) + 2 * sqrt(
            3 - 2 ** 1.5) * t * sin(2 * cf * pi * t) / exp(b * t)) / 2
        a14 = -(2 * t * cos(2 * cf * pi * t) / exp(b * t) - 2 * sqrt(
            3 - 2 ** 1.5) * t * sin(2 * cf * pi * t) / exp(b * t)) / 2

        i = 1j
        gain = abs((-2 * exp(4 * i * cf * pi * t) * t +
                    2 * exp(-(b * t) + 2 * i * cf * pi * t) * t *
                    (cos(2 * cf * pi * t) - sqrt(3 - 2 ** (3. / 2)) *
                     sin(2 * cf * pi * t))) *
                   (-2 * exp(4 * i * cf * pi * t) * t +
                    2 * exp(-(b * t) + 2 * i * cf * pi * t) * t *
                    (cos(2 * cf * pi * t) + sqrt(3 - 2 ** (3. / 2)) *
                     sin(2 * cf * pi * t))) *
                   (-2 * exp(4 * i * cf * pi * t) * t +
                    2 * exp(-(b * t) + 2 * i * cf * pi * t) * t *
                    (cos(2 * cf * pi * t) -
                     sqrt(3 + 2 ** (3. / 2)) * sin(2 * cf * pi * t))) *
                   (-2 * exp(4 * i * cf * pi * t) * t + 2 * exp(
                       -(b * t) + 2 * i * cf * pi * t) * t *
                    (cos(2 * cf * pi * t) + sqrt(3 + 2 ** (3. / 2)) * sin(
                        2 * cf * pi * t))) /
                   (-2 / exp(2 * b * t) - 2 * exp(4 * i * cf * pi * t) +
                    2 * (1 + exp(4 * i * cf * pi * t)) / exp(b * t)) ** 4)

        allfilts = ones(len(cf))

        self.a0, self.a11, self.a12, self.a13, self.a14, self.a2, \
            self.b0, self.b1, self.b2, self.gain = \
            a0 * allfilts, a11, a12, a13, a14, a2 * allfilts, \
            b0 * allfilts, b1, b2, gain

    def filter(self, x):
        """Filters a signal along its last dimension.

        Parameters
        ----------
        x : ndarray
            Signal to filter.

        Returns
        -------
        ndarray
            Filtered signals
        """

        a0, a11, a12, a13, a14, a2 = self.a0, self.a11, self.a12, self.a13, \
            self.a14, self.a2
        b0, b1, b2, gain = self.b0, self.b1, self.b2, self.gain

        output = np.zeros((gain.shape[0], x.shape[-1]))
        for chan in range(gain.shape[0]):
            y1 = ss.lfilter([a0[chan] / gain[chan], a11[chan] / gain[chan],
                             a2[chan] / gain[chan]],
                            [b0[chan], b1[chan], b2[chan]], x)
            y2 = ss.lfilter([a0[chan], a12[chan], a2[chan]],
                            [b0[chan], b1[chan], b2[chan]], y1)
            y3 = ss.lfilter([a0[chan], a13[chan], a2[chan]],
                            [b0[chan], b1[chan], b2[chan]], y2)
            y4 = ss.lfilter([a0[chan], a14[chan], a2[chan]],
                            [b0[chan], b1[chan], b2[chan]], y3)
            output[chan, :] = y4

        return output


def noctave_filtering(x, center_f, fs, width=3, output_time=False):
    """Rectangular nth-octave filtering.

    :x: signal
    :center_f: ndarray, center frequencies, in Hz
    :width: width of the filters, default 3 for 1/3-octave

    Parameters
    ----------
    x : array_like
        Input signal
    center_f : array_like
        List of the center frequencies of the filterbank.
    fs : int
        Sampling frequency of the input signal.
    width : float
         Width of the filters, in fraction of octave. The default value is 3,
         therefore 1/3-octave.
    output_time : bool, optional
        If `True`, also outputs the time output of the filtering. The default
        is to output the RMS value of each band only. Doing the inverse FFT
        is very costly; setting the argument to `False` prevents from doing
        that computation.

    Returns
    -------
    out_rms : ndarray
         RMS power at the output of each filter.
    out_time : ndarray
         Time signals at the output of the filterbank. The shape is (`len(
         center_f) x len(x)`).

    """
    # Use numpy's FFT because SciPy's version of rfft (2 real results per
    # frequency bin) behaves differently from numpy's (1 complex result per
    # frequency bin)
    center_f = np.asarray(center_f, dtype='float')

    n = len(x)
    # TODO Use powers of 2 to calculate the power spectrum, and also, possibly
    # use RFFT instead of the complete fft.
    X = rfft(x)
    X_pow = np.abs(X) ** 2 / n  # Power spectrum
    X_pow[1:] = X_pow[1:] * 2.
    bound_f = np.zeros(len(center_f) + 1)
    bound_f[0] = center_f[0] * 2. ** (- 1. / (2. * width))
    bound_f[1:] = center_f * 2. ** (1. / (2. * width))
    bound_f = bound_f[bound_f < fs / 2]
    # Convert from frequencies to vector indexes. Factor of two is because
    # we consider positive frequencies only.
    bound_idx = np.floor(bound_f / (fs / 2.) * len(X_pow)).astype('int')
    # Initialize arrays
    out_rms = np.zeros(len(center_f))
    out_time = np.zeros((len(center_f), x.shape[-1]), dtype='complex')
    for idx, (l, f) in enumerate(zip(bound_idx[0:], bound_idx[1:])):
        out_time[idx, l:f] = X[l:f]
        out_rms[idx] = np.sqrt(np.sum(X_pow[l:f]) / n)
    if output_time:
        out_time = np.real(irfft(out_time, n=n, axis=-1))
        return out_rms, out_time
    else:
        return out_rms


def mod_filterbank(signal, fs, modf):
    """Implementation of the EPSM-filterbank.

    Parameters
    ----------
    signal : ndarray
        Temporal envelope of a signal
    fs : int
        Sampling frequency of the signal.
    modf : array_like
        List of the center frequencies of the modulation filterbank.

    Returns
    -------
    tuple of ndarray
        Integrated power spectrum at the output of each filter
        Filtered time signals.

    """
    modf = np.asarray(modf)
    fcs = modf[1:]
    fcut = modf[0]
    # Make signal odd length
    signal = signal[0:-1] if (len(signal) % 2) == 0 else signal

    q = 1.     # Q-factor of band-pass filters
    lp_order = 3.     # order of the low-pass filter

    n = signal.shape[-1]  # length of envelope signals
    X = fft(signal)
    X_mag = np.abs(X)
    X_power = np.square(X_mag) / n  # power spectrum
    X_power_pos = X_power[0:np.floor(n / 2).astype('int') + 1]
    # take positive frequencies only and multiply by two to get the same total
    # energy
    X_power_pos[1:] = X_power_pos[1:] * 2

    pos_freqs = np.linspace(0, fs / 2, X_power_pos.shape[-1])
    # Concatenate vector of 0:fs and -fs:1
    freqs = np.concatenate((pos_freqs, -1 * pos_freqs[-1:0:-1]))

    # Initialize transfer function
    TFs = np.zeros((len(fcs) + 1, len(freqs))).astype('complex')
    # Calculating frequency-domain transfer function for each center frequency:
    for k in range(len(fcs)):
        TFs[k + 1, 1:] = 1. / (1. + (1j * q * (freqs[1:] / fcs[k] - fcs[k] /
                                               freqs[1:])))  # p287 Hambley.

    # squared filter magnitude transfer functions
    Wcf = np.square(np.abs(TFs))

    # Low-pass filter squared transfer function, third order Butterworth filter
    # TF from:
    # http://en.wikipedia.org/wiki/Butterworth_filter
    Wcf[0, :] = 1 / (1 + ((2 * pi * freqs / (2 * pi * fcut)) ** (2 * lp_order)))
    # Transfer function of low-pass filter
    TFs[0, :] = np.sqrt(Wcf[0, :])

    # initialize output product:
    vout = np.zeros((len(fcs) + 1, len(pos_freqs)))
    powers = np.zeros(len(modf))

    # ------------ DC-power, --------------------------
    # here divide by two such that a fully modulated tone has an AC-power of 1.
    dc_power = X_power_pos[0] / n / 2
    # ------------------------------------------------
    X_filt = np.zeros((Wcf.shape[0], X.shape[-1]), dtype='complex128')
    filtered_envs = np.zeros_like(X_filt, dtype='float')

    for k, (w, TF) in enumerate(zip(Wcf, TFs)):
        vout[k] = X_power_pos * w[:np.floor(n / 2).astype('int') + 1]
        # Integration estimated as a sum from f > 0
        # integrate envelope power in the passband of the filter. Index goes
        # from 2:end since integration is for f>0
        powers[k] = np.sum(vout[k, 1:]) / n / dc_power
        # Filtering and inverse Fourier transform to get time signal.
        X_filt[k] = X * TF
        filtered_envs[k] = np.real(ifft(X_filt[k]))
    return powers, filtered_envs
