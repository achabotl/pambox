# -*- coding: utf-8 -*-
from __future__ import division, print_function
from numpy import pi, exp, sin, cos, sqrt, abs, ones
import numpy as np
import scipy as sp
import scipy.signal as ss


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

    b, a = sp.signal.butter(N=N, Wn=cutoff * 2. / fs, btype='lowpass')
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
