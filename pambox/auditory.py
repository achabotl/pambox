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
    """Auditory filter bandwith

    :fc: @todo
    :returns: @todo

    """
    # In Hz, according to Glasberg and Moore (1990)
    return 24.7 + fc / 9.265


def gammatone_filtering(signal, center_f=CENTER_F, fs=FS):
    """Filters a signal using a gammatone filterbank

    :signal: @todo
    :returns: @todo

    """
    b, a, _, _, _ = gammatone_make(fs, center_f)
    return 2 * gammatone_apply(signal, b, a)


def lowpass_env_filtering(x, cutoff=150., N=1, fs=FS):
    """Low-pass filters signal

    :x: @todo
    :cutoff: @todo
    :returns: @todo

    """

    b, a = sp.signal.butter(N=N, Wn=cutoff * 2. / fs, btype='lowpass')
    return sp.signal.lfilter(b, a, x)


class GammatoneFilterbank():
    '''
    GammatoneFilterbank

    Input:
        fs ... float, sampling frequency
        cf ... ndarray, center frequencies

    '''
    def __init__(self, cf, fs, b=1.019, order=1, Q=9.26449,
                 min_bw=24.7):
        """

        :cf: center frequencies of the filterbank
        :fs: sampling frequency
        :b: beta of the gammatone filter
        :order:
        :Q: Q-value of the ERB
        :min_bw: minimum bandwidth of an ERB
        """
        try:
            len(cf)
        except TypeError:
            cf = [cf]
        cf = np.asarray(cf)
        self.fs = fs
        T = 1 / self.fs
        self.b, self.erb_order, self.EarQ, self.min_bw = b, order, Q, min_bw
        erb = ((cf / Q) ** order + min_bw ** order) ** (
            1 / order)

        B = b * 2 * pi * erb

        A0 = T
        A2 = 0
        B0 = 1
        B1 = -2 * cos(2 * cf * pi * T) / exp(B * T)
        B2 = exp(-2 * B * T)

        A11 = -(2 * T * cos(2 * cf * pi * T) / exp(B * T) + 2 * sqrt(
            3 + 2 ** 1.5) * T * sin(2 * cf * pi * T) / exp(B * T)) / 2
        A12 = -(2 * T * cos(2 * cf * pi * T) / exp(B * T) - 2 * sqrt(
            3 + 2 ** 1.5) * T * sin(2 * cf * pi * T) / exp(B * T)) / 2
        A13 = -(2 * T * cos(2 * cf * pi * T) / exp(B * T) + 2 * sqrt(
            3 - 2 ** 1.5) * T * sin(2 * cf * pi * T) / exp(B * T)) / 2
        A14 = -(2 * T * cos(2 * cf * pi * T) / exp(B * T) - 2 * sqrt(
            3 - 2 ** 1.5) * T * sin(2 * cf * pi * T) / exp(B * T)) / 2

        i = 1j
        gain = abs((-2 * exp(4 * i * cf * pi * T) * T + \
                    2 * exp(-(B * T) + 2 * i * cf * pi * T) * T * \
                    (cos(2 * cf * pi * T) - sqrt(3 - 2 ** (3. / 2)) * \
                     sin(2 * cf * pi * T))) * \
                   (-2 * exp(4 * i * cf * pi * T) * T + \
                    2 * exp(-(B * T) + 2 * i * cf * pi * T) * T * \
                    (cos(2 * cf * pi * T) + sqrt(3 - 2 ** (3. / 2)) * \
                     sin(2 * cf * pi * T))) * \
                   (-2 * exp(4 * i * cf * pi * T) * T + \
                    2 * exp(-(B * T) + 2 * i * cf * pi * T) * T * \
                    (cos(2 * cf * pi * T) - \
                     sqrt(3 + 2 ** (3. / 2)) * sin(2 * cf * pi * T))) * \
                   (-2 * exp(4 * i * cf * pi * T) * T + 2 * exp(
                       -(B * T) + 2 * i * cf * pi * T) * T * \
                    (cos(2 * cf * pi * T) + sqrt(3 + 2 ** (3. / 2)) * sin(
                        2 * cf * pi * T))) / \
                   (-2 / exp(2 * B * T) - 2 * exp(4 * i * cf * pi * T) + \
                    2 * (1 + exp(4 * i * cf * pi * T)) / exp(B * T)) ** 4)

        allfilts = ones(len(cf))

        self.A0, self.A11, self.A12, self.A13, self.A14, self.A2, self.B0, self.B1, self.B2, self.gain = \
            A0 * allfilts, A11, A12, A13, A14, A2 * allfilts, B0 * allfilts, B1, B2, gain


    def filter(self, x):
        A0, A11, A12, A13, A14, A2, B0, B1, B2, gain = \
            self.A0, self.A11, self.A12, self.A13, self.A14, self.A2, self.B0, self.B1, self.B2, self.gain

        output = np.zeros((gain.shape[0], x.shape[-1]))
        for chan in range(gain.shape[0]):
            y1 = ss.lfilter([A0[chan] / gain[chan], A11[chan] / gain[chan],
                             A2[chan] / gain[chan]],
                            [B0[chan], B1[chan], B2[chan]], x)
            y2 = ss.lfilter([A0[chan], A12[chan], A2[chan]],
                            [B0[chan], B1[chan], B2[chan]], y1)
            y3 = ss.lfilter([A0[chan], A13[chan], A2[chan]],
                            [B0[chan], B1[chan], B2[chan]], y2)
            y4 = ss.lfilter([A0[chan], A14[chan], A2[chan]],
                            [B0[chan], B1[chan], B2[chan]], y3)
            output[chan, :] = y4

        return output
