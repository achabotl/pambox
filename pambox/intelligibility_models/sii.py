# -*- coding: utf-8 -*-
from __future__ import division, print_function

import numpy as np
from numpy import log10, sum, asarray, zeros, ones


class Sii(object):
    """
    Speech intelligibility index model.

    Set the hearing threshold and the type of band importance function.

    Arguments for 'I':
        A scalar having a value of either 0, 1, 2, 3, 4, 5, or 6. The
        Band-importance functions associated with each scalar are
            0: Average speech as specified in Table 3 (DEFAULT)
            1: various nonsense syllable tests where most English phonemes
               occur equally often (as specified in Table B.2)
            2: CID-22 (as specified in Table B.2)
            3: NU6 (as specified in Table B.2)
            4: Diagnostic Rhyme test (as specified in Table B.2)
            5: short passages of easy reading material (as specified in
               Table B.2)
            6: SPIN (as specified in Table B.2)

    :param T: array_like, hearing threshold. 18 values in dB HL
    :param I: int, band importance function selector

    """

    def __init__(self, T=zeros(18), I=0):
        T = asarray(T)

        if len(T) != 18:
            raise ValueError("The length of T should be 18.")
        if I not in range(7):
            raise ValueError("Band importance should be an integer between 0 \
                             and 1.")
        self.T = asarray(T)
        self.I = int(I)

        # Band center frequencies for 1/3rd octave procedure (Table 3)
        self.f = asarray([160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250,
                          1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000])

        # Define Internal Noise Spectrum Level (Table 3)
        self.X = asarray([0.6, -1.7, -3.9, -6.1, -8.2, -9.7, -10.8, -11.9,
                          -12.5, -13.5, -15.4, -17.7, -21.2, -24.2, -25.9,
                          -23.6, -15.8, -7.1])

        # Equivalent Internal Noise Spectrum Level (4.4 Eq. 10)
        self.X = self.X + self.T

        self.BIArr = asarray(
            [0.0083,  0,       0.0365,  0.0168,  0,       0.0114,  0,
             0.0095,  0,       0.0279,  0.013,   0.024,   0.0153,  0.0255,
             0.015,   0.0153,  0.0405,  0.0211,  0.033,   0.0179,  0.0256,
             0.0289,  0.0284,  0.05,    0.0344,  0.039,   0.0558,  0.036,
             0.044,   0.0363,  0.053,   0.0517,  0.0571,  0.0898,  0.0362,
             0.0578,  0.0422,  0.0518,  0.0737,  0.0691,  0.0944,  0.0514,
             0.0653,  0.0509,  0.0514,  0.0658,  0.0781,  0.0709,  0.0616,
             0.0711,  0.0584,  0.0575,  0.0644,  0.0751,  0.066,   0.077,
             0.0818,  0.0667,  0.0717,  0.0664,  0.0781,  0.0628,  0.0718,
             0.0844,  0.0774,  0.0873,  0.0802,  0.0811,  0.0672,  0.0718,
             0.0882,  0.0893,  0.0902,  0.0987,  0.0961,  0.0747,  0.1075,
             0.0898,  0.1104,  0.0938,  0.1171,  0.0901,  0.0755,  0.0921,
             0.0868,  0.112,   0.0928,  0.0932,  0.0781,  0.082,   0.1026,
             0.0844,  0.0981,  0.0678,  0.0783,  0.0691,  0.0808,  0.0922,
             0.0771,  0.0867,  0.0498,  0.0562,  0.048,   0.0483,  0.0719,
             0.0527,  0.0728,  0.0312,  0.0337,  0.033,   0.0453,  0.0461,
             0.0364,  0.0551,  0.0215,  0.0177,  0.027,   0.0274,  0.0306,
             0.0185,  0,       0.0253,  0.0176,  0.024,   0.0145,  0])\
            .reshape(-1, 7)

        self.Ei = asarray([32.41,  33.81,  35.29,  30.77,
                           34.48,  33.92,  37.76,  36.65,
                           34.75,  38.98,  41.55,  42.5,
                           33.98,  38.57,  43.78,  46.51,
                           34.59,  39.11,  43.3,   47.4,
                           34.27,  40.15,  44.85,  49.24,
                           32.06,  38.78,  45.55,  51.21,
                           28.3,   36.37,  44.05,  51.44,
                           25.01,  33.86,  42.16,  51.31,
                           23,     31.89,  40.53,  49.63,
                           20.15,  28.58,  37.7,   47.65,
                           17.32,  25.32,  34.39,  44.32,
                           13.18,  22.35,  30.98,  40.8,
                           11.55,  20.15,  28.21,  38.13,
                           9.33,   16.78,  25.41,  34.41,
                           5.31,   11.47,  18.35,  28.24,
                           2.59,   7.67,   13.87,  23.45,
                           1.13,   5.07,   11.39,  20.72]).reshape(-1, 4)

    def _band_importance(self, test):

        if test not in range(7):
            raise ValueError("Band Importance function must be integer \
                             between 0 and 6.")
        return self.BIArr[:, test].T

    def _speech_spectrum(self, vcl_effort):
        """Return the standard speech spectrum from Table 3.

        The spectrum depends on the vocal effort, possible values are 'normal',
        'raised', 'loud', 'shout'.

        :vcl_effort: string, vocal effort
        :returns:

        """
        efforts = {'normal': 0, 'raised': 1, 'loud': 2, 'shout': 3}
        if vcl_effort not in efforts:
            raise ValueError("Vocal error string not recognized.")
        return self.Ei[:, efforts[vcl_effort]]

    def predict(self, E, N=-50 * ones(18)):
        """o

        :param E: array_like, speech level
        :param N: array_like, noise level
        """

        E[np.isnan(E)] = 0
        N[np.isnan(N)] = 0

        # Self-Speech Masking Spectrum (4.3.2.1 Eq. 5)
        V = E - 24.

        # 4.3.2.2
        B = np.fmax(V, N)

        # Calculate slope parameter Ci (4.3.2.3 Eq. 7)
        C = 0.6 * (B + 10. * log10(self.f) - 6.353) - 80.

        # Initialize Equivalent Masking Spectrum Level (4.3.2.4)
        Z = zeros(18)
        Z[0] = B[0]

        # Calculate Equivalent Masking Spectrum Level (4.3.2.5 Eq. 9)
        for i in range(1, 18):
            Z[i] = 10. * log10(10 ** (0.1 * N[i])
                               + sum(10. ** (0.1 * (B[0:i] + 3.32 * C[0:i]
                                                    * log10(0.89 * self.f[i]
                                                            / self.f[0:i])))))
        # Disturbance Spectrum Level (4.5)
        D = np.fmax(Z, self.X)

        # Level Distortion Factor (4.6 Eq. 11)
        L = 1. - (E - self._speech_spectrum('normal') - 10.) / 160.
        L = np.fmin(1., L)

        # 4.7.1 Eq. 12
        K = (E - D + 15.) / 30.
        K = np.fmin(1., np.fmax(0., K))

        # Band Audibility Function (7.7.2 Eq. 13)
        A = L * K

        # Speech Intelligibility Index (4.8 Eq. 14)
        out = sum(self._band_importance(self.I) * A)
        return np.fmax(out, 0)
