# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import logging

import numpy as np
from numpy import pi

try:
    _ = np.use_fastnumpy  # Use Enthought MKL optimizations
    from numpy.fft import fft, ifft, rfft, irfft
except AttributeError:
    try:
        import mklfft  # MKL FFT optimizations from Continuum Analytics
        from numpy.fft import fft, ifft, rfft, irfft
    except ImportError:
        # Finally, just use Numpy's and Scipy's
        from scipy.fftpack import fft, ifft
        from numpy.fft import rfft, irfft

log = logging.getLogger(__name__)


class EPSMModulationFilterbank(object):
    """Implementation of the EPSM modulation filterbank.

    Parameters
    ----------
    fs : int
        Sampling frequency of the signal.
    modf : array_like
        List of the center frequencies of the modulation filterbank.
    q : float
        Q-factor of the modulation filters. Defaults to 1.
    low_pass_order : float
        Order of the low-pass filter. Defaults to 3.

    Methods
    -------
    filter(signal)
        Filters the signal using the modulation filterbank.

    Notes
    -----
    The envelope power spectrum model (EPSM) filterbank was defined in
    [ewert2000]_ and the implementation was validated against the Matlab
    implementation of [jorgensen2011]_.

    References
    ----------
    .. [ewert2000] S. D. Ewert and T. Dau: Characterizing frequency
        selectivity for envelope fluctuations.. J. Acoust. Soc. Am. 108
        (2000) 1181--1196.

    .. [jorgensen2011] S. Jørgensen and T. Dau: Predicting speech
        intelligibility based on the signal-to-noise envelope power ratio
        after modulation-frequency selective processing. J. Acoust. Soc. Am.
        130 (2011) 1475--1487.

    """

    def __init__(self, fs, modf, q=1., low_pass_order=3.):
        self.fs = fs
        self.modf = np.asarray(modf)
        self.q = q  # Q-factor of band-pass filters
        self.lp_order = low_pass_order  # order of the low-pass filter

    def _calculate_coefficients(self, freqs):
        fcs = self.modf[1:]
        fcut = self.modf[0]
        # Initialize transfer function
        TFs = np.zeros((len(fcs) + 1, len(freqs))).astype('complex')
        # Calculating frequency-domain transfer function for each center
        # frequency:
        for k in range(len(fcs)):
            TFs[k + 1, 1:] = \
                1. / (1. + (1j * self.q * (freqs[1:] / fcs[k] - fcs[k] /
                                           freqs[1:])))  # p287 Hambley.

        # squared filter magnitude transfer functions
        Wcf = np.square(np.abs(TFs))
        # Low-pass filter squared transfer function, third order Butterworth
        # filter
        # TF from:
        # http://en.wikipedia.org/wiki/Butterworth_filter
        Wcf[0, :] = 1 / (
            1 + ((2 * pi * freqs / (2 * pi * fcut)) ** (2 * self.lp_order)))
        # Transfer function of low-pass filter
        TFs[0, :] = np.sqrt(Wcf[0, :])
        return TFs, Wcf

    def filter(self, signal):
        """

        Parameters
        ----------
        signal : ndarray
            Temporal envelope of a signal
        Returns
        -------
        tuple of ndarray
            Integrated power spectrum at the output of each filter
            Filtered time signals.
        """

        # Make signal odd length
        signal = signal[0:-1] if (len(signal) % 2) == 0 else signal

        n = signal.shape[-1]  # length of envelope signals
        X = fft(signal)
        X_mag = np.abs(X)
        X_power = np.square(X_mag) / n  # power spectrum
        X_power_pos = X_power[0:np.floor(n / 2).astype('int') + 1]
        # take positive frequencies only and multiply by two to get the same total
        # energy
        X_power_pos[1:] = X_power_pos[1:] * 2

        pos_freqs = np.linspace(0, self.fs / 2, X_power_pos.shape[-1])
        # Concatenate vector of 0:fs and -fs:1
        freqs = np.concatenate((pos_freqs, pos_freqs[-1:0:-1]))

        TFs, Wcf = self._calculate_coefficients(freqs)

        # initialize output product:
        vout = np.zeros((len(self.modf), len(pos_freqs)))
        powers = np.zeros(len(self.modf))

        # ------------ DC-power, --------------------------
        # here divide by two such that a fully modulated tone has an AC-power of 1.
        dc_power = X_power_pos[0] / n / 2
        # ------------------------------------------------
        X_filt = np.zeros((Wcf.shape[0], X.shape[-1]), dtype='complex128')
        filtered_envs = np.zeros_like(X_filt, dtype='float')

        for k, (w, TF) in enumerate(zip(Wcf, TFs)):
            vout[k] = X_power_pos * w[:np.floor(n / 2).astype('int') + 1]
            # Integration estimated as a sum from f > 0
            # integrate envelope power in the pass-band of the filter. Index goes
            # from 2:end since integration is for f>0
            powers[k] = np.sum(vout[k, 1:]) / n / dc_power
            # Filtering and inverse Fourier transform to get time signal.
            X_filt[k] = X * TF
            filtered_envs[k] = np.real(ifft(X_filt[k]))
        powers[np.isnan(powers)] = 0
        return powers, filtered_envs
