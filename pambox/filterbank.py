# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import scipy as sp
import scipy.signal as ss
import matplotlib.pyplot as plt
from scipy import pi
from numpy.fft import fft, ifft, rfft, irfft
from itertools import izip


def mod_filterbank(signal, fs, modf):
    """Implementation of the EPSM-filterbank.

    :signal: ndarray, temporal envelope of a signal
    :fs: int, sampling frequency
    :returns: ndarray, integrated power spectrum at the output of each filter
              ndarray, filtered time signals.

    """
    modf = np.asarray(modf)
    fcs = modf[1:]
    fcut = modf[0]
    # Make signal odd length
    signal = signal[0:-1] if (len(signal) % 2) == 0 else signal

    Q = 1.     # Q-factor of band-pass filters
    n = 3.     # order of the low-pass filter

    N = signal.shape[-1]  # length of envelope signals
    X = fft(signal)
    X_mag = np.abs(X)
    X_power = np.square(X_mag) / N  # power spectrum
    X_power_pos = X_power[0:np.floor(N / 2) + 1]
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
        TFs[k + 1, 1:] = 1. / (1. + (1j * Q * (freqs[1:] / fcs[k] - fcs[k] /
                                               freqs[1:])))  # p287 Hambley.

    # squared filter magnitude transfer functions
    Wcf = np.square(np.abs(TFs))

    # Low-pass filter squared transfer function, third order Butterworth filter
    # TF from:
    # http://en.wikipedia.org/wiki/Butterworth_filter
    Wcf[0, :] = 1 / (1 + ((2 * pi * freqs / (2 * pi * fcut)) ** (2 * n)))
    # Transfer function of low-pass filter
    TFs[0, :] = np.sqrt(Wcf[0, :])

    # initialize output product:
    Vout = np.zeros((len(fcs) + 1, len(pos_freqs)))
    powers = np.zeros(len(modf))

    # ------------ DC-power, --------------------------
    # here divide by two such that a fully modulated tone has an AC-power of 1.
    DC_power = X_power_pos[0] / N / 2
    # ------------------------------------------------
    X_filt = np.zeros((Wcf.shape[0], X.shape[-1]), dtype='complex128')
    filtered_envs = np.zeros_like(X_filt, dtype='float')

    for k, (w, TF) in enumerate(izip(Wcf, TFs)):
        Vout[k] = X_power_pos * w[:np.floor(N / 2) + 1]
        # Integration estimated as a sum from f > 0
        # integrate envelope power in the passband of the filter. Index goes
        # from 2:end since integration is for f>0
        powers[k] = np.sum(Vout[k, 1:]) / N / DC_power
        # Filtering and inverse Fourier transform to get time signal.
        X_filt[k] = X * TF
        filtered_envs[k] = np.real(ifft(X_filt[k]))
    return powers, filtered_envs


def mfreqz(b, a=1, fs=22050.0):
    """Plot the frequency and phase response.

    :b:
    :a:

    From http://mpastell.com/2010/01/18/fir-with-scipy/

    """
    w, h = ss.freqz(b, a)
    h_dB = 20 * np.log10(abs(h))
    plt.subplot(211)
    if fs:
        f = sp.linspace(0, fs / 2, len(w))
        plt.plot(f, h_dB)
    else:
        plt.plot(w / max(w), h_dB)
    plt.ylim(-150, 5)
    plt.ylabel('Magnitude (db)')
    plt.xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
    plt.title(r'Frequency response')
    plt.subplot(212)
    h_Phase = sp.unwrap(sp.arctan2(sp.imag(h), sp.real(h)))
    plt.plot(w / max(w), h_Phase)
    plt.ylabel('Phase (radians)')
    plt.xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
    plt.title(r'Phase response')
    plt.subplots_adjust(hspace=0.5)


def impz(b, a=1):
    """Plot step and impulse response.

    :b:
    :a:

    From http://mpastell.com/2010/01/18/fir-with-scipy/

    """
    l = len(b)
    impulse = np.repeat(0., l)
    impulse[0] = 1.
    x = np.arange(0, l)
    response = sp.lfilter(b, a, impulse)
    plt.subplot(211)
    plt.stem(x, response)
    plt.ylabel('Amplitude')
    plt.xlabel(r'n (samples)')
    plt.title(r'Impulse response')
    plt.subplot(212)
    step = sp.cumsum(response)
    plt.stem(x, step)
    plt.ylabel('Amplitude')
    plt.xlabel(r'n (samples)')
    plt.title(r'Step response')
    plt.subplots_adjust(hspace=0.5)


def noctave_filtering(x, center_f, fs, width=3):
    """Rectangular nth-octave filtering.

    :x: signal
    :center_f: ndarray, center frequencies, in Hz
    :width: width of the filters, default 3 for 1/3-octave

    Returns:
    :spec: nth-octave spectrum

    """
    # Use numpy's FFT because SciPy's version of rfft (2 real results per
    # frequency bin) behaves differently from numpy's (1 complex result per
    # frequency bin)
    center_f = np.asarray(center_f, dtype='float')

    N = len(x)
    # TODO Use powers of 2 to calculate the power spectrum, and also, possibly
    # use RFFT instead of the complete fft.
    X = rfft(x)
    X_pow = np.abs(X) ** 2 / N  # Power spectrum
    X_pow[1:] = X_pow[1:] * 2.
    bound_f = np.zeros(len(center_f) + 1)
    bound_f[0] = center_f[0] * 2. ** (- 1. / (2. * width))
    bound_f[1:] = center_f * 2. ** (1. / (2. * width))
    bound_f = bound_f[bound_f < fs / 2]
    # Convert from frequencies to vector indexes. Factor of two is because
    # we consider positive frequencies only.
    bound_idx = np.floor(bound_f / (fs / 2.) * len(X_pow))
    # Initialize arrays
    out_rms = np.zeros(len(center_f))
    out_time = np.zeros((len(center_f), x.shape[-1]), dtype='complex')
    for idx, (l, f) in enumerate(zip(bound_idx[0:], bound_idx[1:])):
        out_time[idx, l:f] = X[l:f]
        out_rms[idx] = np.sqrt(np.sum(X_pow[l:f]) / N)
    out_time = np.real(irfft(out_time, n=N, axis=-1))
    return out_time, out_rms


def noctave_center_freq(lowf, highf, width=3):
    """Calculate exact center N-octave space center frequencies.

    In practive, what is often desired is the "simplified" center frequencies,
    so this function is not of much use.

    :lowf: low frequency Hz
    :highf: high frequency Hz
    :width: spacing, 3 for third-octave
    :returns: @todo

    """
    n_centers = np.log2(highf / lowf) * width + 1
    n_octave = np.log2(highf / lowf)
    return lowf * np.logspace(0, n_octave, num=n_centers, base=2)
