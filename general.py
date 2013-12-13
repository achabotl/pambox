from __future__ import division
import numpy as np
import scipy as sp
from scipy.io import wavfile
from numpy import min, log2, ceil, floor, argmin, zeros, arange,  float
from numpy.fft import fft, ifft


def dbspl(x, ac=False, offset=100.0):
    """RMS value of signal (in dB)

    DBSPL(x) computes the SPL (sound pressure level) of the input signal
    measured in dB, using the convention that a pure tone at 100 dB SPL has
    an RMS value of 1.

    DBSPL(x, ac=True) does the same, but considers only the AC component of the
    signal (i.e. the mean is removed).

    See also: setdbspl

    References:
      Auditory Modeling Toolbox, Peter L. Soendergaard
      B. C. J. Moore. An Introduction to the Psychology of Hearing. Academic
      Press, 5th edition, 2003.


    :x: arraly_like, signal
    :ac: bool, consider only the AC component of the signal.
    :offset: float, reference to convert between RMS and dB SPL.
    :returns: @todo

    """
    x = np.asarray(x)
    return 20. * np.log10(rms(x, ac)) + offset


def setdbspl(x, lvl, ac=False, offset=100.0):
    """Set level of signal in dB SPL

    SETDBSPL(insig,lvl) sets the SPL (sound pressure level) of the signal
    insig to lvl dB, using the convention that a pure tone with an RMS value
    of 1 corresponds to 100 dB SPL.

    If the input is an array, it is assumed that each row is a signal.

    SETDBSPL(insig,lvl,ac=True) does the same, but considers only the AC
    component of the signal (i.e. the mean is removed).

    References:
      B. C. J. Moore. An Introduction to the Psychology of Hearing. Academic
      Press, 5th edition, 2003.


    :x: @todo
    :lvl: @todo
    :ac: @todo
    :returns: @todo

    """
    x = np.asarray(x)
    return x / rms(x, ac) * 10. ** ((lvl - offset) / 20.)


def rms(x, ac=False):
    """RMS value of a signal.

    :x: signal
    :ac: bool, default: True
        consider only the AC component of the signal
    :rms: rms value

    """
    x = np.asarray(x)
    if ac:
        return np.linalg.norm((x - np.mean(x)) / np.sqrt(x.shape[-1]))
    else:
        return np.linalg.norm(x / np.sqrt(x.shape[-1]))


def hilbert_envelope(signal):
    """Calculate the hilbert envelope of a signal

    Also does the FFT on the -1 axis.
    :returns: ndarray of the same shape as the input.
    """
    signal = np.asarray(signal)
    N_orig = signal.shape[-1]
    # Next power of 2.
    N = next_pow_2(N_orig)
    y_h = sp.signal.hilbert(signal, N)
    # Return signal with same dimensions as original
    return np.abs(y_h[..., :N_orig])


def next_pow_2(x):
    """Calculate the next power of 2."""
    return int(pow(2, np.ceil(np.log2(x))))


def fftfilt(b, x, *n):
    """Filter the signal x with the FIR filter described by the
    coefficients in b using the overlap-add method. If the FFT
    length n is not specified, it and the overlap-add block length
    are selected so as to minimize the computational cost of
    the filtering operation.

    From: http://projects.scipy.org/scipy/attachment/ticket/837/fftfilt.py
    """

    x = np.asarray(x)
    b = np.asarray(b)

    N_x = len(x)
    N_b = len(b)

    # Determine the FFT length to use:
    if len(n):

        # Use the specified FFT length (rounded up to the nearest
        # power of 2), provided that it is no less than the filter
        # length:
        n = n[0]
        if n != int(n) or n <= 0:
            raise ValueError('n must be a nonnegative integer')
        if n < N_b:
            n = N_b
        N_fft = 2 ** next_pow_2(n)
    else:

        if N_x > N_b:

            # When the filter length is smaller than the signal,
            # choose the FFT length and block size that minimize the
            # FLOPS cost. Since the cost for a length-N FFT is
            # (N/2)*log2(N) and the filtering operation of each block
            # involves 2 FFT operations and N multiplications, the
            # cost of the overlap-add method for 1 length-N block is
            # N*(1+log2(N)). For the sake of efficiency, only FFT
            # lengths that are powers of 2 are considered:
            N = 2 ** arange(ceil(log2(N_b)), floor(log2(N_x)) + 1)
            cost = ceil(N_x / (N - N_b + 1)) * N * (log2(N) + 1)
            N_fft = N[argmin(cost)]

        else:

            # When the filter length is at least as long as the signal,
            # filter the signal using a single block:
            N_fft = 2 ** next_pow_2(N_b + N_x - 1)

    N_fft = int(N_fft)

    # Compute the block length:
    L = int(N_fft - N_b + 1)

    # Compute the transform of the filter:
    H = fft(b, N_fft)

    y = zeros(N_x, float)
    i = 0
    while i <= N_x:
        il = min([i + L, N_x])
        k = min([i + N_fft, N_x])
        yt = ifft(fft(x[i:il], N_fft) * H, N_fft)  # Overlap..
        y[i:k] = y[i:k] + yt[:k - i]  # and add
        i += L
    return np.real(y)


def write_wav(fname, fs, x):
    """Write floating point numpy array to 16 bit wavfile.

    Convenience wrapper around the scipy.io.wavfile.write function. The signal
    so that its maximum value is one.

    The '.wav' extension is added to the file if it is not part of the
    filename string.

    :fname: string, filename with path.
    :fs: sampling frequency
    :x: array_like, N_channel x Length, signal
    :return: nothing
    """
    # Make sure that the channels are the second dimension
    fs = np.int(fs)
    if not fname.endswith('.wav'):
        fname += '.wav'

    if x.shape[0] <= 2:
        x = x.T

    if x is np.float:
        scaled = (x / np.max(np.abs(x)) * (2 ** 15 - 1))
    else:
        scaled = x
    wavfile.write(fname, fs, scaled.astype('int16'))
