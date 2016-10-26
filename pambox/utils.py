# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

from matplotlib import pyplot as plt
import numpy as np
from numpy import min, log2, ceil, argmin, zeros, arange, complex
try:
    _ = np.use_fastnumpy  # Use Enthought MKL optimizations
    from numpy.fft import fft, ifft
except AttributeError:
    try:
        import mklfft  # MKL FFT optimizations from Continuum Analytics
        from numpy.fft import fft, ifft
    except ImportError:
        # Finally, just use Scipy's
        from scipy.fftpack import fft, ifft

import scipy as sp
from scipy import signal as ss
from scipy.io import wavfile
import scipy.special


def dbspl(x, ac=False, offset=0.0, axis=-1):
    """Computes RMS value of signal in dB.

    By default, a signal with an RMS value of 1 will have a level of 0 dB
    SPL.

    Parameters
    ----------
    x : array_like
        Signal for which to caculate the sound-pressure level.
    ac : bool
        Consider only the AC component of the signal, i.e. the mean is
        removed (Default value =  False)
    offset : float
        Reference to convert between RMS and dB SPL.  (Default value = 0.0)
    axis : int
        Axis on which to compute the SPL value (Default value = -1, last axis)

    Returns
    -------
    ndarray
        Sound-pressure levels.

    References
    ----------
    .. [1] Auditory Modeling Toolbox, Peter L. Soendergaard
      B. C. J. Moore. An Introduction to the Psychology of Hearing. Academic
      Press, 5th edition, 2003.

    See also
    --------
    setdbspl
    rms
    """
    x = np.asarray(x)
    return 20. * np.log10(rms(x, ac)) + float(offset)


def setdbspl(x, lvl, ac=False, offset=0.0):
    """Sets the level of signal in dB SPL, along its last dimension.

    Parameters
    ----------
    x : array_like
        Signal.
    lvl : float
        Level, in dB SPL, at which to set the signal.
    ac : bool
        Calculate the AC RMS power of the signal by default (`ac=True`),
        e.g. the mean is removed. If  `False`, considers the non-RMS power.
        (Default value = False)
    offset : float
        Level, in dB SPL, corresponding to an RMS of 1. By default, an RMS of
        1 corresponds to 0 dB SPL, i.e. the default is 0.

    Returns
    -------
    ndarray
        Signal of the same dimension as the original.
    """
    x = np.asarray(x)
    if np.isinf(lvl) and lvl < 0:
        return np.zeros_like(x)

    if x.ndim > 1:
        rms_value = rms(x, ac)[(slice(None),) + (x.ndim - 1) * (np.newaxis, )]
    else:
        rms_value = rms(x, ac)
    return x / rms_value * 10. ** ((lvl - float(offset)) / 20.)


def rms(x, ac=False, axis=-1):
    """Calculates the RMS value of a signal.

    Parameters
    ----------
    x : array_like
        Signal.
    ac : bool
        Consider only the AC component of the signal. (Default value = False)
    axis :
        Axis on which to calculate the RMS value. The default is to calculate
        the RMS on the last dimensions, i.e. axis = -1.

    Returns
    -------
    ndarray
        RMS value of the signal.

    """
    x = np.asarray(x)
    if ac:
        if x.ndim > 1 and axis == -1:
            x_mean = x.mean(axis=axis)[..., np.newaxis]
        else:
            x_mean = x.mean(axis=axis)
        return np.linalg.norm((x - x_mean)
                              / np.sqrt(x.shape[axis]), axis=axis)
    else:
        return np.linalg.norm(x / np.sqrt(x.shape[axis]), axis=axis)


def hilbert(x, N=None, axis=-1):
    """Computes the analytic signal using the Hilbert transform.

    The transformation is done along the last axis by default.

    Parameters
    ----------
    x : array_like
        Signal data.  Must be real.
    N : int, optional
        Number of Fourier components.  Default: ``x.shape[axis]``
    axis : int, optional
        Axis along which to do the transformation.  Default: -1.

    Returns
    -------
    xa : ndarray
        Analytic signal of `x`, of each 1-D array along `axis`

    Notes
    -----

    **NOTE**: This code is a copy-paste from the Scipy codebase. By
    redefining it here, it is possible to take advantage of the speed
    increase provided by  using the MKL's FFT part of Enthough's distribution.

    The analytic signal ``x_a(t)`` of signal ``x(t)`` is:

    .. math:: x_a = F^{-1}(F(x) 2U) = x + i y

    where `F` is the Fourier transform, `U` the unit step function,
    and `y` the Hilbert transform of `x`. [1]_

    In other words, the negative half of the frequency spectrum is zeroed
    out, turning the real-valued signal into a complex signal.  The Hilbert
    transformed signal can be obtained from ``np.imag(hilbert(x))``, and the
    original signal from ``np.real(hilbert(x))``.

    References
    ----------
    .. [1] Wikipedia, "Analytic signal".
           http://en.wikipedia.org/wiki/Analytic_signal

    License
    -------
    This code was copied from Scipy. The following license applies for this
    function:

    Copyright (c) 2001, 2002 Enthought, Inc.
    All rights reserved.

    Copyright (c) 2003-2012 SciPy Developers.
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

      a. Redistributions of source code must retain the above copyright notice,
         this list of conditions and the following disclaimer.
      b. Redistributions in binary form must reproduce the above copyright
         notice, this list of conditions and the following disclaimer in the
         documentation and/or other materials provided with the distribution.
      c. Neither the name of Enthought nor the names of the SciPy Developers
         may be used to endorse or promote products derived from this software
         without specific prior written permission.


    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS
    BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
    OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
    SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
    INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
    CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
    THE POSSIBILITY OF SUCH DAMAGE.
    """
    x = np.asarray(x)
    if np.iscomplexobj(x):
        raise ValueError("x must be real.")
    if N is None:
        N = x.shape[axis]
    if N <= 0:
        raise ValueError("N must be positive.")

    Xf = fft(x, N, axis=axis)
    h = np.zeros(N)
    if N % 2 == 0:
        h[0] = h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        h[0] = 1
        h[1:(N + 1) // 2] = 2

    if len(x.shape) > 1:
        ind = [np.newaxis] * x.ndim
        ind[axis] = slice(None)
        h = h[ind]
    x = ifft(Xf * h, axis=axis)
    return x


def next_pow_2(x):
    """Calculates the next power of 2 of a number.

    Parameters
    ----------
    x : float
        Number for which to calculate the next power of 2.

    Returns
    -------
    int

    """
    return int(pow(2, np.ceil(np.log2(x))))


def fftfilt(b, x, n=None):
    """FIR filtering using the FFT and the overlap-add method.

    Filters the data in `x` using the FIR coefficients in `b`. If `x` is a
    matrix, the rows are filtered. If `b` is a matrix, each filter is applied
    to `x`. If both `b` and `x` are matrices with the same number of rows,
    each row of `x` is filtered with the respective row of `b`.

    Parameters
    ----------
    b : array_like
        Coefficients of the FIR filter.
    x : array_like
        Signal to filter.
    n : int, optional.
        Length of the FFT. If `n` is not provided, a value of `n` will be
        chosen by `fftfilt`. See Notes for details.

    Returns
    -------
    y : ndarray
        Filtered signal.

    Notes
    -----
    Filter the signal `x` with the FIR filter described by the
    coefficients in `b` using the overlap-add method. If the FFT
    length `n` is not specified, it and the overlap-add block length
    are selected so as to minimize the computational cost of
    the filtering operation.

    If `x` is longer than `b`, then `n` and `L` will be chosen as to minimize
    the product of the number of blocks and the number of flops per FFT.

    If a value of `n` is provided, the FFT length will be the next power of 2
    after `n` and each block of data will be of length `N_fft - N_b + 1`. If
    `n` is smaller than the length of `b`, the FFT length will be the length
    of `b`.

    Examples
    --------
    >>> import pambox.utils
    >>> b = [1, 1]
    >>> x = [0, 1, 2, 3, 4, 5]
    >>> y = pambox.utils.fftfilt(b, x)

    The FFT length can also be specified:
    >>> y = pambox.utils.fftfilt(b, x, 16)
    """
    x = np.asarray(x)
    b = np.asarray(b)

    if b.ndim > 1 and x.ndim > 1 and (b.shape[0] != x.shape[0]):
        raise ValueError(
            "b and x must have the same number of dimensions if they have "
            "more than 1.")

    N_x = x.shape[-1]
    N_b = b.shape[-1]

    # Determine the FFT length to use:
    if n:
        # Use the specified FFT length (rounded up to the nearest
        # power of 2), provided that it is no less than the filter
        # length:
        if n != int(n) or n <= 0:
            raise ValueError('n must be a non-negative integer.')
        if n < N_b:
            n = N_b
        N_fft = 2 ** ceil(log2(np.abs(n)))
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
            N = 2 ** arange(ceil(log2(N_b)), 27)
            cost = ceil(N_x / (N - N_b + 1)) * N * (log2(N) + 1)
            N_fft = N[np.argmin(cost)]
        else:
            # When the filter length is at least as long as the signal,
            # filter the signal using a single block:
            N_fft = 2 ** ceil(log2((N_b + N_x - 1)))

    N_fft = int(N_fft)

    # Compute the block length:
    L = int(N_fft - N_b + 1)

    # Compute the transform of the filter:
    B = fft(b, N_fft)

    if b.ndim == 1 and x.ndim > 1:
        # Replicate the rows of B
        B = np.tile(B, (x.shape[0], 1))
    if x.ndim == 1 and B.ndim > 1 :
        x = np.tile(x, (B.shape[0], 1))

    y = zeros(x.shape, np.complex)
    i = 0
    while i <= N_x:
        il = np.min([i + L, N_x])
        k = np.min([i + N_fft, N_x])
        yt = ifft(B * fft(x[..., i:il], N_fft), N_fft)
        y[..., i:k] = y[..., i:k] + yt[..., :k - i]
        i += L
    return np.real(y)


def write_wav(fname, fs, x, normalize=False):
    """Writes floating point numpy array to 16 bit wavfile.

    Convenience wrapper around the scipy.io.wavfile.write function.

    The '.wav' extension is added to the file if it is not part of the
    filename string.

    Inputs of type `np.float` are converted to `int16` before writing to file.

    Parameters
    ----------
    fname : string
        Filename with path.
    fs : int
        Sampling frequency.
    x : array_like
        Signal with the shape N_channels x Length
    normalize : bool
        Scale the signal such that its maximum value is one.

    Returns
    -------
    None

    """
    # Make sure that the channels are the second dimension
    fs = np.int(fs)
    if not fname.endswith('.wav'):
        fname += '.wav'

    if x.shape[0] <= 2:
        x = x.T

    if np.issubdtype(x.dtype, np.float) and normalize:
        scaled = (x / np.max(np.abs(x)) * (2 ** 15 - 1))
    elif np.issubdtype(x.dtype, np.float):
        scaled = x * (2 ** 15 - 1)
    else:
        scaled = x
    wavfile.write(fname, fs, scaled.astype('int16'))

def make_same_length(a, b, extend_first=True):
    """Make two vectors the same length.

    Parameters
    ----------
    a,b : array_like
        Arrays to make of the same length.
    extend_first : bool, optional
        Zero-pad the first array if it is the shortest if `True`. Otherwise,
        cut array `b` to the length of `a`. (Default value = True)

    Returns
    -------
    tuple of ndarrays
        Two arrays with the same length along the last dimension.

    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape[-1] < b.shape[-1]:
        if extend_first:
            c = np.zeros_like(b)
            c[..., :a.shape[-1]] += a
            return c, b
        else:
            return a, b[..., :a.shape[-1]]
    else:
        c = np.zeros_like(a)
        c[..., :b.shape[-1]] += b
        return a, c


def add_signals(a, b):
    """Adds two vectors of different lengths by zero padding the shortest one.

    Parameters
    ----------
    a,b : ndarray
        Arrays to make of the same length.

    Returns
    -------
    ndarray
        Sum of the signal, of the same length as the longest of the two inputs.

    """
    if len(a) < len(b):
        c = b.copy()
        c[:len(a)] += a
    else:
        c = a.copy()
        c[:len(b)] += b
    return c


def int2srt(x, y, srt_at=50.0):
    """Finds intersection using linear interpolation.

    This function finds the x values at which a curve intersects with a
    constant value.

    Parameters
    ----------
    x : array_like
        "x" values
    y : array_like
        "y" values
    srt_at : float
         Value of `y` at which the interception is calculated. (Default value
         = 50.0)

    Returns
    -------
    float

    """
    x = np.asarray(x)
    y = np.asarray(y)

    if x.shape != y.shape:
        raise ValueError('Inputs of different lenghts.')
    srt_at = np.float(srt_at)
    idx = np.nonzero(np.diff(y >= srt_at))[0]
    if not idx.any() and y[0] == srt_at:
        return x[0]

    if len(idx) >= 1:
        idx = idx[0]
        srt = x[idx] + (srt_at - y[idx]) * (x[idx + 1] - x[idx]) \
            / (y[idx + 1] - y[idx])
    else:
        srt = np.nan
    return float(srt)


def psy_fn(x, mu=0., sigma=1.):
    """Calculates a psychometric function with a given mean and variance.

    Parameters
    ----------
    x : array_like
        "x" values of the psychometric functions.
    mu : float, optional
        Value at which the psychometric function reaches 50%, i.e. the mean
        of the distribution. (Default value = 0)
    sigma : float, optional
        Variance of the psychometric function. (Default value = 1)

    Returns
    -------
    pc : ndarray
        Array of "percent correct", between 0 and 100.

    """
    x = np.asarray(x)
    return 100 * sp.special.erfc(-(x - mu) / (np.sqrt(2) * sigma)) / 2


def noctave_center_freq(lowf, highf, width=3):
    """Calculate exact center N-octave space center frequencies.

    In practive, what is often desired is the "simplified" center frequencies,
    so this function is not of much use.

    Parameters
    ----------
    lowf : float
        Lowest frequency.
    highf : float
        Highest frequency
    width : float
         Number of filters per octave. (Default value = 3)

    Returns
    -------
    ndarray
        List of center frequencies.

    """
    n_centers = np.log2(highf / lowf) * width + 1
    n_octave = np.log2(highf / lowf)
    return lowf * np.logspace(0, n_octave, num=n_centers, base=2)


def impz(b, a=1):
    """Plot step and impulse response of an FIR filter.

    b : float
        Forward terms of the FIR filter.
    a : float
        Feedback terms of the FIR filter. (Default value = 1)

    From http://mpastell.com/2010/01/18/fir-with-scipy/

    Returns
    -------
    None

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


def mfreqz(b, a=1, fs=22050.0):
    """Plot the frequency and phase response of an FIR filter.

    From http://mpastell.com/2010/01/18/fir-with-scipy/

    Parameters
    ----------
    b : float
        Forward terms of the FIR filter.
    a : float
        Feedback terms of the FIR filter. (Default value = 1)
    fs : float
        Sampling frequency of the filter. (Default value = 22050.0)

    Returns
    -------
    None

    """
    w, h = ss.freqz(b, a)
    h_db = 20 * np.log10(abs(h))
    plt.subplot(211)
    if fs:
        f = sp.linspace(0, fs / 2, len(w))
        plt.plot(f, h_db)
    else:
        plt.plot(w / max(w), h_db)
    plt.ylim(-150, 5)
    plt.ylabel('Magnitude (db)')
    plt.xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
    plt.title(r'Frequency response')
    plt.subplot(212)
    h_phase = sp.unwrap(sp.arctan2(sp.imag(h), sp.real(h)))
    plt.plot(w / max(w), h_phase)
    plt.ylabel('Phase (radians)')
    plt.xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
    plt.title(r'Phase response')
    plt.subplots_adjust(hspace=0.5)


def read_wav_as_float(path):
    """Reads a wavefile as a float.

    Parameters
    ----------
    path : string
        Path to the wave file.

    Returns
    -------
    wav : ndarray
    """
    _, signal = scipy.io.wavfile.read(path)
    if np.issubdtype(signal.dtype, np.integer):
        return signal.T / np.abs(np.iinfo(signal.dtype).min)
    return signal.T
