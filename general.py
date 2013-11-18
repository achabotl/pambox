from __future__ import division
import numpy as np
import scipy as sp


def dbspl(x, ac=False):
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


    :x: signal
    :ac: @todo
    :returns: @todo

    """
    return 20. * np.log10(rms(x, ac)) + 100.


def setdbspl(x, lvl, ac=False):
    """set level if signal in dB

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
    return x / rms(x, ac) * 10. ** ((lvl - 100.) / 20.)


def rms(x, ac=True):
    """RMS value of a signal

    :x: signal
    :ac: bool, default: True
        consider only the AC component of the signal
    :rms: rms value

    """
    if ac:
        return np.std(x, axis=-1)
    else:
        return np.std(x, axis=-1) + np.mean(x, axis=-1)


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
