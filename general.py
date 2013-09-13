from __future__ import division
import numpy as np
import scipy as sp
from scipy.signal import hilbert


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
    return 20. * np.log10(rms(x, ac=False)) + 100.


def setdbspl(x, lvl, ac=False):
    """set level if signal in dB

    SETDBSPL(insig,lvl) sets the SPL (sound pressure level) of the signal
    insig to lvl dB, using the convention that a pure tone with an RMS value
    of 1 corresponds to 100 dB SPL.

    If the input is a matrix, it is assumed that each row is a signal.

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
    return np.abs(hilbert(signal))
