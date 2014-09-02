Utilities
=========

The :mod:`~pambox.utils` groups together function that are not auditory
processes but that are, nonetheless, useful or essential to manipulate signals.

Signal levels
-------------

`pambox` defines a reference level for digital signals. The convention is
that a signal with a root-mean-square (RMS) value of 1 corresponds to a level
of 0 dB SPL. In other words:

.. math::
  L [dB SPL] = 20 * \log_{10}\frac{P}{Ref},

where :math:`Ref` is 1.

The functions :py:func:`~pambox.utils.setdbspl`,
:py:func:`~pambox.utils.dbspl`, and :py:func:`~pambox.utils.rms` help in
doing this conversion.

Adding signals and adjusting their lengths
------------------------------------------

Adding together signals loaded from disks is often problematic because they
tend to have different lengths. The functions
:py:func:`~pambox.utils.add_signals` and
:py:func:`~pambox.utils.make_same_length` simplify this. The former simply
adds two signals and pads the shortest one with zeros if necessary. The
latter force two signals to be of the same lengths by either zero-padding the
shortest (default) or by cutting the second signal to the length of the
first, for example::

    >>> a = [1, 1]
    >>> b = [2, 2, 2]
    >>> make_same_length(a, b)
    [1, 1, 0], [2, 2, 2]
    >>> make_same_length(a, b, extend_first=False)
    [1, 1], [2, 2]

This can be useful when using models operating in the envelope domain,
as padding with zeros increase the energy at low modulation frequencies.


The :py:func:`~pambox.utils.int2srt` function finds the speech reception
threshold (SRT) for a given intelligibility curve. It is actually a more
general linear interpolation function, but the most common use case in this
toolbox is to find SRTs.

The function :py:func:`~pambox.utils.psy_fn` calculates a psychometric
function based on a mean (that would be the SRT @ 50%) and a standard
deviation. This function can be useful when trying to fit a psychometric
function to a series of data points.


FFT Filtering and general speedups
----------------------------------

FIR filtering is rather slow when using long impulse responses. The function
:py:func:`~pambox.utils.fftfilt` makes such filtering faster by executing the
filtering using the overlap-and-add method in the frequency domain
rather than as a convolution. It is largely inspired from the Matlab
implementation and was adapted from a `suggested addition to Scipy
<http://projects.scipy.org/scipy/attachment/ticket/837/fftfilt.py>`_.
It might be removed from the toolbox if `fftfilt` becomes a part of Scipy.

The function :py:func:`~pambox.utils.next_pow_2` is a convenient way to
obtain the next power of two for a given integer. It's mostly useful when
picking an FFT length.

API
---

.. automodule:: pambox.utils
   :members:
