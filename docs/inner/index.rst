Inner ear processing
====================

This module groups properties and processes of the inner ear, namely
peripheral filtering and envelope extraction.

Filterbanks
-----------

All filterbanks provide a ``filter()`` method that takes only the input signal.
The filterbank's parameters must be defined when creating the filterbank. For example,
here we create a Gammatone filterbank for a sampling frequency of 44.1 kHz and a sequence
of octave-spaced center frequencies::

   >>> import numpy as np
   >>> from pambox.inner import GammatoneFilterbank
   >>> g = GammatoneFilterbank(44100, [250, 500, 1000, 2000, 4000])
   >>> x = np.random.randn(2 * 44100)
   >>> y = g.filter(x)
   >>> y.shape
   (5, 88200)

* :class:`~pambox.inner.GammatoneFilterbank` is a gammatone filterbank which uses Malcom Slaney's implementation.
* :class:`~pambox.inner.RectangularFilterbank` performs bandpass filtering of a signal using rectangular filters.


Envelope extraction
-------------------

* :func:`~pambox.inner.hilbert_envelope` extracts the Hilbert envelope of a
  signal.
* :func:`~pambox.inner.lowpass_env_filtering` low-pass filters a signal using
  a Butterworth filter.


Other functions
---------------

* :func:`~pambox.inner.erb_bandwidth` gives the ERB bandwidth for a given center
  frequencies.


API
---

.. automodule:: pambox.inner
   :members:
