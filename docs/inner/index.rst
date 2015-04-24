Inner ear processing
====================

This module groups properties and processes of the inner ear, namely
peripheral filtering and envelope extraction.

Filterbanks
-----------

All filterbanks provide a ``filter()`` method that takes only the input signal.
The filterbank's parameters must be defined as part of the class.

* :class:`~pambox.inner.GammatoneFilterbank` is a gammatone filterbank which
  uses the Malcom Slaney implementation.
* :class:`~pambox.inner.RectangularFilterbank` performs bandpass filtering of a
  signal using rectangular filters.


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
