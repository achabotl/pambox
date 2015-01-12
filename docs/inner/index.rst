Inner ear processing
====================

This module groups properties and processes of the inner ear, namely
peripheral filtering and envelope extraction.

Filterbanks
-----------

All filterbanks provide a `filter` method that takes only the input signal.
The filterbank's parameters must be defined as part of the class.

* :class:`~pambox.inner.GammatoneFilterbank` is a gammatone filterbank which
  uses the Malcom Slaney implementation.


Envelope extraction
-------------------

* :func:`~pambox.inner.hilbert_envelope` extract the Hilbert envelope of a
  signal.
* :func:`~pambox.inner.lowpass_env_filtering` lowpass filters a signal using
  a Betterworth filter.


Other functions
---------------

* :func:`~pambox.inner.erbbw` give the ERB bandwith given center frequencies.
* :func:`~pambox.inner.noctave_filtering` performs bandpass filtering of a
  signal using rectangular filters.




API
---

.. automodule:: pambox.inner
   :members:
