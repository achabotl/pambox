Signal Distortion and Processing
================================

The :mod:`~pambox.distort` module groups together various distortions and
types of processing that can be applied to signals.


* :func:`~pambox.distort.mix_noise` adds together two signals at a given SNR.
* :func:`~pambox.distort.noise_from_signal` creates a noise with the same
  spectrum as the input signal. Optionally, it can also keep the signal's
  envelope.
* :func:`~pambox.distort.overlap_and_add` reconstructs a signal using the
  overlap and add method.
* :func:`~pambox.distort.phase_jitter` applies phase jitter to a signal.
* :func:`~pambox.distort.spec_sub` applies spectral subtraction to a signal.


API
---

.. automodule:: pambox.distort
    :members:


