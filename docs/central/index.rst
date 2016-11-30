Central auditory processing
===========================


The :mod:`~pambox.central` module regroups what is *not* considered to be part of
the outer, middle, or inner ear. It's a rather broad concept.

It contains:

   - An Equalization--Cancellation (EC) stage, in :py:class:`~pambox.central.EC`.
   - An implementation of the EPSM modulation filterbank in :py:class:`~.pambox.central.EPSMModulationFilterbank`.
   - An Ideal Observer, :py:class:`~pambox.central.IdealObs`, as used in the
     :py:class:`~pambox.speech.Sepsm` model.


API
---

.. automodule:: pambox.central
    :members:
