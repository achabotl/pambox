CHANGES
=======

master (unreleased)

API changes
-----------

Enhancements
------------

Performance
-----------

Bug fixes
---------

- Fixed #14n in the function py:func:`~pambox.central.mod_filterbank` that made
 the filterbank acausal. The filterbank now produces the same time output as using
 Butterworth filter coefficients and the `scipy.signal.filtfilt` function.
