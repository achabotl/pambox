CHANGES
=======

master (unreleased)

API changes
-----------

Enhancements
------------

- `utils.fftfilt` now mirrors Matlab's behavior. Given coefficients `b` and
signal `x`: If `x` is a matrix, the rows are filtered. If `b` is a matrix,
each filter is applied to `x`. If both `b` and `x` are matrices with the same
number of rows, each row of `x` is filtered with the respective row of `b`.
- The `Experiment` class tries to create the output folder if it does not exist.

Performance
-----------

Bug fixes
---------

- Fixed #14n in the function py:func:`~pambox.central.mod_filterbank` that made
 the filterbank acausal. The filterbank now produces the same time output as using
 Butterworth filter coefficients and the `scipy.signal.filtfilt` function.
