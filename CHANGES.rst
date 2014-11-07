CHANGES
=======

master (unreleased)

API changes
-----------
- Added optional `model` parameter to `Experiment.pred_to_pc` to select only
certain models and model outputs for the conversion to percent correct.
- The `speech.Material` class takes directly the path to the sentences and to
 the speech-shaped noise. This allows the user to use auto-complete.
- The py:func:`~pambox.utils.int2srt` function return `np.nan` if the SRT is
not found, instead of returning None.
- Renamed the SRT parameter in pa:func:`~pambox.utils.int2srt` to `srt_at` instead of `srt`.
- Change the default folder for speech experiment outputs to `output`.
- Add parameter to adjust levels before or after the application of the
distortion in speech intelligibility experiments. The default is to apply it
after the distortion.

Enhancements
------------

- `utils.fftfilt` now mirrors Matlab's behavior. Given coefficients `b` and
signal `x`: If `x` is a matrix, the rows are filtered. If `b` is a matrix,
each filter is applied to `x`. If both `b` and `x` are matrices with the same
number of rows, each row of `x` is filtered with the respective row of `b`.
- The `Experiment` class tries to create the output folder if it does not exist.
- The speech material name is saved out the output data frame when running a
speech intelligibility experiment.
- Added the function py:func:`~pambox.speech.material.Material.average_level`
to measure the average level of a speech material.
- Added py:func:`~pambox.speech.experiment.srts_from_df` to convert
intelligibility predictions to SRTs.
- The py:func:`~pambox.speech.experiment.next_masker` function now takes a
dictionary with all the parameters of the experiment. It does not change the
default behavior of the function be default, but it allows for using that
parameter if the `next_masker` function is overriden.
- The name of the columns saved during a speech intelligibility experiment are
defined as class parameters, rather than being hard-coded.
- Add optional `ax` parameter to py:func:`~pambox.speech.experiment.plot_results`.
- Function py:func:`~pambox.speech.experiment.pred_to_pc` can now convert prediction to intelligibility for a specific model.

Performance
-----------

Bug fixes
---------

- Fixed #14 in the function py:func:`~pambox.central.mod_filterbank` that made
the filterbank acausal. The filterbank now produces the same time output as using
Butterworth filter coefficients and the `scipy.signal.filtfilt` function.
- Fix #16: the ideal observer fits the average intelligibility, across all
sentences, to the reference data, rather than trying to fit all sentences at
once.
- Fix #17: Removed unnecessary compensation factor in the sEPSM. It
compensated for the filter bandwidth when computing the bands above threshold
. The tests tolerance had to be adjusted; for the spectral subtraction case,
the relative difference compared to the Matlab code is smaller than 8%. In
the condition with speech-shaped noise only, the difference is smaller than 0
.1%.
- The py:func:`~pambox.speech.material.Material.set_level` function uses
compensates for the reference sentence level using the correct sign.
