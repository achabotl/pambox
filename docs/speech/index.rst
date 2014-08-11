Speech Intelligibility Models and Experiments
=============================================

Introduction
------------

The :mod:`~pambox.speech` module groups together speech intelligibility models
and various other tools to facilitate the creation of speech intelligibility
prediction "experiments".


Speech Intelligibility Models
-----------------------------

Each model presents a standard ``predict`` function that takes the clean speech
signal, the processed speech (or the mixture of the speech and noise),
and the noise alone. The reference level is that a signal with an RMS value
of 1 corresponds to 0 dB SPL.

::

    >>> from pambox.speech import Sepsm
    >>> s = Sepsm()
    >>> res = s.predict_spec(clean, mix, noise)


For models that do take time signals as inputs,
such as the :py:class:`~pambox.speech.Sii`, two other types of interfaces are
defined:

* ``predict_spec`` if the model takes frequency spectra as its inputs. Once
  again, the spectra of the clean speech, of the mixture, and of the noise
  should be provided::

    >>> from pambox.speech import Sii
    >>> s = Sii()
    >>> res = s.predict_spec(clean_spec, mix_spec, noise_spec)


* ``predict_ir`` if the models takes impulse responses as its inputs. The
  function then takes two inputs, the impulse response to the target,
  and the concatenated impulse responses to the maskers::

    >>> from pambox.speech import IrModel
    >>> s = IrModel()
    >>> res = s.predict_ir(clean_ir, noise_irs)

Intelligibility models return a dictionary with **at least** the following key:

* ``p`` (for "predictions"): which is a dictionary with the outputs of the
  model. They keys are the names of the outputs. This allows models to have
  multiple return values. For example, the :py:class:`~pambox.speech.MrSepsm`
  returns two predictions values::

    >>> s = MrSepsm()
    >>> res = s.predict(clean, mix, noise)
    >>> res['p']
    {'lt_snr_env': 10.5, 'snr_env': 20.5}

It might seem a bit over-complicated, but it allows for an easier storing of
the results of an experiment.

Additionally, the models can add another keys to the results dictionary. For
example, a model can return some of its internal attributes,
its internal representation, etc.

Speech Materials
----------------

The :py:class:`~pambox.speech.Material`  class simplifies the
access to the speech files when doing speech intelligibility prediction
experiments.

When creating the class, you have to define:

* where the sentences can be found
* their sampling frequency
* their reference level, in dB SPL (the reference is that a signal with an
  RMS value of 1 corresponds to 0 dB SPL),
* as well as the path to a file where the corresponding speech-shaped noise for
  this particular material can be found.

For example, to create a speech material object for IEEE sentences stored in
the `../stimuli/ieee` folder::

    >>> sm = SpeechMaterial(
    ...    fs=25000,
    ...    root_path='../stimuli/ieee',
    ...    path_to_ssn='ieee_ssn.wav',
    ...    ref_level=74
    ...    name='IEEE'
    ...    )

Each speech file can be loaded using its name::

    >>> x = sm.load_file(sm.list[0])

Or files can be loaded as an iterator::

    >>> all_files = sm.load_files()
    >>> for x in all_files:
    ...    # do some processing on `x`
    ...    pass


By default, the list of files is simply all the files found in
the `root_path`. To overwrite this behavior, simply replace the
:py:func:`~pambox.speech.Material.files_list` function::

    >>> def new_files_list():
    ...     return ['file1.wav', 'file2.wav']
    >>> sm.files_list = new_files_list

It is common that individual sentences of a speech material are not adjusted
to the exact same level. This is typically done to compensate for differences
in intelligibility between sentences. In order to keep the inter-sentence
level difference, it is recommended to use the
:py:func:`~pambox.speech.Material.set_level` method of the speech material.
The code below sets the levelo of the first sentence to 65 dB SPL,
with the reference that a signal with an RMS value of 1 has a level of 0 dB SPL.

    >>> x = sm.load_file(sm.files[0])
    >>> adjusted_x = sm.set_level(x, 65)




API
---

.. automodule:: pambox.speech
   :members:
