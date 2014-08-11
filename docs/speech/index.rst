Speech Intelligibility Models and Experiments
=============================================

Introduction
------------

The :mod:`~pambox.speech` module groups together speech intelligibility models
and various other tools to facilitate the creation of speech intelligibility
prediction "experiments".


Speech Intelligibility Models
-----------------------------

Each model presents a standard `predict` function that takes the clean speech
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

* `predict_spec` if the model takes frequency spectra as its inputs. Once
  again, the spectra of the clean speech, of the mixture, and of the noise
  should be provided::

    >>> from pambox.speech import Sii
    >>> s = Sii()
    >>> res = s.predict_spec(clean_spec, mix_spec, noise_spec)


* `predict_ir` if the models takes impulse responses as its inputs. The
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

When creating the class, you have to define where the sentences can be found,
their sampling frequency, their reference level, in dB SPL (the reference is
that a signal with an RMS value of 1 corresponds to 0 dB SPL),
as well as the path to a file wher

For example, to create a speech material object for IEEE sentences stored in
the `../stimuli/ieee` folder::

    sm = SpeechMaterial(
        fs=25000,
        root_path='../stimuli/ieee',
        path_to_ssn='ieee_ssn.wav',
        ref_level=74
        name='IEEE'
        )

By default, the :py:attr:`~pambox.speech.speechmaterial.SpeechMaterial.files`


API
---

.. automodule:: pambox.speech
   :members:
