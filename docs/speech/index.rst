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
    >>> s.predict_spec(clean, mix, noise)


For models that do take time signals as inputs,
such as the :py:class:`~pambox.speech.Sii`, two other types of interfaces are
defined:

* `predict_spec` if the model takes frequency spectra as its inputs. Once
  again, the spectra of the clean speech, of the mixture, and of the noise
  should be provided::

    >>> from pambox.speech import Sii
    >>> s = Sii()
    >>> s.predict_spec(clean_spec, mix_spec, noise_spec)


* `predict_ir` if the models takes impulse responses as its inputs. The
  function then takes two inputs, the impulse response to the target,
  and the concatenated impulse responses to the maskers::

    >>> from pambox.speech import IrModel
    >>> s = IrModel()
    >>> s.predict_ir(clean_ir, noise_irs)


Speech Materials
----------------

The :py:class:`~pambox.speech.SpeechMaterial`  class simplifies the
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
