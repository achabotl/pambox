Speech Intelligibility Models and Experiments
=============================================

Introduction
------------

The :py:module:`~speech` module groups together speech intelligibility models
 and varios other tools to facilitate the creation of speech intelligibility
 prediction "experiments". Each model presents a standard `predict` function


Speech Intelligibility Models
-----------------------------

:py:class:`~pambox..Sepsm`


:py:class:`~pambox.speech.MrSepsm`



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
