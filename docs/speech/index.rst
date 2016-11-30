Speech Intelligibility Models and Experiments
=============================================

Introduction
------------

The :mod:`~pambox.speech` module groups together speech intelligibility models
and other tools to facilitate the creation of speech intelligibility
prediction "experiments".


.. _speech-intelligibility-models:

Speech Intelligibility Models
-----------------------------

Speech intelligibility models are classes that take at least a ``fs``
argument. All predictions are done via a ``predict`` method with the
signature: ``predict(clean=None, mix=None, noise=None)``.
This signature allows models to require only a subset of the inputs. For example,
blind models might only require the mixture of processed speech and noise: ``predict(mix=noisy_speech)``; or just the
clean signal and the noise: ``predict(clean=speech, noise=noise)``.

The reference level is that a signal with an RMS value of 1 corresponds to 0 dB SPL.

Here is a small example, assuming that we have access to two signals, ``mix`` which is a mixture of the clean speech
and the noise, and ``noise``, which is the noise alone.

::

    >>> from pambox.speech import Sepsm
    >>> s = Sepsm(fs=22050)
    >>> res = s.predict(mix=mix, noise=noise)


For models that do not take time signals as inputs,
such as the :py:class:`~pambox.speech.Sii`, two other types of interfaces are
defined:

* ``predict_spec`` if the model takes frequency spectra as its inputs. Once
  again, the spectra of the clean speech, of the mixture, and of the noise
  should be provided::

    >>> from pambox.speech import Sii
    >>> s = Sii(fs=22050)
    >>> res = s.predict_spec(clean=clean_spec, noise=noise_spec)


* ``predict_ir`` if the models takes impulse responses as its inputs. The
  function then takes two inputs, the impulse response to the target,
  and the concatenated impulse responses to the maskers::

    >>> from pambox.speech import IrModel
    >>> s = IrModel(fs=22050)
    >>> res = s.predict_ir(clean_ir, noise_irs)

Intelligibility models return a dictionary with **at least** the following key:

* ``p`` (for "predictions"): which is a dictionary with the outputs of the
  model. They keys are the names of the outputs. This allows models to have
  multiple return values. For example, the :py:class:`~pambox.speech.MrSepsm`
  returns two prediction values::

    >>> s = MrSepsm(fs=22050)
    >>> res = s.predict(clean, mix, noise)
    >>> res['p']
    {'lt_snr_env': 10.5, 'snr_env': 20.5}

It might seem a bit over-complicated, but it allows for an easier storing of
the results of an experiment.

Additionally, the models can add any other keys to the results dictionary. For
example, a model can return some of its internal attributes, its internal
representation, etc.


.. _speech-materials:

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
    ...    path_to_sentences='../stimuli/ieee',
    ...    path_to_ssn='ieee_ssn.wav',
    ...    ref_level=74
    ...    name='IEEE'
    ...    )

Each speech file can be loaded using its name::

    >>> x = sm.load_file(sm.files[0])

Or files can be loaded as an iterator::

    >>> all_files = sm.load_files()
    >>> for x in all_files:
    ...    # do some processing on `x`
    ...    pass


By default, the list of files is simply all the files found in
the ``path_to_sentences``. To overwrite this behavior, simply replace the
:py:func:`~pambox.speech.Material.files_list` function::

    >>> def new_files_list():
    ...     return ['file1.wav', 'file2.wav']
    >>> sm.files_list = new_files_list

It is common that individual sentences of a speech material are not adjusted
to the exact same level. This is typically done to compensate for differences
in intelligibility between sentences. In order to keep the inter-sentence
level difference, it is recommended to use the
:py:func:`~pambox.speech.Material.set_level` method of the speech material.
The code below sets the level of the first sentence to 65 dB SPL,
with the reference that a signal with an RMS value of 1 has a level of 0 dB SPL.

    >>> x = sm.load_file(sm.files[0])
    >>> adjusted_x = sm.set_level(x, 65)

Accessing the speech-shaped noise corresponding the speech material is done
using the :func:`~pambox.speech.Material.ssn` function:

    >>> ieee_ssn = sm.ssn()

By default, this will return the entirety of the SSN. However, it is often
required to select a section of noise that is the same length as a target
speech signal, therefore, you can get a random portion of the SSN of the same
length as the signal `x` using:

    >>> ssn_section = sm.ssn(x)

If you are given a speech material but you don't know it's average level, you
can use the help function :func:`~pambox.speech.Material.average_level` to
find the average leve, in dB, of all the sentences in the speech material:

    >>> average_level = sm.average_level()

.. _speech-intelligibility-experiments:

Speech Intelligibility Experiment
---------------------------------

Performing speech intelligibility experiments usually involves a tedious
process of looping through all conditions to study, such as different SNRs,
processing conditions, and sentences. The :class:`~pambox.speech.Experiment`
class simplifies and automates the process of going through all the
experimental conditions. It also gathers all the results in a way that is
simple to manipulate, transform, and plot.

Basic Example
~~~~~~~~~~~~~

An experiment requires at least: a model, a speech material, and a list of SNRs.

    >>> from pambox.speech import Experiment, Sepsm, Material
    >>> models = Sepsm()
    >>> material = Material()
    >>> snrs = np.arange(-9,-5, 3)
    >>> exp = Experiment(models, material, snrs, write=False)
    >>> df = exp.run(2)
    >>> df
     Distortion params   Model    Output  SNR  Sentence number      Value
    0             None   Sepsm   snr_env   -9                0   1.432468
    1             None   Sepsm   snr_env   -6                0   5.165170
    2             None   Sepsm   snr_env   -9                1   6.308387
    3             None   Sepsm   snr_env   -6                1  10.314227

Additionally, you can assign a type of processing, such as reverberation,
spectral subtraction, or any arbitrary type of processing. To keep things
simply, let's apply a compression to the mixture and to the noise. Your
distortion function *must return* the clean speech, the mixture, and the
noise alone.

    >>> def compress(clean, noise, power):
    ...     mixture = (clean + noise) ** (1 / power)
    ...     noise = noise ** (1 / power)
    ...     return clean, mixture, noise
    ...
    >>> powers = range(1, 4)
    >>> exp = Experiment(models, material, snrs, mix_signals, powers)
    >>> df = exp.run(2)
    >>> df


If the distortion parameters are stored in a list of dictionaries,
they will be saved in separate columns in the output dataframe. Otherwise,
they will be saved as tuples in the "Distortion params" column.


API
---

.. automodule:: pambox.speech
    :members:
