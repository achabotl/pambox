# -*- coding: utf-8 -*-
"""
The :mod:`pambox.speech.material` module gathers classes to facilitate
working with different speech materials.
"""
from __future__ import absolute_import, division, print_function
import glob
import logging
import os
import random

import numpy as np
import scipy.io.wavfile
from six.moves import zip, range

from pambox import utils

log = logging.getLogger(__name__)


class Material(object):
    """Load and manipulate speech materials for intelligibility experiments"""

    def __init__(self,
                 fs=22050,
                 path_to_sentences='../stimuli/clue/sentencesWAV22',
                 path_to_maskers=None,
                 path_to_ssn='../stimuli/clue/SSN_CLUE22.wav',
                 ref_level=74,
                 name='CLUE',
                 force_mono=False):
        """

        """
        self.fs = fs
        self.path_to_sentences = path_to_sentences
        self.path_to_maskers = path_to_maskers
        self.ref_level = ref_level
        self.name = name
        self._ssn = None
        self._path_to_ssn = None
        self.path_to_ssn = path_to_ssn
        self.force_mono = force_mono
        self._files = None
        self._audio_ext = '.wav'

    @property
    def files(self):
        if not self._files:
            self._files = self.files_list()
        return self._files

    @files.setter
    def files(self, f):
        self._files = f

    @property
    def path_to_ssn(self):
        return self._path_to_ssn

    @path_to_ssn.setter
    def path_to_ssn(self, path):
        if path:
            self._path_to_ssn = path
            self._ssn = self._load_ssn()

    def load_file(self, filename):
        """Read a speech file by name.

        Parameters
        ----------
        filename : string
            Name of the file to read. The file just be in the directory
            defined by `root_path` and `path_to_sentences`.

        Returns
        -------
        ndarray
            Wav file read from disk, as floating point array.
        """
        path = os.path.join(self.path_to_sentences, filename)
        log.info('Reading file %s', path)
        _, int_sentence = scipy.io.wavfile.read(path)
        sent = int_sentence.T / np.iinfo(int_sentence.dtype).min
        if self.force_mono and sent.ndim == 2:
            return sent[1]
        else:
            return sent

    def files_list(self):
        """Return a list of all the .wav files in the `path_to_sentences`
        directory.

        Returns
        -------
        files : list
            List of all files.
        """
        path = os.path.join(self.path_to_sentences, '')
        log.info("Listing files from directory: %s", path)
        return [os.path.basename(each) for each in glob.glob(path + '*' +
                self._audio_ext)]

    def load_files(self, n=None):
        """Read files from disk, starting from the first one.

        Parameters
        ----------
        n : int, optional
            Number of files to read. Default (`None`) is to read all files.

        Returns
        -------
        generator
            Generator where each item is an `ndarray` of the file loaded.
        """
        if not n:
            n = len(self.files)

        for _, name in zip(list(range(n)), self.files):
            yield self.load_file(name)

    def _load_ssn(self):
        try:
            filepath = self.path_to_ssn
            _, int_sentence = scipy.io.wavfile.read(filepath)
            ssn = int_sentence.T / np.iinfo(int_sentence.dtype).min
        except IOError:
            raise IOError('File not found: %s' % filepath)
        return ssn

    @staticmethod
    def pick_section(signal, section=None):
        """Pick section of signal

        Parameters
        ----------
        section : int or ndarray, optional
            If an integer is given, returns section of length `n`
            Alternatively, if `section` is an ndarray the signal returned
            will be of the same length as the `section` signal. If `x` is
            `None`, the full signal is returned.
        Returns
        -------
        ndarray
            Speech-shaped noise signal.
        """
        len_noise = signal.shape[-1]
        if section is None:
            len_sig = len_noise
            ii = 0
        elif isinstance(section, int):
            len_sig = section
            ii = np.random.randint(0, len_noise - len_sig)
        else:
            len_sig = np.asarray(section).shape[-1]
            ii = np.random.randint(0, len_noise - len_sig)
        return signal[..., ii:ii + len_sig]

    def ssn(self, x=None):
        """Returns the speech-shaped noise appropriate for the speech material.

        Parameters
        ----------
        x : int or ndarray, optional
            If an integer is given, returns a speech-shaped noise of length
            `n` Alternatively,  if a sentenced is given,  the speech-shaped
            noise  returned will be of the same length as the input signal.
            If `x` is `None`, the full SSN signal is returned.
        Returns
        -------
        ndarray
            Speech-shaped noise signal.
        """
        section = self.pick_section(self._ssn, x)
        if self.force_mono and section.ndim > 1:
            return section[0]
        return section

    def set_level(self, x, level):
        """Set level of a sentence, in dB.

        Parameters
        ----------
        x : ndarray
            sentence
        level : float
            Level, in dB, at which the sentences are recorded. The reference
            is that and RMS of 1 corresponds to 0 dB SPL.

        Returns
        -------
        array_like
            Adjusted sentences with a `level` db SPL with the reference
            that a signal with an RMS of 1 corresponds to 0 db SPL.
        """
        return x * 10 ** ((level - self.ref_level) / 20)

    def average_level(self):
        """Calculate the average level across all sentences.

        The levels are calculated according to the toolbox's reference
        level.

        Returns
        -------
        mean : float
            Mean level across all sentences, in dB SPL.
        std : float
            Standard deviation of the levels across all sentences.

        See also
        --------
        utils.dbspl
        """
        spl = [utils.dbspl(x) for x in self.load_files()]
        return np.mean(spl), np.std(spl)

    def create_ssn(self, files=None, repetitions=200):
        """Creates a speech-shaped noise from the sentences.

        Creates a speech-shaped noise by randomly adding together sentences
        from the speech material. The output noise is 75% the length of all
        concatenated sentences.

        Parameters
        ----------
        files : list, optional
            List of files to concatenate. Each file should be an `ndarray`.
            If `files` is None, all the files from the speech material
            will be used. They are loaded with the method `load_files()`.
        repetitions : int
            Number of times to superimpose the randomized sentences. The
            default is 120 times.

        Returns
        -------
        ssn : ndarray

        Notes
        -----
        Before each addition, the random stream of sentences is jittered to
        prevent perfect alignment of all sentences. The maximum jitter is
        equal to 25% of the length of the concatenated sentences.
        """
        if files is None:
            files = [each for each in self.load_files()]
        ssn = np.hstack(files)
        n_output = int(0.75 * ssn.shape[-1])
        max_jitter = ssn.shape[-1] - n_output
        ssn = ssn[..., :n_output]
        for _ in range(repetitions):
            random.shuffle(files)
            start = np.random.randint(max_jitter)
            ssn += np.hstack(files)[..., start:start + n_output]
        ssn /= np.sqrt(repetitions)
        return ssn

    def create_filtered_ssn(self, files=None, duration=5):
        """Create speech-shaped noise based on the average long-term spectrum
        of the speech material.

        Parameters
        ----------
        files : list, optional
            List of files to concatenate. Each file should be an `ndarray`.
            If `files` is None, all the files from the speech material
            will be used. They are loaded with the method `load_files()`.
        duration : float, optional
            Duration of the noise, in seconds. The default is 5 seconds.

        Returns
        -------
        ssn : ndarray
        """
        if files is None:
            files = tuple(self.load_files())

        # Find maximum sentence length
        max_len = reduce(lambda x, y: max(x, y.shape[-1]), files, 0)
        n_fft = utils.next_pow_2(max_len)

        # Calculate the average spectra
        LONG_TERM_SPEC = reduce(lambda x, y: (x + np.fft.rfft(y, n_fft)) / 2,
                                files,
                                0)

        average_masker = np.real(np.fft.irfft(LONG_TERM_SPEC, n=n_fft))[..., :max_len]

        n_noise = duration * self.fs
        if average_masker.ndim > 1:
            noise_shape = [average_masker.shape[0], n_noise]
        else:
            noise_shape = [n_noise]

        n_fft_noise = utils.next_pow_2(n_noise)
        NOISE = np.fft.rfft(np.random.randn(*noise_shape), n_fft_noise)
        NOISE /= np.abs(NOISE)
        NOISE *= np.abs(np.fft.rfft(average_masker, n_fft_noise))
        ssn = np.real(np.fft.irfft(NOISE, n_fft_noise))[..., :n_noise]
        return ssn
