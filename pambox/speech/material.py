# -*- coding: utf-8 -*-
"""
The :mod:`pambox.speech.material` module gathers classes to facilitate
working with different speech materials.
"""
from __future__ import division, print_function, absolute_import
import logging
import os

import numpy as np
import scipy.io.wavfile
from six.moves import zip, range

from .. import utils

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

    @property
    def files(self):
        return self.files_list()

    @property
    def path_to_ssn(self):
        return self._path_to_ssn

    @path_to_ssn.setter
    def path_to_ssn(self, path):
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
        """Return a list of all the files in the corpus.

        :return: list of str, list of all CRM files.
        """
        path = os.path.join(self.path_to_sentences)
        log.info("Listing files from directory: %s", path)
        return os.listdir(path)

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

        for _, name in zip(range(n), self.files):
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
        return self.pick_section(self._ssn, x)

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





