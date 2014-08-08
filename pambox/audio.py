"""
:mod:`~pambox.audio` provides a imple wrapper around pyaudio to simplify
sound playback.
"""
from __future__ import division, print_function
import numpy as np
import pyaudio


def play(x, fs=44100, normalize=True):
    """Plays sound.

    Parameters
    ----------
    x : array_like,
       Signal to be player. The shape should be nChannels  x Length.
    fs : int (optional
        Sampling frequency. The default is 44100 Hz.
    normalize : bool
        Normalize the signal such that the maximumal (absolute value) is 1 to
        prevent clipping.

    """
    x = np.asarray(x)
    if normalize:
        x = x / np.abs(x).max()
    if x.shape[0] == 2:
        x = x.T
    channels = x.ndim
    _play_sound(x, fs=fs, channels=channels)


def _play_sound(x, fs=44100, channels=1, output=1, format_=pyaudio.paFloat32):
    """Wrapper around PyAudio to play numpy arrays.

    Parameters
    ----------
    x : ndarray
        Signal
    fs : int
        Sampling frequency, default is 44100 Hz.
    channels : int
        Number of channels, default is 1.
    output : int
        ID of the soundcard where to play back. Default is 1.
    format_: pyaudio Format object
        Format of the signal data, the default is `pyaudio.paFloat32`.

    Returns
    -------
    None

    """
    p = pyaudio.PyAudio()
    stream = p.open(format=format_, channels=channels, rate=fs,
                    output=output)
    stream.write(x.astype(np.float32).tostring())
    stream.close()
    p.terminate()
