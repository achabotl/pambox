"""
Simple wrapper around pyaudio to simplify sound playback
"""
from __future__ import division, print_function
import numpy as np
import pyaudio


def play(x, fs=44100, normalize=True):
    """
    Plays sound.

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
        x /= np.abs(x).max()
    if x.shape[0] == 2:
        x = x.T
    channels = x.ndim
    _play_sound(x, fs=fs, channels=channels)


def _play_sound(x, fs=44100, channels=1, output=1, format_=pyaudio.paFloat32):
    """
    Wrapper around PyAudio to play numpy arrays.

    :param x: ndarray, signal
    :param fs: int, sampling frequency
    :param channels: int, number of channels
    :param output: int, ID of the soundcard where to play back.
    :param format_: pyaudio Format object, Format of the signal data.
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=format_, channels=channels, rate=fs,
                    output=output)
    stream.write(x.astype(np.float32).tostring())
    stream.close()
    p.terminate()
