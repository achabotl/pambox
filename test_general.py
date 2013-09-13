import pytest
import general
from scipy.io import wavfile
import numpy as np
import scipy.io as sio


@pytest.fixture
def speech_raw():
    x = wavfile.read("test_files/test_speech_raw_22050.wav")[1]
    return x / 2. ** 15


@pytest.fixture
def noise_raw():
    x = wavfile.read("test_files/test_noise_raw_22050.wav")[1]
    return x / 2. ** 15


@pytest.fixture
def mix_0dB():
    x = wavfile.read("test_files/test_mix_0dB_22050.wav")[1]
    return x / 2. ** 15


@pytest.fixture
def noise_65dB():
    x_65 = wavfile.read("test_files/test_noise_65dB_22050.wav")[1]
    return x_65 / 2. ** 15


def test_set_level(noise_raw, noise_65dB):
    x65 = general.setdbspl(noise_raw, 65)
    np.testing.assert_allclose(x65, noise_65dB, atol=1e-4)
