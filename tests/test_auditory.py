import pytest
import numpy as np
from pambox import auditory as aud
import scipy.io as sio


def test_lowpass_filtering_of_envelope():
    mat = sio.loadmat("./test_files/test_envelope_filtering.mat")
    envelope = mat['envelope']
    target = mat['filtered_envelope']
    filtered_envelope = aud.lowpass_env_filtering(envelope, 150., 1, 22050.)
    np.testing.assert_allclose(filtered_envelope, target, atol=1e-7)


