import pytest
import numpy as np
from pambox import auditory as aud
import scipy.io as sio
from numpy.testing import assert_allclose


def test_lowpass_filtering_of_envelope():
    mat = sio.loadmat("./test_files/test_hilbert_env_and_lp_filtering_v1.mat",
                      squeeze_me=True)
    envelope = mat['unfiltered_env']
    target = mat['lp_filtered_env']
    filtered_envelope = aud.lowpass_env_filtering(envelope, 150., 1, 22050.)
    assert_allclose(filtered_envelope, target, atol=1e-7)


