from __future__ import division, print_function
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


def test_erb():
    bw = aud.erbbw(1000)
    assert_allclose(bw, 132.63, rtol=1e-4)


# We use a different implementation than the Matlab one and the delay
# are different.
@pytest.mark.xfail
def test_gammatone_filtering():
    mat = sio.loadmat('./test_files/test_gammatone_filtering.mat')
    center_f = mat['midfreq'].squeeze()
    fs = mat['fs'].squeeze()
    signal = mat['signal'].squeeze()
    targets = mat['GT_output'].squeeze()
    target = targets[:,:,0].T
    out = aud.gammatone_filtering(signal, center_f, fs)
    assert_allclose(out, target)
