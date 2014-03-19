from __future__ import division, print_function
import pytest
import numpy as np
from pambox import auditory as aud
import scipy.io as sio
from numpy.testing import assert_allclose
import tests


def test_lowpass_filtering_of_envelope():
    mat = sio.loadmat(tests.__DATA_ROOT__ + "/test_hilbert_env_and_lp_filtering_v1.mat",
                      squeeze_me=True)
    envelope = mat['unfiltered_env']
    target = mat['lp_filtered_env']
    filtered_envelope = aud.lowpass_env_filtering(envelope, 150., 1, 22050.)
    assert_allclose(filtered_envelope, target, atol=1e-7)


def test_erb():
    bw = aud.erbbw(1000)
    assert_allclose(bw, 132.63, rtol=1e-4)


def test_slaney_gammatone_filtering():
    from itertools import product
    mat = sio.loadmat(tests.__DATA_ROOT__ + '/test_slaney_gammatone_coef.mat',
                        squeeze_me=True)
    cf = [63, 1000]
    fs = [22050, 44100]
    for c, f in product(cf, fs):
        g = aud.GammatoneFilterbank(c, f)
        y = g.filter(mat['x'])
        target_file = 'y_%d_%d' % (f, c)
        np.testing.assert_allclose(y[0], mat[target_file])