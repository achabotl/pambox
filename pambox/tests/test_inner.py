# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import csv
import os.path
import pytest

import numpy as np
import scipy.io as sio
from numpy.testing import assert_allclose

from pambox import inner


__DATA_ROOT__ = os.path.join(os.path.dirname(__file__), 'data')


def test_lowpass_filtering_of_envelope():
    mat = sio.loadmat(__DATA_ROOT__ + "/test_hilbert_env_and_lp_filtering_v1.mat",
                      squeeze_me=True)
    envelope = mat['unfiltered_env']
    target = mat['lp_filtered_env']
    filtered_envelope = inner.lowpass_env_filtering(envelope, 150., 1, 22050.)
    assert_allclose(filtered_envelope, target, atol=1e-7)


def test_erb():
    bw = inner.erb_bandwidth(1000)
    assert_allclose(bw, 132.63, rtol=1e-4)


def test_GammatoneFilterbank_filtering():
    from itertools import product
    mat = sio.loadmat(__DATA_ROOT__ + '/test_GammatoneFilterbank_filtering.mat',
                        squeeze_me=True)
    cf = [63, 1000]
    fs = [22050, 44100]
    for c, f in product(cf, fs):
        g = inner.GammatoneFilterbank(f, c)
        y = g.filter(mat['x'])
        target_file = 'y_%d_%d' % (f, c)
        np.testing.assert_allclose(y[0], mat[target_file])


def test_third_octave_filtering_of_noise_():
    with open(os.path.join(__DATA_ROOT__,
                           'test_third_octave_filtering_of_noise.csv')) as \
            csv_file:
        pass
        data_file = csv.reader(csv_file)
        temp = next(data_file)
        n_samples = int(temp[0])
        x = np.empty(n_samples)

        for i, s in enumerate(data_file):
            x[i] = np.asarray(s, dtype=np.float)

    target = np.array([ 151.66437785,  688.6881118 ])
    center_f = [63, 125]
    fs = 22050
    filterbank = inner.RectangularFilterbank(fs, center_f, width=3)
    rms_out = filterbank.filter(x)
    assert_allclose(rms_out, target, rtol=1e-4)


@pytest.mark.parametrize("x, target", [
    ([0, 1, 2, 1, 0],
     [0.70710678, 1.56751612, 2., 1.56751612, 0.70710678]),
    ([0, 1, 2, 1, 0],
     [0.70710678, 1.56751612, 2., 1.56751612, 0.70710678]),
    ([[0, 1], [0, 1]],
     [[0., 1.], [0., 1.]]),
    ([[0, 1, 0], [2, 3, 0]],
     [[0.5, 1., 0.5], [2.5, 3.16227766, 1.5]]),
    ])
def test_hilbert_env_on_2d_array_with_last_dimension(x, target):
    env = inner.hilbert_envelope(x)
    np.testing.assert_allclose(env, target, err_msg="Input was {}".format(x))


def test_envelope_extraction():
    x = np.array(
        [-0.00032745, -0.00031198, -0.00029605, -0.00027965, -0.00026281,
         -0.00024553, -0.00022783, -0.00020972])
    target = np.array(
        [0.00068165, 0.00068556, 0.00068946, 0.00069335, 0.00069725,
         0.00070113, 0.00070502, 0.0007089])
    envelope = inner.hilbert_envelope(x)
    np.testing.assert_allclose(envelope, target, atol=1e-3)
