# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os.path
import pytest
import numpy as np
from scipy import io as sio, io
from pambox import inner
import scipy.io as sio
from numpy.testing import assert_allclose


__DATA_ROOT__ = os.path.join(os.path.dirname(__file__), 'data')


def test_lowpass_filtering_of_envelope():
    mat = sio.loadmat(__DATA_ROOT__ + "/test_hilbert_env_and_lp_filtering_v1.mat",
                      squeeze_me=True)
    envelope = mat['unfiltered_env']
    target = mat['lp_filtered_env']
    filtered_envelope = inner.lowpass_env_filtering(envelope, 150., 1, 22050.)
    assert_allclose(filtered_envelope, target, atol=1e-7)


def test_erb():
    bw = inner.erbbw(1000)
    assert_allclose(bw, 132.63, rtol=1e-4)


def test_slaney_gammatone_filtering():
    from itertools import product
    mat = sio.loadmat(__DATA_ROOT__ + '/test_slaney_gammatone_coef.mat',
                        squeeze_me=True)
    cf = [63, 1000]
    fs = [22050, 44100]
    for c, f in product(cf, fs):
        g = inner.GammatoneFilterbank(c, f)
        y = g.filter(mat['x'])
        target_file = 'y_%d_%d' % (f, c)
        np.testing.assert_allclose(y[0], mat[target_file])


def test_third_octave_filtering_of_noise_():
    mat = sio.loadmat(__DATA_ROOT__ + '/test_third_oct_filt_rms.mat')
    noise = mat['x'].squeeze()
    target_noise_rms = mat['rms_out'].squeeze()
    center_f = mat['midfreq'].squeeze()
    rms_out = inner.noctave_filtering(noise, center_f, 22050, width=3)
    assert_allclose(target_noise_rms, rms_out, rtol=1e-4)


def test_mod_filtering_for_simple_signal():
    signal = np.asarray([1, 0, 1, 0, 1])
    fs = 2205
    modf = np.asarray([1., 2., 4., 8., 16., 32., 64.])
    p, _ = inner.mod_filterbank(signal, fs, modf)
    target = np.asarray([6.69785298e-18, 6.06375859e-06, 2.42555385e-05,
                         9.70302212e-05, 3.88249957e-04, 1.55506496e-03,
                         6.25329663e-03])
    assert_allclose(p, target, rtol=1e-2)


def test_mod_filt_complex():
    """Test modulation filtering with actual speech and noise signals
    """
    mat = sio.loadmat(__DATA_ROOT__ + '/test_mod_filtering.mat')
    x = mat['data'].squeeze()
    fs = mat['fs'].squeeze()
    modf = np.hstack((mat['fcut'].squeeze(), mat['fcs'].squeeze()))
    modf = modf.astype('float')
    target = mat['powers'].squeeze()
    powers, _ = inner.mod_filterbank(x, fs, modf)
    assert_allclose(powers, target)


@pytest.mark.slow
def test_mod_filt_sepsm_v1():
    """Test modulation filtering with actual speech and noise signals
    """
    mat = sio.loadmat(__DATA_ROOT__ + '/test_modFbank_v1.mat')
    x = mat['Env'][:, 0].squeeze()
    fs = mat['fs'].squeeze()
    modf = mat['fcs_EPSM'].squeeze()
    modf = modf.astype('float')
    target = mat['ExcPtn'][:, 0].squeeze()
    powers, _ = inner.mod_filterbank(x, fs, modf)
    assert_allclose(powers, target)


def test_mod_filterbank_for_temporal_outpout():
    """Test modulation filterbank for its temporal output.

    Mostly used in the mr-sEPSM model, where the time output is needed to
    process the envelope power for different window lenghts.
    """
    mat = sio.loadmat(__DATA_ROOT__ + '/test_mr_sepsm_snrenv_mr_v1.mat',
                      squeeze_me=True)
    x = mat['Env'].T[0]
    fs = mat['fs']
    modf = mat['fcs']
    p, t = inner.mod_filterbank(x, fs, modf)
    assert_allclose(p, mat['ExcPtns'].T[0])
    assert_allclose(t, mat['tempOutput'].T[0])
