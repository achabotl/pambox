# -*- coding: utf-8 -*-
from __future__ import division
import os.path
import pytest
from scipy.io import wavfile
import numpy as np
import scipy.io as sio
from pambox.speech import sepsm
from numpy.testing import assert_allclose, assert_array_equal


__DATA_ROOT__ = os.path.join(os.path.dirname(__file__), 'data')


@pytest.fixture
def speech_raw():
    x = wavfile.read(__DATA_ROOT__ + "/test_speech_raw_22050.wav")[1]
    return x / 2. ** 15


@pytest.fixture
def noise_raw():
    x = wavfile.read(__DATA_ROOT__ + "/test_noise_raw_22050.wav")[1]
    return x / 2. ** 15


@pytest.fixture
def mix_0dB():
    x = wavfile.read(__DATA_ROOT__ + "/test_mix_0dB_22050.wav")[1]
    return x / 2. ** 15


@pytest.fixture
def noise_65dB():
    x_65 = wavfile.read(__DATA_ROOT__ + "/test_noise_65dB_22050.wav")[1]
    return x_65 / 2. ** 15


@pytest.fixture
def center_f():
    return np.asarray([63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
                       1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000,
                       6300, 8000])


@pytest.fixture
def mat_snr_env():
    return sio.loadmat(__DATA_ROOT__ + '/test_snr_env.mat')


def test_select_bands_above_threshold(center_f):
    mat = sio.loadmat(__DATA_ROOT__ + "/test_bands_above_threshold_v1.mat",
                      squeeze_me=True)
    noise_rms = mat['mix_rms_out'].squeeze()
    target = mat['bands_to_process'].squeeze()
    c = sepsm.Sepsm(cf=center_f)
    bands_above_thres = c._bands_above_thres(noise_rms)
    # Make 1-based to compare with matlab
    bands_above_thres += 1
    assert_array_equal(bands_above_thres, target)


def test_snr_env():
    tests = [[0.001, [[0.], [0.]]],
             [0.001, [[0.0001], [0]]],
             [0.001, [[0.01], [1]]],
             [0.001, [[0, 0], [0, 0]]],
             [0.001, [[[0, 0],
                       [0, 0]],
                      [[0, 0],
                       [0, 0]]]],
             [0.001, [[[0, 0],
                       [0, 0]],
                      [[0, 0],
                       [0, 0]]]]
    ]

    c = sepsm.Sepsm()
    for target, (p_mix, p_noise) in tests:

        snrenv = c._snr_env(p_mix, p_noise)
        assert_allclose(snrenv, target)


@pytest.mark.slow
def test_sepsm_prediction_snr_min9_db():
    mat = sio.loadmat(__DATA_ROOT__ + "/test_multChanSNRenv.mat", squeeze_me=True,
                      struct_as_record=False)
    target_snr_env = mat['result'].SNRenv
    mix = mat['stim'][0]
    noise = mat['stim'][1]
    c = sepsm.Sepsm()
    res = c.predict(mix, mix, noise)
    assert_allclose(target_snr_env, res['snr_env'], rtol=1e-2)


@pytest.mark.slow
def test_sepsm_predictions_snr_0_kappa_0_8():
    mat = sio.loadmat(__DATA_ROOT__ + '/test_sepsm_spec_sub_0dB_kappa_0_8.mat',
                      squeeze_me=True, struct_as_record=False)
    c = sepsm.Sepsm()
    for ii in range(3):
        mix = mat['mixtures'][ii]
        noise = mat['noises'][ii]
        target = mat['results'][ii].SNRenv
        res = c.predict(mix, mix, noise)
        assert_allclose(target, res['snr_env'], rtol=2e-2)
