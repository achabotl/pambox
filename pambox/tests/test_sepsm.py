# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import os.path
import csv
import pytest
from scipy.io import wavfile
import numpy as np
import scipy.io as sio
from pambox.speech import sepsm
from numpy.testing import assert_allclose, assert_array_equal


__DATA_ROOT__ = os.path.join(os.path.dirname(__file__), 'data')


def test_select_bands_above_threshold():
    center_f = np.asarray([63, 80, 100, 125, 160, 200, 250, 315, 400, 500,
                           630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000,
                           5000, 6300, 8000])
    noise_rms = [142.598279903563, 596.254784935965, 1319.07476787393,
                 1931.80860942992, 2180.13918820141, 1714.49937340166,
                 2009.77926719000, 1130.48579025285, 820.432762207735,
                 1006.49592779826, 1523.47513285058, 921.921756875459,
                 791.901475253190, 1508.59965109920, 825.572455447266,
                 657.161350227808, 626.333420574852, 474.950833753788,
                 331.591691820142, 206.744689750152, 491.003492858161,
                 297.383958806200]
    target = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
              20, 21, 22]
    c = sepsm.Sepsm(cf=center_f)
    bands_above_thres = c._bands_above_thres(noise_rms)
    # Make 1-based to compare with matlab
    bands_above_thres += 1
    assert_array_equal(bands_above_thres, target)


@pytest.fixture
def model():
    return sepsm.Sepsm()


@pytest.mark.parametrize("target, p_mix, p_noise", [
    (0.001, (0., ), (0., )),
    (0.001, (0.0001,), (0,)),
    (0.001, (0.01,), (1,)),
    (0.001, (0, 0), (0, 0)),
    (0.001, ((0, 0), (0, 0)), ((0, 0), (0, 0))),
    (0.001, ((0, 0), (0, 0)), ((0, 0), (0, 0))),
])
def test_snr_env(model, target, p_mix, p_noise):
    snrenv, _ = model._snr_env(p_mix, p_noise)
    assert_allclose(snrenv, target)


@pytest.fixture
def mix_and_noise_snr_min9_db():
    with open(os.path.join(__DATA_ROOT__, 'test_full_sepsm.csv')) as csv_file:
        data_file = csv.reader(csv_file)
        temp = next(data_file)
        n_samples = int(temp[0])
        mix = np.empty(n_samples)
        noise = np.empty(n_samples)

        for i, (m, n) in enumerate(data_file):
            mix[i] = np.asarray(m, dtype=np.float)
            noise[i] = np.asarray(n, dtype=np.float)
    return mix, noise


def test_sepsm_prediction_snr_min9_db(model, mix_and_noise_snr_min9_db):
    mix, noise = mix_and_noise_snr_min9_db

    target_snr_env = 9.57297

    res = model.predict(mix, mix, noise)
    assert_allclose(target_snr_env, res['p']['snr_env'], rtol=1e-3)


@pytest.mark.slow
def test_sepsm_predictions_snr_0_kappa_0_8(model):
    mat = sio.loadmat(__DATA_ROOT__ + '/test_sepsm_spec_sub_0dB_kappa_0_8.mat',
                      squeeze_me=True, struct_as_record=False)
    for ii in range(3):
        mix = mat['mixtures'][ii]
        noise = mat['noises'][ii]
        target = mat['results'][ii].SNRenv
        res = model.predict(mix, mix, noise)
        assert_allclose(target, res['p']['snr_env'], rtol=8e-2)
