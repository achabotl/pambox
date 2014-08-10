# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os.path
import pytest
from pambox.speech.mrsepsm import MrSepsm
import scipy.io as sio
import numpy as np
from numpy.testing import assert_allclose
from six.moves import zip


__DATA_ROOT__ = os.path.join(os.path.dirname(__file__), 'data')


@pytest.fixture
def mr():
    return MrSepsm()


@pytest.fixture
def mat():
    return sio.loadmat(__DATA_ROOT__ + '/test_mr_sepsm_snrenv_mr_v1.mat',
                       squeeze_me=True)


def test_mr_sepsm_mr_env_powers(mr):
    mat = sio.loadmat(__DATA_ROOT__ + '/test_mr_sepsm_mr_env_powers.mat',
                      squeeze_me=True)
    channel_env = mat['env'].T[0]
    mod_channel_envs = mat['mod_channel_envs'].T[0]

    mr_env_powers = mr._mr_env_powers(channel_env, mod_channel_envs)
    for d, target in zip(mr_env_powers,
                         mat['mr_env_powers']):
        assert_allclose(d.compressed(), target[0])


def test_mr_snr_env(mr, mat):
    """Test calculation of SNRenv for a given channel
    """
    mat = mat
    mat_mix = sio.loadmat(__DATA_ROOT__ + '/test_mr_sepsm_mr_snr_env_mix.mat')
    mat_noise = sio.loadmat(__DATA_ROOT__ +
                            '/test_mr_sepsm_mr_snr_env_noise.mat')
    od_mix = np.ma.MaskedArray(mat_mix['data'], mat_mix['mask'])
    od_noise = np.ma.MaskedArray(mat_noise['data'], mat_noise['mask'])
    time_av_snr_env, exc_ptns, mr_snr_env = mr._mr_snr_env(od_mix,
                                                           od_noise)
    assert_allclose(time_av_snr_env, mat['timeAvg_SNRenvs'])


def test_mr_sepsm_time_averaging_of_short_term_snr_env(mat):
    """Test the averaging of the multi-resolution representation of the SNRenv

    Given the OrderedDict of multi-resolution SNRenv values, for a given
    channel, average it over time.
    """
    in_mat = sio.loadmat(__DATA_ROOT__ +
                            '/test_mr_sepsm_time_average_snr.mat',
                      squeeze_me=True)
    mr_snr_env = np.ma.MaskedArray(in_mat['data'], in_mat['mask'])

    mr = MrSepsm()
    t_av = mr._time_average(mr_snr_env)
    assert_allclose(t_av, mat['timeAvg_SNRenvs'])


@pytest.mark.slow
def test_complete_mr_sepsm(mr):
    mat_complete = sio.loadmat(__DATA_ROOT__ +
                               '/test_mr_sepsm_full.mat',
                               squeeze_me=True)
    """Test the prediction by the mr-sEPSM
    """
    mix = mat_complete['mix']
    noise = mat_complete['noise']
    tests = (
        (mix, noise, 17.026659110065896),
    )

    for mix, noise, target in tests:
        res = mr.predict(mix, mix, noise)

        assert_allclose(
            res['snr_env']
            , target
            , rtol=0.01
        )
