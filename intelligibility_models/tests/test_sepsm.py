from __future__ import division
import pytest
from scipy.io import wavfile
import numpy as np
import scipy.io as sio
from pambox.intelligibility_models import sepsm
from numpy.testing import assert_allclose


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


@pytest.fixture
def center_f():
    return np.asarray([63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
                       1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000,
                       6300, 8000])


@pytest.fixture
def mat_snr_env():
    return sio.loadmat('./test_files/test_snr_env.mat')


def test_select_bands_above_threshold(center_f):
    mat = sio.loadmat("./test_files/test_bands_above_threshold_v1.mat",
                      squeeze_me=True)
    noise_rms = mat['mix_rms_out'].squeeze()
    target = mat['bands_to_process'].squeeze()
    c = sepsm.Sepsm(cf=center_f)
    bands_above_thres = c._bands_above_thres(noise_rms)
    # Make 1-based to compare with matlab
    bands_above_thres += 1
    assert np.array_equal(bands_above_thres, target)


def test_snr_env(mat_snr_env):
    mat_snr_env = sio.loadmat('./test_files/test_snr_env_lin.mat',
                              squeeze_me=True)
    env = mat_snr_env['env'].T
    fs = mat_snr_env['fs']
    target_snr_env = mat_snr_env['SNRenv_p_n'][:, 1]
    target_excitation_patterns = mat_snr_env['sEPSM_ExPtns']\
        .T[1, :, :]
    modf = mat_snr_env['fcs_sEPSM'].astype('float')

    signals = (env[0], env[0], env[1])
    c = sepsm.Sepsm(modf=modf)
    snrenv, excitation_patterns = c._snr_env(signals, fs)
    assert_allclose(snrenv, target_snr_env)
    assert_allclose(excitation_patterns[1:, :], target_excitation_patterns)


def test_sepsm_prediction_snr_min9_db():
    mat = sio.loadmat("./test_files/test_multChanSNRenv.mat", squeeze_me=True,
                      struct_as_record=False)
    target_snr_env = mat['result'].SNRenv
    mix = mat['stim'][0]
    noise = mat['stim'][1]
    c = sepsm.Sepsm()
    res = c.predict(mix, mix, noise)
    assert_allclose(target_snr_env, res.snr_env, rtol=1e-2)


def test_sepsm_predictions_snr_0_kappa_0_8():
    mat = sio.loadmat('./test_files/test_sepsm_spec_sub_0dB_kappa_0_8.mat',
                      squeeze_me=True, struct_as_record=False)
    c = sepsm.Sepsm()
    for ii in range(3):
        mix = mat['mixtures'][ii]
        noise = mat['noises'][ii]
        target = mat['results'][ii].SNRenv
        res = c.predict(mix, mix, noise)
        assert_allclose(target, res.snr_env, rtol=1e-2)
