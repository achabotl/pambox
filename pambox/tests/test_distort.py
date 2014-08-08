# -*- coding: utf-8 -*-
from __future__ import division, print_function
import os.path
import pytest
import numpy as np
import scipy.io as sio
from pambox import distort
from pambox import utils


__DATA_ROOT__ = os.path.join(os.path.dirname(__file__), 'data')


@pytest.fixture
def mat_overlap_and_add():
    return sio.loadmat(__DATA_ROOT__ + '/test_overlap_and_add.mat')


def test_overlap_and_add(mat_overlap_and_add):
    powers = mat_overlap_and_add['XNEW'].squeeze().T
    phases = mat_overlap_and_add['yphase'].squeeze().T
    len_window = mat_overlap_and_add['windowLen'].squeeze()
    shift = mat_overlap_and_add['ShiftLen'][0].squeeze()
    target = mat_overlap_and_add['ReconstructedSignal'].squeeze()
    y = distort.overlap_and_add(powers, phases, len_window,  shift)
    np.testing.assert_allclose(y, target, atol=1e-15)


def test_spec_sub_0dB_kappa_1():
    mat = sio.loadmat(__DATA_ROOT__ + '/test_spec_sub_complete.mat',
                      squeeze_me=True)
    signal = mat['input']
    noise = mat['noise']
    w = mat['W']
    padz = mat['padz']
    shift = mat['SP']
    factor = mat['factor']
    target = mat['output_Y']
    target_noise = mat['output_N']
    y_speech, y_noise = distort.spec_sub(signal, noise, factor, w, padz, shift)
    np.testing.assert_allclose(y_speech, target, atol=1e-10)
    np.testing.assert_allclose(y_noise, target_noise, atol=1e-16)


def test_spec_sub_0dB_kappa_0():
    mat = sio.loadmat(__DATA_ROOT__ + '/test_spec_sub_0dB_kappa_0.mat',
                      squeeze_me=True)
    signal = mat['test']
    noise = mat['noise']
    w = mat['W']
    padz = mat['padz']
    shift = mat['SP']
    factor = mat['factor']
    target = mat['processed_test']
    target_noise = mat['processed_noise']
    y_speech, y_noise = distort.spec_sub(signal, noise, factor, w, padz, shift)
    np.testing.assert_allclose(y_speech, target, atol=1e-10)
    np.testing.assert_allclose(y_noise, target_noise, atol=1e-10)


def test_level_adjustment_and_spec_sub_processing():
    mat = sio.loadmat(__DATA_ROOT__ + '/setting_level_and_spec_sub_processing_v1.mat',
                      squeeze_me=True)
    x = mat['x']
    x_adj = mat['x_adjusted']
    speech_level = mat['SPL']
    file_level = mat['sentenceFileLevel']
    noise = mat['noise_glob']
    noise_adj = mat['noise_adjusted']
    snr = mat['snr']
    target_mix = mat['test']
    W = mat['W']
    PADZ = mat['padz']
    SP = mat['SP']
    factor = mat['factor']
    target_processed_speech = mat['processed_test']
    target_processed_noise = mat['processed_noise']

    print(utils.dbspl(x))
    x = x * 10 ** ((speech_level - file_level) / 20)
    print(utils.dbspl(x))
    np.testing.assert_allclose(x, x_adj)

    noise = noise / utils.rms(noise) * 10 ** ((speech_level - snr) / 20)
    np.testing.assert_allclose(noise, noise_adj)

    mixture = noise + x
    np.testing.assert_allclose(target_mix, mixture)

    processed_test, processed_noise = distort.spec_sub(mixture, noise, factor,
                                                       W, PADZ, SP)
    np.testing.assert_allclose(processed_test, target_processed_speech)
    np.testing.assert_allclose(processed_noise, target_processed_noise)


def test_level_adjustment():
    mat = sio.loadmat(__DATA_ROOT__ + '/setting_level_and_spec_sub_processing_v1.mat',
                      squeeze_me=True)
    target_x = mat['x_adjusted']
    target_noise = mat['noise_adjusted']
    target_mix = mat['test']
    speech_level = mat['SPL']
    noise = mat['noise_glob']
    snr = mat['snr']

    clean, mix, noise = distort.mix_noise(target_x, noise, speech_level, snr)

    np.testing.assert_allclose(target_x, target_x)
    np.testing.assert_allclose(mix, target_mix)
    np.testing.assert_allclose(noise, target_noise)
