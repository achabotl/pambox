# -*- coding: utf-8 -*-
from __future__ import division, print_function
import csv
import os.path
import pytest
import numpy as np
import scipy.io as sio
from pambox import distort
from pambox import utils


__DATA_ROOT__ = os.path.join(os.path.dirname(__file__), 'data')


def test_overlap_and_add():
    powers = np.asarray([[0, 0, 0.0552, 0.0849, 0.1539],
                         [0, 0, 0.4514, 0.2585, 0.3045],
                         [0, 0, 1.5847, 0.4918, 0.4065],
                         [0, 0, 3.3438, 1.2580, 0.7740],
                         [0, 0, 5.3481, 1.8157, 2.4120],
                         [0, 0, 6.9678, 2.1765, 4.6699],
                         [0, 0, 7.5320, 3.2693, 6.5841],
                         [0, 0, 6.8844, 4.1498, 7.4230],
                         [0, 0, 5.5596, 3.7663, 7.0639]]).T

    phases = np.asarray([[0, 0, 3.1416, 3.1416, 0],
                         [0.2341, 3.1358, 2.2758, 1.8403, 2.4593],
                         [-2.8007, -1.0503, -0.4921, 0.0804, -1.0717],
                         [0.4176, 1.8466, 2.8773, -2.6004, 0.7771],
                         [-2.6635, -1.4400, -0.0616, 0.3823, -2.7871],
                         [0.5463, 1.5496, -3.0453, 2.9059, 0.2776],
                         [-2.5034, -1.7994, 0.1855, -0.8231, -2.8567],
                         [0.7550, 1.0628, -2.9340, 2.0991, 0.3459],
                         [-2.3103, -2.3792, 0.2128, -0.9727, -2.6737]]).T
    len_window = 4
    shift = 2
    target = np.asarray([0, 0, 0, 0, -0.5447, -0.2305, 1.0300, -0.8635, 0.5541,
                         0.1580, 0.0082, -0.0978])
    y = distort.overlap_and_add(powers, phases, len_window, shift)
    np.testing.assert_allclose(y, target, atol=1e-4)


def test_spec_sub_0dB_kappa_1():
    with open(os.path.join(__DATA_ROOT__, 'test_spec_sub_kappa_1.csv')) as \
            csv_file:
        data_file = csv.reader(csv_file)
        temp = next(data_file)
        n_samples_inputs = int(temp[0])
        n_samples_targets = int(temp[1])
        signal = np.empty(n_samples_targets)
        noise = np.empty(n_samples_targets)
        target = np.empty(n_samples_targets)
        target_noise = np.empty(n_samples_targets)

        for i, (m, n, o, p) in enumerate(data_file):
            signal[i] = np.asarray(m, dtype=np.float)
            noise[i] = np.asarray(n, dtype=np.float)
            target[i] = np.asarray(o, dtype=np.float)
            target_noise[i] = np.asarray(p, dtype=np.float)

        signal = signal[:n_samples_inputs]
        noise = noise[:n_samples_inputs]

    w = 512.
    padz = 512.
    shift = 0.5
    factor = 1

    y_speech, y_noise = distort.spec_sub(signal, noise, factor, w, padz, shift)
    np.testing.assert_allclose(y_speech, target)
    np.testing.assert_allclose(y_noise, target_noise)


def test_spec_sub_0dB_kappa_0():
    with open(os.path.join(__DATA_ROOT__, 'test_spec_sub_kappa_0.csv')) as \
            csv_file:
        data_file = csv.reader(csv_file)
        temp = next(data_file)
        n_samples_inputs = int(temp[0])
        n_samples_targets = int(temp[1])
        signal = np.empty(n_samples_targets)
        noise = np.empty(n_samples_targets)
        target = np.empty(n_samples_targets)
        target_noise = np.empty(n_samples_targets)

        for i, (m, n, o, p) in enumerate(data_file):
            signal[i] = np.asarray(m, dtype=np.float)
            noise[i] = np.asarray(n, dtype=np.float)
            target[i] = np.asarray(o, dtype=np.float)
            target_noise[i] = np.asarray(p, dtype=np.float)

        signal = signal[:n_samples_inputs]
        noise = noise[:n_samples_inputs]
    w = 512.
    padz = 512.
    shift = 0.5
    factor = 0
    y_speech, y_noise = distort.spec_sub(signal, noise, factor, w, padz, shift)
    np.testing.assert_allclose(y_speech, target)
    np.testing.assert_allclose(y_noise, target_noise)


def test_level_adjustment():
    snr = 0
    speech_level = 65
    clean = np.array([28.15291785, 29.23572238, 30.31852691, 34.64974504,
                      41.14657224, 36.81535411, 48.72620397, 45.47779037])
    noise = np.array(
        [-0.01672363, -0.02337646, -0.03231812, -0.03549194, -0.04663086,
         -0.05551147, -0.0401001, -0.04251099])

    target_clean = np.array([28.15291785, 29.23572238, 30.31852691, 34.64974504,
                             41.14657224, 36.81535411, 48.72620397,
                             45.47779037])
    target_mix = np.array(
        [-746.14503442, -1053.08586783, -1465.99813632, -1608.61397482,
         -2117.84479831, -2533.34518452, -1807.89333851, -1922.76499705])
    target_noise = np.array(
        [-774.29795226, -1082.32159021, -1496.31666323, -1643.26371986,
         -2158.99137055, -2570.16053863, -1856.61954247, -1968.24278742])

    clean, mix, noise = distort.mix_noise(clean, noise, speech_level, snr)

    np.testing.assert_allclose(clean, target_clean, rtol=1e-6)
    np.testing.assert_allclose(mix, target_mix, rtol=1e-6)
    np.testing.assert_allclose(noise, target_noise, rtol=1e-6)
