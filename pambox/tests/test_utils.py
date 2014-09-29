# -*- coding: utf-8 -*-
from __future__ import division, print_function
import os.path

import pytest
import numpy as np
from numpy.testing import assert_allclose, dec, TestCase
from scipy import signal

from pambox import utils
from pambox.utils import fftfilt


__DATA_ROOT__ = os.path.join(os.path.dirname(__file__), 'data')


def test_dbspl():
    tests = (
        (([0], True, 0, -1), -np.inf),
        (([1], False, 0, -1), 0),
        (([1], False, 100, -1), 100),
        (([1], True, 0, -1), -np.inf),
        (([10], False, 0, -1), 20),
        (([10, 10], False, 0, -1), 20),
        (([10, 10], False, 0, 1), [20, 20]),
    )
    for (x, ac, offset, axis), target in tests:
        assert_allclose(utils.dbspl(x, ac=ac, offset=offset,
                                               axis=axis), target)


def test_rms_do_ac():
    assert utils.rms([0, 1, 2, 3, 4, 5, 6], ac=True) == 2


def test_rms():
    tests = (
        (([0], True, -1), 0),
        (([1], True, -1), 0),
        (([1], False, -1), 1),
        (([-1], False, -1), 1),
        (([-1], True, -1), 0),
        (([10, 10], False, -1), 10),
        (([10, 10], True, -1), 0),
        (([[0, 1], [0, 1]], True, -1), [0.5, 0.5]),
        (([[0, 1], [0, 1]], False, -1), [0.70710678, 0.70710678]),
        (([[0, 1], [0, 1]], True, 0), [0, 0]),
        (([[0, 1], [0, 1]], False, 0), [0, 1]),
        (([[0, 1], [0, 1]], True, 1), [0.5, 0.5]),
        (([[0, 1], [0, 1]], False, 1), [0.70710678, 0.70710678]),
    )
    for (x, ac, axis), target in tests:
        assert_allclose(utils.rms(x, ac=ac, axis=axis), target)


def test_set_level():
    tests = (
        ((0, 1), 65, 100, (0., 0.02514867)),
        ((0, 1), 65, 0, (0., 2514.86685937)),
        ((0, 1), 100, 100, (0., 1.41421356)),
    )

    for x, level, offset, target in tests:
        y = utils.setdbspl(x, level, offset=offset)
        assert_allclose(y, target, atol=1e-4)


# Can't be done programmatically, because the exact third-octave spacing is not
# exactly the same as the one commonly used.
@pytest.mark.xfail(run=False, reason="Real 3rd-oct != common ones")
def test_third_oct_center_freq_bet_63_12500_hz():
    """Test returns correct center frequencies for third-octave filters

    Between 63 and 12500 Hz.

    """
    center_f = (63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000,
                1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000)
    assert utils.noctave_center_freq(63, 12500, width=3) == center_f


def test_find_calculate_srt_when_found():
    x = np.arange(10)
    y = 20 * x + 4
    assert 2.3 == utils.int2srt(x, y, srt=50)


def test_find_calculate_srt_when_not_found():
    x = np.arange(10)
    y = 2 * x + 4
    assert None == utils.int2srt(x, y, srt=50)


def test_find_srt_when_srt_at_index_zero():
    x = [0, 1]
    y = [50, 51]
    assert 0 == utils.int2srt(x, y, srt=50)


def test_make_same_length_with_padding():
    tests = (
        (([1], [1, 1]), ([1, 0], [1, 1])),
        (([1, 1], [1, 1]), ([1, 1], [1, 1])),
        (([1, 1], [1]), ([1, 1], [1, 0])),
        (([1], [1, 1], False), ([1], [1])),
    )

    for inputs, targets in tests:
        assert_allclose(utils.make_same_length(*inputs), targets)


def test_psy_fn():
    x = -3.0
    mu = 0.
    sigma = 1.0
    target = 0.13498980316300957
    y = utils.psy_fn(x, mu, sigma)
    assert_allclose(y, target)


class _TestFFTFilt(TestCase):
    dt = None

    def test_fftfilt(self):
        dt = 1e-6
        fs = 1/dt
        u = np.random.rand(10**6)
        f = 10**4
        b = signal.firwin(50, f/fs)

        u_lfilter = signal.lfilter(b, 1, u)
        u_fftfilt = fftfilt(b, u)
        assert_allclose(u_lfilter, u_fftfilt)

    def test_rank1(self):
        dec.knownfailureif(
            self.dt in [np.longdouble, np.longcomplex],
            "Type %s is not supported by fftpack" % self.dt)(lambda:  None)()

        x = np.arange(6).astype(self.dt)

        # Test simple FIR
        b = np.array([1, 1]).astype(self.dt)
        y_r = np.array([0, 1, 3, 5, 7, 9.]).astype(self.dt)
        assert_allclose(fftfilt(b, x), y_r, atol=1e-6)

        # Test simple FIR with FFT length
        b = np.array([1, 1]).astype(self.dt)
        y_r = np.array([0, 1, 3, 5, 7, 9.]).astype(self.dt)
        n = 12
        assert_allclose(fftfilt(b, x, n), y_r, atol=1e-6)

        # Test simple FIR with FFT length which is a power of 2
        b = np.array([1, 1]).astype(self.dt)
        y_r = np.array([0, 1, 3, 5, 7, 9.]).astype(self.dt)
        n = 32
        assert_allclose(fftfilt(b, x, n), y_r, atol=1e-6)

        # Test simple FIR with FFT length
        b = np.array(np.ones(6)).astype(self.dt)
        y_r = np.array([0, 1, 3, 6, 10, 15]).astype(self.dt)
        assert_allclose(fftfilt(b, x), y_r, atol=1e-6)

    def test_rank2_x_longer_than_b(self):
        dec.knownfailureif(
            self.dt in [np.longdouble, np.longcomplex],
            "Type %s is not supported by fftpack" % self.dt)(lambda:  None)()

        shape = (4, 3)
        x = np.linspace(0, np.prod(shape) - 1, np.prod(shape)).reshape(shape)
        x = x.astype(self.dt)

        b = np.array([1, 1]).astype(self.dt)

        y_r2 = np.array([[0, 1, 3], [3, 7, 9], [6, 13, 15], [9, 19, 21]],
                           dtype=self.dt)

        y = fftfilt(b, x)
        assert_allclose(y, y_r2)

    def test_rank2_b_longer_than_x(self):
        dec.knownfailureif(
            self.dt in [np.longdouble, np.longcomplex],
            "Type %s is not supported by fftpack" % self.dt)(lambda:  None)()

        shape = (4, 3)
        x = np.linspace(0, np.prod(shape) - 1, np.prod(shape)).reshape(shape)
        x = x.astype(self.dt)

        b = np.array([1, 1, 1, 1]).astype(self.dt)

        y_r2 = np.array([[0, 1, 3], [3, 7, 12], [6, 13, 21], [9, 19, 30]],
                        dtype=self.dt)

        y = utils.fftfilt(b, x)
        assert_allclose(y, y_r2, atol=1e-6)

    def test_b_rank2(self):
        dec.knownfailureif(
            self.dt in [np.longdouble, np.longcomplex],
            "Type %s is not supported by fftpack" % self.dt)(lambda:  None)()

        x = np.linspace(0, 5, 6).astype(self.dt)

        b = np.array([[1, 1], [2, 2]]).astype(self.dt)

        y_r2 = np.array([[0, 1, 3, 5, 7, 9], [0, 2, 6, 10, 14, 18]],
                        dtype=self.dt)
        y = utils.fftfilt(b, x)
        assert_allclose(y, y_r2)

        b = np.array([[1, 1], [2, 2], [3, 3]]).astype(self.dt)

        y_r2 = np.array([[0, 1, 3, 5, 7, 9],
                         [0, 2, 6, 10, 14, 18],
                         [0, 3, 9, 15, 21, 27]],
                        dtype=self.dt)
        y = utils.fftfilt(b, x)
        assert_allclose(y, y_r2, atol=1e-6)

    def test_b_and_x_of_same_dim(self):
        dec.knownfailureif(
            self.dt in [np.longdouble, np.longcomplex],
            "Type %s is not supported by fftpack" % self.dt)(lambda:  None)()

        shape = (2, 5)
        x = np.linspace(0, np.prod(shape) - 1, np.prod(shape)).reshape(shape)
        x = x.astype(self.dt)

        b = np.array([[1, 1], [2, 2]]).astype(self.dt)

        y_r2 = np.array([[0, 1, 3, 5, 7], [10, 22, 26, 30, 34]],
                        dtype=self.dt)
        y = utils.fftfilt(b, x)
        assert_allclose(y, y_r2, atol=1e-6)


class TestFFTFiltFloat32(_TestFFTFilt):
    dt = np.float32


class TestFFTFiltFloat64(_TestFFTFilt):
    dt = np.float64

