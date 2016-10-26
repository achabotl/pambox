from __future__ import absolute_import, division, print_function

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from pambox.central import EC


class TestECMethodsInputs:

    @classmethod
    def setup_class(cls):
        cls.ec = EC(22050)

    def test_equalize_different_shapes(self):
        with pytest.raises(ValueError):
            self.ec.equalize((1, 2, 3), ((1, 2, 3), (4, 5, 6)), 0)

    def test_equalize_more_than_2d(self):
        with pytest.raises(ValueError):
            self.ec.equalize(np.ones((1, 2, 3)), (0,), 0)

    def test_equalize_with_2d_inputs_but_only_one_integer_cf(self):
        with pytest.raises(ValueError):
            self.ec.equalize(np.ones((2, 2)), np.ones((2, 2)), 1)

    def test_equalize_with_2d_inputs_but_only_one_iterable_cf(self):
        with pytest.raises(ValueError):
            self.ec.equalize(np.ones((2, 2)), np.ones((2, 2)), (1,))

    def test_cancel_different_shapes(self):
        with pytest.raises(ValueError):
            self.ec.cancel((1, 2, 3), ((1, 2, 3), (4, 5, 6)), 1, 0)

    def test_cancel_more_than_2d(self):
        with pytest.raises(ValueError):
            self.ec.cancel(np.ones((1, 2, 3)), (0,), 1, 0)


class TestECReturnValues:

    @classmethod
    def setup_class(cls):
        cls.ec = EC(22050)

    @pytest.mark.parametrize("x, delay, target", [
        ((1, 0, 0), 1, (0, 1, 0)),
        ((1, 0, 0), -1, (0, 0, 1)),
    ])
    def test_shift(self, x, delay, target):
        x = np.asarray(x)
        delay = delay / self.ec.fs
        out = self.ec._shift(x, delay)
        assert_allclose(out, target, atol=1e-15)

    @pytest.mark.parametrize("left, right, cf, target", [
        ((1, 0, 0), (1, 0, 0), 100, 0),
        ((2, 1, 0, 0), (0, 2, 1, 0), 1000, -1),
        ((0, 2, 1, 0), (2, 1, 0, 0), 1000, 1),
        ((1, 0, 0, 0), (0, 1, 0, 0), 1000, -1),
        ((0, 0, 0, 0), (0, 0, 0, 0), 1000, 0),
    ])
    def test_find_tau(self, left, right, cf, target):
        left = np.asarray(left)
        right = np.asarray(right)
        tau = self.ec._find_tau(left, right, cf)
        assert tau == target / self.ec.fs

    def test_ec_cancel_2d_input_with_single_window(self):
        ec = EC(22050, win_len=None)
        out = ec.cancel([(2, 2), (2, 2)], [(1, 1,), (1, 1)], (1, 2), (0, 0))
        assert_allclose(out, ((0.5, 0.5), (0, 0)), atol=1e-15)

    def test_ec_cancel_1d_input_with_single_window(self):
        ec = EC(22050, win_len=None)
        out = ec.cancel((2, 2), (1, 1,), 1, 0)
        assert_allclose(out, (0.5, 0.5))

    def test_single_channel_cancel_with_tau_equals_zero(self):
        ec = EC(4, win_len=None)
        out = ec._single_chan_cancel(np.array((1., 1.)), np.array((1., 1.)), 1, 0)
        assert_array_equal(out, (0, 0))

    def test_single_channel_cancel_with_tau_equals_one(self):
        ec = EC(4, win_len=None)
        out = ec._single_chan_cancel(np.array((0., 1.)), np.array((1., 0.)), 1, -0.25)
        assert_allclose(out, (0., 0.), atol=1e-16)

    def test_single_channel_cancel_with_overlap(self):
        ec = EC(4, win_len=0.5, overlap=0.5)
        out = ec._single_chan_cancel(np.ones(4), np.ones(4),
                                     (1, 1, 1), (0, 0, 0))
        assert_array_equal(out, (0, 0, 0, 0))

    def test_single_channel_cancel_with_without_overlap(self):
        ec = EC(4, win_len=0.5, overlap=0)
        out = ec._single_chan_cancel(np.ones(4), np.ones(4), (1, 1), (1, 1))
        assert_array_equal(out, (0, 0, 0, 0))

    @pytest.mark.parametrize("fs, left, right, win_len, overlap, "
                             "target_shape", [
        (4, (0, 0, 0, 0), (0, 0, 0, 0), 0.5, 0, 2),
        (4, (0, 0, 0, 0), (0, 0, 0, 0), 0.5, 0.5, 3),
        (4, (0, 0, 0, 0), (0, 0, 0, 0), 0.25, None, 4),
        (4, (0, 0, 0, 0), (0, 0, 0, 0), None, None, 1),
        (4, (0, 0, 0, 0), (0, 0, 0, 0), None, 0.5, 1),
    ])
    def test_number_of_alpha_and_tau(self, fs, left, right, win_len, overlap,
                                     target_shape):
        ec = EC(fs, win_len=win_len, overlap=overlap)
        alphas, taus = ec.equalize(left, right, 100)
        assert len(alphas) == target_shape

    def test_find_tau_returns_zero_if_outwise_of_allowed_range(self):
        ec = EC(10, win_len=0.5, overlap=0)
        tau = ec._find_tau(np.arange(10), np.arange(10), 63)
        assert_array_equal(tau, 0)

    def test_equalization_of_2d_signal(self):
        ec = EC(4, win_len=1, overlap=None)
        alphas, taus = ec.equalize([(1, 0, 0, 0), (1, 0, 0, 0)],
                                   [(1, 0, 0, 0), (1, 0, 0, 0)],
                                   (100, 200))
        assert_array_equal(alphas, ((1,), (1,)))
        assert_array_equal(taus, ((0, ), (0, )))

    @pytest.mark.parametrize('x', [
        np.array((1, 1, 1)),
        np.array(((1, 1, 1),
                  (1, 1, 1))),
    ])
    def test_return_shape_when_creating_jitter(self, x):
        input_shape = x.shape
        alphas, deltas = self.ec.create_jitter(x)
        assert alphas.shape == input_shape and deltas.shape == input_shape

    @pytest.mark.parametrize("x, alphas, deltas, target", [
        (np.arange(4), np.zeros(4), np.zeros(4), np.arange(4)),
        (np.arange(4), 0.5 * np.ones(4), np.zeros(4), (0, 0.5, 1, 1.5)),
        (np.arange(4), np.zeros(4), np.ones(4), (1, 2, 3, 3)),
        (np.arange(4), np.zeros(4), -np.ones(4), (0, 0, 1, 2)),
        (np.arange(6).reshape((2, 3)), np.zeros((2, 3)),
         np.array(((1, 1, 1), (-1, -1, -1))), ((1, 2, 2), (3, 3, 4))),
    ])
    def test_return_shape_when_applying_jitter(self, x, alphas, deltas, target):
        out = self.ec.apply_jitter(x, alphas, deltas)
        assert_array_equal(out, target)

    def test_setting_out_value_to_input_when_applying_jitter(self):
        x = np.arange(4, dtype='float')
        alphas = np.array((1, 1, 1, 1))
        deltas = np.zeros(4, dtype='int')
        out = self.ec.apply_jitter(x, alphas, deltas, out=x)
        assert_array_equal(out, x)

    def test_out_value_is_different_from_input_when_applying_jitter(self):
        x = np.arange(4, dtype='float')
        alphas = np.array((1, 1, 1, 1))
        deltas = np.zeros(4, dtype='int')
        out = self.ec.apply_jitter(x, alphas, deltas)
        with pytest.raises(AssertionError):
            assert_array_equal(out, x)

    @pytest.mark.parametrize("x, target", [
        ((1, 1, 1), (0.7553155, 0.7553155, 0.55898691)),
        (((1, 1, 1), (1, 1, 1)), ((0.7553155, 0.8999607, 0.7553155),
                                  (0.5331105, 0.5331105, 1.24431947))),
    ])
    def test_jitter_of_signal(self, x, target):
        np.random.seed(0)
        out = self.ec.jitter(x)
        assert_allclose(out, target, atol=1e-5)

    @pytest.mark.parametrize("n_samples, win_len, step, target", [
        (7, 2, 1, 6),
        (7, 4, 2, 2),
        (7, 2, 2, 3),
        (7, 4, 4, 1),
        (7, 4, 1, 4),
        (7, 3, 1, 5),
    ])
    def test_number_of_valid_windows(self, n_samples, win_len, step, target):
        n = self.ec._n_valid_windows(n_samples, win_len, step)
        assert n == target
