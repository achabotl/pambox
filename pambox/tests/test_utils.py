# -*- coding: utf-8 -*-
from __future__ import division, print_function
import os.path
import pytest
from pambox import utils
import numpy as np


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
        np.testing.assert_allclose(utils.dbspl(x, ac=ac, offset=offset,
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
        np.testing.assert_allclose(utils.rms(x, ac=ac, axis=axis), target)


def test_set_level():
    tests = (
        ((0, 1), 65, 100, (0., 0.02514867)),
        ((0, 1), 65, 0, (0., 2514.86685937)),
        ((0, 1), 100, 100, (0., 1.41421356)),
    )

    for x, level, offset, target in tests:
        y = utils.setdbspl(x, level, offset=offset)
        np.testing.assert_allclose(y, target, atol=1e-4)


def test_envelope_extraction():
    x = np.array(
        [-0.00032745, -0.00031198, -0.00029605, -0.00027965, -0.00026281,
         -0.00024553, -0.00022783, -0.00020972])
    target = np.array(
        [0.00068165, 0.00068556, 0.00068946, 0.00069335, 0.00069725,
         0.00070113, 0.00070502, 0.0007089])
    envelope = utils.hilbert_envelope(x)
    np.testing.assert_allclose(envelope, target, atol=1e-3)


def test_hilbert_env_on_2d_array_with_last_dimension():
    tests = (
        ([0.70710678, 1.56751612, 2., 1.56751612, 0.70710678],
         [0, 1, 2, 1, 0]),
        ([0.70710678, 1.56751612, 2., 1.56751612, 0.70710678],
         [0, 1, 2, 1, 0]),
        ([[0., 1.], [0., 1.]],
         [[0, 1], [0, 1]]),
        ([[0.5, 1., 0.5], [2.5, 3.16227766, 1.5]],
         [[0, 1, 0], [2, 3, 0]]),
    )

    for target, x in tests:
        env = utils.hilbert_envelope(x)
        np.testing.assert_allclose(env, target,
                                   err_msg="Input was {}".format(x))


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
        np.testing.assert_allclose(utils.make_same_length(*inputs),
                                   targets)


def test_psy_fn():
    x = -3.0
    mu = 0.
    sigma = 1.0
    target = 0.13498980316300957
    y = utils.psy_fn(x, mu, sigma)
    np.testing.assert_allclose(y, target)
