# -*- coding: utf-8 -*-
from __future__ import division, print_function
import pytest
import scipy.io as sio
from pambox import idealobs
import numpy as np
from tests import __DATA_ROOT__


@pytest.fixture
def data():
    return np.asarray([0.28032187,   1.07108181,   3.35513227,   8.66774961,
                       18.61914334,  33.63172026,  51.87228063,  69.72236134,
                       83.79127082,  92.72205919,  97.28779782,  99.16754416])


@pytest.fixture
def idealobs_parameters():
    return (3.74647303e+00, 5.15928999e-02, -9.09197905e-07, 8000.)


@pytest.fixture
def snr():
    return np.arange(-9, 3, 1)


@pytest.fixture
def snrenv(snr):
    return 10. ** np.linspace(-2, 2, len(snr))


def test_fit_obs(data, snrenv, idealobs_parameters):
    c = idealobs.IdealObs()
    c.fit_obs(snrenv, data)
    params = c.get_params()
    res = [params['k'], params['q'], params['sigma_s']]
    np.testing.assert_allclose(res, idealobs_parameters[0:3], atol=1e-5)


def test_psy_fn():
    mat = sio.loadmat(__DATA_ROOT__ + '/test_psychometric_function.mat')
    x = mat['x'][0]
    mu = 0.
    sigma = 1.0
    target = mat['p'][0] * 100
    y = idealobs.psy_fn(x, mu, sigma)
    np.testing.assert_allclose(y, target)


def test_snr_env_to_pc(snrenv, idealobs_parameters, data):
    c = idealobs.IdealObs(k=0.81, q=0.5, sigma_s=0.6, m=8000.)
    snrenvs = np.asarray([2.6649636, 6.13623543, 13.1771355, 24.11754981,
                          38.35865445, 55.59566425])
    pc = c.snrenv_to_pc(snrenvs)
    target = np.asarray([1.62223958e-02, 4.52538073e-01, 1.02766152e+01,
                         5.89991555e+01, 9.57537063e+01, 9.99301187e+01])
    np.testing.assert_allclose(pc, target, atol=1e-4)


def test_get_params():
    p = {'k': 1, 'q': 2, 'sigma_s': 0.5, 'm': 800}
    c = idealobs.IdealObs(**p)
    assert p == c.get_params()


def test_fit_obs_set_m_and_sigma_s(data, snrenv):
    c = idealobs.IdealObs()

    tests = (((1.42765076, 0.390529, 0.6, 12), (0.6, 12)),
             ((3.6590, 0.10341, 0.6, 8000), (0.6, 8000)),
             ((3.7464, 0.05159, -1.2144e-4, 8000), (None, 8000)))

    for target, values in tests:
        sigma_s, m = values
        c.fit_obs(snrenv, data, sigma_s=sigma_s, m=m)
        params = c.get_params()
        res = [params['k'], params['q'], params['sigma_s'], params['m']]
        np.testing.assert_allclose(res, target, atol=1e-4)

