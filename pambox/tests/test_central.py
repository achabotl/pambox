# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import os.path
from numpy.testing import assert_allclose
import pytest
from pambox import central
import numpy as np


__DATA_ROOT__ = os.path.join(os.path.dirname(__file__), 'data')


@pytest.fixture
def data():
    return np.asarray([0.28032187,   1.07108181,   3.35513227,   8.66774961,
                       18.61914334,  33.63172026,  51.87228063,  69.72236134,
                       83.79127082,  92.72205919,  97.28779782,  99.16754416])


@pytest.fixture
def central_parameters():
    return 3.74647303e+00, 5.15928999e-02, -9.09197905e-07, 8000.


@pytest.fixture
def snr():
    return np.arange(-9, 3, 1)


@pytest.fixture
def snrenv(snr):
    return 10. ** np.linspace(-2, 2, len(snr))


def test_fit_obs(data, snrenv, central_parameters):
    c = central.IdealObs()
    c.fit_obs(snrenv, data)
    params = c.get_params()
    res = [params['k'], params['q'], params['sigma_s']]
    np.testing.assert_allclose(res, central_parameters[0:3], atol=1e-5)


def test_snr_env_to_pc(snrenv, central_parameters, data):
    c = central.IdealObs(k=0.81, q=0.5, sigma_s=0.6, m=8000.)
    snrenvs = np.asarray([2.6649636, 6.13623543, 13.1771355, 24.11754981,
                          38.35865445, 55.59566425])
    pc = c.transform(snrenvs)
    target = np.asarray([1.62223958e-02, 4.52538073e-01, 1.02766152e+01,
                         5.89991555e+01, 9.57537063e+01, 9.99301187e+01])
    np.testing.assert_allclose(pc, target, atol=1e-4)


def test_get_params():
    p = {'k': 1, 'q': 2, 'sigma_s': 0.5, 'm': 800}
    c = central.IdealObs(**p)
    assert p == c.get_params()


def test_fit_obs_set_m_and_sigma_s(data, snrenv):
    c = central.IdealObs()

    tests = (((1.42765076, 0.390529, 0.6, 12), (0.6, 12)),
             ((3.6590, 0.10341, 0.6, 8000), (0.6, 8000)),
             ((3.7464, 0.05159, -1.2144e-4, 8000), (None, 8000)))

    for target, values in tests:
        sigma_s, m = values
        c.fit_obs(snrenv, data, sigma_s=sigma_s, m=m)
        params = c.get_params()
        res = [params['k'], params['q'], params['sigma_s'], params['m']]
        np.testing.assert_allclose(res, target, atol=1e-4)


def test_mod_filtering_for_simple_signal():
    signal = np.asarray([1, 0, 1, 0, 1])
    fs = 2205
    modf = np.asarray([1., 2., 4., 8., 16., 32., 64.])
    mfb = central.EPSMModulationFilterbank(fs, modf)
    p, _ = mfb.filter(signal)
    target = np.asarray([6.69785298e-18, 6.06375859e-06, 2.42555385e-05,
                         9.70302212e-05, 3.88249957e-04, 1.55506496e-03,
                         6.25329663e-03])
    assert_allclose(p, target, rtol=1e-2)
