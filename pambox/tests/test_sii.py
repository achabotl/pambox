# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import os.path

import numpy as np
from numpy.testing import assert_allclose
from pandas import read_csv
import pytest

from pambox.speech.sii import SII


__DATA_ROOT__ = os.path.join(os.path.dirname(__file__), 'data')


@pytest.fixture(scope='module')
def data():
    d = read_csv(__DATA_ROOT__ + '/test_sii.csv')
    return list(d.itertuples())


@pytest.mark.parametrize('_, E, N, T, I, target', data())
def test_sii(_, E, N, T, I, target):
    """@todo: Docstring for test_sii.
    :returns: @todo

    """
    s = SII(T=T * np.ones(18), I=I)
    ss = s.predict_spec(clean=E*np.ones(18), noise=N*np.ones(18))
    assert_allclose(ss['p']['sii'], target, rtol=1e-4)

