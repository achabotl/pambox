# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import os.path
from pandas import read_csv
from pambox.speech.sii import Sii
import numpy as np
from numpy.testing import assert_allclose


__DATA_ROOT__ = os.path.join(os.path.dirname(__file__), 'data')


def test_sii():
    """@todo: Docstring for test_sii.
    :returns: @todo

    """
    data = read_csv(__DATA_ROOT__ + '/test_sii.csv')
    for _, E, N, T, I, SII in data.itertuples():
        s = Sii(T=T*np.ones(18), I=I)
        ss = s.predict_spec(E*np.ones(18), N*np.ones(18))
        assert_allclose(ss['p']['sii'], SII, rtol=1e-4)

