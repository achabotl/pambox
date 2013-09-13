import pytest
import filterbank
import numpy as np


def test_mod_filtering_for_simple_signal():
    signal = np.array([1, 0, 1, 0, 1])
    fs = 2205
    p = filterbank.mod_filterbank(signal, fs)
    target = np.array([6.69785298e-18, 6.06375859e-06, 2.42555385e-05,
                      9.70302212e-05, 3.88249957e-04, 1.55506496e-03,
                      6.25329663e-03])
    np.testing.assert_allclose(p, target, rtol=1e-2)
