from __future__ import division, print_function
from pandas import read_csv
from pambox.intelligibility_models import sii
from numpy import ones
from numpy.testing import assert_allclose



def test_sii():
    """@todo: Docstring for test_sii.
    :returns: @todo

    """
    data = read_csv('./test_files/test_sii.csv')
    for _, E, N, T, I, SII in data.itertuples():
        s = sii.Sii(T=T*ones(18), I=I)
        ss = s.predict(E*ones(18), N*ones(18))
        assert_allclose(ss, SII, rtol=1e-4)

