# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
from numpy.testing import assert_allclose
import scipy.io as sio
import pytest
from tests import __DATA_ROOT__

from pambox.intelligibility_models.slidingmrsepsm import SlidingMrSepsm


@pytest.fixture
def mat_complete():
    return sio.loadmat(__DATA_ROOT__ + '/test_mr_sepsm_full_prediction.mat',
                       squeeze_me=True)


def test_sliding_mr_sepsm_with_section(mat_complete):
    mix = mat_complete['test']
    noise = mat_complete['noise']
    fs = 22050

    smr = SlidingMrSepsm(fs=fs)

    sections = ((0.9, 1.5), )

    res = smr.predict(mix, mix, noise, sections)

    assert_allclose(
        res['sections_snr_env']
        , 6.98
        , rtol=0.01
    )


def test_sliding_mr_sepsm_without_section(mat_complete):
    mix = mat_complete['test']
    noise = mat_complete['noise']
    fs = 22050

    smr = SlidingMrSepsm(fs=fs)

    res = smr.predict(mix, mix, noise)
    print(mat_complete['tmp']['SNRenv'].astype('float'))
    print(res['snr_env'])

    assert_allclose(
        res['snr_env']
        , 15.317
        , rtol=0.01
    )
