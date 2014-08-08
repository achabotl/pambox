# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os.path
from numpy.testing import assert_allclose
import scipy.io as sio
import pytest
from pambox.speech.slidingmrsepsm import SlidingMrSepsm

__DATA_ROOT__ = os.path.join(os.path.dirname(__file__), 'data')


@pytest.fixture
def mat_complete():
    return sio.loadmat(__DATA_ROOT__ + '/test_mr_sepsm_full_prediction.mat',
                       squeeze_me=True)


@pytest.mark.slow
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


@pytest.mark.slow
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
