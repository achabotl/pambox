# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import os.path

import numpy as np
from numpy.testing import assert_allclose, dec, TestCase

from pambox.speech import material
from pambox import utils

__DATA_ROOT__ = os.path.join(os.path.dirname(__file__), 'data')


def test_set_level():
    # Set the reference level to 100 dB
    ref_level = 100
    c = material.Material(path_to_ssn=os.path.join(__DATA_ROOT__,
                                                   'dummy_ssn.wav'),
                          ref_level=ref_level)
    # But actually create a 97 dB signal.
    sentence_level = 97  # dB SPL
    target = sentence_level
    x = np.random.randn(100)
    x = utils.setdbspl(x, sentence_level)
    # So when setting the level to the reference, we should get the actual
    # sentence level.
    level = utils.dbspl(c.set_level(x, ref_level))
    assert_allclose(level, target)

    # Now if we set a target different from the reference level
    sentence_level = 97
    x = utils.setdbspl(x, sentence_level)
    level = utils.dbspl(c.set_level(x, ref_level + 3))
    assert_allclose(level, sentence_level + 3)
