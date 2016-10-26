# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import os

import numpy as np
from numpy import asarray
from numpy.testing import assert_array_equal
import pytest

from pambox.speech import BsEPSM


__DATA_ROOT__ = os.path.join(os.path.dirname(__file__), 'data')


class TestBinauralMrSepsmReturnValues(object):

    @classmethod
    def setup_class(cls):
        cls.b = BsEPSM()

    @pytest.mark.parametrize("left, right, left_idx, right_idx, target", [
        # Take best of each side
        (asarray((0, 1, 0, 1)), asarray((0, 0, 1, 1)),
            range(4), range(4), (0, 1, 1, 1)),
        # Take left only
        (asarray((0, 1, 0, 1)), asarray((0, 0, 1, 1)),
            np.zeros(4, dtype='int'), range(4), (0, 0, 1, 1)),
        # Take right only
        (asarray((0, 1, 0, 1)), asarray((0, 0, 1, 1)),
            range(4), np.zeros(4, dtype='int'), (0, 1, 0, 1)),
        # Pick from both sides
        (asarray((0, 1, 0, 5)), asarray((0, 0, 3, 4)),
            (0, 1), (2, 3), (0, 1, 3, 4)),

    ])
    def test_better_ear(self, left, right, left_idx, right_idx, target):
        be = self.b._better_ear(left, right, left_idx, right_idx)
        assert_array_equal(be, target)

    @pytest.mark.parametrize("left, right, target", [
        ({'bands_above_thres_idx': (0, 1)},
         {'bands_above_thres_idx': (2, 3)},
         ()),
        ({'bands_above_thres_idx': (2, 3)},
         {'bands_above_thres_idx': (2, 3)},
         (2, 3)),
        ({'bands_above_thres_idx': (0, 1)},
         {'bands_above_thres_idx': (1, 3)},
         1),
    ])
    def test_calculate_bu_bands_above_threshold(self, left, right, target):
        b = BsEPSM(cf=range(4))
        bu_mask = b._calc_bu_bands_above_thres(left, right)
        assert_array_equal(bu_mask, target)

    @pytest.mark.parametrize("left, right, target", [
        ({'bands_above_thres_idx': (1, 2)},
         {'bands_above_thres_idx': (2, 3)},
         (1, 2, 3)),
    ])
    def test_calculate_be_bands_above_threshold(self, left, right, target):
        b = BsEPSM(cf=range(4))
        be_mask = b._calc_be_bands_above_thres(left, right)
        assert_array_equal(be_mask, target)

    @pytest.mark.parametrize("left, right, target", [
        # Take best of each side
        ({'mr_snr_env_matrix': asarray((0, 1, 0, 1)), 'bands_above_thres_idx':
            range(4)},
         {'mr_snr_env_matrix': asarray((0, 0, 1, 1)), 'bands_above_thres_idx':
             range(4)},
         (0, 1, 1, 1)),
        # Take left only
        ({'mr_snr_env_matrix': asarray((0, 1, 0, 1)), 'bands_above_thres_idx':
            np.zeros(4, dtype='int')},
         {'mr_snr_env_matrix': asarray((0, 0, 1, 1)), 'bands_above_thres_idx':
             range(4)},
         (0, 0, 1, 1)),
        # Take right only
        ({'mr_snr_env_matrix': asarray((0, 1, 0, 1)), 'bands_above_thres_idx':
            range(4)},
         {'mr_snr_env_matrix': asarray((0, 0, 1, 1)), 'bands_above_thres_idx':
             np.zeros(4, dtype='int')},
         (0, 1, 0, 1)),
        # Pick from both sides
        ({'mr_snr_env_matrix': asarray((0, 1, 0, 5)), 'bands_above_thres_idx':
            (0, 1)},
         {'mr_snr_env_matrix': asarray((0, 0, 3, 4)), 'bands_above_thres_idx':
             (2, 3)},
         (0, 1, 3, 4)),
    ])
    def test_apply_be_process(self, left, right, target):
        out = self.b._apply_be_process(left, right)
        assert_array_equal(out, target)


