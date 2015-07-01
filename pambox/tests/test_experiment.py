# -*- coding: utf-8 -*-
from __future__ import division, print_function
import os.path

import numpy as np
from numpy.testing import assert_allclose
import pytest

from pambox.speech import Experiment


__DATA_ROOT__ = os.path.join(os.path.dirname(__file__), 'data')

class TestExperiment(object):
    @pytest.mark.parametrize("fixed_target, target, masker, snr, exp_target, "
                             "exp_masker", (
        (True, [0, 1], [0, 1], 0, [0, 2514.86685937], [0, 2514.86685937]),
        (True, [0, 1], [0, 1], 5, [0, 2514.86685937], [0, 1414.21356237]),
        (False, [0, 1], [0, 1], 0, [0, 2514.86685937], [0, 2514.86685937]),
        (False, [0, 1], [0, 1], 5, [0, 4472.135955], [0, 2514.86685937]),
    ))
    def test_adjust_levels(self, fixed_target,
                           target, masker, snr,
                           exp_target, exp_masker):
        exp = Experiment([], [], [], fixed_target=fixed_target, fixed_level=65)
        target, masker = exp.adjust_levels(target, masker, snr)
        assert_allclose(target, exp_target, atol=1e-6)
        assert_allclose(masker, exp_masker, atol=1e-6)

    @classmethod
    def distort_passthrough(target, masker, *args, **kwargs):
        return target, masker

    def test_preprocessing(self):
        params = {}
        target = np.asarray([0, 1])
        masker = np.asarray([0, 1])
        snr = 0

        exp_target = [0, 1]
        exp_mix = [0, 2]
        exp_masker = [0, 1]

        exp = Experiment([], [], [], distortion=self.distort_passthrough,
                           dist_params=params, adjust_levels_bef_proc=True,
                           fixed_level=-3.0102999566398125)
        target, mix, masker = exp.preprocessing(target, masker, snr, params)
        assert_allclose(target, exp_target)
        assert_allclose(mix, exp_mix)
        assert_allclose(masker, exp_masker)



