from __future__ import division, print_function
import numpy as np
from collections import namedtuple
from itertools import izip
from pambox.intelligibility_models.mrsepsm import MrSepsm


class BinauralMrSepsm(MrSepsm):
    """Binaural implementation of the sEPSM model. """

    def __init__(self, fs=22050, cf=MrSepsm._default_center_cf,
                 modf=MrSepsm._default_modf,
                 downsamp_factor=10,
                 noise_floor=0.001, snr_env_limit=0.001):
        """@todo: to be defined1. """
        MrSepsm.__init__(self, fs, cf, modf, downsamp_factor, noise_floor,
                         snr_env_limit)


    def _mr_better_ear(self, res):
        """Calculate the multi-resolution better-ear SNRenv.

        Args:
            (self, res)

        Returns:

        """
        # The mr-SNRenv is not a numpy matrix but a N_s x N_m list of N_x long
        # numpy arrays, where N_s is the number of bands above threshold and N_m is
        #  the number of modulation channels, which is constant.



    def predict(self, clean, mixture, noise):
        """Predict intelligibility.

        :clean: @todo
        :mixture: @todo
        :noise: @todo
        :returns: @todo

        """
        binaural_res = [MrSepsm.predict(self, c, m, n)
                        for c, m, n in izip(clean, mixture, noise)]
        Ears = namedtuple('Ears', ['left', 'right'])
        ears_res = Ears(*binaural_res)

        # Better ear (BE)
        be_matrix = self._better_ear(binaural_res)
        be_mask = self._get_bands_above_threshold(binaural_res)
        be_snr_env = MrSepsm._optimal_combination(self, be_matrix, be_mask)

        res = {
            'be_snr_env': be_snr_env,
            'be_matrix': be_matrix,
            'be_above_thres': be_mask,
            'ears': ears_res
        }
        return res  # Results for each ear's sEPSM model.

    def _better_ear(self, res):
        """Return the better-ear SNRenv for bands above threshold only.

        :param res: Results from Sepsm.predict()
        :type res: namedtuple
        :returns: List of 3 ndarray, for clean speech, mixture, and noise which
            contains the better ear SNRenv.

        """
        be_snr_env = np.zeros_like(res[0]['snr_env_matrix'])
        for ii in range(2):
            idx = res[ii]['bands_above_thres_idx']
            be_snr_env[idx] = \
                np.maximum(be_snr_env[idx], res[ii]['snr_env_matrix'][idx])
        return be_snr_env

    def _get_bands_above_threshold(self, res):
        """Return a mask for bands above threshold in at least one ear.

        :param res: Results return by the Sepsm.predict() function.
        :type res: structured array.
        :returns: ndarray of booleans -- The mask, it has the same dimensions
            as res.snr_env_matrix.

        """
        at = {}
        for ii in range(2):
            at[ii] = np.zeros((len(self.cf), len(self.modf)))
            at[ii][res[ii]['bands_above_thres_idx']] = 1
        all_above_thres = np.logical_or(at[0], at[1])
        return all_above_thres
