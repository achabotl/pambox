from __future__ import absolute_import, division, print_function
from collections import namedtuple
import logging

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import plot, xlabel, ylabel, legend
from scipy.signal import butter, filtfilt

from pambox.central import EC
from pambox.speech import MrSepsm

log = logging.getLogger(__name__)


Ears = namedtuple('Ears', ['left', 'right'])


class BsEPSM(MrSepsm):

    """Binaural implementation of the sEPSM model.

    Implementation used in [chabot-leclerc2016]_.

    References
    ----------
    .. [chabot-leclerc2016]

    """

    def __init__(self,
                 fs=22050,
                 name='BinauralMrSepsm',
                 cf=MrSepsm._default_center_cf,
                 modf=MrSepsm._default_modf,
                 downsamp_factor=10,
                 noise_floor=0.001,
                 snr_env_limit=0.001,
                 sigma_e=0.25,
                 sigma_d=105e-6,
                 fast_cancel=True,
                 debug=False,
                 win_len=0.02,
                 ec_padding_windows=10
                 ):
        """@todo: to be defined1. """
        MrSepsm.__init__(self, fs, cf, modf, downsamp_factor, noise_floor,
                         snr_env_limit, output_time_signals=True)
        self.name = name
        self.overlap = 0.5
        self.win_len = win_len
        self.sigma_e = sigma_e
        self.sigma_d = sigma_d
        self.ec_padding_windows = ec_padding_windows
        self.fast_cancel = fast_cancel
        self._key_alpha = 'alpha'
        self._key_tau = 'tau'
        self.debug = debug
        self.env_lp_cutoff = 770  # Hz, from breebaart2001binaurala
        self.env_lp_order = 5  # from breebaart2001binaurala
        self._ec_process = EC(fs, win_len=self.win_len, overlap=self.overlap,
                              sigma_d=self.sigma_d, sigma_e=self.sigma_e,
                              fast_cancel=fast_cancel,
                              padding_windows=self.ec_padding_windows)

    def _ec_equalize(self, left, right):
        alphas, taus = self._ec_process.equalize(left, right, self.cf)
        return alphas, taus

    def _ec_cancel(self, left, right, alphas, taus):
        cancelled_mix = self._ec_process.cancel(
            left, right, alphas, taus)
        cancelled_mix = np.abs(cancelled_mix)
        return cancelled_mix

    def _apply_bu_process(self, left, right, bands=None):
        """Apply EC process between left and right envelopes (mix and noise),
        apply sEPSM processing to the resulting signals and calculate the
        SNR_env.

        Parameters
        ----------
        left, right : dictionary
            Outputs of the mr-sEPSM model. The dictionaries must have a
            'chan_envs' key.
        bands : list of int
            Indices of the channels to process. If `None`, all channels are
            processes. Defaults to None.

        Returns
        -------
        bu_mr_snr_env_matrix : ndarray
            Multi-resolution SNRenv matrix of shape N_CHAN x N_SAMPLES.
        alphas : ndarray of float
            Optimal gains, in samples, calculated by the "equalize"
            process.
        tau : ndarray of integers
            Optimal time delays, in samples, calculated by the "equalize"
            process.

        """
        left_mix_envs = left['chan_envs'][-2]
        right_mix_envs = right['chan_envs'][-2]
        left_noise_envs = left['chan_envs'][-1]
        right_noise_envs = right['chan_envs'][-1]

        # ... then we find the alpha and tau parameters the minimize the noise
        # energy...
        alphas, taus = self._ec_equalize(left_noise_envs, right_noise_envs)
        # ... then we perform the cancellation with those alphas and taus for
        #  the mixture and noise...
        cancelled_mix = self._ec_cancel(left_mix_envs, right_mix_envs, alphas,
                                        taus)
        cancelled_noise = self._ec_cancel(left_noise_envs, right_noise_envs,
                                          alphas, taus)

        # ... then we apply the same processing as the mr-sEPSM, until we
        # have the multi-resolution excitation patterns...
        mix_mr_exc_ptns = self._apply_sepsm_processing(cancelled_mix[
            np.newaxis])
        noise_mr_exc_ptns = self._apply_sepsm_processing(cancelled_noise[
            np.newaxis])

        # --- BU SNRenv ---
        # ... we can finally calculate the BU SNRenv by calculating the
        # SNRenv.
        bu_mr_snr_env_matrix, _ = self._snr_env(
            mix_mr_exc_ptns,
            noise_mr_exc_ptns
        )
        return bu_mr_snr_env_matrix, alphas, taus, mix_mr_exc_ptns, noise_mr_exc_ptns

    def _apply_sepsm_processing(self, envs):
        filtered_envs, _ = self._mod_filtering(envs)
        mr_exc_ptns = self._mr_env_powers(envs, filtered_envs)
        return mr_exc_ptns[0]

    def _apply_be_process(self, left, right):
        # --- Better ear (BE) ---
        be_mr_snr_env_matrix = self._better_ear(
            left['mr_snr_env_matrix'],
            right['mr_snr_env_matrix'],
            left['bands_above_thres_idx'],
            right['bands_above_thres_idx']
        )
        return be_mr_snr_env_matrix

    def _better_ear(self, left, right, left_idx, right_idx):
        """Return the better-ear SNRenv for bands above threshold only.

        Parameters
        ----------
        left, right: ndarray
            SNR_env values, of shape (N_CHAN, N_WIN).
        left_idx, right_idx: array_like
            Index of the bands above threshold for the left and right ear,
            respectively.

        """
        left_idx = np.asarray(left_idx)
        right_idx = np.asarray(right_idx)

        be_snr_env = np.zeros_like(left)
        for side, idx in zip((left, right), (left_idx, right_idx)):
            try:
                be_snr_env[idx] = np.maximum(be_snr_env[idx], side[idx])
            except IndexError:
                # BE SNRenv is not modified.
                pass
        return be_snr_env

    def _calc_bu_bands_above_thres(self, left, right):
        """Calculate bands above threshold for binaural unmasking.

        A band is considered above threshold if both bands are above
        threshold (logical 'and').

        Parameters
        ----------
        left, right : dictionaries
            Outputs from the mr-sEPSM prediction. Must have a
            'bands_above_thres_idx' key.

        Returns
        -------
        idx : array
            Indices of the bands that are above threshold in at least one side.

        """

        left_bands_idx = left["bands_above_thres_idx"]
        right_bands_idx = right["bands_above_thres_idx"]

        # BU mask is when _both sides_ are above threshold.
        indices = list(set(left_bands_idx) & set(right_bands_idx))
        return indices

    def _calc_be_bands_above_thres(self, left, right):
        """True if at least one side is above threshold.

        Parameters
        ----------
        left, right : dictionaries
            Outputs from the mr-sEPSM prediction. Must have a
            'bands_above_thres_idx' key.

        Returns
        -------
        idx : array
            Indices of the bands that are above threshold in at least one side.
        """

        left_bands_idx = left["bands_above_thres_idx"]
        right_bands_idx = right["bands_above_thres_idx"]

        indices = list(set(left_bands_idx) | set(right_bands_idx))
        return indices

    def _apply_ba_process(self, be, bu, be_indices, bu_indices):
        """Applies the binaural-advantage process.

        The BA advantage selection is actually the exact same thing as the BE
        process: only bands above threshold for *that* signal are
        considered for the comparison.

        Parameters
        ----------
        be, bu : ndarray
            Better-ear and Binaural-unmasking SNRenv.
        be_indices, bu_indices : lists of integers
            List of the indices for the channels that were above threshold
            for each input.

        Returns
        -------
        ba : ndarray
            Combination of the better-ear and binaural unmasking SNRenv.

        """

        ba = self._better_ear(be, bu, be_indices, bu_indices)
        return ba

    def predict(self, clean=None, mixture=None, noise=None):
        """Predict intelligibility.

        Parameters
        ----------
        clean, mixture, noise : ndarray
            Binaural input signals.

        Returns
        -------
        res : dict
            Model predictions and internal values. Model predictions are
            stored as a dictionary under the key `'p'`.

        """
        # Calculate the mr-sEPSM prediction for each ear in one call..
        binaural_res = [super(BsEPSM, self).predict(clean=c, mix=m, noise=n)
                        for c, m, n in zip(clean, mixture, noise)]
        # ... and save them independently...
        ears_res = Ears(*binaural_res)
        log.debug('Left bands above threshold {}.'.format(
            ears_res.left["bands_above_thres_idx"]))
        log.debug('Right bands above threshold {}.'.format(ears_res.right[
            "bands_above_thres_idx"]))

        # ... then apply the binaural unmasking (BU) process, which includes the
        # EC process and the mr-sEPSM process applied to the cancelled
        # signals...
        bu_mr_snr_env_matrix, alphas, taus, bu_mix_mr_exc_ptns, \
            bu_noise_mr_exc_ptns \
            = self._apply_bu_process(ears_res.left, ears_res.right)

        # ... in "parallel", we apply the better-ear (BE) process to the
        # multi-resolution SNRenv...
        be_mr_snr_env_matrix = self._apply_be_process(ears_res.left,
                                                      ears_res.right)

        # ... then we select the bands that are considered "above threshold"
        # for the BU, BE and binaural advantage (BA)...
        bu_idx_above_thres = self._calc_bu_bands_above_thres(ears_res.left,
                                                             ears_res.right)
        log.debug('BU bands above threshold {}.'.format(bu_idx_above_thres))
        be_idx_above_thres = self._calc_be_bands_above_thres(ears_res.left,
                                                             ears_res.right)
        log.debug('BE bands above threshold {}.'.format(be_idx_above_thres))
        ba_idx_above_thres = list(
            set(be_idx_above_thres) | set(bu_idx_above_thres))
        log.debug('BA bands above threshold {}.'.format(ba_idx_above_thres))


        # ... then we combine the BE and BU as part of the "binaural
        # advantage"...
        ba_mr_snr_env_matrix = self._apply_ba_process(
            be_mr_snr_env_matrix,
            bu_mr_snr_env_matrix,
            be_idx_above_thres,
            bu_idx_above_thres)

        # ... we can now averaging over time the multi-resolution
        # representation...
        time_av_bu_snr_env = self._time_average(bu_mr_snr_env_matrix)
        time_av_be_snr_env = self._time_average(be_mr_snr_env_matrix)
        time_av_ba_snr_env = self._time_average(ba_mr_snr_env_matrix)

        # ... and combine the SNRenv for the bands that are above threshold
        # for each output type.
        bu_snr_env = self._optimal_combination(
            time_av_bu_snr_env,
            bu_idx_above_thres)
        be_snr_env = self._optimal_combination(
            time_av_be_snr_env,
            be_idx_above_thres)
        ba_snr_env = self._optimal_combination(
            time_av_ba_snr_env,
            ba_idx_above_thres)

        # Additional variation, where the multi-resolution representation is
        # not average over time at first. The whole mr representation is
        # combined optimally.
        full_bu_snr_env = self._optimal_combination(
            bu_mr_snr_env_matrix,
            bu_idx_above_thres
        )
        full_be_snr_env = self._optimal_combination(
            be_mr_snr_env_matrix,
            be_idx_above_thres
        )
        full_ba_snr_env = self._optimal_combination(
            ba_mr_snr_env_matrix,
            ba_idx_above_thres
        )

        res = {
            'p': {
                'be_snr_env': be_snr_env,
                'bu_snr_env': bu_snr_env,
                'ba_snr_env': ba_snr_env,
                'full_be_snr_env': full_be_snr_env,
                'full_bu_snr_env': full_bu_snr_env,
                'full_ba_snr_env': full_ba_snr_env,
                'snr_env_l': ears_res.left['p']['snr_env'],
                'snr_env_r': ears_res.right['p']['snr_env']
            },
        }
        if self.debug:
            res.update({
                'be_matrix': be_mr_snr_env_matrix,
                'bu_matrix': bu_mr_snr_env_matrix,
                'ba_matrix': ba_mr_snr_env_matrix,
                'be_idx_above_threshold': be_idx_above_thres,
                'bu_idx_above_threshold': bu_idx_above_thres,
                'ba_idx_above_threshold': ba_idx_above_thres,
                'ears': ears_res,
                'time_av_be_snr_env': time_av_be_snr_env,
                'time_av_ba_snr_env': time_av_ba_snr_env,
                'time_av_bu_snr_env': time_av_bu_snr_env,
                'bu_mix_mr_exc_ptns': bu_mix_mr_exc_ptns,
                'bu_noise_mr_exc_ptns': bu_noise_mr_exc_ptns,
                self._key_alpha: alphas,
                self._key_tau: taus
            })
        return res  # Results for each ear's sEPSM model.

    def plot_alpha(self, res):
        alphas = res[self._key_alpha]
        t = np.arange(alphas.shape[-1]) * self.overlap * self.win_len
        plot(t, alphas.T)
        xlabel('Time (sec)')
        ylabel('$alpha_0$ gains (L / R)')
        legend(self.cf,
               loc='outside',
               bbox_to_anchor=(1.05, 1))

    def plot_alpha_hist(self, res, ymax=None):
        alphas = res[self._key_alpha]
        plt.boxplot(alphas.T, labels=self.cf)
        plt.setp(plt.xticks()[1], rotation=30)
        xlabel('Channel frequency (Hz)')
        ylabel(r'$\alpha_0$ gains (L / R)')
        plt.ylim([0, ymax])
        plt.xticks(rotation=30)

    def plot_tau(self, res):
        tau = res[self._key_tau]
        t = np.arange(tau.shape[-1]) * self.overlap * self.win_len
        plot(t, tau.T)
        xlabel('Time (sec)')
        ylabel('Tau (s)')
        legend(self.cf,
               loc='outside',
               bbox_to_anchor=(1.05, 1))

    def plot_tau_hist(self, res, cfs=None, bins=None, return_ax=False):
        """Plot histogram of tau values.

        Parameters
        ----------
        res : dict
            Results from the `predict` function.
        cfs : list
            Index of center frequencies to plot.
        bins : int
            Number of bins in the histogram. If `None`, uses bins between
            -700 us and 700 us. Default is `None`.
        return_ax : bool, optional
            If True, returns the figure Axes. Default is False.
        """
        taus = res[self._key_tau]

        edges = np.max(np.abs(taus))

        if bins is None:
            bins = np.arange(-edges, edges, 20e-6)

        # Put together all ITDs if no particular channel is chosen...
        if cfs is None:
            fig, ax = plt.subplots(1, 1)
            ax.hist(taus.ravel(), bins=bins)
            ax.set_ylabel('Count for all channels')
        else:
            # ... or create N subplots if more than one channel is chosen.
            try:
                iter(cfs)
            except TypeError:
                cfs = (cfs, )
            cfs = cfs[::-1]
            fig, axes = plt.subplots(len(cfs), 1, sharex=True, sharey=True)
            try:
                iter(axes)
            except TypeError:
                axes = (axes,)
            for ax, cf in zip(axes, cfs):
                ax.hist(taus[cf], bins=bins)
            [ax.set_ylabel('@ {} Hz'.format(self.cf[i_cf]))
             for ax, i_cf in zip(axes, cfs)]

        ax.set_xlabel('Interaural delay (ms)')
        # ax.set_xlim((-800e-6, 800e-6))
        ticks = ax.get_xticks()
        ax.set_xticklabels(ticks * 1e3)
        if return_ax and cfs is None:
            return axes
        else:
            return ax

    def _extract_env(self, channel_sigs):
        """Extracts the envelope via half-wave rectification and low-pass
        filtering and jitters the envelopes.

        Parameters
        ----------
        channel_sigs : ndarray
            Peripheral subband signals.

        Returns
        -------
        env : ndarray

        """
        envelopes = np.maximum(channel_sigs, 0)
        b, a = butter(self.env_lp_order, self.env_lp_cutoff * 2. / self.fs)
        envelopes = filtfilt(b, a, envelopes)

        epsilons, deltas = self._ec_process.create_jitter(envelopes[0])
        for i_sig, signal in enumerate(envelopes):
            envelopes[i_sig] = self._ec_process.apply_jitter(signal, epsilons, deltas)
        return envelopes

    def _mod_sensitivity(self, envs):
        """Doesn't do anything to the envelopes"""
        return envs