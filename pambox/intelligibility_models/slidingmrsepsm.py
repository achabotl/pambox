from __future__ import division, print_function, absolute_import
from six.moves import zip
import numpy as np
from pambox.intelligibility_models.mrsepsm import MrSepsm
from pambox import filterbank, general, auditory


class SlidingMrSepsm(MrSepsm):
    """
    Sliding mr-sEPSM.


    :param min_win: float, minimum length of a window, in ms. (Default is
    `None`.)
    """
    _default_modf = (1., 2., 4., 8., 16., 32., 64., 128., 256.)

    def __init__(self, fs=22050,
                 cf=(63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
                     1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300,
                     8000),
                 modf=(1., 2., 4., 8., 16., 32., 64., 128., 256.),
                 downsamp_factor=10,
                 noise_floor=0.001,
                 snr_env_limit=0.001):
        self.mr = super(SlidingMrSepsm,
                        self).__init__(fs=fs,
                                       cf=cf,
                                       modf=modf,
                                       downsamp_factor=downsamp_factor,
                                       noise_floor=noise_floor,
                                       snr_env_limit=snr_env_limit)

    @staticmethod
    def _inc_sliding_window(x, win=2, step=1):
        n = len(x)
        step = int(step)
        win = int(win)

        out = np.ma.masked_all((n // step + 1, win))
        for ii, i_step in enumerate(range(0, n + 1, step)):
            i_low = max([0, i_step - win // 2])
            i_max = min([i_step + win // 2, n])
            out[ii, :(i_max-i_low)] = x[i_low:i_max]
            out.mask[ii, :(i_max-i_low)] = False
        return out

    def _mr_env_powers(self, channel_env, filtered_envs):
        len_env = filtered_envs.shape[-1]
        win_durations = 1. / np.asarray(self.modf, dtype='float')
        win_lengths = np.floor(win_durations * self.fs / self.downsamp_factor)
        swin_length = win_lengths.min()
        n_segment = np.ceil(len_env / swin_length)

        # DC power used for normalization. Divided by 2 such that a fully
        # modulated signal has # an AC-power of 1.
        dc_power = np.mean(channel_env) ** 2 / 2

        # Create a masked array of zeros, where all entries are hidden.
        mr_env_powers = np.zeros((len(self.modf), n_segment))

        for i_modf, (env, win_length) in enumerate(
                zip(filtered_envs, win_lengths)):
            slices = self._inc_sliding_window(env, win_length, swin_length)
            mr_env_powers[i_modf, :] = np.var(slices, ddof=1,
                                              axis=-1) / dc_power
        return mr_env_powers

    def predict(self, clean, mixture, noise, sections=None):

        fs_new = self.fs / self.downsamp_factor
        n = len(clean)
        n_modf = len(self.modf)
        n_cf = len(self.cf)

        # Process only the mixture and noise if the clean speech is the same
        # as the noise.
        if (clean is None) or (np.array_equal(clean, mixture)):
            signals = (mixture, noise)
        else:
            signals = (clean, mixture, noise)

        downsamp_chan_envs = np.zeros((len(signals),
                                       np.ceil(n / self.downsamp_factor)
                                       .astype('int')))
        if (downsamp_chan_envs.shape[-1] % 2) == 0:
            len_offset = 1
        else:
            len_offset = 0
        chan_mod_envs = np.zeros((len(signals),
                                  len(self.modf),
                                  downsamp_chan_envs.shape[-1] - len_offset))
        time_av_mr_snr_env_matrix = np.zeros((n_cf, n_modf))
        lt_exc_ptns = np.zeros((len(signals), n_cf, n_modf))
        mr_snr_env_matrix = []
        mr_exc_ptns = []

        # find bands above threshold
        filtered_rms_mix = filterbank.noctave_filtering(mixture, self.cf,
                                                        self.fs, width=3)
        bands_above_thres_idx = self._bands_above_thres(filtered_rms_mix)

        for idx_band in bands_above_thres_idx:
            channel_envs = \
                [self._peripheral_filtering(signal, self.cf[idx_band])
                 for signal in signals]

            for i_sig, channel_env in enumerate(channel_envs):
                # Extract envelope
                tmp_env = general.hilbert_envelope(channel_env).squeeze()

                # Low-pass filtering
                tmp_env = auditory.lowpass_env_filtering(tmp_env, 150.0,
                                                         n=1, fs=self.fs)
                # Downsample the envelope for faster processing
                downsamp_chan_envs[i_sig] = tmp_env[::self.downsamp_factor]

                # Sub-band modulation filtering
                lt_exc_ptns[i_sig, idx_band], chan_mod_envs[i_sig] = \
                    filterbank.mod_filterbank(downsamp_chan_envs[i_sig],
                                              fs_new,
                                              self.modf)

            chan_mr_exc_ptns = []
            for chan_env, mod_envs in zip(downsamp_chan_envs, chan_mod_envs):
                chan_mr_exc_ptns.append(self._mr_env_powers(chan_env, mod_envs))
            mr_exc_ptns.append(chan_mr_exc_ptns)

            time_av_mr_snr_env_matrix[idx_band], _, chan_mr_snr_env_matrix \
                = self._mr_snr_env(*chan_mr_exc_ptns[-2:])  # Select only the
                # env powers from the mixture and the noise, even if we
                # calculated the envelope powers for the clean speech.
            mr_snr_env_matrix.append(chan_mr_snr_env_matrix)

        # Pick only sections that were selected
        section_snr_envs = None
        if sections:
            section_snr_envs = np.zeros((len(self.cf),
                                         len(self.modf),
                                         len(sections)))

            for ii, each in zip(bands_above_thres_idx, mr_snr_env_matrix):
                section_snr_envs[ii] = self.snr_env_for_sections(each,
                                                                 sections)

        lt_snr_env_matrix = super(MrSepsm, self)._snr_env(*lt_exc_ptns[-2:])
        lt_snr_env = self._optimal_combination(lt_snr_env_matrix,
                                               bands_above_thres_idx)

        snr_env = self._optimal_combination(time_av_mr_snr_env_matrix,
                                            bands_above_thres_idx)

        try:
            sections_snr_env = self._optimal_combination(
                np.mean(section_snr_envs,
                        axis=-1),
                bands_above_thres_idx)
        except IndexError:
            sections_snr_env = None

        res = {
            'snr_env': snr_env,
            'snr_env_matrix': time_av_mr_snr_env_matrix,
            # .snr_env_matrix': snr_env_matrix

            # Output of what is essentially the sEPSM.
            'lt_snr_env': lt_snr_env,
            'lt_snr_env_matrix': lt_snr_env_matrix,
            'lt_exc_ptns': lt_exc_ptns,
            # Outputs of the selection process
            'sections_snr_env': sections_snr_env,
            'per_section_snr_env': section_snr_envs,

            # .mr_snr_env': mr_snr_env
            'mr_snr_env_matrix': mr_snr_env_matrix,
            'mr_exc_ptns': mr_exc_ptns,

            'bands_above_thres_idx': bands_above_thres_idx
        }
        return res

    def snr_env_for_sections(self, snr_envs, sections):
        """Calculate the SNRenv for selected sections of signal.

        Parameters
        ----------

        snr_envs : array_like
            Multi-resolution SNRenv values for the current channel. Expect an
            (n_modf, n_windows) array.
        sections : list
            Pairs of start-stop times, in seconds, delimiting the beginning
            and end of each section.

        Returns
        -------
        out : ndarray
            Time-averaged SNRenv for each section. The output array has the
            shaped (n_modf, n_sections).
        """

        # The small "unit of time" at this point is the time window for the
        # highest modulation filter.
        win_dur = 1 / np.max(self.modf)
        n_sections = len(sections)
        sec_snr_env = np.zeros((snr_envs.shape[0], n_sections))
        for i_sec, section in enumerate(sections):
            i_beg, i_end = [each // win_dur for each in section]
            sec_snr_env[:, i_sec] = np.mean(snr_envs[:, i_beg:i_end], axis=-1)
        return sec_snr_env


if __name__ == "__main__":

    import scipy.io as sio
    from tests import __DATA_ROOT__
    from pprint import pprint

    mat_complete = sio.loadmat(__DATA_ROOT__ +
                               '/test_mr_sepsm_full_prediction.mat',
                               squeeze_me=True)
    mix = mat_complete['test']
    noise = mat_complete['noise']
    fs = 22050

    smr = SlidingMrSepsm(fs=fs)

    sections = [(0.9, 1.5)]

    res = smr.predict(mix, mix, noise, sections)
    print("With sections")
    pprint(res['snr_env'])
    pprint(res['sections_snr_env'])
    # pprint(np.mean(res['per_section_snr_env'], axis=-1))

    smr.min_win = 0.02
    res = smr.predict(mix, mix, noise)
    print('No Sections, but with minimum window length')
    pprint(res['snr_env'])
    pprint(res['sections_snr_env'])
