from __future__ import division, print_function, absolute_import
from itertools import izip
import numpy as np
from pambox.intelligibility_models.mrsepsm import MrSepsm


class SlidingMrSepsm(MrSepsm):
    _default_modf = (1., 2., 4., 8., 16., 32., 64., 128., 256.)


    def __init__(self, fs=22050,
                 cf=(63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000,
                1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000),
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
                izip(filtered_envs, win_lengths)):
            win_iter = self._inc_sliding_window(env, win_length, swin_length)
            for i_win, each in enumerate(win_iter):
                mr_env_powers[i_modf, i_win] = np.var(each, ddof=1) / dc_power
        return mr_env_powers