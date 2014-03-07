from __future__ import division
import numpy as np
from collections import namedtuple, OrderedDict
from itertools import izip

from pambox.intelligibility_models.sepsm import Sepsm
from pambox import general
from pambox import filterbank
from pambox import auditory


CENTER_F = (63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000,
            1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000)


class MrSepsm(Sepsm):
    """Docstring for MrSepsm. """

    def __init__(self, fs=22050, cf=CENTER_F,
                 modf=(1., 2., 4., 8., 16., 32., 64., 128., 256.),
                 downsamp_factor=10,
                 noise_floor=0.001, snr_env_limit=0.001):
        """@todo: to be defined1. """
        Sepsm.__init__(self, fs, cf, modf, downsamp_factor, noise_floor,
                       snr_env_limit)

    def _mr_env_powers(self, channel_env, filtered_envs):
        win_durations = 1. / np.asarray(self.modf, dtype='float')
        win_lengths = [np.floor(win * self.fs / self.downsamp_factor) for win in
                       win_durations]

        # DC power used for normalization. Divided by 2 such that a fully
        # modulated signal has # an AC-power of 1.
        dc_power = np.mean(channel_env) ** 2 / 2

        mr_env_powers = []

        for i_modf, (mod_cf, env, win_length) in enumerate(
                izip(self.modf, filtered_envs, win_lengths)):
            tmp_env_powers = np.zeros(np.ceil(env.shape / win_length))
            for ii, hop in enumerate(
                    range(0, env.shape[0] - int(win_length), int(win_length))):
                # Normalize the variance by N-1, like in MATLAB.
                tmp_env_powers[ii] = np.var(env[hop:hop + win_length],
                                            ddof=1) / dc_power
            else:  # Calculate variance for the last incomplete window.
                ii += 1
                hop = hop + win_length
                tmp_env_powers[ii] = np.var(env[hop:], ddof=1) / dc_power
                mr_env_powers.append(tmp_env_powers)

        return np.asarray(mr_env_powers)

    def _time_average(self, mr_snr_env):
        snr_env_matrix = np.zeros(len(self.modf))
        for ii, v in enumerate(mr_snr_env):
            snr_env_matrix[ii] = v.mean()
        return snr_env_matrix

    def _mr_snr_env(self, mr_env_powers_mix, mr_env_powers_noise):
        """Calculate the multi-resolution SNRenv.

        :param mr_env_powers_mix:
        :param mr_env_powers_noise:
        :returns: tuple( ndarray, list), time-average of the mr-SNRenv, and mr-SNRenv.
        """
        mr_snr_env = []
        for ii, (modf, p_mix, p_noise) in enumerate(izip(self.modf,
                                                         mr_env_powers_mix,
                                                         mr_env_powers_noise)):
            # noisefloor cannot exceed the mix, since they exist at the same time
            p_noise = np.minimum(p_noise, p_mix)

            # the noisefloor restricted to minimum 0.01 reflecting and internal
            # noise threshold
            p_mix = np.maximum(p_mix, self.noise_floor)
            p_noise = np.maximum(p_noise, self.noise_floor)

            # calculation of snrenv
            mr_snr_env.append((p_mix - p_noise) / p_noise)
            mr_snr_env[-1] = np.maximum(mr_snr_env[-1], self.snr_env_limit)

        snr_env_matrix = self._time_average(mr_snr_env)

        return snr_env_matrix, [p_mix, p_noise], mr_snr_env

    def predict(self, clean, mixture, noise):

        fs_new = self.fs / self.downsamp_factor
        N = len(clean)
        N_modf = len(self.modf)
        N_cf = len(self.cf)

        if (clean is None) or (np.array_equal(clean, mixture)):
            signals = (mixture, noise)
        else:
            signals = (clean, mixture, noise)

        downsamp_chan_envs = np.zeros((len(signals),
                                       np.ceil(N / self.downsamp_factor)))
        mod_channel_envs = np.zeros((len(signals),
                                     len(self.modf),
                                     downsamp_chan_envs.shape[-1] - 1))
        snr_env_lin = np.zeros((N_cf, N_modf))
        lt_exc_ptns = np.zeros((3, N_cf, N_modf))
        mr_snr_env_lin = OrderedDict()

        # find bands above threshold
        filtered_rms_mix = filterbank.noctave_filtering(mixture, self.cf,
                                                        self.fs, width=3)
        bands_above_thres_idx = self._bands_above_thres(filtered_rms_mix)

        for idx_band in bands_above_thres_idx:
            channel_envs = \
                np.asarray(
                    [self._peripheral_filtering(signal, self.cf[idx_band])
                     for signal in signals])

            for ii, channel_env in enumerate(channel_envs):
                # Extract envelope
                tmp_env = general.hilbert_envelope(channel_env).squeeze()

                # Low-pass filtering
                tmp_env = auditory.lowpass_env_filtering(tmp_env, 150.0,
                                                         N=1, fs=self.fs)
                # Downsample the envelope for faster processing
                downsamp_chan_envs[ii] = tmp_env[::self.downsamp_factor]

                # Sub-band modulation filtering
                lt_exc_ptns[ii, idx_band], mod_channel_envs[ii] = \
                    filterbank.mod_filterbank(downsamp_chan_envs[ii], fs_new, self.modf)

            mr_env_powers = []
            for chan_env, mod_envs in izip(downsamp_chan_envs,
                                           mod_channel_envs):
                mr_env_powers.append(self._mr_env_powers(chan_env, mod_envs))

            snr_env_lin[idx_band], _, mr_snr_env_lin[idx_band] \
                = self._mr_snr_env(*mr_env_powers[-2:])  # Select only the env
            # powers from the mixture and the noise, even if we calculated the
            # envelope powers for the clean speech.

        snr_env = self._optimal_combination(snr_env_lin, bands_above_thres_idx)

        res = namedtuple('Results', ['snr_env', 'snr_env_matrix', 'exc_ptns',
                                     'bands_above_thres_idx'])
        res.snr_env = snr_env
        res.snr_env_db = 10 * np.log10(snr_env)
        res.snr_env_matrix = snr_env_lin
        # res.exc_ptns = mr_env_powers
        res.bands_above_thres_idx = bands_above_thres_idx
        return res


