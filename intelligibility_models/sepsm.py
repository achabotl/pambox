from __future__ import division
import numpy as np
import scipy as sp
import matplotlib.pylab as plt
from collections import namedtuple

from pambox import general
from pambox import filterbank
from pambox import auditory


CENTER_F = (63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000,
            1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000)
HT_DIFFUSE = (37.5, 31.5, 26.5, 22.1, 17.9, 14.4, 11.4, 8.4, 5.8, 3.8, 2.1,
              1.0, 0.8, 1.9, 0.5, -1.5, -3.1, -4.0, -3.8, -1.8, 2.5, 6.8)
MODF = (1., 2., 4., 8., 16., 32., 64.)
FS = 22050.0


class Sepsm(object):

    """Implement the sEPSM intelligibility model"""

    def __init__(self, fs=22050, cf=CENTER_F, modf=MODF, downsamp_factor=10,
                 noise_floor=0.01, snr_env_limit=0.001):
        """@todo: to be defined1. """
        self.fs = fs
        self.cf = cf
        self.modf = modf
        self.downsamp_factor = downsamp_factor
        self.noise_floor = noise_floor
        self.snr_env_limit = snr_env_limit

    def _peripheral_filtering(self, signal, center_f):
        b, a, _, _, _ = auditory.gammatone_make(self.fs, center_f)
        # Factor 2 to do like in the sEPSM, although I don't really understand
        # why...
        y = 2 * auditory.gammatone_apply(signal, b, a)
        return y

    def _bands_above_thres(self, x):
        """Select bands above threshold



        Parameters
        ----------
        x : array_like, rms value of each peripheral channel.

        Returns
        -------
        idx : array_like, indexes of the bands that are above threshold.

        """
        noise_rms_db = 20 * np.log10(x)
        # convert to spectrum level according to ANSI 1997
        noise_spec_level_corr = noise_rms_db \
            - 10.0 * np.log10(sp.array(self.cf) * 0.231)
        max_idx = min(len(noise_spec_level_corr), len(HT_DIFFUSE))
        b = noise_spec_level_corr[:max_idx] > HT_DIFFUSE[:max_idx]
        idx = np.arange(len(noise_rms_db))
        return idx[b]

    def _snr_env(self, signals, fs):
        """calculate SNR_env for a signal mixture and a noise.

        :signals: namedtuple of ndarrays, channel envelopes for the clean
            speech, mixture and noise alone, in that order
        :fs: int, sampling frequency at which to do the modulation analysis.
        :returns: ndarray, ndarray
            lin snrenv values and modulation excitation patterns

        """
        fs = np.array(fs, dtype='float')
        signals = [np.array(signal, dtype='float') for signal in signals]

        exc_ptns = sp.empty((3, len(self.modf)))
        # for each stimulus
        for ii, signal in enumerate(signals):
            # modulation filtering
            exc_ptns[ii] = filterbank.mod_filterbank(signal, fs,
                                                       self.modf)

        # set nan values to zero
        exc_ptns[np.isnan(exc_ptns)] = 0

        # noisefloor cannot exceed the mix, since they exist at the same time
        exc_ptns[2] = np.minimum(exc_ptns[2], exc_ptns[1])

        # the noisefloor restricted to minimum 0.01 reflecting and internal
        # noise threshold
        exc_ptns[1] = np.maximum(exc_ptns[1], self.noise_floor)
        exc_ptns[2] = np.maximum(exc_ptns[2], self.noise_floor)

        # calculation of snrenv
        snr_env = (exc_ptns[1] - exc_ptns[2]) / (exc_ptns[2])
        snr_env = np.maximum(snr_env, self.snr_env_limit)

        return snr_env, exc_ptns

    def predict(self, clean, mixture, noise):
        """Predicts intelligibility

        :clean: @todo
        :mix: @todo
        :noise: @todo
        :returns: @todo

        """
        fs_new = self.fs / self.downsamp_factor
        N = len(clean)
        N_modf = len(self.modf)
        N_cf = len(self.cf)

        # find bands above threshold
        filtered_mix_rms = filterbank.noctave_filtering(mixture, self.cf,
                                                        self.fs, width=3)
        bands_above_thres_idx = self._bands_above_thres(filtered_mix_rms)

        snr_env_lin = np.zeros((N_cf, N_modf))
        exc_ptns = np.zeros((3, N_cf, N_modf))
        # For each band above threshold,
        # (actually, for debug purposes, maybe I should keep all...)
        for idx_band in bands_above_thres_idx:
            # Peripheral filtering, of just the band we process
            filtered_signals = \
                np.array([self._peripheral_filtering(signal,
                                                     self.cf[idx_band])
                          for signal in [clean, mixture, noise]])

            downsamp_env = np.empty((3, np.ceil(N / self.downsamp_factor)))
            for i, signal in enumerate(filtered_signals):
                # Extract envelope
                tmp_env = general.hilbert_envelope(signal).squeeze()

                # Low-pass filtering
                tmp_env = auditory.lowpass_env_filtering(tmp_env, 150.0,
                                                         N=1, fs=self.fs)
                # Downsample the envelope for faster processing
                downsamp_env[i] = tmp_env[::self.downsamp_factor]

            # Calculate SNRenv for the current channel
            snr_env_lin[idx_band], exc_ptns_tmp = self._snr_env(downsamp_env,
                                                                fs_new)
            for i_sig in range(3):
                exc_ptns[i_sig, idx_band, :] = exc_ptns_tmp[i_sig]

        snr_env = np.sqrt(np.sum(snr_env_lin[bands_above_thres_idx] ** 2,
                                 axis=-1))
        snr_env = np.sqrt(np.sum(snr_env ** 2))

        res = namedtuple('Results', ['snr_env', 'snr_env_matrix', 'exc_ptns',
                                     'bands_above_thres_idx'])
        res.snr_env = snr_env
        res.snr_env_matrix = snr_env_lin
        res.exc_ptns = exc_ptns
        res.bands_above_thres_idx = bands_above_thres_idx

        return res

def plot_mod_powers(mod_powers_all, cf, modf):
    # File / Factors / SNR / SIGNAL / CF / MODF

    # Make a 4x4 figure, with the excitation patterns for 4 CF and 4
    # FACTOR/SNR conditions
    f = plt.figure()
    i_ax = 0
    ax = []
    for base, i_snr in enumerate([0, 3, 6, 8]):
        base = base * 4 + 1
        for i, i_cf in enumerate([6, 9, 12, 15]):
            ax.append(f.add_subplot(4, 4, base + i))
            for i_sig in range(3):
                ax[i_ax].plot(mod_powers_all[0, 0, i_snr, i_sig, i_cf, :])
            i_ax += 1
    plt.legend(('Clean', 'Mix', 'Noise'))
    for ii in range(12, 16):
        ax[ii].set_xticklabels(modf)
    for ii in range(0, 16, 4):
        ax[ii].set_ylabel()
