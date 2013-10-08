from __future__ import division
import numpy as np
import scipy as sp
import scipy.io.wavfile
import pandas as pd
import matplotlib.pylab as plt

from pambox import general
from pambox import filterbank
from pambox import auditory
from pambox import distort
from pambox import idealobs


MIDFREQ = (63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000,
           1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000)
HT_DIFFUSE = (37.5, 31.5, 26.5, 22.1, 17.9, 14.4, 11.4, 8.4, 5.8, 3.8, 2.1,
              1.0, 0.8, 1.9, 0.5, -1.5, -3.1, -4.0, -3.8, -1.8, 2.5, 6.8)
MODF = (1., 2., 4., 8., 16., 32., 64.)
FS = 22050.0


class Sepsm(object):

    """Implement the sEPSM intelligibility model"""

    def __init__(self, fs=22050, cf=MIDFREQ, modf=MODF, downsamp_factor=10,
                 noise_floor=0.01, snrenv_limit=-30):
        """@todo: to be defined1. """
        self.fs = fs
        self.cf = cf
        self.modf = modf
        self.downsamp_factor = downsamp_factor
        self.noise_floor = noise_floor
        self.snrenv_limit = snrenv_limit

    def _auditory_processing(self, signal, center_f):
        b, a, _, _, _ = auditory.gammatone_make(self.fs, center_f)
        y = auditory.gammatone_apply(signal, b, a)
        return general.hilbert_envelope(y)

    def _bands_above_thres(self, x):
        noise_rms_db = general.dbspl(x)
        # convert to spectrum level according to ansi 1997
        noise_spec_level_corr = noise_rms_db \
            - 10.0 * sp.log10(sp.array(self.cf) * 0.231)
        max_idx = min(len(noise_spec_level_corr), len(HT_DIFFUSE))
        return noise_spec_level_corr[:max_idx] > HT_DIFFUSE[:max_idx]

    def _snr_env(self, signals, fs):
        """calculate SNR_env for a signal mixture and a noise.

        :signals: namedtuple of ndarrays, channel envelopes for the clean
            speech, mixture and noise alone, in that order
        :fs: int, sampling frequency at which to do the modulation analysis.
        :returns: ndarray, namedtuple, log snrenv values and modulation
            excitation patters

        """
        fs = np.array(fs, dtype='float')
        signals = [np.array(signal, dtype='float') for signal in signals]

        mod_powers = sp.empty((3, len(self.modf)))
        # for each stimulus
        for ii, signal in enumerate(signals):
            # modulation filtering
            mod_powers[ii] = filterbank.mod_filterbank(signal, fs,
                                                       self.modf)

        # set nan values to zero
        mod_powers[np.isnan(mod_powers)] = 0

        # noisefloor cannot exceed the mix, since they exist at the same time
        mod_powers[2] = np.minimum(mod_powers[2], mod_powers[1])

        # the noisefloor restricted to minimum 0.01 reflecting and internal
        # noise threshold
        mod_powers[1] = np.maximum(mod_powers[1], self.noise_floor)
        mod_powers[2] = np.maximum(mod_powers[2], self.noise_floor)

        # calculation of snrenv
        snr_env_db = 10 * np.log10((mod_powers[1] - mod_powers[2])
                                   / (mod_powers[2]))

        # snrenv - values are truncated to minimum -30 db.
        return sp.maximum(self.snrenv_limit, snr_env_db), mod_powers

    def predict(self, clean, mix, noise):
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
        filtered_noise, _ = filterbank.noctave_filtering(noise, self.cf,
                                                         self.fs, width=3)
        bands_above_thres_idx = self._bands_above_thres(filtered_noise)

        snr_env_dbs = np.empty((N_cf, N_modf))
        mod_powers = np.empty((3, N_cf, N_modf))
        # For each band above threshold,
        # (actually, for debug purposes, maybe I should keep all...)
        for idx_band, _ in enumerate(bands_above_thres_idx):
            # Peripheral filtering, of just the band we process
            filtered_signals = \
                np.array([self._auditory_processing(signal,
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
            snr_env_dbs[idx_band], mod_powers_tmp \
                = self._snr_env([downsamp_env[0], downsamp_env[1],
                                 downsamp_env[2]], fs_new)
            for i_sig in range(3):
                mod_powers[i_sig, idx_band, :] = mod_powers_tmp[i_sig]

        return snr_env_dbs, mod_powers


def load_clue_file(file_number):
    root_name = '../stimuli/clue/sentencesWAV22/adjustedCLUEsent'
    name = root_name + '{:03d}'.format(file_number) + '.wav'
    int_sentence = scipy.io.wavfile.read(name)
    return np.array(int_sentence[1].astype('float') / int_sentence[0])


def load_clue_files(n_files):
    root_name = '../stimuli/clue/sentencesWAV22/adjustedCLUEsent'
    names = [root_name + '{:03d}'.format(i) + '.wav'
             for i in range(1, n_files + 1)]
    int_sentences = [scipy.io.wavfile.read(name) for name in names]
    return np.array([int_sentence[1].astype('float') / int_sentence[0]
                     for int_sentence in int_sentences])


if __name__ == '__main__':

    # Set conditions
    factors = [0, 1, 8]
    snrs = np.arange(-12, 5, 4)

    # Read files
    n_files = 1
    sentences = load_clue_files(n_files)

    # Set speech level
    speech_level = 65
    sentences = np.array([general.setdbspl(sentence, speech_level)
                          for sentence in sentences])

    # load noise
    full_noise = scipy.io.wavfile.read('../stimuli/clue/SSN_CLUE22.wav')
    full_noise = full_noise[1].astype('float') / full_noise[0]

    distortion_name = 'SpecSub'
    factor = 0
    CF = np.array(MIDFREQ)
    FS = 22050
    W = 512
    PADZ = 512
    SHIFT_P = 0.5

    columns = ['File', 'SNR', 'Distortion', 'Factor', 'SNRenv']
    df_results = pd.DataFrame(columns=columns)
    mod_columns = columns + ['Signal']
    df_mod_powers = pd.DataFrame(columns=mod_columns)

    sepsm = Sepsm(fs=FS, cf=CF, modf=MODF)

    # for each sentence
    mod_powers_all = np.empty((n_files, len(factors), len(snrs),
                               3, len(CF), len(MODF)))
    # for i_sent, sentence in enumerate(sentences):
    for i_sent, sentence in enumerate(sentences):
        print('Starting sentence %d...' % (i_sent + 1))
        N = len(sentence)

        # For each set of distortion parameters
        for i_snr, snr in enumerate(snrs):
            signals = distort.mix_noise(sentence, full_noise, snr)

            for i_factor, factor in enumerate(factors):

                # Add noise and apply distortion
                mixture, noise = distort.spec_sub(signals[1], signals[2],
                                                  factor, W, PADZ, SHIFT_P)

                # Truncate signals to remove silences at beginning and end
                mixture = mixture[1.5 * W:N]
                noise = noise[1.5 * W:N]
                clean = sentence[:len(mixture)]

                snr_env_dbs, mod_powers = sepsm.predict(clean, mixture, noise)

                snr_env_lins = 10 ** (snr_env_dbs / 10)

                mod_powers_all[i_sent, i_factor, i_snr] = mod_powers

                for ii, signal_type in enumerate(['Clean', 'Mix', 'Noise']):
                    df_mod_powers_tmp = pd.DataFrame(mod_powers[ii],
                                                     columns=MODF)
                    df_mod_powers_tmp['File'] = i_sent
                    df_mod_powers_tmp['Signal'] = signal_type
                    df_mod_powers_tmp['Channel'] = CF
                    df_mod_powers_tmp['SNR'] = snr
                    df_mod_powers_tmp['Factor'] = factor
                    df_mod_powers_tmp['Distortion'] = distortion_name

                df_mod_powers = df_mod_powers.append(df_mod_powers_tmp)
                df_results = df_results.\
                    append({'File': i_sent + 1,
                            'SNR': snr,
                            'Distortion': distortion_name,
                            'Factor': factor,
                            'SNRenv': np.sqrt(np.sum(snr_env_lins ** 2))
                            }, ignore_index=True)

        print('Done with sentence %d...' % (i_sent + 1))

    # Convert to percent correct
    ido = idealobs.IdealObs()
    to_pc = lambda x: ido.snrenv_to_pc(x)
    df_results['Intelligibility'] = df_results['SNRenv'].map(to_pc)

    print(df_results)


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



# for key, grp in df_results.groupby(['Factor','SNR']):
    # print(key)
    # plot(grp.SNR, grp.Intelligibility.mean(), label=key)
