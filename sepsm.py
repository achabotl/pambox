from __future__ import division
import numpy as np
import scipy as sp
import scipy.io.wavfile
import general
import filterbank
import auditory
import distort
from collections import namedtuple


MIDFREQ = (63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000,
           1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000)
HT_DIFFUSE = (37.5, 31.5, 26.5, 22.1, 17.9, 14.4, 11.4, 8.4, 5.8, 3.8, 2.1,
              1.0, 0.8, 1.9, 0.5, -1.5, -3.1, -4.0, -3.8, -1.8, 2.5, 6.8)
FS = 22050.0


class SepsmSignals(namedtuple('Signals', 'clean mix noise')):
    pass


def auditory_processing(signal, center_f=MIDFREQ, fs=FS):
    # Gammatone filering
    b, a, _, _, _ = auditory.gammatone_make(fs, center_f)
    y = auditory.gammatone_apply(signal, b, a)
    return general.hilbert_envelope(y)


def bands_above_thres(x, c, fs):
    noise_rms_db = general.dbspl(x)
    # Convert to spectrum level according to ANSI 1997
    noise_spec_level_corr = noise_rms_db - 10.0 * sp.log10(sp.array(c) * 0.231)
    return noise_spec_level_corr > HT_DIFFUSE[0:len(noise_spec_level_corr)]


def srn_env(signals, fs):
    """Calculate SNR_env for a signal mixture and a noise.


    :signals: namedtuple of ndarrays, channel envelopes for the clean speech,
              mixture and noise alone, in that order
    :fs:      int, sampling frequency
    :returns: ndarray, namedtuple, log SNRenv values and modulation excitation
              patters

    """

    mod_powers = sp.zeros((3, 7))
    # For each stimulus
    for ii, signal in enumerate(signals):
        # Modulation filtering
        mod_powers[ii] = filterbank.mod_filterbank(signal, fs)

    # Set NaN values to zero
    mod_powers[np.isnan(mod_powers)] = 0

    # Noisefloor cannot exceed the mix, since they exist at the same time
    mod_powers[2] = sp.minimum(mod_powers[2], mod_powers[1])

    # The noisefloor restricted to minimum 0.01 reflecting and internal noise
    # threshold
    mod_powers[1] = sp.maximum(mod_powers[1], 0.01)
    mod_powers[2] = sp.maximum(mod_powers[2], 0.01)

    # calculation of SNRenv
    snr_env = 10 * sp.log10((mod_powers[1] - mod_powers[2]) /
                            (mod_powers[2]))

    # SNRenv - values are truncated to minimum -30 dB.
    return sp.maximum(-30, snr_env), mod_powers
