# -*- coding: utf-8 -*-
"""
:mod:`pambox.distort` regroups various types of distortions and processings
that can be applied to signals.
"""
from __future__ import division, print_function

import numpy as np
import scipy as sp
from six.moves import zip
from scipy.io import wavfile

from pambox import utils
from pambox.utils import fftfilt, hilbert
import six

try:
    _ = np.use_fastnumpy  # MKL FFT optimizations from Enthought.
    from numpy.fft import fft, ifft, rfft, irfft
except AttributeError:
    try:
        import mklfft  # MKL FFT optimizations from Continuum Analytics
        from numpy.fft import fft, ifft, rfft, irfft
    except ImportError:
        from scipy.fftpack import fft, ifft
        from numpy.fft import rfft, irfft


def mix_noise(clean, noise, sent_level, snr=None):
    """Mix a signal signal noise at a given signal-to-noise ratio.

    Parameters
    ----------
    clean : ndarray
        Clean signal.
    noise : ndarray
        Noise signal.
    sent_level : float
        Sentence level, in dB SPL.
    snr :
        Signal-to-noise ratio at which to mix the signals, in dB. If snr is
        `None`,  no noise is mixed with the signal (Default value = None)

    Returns
    -------
    tuple of ndarrays
        Returns the clean signal, the mixture, and the noise.

    """

    # Pick a random section of the noise
    n_clean = len(clean)
    n_noise = len(noise)
    if n_noise > n_clean:
        start_idx = np.random.randint(n_noise - n_clean)
        noise = noise[start_idx:start_idx + n_clean]

    if snr is not None:
        # Get speech level and set noise level accordingly
        # clean_level = utils.dbspl(clean)
        # noise = utils.setdbspl(noise, clean_level - snr)
        noise = noise / utils.rms(noise) * 10 ** ((sent_level - snr) / 20)
        mix = clean + noise
    else:
        mix = clean

    return clean, mix, noise


def phase_jitter(x, a):
    """
    Apply phase jitter to a signal.

    The expression of phase jitter is:

    .. math:: y(t) = s(t) * cos(\Phi(t)),

    where :math:`\Phi(t)` is a random process uniformly distributed over
    :math:`[0, 2\pi\alpha]`. The effect of the jitter when \alpha is 0.5 or 1
    is to completely destroy the carrier signal, effictively yielding
    modulated white noise.

    Parameters
    ----------
    x : ndarray
       Signal
    a : float
        Phase jitter parameter, typically between 0 and 1, but it can be
        anything.

    Returns
    -------
    ndarray
        Processed signal of the same dimension as the input signal.

    """
    n = len(x)
    return x * np.cos(2 * np.pi * a * np.random.random_sample(n))


def reverb(x, rt):
    """
    Applies reverberation to a signal.

    Parameters
    ----------
    x : ndarray
       Input signal.
    rt : float
        Reverberation time


    Returns
    -------
    ndarray
        Processed signal.

    """
    pass


def spec_sub(x, noise, factor, w=1024 / 2., padz=1024 / 2., shift_p=0.5):
    """
    Apply spectral subtraction to a signal.

    The defaul values of the parameters are typical for a sampling frequency of
    44100 Hz. Note that (W+padz) is the final frame window and hence the fft
    length (it is normally chose as a power of 2).

    Parameters
    ----------
    x : ndarray
        Input signal
    noise :
        Input noise signal
    factor : float
        Noise subtraction factor, must be larger than 0.
    w : int
        Frame length, in samples. (Default value = 1024 / 2.)
    padz : int
        Zero padding (pad with padz/2 from the left and the right) (Default
        value = 1024 / 2.)
    shift_p : float
         Shift percentage (overlap) between each window, in fraction of the
         window size (Default value = 0.5)

    Returns
    -------
    clean_estimate : ndarray
        Estimate of the clean signal.
    noise_estimate : ndarray
        Estimate of the noisy signal.

    """
    wnd = np.hanning(w + 2)  # create hanning window with length = W
    wnd = wnd[1:-1]

    stim = np.vstack((x, noise))

    len_signal = stim.shape[-1]  # Signal length
    shift_p_indexes = np.floor(w * shift_p)
    n_segments = np.floor((len_signal - w) / shift_p_indexes + 1)
    len_segment = w + padz * 2 * shift_p
    y = np.empty((2, n_segments, len_segment))
    # Initialize arrays for spectral subtraction. Use only positive
    # frequencies.
    Y_hat = np.empty((n_segments, len_segment / 2 + 1))
    PN_hat = Y_hat.copy()

    # For each signal
    for k in range(2):
        # CUT THE APPROPRIATE SIGNAL FRAMES
        indexes = np.tile(np.arange(w), (n_segments, 1))
        index_shift = np.arange(n_segments) * shift_p_indexes
        indexes = indexes + index_shift[:, np.newaxis]
        y_tmp = stim[k]
        y_tmp = y_tmp[indexes.astype('int')] * wnd
        # PAD WITH ZEROS
        pad = np.zeros((n_segments, padz / 2))
        y_pad = np.hstack((pad, y_tmp, pad))
        y[k, :, :] = y_pad

    # FREQUENCY DOMAIN

    # signal:
    Y = fft(y[0])
    # YY = Y(1:round(end/2)+1,:); # Half window (exploit the symmetry)
    YY = Y[:, :(len_segment / 2 + 1)]  # Half window (exploit the symmetry)
    YPhase = np.angle(YY)  # Phase
    Y1 = np.abs(YY)  # Spectrum
    Y2 = Y1 ** 2  # Power Spectrum

    # noise:
    Y_N = fft(y[1])
    YY_N = Y_N[:, :(len_segment / 2 + 1)]  # Half window (exploit the symmetry)
    Y_NPhase = np.angle(YY_N)  # Phase
    Y_N1 = np.abs(YY_N)  # Spectrum
    Y_N2 = Y_N1 ** 2  # Power Spectrum

    # The noise "estimate" is simply the average of the noise power
    # spectral density in the frame:
    P_N = Y_N2.mean(axis=-1)

    Y_hat = Y2 - factor * P_N[:, np.newaxis]  # subtraction
    Y_hat = np.maximum(Y_hat, 0)  # Make the minima equal zero
    PN_hat = Y_N2 - factor * P_N[:, np.newaxis]  # subtraction for noise alone
    # PN_hat = np.maximum(PN_hat, 0)
    PN_hat[Y_hat == 0] = 0

    Y_hat[0:2, :] = 0
    PN_hat[0:2, :] = 0
    # Combining the estimated power spectrum with the original noisy phase,
    # and add the frames using an overlap-add technique
    output_Y = overlap_and_add(np.sqrt(Y_hat), YPhase, (w + padz), shift_p * w)
    output_N = overlap_and_add(np.sqrt(PN_hat.astype('complex')),
                               Y_NPhase, (w + padz), shift_p * w)

    return output_Y, output_N


def overlap_and_add(powers, phases, len_window, shift_size):
    """Reconstruct a signal with the overlap and add method.

    Parameters
    ----------
    powers : ndarray
        Magnitude of the power spectrum of the signal to reconstruct.
    phases : ndarray
        Phase of the signal to reconstruct.
    len_window : int
        Frame length, in samples.
    shift_size : int
        Shift length. For non overlapping signals, in would equal `len_window`.
        For 50% overlapping signals, it would be `len_window/2`.

    Returns
    -------
    ndarray
        Reconstructed time-domain signal.

    """
    len_window = int(len_window)
    shift_size = int(shift_size)
    n_frames, len_frame = powers.shape
    spectrum = powers * np.exp(1j * phases)
    signal = np.zeros(n_frames * shift_size + len_window - shift_size)

    # Create full spectrum, by joining conjugated positive spectrum
    if len_window % 2:
        # Do no duplicate the DC bin
        spectrum = np.hstack((spectrum, np.conj(np.fliplr(spectrum[:, 1:]))))
    else:
        # If odd-numbered, do not duplicated the DC ans FS/2 bins
        spectrum = np.hstack((spectrum,
                              np.conj(np.fliplr(spectrum[:, 1:-1]))))

    signal = np.zeros((n_frames - 1) * shift_size + len_window)

    for i_frame, hop in enumerate(range(0,
                                        len(signal) - int(len_window) + 1,
                                        int(shift_size))):
        signal[hop:hop + len_window] \
            += np.real(ifft(spectrum[i_frame], len_window))
    return signal


class WestermannCrm(object):
    """Applies HRTF and BRIR for a given target and masker distance.

    Parameters
    ----------
    fs : int
         Samping frequenc of the process. (Default value = 40000)

    Attributes
    ----------
    brir : dict
        Binaural room impulse responses for each distance.
    delays : dict
        Delay until the first peak in the BRIR for each distance.
    dist : ndarray
        List of the valid distances (0.5, 2, 5, and 10 meters).

    References
    ----------
    .. [1] A. Westermann and J. M. Buchholz: Release from masking through
        spatial separation in distance in hearing impaired listeners.
        Proceedings of Meetings on Acoustics 19 (2013) 050156.
    """

    def __init__(self, fs=40000):
        self.dist = np.asarray([0.5, 2, 5, 10])
        self.fs = fs
        self.brir = self._load_brirs()
        self.delays = self._find_delay()

    def _load_brirs(self):
        """Loads BRIRs from file."""
        brirs = {}
        for d in self.dist:
            fname = '../stimuli/crm/brirs_{fs}/aud{d_str}m.wav'.format(
                fs=self.fs,
                d_str=self._normalize_fname(d)
            )
            wav = wavfile.read(fname)
            brirs[d] = np.array(wav[1].astype('float') / 2. ** 15).T
        return brirs

    def _find_delay(self):
        """Calculates the delay of the direct sound, in samples."""
        delays = {}
        for k, v in six.iteritems(self.brir):
            x = np.mean(v, axis=0)
            delays[k] = np.abs(x).argmax()
        return delays

    @staticmethod
    def _normalize_fname(d):
        """

        Parameters
        ----------
        d : float

        Returns
        -------

        """
        if d > 1:
            d_str = str('%d' % d)
        else:
            d_str = str(d).replace('.', '')
        return d_str

    def _load_eqfilt(self, tdist, mdist):
        """
        Returns the equalization filter for the pair of target and masker.

        Parameters
        ----------
        tdist : float
            Target distance in meters. Must be in the set (0.5, 2, 5, 10).
        mdist :
            Masker distance in meters. Must be in the set (0.5, 2, 5, 10).

        Returns
        -------
        ndarray
            Equalization filter.

        """
        eqfilt_name = 't{}m_m{}m.mat'.format(self._normalize_fname(tdist),
                                             self._normalize_fname(mdist))
        eqfilt_path = '../stimuli/crm/eqfilts_{}/{}'.format(self.fs,
                                                            eqfilt_name)
        try:
            eqfilt = sp.io.loadmat(eqfilt_path, squeeze_me=True)
        except IOError:
            raise IOError('Cannot file file %s' % eqfilt_path)
        return eqfilt

    def apply(self, x, m, tdist, mdist, align=True):
        """Applies the "Westermann" distortion to a target and masker.

        target and masker are not co-located, the masker is equalized before
        applying the BRIR, so that both the target and masker will have the
        same average spectrum after the BRIR filtering.

        By default, the delay introduced by the BRIR is compensated for,
        such that the maxiumum of the BRIR happen simulatenously.

        Parameters
        ----------
        x : ndarray
            Mono clean speech signal of length `N`.
        m : ndarray
            Mono masker signal of length `N`.
        tdist : float
            Target distance, in meters.
        mdist : float
            Masker distance, in meters.
        align : bool
            Compensate for the delay in the BRIRs with distance  (default is
            `True`).

        Returns
        -------
        mix : (2, N) ndarray
            Mixture processesed by the BRIRs.
        noise : (2, N)
            Noise alone processed by the BRIRs.
        """
        if tdist not in self.dist or mdist not in self.dist:
            raise ValueError('The distance values are incorrect.')

        n_orig = x.shape[-1]

        # Filter target with BRIR only
        out_x = np.asarray([fftfilt(b, x) for b in self.brir[tdist]])

        # Equalized masker and then apply the BRIR
        if tdist == mdist:
            m = [m, m]
        else:
            eqfilt = self._load_eqfilt(tdist, mdist)
            m = [fftfilt(b, m) for b in [eqfilt['bl'], eqfilt['br']]]

        out_m = np.asarray([fftfilt(b, chan) for b, chan
                            in zip(self.brir[mdist], m)])

        if align:
            i_x, i_m = self._calc_aligned_idx(tdist, mdist)
        else:
            i_x = 0
            i_m = 0

        # Pad with zeros if necessary, so that the lengths stay the same
        out_x, out_m = utils.make_same_length(out_x[:, i_x:], out_m[:, i_m:])
        return out_x, out_m

    def _calc_aligned_idx(self, tdist, mdist):
        """Calculates the index of the required delay to align the max of the
        BRIRs

        Parameters
        ----------
        tdist :
            float, distance to target, in meters
        mdist :
            float, distance to masker, in meters
            :return: tuple, index of the target and masker.

        Returns
        -------
        i_x : int
            Index of earliest peak in the signal.
        i_m : int
            Index of the earliest peak in the maskers.
        """
        # location of earliest peak
        m_is_shortest = np.argmin([self.delays[tdist], self.delays[mdist]])
        if m_is_shortest:
            i_x = self.delays[tdist] - self.delays[mdist]
            i_m = 0
        else:
            i_x = 0
            i_m = self.delays[mdist] - self.delays[tdist]
        return i_x, i_m


def noise_from_signal(x, fs=40000, keep_env=False):
    """Create a noise with same spectrum as the input signal.

    Parameters
    ----------
    x : array_like
        Input signal.
    fs : int
         Sampling frequency of the input signal. (Default value = 40000)
    keep_env : bool
         Apply the envelope of the original signal to the noise. (Default
         value = False)

    Returns
    -------
    ndarray
        Noise signal.

    """
    x = np.asarray(x)
    n_x = x.shape[-1]
    n_fft = utils.next_pow_2(n_x)
    X = rfft(x, utils.next_pow_2(n_fft))
    # Randomize phase.
    noise_mag = np.abs(X) * np.exp(
        2 * np.pi * 1j * np.random.random(X.shape[-1]))
    noise = np.real(irfft(noise_mag, n_fft))
    out = noise[:n_x]

    if keep_env:
        env = np.abs(hilbert(x))
        [bb, aa] = sp.signal.butter(6, 50 / (fs / 2))  # 50 Hz LP filter
        env = sp.signal.filtfilt(bb, aa, env)
        out *= env

    return out
