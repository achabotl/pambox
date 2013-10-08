from __future__ import division
from __future__ import print_function
import numpy as np
import scipy as sp
import general


def mix_noise(clean, noise, snr=None):
    """Mix a signal signal noise at a given signal-to-noise ratio

    :x: ndarray, clean signal.
    :noise: ndarray, noise signal.
    :snr: float, signal-to-noise ratio. If ignored, no noise is mixed, i.e.
          clean == mix.
    :returns: tuple, returns the clean signal, the mixture, and the noise
              alone.

    """
    N = len(clean)
    # Pick a random section of the noise
    nNoise = len(noise)
    # Select a random portion of the noise.
    startIdx = np.random.randint(nNoise - N)
    noise = noise[startIdx:startIdx + N]

    # Get speech level and set noise level accordingly
    clean_level = general.dbspl(clean)
    noise = general.setdbspl(noise, clean_level - snr)

    if snr is not None:
        mix = clean + noise
    else:
        mix = clean

    return clean, mix, noise


def phase_jitter(x, a):
    """Apply phase jitter to a signal

    :x: ndarray, signal.
    :a: float, phase jitter parameter, between 0 and 1.
    :returns: ndarray, processed signal

    """
    N = len(x)
    return x * np.cos(2 * np.pi * a * np.random.random_sample(N))


def reverb(x, rt):
    """@todo: Docstring for reverb.

    :x: @todo
    :rt: @todo
    :returns: @todo

    """
    pass


def spec_sub(x, noise, factor, w=1024/2., padz=1024/2., shift_p=0.5):
    """Apply spectral subtraction to a signal

    Typical values of the parameters, for a sampling frequency of 44100 Hz
    W = 1024
    padz = 1024; %zero padding (pad with padz/2 from the left and padz/2 from
    the right )Note that (W+padz) is the final frame window and hence the fft
    length (it is normally chose as a power of 2)
    shift_p = 0.5 # %Shift percentage is 50%

    :x: ndarray, signal
    :noise: ndarray, noise
    :factor: float, the over-subtraction factor
    :w: int, frame length
    :padz: int, zero padding (pad with padz/2 from the left and the right)
    :shift_p: float, shift percentage (overlap)
    :returns: tuple of ndarrays, estimate of clean signal and estimate of
              noisy signal.

    """
    wnd = np.hanning(w)  # create hanning window with length = W

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
    Y = sp.fftpack.fft(y[0])
    #YY = Y(1:round(end/2)+1,:); # Half window (exploit the symmetry)
    YY = Y[:, :(len_segment / 2 + 1)]  # Half window (exploit the symmetry)
    YPhase = np.angle(YY)  # Phase
    Y1 = np.abs(YY)  # Spectrum
    Y2 = Y1 ** 2  # Power Spectrum

    #  noise:
    Y_N = sp.fftpack.fft(y[1])
    YY_N = Y_N[:, :(len_segment / 2 + 1)]  # Half window (exploit the symmetry)
    Y_NPhase = np.angle(YY_N)  # Phase
    Y_N1 = np.abs(YY_N)  # Spectrum
    Y_N2 = Y_N1 ** 2  # Power Spectrum

    # The noise "estimate" is simply the average of the noise power
    # spectral density in the frame:
    P_N = Y_N2.mean(axis=-1)
    Y_hat = Y2 - factor * P_N[:, np.newaxis]     # subtraction
    PN_hat = Y_N2 - factor * P_N[:, np.newaxis]  # subtraction for noise alone
    Y_hat = np.maximum(Y_hat, 0)  # Make the minima equal zero
    PN_hat = np.maximum(PN_hat, 0)
    Y_hat[0:2, :] = 0
    PN_hat[0:2, :] = 0

    # Combining the estimated power spectrum with the original noisy phase,
    # and add the frames using an overlap-add technique
    output_Y = overlap_and_add(np.sqrt(Y_hat), YPhase,
                               (w + padz), shift_p * w)
    output_N = overlap_and_add(np.sqrt(PN_hat), Y_NPhase,
                               (w + padz), shift_p * w)
    return output_Y, output_N


def overlap_and_add(powers, phases, len_window, shift_size):
    """Reconstruct a signal with the overlap and add method

    :powers: array-like,
    :phases: array-like,
    :len_window: int, window length
    :shift_size: int, shift length. For non overlapping signals, in would
    equal len_frame. For 50% overlapping signals, it would be len_frame/2.
    :returns: array-like, reconstructed time-domain signal

    """
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
            += np.real(sp.fftpack.ifft(spectrum[i_frame], len_window))
    return signal
