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

    if snr:
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
