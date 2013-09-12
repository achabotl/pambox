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
