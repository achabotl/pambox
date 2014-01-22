import numpy as np
import scipy as sp
import general


def st_tmr(signal, noise, fs):
    """Calculate spectro-temporal target-to-masker ratio

    Implemented as described in Schoenmaker and van de Par, 2013


    Schoenmaker, E and van de Par, S. "Auditory streaming in cocktail parties:
    Better-ear versus binaural processing," AIA-DAGA 2013, Merano.

    :signal: ndarray, mono or stereo speech
    :noise: @todo
    :fs: int, sampling frequency
    :returns: float, TMR distribution, in dB

    """
    def stft(x, wnd, framesz, len_overlap):
        frames = np.asarray([sp.fftpack.fft(wnd * x[hop:hop + framesz])
                             for hop in range(0, len(x) - framesz,
                                              len_overlap)])
        return frames

    len_win = int(0.023 * fs)  # 23 m windows
    len_overlap = int(0.5 * len_win)  # 50% overlap
    wnd = np.sqrt(np.hanning(len_win))  # square-root Hanning window

    # Split in overlapping windowed frames.
    signal_frames = stft(signal, wnd, len_win, len_overlap)
    noise_frames = stft(noise, wnd, len_win, len_overlap)

    # Calculate TMR
    tmr = 20 * np.log10(np.abs(signal_frames) / np.abs(noise_frames))
    return tmr


def long_term_tmr(signal, noise):
    signal_rms = general.rms(signal)
    noise_rms = general.rms(noise)
    return 10 * np.log10(signal_rms / noise_rms)
