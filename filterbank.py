import numpy as np
import scipy as sp
import scipy.signal as ss
import matplotlib.pyplot as plt


def mfreqz(b,a=1, fs=[]):
    """Plot the frequency and phase response

    :b:
    :a:

    From http://mpastell.com/2010/01/18/fir-with-scipy/
    """
    w,h = ss.freqz(b,a)
    h_dB = 20 * np.log10(abs(h))
    plt.subplot(211)
    if fs:
        f = sp.linspace(0, fs/2, len(w))
        plt.plot(f,h_dB)
    else:
        plt.plot(w/max(w),h_dB)
    plt.ylim(-150, 5)
    plt.ylabel('Magnitude (db)')
    plt.xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
    plt.title(r'Frequency response')
    plt.subplot(212)
    h_Phase = sp.unwrap(sp.arctan2(sp.imag(h), sp.real(h)))
    plt.plot(w/max(w),h_Phase)
    plt.ylabel('Phase (radians)')
    plt.xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
    plt.title(r'Phase response')
    plt.subplots_adjust(hspace=0.5)


def impz(b,a=1):
    """Plot step and impulse response

    :b:
    :a:

    From http://mpastell.com/2010/01/18/fir-with-scipy/
    """
    l = len(b)
    impulse = repeat(0.,l); impulse[0] =1.
    x = arange(0,l)
    response = sp.lfilter(b,a,impulse)
    plt.subplot(211)
    plt.stem(x, response)
    plt.ylabel('Amplitude')
    plt.xlabel(r'n (samples)')
    plt.title(r'Impulse response')
    plt.subplot(212)
    step = sp.cumsum(response)
    plt.stem(x, step)
    plt.ylabel('Amplitude')
    plt.xlabel(r'n (samples)')
    plt.title(r'Step response')
    plt.subplots_adjust(hspace=0.5)
