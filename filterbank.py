import numpy as np
import scipy as sp
import scipy.signal as ss
import matplotlib.pyplot as plt


def middlepad(f, L, axis=-1):
    """Symmetrically zero-extends or cuts a function
    Usage:  h=middlepad(f,L);
            h=middlepad(f,L,dim);
            h=middlepad(f,L,...);

    MIDDLEPAD(f,L) cuts or zero-extends f to length L by inserting
    zeros in the middle of the vector, or by cutting in the middle
    of the vector.

    If f is whole-point even, MIDDLEPAD(f,L) will also be whole-point
    even.

    MIDDLEPAD(f,L,dim) does the same along dimension dim.

    If f has even length, then f will not be purely zero-extended, but
    the last element will be repeated once and multiplied by 1/2.
    That is, the support of f will increase by one!

    Adding the flag 'wp' as the last argument will cut or extend whole point
    even functions.  Adding 'hp' will do the same for half point even
    functions.

    See also:  isevenfunction, fir2long, fftresample

    Url: http://ltfat.sourceforge.net/doc/fourier/middlepad.php

    :arg1: @todo
    :returns: @todo

    """
    pass

def postpad(x, L, C=0, axis=-1):
    """Pads or truncates a vector x to a specified length L.

    POSTPAD(x,L) will add zeros to the end of the vector x, until the
    result has length L. If L is less than the length of the signal, it
    will be truncated. POSTPAD works along the first non-singleton
    dimension.

    POSTPAD(x,L,C) will add entries with a value of C instead of zeros.

    POSTPAD(x,L,C,dim) works along dimension dim instead of the first
    non-singleton.

    See also: middlepad

    Url: http://ltfat.sourceforge.net/doc/mex/postpad.php

    :x: @todo
    :L: @todo
    :returns: @todo

    """
    Ls = x.shape[0]
    W = 0  # FIXME
    if Ls < L:
        x = np.concatenate(x, C * np.ones(L - Ls, W))
    else:
        x = x[:L, :]
    return x


def gcd(a, b):
    """Return greatest common divisor using Euclid's Algorithm."""
    while b:
        a, b = b, a % b
    return a


def lcm(a, b):
    """Return lowest common multiple."""
    return a * b // gcd(a, b)


def _filterbank_length(Ls, a):
    """Filterbank length from signal
    """
    lcm_a = a[[0]]
    for m in range(1, a.shape[0]):
        lcm_a = lcm(lcm_a, a[m])
    return np.ceil(Ls / lcm_a) * lcm_a


def fir2long(gin, Llong):
    """FIR2LONG   Extend FIR window to LONG

     Usage:  g=fir2long(g,Llong);

    FIR2LONG(g,Llong) will extend the FIR window g to a length Llong*
    window by inserting zeros. Note that this is a slightly different
    behaviour than MIDDLEPAD.

    FIR2LONG can also be used to extend a FIR window to a longer FIR
    window, for instance in order to satisfy the usual requirement that the
    window length should be divisible by the number of channels.

    If the input to FIR2LONG is a cell, `fir2long` will recurse into
    the cell array.

    See also:  long2fir, middlepad

    Url: http://ltfat.sourceforge.net/doc/fourier/fir2long.php

    :arg1: @todo
    :returns: @todo

    """
    Lfir = gin.shape[-1]
    if Lfir > Llong:
        # Raise error 'Llong must be larger than length of window.'
        pass

    if Lfir // 2 == 0:
        return middlepad(gin, Llong, 'hp')
    else:
        return middlepad(gin, Llong)


def comp_ufilterbank_fft(f, g, a):
    """@todo: Docstring for comp_ufilterbank_fft.

    :f: @todo
    :gw: @todo
    :a): @todo
    :returns: @todo

    """
    L = f.shape[0]
    W = f.shape[1]
    M = g.shape[1]

    N = L / a

    c = np.empty((N, M, W))

    G = sp.fft.fft(fir2long(g, L))

    for w in np.arange(W):
        F = fft(f[:, w])
        for m in np.arange(M):
            c[:, m, w] = ifft(np.sum(reshape(F, G[:, m], N, a), 2)) / a

    if np.isreal(f) and np.isreal(g):
        c = np.real(c)
    return c


def ufilterbank(f, g, a=np.array(1.)):
    """@todo: Docstring for ufilterbank.

    :arg1: @todo
    :returns: @todo

    """
    a = a[[0]]
    Ls = f.shape[0]
    L = _filterbank_length(Ls, a)
    g, M = filterbankwin(g, a, L, 'normal')

    N = L / a

    f = np.pad(f, (L,), 'constant', constant_value=(0))

    gw = np.zeros((L, M))
    for ii in range(M):
        gw[ii,:] = fir2long(g[ii], L)

    return comp_ufilterbank_fft(f, gw, a)


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
