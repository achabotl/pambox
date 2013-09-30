from numpy import pi, exp, cos
import numpy as np
import scipy as sp
import scipy.signal as ss
import filterbank


MIDFREQ = np.array([63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
                   1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000])
FS = np.array([22050.])


def erbbw(fc):
    """Auditory filter bandwith

    :fc: @todo
    :returns: @todo

    """
    # In Hz, according to Glasberg and Moore (1990)
    return 24.7 + fc / 9.265


def gammatone_filtering(signal, center_f=MIDFREQ, fs=FS):
    """Filters a signal using a gammatone filterbank

    :signal: @todo
    :returns: @todo

    """
    b, a, _, _, _ = gammatone_make(fs, center_f)
    return gammatone_apply(signal, b, a)


def lowpass_env_filtering(x, cutoff=150., N=1, fs=FS):
    """Low-pass filters signal

    :x: @todo
    :cutoff: @todo
    :returns: @todo

    """
    b, a = sp.signal.butter(N=N, Wn=cutoff * 2. / fs, btype='lowpass')
    return sp.signal.lfilter(b, a, x)


def gammatone_make(fs, cf, beta=1.019):
    '''
    GammaToneMake(fs,cf)

    Input:
        fs ... float, sampling frequency
        cf ... ndarray, center frequencies

    Output:
        forward ... "b"-coefficients for the linear filter
        feedback ... "a"-coefficients for the linear filter
        cf ... center frequency
        ERB ... Equivalent Rectangular Bandwidth
        B ... Gammatone filter parameter in Roy Patterson's ear model

    Computes the filter coefficients for a bank of Gammatone filters. The
    results are returned as arrays of filter coefficients. Each row of the
    filter arrays (forward and feedback) can be passed to the SciPy "lfilter"
    function.
    '''
    T = 1 / float(fs)
    pi = np.pi
    ERB = 24.7 + cf / 9.265  # In Hz, according to Glasberg and Moore (1990)
    # B = 1.019 * 2 * pi * ERB    # in rad here. Note: some models require B in Hz (NC)
    B = beta * 2 * pi * ERB    # in rad here. Note: some models require B in Hz (NC)

    gain = abs( \
     (-2*np.exp(4*1j*cf*pi*T)*T + 2*np.exp(-(B*T) + 2*1j*cf*pi*T) * T * (np.cos(2*cf*pi*T) - np.sqrt(3 - 2**(3./2))*np.sin(2*cf*pi*T))) \
    * (-2*np.exp(4*1j*cf*pi*T)*T + 2*np.exp(-(B*T) + 2*1j*cf*pi*T) * T * (np.cos(2*cf*pi*T) + np.sqrt(3 - 2**(3./2))*np.sin(2*cf*pi*T))) \
    * (-2*np.exp(4*1j*cf*pi*T)*T + 2*np.exp(-(B*T) + 2*1j*cf*pi*T) * T * (np.cos(2*cf*pi*T) - np.sqrt(3 + 2**(3./2))*np.sin(2*cf*pi*T))) \
    * (-2*np.exp(4*1j*cf*pi*T)*T + 2*np.exp(-(B*T) + 2*1j*cf*pi*T) * T * (np.cos(2*cf*pi*T) + np.sqrt(3 + 2**(3./2))*np.sin(2*cf*pi*T))) \
    / (-2 / np.exp(2*B*T) - 2*np.exp(4*1j*cf*pi*T) + 2*(1 + np.exp(4*1j*cf*pi*T))/np.exp(B*T))**4 )

    if np.isscalar(cf):
        len_cf = 1
    else:
        len_cf = len(cf)
    feedback = np.zeros((len_cf, 9))
    forward = np.zeros((len_cf, 5))

    forward[:, 0] =    T**4 / gain
    forward[:, 1] = -4*T**4 * np.cos(2*cf*pi*T) / np.exp(B*T)   / gain
    forward[:, 2] = 6 * T**4 * np.cos(4*cf*pi*T) / np.exp(2*B*T) / gain
    forward[:, 3] = -4*T**4 * np.cos(6*cf*pi*T) / np.exp(3*B*T) / gain
    forward[:, 4] =    T**4 * np.cos(8*cf*pi*T) / np.exp(4*B*T) / gain

    feedback[:, 0] = np.ones(len_cf)
    feedback[:, 1] = -8 * np.cos(2*cf*pi*T) / np.exp(B*T)
    feedback[:, 2] =  4 * (4 + 3*np.cos(4*cf*pi*T)) / np.exp(2*B*T)
    feedback[:, 3] = -8 * (6*np.cos(2*cf*pi*T) + np.cos(6*cf*pi*T)) / np.exp(3*B*T)
    feedback[:, 4] =  2 * (18 + 16*np.cos(4*cf*pi*T) + np.cos(8*cf*pi*T)) / np.exp(4*B*T)
    feedback[:, 5] = -8 * (6*np.cos(2*cf*pi*T) + np.cos(6*cf*pi*T)) / np.exp(5*B*T)
    feedback[:, 6] =  4 * (4 + 3*np.cos(4*cf*pi*T)) / np.exp(6*B*T)
    feedback[:, 7] = -8 * np.cos(2*cf*pi*T) / np.exp(7*B*T)
    feedback[:, 8] = np.exp(-8*B*T)

    return (forward, feedback, cf, ERB, B)


def gammatone_apply(x, forward, feedback):
    '''
    This function filters the waveform x with the array of filters
    specified by the forward and feedback parameters. Each row
    of the forward and feedback parameters are the parameters
    to the SciPy function "lfilter".
    '''

    # Allocate the memory
    rows, _ = np.shape(feedback)
    y = np.zeros((rows, len(x)))

    # Filter the signal
    for ii in range(rows):
        y[ii,:] = ss.lfilter(forward[ii,:], feedback[ii,:], x)

    return y

