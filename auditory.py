from numpy import pi, exp, cos
import numpy as np
import scipy as sp
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


def gammatonefir(fc, fs, n=5000, betamul=1.0183,
                 ftype='complex', phase='causal'):
    """Gammatone filter coefficients

       Usage: b = gammatonefir(fc,fs,n,betamul);
              b = gammatonefir(fc,fs,n);
              b = gammatonefir(fc,fs);

       Input parameters:
          fc    :  center frequency in Hz.
          fs    :  sampling rate in Hz.
          n     :  filter order.
          beta  :  bandwidth of the filter.
          ftype :  outpout, 'real' or 'complex' (default)
          phase :  'peakphase' or 'causal' (default)

       Output parameters:
          b     :  FIR filters as columns

       GAMMATONEFIR(fc,fs,n,betamul) computes the filter coefficients of a
       digital FIR gammatone filter of length n with center frequency fc,
       4th order rising slope, sampling rate fs and bandwith determined by
       betamul. The bandwidth beta of each filter is determined as betamul*
       times AUDFILTBW of the center frequency of corresponding filter.

       GAMMATONEFIR(fc,fs,n) will do the same but choose a filter bandwidth
       according to Glasberg and Moore (1990). betamul is choosen to be
       1.0183.

       GAMMATONEFIR(fc,fs) will do as above and choose a sufficiently long
       filter to accurately represent the lowest subband channel.

       If fc is a vector, each entry of fc is considered as one center
       frequency, and the corresponding coefficients are returned as column
       vectors in the output.

       The inpulse response of the gammatone filter is given by

           g(t) = a*t^(n-1)*cos(2*pi*fc*t)*exp(-2*pi*beta*t)


       The gammatone filters as implemented by this function generate
       complex valued output, because the filters are modulated by the
       exponential function. Using real on the output will give the
       coefficients of the corresponding cosine modulated filters.

       References:
         A. Aertsen and P. Johannesma. Spectro-temporal receptive fields of
         auditory neurons in the grassfrog. I. Characterization of tonal and
         natural stimuli. Biol. Cybern, 38:223-234, 1980.

         B. R. Glasberg and B. Moore. Derivation of auditory filter shapes from
         notched-noise data. Hearing Research, 47(1-2):103, 1990.

    """

    nchannels = fc.size
    beta = betamul * erbbw(fc)

    # FIXME: Calculate a good value for n, probably from the filters frequency

    # Initialize memtory
    out = np.empty((nchannels, n), dtype='complex')

    for ii in range(nchannels):
        delay = 3 / (2 * pi * beta[ii])
        scalconst = 2 * (2 * pi * beta[
                         ii]) ** 4 / sp.misc.factorial(4 - 1) / fs

        nfirst = np.ceil(fs * delay)

        if nfirst > n / 2:
            pass
            # error(['%s: The desired filter length is too short to accomodate
            # the ' ... #'beginning of the filter. Please choose a filter
            # length of ' ... #'at least %i samples.'], upper(mfilename),
            # nfirst *2)

        nlast = n / 2

        t = np.hstack((np.arange(nlast) / fs + delay,
                       np.arange(nfirst) / fs - nfirst / fs + delay))

        # g(t) = a*t ^ (n-1)*cos(2*pi*fc*t)*exp(-2*pi*beta*t)
        if ftype is 'real':
            bwork = scalconst * t ** (4 - 1) * cos(2 * pi * fc[ii] * t) \
                * exp(-2 * pi * beta[ii] * t)
        else:  # Is complex
            bwork = scalconst * t ** (4 - 1) * exp(2 * pi * 1j * fc[ii] * t) \
                * exp(-2 * pi * beta[ii] * t)

        if phase is 'peakphase':
            bwork = bwork * exp(-2 * pi * 1j * fc[ii] * delay)

        # Insert zeros before the start of the signal.
        out[ii] = np.hstack((bwork[0:nlast],
                             np.zeros((n - nlast - nfirst)),
                             bwork[nlast:nlast + nfirst]))

    return out


def gammatone_filtering(signal, center_f=MIDFREQ, fs=FS):
    """Filters a signal using a gammatone filterbank

    :signal: @todo
    :returns: @todo

    """
    b = gammatonefir(center_f, fs)
    return filterbank.ufilterbank(signal, b)


def lowpass_env_filtering(x, cutoff=150., N=1, fs=FS):
    """Low-pass filters signal

    :x: @todo
    :cutoff: @todo
    :returns: @todo

    """
    b, a = sp.signal.butter(N=N, Wn=cutoff * 2. / fs, btype='lowpass')
    return sp.signal.lfilter(b, a, x)
