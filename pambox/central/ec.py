from __future__ import division, absolute_import

import numpy as np
from numpy import (
    sqrt,
    minimum,
)
try:
    _ = np.use_fastnumpy  # Use Enthought MKL optimizations
    from numpy.fft import rfft, irfft, rfftfreq
except AttributeError:
    try:
        import mklfft  # MKL FFT optimizations from Continuum Analytics
        from numpy.fft import rfft, irfft, rfftfreq
    except ImportError:
        # Finally, just use Numpy's and Scipy's
        from numpy.fft import rfft, irfft, rfftfreq
from scipy.signal import fftconvolve


class EC(object):

    """Equalization-Cancellation process used by the STEC model [wan2014]_.

    The `equalize` method finds the optimal gains and delays that minimizes
    the energy of the cancelled signal.

    The `cancel` method uses the gains and delays found by the `equalize`
    method to "cancel" the two signals.

    The `jitter` method applies amplitude and time jitters to the input as a
    form of internal noise.

    Examples
    --------
    >>> ec = EC()
    >>> alphas, taus = ec.equalize(left, right, cf)
    >>> y = ec.cancel(left, right, alphas, taus)

    References
    ----------
    .. [wan2014] Wan, R., Durlach, N. I., and Colburn, H. S. (2014).
    "Application of a short-time version of the Equalization-Cancellation
    model to speech intelligibility experiments with speech maskers",
    The Journal of the Acoustical Society of America, 136(2), 768--776

    """

    def __init__(self, fs, win_len=None, overlap=0.5, sigma_e=0.25,
                 sigma_d=105e-6, padding_windows=10, fast_cancel=True):
        """Equalization--Cancellation process.

        Parameters
        -----------
        fs : int
            Sampling frequency of the EC process.
        win_len : float
            Duration of a window, in seconds, where to apply the EC process. If
            `None`, the EC process is applied to the whole signal. Defaults to
            `None`.
        overlap : float
            Overlap between windows, in fraction of window. Defaults to 0.5 (i.e.
            50%).
        sigma_e : float
            Mean value of the amplitude jitter in the EC process. Default is 0.25
            as reported by Durlach (1963).
        sigma_d : float
            Mean duration of the time jitter. Default is 105us, as reported by
            Durlach (1963).

        """
        self.fs = fs
        self.win_len = win_len
        self.overlap = overlap
        self.sigma_e = sigma_e
        self.sigma_d = sigma_d
        self.padding_windows = padding_windows
        self.fast_cancel = fast_cancel

    def equalize(self, left, right, cf):
        """Finds the optimal gains and delays that minimize the energy of the
        cancelled signals.

        Parameters
        ----------
        left, right : ndarrays
            Signals for which to find the optimal parameters. They can be 1D
            or 2D. If they are 2D, the signals are cancelled along the last
            dimension.
        cf : float or list of floats
            Center frequency of the channel at which the equalization takes
            place. If the inputs are multi-channel, then cf must be a list of
            center frequencies.

        Returns
        -------
        alphas : ndarray
            Optimal gains. The shape depends on the input signals and on the
            `win_len` and ``overlap`` attributes.
        taus : ndarrays
            Optimal delays in seconds. The shape depends on the input signals
            and on the ``win_len`` and `overlap` attributes.
        """

        left = np.asanyarray(left, dtype='float')
        right = np.asanyarray(right, dtype='float')
        if left.ndim > 2 or right.ndim > 2:
            raise ValueError("Input signals must have at most 2 dimensions.",
                             left.ndim, right.ndim)
        if left.shape != right.shape:
            raise ValueError("Both inputs must have must have the same shape.",
                             left.shape, right.shape)
        if left.ndim == 2:
            try:
                if len(cf) != left.shape[0]:
                    raise ValueError("cf must have as many values as there "
                                     "are channels in the inputs.")
            except TypeError:
                raise ValueError("cf must be iterable if there are more than "
                                 "one channel.")

        if left.ndim == 1 and right.ndim == 1:
            # Use the whole signal.
            alphas, taus = self._equalization(left, right, cf)
        else:  # the inputs are 2D
            alphas = []
            taus = []
            for i_chan, cf in enumerate(cf):
                chan_alphas, chan_taus = self._equalization(
                    left[i_chan], right[i_chan], cf)
                alphas.append(chan_alphas)
                taus.append(chan_taus)
            alphas = np.array(alphas)
            taus = np.asarray(taus)

        return alphas, taus

    def _equalization(self, left, right, cf):
        """Equalize two signals.

        Parameters
        ----------
        left, right: array
            Single dimension array for left and right signal.
        cf : float
            Center frequency at which the equalization takes place.

        Returns
        -------
        alphas : array
            Gains for each window.
        taus : array
            Time delays for each window, in seconds.

        Notes
        -----
        The window duration is set by the attribute ``win_len``, in seconds,
        and the overlap between windows by ``overlap``, in fraction (e.g. 0.5
        for 50 % overlap).
        """
        n = left.shape[-1]
        if self.win_len is None:
            win = n
            step = n
        else:
            win = int(self.win_len * self.fs)
            if self.overlap:
                step = int(win * self.overlap)
            else:
                step = win

        n_valid_windows = self._n_valid_windows(n, win, step)
        alphas = np.zeros(n_valid_windows)
        taus = np.zeros(n_valid_windows)

        for i_frame, hop in enumerate(range(0, n - win + 1, step)):
            a, tau = self._calculate_alpha_tau(left[hop:hop + win],
                                               right[hop:hop + win], cf=cf)
            alphas[i_frame] = a
            taus[i_frame] = tau
        return alphas, taus

    @staticmethod
    def _n_valid_windows(n_samples, win_len, step):
        """Calculate the number of valid windows, considering overlap.

        Parameters
        ----------
        n_samples : int
            Length of vector.
        win_len : int
            Window length, in samples.
        step : int
            Number of samples between frames, essentially overlap * window
            length.

        Returns
        -------
        n_windows : int
            Number of valid windows.

        """
        valid = np.maximum(n_samples - win_len, 0)
        n_windows = valid // step
        return n_windows + 1

    def _calculate_alpha_tau(self, left, right, cf):
        """Finds optimal parameters for the EC process.

        Performs equations (1) in Wan et al. (2014).

        Parameters
        ----------
        left, right : ndarray
        w : float
            Center frequency of the channel, in Hz.

        Returns
        -------
        a : float
            Level equalization parameter
        tau : float
            Delay, in seconds, that should be applied to `right` in order to
            get close too `left`. Could also be explained as the delay
            applied to `left`, with respect to `right`.

        """
        E_L = left.dot(left.T)
        E_R = right.dot(right.T)
        # Alpha parameter for level equalization
        alpha = sqrt(E_L / E_R)

        tau = self._find_tau(left, right, cf)
        return alpha, tau

    def _find_tau(self, left, right, cf):
        """ Returns the delay (in seconds) of the maximum of the cross-correlation
        of two signals.
        """
        left = np.asanyarray(left)
        right = np.asanyarray(right)

        left = left - np.mean(left)
        right = right - np.mean(right)

        if left.dot(left) == 0 or right.dot(right) == 0:
            return 0
        else:
            n_samples = left.shape[-1]
            # Cross correlation
            # It should be normalized, according to the definition, but
            # we only need the max value, so it is not necessary to compute it.
            rho = fftconvolve(left, right[::-1], 'full')

            # Eq 6, we have to find tau_0 in the range where |tau| < fs / cf_0
            # i.e. (pi / omega_0)
            max_delay_in_samples = minimum(
                np.floor(np.pi / (2 * np.pi * cf) * self.fs),
                n_samples // 2)
            # First we limit the range to -fs/cf_0 < tau < fs/cf_0...
            allowed_range = np.arange(-max_delay_in_samples,
                                      max_delay_in_samples + 1, dtype=int)
            # ... then we find where the maximum is that range.
            tau = allowed_range[rho[allowed_range + n_samples - 1].argmax()]
            return tau / self.fs

    def cancel(self, left, right, alpha, tau):
        """Cancel left and right signal using gains and delays.

        Parameters
        ----------
        left, right : array_like
            Signals for which to find the optimal parameters. They can be 1D
            or 2D. If they are 2D, the signals are cancelled along the last
            dimension.
        alpha : ndarray
            Optimal amplitude cancellation gains.
        tau : ndarray
            Optimal cancellation delays.

        Returns
        -------
        y : ndarray


        """
        left = np.asanyarray(left, dtype='float')
        right = np.asanyarray(right, dtype='float')
        alpha = np.asanyarray(alpha)
        tau = np.asanyarray(tau)

        if left.ndim > 2 or right.ndim > 2:
            raise ValueError("Input signals must have at most 2 dimensions.",
                             left.ndim, right.ndim)
        if left.shape != right.shape:
            raise ValueError("Both inputs must have must have the same shape.",
                             left.shape, right.shape)

        if left.ndim == 1 and right.ndim == 1:
            out = self._single_chan_cancel(left, right, alpha, tau)
        else:  # the inputs are 2D
            out = np.zeros_like(left)
            for i_chan, (chan_alpha, chan_tau) in enumerate(zip(alpha, tau)):
                out[i_chan, :] = self._single_chan_cancel(
                    left[i_chan],
                    right[i_chan],
                    chan_alpha,
                    chan_tau)
        return out

    def _single_chan_cancel(self, left, right, alphas, taus):
        """Equalize two signals.

        Parameters
        ----------
        left, right: ndarrays
            Single dimension array for left and right signal.
        alphas : ndarray
            Gains for each window.
        taus : ndarray
            Time delays for each window, in samples.

        Returns
        -------
        out : ndarray
            Cancelled signals.

        Notes
        -----
        The window duration is set by the attribute `win_len`, in samples,
        and the overlap between windows by `overlap`, in fraction (e.g. 0.5
        for 50 % overlap).

        """

        n = left.shape[-1]
        if self.win_len is None:
            win = n
            step = n
            # Make sure the alphas and taus are iterable.
            try:
                iter(alphas)
            except TypeError:
                alphas = (alphas,)
            try:
                iter(taus)
            except TypeError:
                taus = (taus,)
        else:
            win = int(self.win_len * self.fs)
            if self.overlap:
                step = int(win * self.overlap)
            else:
                step = win

        out = np.zeros_like(left)
        extra = self.padding_windows * win
        for i_frame, (a, tau, hop) in enumerate(
                zip(alphas, taus, range(0, n - win + 1, step))):
            if tau == 0:
                out[hop:hop + win] += 1 / sqrt(a) * left[hop:hop + win] \
                    - sqrt(a) * right[hop:hop + win]
            else:
                if self.fast_cancel:
                    # Shift only a section of the signal, instead of the its
                    # entirety. The "window" size is defined by the `padding_windows`
                    # parameter. The size of the original window is increased
                    #  by 2*padding_windows (one before, one after).
                    lower = np.maximum(hop - extra, 0)
                    if lower == 0:
                        new_hop = hop
                    else:
                        new_hop = extra
                    upper = np.minimum(hop + win + extra, n)
                    out[hop:hop + win] += (
                        1 / sqrt(a) * self._shift(left[lower:upper], -tau / 2)
                        - sqrt(a) * self._shift(right[lower:upper], tau / 2)
                        )[new_hop:new_hop + win]
                else:
                    out[hop:hop + win] += 1 / sqrt(a) \
                        * self._shift(left, -tau / 2)[hop:hop + win] \
                        - sqrt(a) * self._shift(right, tau / 2)[hop:hop + win]
        if self.overlap:
            out *= self.overlap
        return out

    def _shift(self, x, delay):
        """Shift signal according to a delay and pads with zeros.

        Parameters
        ----------
        x : array
            Signal.
        delay : int
            Delay in seconds. Positive values correspond to a delay in time,
            i.e. the signal "starts later". Negative values correspond to a
            signal starting "earlier".

        Returns
        -------
        out : ndarray
            Delayed signal
        """
        n = x.shape[-1]
        y = rfft(x)
        w = rfftfreq(n, 1 / self.fs) * 2 * np.pi
        y *= np.exp(-1j * w * delay)
        return np.real(irfft(y, n))

    def jitter(self, x, out=None):
        """Applies amplitude and time jitter to a signal.

        Parameters
        ----------
        x : array_like
            Input signal, will be casted to 'float'. It can be one or 2
            dimensional.
        out : None or array_like
            Define where to write the jitter signal. Defaults to `None`,
            i.e. creates a new array. Can be used to jitter an array "in
            place".

        Returns
        -------
        out : ndarray
            Jittered signal.

        Notes
        -----
        The amplitude jitters are taken from a normal Gaussian distribution
        with a mean of zero and a standard distribution of ``sigma_e``. The time
        jitters are taken from a normal Gaussian distribution with mean zero
        and standard distribution ``sigma_d`` in seconds. The default jitter
        values come from [durlach1963]_.

        References
        ----------
        .. [durlach1963] Durlach, N. I. (1963). "Equalization and
        Cancellation Theory of Binaural Masking-Level Differences", J. Acoust.
        Soc. Am., 35(), 1206--1218


        """

        x = np.asanyarray(x, dtype='float')

        epsilons, deltas = self.create_jitter(x)
        out = self.apply_jitter(x, epsilons, deltas, out=out)
        return out

    def create_jitter(self, x):
        """Create amplitude and time jitter for a signal.

        Parameters
        ----------
        x : ndarray
            Input signal.

        Returns
        -------
        alphas : ndarray of floats
            Amplitude jitters.
        deltas : ndarray of ints
            Jitter indices.

        """
        n_x = x.shape[-1]

        # Amplitude jitter
        a_jitter = self.sigma_e * np.random.randn(*x.shape)

        # Time jitter
        if x.ndim > 1:
            idx = np.tile(np.arange(n_x, dtype='float'), (x.shape[0], 1))
        else:
            idx = np.arange(n_x, dtype='float')
        t_jitter = self.sigma_d * self.fs * np.random.randn(*idx.shape)
        return a_jitter, t_jitter

    @staticmethod
    def apply_jitter(x, epsilons, deltas, out=None):
        """Apply jitter to a signal

        Parameters
        ----------
        x : ndarray
            Input signal.
        epsilons : ndarray of floats
            Amplitude jitter coefficients.
        deltas : ndarray of ints
            Time jitters, they have to be integers because they will be
            used as indices.
        out : array or None
            Array where to write the output. If None, which is the default,
            the function returns a new array.

        Returns
        -------
        out : ndarray
            Jittered signal.

        """

        n_cf = x.shape[0]
        n_x = x.shape[-1]
        if x.ndim > 1:
            chan_idx = np.tile(np.arange(n_cf)[np.newaxis].T, (1, n_x))
            idx = np.tile(np.arange(n_x, dtype='float'), (x.shape[0], 1))
        else:
            # Single channel
            chan_idx = Ellipsis
            idx = np.arange(n_x, dtype='float')

        # Apply the jitter to the idx.
        idx += deltas
        # Limit the indices to the length of the array
        idx = np.clip(idx, 0, n_x - 1, out=idx)
        idx = np.round(idx, out=idx).astype('int')

        # Create indices for channels, it's a n_cf x n_x array, where each row
        # is filled with the row number.
        # Same for the "ear" dimension
        # ear_idx = np.tile(np.arange(2)[np.newaxis].T, (n_cf, 1, n_x))

        if out is None:
            out = x * (1 - epsilons)
            out[..., :] = out[chan_idx, idx]
        else:
            x *= (1 - epsilons)
            x[..., :] = x[chan_idx, idx]
            out[...] = x
        return out
