# -*- coding: utf-8 -*-
from __future__ import division
from scipy.stats import norm
from scipy.optimize import leastsq
from scipy.special import erfc
from numpy import sqrt
from numpy import arange
from numpy import linspace


class IdealObs(object):
    """Statistical ideal observer.

    Converts input values (usually SNRenv) to a percentage.

    Parameters
    ----------
    k : float, optional
         (Default value = sqrt(1.2)
    q : float, optional
        (Default value = 0.5)
    sigma_s : float, optional
        (Default value = 0.6)
    m : int, optional
        Number of words in the vocabulary. (Default value = 8000)

    Examples
    --------
    """

    def __init__(self, k=sqrt(1.2), q=0.5, sigma_s=0.6, m=8000.):
        """@todo: to be defined1.

        :k: @todo
        :sigma_s: @todo
        :q: @todo
        :m: @todo

        """
        self.k = k
        self.q = q
        self.sigma_s = sigma_s
        self.m = m

    def get_params(self):
        """Returns the parameters of the ideal observer as dict.

        Parameters
        ----------
        None

        Returns
        -------
        params : dict
            Dictionary of internal parameters of the ideal observer.
        """
        return {'k': self.k, 'q': self.q, 'sigma_s': self.sigma_s, 'm': self.m}

    def fit_obs(self, snrenv, pcdata, sigma_s=None, m=None):
        """Finds the parameters of the ideal observer.

        Finds the paramaters `k`, `q`, and `sigma_s`, that minimize the
        least-square error between a data set and transformed SNRenv.

        By default the `m` parameter is fixed and the property `m` is used.
        It can also be defined as an optional parameter.

        It is also possible to fix the `sigma_s` parameter by passing it as
        an optional argument. Otherwise, it is optimized with `k` and `q`.

        Parameters
        ----------
        snrenv : ndarray
            The linear SNRenv values that are to be converted to percent
            correct.
        pcdata : ndarray
            The data, in percentage between 0 and 1, of correctly understood
            tokens. Must be the same shape as `snrenv`.
        sigma_s : float, optional
             (Default value = None)
        m : float, optional
             (Default value = None)

        Returns
        -------
        self

        """
        errfc = lambda p, snr, data: self._snrenv_to_pc(snrenv, p[0], p[1],
                                                        p[2], self.m) - data
        p0 = [self.k, self.q, self.sigma_s]
        res = leastsq(errfc, p0, args=(snrenv, pcdata))[0]
        self.k, self.q, self.sigma_s = res
        return self

    @staticmethod
    def _snrenv_to_pc(snrenv, k=None, q=None, sigma_s=None, m=None):
        """Converts SNRenv values to percent correct using an ideal observer.

        Parameters
        ----------
        snrenv : array_like
            linear values of SNRenv

        Returns
        -------
        pc : array_like
            Percentage of correctly understood speech units.

        """
        Un = norm.ppf(1.0 - 1.0 / m)
        sn = 1.28255 / Un
        un = Un + 0.577 / Un
        dp = k * snrenv ** q
        return norm.cdf(dp, un, sqrt(sigma_s ** 2 + sn ** 2)) * 100

    def snrenv_to_pc(self, snrenv):
        """Converts SNRenv values to a percent correct.

        Parameters
        ---------
        snrenv : array_like
            linear values of SNRenv

        Returns
        -------
        pc : ndarray
            Percentage of correctly understood speech units.

        """
        return self._snrenv_to_pc(snrenv, self.k, self.q, self.sigma_s, self.m)


def psy_fn(x, mu=0., sigma=1.):
    """Calculates a psychometric function with a given mean and variance.

    Parameters
    ----------
    x : array_like
        "x" values of the psychometric functions.
    mu : float, optional
        Value at which the psychometric function reaches 50%, i.e. the mean
        of the distribution. (Default value = 0)
    sigma : float, optional
        Variance of the psychometric function. (Default value = 1)

    Returns
    -------
    pc : ndarray
        Array of "percent correct", between 0 and 100.

    Outputs:
    pc: array of values between 0 and 100
    """
    return erfc(-(x - mu) / (sqrt(2) * sigma)) / 2


if __name__ == '__main__':
    from pylab import plot, figure, show, xlabel, ylabel, legend
    snr = arange(-9, 3, 1)
    snrenv = 10 ** linspace(-2, 2, len(snr))
    c = IdealObs()
    data = psy_fn(snr, -3.1, 2.13) * 100
    c.fit_obs(snrenv, data)

    figure
    plot(snr, data, 'b--', label='Data')
    plot(snr, c.snrenv_to_pc(snrenv), 'r--', label='Model')
    xlabel('SNR [dB]')
    ylabel('Percent correct')
    legend(loc='upper left')
    show()

    params = c.get_params()
    print(('Optimized parameters found:\nk = {:.3f}\nq = {:.3f}\n'
           'sigma_s = {:.3f}\nm = {:.3f}').format(params['k'], params['q'],
                                                  params['sigma_s'],
                                                  params['m']))
