from __future__ import division
from scipy.stats import norm
from scipy.optimize import leastsq
from scipy.special import erfc
from numpy import sqrt
from numpy import arange
from numpy import linspace


class IdealObs(object):

    """Docstring for IdealObs. """

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
        return {'k': self.k, 'q': self.q, 'sigma_s': self.sigma_s, 'm': self.m}

    def fit_obs(self, snrenv, pcdata):
        """Find the optimal parameters for the ideal observer.
        snrenv: the linear SNRenv values that are to be converted to percent
            correct.
        pcdata: the data, in percentage of correctly understood tokens.
        """
        errfc = lambda p, snr, data: self._snrenv_to_pc(snrenv, p[0], p[1],
                                                        p[2], self.m) - data
        p0 = [self.k, self.q, self.sigma_s]
        res = leastsq(errfc, p0, args=(snrenv, pcdata))[0]
        self.k, self.q, self.sigma_s = res
        return self

    def _snrenv_to_pc(self, snrenv, k, q, sigma_s, m):
        """Convert SNR_env values to a percent correct using a ideal observer

        Uses specified parameters of the ideal observer so that we can do the
        parameter optimization.

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
        """Convert SNR_env values to a percent correct using a ideal observer


        Parameters
        ---------
        snrenv : array_like
            linear values of SNRenv

        Returns
        -------
        pc : ndarray
            Percentage of correctly understood speech units.

        Reference: Jorgensen and Dau (2011)
        """
        return self._snrenv_to_pc(snrenv, self.k, self.q, self.sigma_s, self.m)


def psy_fn(x, mu=0, sigma=1):
    """Psychometric function
    Convert x values to percent correct
    Inputs:
    x: the input array
    mu: mean of the distribution (default 0)
    sigma: standard deviation of the distribution (default 1)

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
