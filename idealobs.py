from __future__ import division
from scipy.stats import norm
from scipy.optimize import leastsq
from scipy.special import erfc
from numpy import sqrt
from numpy import arange
from numpy import array
from numpy import linspace


# Defaults
K = sqrt(1.2)
Q = 0.5
SIGMA_S = 0.6
M = 8000.0


def snrenv_to_pc(snrenv, k=K, q=Q, sigma_s=SIGMA_S, m=M):
    """Convert SNR_env values to a percent correct using a ideal observer
    snrenv: linear values of SNRenv
    Reference: Jorgensen and Dau (2011)
    Author: Alexandre Chabot-Leclerc (alech AT elektro.dtu.dk)
    """
    Un = norm.ppf(1.0 - 1.0 / m)
    sn = 1.28255 / Un
    un = Un + 0.577 / Un
    dp = k * snrenv**q
    return norm.cdf(dp, un, sqrt(sigma_s**2 + sn**2)) * 100


def fit_obs(snrenv, pcdata):
    """Find the optimal parameters for the ideal observer.
    snrenv: the linear SNRenv values that are to be converted to percent
        correct.
    pcdata: the data, in percentage of correctly understood tokens.
    """
    errfc = lambda p, snr, data: snrenv_to_pc(snrenv, p[0], p[1], p[2], M) - data
    p0 = [K, Q, SIGMA_S]
    res = leastsq(errfc, p0, args=(snrenv, pcdata))
    return res


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
    data = psy_fn(snr, -3.1, 2.13) * 100
    popt = fit_obs(snrenv, data)

    figure
    plot(snr, data, 'b--', label='Data')
    plot(snr, snrenv_to_pc(snrenv, *popt[0], m=M), 'r--', label='Model')
    xlabel('SNR [dB]')
    ylabel('Percent correct')
    legend(loc='upper left')
    show()
    k, q, ss = popt[0]
    print(('Optimized parameters found:\nk = {:.3f}\nq = {:.3f}\n'
           'sigma_s = {:.3f}\nm = {:.3f}').format(k, q, ss, M))

