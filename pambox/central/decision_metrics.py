import logging

import numpy as np
from scipy.optimize import leastsq
from scipy.stats import norm


log = logging.getLogger(__name__)


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

    Notes
    -----
    Implemented as described in [jorgensen2011]_.

    Examples
    --------

    Converting values to percent correct using the default parameters
    of the ideal observer:

    >>> from pambox import central
    >>> obs = central.IdealObs()
    >>> obs.transform((0, 1, 2, 3))

    """
    def __init__(self, k=np.sqrt(1.2), q=0.5, sigma_s=0.6, m=8000.):
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

    def fit_obs(self, values, pcdata, sigma_s=None, m=None, tries=10):
        """Finds the parameters of the ideal observer.

        Finds the paramaters `k`, `q`, and `sigma_s`, that minimize the
        least-square error between a data set and transformed SNRenv.

        By default the `m` parameter is fixed and the property `m` is used.
        It can also be defined as an optional parameter.

        It is also possible to fix the `sigma_s` parameter by passing it as
        an optional argument. Otherwise, it is optimized with `k` and `q`.

        Parameters
        ----------
        values : ndarray
            The linear SNRenv values that are to be converted to percent
            correct.
        pcdata : ndarray
            The data, in percentage between 0 and 1, of correctly understood
            tokens. Must be the same shape as `values`.
        sigma_s : float, optional
             (Default value = None)
        m : float, optional
             (Default value = None)
        tries : int, optional
            How many attempts to fit the observer if the start values do not
            converge. The default is 10 times.

        Returns
        -------
        self

        """

        values = np.asarray(values)
        pcdata = np.asarray(pcdata)

        if m is None:
            m = self.m
        else:
            self.m = m

        # Set default values for optimization
        p0 = [self.k, self.q, self.sigma_s]
        fixed_params = {'m': m}
        if sigma_s is not None:
            p0 = p0[:2]
            fixed_params['sigma_s'] = sigma_s

        # Reshape the array to have `N` predictions and define the cost
        # function to average over those predictions.
        if values.shape != pcdata.shape:
            values = values.reshape((-1, len(pcdata)))

            def errfc(p, fixed):
                return np.mean(self._transform(values, *p, **fixed), axis=0
                               ) - pcdata
        # They have the same shape, the array should not be averaged
        else:
            def errfc(p, fixed):
                return self._transform(values, *p, **fixed) - pcdata

        for try_id in range(tries):
            (x, _, _, errmsg, ier) = leastsq(errfc, p0, args=fixed_params,
                                             maxfev=10000, full_output=True)
            if ier in [1, 2, 3, 4]:
                break
            else:
                p0 = 2 * np.random.random_sample(len(p0))
                log.error("Optimal parameters not found: " + errmsg)

        if sigma_s:
            self.k, self.q = x
            self.sigma_s = sigma_s
        else:
            self.k, self.q, self.sigma_s = x
        return self

    @staticmethod
    def _transform(values, k=None, q=None, sigma_s=None, m=None):
        """Converts SNRenv values to percent correct using an ideal observer.

        Parameters
        ----------
        values : array_like
            linear values of SNRenv
        k : float
            k parameter (Default value = None)
        q : float
            q parameter (Default value = None)
        sigma_s : float
            sigma_s parameter (Default value = None)
        m : float
            m parameter, number of words in the vocabulary. (Default value =
            None)

        Returns
        -------
        pc : ndarray
            Array of intelligibility percentage values, of the same shape as
            `values`.

        """
        un = norm.ppf(1.0 - 1.0 / m)
        sn = 1.28255 / un
        un += 0.577 / un
        dp = k * values ** q
        return norm.cdf(dp, un, np.sqrt(sigma_s ** 2 + sn ** 2)) * 100

    def transform(self, values):
        """Converts inputs values to a percent correct.

        Parameters
        ----------
        values : array_like
            Linear values to transform.

        Returns
        -------
        pc : ndarray
            Array of intelligibility percentage values, of the same shape as
            `values`.

        """
        values = np.asarray(values)
        return self._transform(values, self.k, self.q, self.sigma_s, self.m)