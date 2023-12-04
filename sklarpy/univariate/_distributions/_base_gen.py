# Contains a base class for SklarPy univariate probability models.
import numpy as np
import scipy.optimize
import scipy.integrate

from sklarpy.utils._errors import FitError
from sklarpy.utils._input_handlers import check_params

__all__ = ['base_gen']


class base_gen:
    """Base class for SklarPy univariate probability models."""
    _NAME: str
    _NUM_PARAMS: int

    def _argcheck(self, params) -> None:
        """Checks parameters and raises an error if required.

        Parameters
        ----------
        params: tuple
            The parameters which define the univariate model.
        """
        check_params(params)

        num_params: int = len(params)
        if num_params != self._NUM_PARAMS:
            raise ValueError(f"expected {self._NUM_PARAMS} parameters, but "
                             f"{num_params} given.")

    def _logpdf_single(self, xi: float, *params) -> float:
        """Returns the log-pdf value for a single observation.

        Parameters
        ----------
        xi: float
            A single observation.
        params: tuple
            The parameters which define the univariate model.

        Returns
        -------
        single_logpdf : float
            The log-pdf value for a single observation.
        """

    def logpdf(self, x, *params) -> np.ndarray:
        """The logarithm of the probability density/mass function.

        Parameters
        ----------
        x: Union[float, int, np.ndarray]
            The value/values to calculate the logpdf values of.
        params: tuple
            The parameters which define the univariate model.

        Returns
        -------
        logpdf_values: np.ndarray
            An array of logpdf values
        """
        self._argcheck(params)
        return np.vectorize(self._logpdf_single, otypes=[float])(x, *params)

    def pdf(self, x, *params) -> np.ndarray:
        """The probability density/mass function.

        Parameters
        ----------
        x: Union[float, int, np.ndarray]
            The value/values to calculate the pdf/pmf values of.
        params: tuple
            The parameters which define the univariate model.

        Returns
        -------
        pdf_values: np.ndarray
            An array of pdf values
        """
        return np.exp(self.logpdf(x, *params))

    def _pdf_single(self, xi: float, *params) -> float:
        """Returns the pdf value for a single observation.

        Used when computing the cdf via numerical integreation.
        """
        return np.exp(self._logpdf_single(xi, *params))

    def _cdf_single(self, xi: float, *params) -> float:
        """Returns the cdf value for a single observation.

        Parameters
        ----------
        xi: float
            A single observation.
        params: tuple
            The parameters which define the univariate model.

        Returns
        -------
        single_cdf : float
            The cdf value for a single observation.
        """
        left: float = self.support(params)[0]
        if xi <= left:
            return 0
        return float(
            scipy.integrate.quad(self._pdf_single, left, xi, params)[0]
        )

    def cdf(self, x, *params) -> np.ndarray:
        """The cumulative distribution function.

        Parameters
        ---------
        x: Union[float, int, np.ndarray]
            The value/values to calculate the cdf values, P(X<=x) of.
        params: tuple
            The parameters which define the univariate model.

        Returns
        -------
        cdf_values: np.ndarray
            An array of cdf values
        """
        self._argcheck(params)
        return np.vectorize(self._cdf_single, otypes=[float])(x, *params)

    def support(self, *params) -> tuple:
        """The support function of the distribution.

        Parameters
        ----------
        params: tuple
            The parameters which define the univariate model.

        Returns
        --------
        support: tuple
            The support of the specified distribution.
        """

    def _ppf_to_solve(self, xi, qi, *params) -> float:
        """Used to calculate the ppf function value using numerical
        optimization."""
        return self._cdf_single(xi, *params) - qi

    def _ppf_single(self, qi: float, *params) -> float:
        """Returns the ppf value for a single cdf observation."""
        # Code adapted from scipy code
        factor = 2.
        left, right = self.support(*params)

        if np.isinf(left):
            left = min(-factor, right)
            while self._ppf_to_solve(left, qi, *params) > 0.:
                left, right = left * factor, left
            # left is now such that cdf(left) <= q
            # if right has changed, then cdf(right) > q

        if np.isinf(right):
            right = max(factor, left)
            while self._ppf_to_solve(right, qi, *params) < 0.:
                left, right = right, right * factor
            # right is now such that cdf(right) >= q

        return scipy.optimize.brentq(self._ppf_to_solve, left, right,
                                     args=(qi, *params))

    def ppf(self, q, *params) -> np.ndarray:
        """The cumulative inverse / quartile function.

        Parameters
        ----------
        q: Union[float, int, np.ndarray]
            The quartile values to calculate cdf^-1(q) of.
        params: tuple
            The parameters which define the univariate model.

        Returns
        -------
        ppf_values: np.ndarray
            An array of quantile values.
        """
        return np.vectorize(self._ppf_single, otypes=[float])(q, *params)

    def _get_default_bounds(self, data: np.ndarray, *args) -> tuple:
        pass

    def _get_additional_args(self, data: np.ndarray) -> tuple:
        return tuple()

    def _theta_to_params(self, theta: np.ndarray, *args) -> tuple:
        return tuple(theta)

    def _neg_loglikelihood(self, theta: np.ndarray, data: np.ndarray, *args
                           ) -> float:
        params: tuple = self._theta_to_params(theta, *args)
        if params is None:
            return np.inf
        return -np.sum(self.logpdf(data, *params))

    def fit(self, data: np.ndarray) -> tuple:
        """Used to fit the distribution to the data.

        Parameters
        -----------
        data : data_iterable
            The data to fit to the distribution too.

        Returns
        -------
        params: tuple
            The parameters of the distribution, optimized to fit the data using
            MLE.
        """
        bounds: tuple = self._get_default_bounds(data=data)
        args: tuple = self._get_additional_args(data)
        res = scipy.optimize.differential_evolution(
            self._neg_loglikelihood, bounds, args=(data, *args))

        if not res['success']:
            raise FitError(f"Unable to fit {self._NAME} Distribution to data.")
        return self._theta_to_params(res['x'], *args)
