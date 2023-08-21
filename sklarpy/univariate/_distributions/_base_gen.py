import numpy as np
import scipy.optimize
import scipy.integrate

from sklarpy._utils import check_params, FitError

__all__ = ['base_gen']


class base_gen:
    _NAME: str
    _NUM_PARAMS: int

    def _argcheck(self, params) -> None:
        check_params(params)

        num_params: int = len(params)
        if num_params != self._NUM_PARAMS:
            raise ValueError(f"expected {self._NUM_PARAMS} parameters, but {num_params} given.")

    def _logpdf_single(self, xi: float, *params) -> float:
        pass

    def logpdf(self, x, *params):
        self._argcheck(params)
        return np.vectorize(self._logpdf_single, otypes=[float])(x, *params)

    def pdf(self, x, *params):
        return np.exp(self.logpdf(x, *params))

    def _pdf_single(self, xi: float, *params) -> float:
        return np.exp(self._logpdf_single(xi, *params))

    def _cdf_single(self, xi: float, *params) -> float:
        left: float = self.support(params)[0]
        return float(scipy.integrate.quad(self._pdf_single, left, xi, params)[0])

    def cdf(self, x, *params):
        self._argcheck(params)
        return np.vectorize(self._cdf_single, otypes=[float])(x, *params)

    def support(self, *params) -> tuple:
        pass

    def _ppf_to_solve(self, xi, qi, *params):
        return self._cdf_single(xi, *params) - qi

    def _ppf_single(self, qi: float, *params):
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

        return scipy.optimize.brentq(self._ppf_to_solve, left, right, args=(qi, *params))

    def ppf(self, q, *params):
        return np.vectorize(self._ppf_single, otypes=[float])(q, *params)

    def get_default_bounds(self, data: np.ndarray, *args) -> tuple:
        pass

    def fit(self, data: np.ndarray) -> tuple:
        def neg_loglikelihood(params: np.ndarray):
            return -np.sum(self.logpdf(data, *params))

        bounds: tuple = self.get_default_bounds(data=data)
        res = scipy.optimize.differential_evolution(neg_loglikelihood, bounds)
        if not res['success']:
            raise FitError(f"Unable to fit {self._NAME} Distribution to data.")
        return tuple(res['x'])
