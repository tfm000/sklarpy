# Standard parametrization of the Generalized Hyperbolic distribution
import numpy as np
import scipy.optimize
import scipy.integrate

from sklarpy.misc import kv
from sklarpy._utils import check_params, FitError

__all__ = ['_gh']


class gh_gen:
    def _argcheck(self, params) -> None:
        check_params(params)
        num_params: int = len(params)
        if num_params != 6:
            raise ValueError(f"expected 6 parameters, but {num_params} given.")
        if not (params[1] > 0) and (params[2] > 0) and (params[-2] > 0):
            raise ValueError(f"chi, psi and scale parameters must all be strictly positive.")

    def _logpdf_single(self, xi: float, lamb: float, chi: float, psi: float, loc: float, scale: float, skew: float) -> float:
        q: float = chi + (((xi - loc)/scale) ** 2)
        p: float = psi + ((skew/scale) ** 2)
        r: float = np.sqrt(chi * psi)
        s: float = 0.5 - lamb

        log_c: float = (lamb * (np.log(psi) - np.log(r))) + (s * np.log(p)) - (0.5 * np.log(2*np.pi)) - np.log(scale) - kv.logkv(lamb, r)
        log_h: float = ((xi - loc) * skew * (scale ** - 2)) + kv.logkv(-s, np.sqrt(p*q)) - 0.5 * s * (np.log(p) + np.log(q))
        return log_c + log_h

    def logpdf(self, x, *params):
        self._argcheck(params)
        return np.vectorize(self._logpdf_single, otypes=[float])(x, *params)

    def pdf(self, x, *params):
        return np.exp(self.logpdf(x, *params))

    def _pdf_single(self, xi: float, *params) -> float:
        return np.exp(self._logpdf_single(xi, *params))

    def _cdf_single(self, xi: float, *params) -> float:
        return float(scipy.integrate.quad(self._pdf_single, -np.inf, xi, params)[0])

    def cdf(self, x, *params):
        self._argcheck(params)
        return np.vectorize(self._cdf_single, otypes=[float])(x, *params)

    def support(self, *params):
        return -np.inf, np.inf

    def _ppf_single(self, qi: float, *params):
        def to_solve(xi):
            return self._cdf_single(xi, *params) - qi
        res = scipy.optimize.root(to_solve, params[3])
        return float(res['x']) if res['success'] else np.nan

    def ppf(self, q, *params):
        return np.vectorize(self._ppf_single, otypes=[float])(q, *params)

    def fit(self, data: np.ndarray) -> tuple:
        def neg_loglikelihood(params: np.ndarray):
            return -np.sum(self.logpdf(data, *params))

        xmin, xmax = data.min(), data.max()
        xextreme = max(abs(xmin), abs(xmax))
        eps: float = 10**-5
        bounds: tuple = ((-10, 10), (eps, 10), (eps, 10), (xmin, xmax), (eps, 2*(xmax-xmin)), (-xextreme, xextreme))
        res = scipy.optimize.differential_evolution(neg_loglikelihood, bounds)
        if not res['success']:
            raise FitError("Unable to fit Generalized Hyperbolic Distribution to data.")
        return tuple(res['x'])


_gh: gh_gen = gh_gen()
