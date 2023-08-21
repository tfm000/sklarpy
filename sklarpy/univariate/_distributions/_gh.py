# Standard parametrization of the Generalized Hyperbolic distribution
import numpy as np

from sklarpy.misc import kv
from sklarpy.univariate._distributions._base_gen import base_gen

__all__ = ['_gh']


class gh_gen(base_gen):
    _NAME = 'Generalized Hyperbolic'
    _NUM_PARAMS = 6

    def _argcheck(self, params) -> None:
        super()._argcheck(params)

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

    def support(self, *params):
        return -np.inf, np.inf

    def get_default_bounds(self, data: np.ndarray, eps: float = 10 ** -5) -> tuple:
        xmin, xmax = data.min(), data.max()
        xextreme = max(abs(xmin), abs(xmax))
        return ((-10, 10), (eps, 10), (eps, 10), (xmin, xmax), (eps, 2 * (xmax - xmin)), (-xextreme, xextreme))


_gh: gh_gen = gh_gen()
