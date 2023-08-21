# Standard parametrization of the Skewed T distribution
import numpy as np
import scipy.special

from sklarpy.misc import kv
from sklarpy.univariate._distributions._base_gen import base_gen

__all__ = ['_skewed_t']


class skewed_t_gen(base_gen):
    _NAME = 'Skewed T'
    _NUM_PARAMS = 4

    def _argcheck(self, params) -> None:
        super()._argcheck(params)

        if not (params[0] > 0) and (params[2] > 0):
            raise ValueError(f"dof and scale parameters must be strictly positive.")

    def _logpdf_single(self, xi: float, dof: float, loc: float, scale: float, skew: float) -> float:
        q: float = dof + (((xi - loc) * scale) ** 2)
        p: float = (skew / scale) ** 2
        s: float = 0.5 * (1 + dof)
        m: float = np.sqrt(q * p)

        log_c: float = ((1-s) * np.log(2)) - scipy.special.loggamma(dof/2) - 0.5 * np.log(np.pi*dof*(scale**2))
        log_h: float = ((xi - loc) * skew * (scale ** -2)) + kv.logkv(s, m) - s * (np.log(q/dof) - np.log(m))
        return log_c + log_h

    def support(self, *params):
        return -np.inf, np.inf

    def get_default_bounds(self, data: np.ndarray, eps: float = 10 ** -5) -> tuple:
        xmin, xmax = data.min(), data.max()
        xextreme = max(abs(xmin), abs(xmax))
        return ((2.01, 10), (xmin, xmax), (eps, 2 * (xmax - xmin)), (-xextreme, xextreme))


_skewed_t: skewed_t_gen = skewed_t_gen()
