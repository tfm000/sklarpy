# Standard parametrization of the Skewed T distribution
# Note there are convergence problems in the fit
import numpy as np
import scipy.special

from sklarpy.misc import kv
from sklarpy.univariate._distributions._base_gen import base_gen
from sklarpy.univariate._distributions._gh import gh_gen

__all__ = ['_skewed_t']


class skewed_t_gen(gh_gen):
    """The univariate Skewed-T distribution, with the parametrization specified
    by McNeil et al."""
    _NAME = 'Skewed T'
    _NUM_PARAMS = 4

    def _argcheck(self, params) -> None:
        base_gen._argcheck(self, params)

        if not (params[0] > 0) and (params[2] > 0):
            raise ValueError("dof and scale parameters must be strictly "
                             "positive.")

    def _logpdf_single(self, xi: float, dof: float, loc: float, scale: float,
                       skew: float) -> float:
        q: float = dof + (((xi - loc) * scale) ** 2)
        p: float = (skew / scale) ** 2
        s: float = 0.5 * (1 + dof)
        m: float = np.sqrt(q * p)

        log_c: float = float(
            ((1 - s) * np.log(2))
            - scipy.special.loggamma(dof / 2)
            - 0.5 * np.log(np.pi * dof * (scale ** 2))
        )
        log_h: float = float(
            ((xi - loc) * skew * (scale ** -2))
            + kv.logkv(s, m)
            - s * (np.log(q / dof) - np.log(m))
        )
        return log_c + log_h

    @staticmethod
    def _exp_w(params: tuple) -> float:
        alpha_beta: float = params[1] / 2
        return alpha_beta / (alpha_beta - 1)

    @staticmethod
    def _var_w(params: tuple) -> float:
        alpha_beta: float = params[1] / 2
        return (alpha_beta ** 2) / (((alpha_beta - 1) ** 2) * (alpha_beta - 2))

    def _get_default_bounds(self, data: np.ndarray, *args) -> tuple:
        xmin, xmax = data.min(), data.max()
        xextreme = float(max(abs(xmin), abs(xmax)))
        return (3, 10), (-xextreme, xextreme)

    def _theta_to_params(self, theta: np.ndarray, mean: np.ndarray, var: float
                         ) -> tuple:
        dof, gamma = theta
        gh_theta: np.ndarray = np.array([-0.5 * dof, dof, 0, gamma])
        gh_params = super()._theta_to_params(gh_theta, mean, var)

        if gh_params is None:
            return gh_params
        return gh_params[1], *gh_params[-3:]


_skewed_t: skewed_t_gen = skewed_t_gen()
