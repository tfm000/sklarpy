# Standard parametrization of the Generalized Hyperbolic distribution
import numpy as np

from sklarpy.misc import kv
from sklarpy.univariate._distributions._base_gen import base_gen

__all__ = ['_gh']


class gh_gen(base_gen):
    """The univariate Generalized Hyperbolic distribution,
    with the parametrization specified by McNeil et al."""
    _NAME = 'Generalized Hyperbolic'
    _NUM_PARAMS = 6

    def _argcheck(self, params) -> None:
        super()._argcheck(params)

        if not (params[1] > 0) and (params[2] > 0) and (params[-2] > 0):
            raise ValueError("chi, psi and scale parameters must all be "
                             "strictly positive.")

    def _logpdf_single(self, xi: float, lamb: float, chi: float, psi: float,
                       loc: float, scale: float, skew: float) -> float:
        q: float = chi + (((xi - loc) / scale) ** 2)
        p: float = psi + ((skew / scale) ** 2)
        r: float = np.sqrt(chi * psi)
        s: float = 0.5 - lamb

        log_c: float = float(
            (lamb * (np.log(psi) - np.log(r)))
            + (s * np.log(p))
            - (0.5 * np.log(2 * np.pi))
            - np.log(scale)
            - kv.logkv(lamb, r)
        )
        log_h: float = float(
            ((xi - loc) * skew * (scale ** - 2))
            + kv.logkv(-s, np.sqrt(p * q))
            - 0.5 * s * (np.log(p) + np.log(q))
        )
        return log_c + log_h

    def support(self, *params):
        return -np.inf, np.inf

    def _get_default_bounds(self, data: np.ndarray, eps: float = 10 ** -5
                            ) -> tuple:
        xmin, xmax = data.min(), data.max()
        xextreme = float(max(abs(xmin), abs(xmax)))
        return (-10, 10), (eps, 10), (eps, 10), (-xextreme, xextreme)

    def _get_additional_args(self, data: np.ndarray) -> tuple:
        return data.mean(), data.var()

    @staticmethod
    def _exp_w_a(params: tuple, a: float) -> float:
        """Calculates one of the moments of the distribution W, E[W^a].

        Parameters
        ----------
        params : tuple
            The parameters which define the model, in tuple form.
        a: float
            The order of the moment.

        Returns
        -------
        exp_w_a : float
            E[W^a]
        """
        lamb, chi, psi = params[:3]
        r: float = np.sqrt(chi * psi)
        if r > 100:
            # tends to 1 as r -> inf
            bessel_val: float = 1.0
        else:
            bessel_val: float = kv.kv(lamb + a, r) / kv.kv(lamb, r)
        return ((chi / psi) ** (a / 2)) * bessel_val

    @staticmethod
    def _exp_w(params: tuple) -> float:
        """Calculates the expectation of the distribution W, E[W].

        Parameters
        ----------
        params : tuple
            The parameters which define the model, in tuple form.

        Returns
        -------
        exp_w : float
            E[W]
        """
        return gh_gen._exp_w_a(params, 1)

    @staticmethod
    def _var_w(params: tuple) -> float:
        """Calculates the variance of the distribution W, var(W).

        Parameters
        ----------
        params : tuple
            The parameters which define the model, in tuple form.

        Returns
        -------
        var_w: float
            var(w)
        """
        return gh_gen._exp_w_a(params, 2) - (gh_gen._exp_w_a(params, 1) ** 2)

    def _theta_to_params(self, theta: np.ndarray, mean: np.ndarray, var: float
                         ) -> tuple:
        exp_w: float = self._exp_w(theta)
        var_w: float = self._var_w(theta)

        gamma: float = theta[-1]
        loc: float = mean - (exp_w * gamma)
        scale_sq: float = (var - (var_w * (gamma ** 2))) / exp_w
        if scale_sq <= 0:
            return None

        scale: float = scale_sq ** - 0.5
        return *theta[:3], loc, scale, gamma


_gh: gh_gen = gh_gen()
