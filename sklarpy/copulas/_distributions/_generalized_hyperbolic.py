# Contains code for the generalized hyperbolic copula model
import numpy as np
from typing import Union, Callable

from sklarpy.copulas._prefit_dicts import PreFitCopula
from sklarpy._other import Params
from sklarpy.univariate import gh

__all__ = ['gen_hyperbolic_copula_gen']


class gen_hyperbolic_copula_gen(PreFitCopula):
    """The Multivariate Generalized Hyperbolic copula model."""
    def _u_g_pdf(self, func: Callable, arr: np.ndarray,
                 copula_params: Union[Params, tuple], **kwargs) -> np.ndarray:
        shape: tuple = arr.shape
        output: np.ndarray = np.full(shape, np.nan)
        for i in range(shape[1]):
            gh_params: tuple = (
                copula_params[0], copula_params[1], copula_params[2],
                0.0, 1.0, float(copula_params[-1][i])
            )
            output[:, i] = func(arr[:, i], gh_params, **kwargs)
        return output

    def _g_to_u(self, g: np.ndarray, copula_params: Union[Params, tuple]) \
            -> np.ndarray:
        return self._u_g_pdf(func=gh.cdf_approx, arr=g,
                             copula_params=copula_params, num_points=10)

    def _u_to_g(self, u: np.ndarray, copula_params: Union[Params, tuple]):
        return self._u_g_pdf(func=gh.ppf_approx, arr=u,
                             copula_params=copula_params, num_points=10)

    def _h_logpdf_sum(self, g: np.ndarray, copula_params: Union[Params, tuple]
                      ) -> np.ndarray:
        pdf_vals: np.ndarray = self._u_g_pdf(func=gh.pdf, arr=g,
                                             copula_params=copula_params)
        return np.log(pdf_vals).sum(axis=1)
