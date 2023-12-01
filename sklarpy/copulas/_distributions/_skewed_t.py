# Contains code for the skewed-t copula model
import numpy as np
from typing import Union, Callable

from sklarpy.copulas._distributions._generalized_hyperbolic import \
    gen_hyperbolic_copula_gen
from sklarpy.utils._params import Params
from sklarpy.univariate.distributions import skewed_t

__all__ = ['skewed_t_copula_gen']


class skewed_t_copula_gen(gen_hyperbolic_copula_gen):
    """The Multivariate Skewed-T copula model."""
    def _u_g_pdf(self, func: Callable, arr: np.ndarray,
                 copula_params: Union[Params, tuple], **kwargs) -> np.ndarray:
        shape: tuple = arr.shape
        output: np.ndarray = np.full(shape, np.nan)
        for i in range(shape[1]):
            skewed_t_params: tuple = (
                copula_params[0], 0.0, 1.0, float(copula_params[-1][i])
            )
            output[:, i] = func(arr[:, i], skewed_t_params, **kwargs)
        return output

    def _g_to_u(self, g: np.ndarray, copula_params: Union[Params, tuple]) \
            -> np.ndarray:
        return self._u_g_pdf(func=skewed_t.cdf_approx, arr=g,
                             copula_params=copula_params, num_points=10)

    def _u_to_g(self, u: np.ndarray, copula_params: Union[Params, tuple]):
        return self._u_g_pdf(func=skewed_t.ppf_approx, arr=u,
                             copula_params=copula_params, num_points=10)

    def _h_logpdf_sum(self, g: np.ndarray, copula_params: Union[Params, tuple]
                      ) -> np.ndarray:
        pdf_vals: np.ndarray = self._u_g_pdf(func=skewed_t.pdf, arr=g,
                                             copula_params=copula_params)
        return np.log(pdf_vals).sum(axis=1)
