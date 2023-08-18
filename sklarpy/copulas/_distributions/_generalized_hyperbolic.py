import numpy as np
from typing import Union

from sklarpy.copulas._prefit_dicts import PreFitCopula
from sklarpy._other import Params
from sklarpy.univariate import gh

__all__ = ['gen_hyperbolic_copula_gen']


class gen_hyperbolic_copula_gen(PreFitCopula):
    def _g_to_u(self, g: np.ndarray, copula_params: Union[Params, tuple]) -> np.ndarray:
        shape: tuple = g.shape
        u: np.ndarray = np.full(shape, np.nan)
        for i in range(shape[1]):
            gh_params: tuple = copula_params[0], copula_params[1], copula_params[2], 0.0, 1.0, float(copula_params[-1][i])
            u[:, i] = gh.cdf_approx(g[:, i], gh_params, num_points=10)
        return u

    def _u_to_g(self, u: np.ndarray, copula_params: Union[Params, tuple]):
        shape: tuple = u.shape
        g: np.ndarray = np.full(shape, np.nan)
        for i in range(shape[1]):
            gh_params: tuple = copula_params[0], copula_params[1], copula_params[2], 0.0, 1.0, float(copula_params[-1][i])
            g[:, i] = gh.ppf_approx(u[:, i], gh_params, num_points=10)
        return g

    def _h_logpdf_sum(self, g: np.ndarray, copula_params: Union[Params, tuple]) -> np.ndarray:
        shape: tuple = g.shape
        pdf_vals: np.ndarray = np.full(shape, np.nan)
        for i in range(shape[1]):
            gh_params: tuple = copula_params[0], copula_params[1], copula_params[2], 0.0, 1.0, float(copula_params[-1][i])
            pdf_vals[:, i] = gh.pdf(g[:, i], gh_params)
        return np.log(pdf_vals).sum(axis=1)
