import numpy as np
import scipy.stats
from typing import Union

from sklarpy.copulas._prefit_dicts import PreFitCopula
from sklarpy._other import Params

__all__ = ['student_t_copula_gen']


class student_t_copula_gen(PreFitCopula):
    def _g_to_u(self, g: np.ndarray, copula_params: Union[Params, tuple]) -> np.ndarray:
        return scipy.stats.t.cdf(g, df=copula_params[-1])

    def _u_to_g(self, u: np.ndarray, copula_params: Union[Params, tuple]) -> np.ndarray:
        return scipy.stats.t.ppf(u, df=copula_params[-1])

    def _h_logpdf_sum(self, g: np.ndarray, copula_params: Union[Params, tuple]) -> np.ndarray:
        return scipy.stats.t.logpdf(g, df=copula_params[-1]).sum(axis=1)
