import numpy as np
import scipy.stats
from typing import Union

from sklarpy.copulas._prefit_dicts import PreFitCopula
from sklarpy._other import Params

__all__ = ['student_t_copula_gen']


class student_t_copula_gen(PreFitCopula):
    def _z_to_u(self, z: np.ndarray, copula_params: Union[Params, tuple]) -> np.ndarray:
        copula_params_tuple: tuple = self._mv_object._get_params(copula_params, check_params=False)
        return scipy.stats.t.cdf(z, df=copula_params_tuple[-1])

    def _u_to_z(self, u: np.ndarray, copula_params: Union[Params, tuple]) -> np.ndarray:
        copula_params_tuple: tuple = self._mv_object._get_params(copula_params, check_params=False)
        return scipy.stats.t.ppf(u, df=copula_params_tuple[-1])
