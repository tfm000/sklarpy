import numpy as np
import scipy.stats
from typing import Union

from sklarpy.copulas._prefit_dicts import PreFitCopula
from sklarpy._other import Params

__all__ = ['gaussian_copula_gen']


class gaussian_copula_gen(PreFitCopula):
    def _z_to_u(self, z: np.ndarray, copula_params: Union[Params, tuple]) -> np.ndarray:
        return scipy.stats.norm.cdf(x=z)

    def _u_to_z(self, u: np.ndarray, copula_params: Union[Params, tuple]) -> np.ndarray:
        return scipy.stats.norm.ppf(q=u)
