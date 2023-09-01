# Contains code for the gaussian copula model
import numpy as np
import scipy.stats
from typing import Union

from sklarpy.copulas._prefit_dists import PreFitCopula
from sklarpy._other import Params

__all__ = ['gaussian_copula_gen']


class gaussian_copula_gen(PreFitCopula):
    """The Multivariate Gaussian copula model."""
    def _g_to_u(self, g: np.ndarray, copula_params_tuple: Union[Params, tuple]
                ) -> np.ndarray:
        return scipy.stats.norm.cdf(x=g)

    def _u_to_g(self, u: np.ndarray, copula_params: Union[Params, tuple]) \
            -> np.ndarray:
        return scipy.stats.norm.ppf(q=u)

    def _h_logpdf_sum(self, g: np.ndarray, copula_params: Union[Params, tuple]
                      ) -> np.ndarray:
        return scipy.stats.norm.logpdf(g).sum(axis=1)
