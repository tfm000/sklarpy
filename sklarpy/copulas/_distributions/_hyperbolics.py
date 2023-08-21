import numpy as np
from typing import Union, Callable

from sklarpy.copulas._distributions._generalized_hyperbolic import gen_hyperbolic_copula_gen
from sklarpy._other import Params

__all__ = ['marginal_hyperbolic_copula_gen', 'hyperbolic_copula_gen', 'nig_copula_gen']


class hyperbolic_copula_base_gen(gen_hyperbolic_copula_gen):
    def _get_lamb(self, copula_params: Union[Params, tuple]) -> float:
        pass

    def _u_g_pdf(self, func: Callable, arr: np.ndarray, copula_params: Union[Params, tuple], **kwargs) -> np.ndarray:
        lamb: float = self._get_lamb(copula_params=copula_params)
        copula_params: tuple = lamb, copula_params[0], copula_params[1], copula_params[2], copula_params[3], copula_params[4]
        return super()._u_g_pdf(func=func, arr=arr, copula_params=copula_params, **kwargs)


class marginal_hyperbolic_copula_gen(hyperbolic_copula_base_gen):
    def _get_lamb(self, copula_params: Union[Params, tuple]) -> float:
        return 1.0


class hyperbolic_copula_gen(hyperbolic_copula_base_gen):
    def _get_lamb(self, copula_params: Union[Params, tuple]) -> float:
        d: int = copula_params[2].size
        return 0.5 * (d + 1)


class nig_copula_gen(hyperbolic_copula_base_gen):
    def _get_lamb(self, copula_params: Union[Params, tuple]) -> float:
        return -0.5
