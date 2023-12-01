# Contains code for the symmetric generalized hyperbolic copula model
import numpy as np
from typing import Union, Callable

from sklarpy.copulas._distributions._generalized_hyperbolic import \
    gen_hyperbolic_copula_gen
from sklarpy.utils._params import Params

__all__ = ['sym_gen_hyperbolic_copula_gen']


class sym_gen_hyperbolic_copula_gen(gen_hyperbolic_copula_gen):
    """The Multivariate Symmetric Generalized Hyperbolic copula model."""
    def _u_g_pdf(self, func: Callable, arr: np.ndarray,
                 copula_params: Union[Params, tuple], **kwargs) -> np.ndarray:
        loc: np.ndarray = copula_params[3]
        copula_params: tuple = (
            copula_params[0], copula_params[1], copula_params[2],
            loc, copula_params[4], np.zeros(loc.shape, dtype=float)
        )
        return super()._u_g_pdf(func=func, arr=arr,
                                copula_params=copula_params, **kwargs)
