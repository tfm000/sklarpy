# Contains code for hyperbolic copula models
import numpy as np
from typing import Union, Callable

from sklarpy.copulas._distributions._generalized_hyperbolic import \
    gen_hyperbolic_copula_gen
from sklarpy.utils._params import Params

__all__ = ['marginal_hyperbolic_copula_gen', 'hyperbolic_copula_gen',
           'nig_copula_gen']


class hyperbolic_copula_base_gen(gen_hyperbolic_copula_gen):
    def _get_lamb(self, copula_params: Union[Params, tuple]) -> float:
        """Returns the generalized hyperbolic lambda (lamb) parameter
        associated with the distribution.

        Parameters
        ------------
        copula_params: Union[Params, tuple]
            The parameters of the multivariate distribution used to specify
            your copula distribution.
            Can be a Params object or the specific multivariate distribution
            or a tuple containing these parameters in the correct order.

        Returns
        --------
        lamb: float
            The generalized hyperbolic lambda (lamb) parameter associated with
            the distribution
        """
        pass

    def _u_g_pdf(self, func: Callable, arr: np.ndarray,
                 copula_params: Union[Params, tuple], **kwargs) -> np.ndarray:
        lamb: float = self._get_lamb(copula_params=copula_params)
        copula_params: tuple = lamb, *copula_params
        return super()._u_g_pdf(func=func, arr=arr,
                                copula_params=copula_params, **kwargs)


class marginal_hyperbolic_copula_gen(hyperbolic_copula_base_gen):
    """The Multivariate Marginal Hyperbolic copula model."""
    def _get_lamb(self, copula_params: Union[Params, tuple]) -> float:
        return 1.0


class hyperbolic_copula_gen(hyperbolic_copula_base_gen):
    """The Multivariate Hyperbolic copula model."""
    def _get_lamb(self, copula_params: Union[Params, tuple]) -> float:
        d: int = copula_params[2].size
        return 0.5 * (d + 1)


class nig_copula_gen(hyperbolic_copula_base_gen):
    """The Multivariate Normal-Inverse Gaussian (NIG) copula model."""
    def _get_lamb(self, copula_params: Union[Params, tuple]) -> float:
        return -0.5
