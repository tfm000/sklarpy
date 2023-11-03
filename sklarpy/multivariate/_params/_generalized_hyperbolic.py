# Contains code for holding Generalized Hyperbolic parameters
import numpy as np

from sklarpy.multivariate._params._symmetric_generalized_hyperbolic import \
    MultivariateSymGenHyperbolicParams
from sklarpy.multivariate._distributions._generalized_hyperbolic import \
    multivariate_gen_hyperbolic_gen

__all__ = ['MultivariateGenHyperbolicParams']


class MultivariateGenHyperbolicParams(MultivariateSymGenHyperbolicParams):
    """Contains the fitted parameters of a Multivariate Generalized Hyperbolic
    distribution."""
    _DIST_GENERATOR = multivariate_gen_hyperbolic_gen

    @property
    def gamma(self) -> np.ndarray:
        """The gamma / skewness parameter of the distribution.

        Returns
        -------
        gamma : np.ndarray
            The gamma / skewness parameter of the distribution.
        """
        return self.to_dict['gamma']

    @property
    def mean(self) -> np.ndarray:
        return self.loc + (self.exp_w * self.gamma)

    @property
    def cov(self) -> np.ndarray:
        gamma: np.ndarray = self.gamma.reshape((self.gamma.size, 1))
        return (self.exp_w * self.shape) + (self.var_w * gamma @ gamma.T)
