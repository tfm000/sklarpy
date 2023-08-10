import numpy as np

from sklarpy.multivariate._params._symmetric_generalized_hyperbolic import MultivariateSymGenHyperbolicParams
from sklarpy.multivariate._distributions._generalized_hyperbolic import multivariate_gen_hyperbolic_gen

__all__ = ['MultivariateGenHyperbolicParams']


class MultivariateGenHyperbolicParams(MultivariateSymGenHyperbolicParams):
    """Contains the fitted parameters of a Multivariate Generalized Hyperbolic distribution."""
    _DIST_GENERATOR = multivariate_gen_hyperbolic_gen

    @property
    def lamb(self) -> float:
        """The lambda parameter of the Multivariate Generalized Hyperbolic distribution
        and the underlying GIG distribution for W."""

    @property
    def chi(self) -> float:
        """The chi parameter of the Multivariate Generalized Hyperbolic distribution
        and the underlying GIG distribution for W."""

    @property
    def psi(self) -> float:
        """The psi parameter of the Multivariate Generalized Hyperbolic distribution
        and the underlying GIG distribution for W."""

    @property
    def loc(self) -> np.ndarray:
        """The location / mu parameter.
        Note this is not the same as the mean vector in general.
        """

    @property
    def gamma(self) -> np.ndarray:
        """The gamma/skewness parameter."""
        return self.to_dict['gamma']

    @property
    def mean(self) -> np.ndarray:
        """The mean vector of the Multivariate Generalized Hyperbolic distribution.
        Note this is not the same as the location / loc vector in general.
        """
        return self.loc + (self.w_mean * self.gamma)

    @property
    def cov(self) -> np.ndarray:
        """The covariance matrix of the Multivariate Generalized Hyperbolic distribution.
        Note this is not the same as the shape vector in general.
        """
        gamma: np.ndarray = self.gamma.reshape((self.gamma.size, 1))
        return (self.w_mean * self.shape) + (self.w_variance * gamma @ gamma.T)
