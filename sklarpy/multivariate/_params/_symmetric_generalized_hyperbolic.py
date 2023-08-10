import numpy as np
import scipy.special

from sklarpy._other import Params

__all__ = ['MultivariateSymGenHyperbolicParams']


class MultivariateSymGenHyperbolicParams(Params):
    """Contains the fitted parameters of a Multivariate Symmetric Generalized Hyperbolic distribution."""

    @property
    def lamb(self) -> float:
        """The lambda parameter of the Multivariate Symmetric Generalized Hyperbolic distribution
        and the underlying GIG distribution for W."""
        return self.to_dict['lamb']

    @property
    def chi(self) -> float:
        """The chi parameter of the Multivariate Symmetric Generalized Hyperbolic distribution
        and the underlying GIG distribution for W."""
        return self.to_dict['chi']

    @property
    def psi(self) -> float:
        """The psi parameter of the Multivariate Symmetric Generalized Hyperbolic distribution
        and the underlying GIG distribution for W."""
        return self.to_dict['psi']

    @property
    def loc(self) -> np.ndarray:
        """The location / mu parameter..
        """
        return self.to_dict['loc']

    @property
    def shape(self) -> np.ndarray:
        """The shape parameter.
        Note this is not the same as the covariance matrix in general.
        """
        return self.to_dict['shape']

    @property
    def w_mean(self) -> float:
        """The mean of the rv W ~ GIG(chi, psi, lambda)"""
        r: float = np.sqrt(self.chi * self.psi)
        return np.sqrt(self.chi / self.psi) * (scipy.special.kv(self.lamb + 1, r) / scipy.special.kv(self.lamb, r))

    @property
    def w_variance(self) -> float:
        """The variance of the rv W ~ GIG(chi, psi, lambda)"""
        r: float = np.sqrt(self.chi * self.psi)
        e2: float = (self.chi / self.psi) * (scipy.special.kv(self.lamb + 2, r) / scipy.special.kv(self.lamb, r))
        return e2 - (self.w_mean ** 2)

    @property
    def mean(self) -> np.ndarray:
        """The mean vector of the Multivariate Symmetric Generalized Hyperbolic distribution.
        """
        return self.loc

    @property
    def cov(self) -> np.ndarray:
        """The covariance matrix of the Multivariate Symmetric Generalized Hyperbolic distribution.
        Note this is not the same as the shape vector in general.
        """
        return self.w_mean * self.shape

    @property
    def covariance_matrix(self) -> np.ndarray:
        """The covariance matrix of the Multivariate Symmetric Generalized Hyperbolic distribution.
        Note this is not the same as the shape vector in general.
        """
        return self.cov
