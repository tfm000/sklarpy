import numpy as np

from sklarpy._other import Params

__all__ = ['MultivariateStudentTParams']


class MultivariateStudentTParams(Params):
    """Contains the fitted parameters of a Multivariate Student T distribution"""

    @property
    def mean(self) -> np.ndarray:
        """The mean/location/mu parameter."""
        return self.to_dict['mu']

    @property
    def mu(self) -> np.ndarray:
        """The mean/location/mu parameter."""
        return self.mean

    @property
    def loc(self) -> np.ndarray:
        """The mean/location/mu parameter."""
        return self.mean

    @property
    def shape(self) -> np.ndarray:
        """The shape parameter of the multivariate Student T distribution.
        Note this is not the same as the covariance matrix.
        """
        return self.to_dict['shape']

    @property
    def dof(self) -> float:
        """The degrees of freedom parameter of the multivariate Student T distribution."""
        return self.to_dict['dof']

    @property
    def degrees_of_freedom(self) -> float:
        """The degrees of freedom parameter of the multivariate Student T distribution."""
        return self.dof

    @property
    def cov(self) -> np.ndarray:
        """The covariance matrix of the multivariate Student T distribution.
        Note this is not the same as the shape matrix.
        """
        shape: np.ndarray = self.shape
        return shape * self.dof / (self.dof - 2)

    @property
    def covariance_matrix(self) -> np.ndarray:
        """The covariance matrix of the multivariate Student T distribution.
        Note this is not the same as the shape matrix.
        """
        return self.cov