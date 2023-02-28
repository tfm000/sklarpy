import numpy as np

from sklarpy._other import Params

__all__ = ['GaussianCopulaParams', 'StudentTCopulaParams']


class GaussianCopulaParams(Params):
    """Contains the fitted parameters of a Gaussian Copula."""
    @property
    def corr(self) -> np.ndarray:
        """The fitted correlation matrix."""
        return self.to_dict['corr']

    @property
    def correlation_matrix(self) -> np.ndarray:
        """The fitted correlation matrix."""
        return self.corr


class StudentTCopulaParams(GaussianCopulaParams):
    """Contains the fitted parameters of a Student-T Copula with one degree of freedom."""
    @property
    def dof(self) -> float:
        """The fitted degrees of freedom parameter."""
        return self.to_dict['dof']

    @property
    def degrees_of_freedom(self) -> float:
        """The fitted degrees of freedom parameter."""
        return self.dof
