# Contains code for holding Gaussian / Normal parameters
import numpy as np

from sklarpy.utils._params import Params

__all__ = ['MvtNormalParams']


class MvtNormalParams(Params):
    """Contains the fitted parameters of a Multivariate Gaussian / Normal
    distribution."""

    @property
    def mean(self) -> np.ndarray:
        """The mean vector of the distribution.

        Returns
        -------
        mean: np.ndarray
            The mean vector of the distribution.
        """
        return self.loc

    @property
    def loc(self) -> np.ndarray:
        """The location parameter of the distribution.

        Returns
        -------
        loc: np.ndarray
            The location parameter of the distribution.
        """
        return self.to_dict['loc']

    @property
    def shape(self) -> np.ndarray:
        """The shape parameter of the distribution.

        Returns
        -------
        shape: np.ndarray
            The shape parameter of the distribution.
        """
        return self.to_dict['shape']

    @property
    def cov(self) -> np.ndarray:
        """The covariance matrix of the distribution.

        Returns
        -------
        cov: np.ndarray
            The covariance matrix of the distribution.
        """
        return self.shape

    @property
    def covariance_matrix(self) -> np.ndarray:
        """The covariance matrix of the distribution.

        Returns
        -------
        cov: np.ndarray
            The covariance matrix of the distribution.
        """
        return self.cov
