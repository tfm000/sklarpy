import numpy as np

from sklarpy._other import Params


class MultivariateNormalParams(Params):
    """Contains the fitted parameters of a Multivariate Normal distribution."""

    @property
    def mean(self) -> np.ndarray:
        """The mean/location/mu parameter."""
        return self.loc

    @property
    def mu(self) -> np.ndarray:
        """The mean/location/mu parameter."""
        return self.mean

    @property
    def loc(self) -> np.ndarray:
        """The mean/location/mu parameter."""
        return self.to_dict['loc']

    @property
    def shape(self) -> np.ndarray:
        """The fitted covariance/shape matrix."""
        return self.to_dict['shape']

    @property
    def cov(self) -> np.ndarray:
        """The fitted covariance/shape matrix."""
        return self.shape

    @property
    def covariance_matrix(self) -> np.ndarray:
        """The fitted covariance/shape matrix."""
        return self.shape

