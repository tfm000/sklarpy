import numpy as np

from sklarpy.multivariate._distributions._params._hyperbolics import MultivariateMarginalHyperbolicParams, MultivariateHyperbolicParams, MultivariateNIGParams


__all__ = ['MultivariateSymMarginalHyperbolicParams', 'MultivariateHyperbolicParams', 'MultivariateNIGParams']


class MultivariateSymMarginalHyperbolicParams(MultivariateMarginalHyperbolicParams):
    @property
    def gamma(self) -> np.ndarray:
        return np.zeros(self.loc.shape, dtype=float)


class MultivariateSymHyperbolicParams(MultivariateHyperbolicParams):
    @property
    def gamma(self) -> np.ndarray:
        return np.zeros(self.loc.shape, dtype=float)


class MultivariateSymNIGParams(MultivariateNIGParams):
    @property
    def gamma(self) -> np.ndarray:
        return np.zeros(self.loc.shape, dtype=float)