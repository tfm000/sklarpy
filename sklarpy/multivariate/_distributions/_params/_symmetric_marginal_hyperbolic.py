import numpy as np

from sklarpy.multivariate._distributions._params._marginal_hyperbolic import MultivariateMarginalHyperbolicParams


__all__ = ['MultivariateSymMarginalHyperbolicParams']


class MultivariateSymMarginalHyperbolicParams(MultivariateMarginalHyperbolicParams):
    @property
    def gamma(self) -> np.ndarray:
        return np.zeros(self.loc.shape, dtype=float)
