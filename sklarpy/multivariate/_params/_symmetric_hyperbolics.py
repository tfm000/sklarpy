import numpy as np

from sklarpy.multivariate._params._hyperbolics import MultivariateMarginalHyperbolicParams, MultivariateHyperbolicParams, MultivariateNIGParams
from sklarpy.multivariate._distributions._symmetric_hyperbolics import multivariate_sym_hyperbolic_gen, multivariate_sym_marginal_hyperbolic_gen, multivariate_sym_nig_gen

__all__ = ['MultivariateSymMarginalHyperbolicParams', 'MultivariateSymHyperbolicParams', 'MultivariateSymNIGParams']


class MultivariateSymHyperbolicParamsBase:
    """Base Class to be inherited by multivariate symmetric hyperbolics."""

    @property
    def gamma(self) -> np.ndarray:
        return np.zeros(self.loc.shape, dtype=float)


class MultivariateSymMarginalHyperbolicParams(MultivariateMarginalHyperbolicParams, MultivariateSymHyperbolicParamsBase):
    _DIST_GENERATOR = multivariate_sym_marginal_hyperbolic_gen


class MultivariateSymHyperbolicParams(MultivariateHyperbolicParams, MultivariateSymHyperbolicParamsBase):
    _DIST_GENERATOR = multivariate_sym_hyperbolic_gen


class MultivariateSymNIGParams(MultivariateNIGParams, MultivariateSymHyperbolicParamsBase):
    _DIST_GENERATOR = multivariate_sym_nig_gen
