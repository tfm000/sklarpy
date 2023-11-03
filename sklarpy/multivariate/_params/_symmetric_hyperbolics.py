# Contains code for holding the parameters of Symmetric Hyperbolic models
import numpy as np

from sklarpy.multivariate._params._hyperbolics import \
    MultivariateMarginalHyperbolicParams, MultivariateHyperbolicParams, \
    MultivariateNIGParams
from sklarpy.multivariate._distributions._symmetric_hyperbolics import \
    multivariate_sym_hyperbolic_gen, multivariate_sym_marginal_hyperbolic_gen,\
    multivariate_sym_nig_gen

__all__ = ['MultivariateSymMarginalHyperbolicParams',
           'MultivariateSymHyperbolicParams', 'MultivariateSymNIGParams']


class MultivariateSymHyperbolicParamsBase:
    """Base Class containing the fitted parameters of a Multivariate Symmetric
    Hyperbolic distribution."""

    @property
    def gamma(self) -> np.ndarray:
        return np.zeros(self.loc.shape, dtype=float)


class MultivariateSymMarginalHyperbolicParams(
    MultivariateSymHyperbolicParamsBase, MultivariateMarginalHyperbolicParams):
    """Contains the fitted parameters of a Multivariate Symmetric Marginal
    Hyperbolic distribution."""
    _DIST_GENERATOR = multivariate_sym_marginal_hyperbolic_gen


class MultivariateSymHyperbolicParams(MultivariateSymHyperbolicParamsBase,
                                      MultivariateHyperbolicParams):
    """Contains the fitted parameters of a Multivariate Symmetric Hyperbolic
    distribution."""
    _DIST_GENERATOR = multivariate_sym_hyperbolic_gen


class MultivariateSymNIGParams(MultivariateSymHyperbolicParamsBase,
                               MultivariateNIGParams):
    """Contains the fitted parameters of a Multivariate Symmetric
    Normal-Inverse Gaussian (NIG) distribution."""
    _DIST_GENERATOR = multivariate_sym_nig_gen
