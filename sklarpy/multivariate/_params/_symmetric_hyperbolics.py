# Contains code for holding the parameters of Symmetric Hyperbolic models
import numpy as np

from sklarpy.multivariate._params._hyperbolics import (
    MvtMHParams, MvtHyperbolicParams, MvtNIGParams)
from sklarpy.multivariate._distributions._symmetric_hyperbolics import \
    multivariate_sym_hyperbolic_gen, multivariate_sym_marginal_hyperbolic_gen,\
    multivariate_sym_nig_gen

__all__ = ['MvtSMHParams', 'MvtSHyperbolicParams', 'MvtSNIGParams']


class MvtSHyperbolicParamsBase:
    """Base Class containing the fitted parameters of a Multivariate Symmetric
    Hyperbolic distribution."""

    @property
    def gamma(self) -> np.ndarray:
        return np.zeros(self.loc.shape, dtype=float)


class MvtSMHParams(MvtSHyperbolicParamsBase, MvtMHParams):
    """Contains the fitted parameters of a Multivariate Symmetric Marginal
    Hyperbolic distribution."""
    _DIST_GENERATOR = multivariate_sym_marginal_hyperbolic_gen


class MvtSHyperbolicParams(MvtSHyperbolicParamsBase, MvtHyperbolicParams):
    """Contains the fitted parameters of a Multivariate Symmetric Hyperbolic
    distribution."""
    _DIST_GENERATOR = multivariate_sym_hyperbolic_gen


class MvtSNIGParams(MvtSHyperbolicParamsBase, MvtNIGParams):
    """Contains the fitted parameters of a Multivariate Symmetric
    Normal-Inverse Gaussian (NIG) distribution."""
    _DIST_GENERATOR = multivariate_sym_nig_gen
