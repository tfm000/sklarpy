# Contains code for holding the parameters of Hyperbolic models
from sklarpy.multivariate._params._generalized_hyperbolic import MvtGHParams
from sklarpy.multivariate._distributions._hyperbolics import \
    multivariate_nig_gen, multivariate_hyperbolic_gen,\
    multivariate_marginal_hyperbolic_gen


__all__ = ['MvtMHParams', 'MvtHyperbolicParams', 'MvtNIGParams']


class MvtMHParams(MvtGHParams):
    """Contains the fitted parameters of a Multivariate Marginal Hyperbolic
    distribution."""
    _DIST_GENERATOR = multivariate_marginal_hyperbolic_gen

    @property
    def lamb(self) -> float:
        return 1.0


class MvtHyperbolicParams(MvtGHParams):
    """Contains the fitted parameters of a Multivariate Hyperbolic
    distribution."""
    _DIST_GENERATOR = multivariate_hyperbolic_gen

    @property
    def lamb(self) -> float:
        return 0.5 * (self.loc.size + 1)


class MvtNIGParams(MvtGHParams):
    """Contains the fitted parameters of a Multivariate Normal-Inverse
    Gaussian (NIG) distribution."""
    _DIST_GENERATOR = multivariate_nig_gen

    @property
    def lamb(self) -> float:
        return -0.5
