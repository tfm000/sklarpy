from sklarpy.multivariate._params._generalized_hyperbolic import MultivariateGenHyperbolicParams
from sklarpy.multivariate._distributions._hyperbolics import multivariate_nig_gen, multivariate_hyperbolic_gen, multivariate_marginal_hyperbolic_gen


__all__ = ['MultivariateMarginalHyperbolicParams', 'MultivariateHyperbolicParams', 'MultivariateNIGParams']


class MultivariateMarginalHyperbolicParams(MultivariateGenHyperbolicParams):
    _DIST_GENERATOR = multivariate_marginal_hyperbolic_gen

    @property
    def lamb(self) -> float:
        """The lambda parameter of the Multivariate Generalized Hyperbolic distribution
        and the underlying GIG distribution for W."""
        return 1.0


class MultivariateHyperbolicParams(MultivariateGenHyperbolicParams):
    _DIST_GENERATOR = multivariate_hyperbolic_gen

    @property
    def lamb(self) -> float:
        """The lambda parameter of the Multivariate Generalized Hyperbolic distribution
        and the underlying GIG distribution for W."""
        d: int = self.loc.size
        return 0.5 * (d + 1)


class MultivariateNIGParams(MultivariateGenHyperbolicParams):
    _DIST_GENERATOR = multivariate_nig_gen

    @property
    def lamb(self) -> float:
        """The lambda parameter of the Multivariate Generalized Hyperbolic distribution
        and the underlying GIG distribution for W."""
        return -0.5
