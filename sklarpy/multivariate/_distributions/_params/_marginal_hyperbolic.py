from sklarpy.multivariate._distributions._params._generalized_hyperbolic import MultivariateGenHyperbolicParams


__all__ = ['MultivariateMarginalHyperbolicParams']


class MultivariateMarginalHyperbolicParams(MultivariateGenHyperbolicParams):
    @property
    def lamb(self) -> float:
        """The lambda parameter of the Multivariate Generalized Hyperbolic distribution
        and the underlying GIG distribution for W."""
        return 1.0
