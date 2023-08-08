from sklarpy.multivariate._distributions._params._generalized_hyperbolic import MultivariateGenHyperbolicParams


__all__ = ['MultivariateMarginalHyperbolicParams', 'MultivariateHyperbolicParams', 'MultivariateNIGParams']


class MultivariateMarginalHyperbolicParams(MultivariateGenHyperbolicParams):
    @property
    def lamb(self) -> float:
        """The lambda parameter of the Multivariate Generalized Hyperbolic distribution
        and the underlying GIG distribution for W."""
        return 1.0


class MultivariateHyperbolicParams(MultivariateGenHyperbolicParams):
    @property
    def lamb(self) -> float:
        """The lambda parameter of the Multivariate Generalized Hyperbolic distribution
        and the underlying GIG distribution for W."""
        d: int = self.loc.size
        return 0.5 * (d + 1)


class MultivariateNIGParams(MultivariateGenHyperbolicParams):
    @property
    def lamb(self) -> float:
        """The lambda parameter of the Multivariate Generalized Hyperbolic distribution
        and the underlying GIG distribution for W."""
        return -0.5