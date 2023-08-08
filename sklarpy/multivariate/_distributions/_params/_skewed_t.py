from sklarpy.multivariate._distributions._params._generalized_hyperbolic import MultivariateGenHyperbolicParams, __all__

__all__ = ['MultivariateSkewedTParams']


class MultivariateSkewedTParams(MultivariateGenHyperbolicParams):
    @property
    def dof(self) -> float:
        return self.to_dict['dof']

    @property
    def lamb(self) -> float:
        return -self.dof/2

    @property
    def chi(self) -> float:
        return self.dof

    @property
    def psi(self) -> float:
        return 0.0
