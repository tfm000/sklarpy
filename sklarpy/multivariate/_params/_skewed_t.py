# Contains code for holding Skewed-T parameters
from sklarpy.multivariate._params._generalized_hyperbolic import \
    MultivariateGenHyperbolicParams
from sklarpy.multivariate._params._student_t import MultivariateStudentTParams
from sklarpy.multivariate._distributions._skewed_t import \
    multivariate_skewed_t_gen

__all__ = ['MultivariateSkewedTParams']


class MultivariateSkewedTParams(MultivariateGenHyperbolicParams,
                                MultivariateStudentTParams):
    """Contains the fitted parameters of a Multivariate Skewed-T distribution.
    """
    _DIST_GENERATOR = multivariate_skewed_t_gen

    @property
    def lamb(self) -> float:
        return -self.dof/2

    @property
    def chi(self) -> float:
        return self.dof

    @property
    def psi(self) -> float:
        return 0.0
