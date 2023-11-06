# Contains code for holding Skewed-T parameters
from sklarpy.multivariate._params._generalized_hyperbolic import MvtGHParams
from sklarpy.multivariate._params._student_t import MvtStudentTParams
from sklarpy.multivariate._distributions._skewed_t import \
    multivariate_skewed_t_gen

__all__ = ['MvtSkewedTParams']


class MvtSkewedTParams(MvtGHParams, MvtStudentTParams):
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
