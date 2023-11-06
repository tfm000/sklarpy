# Contains code for holding Student-T parameters
import numpy as np

from sklarpy.multivariate._params._normal import MvtNormalParams

__all__ = ['MvtStudentTParams']


class MvtStudentTParams(MvtNormalParams):
    """Contains the fitted parameters of a Multivariate Student-T
    distribution."""

    @property
    def dof(self) -> float:
        """The degrees of freedom parameter of the Multivariate Student-T
        distribution.

        Returns
        -------
        dof: float
            The degrees of freedom parameter of the distribution.
        """
        return self.to_dict['dof']

    @property
    def degrees_of_freedom(self) -> float:
        """The degrees of freedom parameter of the Multivariate Student-T
        distribution.

        Returns
        -------
        dof: float
            The degrees of freedom parameter of the distribution.
        """
        return self.dof

    @property
    def cov(self) -> np.ndarray:
        return self.shape * self.dof / (self.dof - 2) if self.dof > 2 \
            else np.full(self.shape.shape, np.nan, dtype=float)
