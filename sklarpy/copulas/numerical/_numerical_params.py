from typing import Callable

import numpy as np

from sklarpy._other import Params

__all__ = ['EmpiricalCopulaParams', 'GaussianKdeCopulaParams']


class NumericalCopulaParams(Params):
    """Contains the fitted pdf and cdf interpolation functions for numerical copulas."""
    @property
    def pdf_func(self) -> Callable:
        """The fitted pdf function."""
        return self.to_dict['pdf_func']

    @property
    def cdf_func(self) -> Callable:
        """The fitted cdf function."""
        return self.to_dict['cdf_func']

    @property
    def umins(self) -> np.ndarray:
        """The minimum values of each variable's marginal cdf."""
        return self.to_dict['umins']

    @property
    def umaxs(self) -> np.ndarray:
        """The maximum values of each variable's marginal cdf."""
        return self.to_dict['umaxs']

    @property
    def num_variables(self) -> int:
        """The number of variables fitted."""
        return self.to_dict['num_variables']


class EmpiricalCopulaParams(NumericalCopulaParams):
    pass


class GaussianKdeCopulaParams(NumericalCopulaParams):
    pass
