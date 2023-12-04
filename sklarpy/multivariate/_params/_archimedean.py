# Contains code for holding Archimedean copula parameters
from sklarpy.utils._params import Params

__all__ = ['MvtClaytonParams', 'MvtGumbelParams',
           'BvtFrankParams']


class MvtArchimedeanParamsBase(Params):
    """Base class containing the fitted parameters of an Archimedean Copula."""
    @property
    def theta(self) -> float:
        """Returns the theta parameter defining the Archimedean distribution.

        Returns
        --------
        theta : float
            The theta parameter defining the Archimedean distribution.
        """
        return self.to_dict['theta']

    @property
    def d(self) -> int:
        """Returns the dimension of the dataset the Archimedean distribution
        was fitted too.

        Returns
        -------
        d: int
            The dimension of the dataset the Archimedean distribution
            was fitted too.
        """
        return self.to_dict['d']


class MvtClaytonParams(MvtArchimedeanParamsBase):
    """Contains the fitted parameters of a Clayton Copula."""


class MvtGumbelParams(MvtArchimedeanParamsBase):
    """Contains the fitted parameters of a Gumbel Copula."""


class BvtFrankParams(MvtArchimedeanParamsBase):
    """Contains the fitted parameters of a Frank Copula."""
