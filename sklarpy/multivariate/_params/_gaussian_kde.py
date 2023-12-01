# Contains code for holding Gaussian KDE parameters.
from sklarpy.utils._params import Params

__all__ = ['MvtGaussianKDEParams']


class MvtGaussianKDEParams(Params):
    """Contains the fitted parameters of a Multivariate Gaussian KDE
    distribution."""
    @property
    def kde(self):
        """Returns the scipy gaussian_kde object fitted to a given dataset.

        Returns
        -------
        kde:
            A fitted scipy gaussian_kde object.
        """
        return self.to_dict['kde']
