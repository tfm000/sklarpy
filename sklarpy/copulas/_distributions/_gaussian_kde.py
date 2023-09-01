# Contains code for the gaussian kde copula model
from sklarpy.copulas._prefit_dists import PreFitCopula

__all__ = ['gaussian_kde_copula_gen']


class gaussian_kde_copula_gen(PreFitCopula):
    """The Multivariate Gaussian KDE copula model."""
