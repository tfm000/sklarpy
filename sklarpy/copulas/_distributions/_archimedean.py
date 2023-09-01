# Contains code for archimedean copula models
from sklarpy.copulas._prefit_dists import PreFitCopula

__all__ = ['clayton_copula_gen', 'gumbel_copula_gen', 'frank_copula_gen']


class clayton_copula_gen(PreFitCopula):
    """The Multivariate Clayton copula model."""


class gumbel_copula_gen(PreFitCopula):
    """The Multivariate Gumbel copula model."""


class frank_copula_gen(PreFitCopula):
    """The Bivariate Frank copula model."""
