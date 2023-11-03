# Contains code for the symmetric hyperbolic copula models
from sklarpy.copulas._distributions._symmetric_generalized_hyperbolic import \
    sym_gen_hyperbolic_copula_gen
from sklarpy.copulas._distributions._hyperbolics import \
    marginal_hyperbolic_copula_gen, hyperbolic_copula_gen, nig_copula_gen

__all__ = ['sym_marginal_hyperbolic_copula_gen', 'sym_hyperbolic_copula_gen',
           'sym_nig_copula_gen']


class sym_marginal_hyperbolic_copula_gen(marginal_hyperbolic_copula_gen,
                                         sym_gen_hyperbolic_copula_gen):
    """The Multivariate Symmetric Marginal Hyperbolic copula model."""


class sym_hyperbolic_copula_gen(hyperbolic_copula_gen,
                                sym_gen_hyperbolic_copula_gen):
    """The Multivariate Symmetric Hyperbolic copula model."""


class sym_nig_copula_gen(nig_copula_gen, sym_gen_hyperbolic_copula_gen):
    """The Multivariate Symmetric Normal-Inverse Gaussian (NIG) copula model.
    """
