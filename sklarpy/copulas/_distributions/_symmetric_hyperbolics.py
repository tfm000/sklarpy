import numpy as np
from typing import Union, Callable

from sklarpy.copulas._distributions._symmetric_generalized_hyperbolic import sym_gen_hyperbolic_copula_gen
from sklarpy.copulas._distributions._hyperbolics import marginal_hyperbolic_copula_gen, hyperbolic_copula_gen, nig_copula_gen

__all__ = ['sym_marginal_hyperbolic_copula_gen', 'sym_hyperbolic_copula_gen', 'sym_nig_copula_gen']


class sym_marginal_hyperbolic_copula_gen(marginal_hyperbolic_copula_gen, sym_gen_hyperbolic_copula_gen):
    pass


class sym_hyperbolic_copula_gen(hyperbolic_copula_gen, sym_gen_hyperbolic_copula_gen):
    pass


class sym_nig_copula_gen(nig_copula_gen, sym_gen_hyperbolic_copula_gen):
    pass
