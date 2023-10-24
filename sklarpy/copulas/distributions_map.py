# Grouping copula models
__all__ = ['distributions_map']

###############################################################################
# Parametric
###############################################################################
all_parametric_names: tuple = (
    'clayton_copula',
    'gumbel_copula',
    'frank_copula',
    'gaussian_copula',
    'gen_hyperbolic_copula',
    'marginal_hyperbolic_copula',
    'hyperbolic_copula',
    'nig_copula',
    'skewed_t_copula',
    'student_t_copula',
    'sym_gen_hyperbolic_copula',
    'sym_marginal_hyperbolic_copula',
    'sym_hyperbolic_copula',
    'sym_nig_copula',
)

###############################################################################
# Numerical/Non-Parametric
###############################################################################
all_numerical_names: tuple = ('gaussian_kde_copula',)

###############################################################################
# Distribution Categories/Map
###############################################################################
all_distributions: tuple = (*all_numerical_names,
                            *all_parametric_names)

distributions_map: dict = {
    'all': all_distributions,
    'all parametric': all_parametric_names,
    'all numerical': all_numerical_names,
}
