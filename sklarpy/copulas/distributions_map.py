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
    'gh_copula',
    'mh_copula',
    'hyperbolic_copula',
    'nig_copula',
    'skewed_t_copula',
    'student_t_copula',
    'sgh_copula',
    'smh_copula',
    'shyperbolic_copula',
    'snig_copula',
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
