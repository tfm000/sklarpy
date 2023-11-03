# Grouping multivariate distributions
__all__ = ['distributions_map']

###############################################################################
# Continuous (Parametric)
###############################################################################
continuous_parametric_names: tuple = (
    'multivariate_normal',
    'multivariate_student_t',
    'multivariate_gen_hyperbolic',
    'multivariate_marginal_hyperbolic',
    'multivariate_hyperbolic',
    'multivariate_nig',
    'multivariate_skewed_t',
    'multivariate_sym_gen_hyperbolic',
    'multivariate_sym_marginal_hyperbolic',
    'multivariate_sym_hyperbolic',
    'multivariate_sym_nig')

###############################################################################
# Discrete (Parametric)
###############################################################################
discrete_parametric_names: tuple = tuple()

###############################################################################
# Numerical/Non-Parametric
###############################################################################
continuous_numerical_names: tuple = ('multivariate_gaussian_kde',)
discrete_numerical_names: tuple = tuple()

###############################################################################
# Distribution Categories/Map
###############################################################################
all_continuous_names: tuple = (*continuous_numerical_names,
                               *continuous_parametric_names)

all_discrete_names: tuple = (*discrete_parametric_names,
                             *discrete_numerical_names)

all_parametric_names: tuple = (*continuous_parametric_names,
                               *discrete_parametric_names)

all_numerical_names: tuple = (*continuous_numerical_names,
                              *discrete_numerical_names)

all_distributions: tuple = (*all_continuous_names,
                            *all_discrete_names)

distributions_map: dict = {
    'all': all_distributions,
    'all continuous': all_continuous_names,
    'all discrete': all_discrete_names,
    'all parametric': all_parametric_names,
    'all numerical': all_numerical_names,
    'all continuous parametric': continuous_parametric_names,
    'all discrete parametric': discrete_parametric_names,
    'all continuous numerical': continuous_numerical_names,
    'all discrete numerical': discrete_numerical_names,
}
