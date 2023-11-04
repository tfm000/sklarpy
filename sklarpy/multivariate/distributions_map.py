# Grouping multivariate distributions
__all__ = ['distributions_map']

###############################################################################
# Continuous (Parametric)
###############################################################################
continuous_parametric_names: tuple = (
    'mvt_normal',
    'mvt_student_t',
    'mvt_gh',
    'mvt_mh',
    'mvt_hyperbolic',
    'mvt_nig',
    'mvt_skewed_t',
    'mvt_sgh',
    'mvt_smh',
    'mvt_shyperbolic',
    'mvt_snig')

###############################################################################
# Discrete (Parametric)
###############################################################################
discrete_parametric_names: tuple = tuple()

###############################################################################
# Numerical/Non-Parametric
###############################################################################
continuous_numerical_names: tuple = ('mvt_gaussian_kde',)
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
