# Grouping univariate distributions
import scipy.stats
from collections import deque

__all__ = ['distributions_map']


def get_scipy_names(names, required, rename_dict):
    """Returns the names of the distributions available in scipy.stats."""
    scipy_names: deque = deque()
    for name in names:
        dist = eval(f"scipy.stats.{name}")
        dist_dir = dir(dist)
        for req in required:
            if req not in dist_dir:
                continue
        scipy_names.append(name)

    for name in rename_dict:
        if name in scipy_names:
            scipy_names.remove(name)
        else:
            rename_dict.pop(name)
    return scipy_names, rename_dict


###############################################################################
# Continuous (Parametric)
###############################################################################
cp_required: tuple = ('pdf', 'cdf', 'ppf', 'support', 'fit')
cp_rename_dict: dict = {'norm': 'normal', 't': 'student_t'}
scipy_cp_names, cp_rename_dict = get_scipy_names(
    scipy.stats._continuous_distns._distn_names, cp_required, cp_rename_dict
)

sklarpy_cp_names: tuple = ('gh', 'gig', 'ig')

continuous_parametric_names: tuple = (
    *scipy_cp_names,
    *sklarpy_cp_names,
    *cp_rename_dict.values()
)

target_common_cp_names: tuple = (
    'cauchy', 'chi2', 'expon', 'gamma', 'lognorm', 'normal',
    'powerlaw', 'rayleigh', 'student_t', 'uniform'
)

common_continuous_parametric_names: tuple = tuple(
    name for name in target_common_cp_names if name in scipy_cp_names
)

target_multimodal_cp_names: tuple = ('arcsine', 'beta')
continuous_multimodal_parametric_names: tuple = tuple(
    name for name in target_multimodal_cp_names if name in scipy_cp_names
)

###############################################################################
# Discrete (Parametric)
###############################################################################
dp_required: tuple = ('pmf', 'cdf', 'ppf', 'support')
dp_rename_dict: dict = {'poisson': 'poisson', 'planck': 'planck',
                        'dlaplace': 'discrete_laplace',
                        'randint': 'discrete_uniform', 'geom': 'geometric'
                        }
_, dp_rename_dict = get_scipy_names(
    scipy.stats._discrete_distns._distn_names, dp_required, dp_rename_dict
)

discrete_parametric_names: tuple = tuple(dp_rename_dict.values())

target_common_dp_names: tuple = (
    'discrete_laplace', 'discrete_uniform', 'geometric', 'poisson'
)

common_discrete_parametric_names: tuple = tuple(
    name for name in target_common_dp_names
    if name in discrete_parametric_names
)

discrete_multimodal_parametric_names: tuple = ()

###############################################################################
# Numerical/Non-Parametric
###############################################################################
continuous_numerical_names: tuple = ('gaussian_kde', 'empirical')
discrete_numerical_names: tuple = ('discrete_empirical',)


###############################################################################
# Distribution Categories/Map
###############################################################################
all_continuous_names: tuple = (*continuous_parametric_names,
                               *continuous_numerical_names)

all_discrete_names: tuple = (*discrete_parametric_names,
                             *discrete_numerical_names)

all_common_names: tuple = (*common_continuous_parametric_names,
                           *common_discrete_parametric_names)

all_multimodal_names: tuple = (*continuous_multimodal_parametric_names,
                               *discrete_multimodal_parametric_names)

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
    'all common': all_common_names,
    'all multimodal': all_multimodal_names,
    'all parametric': all_parametric_names,
    'all numerical': all_numerical_names,
    'all continuous parametric': continuous_parametric_names,
    'all discrete parametric': discrete_parametric_names,
    'all continuous numerical': continuous_numerical_names,
    'all discrete numerical': discrete_numerical_names,
    'common continuous': common_continuous_parametric_names,
    'common discrete': common_discrete_parametric_names,
    'continuous multimodal': continuous_multimodal_parametric_names,
    'discrete multimodal': discrete_multimodal_parametric_names,
}
