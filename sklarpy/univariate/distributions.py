# Contains univariate probability distributions
import scipy.stats

from sklarpy.univariate._prefit_dists import \
    PreFitParametricContinuousUnivariate, PreFitParametricDiscreteUnivariate, \
    PreFitNumericalContinuousUnivariate, PreFitNumericalDiscreteUnivariate
from sklarpy.univariate._distributions import discrete_empirical_fit, \
    continuous_empirical_fit, _gig, _ig, _gh, _skewed_t, kde_fit, \
    discrete_laplace_fit, discrete_uniform_fit, geometric_fit, planck_fit, \
    poisson_fit
from sklarpy.univariate.distributions_map import distributions_map, \
    scipy_cp_names, cp_rename_dict, sklarpy_cp_names, dp_rename_dict

__all__ = [*distributions_map['all']]


###############################################################################
# Continuous (Parametric)
###############################################################################
def build_cp_str(dist: str, name: str) -> str:
    s: str = f"{name}: PreFitParametricContinuousUnivariate = " \
             f"PreFitParametricContinuousUnivariate('{name}', {dist}.pdf, " \
             f"{dist}.cdf, {dist}.ppf, {dist}.support, {dist}.fit)"
    if 'rvs' in dir(eval(dist)):
        s = f"{s[:-1]}, {dist}.rvs)"
    return s


for name in scipy_cp_names:
    dist: str = f"scipy.stats.{name}"
    s: str = build_cp_str(dist, name)
    exec(s)

for scipy_name, sklarpy_name in cp_rename_dict.items():
    dist: str = f"scipy.stats.{scipy_name}"
    s: str = build_cp_str(dist, sklarpy_name)
    exec(s)

for name in sklarpy_cp_names:
    dist: str = f"_{name}"
    s: str = build_cp_str(dist, name)
    exec(s)

skewed_t = PreFitParametricContinuousUnivariate(
    'skewed_t', _skewed_t.pdf, _skewed_t.cdf,
    _skewed_t.ppf, _skewed_t.support, _skewed_t.fit
)

###############################################################################
# Discrete (Parametric)
###############################################################################
for scipy_name, sklarpy_name in dp_rename_dict.items():
    dist: str = f"scipy.stats.{scipy_name}"
    s: str = f"{sklarpy_name}: PreFitParametricDiscreteUnivariate = " \
             f"PreFitParametricDiscreteUnivariate('{sklarpy_name}', " \
             f"{dist}.pmf, {dist}.cdf, {dist}.ppf, {dist}.support, " \
             f"{sklarpy_name}_fit)"
    if 'rvs' in dir(eval(dist)):
        s = f"{s[:-1]}, {dist}.rvs)"
    exec(s)

###############################################################################
# Numerical/Non-Parametric
###############################################################################
gaussian_kde = PreFitNumericalContinuousUnivariate('gaussian-kde', kde_fit)

empirical = PreFitNumericalContinuousUnivariate(
    'empirical', continuous_empirical_fit
)

discrete_empirical = PreFitNumericalDiscreteUnivariate(
    'discrete-empirical', discrete_empirical_fit
)
