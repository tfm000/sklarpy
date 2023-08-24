import numpy as np
import sys

from sklarpy.multivariate._distributions._gaussian_kde import multivariate_gaussian_kde_gen
from sklarpy.multivariate._distributions._generalized_hyperbolic import multivariate_gen_hyperbolic_gen
from sklarpy.multivariate._distributions._hyperbolics import multivariate_hyperbolic_gen, multivariate_nig_gen, multivariate_marginal_hyperbolic_gen
from sklarpy.multivariate._distributions._normal import multivariate_normal_gen
from sklarpy.multivariate._distributions._skewed_t import multivariate_skewed_t_gen
from sklarpy.multivariate._distributions._student_t import multivariate_student_t_gen
from sklarpy.multivariate._distributions._symmetric_generalized_hyperbolic import multivariate_sym_gen_hyperbolic_gen
from sklarpy.multivariate._distributions._symmetric_hyperbolics import multivariate_sym_hyperbolic_gen, multivariate_sym_nig_gen, multivariate_sym_marginal_hyperbolic_gen
from sklarpy.multivariate._distributions._archimedean import multivariate_clayton_gen, multivariate_gumbel_gen, bivariate_frank_gen

from sklarpy.multivariate._params._gaussian_kde import MultivariateGaussianKDEParams
from sklarpy.multivariate._params._generalized_hyperbolic import MultivariateGenHyperbolicParams
from sklarpy.multivariate._params._hyperbolics import MultivariateMarginalHyperbolicParams, MultivariateHyperbolicParams, MultivariateNIGParams
from sklarpy.multivariate._params._normal import MultivariateNormalParams
from sklarpy.multivariate._params._skewed_t import MultivariateSkewedTParams
from sklarpy.multivariate._params._student_t import MultivariateStudentTParams
from sklarpy.multivariate._params._symmetric_generalized_hyperbolic import MultivariateSymGenHyperbolicParams
from sklarpy.multivariate._params._symmetric_hyperbolics import MultivariateSymMarginalHyperbolicParams, MultivariateSymHyperbolicParams, MultivariateSymNIGParams
from sklarpy.multivariate._params._archimedean import MultivariateClaytonParams, MultivariateGumbelParams, BivariateFrankParams

__all__ = ['multivariate_gaussian_kde', 'multivariate_gen_hyperbolic', 'multivariate_marginal_hyperbolic', 'multivariate_hyperbolic', 'multivariate_nig',
           'multivariate_normal', 'multivariate_skewed_t', 'multivariate_student_t', 'multivariate_sym_gen_hyperbolic',
           'multivariate_sym_marginal_hyperbolic', 'multivariate_sym_hyperbolic', 'multivariate_sym_nig']


########################################################################################################################
# Numerical/Non-Parametric
########################################################################################################################
multivariate_gaussian_kde: multivariate_gaussian_kde_gen = multivariate_gaussian_kde_gen(name='multivariate_gaussian_kde', params_obj=MultivariateGaussianKDEParams, num_params=1, max_num_variables=np.inf)

########################################################################################################################
# Continuous (Parametric)
########################################################################################################################
multivariate_gen_hyperbolic: multivariate_gen_hyperbolic_gen = multivariate_gen_hyperbolic_gen(name="multivariate_gen_hyperbolic", params_obj=MultivariateGenHyperbolicParams, num_params=6, max_num_variables=np.inf)
multivariate_marginal_hyperbolic: multivariate_marginal_hyperbolic_gen = multivariate_marginal_hyperbolic_gen(name='multivariate_marginal_hyperbolic', params_obj=MultivariateMarginalHyperbolicParams, num_params=5, max_num_variables=np.inf)
multivariate_hyperbolic: multivariate_hyperbolic_gen = multivariate_hyperbolic_gen(name='multivariate_hyperbolic', params_obj=MultivariateHyperbolicParams, num_params=5, max_num_variables=np.inf)
multivariate_nig: multivariate_nig_gen = multivariate_nig_gen(name='multivariate_nig', params_obj=MultivariateNIGParams, num_params=5, max_num_variables=np.inf)
multivariate_normal: multivariate_normal_gen = multivariate_normal_gen(name="multivariate_normal", params_obj=MultivariateNormalParams, num_params=2, max_num_variables=np.inf)
multivariate_skewed_t: multivariate_skewed_t_gen = multivariate_skewed_t_gen(name='multivariate_skewed_t', params_obj=MultivariateSkewedTParams, num_params=4, max_num_variables=np.inf)
multivariate_student_t: multivariate_student_t_gen = multivariate_student_t_gen(name="multivariate_student_t", params_obj=MultivariateStudentTParams, num_params=3, max_num_variables=np.inf)
multivariate_sym_gen_hyperbolic: multivariate_sym_gen_hyperbolic_gen = multivariate_sym_gen_hyperbolic_gen(name="multivariate_sym_gen_hyperbolic", params_obj=MultivariateSymGenHyperbolicParams, num_params=5, max_num_variables=np.inf)
multivariate_sym_marginal_hyperbolic: multivariate_sym_marginal_hyperbolic_gen = multivariate_sym_marginal_hyperbolic_gen(name='multivariate_sym_marginal_hyperbolic', params_obj=MultivariateSymMarginalHyperbolicParams, num_params=4, max_num_variables=np.inf)
multivariate_sym_hyperbolic: multivariate_sym_hyperbolic_gen = multivariate_sym_hyperbolic_gen(name='multivariate_sym_hyperbolic', params_obj=MultivariateSymHyperbolicParams, num_params=4, max_num_variables=np.inf)
multivariate_sym_nig: multivariate_sym_nig_gen = multivariate_sym_nig_gen(name='multivariate_sym_nig', params_obj=MultivariateSymNIGParams, num_params=4, max_num_variables=np.inf)

########################################################################################################################
# Copula Only
########################################################################################################################
multivariate_clayton: multivariate_clayton_gen = multivariate_clayton_gen(name='multivariate_clayton', params_obj=MultivariateClaytonParams, num_params=2, max_num_variables=np.inf)
multivariate_gumbel: multivariate_gumbel_gen = multivariate_gumbel_gen(name='multivariate_gumbel', params_obj=MultivariateGumbelParams, num_params=2, max_num_variables=sys.getrecursionlimit())
bivariate_frank: bivariate_frank_gen = bivariate_frank_gen(name='bivariate_frank', params_obj=BivariateFrankParams, num_params=2, max_num_variables=2)
