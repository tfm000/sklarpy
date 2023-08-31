# Contains code for building copula models
from sklarpy.copulas._distributions._gaussian import gaussian_copula_gen
from sklarpy.copulas._distributions._gaussian_kde import gaussian_kde_copula_gen
from sklarpy.copulas._distributions._generalized_hyperbolic import gen_hyperbolic_copula_gen
from sklarpy.copulas._distributions._hyperbolics import marginal_hyperbolic_copula_gen, hyperbolic_copula_gen, nig_copula_gen
from sklarpy.copulas._distributions._skewed_t import skewed_t_copula_gen
from sklarpy.copulas._distributions._student_t import student_t_copula_gen
from sklarpy.copulas._distributions._symmetric_generalized_hyperbolic import sym_gen_hyperbolic_copula_gen
from sklarpy.copulas._distributions._symmetric_hyperbolics import sym_marginal_hyperbolic_copula_gen, sym_hyperbolic_copula_gen, sym_nig_copula_gen
from sklarpy.copulas._distributions._archimedean import clayton_copula_gen, gumbel_copula_gen, frank_copula_gen

from sklarpy.multivariate import multivariate_normal, multivariate_gaussian_kde, multivariate_gen_hyperbolic, \
    multivariate_marginal_hyperbolic, multivariate_hyperbolic, multivariate_nig, \
    multivariate_student_t, multivariate_skewed_t, multivariate_sym_gen_hyperbolic, \
    multivariate_sym_marginal_hyperbolic, multivariate_sym_hyperbolic, multivariate_sym_nig

from sklarpy.multivariate.distributions import multivariate_clayton, multivariate_gumbel, bivariate_frank

__all__ = ['gaussian_copula', 'gaussian_kde_copula', 'gen_hyperbolic_copula',
           'marginal_hyperbolic_copula', 'hyperbolic_copula', 'nig_copula',
           'skewed_t_copula', 'student_t_copula', 'sym_gen_hyperbolic_copula',
           'sym_marginal_hyperbolic_copula', 'sym_hyperbolic_copula', 'sym_nig_copula',
           'clayton_copula', 'gumbel_copula', 'frank_copula'
           ]

########################################################################################################################
# Numerical/Non-Parametric
########################################################################################################################
gaussian_kde_copula: gaussian_kde_copula_gen = gaussian_kde_copula_gen(name="gaussian_kde", mv_object=multivariate_gaussian_kde)

########################################################################################################################
# Parametric
########################################################################################################################
gaussian_copula: gaussian_copula_gen = gaussian_copula_gen(name="gaussian", mv_object=multivariate_normal)
gen_hyperbolic_copula: gen_hyperbolic_copula_gen = gen_hyperbolic_copula_gen(name="gen_hyperbolic", mv_object=multivariate_gen_hyperbolic)
marginal_hyperbolic_copula: marginal_hyperbolic_copula_gen = marginal_hyperbolic_copula_gen(name="marginal_hyperbolic", mv_object=multivariate_marginal_hyperbolic)
hyperbolic_copula: hyperbolic_copula_gen = hyperbolic_copula_gen(name="hyperbolic", mv_object=multivariate_hyperbolic)
nig_copula: nig_copula_gen = nig_copula_gen(name="nig", mv_object=multivariate_nig)
skewed_t_copula: skewed_t_copula_gen = skewed_t_copula_gen(name="skewed_t", mv_object=multivariate_skewed_t)
student_t_copula: student_t_copula_gen = student_t_copula_gen(name="student_t", mv_object=multivariate_student_t)
sym_gen_hyperbolic_copula: sym_gen_hyperbolic_copula_gen = sym_gen_hyperbolic_copula_gen(name="sym_gen_hyperbolic", mv_object=multivariate_sym_gen_hyperbolic)
sym_marginal_hyperbolic_copula: sym_marginal_hyperbolic_copula_gen = sym_marginal_hyperbolic_copula_gen(name="sym_marginal_hyperbolic", mv_object=multivariate_sym_marginal_hyperbolic)
sym_hyperbolic_copula: sym_hyperbolic_copula_gen = sym_hyperbolic_copula_gen(name="sym_hyperbolic", mv_object=multivariate_sym_hyperbolic)
sym_nig_copula: sym_nig_copula_gen = sym_nig_copula_gen(name="sym_nig", mv_object=multivariate_sym_nig)
clayton_copula: clayton_copula_gen = clayton_copula_gen(name="clayton", mv_object=multivariate_clayton)
gumbel_copula: gumbel_copula_gen = gumbel_copula_gen(name="gumbel", mv_object=multivariate_gumbel)
frank_copula: frank_copula_gen = frank_copula_gen(name="frank", mv_object=bivariate_frank)
