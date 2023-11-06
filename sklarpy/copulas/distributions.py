# Contains code for building copula models
from sklarpy.copulas._distributions._gaussian import gaussian_copula_gen
from sklarpy.copulas._distributions._gaussian_kde import (
    gaussian_kde_copula_gen)
from sklarpy.copulas._distributions._generalized_hyperbolic import (
    gen_hyperbolic_copula_gen)
from sklarpy.copulas._distributions._hyperbolics import (
    marginal_hyperbolic_copula_gen, hyperbolic_copula_gen, nig_copula_gen)
from sklarpy.copulas._distributions._skewed_t import skewed_t_copula_gen
from sklarpy.copulas._distributions._student_t import student_t_copula_gen
from sklarpy.copulas._distributions._symmetric_generalized_hyperbolic import (
    sym_gen_hyperbolic_copula_gen)
from sklarpy.copulas._distributions._symmetric_hyperbolics import (
    sym_marginal_hyperbolic_copula_gen, sym_hyperbolic_copula_gen,
    sym_nig_copula_gen)
from sklarpy.copulas._distributions._archimedean import (
    clayton_copula_gen, gumbel_copula_gen, frank_copula_gen)

from sklarpy.multivariate import (
    mvt_normal, mvt_gaussian_kde, mvt_gh, mvt_mh, mvt_hyperbolic, mvt_nig,
    mvt_student_t, mvt_skewed_t, mvt_sgh, mvt_smh, mvt_shyperbolic, mvt_snig)

from sklarpy.multivariate.distributions import (
    mvt_clayton, mvt_gumbel, bvt_frank)

__all__ = ['gaussian_copula', 'gaussian_kde_copula', 'gh_copula', 'mh_copula',
           'hyperbolic_copula', 'nig_copula', 'skewed_t_copula',
           'student_t_copula', 'sgh_copula', 'smh_copula',
           'shyperbolic_copula', 'snig_copula', 'clayton_copula',
           'gumbel_copula', 'frank_copula']

###############################################################################
# Numerical/Non-Parametric
###############################################################################
gaussian_kde_copula: gaussian_kde_copula_gen = \
    gaussian_kde_copula_gen(
        name="gaussian_kde", mv_object=mvt_gaussian_kde)

###############################################################################
# Parametric
###############################################################################
gaussian_copula: gaussian_copula_gen = \
    gaussian_copula_gen(name="gaussian", mv_object=mvt_normal)

gh_copula: gen_hyperbolic_copula_gen = gen_hyperbolic_copula_gen(
    name="gh", mv_object=mvt_gh)

mh_copula: marginal_hyperbolic_copula_gen = marginal_hyperbolic_copula_gen(
    name="mh", mv_object=mvt_mh)

hyperbolic_copula: hyperbolic_copula_gen = hyperbolic_copula_gen(
    name="hyperbolic", mv_object=mvt_hyperbolic)

nig_copula: nig_copula_gen = nig_copula_gen(name="nig", mv_object=mvt_nig)

skewed_t_copula: skewed_t_copula_gen = skewed_t_copula_gen(
    name="skewed_t", mv_object=mvt_skewed_t)

student_t_copula: student_t_copula_gen = student_t_copula_gen(
    name="student_t", mv_object=mvt_student_t)

sgh_copula: sym_gen_hyperbolic_copula_gen = sym_gen_hyperbolic_copula_gen(
    name="sgh", mv_object=mvt_sgh)

smh_copula: sym_marginal_hyperbolic_copula_gen = \
    sym_marginal_hyperbolic_copula_gen(name="smh", mv_object=mvt_smh)

shyperbolic_copula: sym_hyperbolic_copula_gen = sym_hyperbolic_copula_gen(
    name="shyperbolic", mv_object=mvt_shyperbolic)

snig_copula: sym_nig_copula_gen = sym_nig_copula_gen(
    name="snig", mv_object=mvt_snig)

clayton_copula: clayton_copula_gen = \
    clayton_copula_gen(name="clayton", mv_object=mvt_clayton)

gumbel_copula: gumbel_copula_gen = \
    gumbel_copula_gen(name="gumbel", mv_object=mvt_gumbel)

frank_copula: frank_copula_gen = \
    frank_copula_gen(name="frank", mv_object=bvt_frank)
