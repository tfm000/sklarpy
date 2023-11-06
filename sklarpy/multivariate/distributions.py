# Contains multivariate probability distributions
import numpy as np
import sys

from sklarpy.multivariate._distributions._gaussian_kde import \
    multivariate_gaussian_kde_gen
from sklarpy.multivariate._distributions._generalized_hyperbolic import \
    multivariate_gen_hyperbolic_gen
from sklarpy.multivariate._distributions._hyperbolics import \
    multivariate_hyperbolic_gen, multivariate_nig_gen, \
    multivariate_marginal_hyperbolic_gen
from sklarpy.multivariate._distributions._normal import multivariate_normal_gen
from sklarpy.multivariate._distributions._skewed_t import \
    multivariate_skewed_t_gen
from sklarpy.multivariate._distributions._student_t import \
    multivariate_student_t_gen
from sklarpy.multivariate._distributions._symmetric_generalized_hyperbolic \
    import multivariate_sym_gen_hyperbolic_gen
from sklarpy.multivariate._distributions._symmetric_hyperbolics import \
    multivariate_sym_hyperbolic_gen, multivariate_sym_nig_gen, \
    multivariate_sym_marginal_hyperbolic_gen
from sklarpy.multivariate._distributions._archimedean import \
    multivariate_clayton_gen, multivariate_gumbel_gen, bivariate_frank_gen

from sklarpy.multivariate._params._gaussian_kde import \
    MvtGaussianKDEParams
from sklarpy.multivariate._params._generalized_hyperbolic import \
    MvtGHParams
from sklarpy.multivariate._params._hyperbolics import \
    MvtMHParams, MvtHyperbolicParams, \
    MvtNIGParams
from sklarpy.multivariate._params._normal import MvtNormalParams
from sklarpy.multivariate._params._skewed_t import MvtSkewedTParams
from sklarpy.multivariate._params._student_t import MvtStudentTParams
from sklarpy.multivariate._params._symmetric_generalized_hyperbolic import \
    MvtSGHParams
from sklarpy.multivariate._params._symmetric_hyperbolics import \
    MvtSMHParams, MvtSHyperbolicParams, \
    MvtSNIGParams
from sklarpy.multivariate._params._archimedean import (
    MvtClaytonParams, MvtGumbelParams, BvtFrankParams)

__all__ = [
    'mvt_gaussian_kde',
    'mvt_gh',
    'mvt_mh',
    'mvt_hyperbolic',
    'mvt_nig',
    'mvt_normal',
    'mvt_skewed_t',
    'mvt_student_t',
    'mvt_sgh',
    'mvt_smh',
    'mvt_shyperbolic',
    'mvt_snig'
]


###############################################################################
# Numerical/Non-Parametric
###############################################################################
mvt_gaussian_kde: multivariate_gaussian_kde_gen = \
    multivariate_gaussian_kde_gen(
        name='mvt_gaussian_kde',
        params_obj=MvtGaussianKDEParams,
        num_params=1, max_num_variables=np.inf
    )

###############################################################################
# Continuous (Parametric)
###############################################################################
mvt_gh: multivariate_gen_hyperbolic_gen = multivariate_gen_hyperbolic_gen(
    name="mvt_gh",
    params_obj=MvtGHParams,
    num_params=6, max_num_variables=np.inf
    )

mvt_mh: multivariate_marginal_hyperbolic_gen = (
    multivariate_marginal_hyperbolic_gen(
        name='mvt_mh',
        params_obj=MvtMHParams,
        num_params=5, max_num_variables=np.inf
    ))

mvt_hyperbolic: multivariate_hyperbolic_gen = \
    multivariate_hyperbolic_gen(
        name='mvt_hyperbolic',
        params_obj=MvtHyperbolicParams, num_params=5,
        max_num_variables=np.inf
    )

mvt_nig: multivariate_nig_gen = multivariate_nig_gen(
    name='mvt_nig', params_obj=MvtNIGParams,
    num_params=5, max_num_variables=np.inf
)

mvt_normal: multivariate_normal_gen = multivariate_normal_gen(
    name="mvt_normal", params_obj=MvtNormalParams,
    num_params=2, max_num_variables=np.inf
)

mvt_student_t: multivariate_student_t_gen = multivariate_student_t_gen(
    name="mvt_student_t", params_obj=MvtStudentTParams,
    num_params=3, max_num_variables=np.inf
    )

mvt_skewed_t: multivariate_skewed_t_gen = multivariate_skewed_t_gen(
    name='mvt_skewed_t', params_obj=MvtSkewedTParams,
    num_params=4, max_num_variables=np.inf, mvt_t=mvt_student_t
)

mvt_sgh: multivariate_sym_gen_hyperbolic_gen = (
    multivariate_sym_gen_hyperbolic_gen(
        name="mvt_sgh", params_obj=MvtSGHParams,
        num_params=5, max_num_variables=np.inf
    ))

mvt_smh: multivariate_sym_marginal_hyperbolic_gen = (
    multivariate_sym_marginal_hyperbolic_gen(
        name='mvt_smh', params_obj=MvtSMHParams,
        num_params=4, max_num_variables=np.inf
    ))

mvt_shyperbolic: multivariate_sym_hyperbolic_gen = (
    multivariate_sym_hyperbolic_gen(
        name='mvt_shyperbolic', params_obj=MvtSHyperbolicParams,
        num_params=4, max_num_variables=np.inf
    ))

mvt_snig: multivariate_sym_nig_gen = multivariate_sym_nig_gen(
    name='mvt_snig', params_obj=MvtSNIGParams,
    num_params=4, max_num_variables=np.inf
)

###############################################################################
# Copula Only
###############################################################################
mvt_clayton: multivariate_clayton_gen = multivariate_clayton_gen(
    name='mvt_clayton', params_obj=MvtClaytonParams,
    num_params=2, max_num_variables=np.inf
)

mvt_gumbel: multivariate_gumbel_gen = multivariate_gumbel_gen(
    name='mvt_gumbel', params_obj=MvtGumbelParams,
    num_params=2, max_num_variables=sys.getrecursionlimit()
)

bvt_frank: bivariate_frank_gen = bivariate_frank_gen(
    name='bvt_frank', params_obj=BvtFrankParams,
    num_params=2, max_num_variables=2
)
