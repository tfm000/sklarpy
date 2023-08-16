from sklarpy.copulas._distributions._gaussian import gaussian_copula_gen
from sklarpy.copulas._distributions._gaussian_kde import gaussian_kde_copula_gen

from sklarpy.copulas._distributions._student_t import student_t_copula_gen

from sklarpy.multivariate import multivariate_normal
from sklarpy.multivariate import multivariate_gaussian_kde

from sklarpy.multivariate import multivariate_student_t

__all__ = ['gaussian_copula', 'gaussian_kde_copula', 'student_t_copula']

########################################################################################################################
# Numerical/Non-Parametric
########################################################################################################################


########################################################################################################################
# Parametric
########################################################################################################################
gaussian_copula: gaussian_copula_gen = gaussian_copula_gen(name="gaussian", mv_object=multivariate_normal)
gaussian_kde_copula: gaussian_kde_copula_gen = gaussian_kde_copula_gen(name="gaussian_kde", mv_object=multivariate_gaussian_kde)

student_t_copula: student_t_copula_gen = student_t_copula_gen(name="student_t", mv_object=multivariate_student_t)
