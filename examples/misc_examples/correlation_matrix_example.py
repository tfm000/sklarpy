# This file contains examples of how to use the CorrelationMatrix class in
# SklarPy. Here we calculate both the covariance and correlation matrix
# estimators.
import numpy as np
import pandas as pd

# specifying the parameters of the multivariate hyperbolic distribution we are
# generating from
my_loc = np.array([1, -3], dtype=float)
my_shape = np.array([[1, 0.7], [0.7, 1]], dtype=float)
my_chi = 1.7
my_psi = 4.5
my_gamma = np.array([2.3, -4.3], dtype=float)
my_params = (my_chi, my_psi, my_loc, my_shape, my_gamma)

# generating multivariate hyperbolic random variables
from sklarpy.multivariate import mvt_hyperbolic

num_generate: int = 1000
rvs: np.ndarray = mvt_hyperbolic.rvs(num_generate, my_params)
rvs_df: pd.DataFrame = pd.DataFrame(rvs, columns=['Process A', 'Process B'],
                                    dtype=float)

# calculating covariance matrix and correlation matrix estimators
from sklarpy.misc import CorrelationMatrix

cmatrix: CorrelationMatrix = CorrelationMatrix(rvs_df)
corr_estimator: np.ndarray = cmatrix.corr(method='laloux_pp_kendall')
cov_estimator: np.ndarray = cmatrix.cov(method='spearman')

# printing our estimator values
print('Correlation Matrix Estimator\n----------------------------')
print(corr_estimator)

print('\nCovariance Matrix Estimator\n---------------------------')
print(cov_estimator)
