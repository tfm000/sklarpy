.. _misc:

####################
Miscellaneous Tools
####################

This SklarPy package contains functions / objects which are both implemented across SklarPy and also intended for user use.

CorrelationMatrix
------------------
CorrelationMatrix is a SklarPy class which allows the user to estimate correlation and covariance matrices using a number of different estimators.

This code is inspired by the methods described by Xu, Brin (2016) and implements the following estimators:

- pearson
- spearman
- kendall
- pp-kendall
- rm-pearson
- rm-spearman
- rm-kendall
- rm-pp-kendall
- laloux-pearson
- laloux-spearman
- laloux-kendall
- laloux-pp-kendall

rm stands for the technique described by Rousseeuw and Molenberghs (1993) and laloux for that by Laloux et al. (2000).

The corr method allows you to calculate correlation matrices, whilst cov allows you to calculate covariance matrices.

debye
-----
This function allows the user to easily evaluate any member of the Debye function family.

gradient_1d
------------
This function allows the user to calculate the numerical first derivative / gradient of a given 1-d function.

kv
---
This class allows the user to easily evaluate the Modified Bessel function of the 2nd kind, in addition to its log-values.
Limiting cases of the family parameter, v, and value, z, are also implemented.

CorrelationMatrix Example
--------------------------

Here we calculate both the covariance and correlation matrix estimators::

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

Calculating PP-Kendall Correlation Matrix with Laloux's adjustments::

    corr_estimator: np.ndarray = cmatrix.corr(method='laloux_pp_kendall')
    print(corr_estimator)

.. code-block:: text

    [[ 1.         -0.53750912]
     [-0.53750912  1.        ]]

Calculating Spearman's Covariance Matrix::

    cov_estimator: np.ndarray = cmatrix.cov(method='spearman')
    print(cov_estimator)

.. code-block:: text

    [[ 3.02797258 -2.68535942]
     [-2.68535942  8.68778502]]