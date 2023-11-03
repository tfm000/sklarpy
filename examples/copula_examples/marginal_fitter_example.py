# This file contains examples of how to use the MarginalFitter object to
# find the marginal distributions of your multivariate dataset
import numpy as np
import pandas as pd

# specifying the parameters of the multivariate normal distribution we are
# sampling from
num_generate: int = 1000
my_mu: np.ndarray = np.array([33, 44], dtype=float)
my_corr: np.ndarray = np.array([[1, 0.7], [0.7, 1]], dtype=float)
my_sig: np.ndarray = np.array([1.3, 2.5])
my_cov: np.ndarray = np.diag(my_sig) @ my_corr @ np.diag(my_sig)
my_mvn_params: tuple = (my_mu, my_cov)

# generating multivariate random normal variables
from sklarpy.multivariate import multivariate_normal

rvs: np.ndarray = multivariate_normal.rvs(num_generate, my_mvn_params)
rvs_df: pd.DataFrame = pd.DataFrame(rvs, columns=['Wife Age', 'Husband Age'
                                                  ], dtype=float)

# applying MarginalFitter to our random variables
from sklarpy.copulas import MarginalFitter

mfitter: MarginalFitter = MarginalFitter(rvs_df)
mfitter.fit({'pvalue': 0.01})

# printing out a summary of our fits
print(mfitter.summary)
print(mfitter.marginals)

# calculating the marginal cdf values
mcdf_values: pd.DataFrame = mfitter.marginal_cdfs()
print(mcdf_values)

# we can also go beyond the bivariate setting
data: np.ndarray = np.full((num_generate, 10), np.NaN)
data[:, :2] = np.random.poisson(4, (num_generate, 2))
data[:, 2] = np.random.randint(-5, 5, (num_generate,))
data[:, 3] = data[:, :2].sum(axis=1)
data[:, 4] = data[:, 0] + data[:, 3]
data[:, 5] = np.random.normal(4, 2, (num_generate,))
data[:, 6] = np.random.gamma(2, 1, (num_generate,))
data[:, 7:9] = np.random.standard_t(3, (num_generate, 2))
data[:, 9] = np.random.uniform(0, 1, (num_generate,))

mfitter2: MarginalFitter = MarginalFitter(data).fit()
print(mfitter2.marginals)

# we can plot our marginal distributions
mfitter2.pairplot()
