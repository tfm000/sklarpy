# This file contains examples of how to use multivariate distributions in
# SklarPy.
# Here we use the multivariate normal and multivariate symmetric hyperbolic
# distributions, though all methods and attributes are generalized.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# specifying the parameters of the multivariate normal distribution we are
# sampling from
my_mu: np.ndarray = np.array([33, 44], dtype=float)
my_corr: np.ndarray = np.array([[1, 0.7], [0.7, 1]], dtype=float)
my_sig: np.ndarray = np.array([1.3, 2.5])
my_cov: np.ndarray = np.diag(my_sig) @ my_corr @ np.diag(my_sig)
my_mvn_params: tuple = (my_mu, my_cov)

# generating multivariate random normal variables
from sklarpy.multivariate import multivariate_normal

rvs: np.ndarray = multivariate_normal.rvs(1000, my_mvn_params)
rvs_df: pd.DataFrame = pd.DataFrame(rvs, columns=['Wife Age', 'Husband Age'],
                                    dtype=float)

# fitting a symmetric hyperbolic dist to our generated data using
# Maximum Likelihood Estimation
from sklarpy.multivariate import multivariate_sym_hyperbolic

fitted_msh = multivariate_sym_hyperbolic.fit(rvs_df, method='mle',
                                             show_progress=True)

# printing our fitted parameters
print(fitted_msh.params.to_dict)
print(fitted_msh.params.cov)

# printing a summary of our fit
print(fitted_msh.summary)

# can plot
fitted_msh.pdf_plot(show=False)
fitted_msh.mc_cdf_plot(show=False)
fitted_msh.marginal_pairplot(show=False)
plt.show()

# can also save parameters
fitted_msh.params.save()

# we can then easily load these parameters and use them to fit another
# distribution of the same type
from sklarpy import load

loaded_msh_params = load('multivariate_sym_hyperbolic.pickle')
param_fitted_msh = multivariate_sym_hyperbolic.fit(params=loaded_msh_params)
print(param_fitted_msh.params.to_dict)
param_fitted_msh.pdf_plot(axes_names=rvs_df.columns)
