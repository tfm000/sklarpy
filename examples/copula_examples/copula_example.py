# This file contains examples of how to use copula distributions in SklarPy.
# Here we use the skewed-t copula, though all methods and attributes are
# generalized.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# fitting a skewed-t copula to our generated data using
# Maximum Likelihood Estimation
from sklarpy.copulas import skewed_t_copula

fitted_copula = skewed_t_copula.fit(
    data=rvs_df, method='mle',
    univariate_fitter_options={'significant': False}, show_progress=True)

# prining our fitted parameters
print(fitted_copula.copula_params.to_dict)
print(fitted_copula.mdists)
print(fitted_copula.copula_params.cov)

# printing a summary of our fit
print(fitted_copula.summary)

# plotting our fit
fitted_copula.copula_pdf_plot(show=False)
fitted_copula.pdf_plot(show=False)
fitted_copula.mc_cdf_plot(show=False)
fitted_copula.marginal_pairplot(show=False)
plt.show()

# we can also save our fitted copula, as well as its parameters
fitted_copula.save()

# we can then easily load this object
from sklarpy import load

loaded_copula = load('skewed_t.pickle')
print(loaded_copula.summary)
