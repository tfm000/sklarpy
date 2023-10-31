# This file contains examples of how to use continuous, univariate
# distributions in SklarPy.
# Here we use the normal and gamma distributions, though all methods and
# attributes are generalized.
import numpy as np
import pandas as pd

# generating random variables
from sklarpy.univariate import normal

num_generate: int = 100

# generating a 1d array of N(1, 1) random variables
normal_rvs1: np.ndarray = normal.rvs((num_generate,), (1, 1))
# generating a 1d array of N(2, 3) random variables
normal_rvs2: np.ndarray = normal.rvs((num_generate,), (0, 3))
rvs = normal_rvs1 * normal_rvs2

# fitting a gamma distribution to our product of normal random variables
from sklarpy.univariate import gamma

fitted_gamma = gamma.fit(rvs)

# we can easily retrieve the fitted parameters
fitted_gamma_params: tuple = fitted_gamma.params
print(fitted_gamma_params)

# we can also print a summary of our fit
summary: pd.DataFrame = fitted_gamma.summary
print(summary)

# we can plot our fit
fitted_gamma.plot()

# and save our fitted model
fitted_gamma.save()

# we can then easily reload our saved model
from sklarpy import load

loaded_fitted_gamma = load('gamma.pickle')

# which we can quickly verify is the same as the above
print(loaded_fitted_gamma.params)

# we can also fit a distribution to a tuple of its parameters, without any data
normal_params: tuple = (0, 3)
fitted_normal = normal.fit(params=normal_params)
print(fitted_normal.summary)
fitted_normal.plot()
