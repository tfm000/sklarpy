# This file contains examples of how to use discrete, univariate distributions in SklarPy.
# These work in exactly the same way as continuous distributions, as shown below.
import numpy as np
import pandas as pd

# generating random variables
from sklarpy.univariate import poisson

num_generate: int = 10000
poisson_rvs: np.ndarray = poisson.rvs((num_generate, ), (4,))
rvs_df: pd.DataFrame = pd.DataFrame(poisson_rvs, columns=['rvs'], dtype=int)

# fitting a poisson distribution to a dataframe of rvs
fitted_poisson = poisson.fit(rvs_df)

# we can easily retrieve the fitted parameters
fitted_poisson_params: tuple = fitted_poisson.params
print(fitted_poisson_params)

# we can also print a summary of our fit
summary: pd.DataFrame = fitted_poisson.summary
print(summary)

# we can plot our fit
fitted_poisson.plot()

# and save our fitted model
fitted_poisson.save()

# we can then easily reload our saved model
from sklarpy import load

loaded_fitted_poisson = load('poisson.pickle')

# which we can quickly verify is the same as the above
print(loaded_fitted_poisson.param)
