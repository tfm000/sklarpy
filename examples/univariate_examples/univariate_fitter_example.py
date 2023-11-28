# This file contains examples of how to use the UnivariateFitter object to
# find the best probability distribution to represent your data.
import numpy as np

# generating random variables
from sklarpy.univariate import normal

num_generate: int = 10000
# generating a 1d array of N(1, 1) random variables
normal_rvs1: np.ndarray = normal.rvs((num_generate,), (1, 1))
# generating a 1d array of N(2, 3) random variables
normal_rvs2: np.ndarray = normal.rvs((num_generate,), (0, 3))
rvs = normal_rvs1 * normal_rvs2

# applying UnivariateFitter to our product of normal random variables
from sklarpy.univariate import UnivariateFitter

ufitter: UnivariateFitter = UnivariateFitter(rvs)
ufitter.fit()

# printing out the summary of our fits
from sklarpy import print_full
print_full()

print(ufitter.get_summary())

# finding our best fit
best_fit = ufitter.get_best(significant=False)
print(best_fit.summary)
best_fit.plot()

# we can also save our UnivariateFitter object
ufitter.save()

# we can then easily reload this
from sklarpy import load

loaded_ufitter = load('UnivariateFitter.pickle')
loaded_best_fit = loaded_ufitter.get_best(significant=False)
print(loaded_best_fit.summary)
