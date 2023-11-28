.. _univariate:

#########################
Univariate Distributions
#########################

This SklarPy package contains many different univariate distributions in addition to objects allowing for easy fitting.
With the exception of a handful of distributions, all univariate distribution objects are wrappers of scipy.stats univariate distributions, with added functionalities for plotting, fitting and saving.

There is also the UnivariateFitter object, which allows for easy fitting of univariate distributions to data and for determining the best / statistically significant distribution(s).

.. automodule:: sklarpy.univariate
    :members:
    :exclude-members: UnivariateFitter

    .. autoclass:: UnivariateFitter
        :members:

    .. automethod:: __init__

    .. automethod:: fit

    .. caution::

        If 'use_processpoolexecutor' is set to True, the UnivariateFitter object will use the ProcessPoolExecutor to parallelize the fitting process. However, if the code is ran outside 'if __name__ == "__main__":', you may receive a runtime error.

    .. automethod:: plot

    .. automethod:: fitted_distributions

