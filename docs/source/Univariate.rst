.. _univariate:

#########################
Univariate Distributions
#########################

This SklarPy package contains many different univariate distributions in addition to objects allowing for easy fitting.
With the exception of a handful of distributions, all univariate distribution objects are wrappers of scipy.stats univariate distributions, with added functionalities for plotting, fitting and saving.
This means that the distributions available in SklarPy are the same as those available in your installed version of scipy.

There is also the UnivariateFitter object, which allows for easy fitting of univariate distributions to data and for determining the best / statistically significant distribution(s).

Why is my interpreter unable to find univariate distributions?
--------------------------------------------------------------

If you try::

    from sklarpy.univariate import normal

You will likely find that your interpreter flags an error along the lines of "cannot find reference 'normal' in __init__.py".
Do not worry, this is to be expected an a side effect of the dynamic way SklarPy univariate distributions are created from scipy.stats distributions.
At runtime, *your code will work without any errors*!

But how do I know which distributions are available?
----------------------------------------------------
Good question! You can use the following code to print out a list of all available univariate distributions::

    from sklarpy.univariate import distributions_map
    print(distributions_map)

For scipy version 1.11.4 you should get an output along the lines of:

.. code-block:: text

    {'all': ('ksone', 'kstwo', 'kstwobign', 'alpha', 'anglit', 'arcsine', 'beta', 'betaprime', 'bradford', 'burr', 'burr12', 'fisk', 'cauchy', 'chi', 'chi2', 'cosine', 'dgamma', 'dweibull', 'expon', 'exponnorm', 'exponweib', 'exponpow', 'fatiguelife', 'foldcauchy', 'f', 'foldnorm', 'weibull_min', 'truncweibull_min', 'weibull_max', 'genlogistic', 'genpareto', 'genexpon', 'genextreme', 'gamma', 'erlang', 'gengamma', 'genhalflogistic', 'genhyperbolic', 'gompertz', 'gumbel_r', 'gumbel_l', 'halfcauchy', 'halflogistic', 'halfnorm', 'hypsecant', 'gausshyper', 'invgamma', 'invgauss', 'geninvgauss', 'norminvgauss', 'invweibull', 'johnsonsb', 'johnsonsu', 'laplace', 'laplace_asymmetric', 'levy', 'levy_l', 'logistic', 'loggamma', 'loglaplace', 'lognorm', 'gibrat', 'maxwell', 'mielke', 'kappa4', 'kappa3', 'moyal', 'nakagami', 'ncx2', 'ncf', 'nct', 'pareto', 'lomax', 'pearson3', 'powerlaw', 'powerlognorm', 'powernorm', 'rdist', 'rayleigh', 'loguniform', 'reciprocal', 'rice', 'recipinvgauss', 'semicircular', 'skewcauchy', 'skewnorm', 'trapezoid', 'trapz', 'triang', 'truncexpon', 'truncnorm', 'truncpareto', 'tukeylambda', 'uniform', 'vonmises', 'vonmises_line', 'wald', 'wrapcauchy', 'gennorm', 'halfgennorm', 'crystalball', 'argus', 'studentized_range', 'rel_breitwigner', 'gh', 'gig', 'ig', 'normal', 'student_t', 'gaussian_kde', 'empirical', 'poisson', 'planck', 'discrete_laplace', 'discrete_uniform', 'geometric', 'discrete_empirical'), 'all continuous': ('ksone', 'kstwo', 'kstwobign', 'alpha', 'anglit', 'arcsine', 'beta', 'betaprime', 'bradford', 'burr', 'burr12', 'fisk', 'cauchy', 'chi', 'chi2', 'cosine', 'dgamma', 'dweibull', 'expon', 'exponnorm', 'exponweib', 'exponpow', 'fatiguelife', 'foldcauchy', 'f', 'foldnorm', 'weibull_min', 'truncweibull_min', 'weibull_max', 'genlogistic', 'genpareto', 'genexpon', 'genextreme', 'gamma', 'erlang', 'gengamma', 'genhalflogistic', 'genhyperbolic', 'gompertz', 'gumbel_r', 'gumbel_l', 'halfcauchy', 'halflogistic', 'halfnorm', 'hypsecant', 'gausshyper', 'invgamma', 'invgauss', 'geninvgauss', 'norminvgauss', 'invweibull', 'johnsonsb', 'johnsonsu', 'laplace', 'laplace_asymmetric', 'levy', 'levy_l', 'logistic', 'loggamma', 'loglaplace', 'lognorm', 'gibrat', 'maxwell', 'mielke', 'kappa4', 'kappa3', 'moyal', 'nakagami', 'ncx2', 'ncf', 'nct', 'pareto', 'lomax', 'pearson3', 'powerlaw', 'powerlognorm', 'powernorm', 'rdist', 'rayleigh', 'loguniform', 'reciprocal', 'rice', 'recipinvgauss', 'semicircular', 'skewcauchy', 'skewnorm', 'trapezoid', 'trapz', 'triang', 'truncexpon', 'truncnorm', 'truncpareto', 'tukeylambda', 'uniform', 'vonmises', 'vonmises_line', 'wald', 'wrapcauchy', 'gennorm', 'halfgennorm', 'crystalball', 'argus', 'studentized_range', 'rel_breitwigner', 'gh', 'gig', 'ig', 'normal', 'student_t', 'gaussian_kde', 'empirical'), 'all discrete': ('poisson', 'planck', 'discrete_laplace', 'discrete_uniform', 'geometric', 'discrete_empirical'), 'all common': ('cauchy', 'chi2', 'expon', 'gamma', 'lognorm', 'powerlaw', 'rayleigh', 'uniform', 'discrete_laplace', 'discrete_uniform', 'geometric', 'poisson'), 'all multimodal': ('arcsine', 'beta'), 'all parametric': ('ksone', 'kstwo', 'kstwobign', 'alpha', 'anglit', 'arcsine', 'beta', 'betaprime', 'bradford', 'burr', 'burr12', 'fisk', 'cauchy', 'chi', 'chi2', 'cosine', 'dgamma', 'dweibull', 'expon', 'exponnorm', 'exponweib', 'exponpow', 'fatiguelife', 'foldcauchy', 'f', 'foldnorm', 'weibull_min', 'truncweibull_min', 'weibull_max', 'genlogistic', 'genpareto', 'genexpon', 'genextreme', 'gamma', 'erlang', 'gengamma', 'genhalflogistic', 'genhyperbolic', 'gompertz', 'gumbel_r', 'gumbel_l', 'halfcauchy', 'halflogistic', 'halfnorm', 'hypsecant', 'gausshyper', 'invgamma', 'invgauss', 'geninvgauss', 'norminvgauss', 'invweibull', 'johnsonsb', 'johnsonsu', 'laplace', 'laplace_asymmetric', 'levy', 'levy_l', 'logistic', 'loggamma', 'loglaplace', 'lognorm', 'gibrat', 'maxwell', 'mielke', 'kappa4', 'kappa3', 'moyal', 'nakagami', 'ncx2', 'ncf', 'nct', 'pareto', 'lomax', 'pearson3', 'powerlaw', 'powerlognorm', 'powernorm', 'rdist', 'rayleigh', 'loguniform', 'reciprocal', 'rice', 'recipinvgauss', 'semicircular', 'skewcauchy', 'skewnorm', 'trapezoid', 'trapz', 'triang', 'truncexpon', 'truncnorm', 'truncpareto', 'tukeylambda', 'uniform', 'vonmises', 'vonmises_line', 'wald', 'wrapcauchy', 'gennorm', 'halfgennorm', 'crystalball', 'argus', 'studentized_range', 'rel_breitwigner', 'gh', 'gig', 'ig', 'normal', 'student_t', 'poisson', 'planck', 'discrete_laplace', 'discrete_uniform', 'geometric'), 'all numerical': ('gaussian_kde', 'empirical', 'discrete_empirical'), 'all continuous parametric': ('ksone', 'kstwo', 'kstwobign', 'alpha', 'anglit', 'arcsine', 'beta', 'betaprime', 'bradford', 'burr', 'burr12', 'fisk', 'cauchy', 'chi', 'chi2', 'cosine', 'dgamma', 'dweibull', 'expon', 'exponnorm', 'exponweib', 'exponpow', 'fatiguelife', 'foldcauchy', 'f', 'foldnorm', 'weibull_min', 'truncweibull_min', 'weibull_max', 'genlogistic', 'genpareto', 'genexpon', 'genextreme', 'gamma', 'erlang', 'gengamma', 'genhalflogistic', 'genhyperbolic', 'gompertz', 'gumbel_r', 'gumbel_l', 'halfcauchy', 'halflogistic', 'halfnorm', 'hypsecant', 'gausshyper', 'invgamma', 'invgauss', 'geninvgauss', 'norminvgauss', 'invweibull', 'johnsonsb', 'johnsonsu', 'laplace', 'laplace_asymmetric', 'levy', 'levy_l', 'logistic', 'loggamma', 'loglaplace', 'lognorm', 'gibrat', 'maxwell', 'mielke', 'kappa4', 'kappa3', 'moyal', 'nakagami', 'ncx2', 'ncf', 'nct', 'pareto', 'lomax', 'pearson3', 'powerlaw', 'powerlognorm', 'powernorm', 'rdist', 'rayleigh', 'loguniform', 'reciprocal', 'rice', 'recipinvgauss', 'semicircular', 'skewcauchy', 'skewnorm', 'trapezoid', 'trapz', 'triang', 'truncexpon', 'truncnorm', 'truncpareto', 'tukeylambda', 'uniform', 'vonmises', 'vonmises_line', 'wald', 'wrapcauchy', 'gennorm', 'halfgennorm', 'crystalball', 'argus', 'studentized_range', 'rel_breitwigner', 'gh', 'gig', 'ig', 'normal', 'student_t'), 'all discrete parametric': ('poisson', 'planck', 'discrete_laplace', 'discrete_uniform', 'geometric'), 'all continuous numerical': ('gaussian_kde', 'empirical'), 'all discrete numerical': ('discrete_empirical',), 'common continuous': ('cauchy', 'chi2', 'expon', 'gamma', 'lognorm', 'powerlaw', 'rayleigh', 'uniform'), 'common discrete': ('discrete_laplace', 'discrete_uniform', 'geometric', 'poisson'), 'continuous multimodal': ('arcsine', 'beta'), 'discrete multimodal': ()}

So you have a lot to choose from!

.. automodule:: sklarpy.univariate.univariate_fitter
    :members:
    :exclude-members: UnivariateFitter

    .. autoclass:: UnivariateFitter
        :members:

    .. automethod:: __init__

    .. automethod:: fit

    .. caution::

        If 'use_processpoolexecutor' is set to True, the UnivariateFitter object will use the ProcessPoolExecutor to parallelize the fitting process. However, if the code is ran outside 'if __name__ == "__main__":', you may receive a runtime error.

    .. automethod:: get_summary

    .. automethod:: get_best

    .. automethod:: plot

    .. automethod:: fitted_distributions


.. automodule:: sklarpy.univariate._prefit_dists
    :members:
    :exclude-members: PreFitUnivariateBase, PreFitNumericalUnivariateBase


.. automodule:: sklarpy.univariate._fitted_dists
    :members:
    :exclude-members: FittedUnivariateBase

Continuous Example
---------------------
Here we use the normal and gamma distributions, though all methods and attributes are generalized.::

    import numpy as np
    import pandas as pd

    # generating random variables
    from sklarpy.univariate import normal

    num_generate: int = 1000

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

.. code-block:: text:

    (9754.44976841112, -411.8704014945831, 0.042211986922603084)

We can also print a summary of our fit::

    summary: pd.DataFrame = fitted_gamma.summary
    print(summary)

.. code-block:: text:

                                                                  summary
    Parametric/Non-Parametric                                  Parametric
    Discrete/Continuous                                        continuous
    Distribution                                                    gamma
    #Params                                                             3
    param0                                                    9754.449768
    param1                                                    -411.870401
    param2                                                       0.042212
    Support                                     (-411.8704014945831, inf)
    Fitted Domain                 (-20.13664960054484, 17.86802768972715)
    Cramér-von Mises statistic                                   3.411862
    Cramér-von Mises p-value                                          0.0
    Cramér-von Mises @ 10%                                          False
    Cramér-von Mises @ 5%                                           False
    Cramér-von Mises @ 1%                                           False
    Kolmogorov-Smirnov statistic                                 0.094371
    Kolmogorov-Smirnov p-value                                        0.0
    Kolmogorov-Smirnov @ 10%                                        False
    Kolmogorov-Smirnov @ 5%                                         False
    Kolmogorov-Smirnov @ 1%                                         False
    Likelihood                                                        0.0
    Log-Likelihood                                           -2846.513514
    AIC                                                       5699.027028
    BIC                                                       5713.750294
    Sum of Squared Error                                        12.319097
    #Fitted Data Points                                              1000

And plot our fitted distribution::

    fitted_gamma.plot()

.. image:: media/univariate_continuous_example_figure1.png
    :alt: gamma plot
    :align: center

And save::

    fitted_gamma.save()


We can then easily reload our saved model::

    from sklarpy import load

    loaded_fitted_gamma = load('gamma.pickle')

