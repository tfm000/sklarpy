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