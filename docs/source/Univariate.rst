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
Do not worry, this is to be expected as a side effect of the dynamic way SklarPy univariate distributions are created from scipy.stats distributions.
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

Name differences between SklarPy and SciPy
-------------------------------------------
Whilst we have generally kept most of the distribution names consistent with SciPy, there are a few notable exceptions.
These are:

.. csv-table:: Distribution Name Discrepancies
    :file: univariate_table.csv
    :header-rows: 1

PreFitUnivariateBase
---------------------
This class and its subclasses contain the following methods / functions:

- pdf (probability density function)
- cdf (cumulative distribution function)
- ppf (percent point function / cumulative inverse function)
- support
- ppf_approx (approximate ppf)
- cdf_approx (approximate cdf)
- rvs (random variate generator / sampler)
- logpdf (log of the probability density function)
- likelihood (likelihood function)
- loglikelihood (log of the likelihood function)
- aic (Akaike information criterion)
- bic (Bayesian information criterion)
- sse (Sum of squared errors)
- gof (goodness of fit)
- plot (plotting)
- fit (fitting the distribution to data)

Many / all of these methods take params as an argument. This is a tuple containing the parameters of the associated scipy.stats distribution object.

ppf_approx and cdf_approx are approximations of the ppf and cdf functions respectively, which may be useful for distributions where the cdf and therefore ppf functions require numerical integration to evaluate.

FittedUnivariateBase
---------------------
This class is the fitted version of PreFitUnivariateBase's subclasses.
It implements the same methods as PreFitUnivariateBase, but does not require params as an argument.
It also implements the following additional methods and attributes:

- summary (summary of the distribution fit)
- params (the fitted parameters)
- fitted domain (the domain over which the distribution is fitted)
- fitted_num_data_points (the number of data points used to fit the distribution)
- save (save the fitted distribution to a pickle file)

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
Here we use the normal and gamma distributions, though all methods and attributes are generalized::

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

.. code-block:: text

    (9754.44976841112, -411.8704014945831, 0.042211986922603084)

We can also print a summary of our fit::

    summary: pd.DataFrame = fitted_gamma.summary
    print(summary)

.. code-block:: text

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

.. image:: https://github.com/tfm000/sklarpy/blob/main/media/univariate_continuous_example_figure1.png?raw=true
    :alt: gamma plot
    :align: center

And save::

    fitted_gamma.save()


We can then easily reload our saved model::

    from sklarpy import load

    loaded_fitted_gamma = load('gamma.pickle')



Discrete Example
---------------------
Here we use the poisson distribution, though all methods and attributes are generalized.
We see this works in exactly the same way as continuous distributions.::

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

.. code-block:: text

    (3.992,)

We can also print a summary of our fit::

    summary: pd.DataFrame = fitted_poisson.summary
    print(summary)

.. code-block:: text

                                   summary
    Parametric/Non-Parametric   Parametric
    Discrete/Continuous           discrete
    Distribution                   poisson
    #Params                              1
    param0                           3.985
    Support                       (0, inf)
    Fitted Domain                  (0, 12)
    chi-square statistic          7.059903
    chi-square p-value                 1.0
    chi-square @ 10%                  True
    chi-square @ 5%                   True
    chi-square @ 1%                   True
    Likelihood                         0.0
    Log-Likelihood            -2100.955867
    AIC                        4203.911734
    BIC                        4208.819489
    Sum of Squared Error          0.044802
    #Fitted Data Points               1000

And plot our fitted distribution::

    fitted_poisson.plot()

.. image:: https://github.com/tfm000/sklarpy/blob/main/media/univariate_discrete_example_figure1.png?raw=true
    :alt: poisson plot
    :align: center

And save::

    fitted_poisson.save()


We can then easily reload our saved model::

    from sklarpy import load

    loaded_fitted_poisson = load('poisson.pickle')

UnivariateFitter Example
-------------------------
Here we use the UnivariateFitter object to fit a distribution to a dataset::

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

.. code-block:: text

             Parametric/Non-Parametric Discrete/Continuous Distribution #Params      param0      param1      param2                                    Support                              Fitted Domain Cramér-von Mises statistic Cramér-von Mises p-value Cramér-von Mises @ 10% Cramér-von Mises @ 5% Cramér-von Mises @ 1% Kolmogorov-Smirnov statistic Kolmogorov-Smirnov p-value Kolmogorov-Smirnov @ 10% Kolmogorov-Smirnov @ 5% Kolmogorov-Smirnov @ 1% Likelihood Log-Likelihood          AIC          BIC Sum of Squared Error #Fitted Data Points
    chi2                    Parametric          continuous         chi2       3  448.683161  -68.423622     0.15222                  (-68.42362151895298, inf)  (-24.241200503425766, 21.971575538054054)                   3.955007                      0.0                  False                 False                 False                     0.099469                        0.0                    False                   False                   False        0.0   -2916.834582  5839.669164   5854.39243             12.84073                1000
    powerlaw                Parametric          continuous     powerlaw       3    1.485383  -24.284621   46.256197    (-24.28462141839885, 21.97157553805406)  (-24.241200503425766, 21.971575538054054)                  53.515366                      0.0                  False                 False                 False                     0.393459                        0.0                    False                   False                   False        0.0   -3765.295723  7536.591446  7551.314712              23.1246                1000
    cauchy                  Parametric          continuous       cauchy       2   -0.141171    1.744522         NaN                                (-inf, inf)  (-24.241200503425766, 21.971575538054054)                   0.223919                 0.225566                   True                  True                  True                      0.03747                   0.117619                     True                    True                    True        0.0   -2848.628202  5701.256403  5711.071914             7.057125                1000
    expon                   Parametric          continuous        expon       2  -24.241201   24.121323         NaN                 (-24.241200503425766, inf)  (-24.241200503425766, 21.971575538054054)                  68.507136                      0.0                  False                 False                 False                     0.465333                        0.0                    False                   False                   False        0.0    -4183.09624   8370.19248  8380.007991            24.962541                1000
    lognorm                 Parametric          continuous      lognorm       3    0.024195 -185.928209  185.754474                 (-185.92820884247777, inf)  (-24.241200503425766, 21.971575538054054)                   3.726801                      0.0                  False                 False                 False                     0.093801                        0.0                    False                   False                   False        0.0   -2910.878606  5827.757211  5842.480477            12.702458                1000
    rayleigh                Parametric          continuous     rayleigh       2  -24.268255   17.360527         NaN                    (-24.268254515672, inf)  (-24.241200503425766, 21.971575538054054)                  45.036613                      0.0                  False                 False                 False                     0.364332                        0.0                    False                   False                   False        0.0   -3548.608918  7101.217836  7111.033346            21.635708                1000
    gamma                   Parametric          continuous        gamma       3  614.186953 -110.593183    0.179857                  (-110.5931825074225, inf)  (-24.241200503425766, 21.971575538054054)                   3.612011                      0.0                  False                 False                 False                     0.094024                        0.0                    False                   False                   False        0.0   -2911.657958  5829.315916  5844.039182            12.618159                1000
    uniform                 Parametric          continuous      uniform       2  -24.241201   46.212776         NaN  (-24.241200503425766, 21.971575538054054)  (-24.241200503425766, 21.971575538054054)                  43.325309                      0.0                  False                 False                 False                     0.328626                        0.0                    False                   False                   False        0.0   -3833.256298  7670.512595  7680.328106            23.507262                1000

finding our best fit::

    best_fit = ufitter.get_best(significant=False)
    print(best_fit.summary)
    best_fit.plot()

.. code-block:: text

                                                                   summary
    Parametric/Non-Parametric                                   Parametric
    Discrete/Continuous                                         continuous
    Distribution                                                    cauchy
    #Params                                                              2
    param0                                                       -0.070741
    param1                                                        1.642212
    Support                                                    (-inf, inf)
    Fitted Domain                 (-16.627835918238397, 20.41344998969709)
    Cramér-von Mises statistic                                    0.272381
    Cramér-von Mises p-value                                      0.162046
    Cramér-von Mises @ 10%                                            True
    Cramér-von Mises @ 5%                                             True
    Cramér-von Mises @ 1%                                             True
    Kolmogorov-Smirnov statistic                                  0.034967
    Kolmogorov-Smirnov p-value                                    0.169277
    Kolmogorov-Smirnov @ 10%                                          True
    Kolmogorov-Smirnov @ 5%                                           True
    Kolmogorov-Smirnov @ 1%                                           True
    Likelihood                                                         0.0
    Log-Likelihood                                            -2791.769256
    AIC                                                        5587.538511
    BIC                                                        5597.354022
    Sum of Squared Error                                           9.18869
    #Fitted Data Points                                               1000

.. image:: https://github.com/tfm000/sklarpy/blob/main/media/univariate_fitter_example_figure1.png?raw=true
    :alt: poisson plot
    :align: center

We can also save our UnivariateFitter object::

    ufitter.save()

We can then easily reload this::

    from sklarpy import load

    loaded_ufitter = load('UnivariateFitter.pickle')
    loaded_best_fit = loaded_ufitter.get_best(significant=False)
