.. _copulas:

##############
Copula Models
##############

This SklarPy package contains many different copula models.
Unlike univariate distributions, these are not wrappers of scipy objects.

All implemented copula models are able to be fitted to both multivariate numpy and pandas data and contain easy saving and plotting methods.

An important concept to remember when using these models that they are composed of 2 overall parts:
1. The marginal distributions. These are univariate distributions of each random variable.
2. The copula distribution. This multivariate model captures the dependence structure between the variables.

The overall multivariate joint distribution is created by combining these two parts.

Which copula models are implemented?
------------------------------------
Currently, the following copula models are implemented:

.. csv-table:: Copula Models
    :file: copula_table.csv
    :header-rows: 1

All Normal-Mixture models use the parameterization specified by McNeil, Frey and Embrechts (2005).

MarginalFitter
--------------
This class is used to fit multiple univariate distributions to data easily and evaluate their methods.
It implements the following methods and attributes:

- marginal_logpdf (log of the probability density functions of the marginal distributions)
- marginal_pdfs (the probability density functions of the fitted marginal distributions)
- marginal_cdfs (the cumulative distribution functions of the fitted marginal distributions)
- marginal_ppfs (the percent point functions / inverse cdfs of the fitted marginal distributions)
- marginal_rvs (random variate generators / samplers of the fitted marginal distributions)
- pairplot (pairplot of the fitted marginal distributions)
- marginals (the fitted marginal distributions as a dictionary)
- summary (a summary of the fitted marginal distributions)
- num_variables (the number of variables present in the original dataset)
- fitted (whether the marginal distributions have been fitted to data)
- fit (fitting the marginal distributions to data)

PreFitCopula
-------------
This is the base class for all copula models. It implements the following methods and attributes:

- logpdf (log of the probability density function of the overall joint distribution)
- pdf (probability density function of the overall joint distribution)
- cdf (cumulative distribution function of the overall joint distribution)
- mc_cdf (Monte Carlo approximation of the cumulative distribution function of the overall joint distribution)
- rvs (random variate generator / sampler of the overall joint distribution)
- copula_logpdf (log of the probability density function of the copula distribution)
- copula_pdf (probability density function of the copula distribution)
- copula_cdf (cumulative distribution function of the copula distribution)
- copula_mc_cdf (Monte Carlo approximation of the cumulative distribution function of the copula distribution)
- copula_rvs (random variate generator / sampler of the copula distribution)
- num_marginal_params (number of parameters in the marginal distributions)
- num_copula_params (number of parameters in the copula distribution)
- num_scalar_params (number of scalar parameters in the overall joint distribution)
- num_params (number of parameters in the overall joint distribution)
- likelihood (likelihood of the overall joint distribution)
- loglikelihood (log of the likelihood of the overall joint distribution)
- aic (Akaike Information Criterion of the overall joint distribution)
- bic (Bayesian Information Criterion of the overall joint distribution)
- marginal_pairplot (pairplot of the marginal distributions)
- pdf_plot (plot of the probability density function of the overall joint distribution)
- cdf_plot (plot of the cumulative distribution function of the overall joint distribution)
- mc_cdf_plot (plot of the Monte Carlo approximation of the cumulative distribution function of the overall joint distribution)
- copula_pdf_plot (plot of the probability density function of the copula distribution)
- copula_cdf_plot (plot of the cumulative distribution function of the copula distribution)
- copula_mc_cdf_plot (plot of the Monte Carlo approximation of the cumulative distribution function of the copula distribution)
- fit (fitting the overall joint distribution to data)

mc_cdf and copula_mc_cdf are numerical approximations of their respective cumulative distribution functions.
These are usually necessary as the analytical forms of these functions are often not available and numerical integration is computationally expensive.

Also note that pdf and cdf plots are only implemented for 2-dimensional distributions.

FittedCopula
------------
This class is the fitted version of PreFitCopula's subclasses.
It implements the same methods as PreFitCopula, but does not require copula_params or mdists as arguments.
It also implements the following additional methods and attributes:

- copula_params (the fitted parameters of the copula distribution)
- mdists (the fitted univariate marginal distributions)
- num_variables (the number of variables the distribution is fitted too)
- fitted_num_data_points (the number of observations used to fit the distribution)
- converged (whether the fitting algorithm converged)
- summary (a summary of the overall fitted distribution)
- save (save the overall fitted distribution object)

MarginalFitter Example
-----------------------
Generating data and fitting marginal distributions::

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
    from sklarpy.multivariate import mvt_normal

    rvs: np.ndarray = mvt_normal.rvs(num_generate, my_mvn_params)
    rvs_df: pd.DataFrame = pd.DataFrame(rvs, columns=['Wife Age', 'Husband Age'
                                                      ], dtype=float)

    # applying MarginalFitter to our random variables
    from sklarpy.copulas import MarginalFitter

    mfitter: MarginalFitter = MarginalFitter(rvs_df)
    mfitter.fit({'pvalue': 0.01})

    # printing out a summary of our fits
    from sklarpy import print_full
    print_full()

    print(mfitter.summary)


.. code-block:: text

                                                                  Wife Age                              Husband Age
    Parametric/Non-Parametric                                   Parametric                               Parametric
    Discrete/Continuous                                         continuous                               continuous
    Distribution                                                   lognorm                                  lognorm
    #Params                                                              3                                        3
    param0                                                        0.000005                                 0.000001
    param1                                                  -262115.561308                          -2097116.799667
    param2                                                   262148.497841                           2097160.700641
    Support                                     (-262115.56130758836, inf)               (-2097116.7996667635, inf)
    Fitted Domain                 (28.438692411392555, 36.673753788627785)  (35.20033323448715, 51.735336956575935)
    Cramér-von Mises statistic                                    0.124954                                 0.102395
    Cramér-von Mises p-value                                      0.475847                                 0.573349
    Cramér-von Mises @ 10%                                            True                                     True
    Cramér-von Mises @ 5%                                             True                                     True
    Cramér-von Mises @ 1%                                             True                                     True
    Kolmogorov-Smirnov statistic                                  0.032827                                 0.024709
    Kolmogorov-Smirnov p-value                                    0.226385                                  0.56612
    Kolmogorov-Smirnov @ 10%                                          True                                     True
    Kolmogorov-Smirnov @ 5%                                           True                                     True
    Kolmogorov-Smirnov @ 1%                                           True                                     True
    Likelihood                                                         0.0                                      0.0
    Log-Likelihood                                            -1666.824453                             -2382.153726
    AIC                                                        3339.648906                              4770.307452
    BIC                                                        3354.372172                              4785.030718
    Sum of Squared Error                                         16.819752                                 6.322994
    #Fitted Data Points                                               1000                                     1000

Printing Marginals::

    print(mfitter.marginals)

.. code-block:: text

    {0: lognorm(0.0, -262115.56, 262148.5), 1: lognorm(0.0, -2097116.8, 2097160.7)}

Calculating marginal cdf values::

    mcdf_values: pd.DataFrame = mfitter.marginal_cdfs()
    print(mcdf_values)

.. code-block:: text

         Wife Age  Husband Age
    0    0.446886     0.676438
    1    0.162115     0.107338
    2    0.631869     0.461236
    3    0.182751     0.589056
    4    0.827908     0.870150
    ..        ...          ...
    995  0.732827     0.523818
    996  0.457342     0.372388
    997  0.319827     0.598163
    998  0.476477     0.350149
    999  0.353060     0.323429

Producing a pairplot of the marginals::

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

    mfitter2.pairplot()

.. image:: https://github.com/tfm000/sklarpy/blob/docs/readthedocs/media/mfitter_pairplot.png?raw=true
   :alt: MarginalFitter Pair-Plot
   :scale: 60%
   :align: center

Copula Example
--------------
Here we use the generalized hyperbolic copula, though all methods and attributes are generalized.::

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

    # fitting a generalized hyperbolic copula to our generated data using
    # Maximum Likelihood Estimation
    from sklarpy.copulas import gh_copula

    fitted_copula = gh_copula.fit(
        data=rvs_df, method='mle',
        univariate_fitter_options={'significant': False}, show_progress=True)

    # prining our fitted parameters
    from sklarpy import print_full
    print_full()

    print(fitted_copula.copula_params.to_dict)

.. code-block:: text

    {'lamb': -10.0, 'chi': 8.460830761870396, 'psi': 10.0,
    'loc': array([[0.], [0.]]),
    'shape': array([[ 1.       , -0.5214283],
                    [-0.5214283,  1.       ]]),
    'gamma': array([[0.99848424], [0.94696141]])}

Printing marginal distributions::

    print(fitted_copula.mdists)

.. code-block:: text

    {0: lognorm(0.38, -0.78, 4.02), 1: lognorm(0.0, -1276.15, 1268.45)}

Printing covariance parameters::

    print(fitted_copula.copula_params.cov)

.. code-block:: text

    [[ 0.39404386 -0.18821382]
    [-0.18821382  0.3928638 ]]

Printing a summargy of our joint fit::

    print(fitted_copula.summary)

.. code-block:: text

                                  Joint Distribution           gh                                     summary                                     summary
    Distribution                  Joint Distribution       mvt_gh                                     lognorm                                      cauchy
    #Variables                                     2            2                                         NaN                                         NaN
    #Params                                       11            6                                           3                                           2
    #Scalar Params                                11            6                                         NaN                                         NaN
    Converged                                   True         True                                         NaN                                         NaN
    Likelihood                                   0.0          0.0                                         0.0                                         0.0
    Log-Likelihood                      -4298.311941 -1032.490682                                -1880.434874                                -2561.765741
    AIC                                  8618.623881  2076.981365                                 3766.869748                                 5127.531482
    BIC                                  8672.609189  2106.427896                                 3781.593014                                 5137.346993
    #Fitted Data Points                         1000         1000                                        1000                                        1000
    Parametric/Non-Parametric                    NaN          NaN                                  Parametric                                  Parametric
    Discrete/Continuous                          NaN          NaN                                  continuous                                  continuous
    param0                                       NaN          NaN                                    0.328725                                   -6.937913
    param1                                       NaN          NaN                                   -1.596967                                    1.485756
    param2                                       NaN          NaN                                    4.826054                                         NaN
    Support                                      NaN          NaN                  (-1.5969673012994325, inf)                                 (-inf, inf)
    Fitted Domain                                NaN          NaN  (0.030085402918948567, 10.416203209871883)  (-28.483718062724616, -2.8836636097027206)
    Cramér-von Mises statistic                   NaN          NaN                                    0.055878                                    3.834238
    Cramér-von Mises p-value                     NaN          NaN                                    0.840024                                         0.0
    Cramér-von Mises @ 10%                       NaN          NaN                                        True                                       False
    Cramér-von Mises @ 5%                        NaN          NaN                                        True                                       False
    Cramér-von Mises @ 1%                        NaN          NaN                                        True                                       False
    Kolmogorov-Smirnov statistic                 NaN          NaN                                    0.018599                                    0.128949
    Kolmogorov-Smirnov p-value                   NaN          NaN                                    0.872994                                         0.0
    Kolmogorov-Smirnov @ 10%                     NaN          NaN                                        True                                       False
    Kolmogorov-Smirnov @ 5%                      NaN          NaN                                        True                                       False
    Kolmogorov-Smirnov @ 1%                      NaN          NaN                                        True                                       False
    Sum of Squared Error                         NaN          NaN                                   11.475127                                    8.464622

Plotting our fit::

    fitted_copula.copula_pdf_plot(show=False)
    fitted_copula.pdf_plot(show=False)
    fitted_copula.mc_cdf_plot(show=False)
    plt.show()

.. image:: https://github.com/tfm000/sklarpy/blob/docs/readthedocs/media/PDF_Gh_PDF_Plot_Plot2.png?raw=true
   :alt: Generalized Hyperbolic PDF
   :scale: 60%
   :align: center

.. image:: https://github.com/tfm000/sklarpy/blob/docs/readthedocs/media/Copula_PDF_Gh_Copula_PDF_Plot_Plot2.png?raw=true
   :alt: Generalized Hyperbolic Copula PDF
   :scale: 60%
   :align: center

.. image:: https://github.com/tfm000/sklarpy/blob/docs/readthedocs/media/MC_CDF_Gh_MC_PDF_Plot_Plot2.png?raw=true
   :alt: Generalized Hyperbolic CDF
   :scale: 60%
   :align: center

Saving our fitted copula::

    fitted_copula.save()

We can then easily reload this object later::

    from sklarpy import load

    loaded_copula = load('gh.pickle')
    print(loaded_copula.summary)