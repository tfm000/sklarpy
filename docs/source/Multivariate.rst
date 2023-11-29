.. _multivariate:

############################
Multivariate Distributions
############################

This SklarPy package contains many different multivariate distributions.
Unlike univariate distributions, these are not wrappers of scipy objects (with the exceptions of mvt_normal and mvt_student_t).

All implemented multivariate distributions are able to be fitted to both multivariate numpy and pandas data and contain easy saving and plotting methods.

Which multivariate distributions are implemented?
------------------------------------------------
Currently, the following multivariate distributions are implemented:

.. csv-table:: Multivariate Distributions
    :file: mvt_table.csv
    :header-rows: 1

All Normal-Mixture models use the parameterization specified by McNeil, Frey and Embrechts (2005).

PreFitContinuousMultivariate
----------------------------
This is the base class for all multivariate distributions. It implements the following methods and attributes:

- logpdf (log of the probability density function)
- pdf (probability density function)
- cdf (cumulative distribution function)
- mc_cdf (Monte Carlo approximation of the cumulative distribution function)
- rvs (random variate generator / sampler)
- likelihood (likelihood function)
- loglikelihood (log of the likelihood function)
- aic (Akaike Information Criterion)
- bic (Bayesian Information Criterion)
- marginal_pairplot (pairplot of the marginal distributions)
- pdf_plot (plot of the probability density function)
- cdf_plot (plot of the cumulative distribution function)
- mc_cdf_plot (plot of the Monte Carlo approximation of the cumulative distribution function)
- num_params (The number of parameters in the distribution)
- num_scalar_params (The number of scalar values across all parameters in the distribution)
- fit (fitting the distribution to data)

mc_cdf is a numerical approximation of the cumulative distribution function. This is usually necessary for distributions that do not have a closed form cumulative density function, as the numerical integration alternative is computationally expensive.

num_params is the number of parameter objects in the distribution, i.e. a vector / matrix is counted as 1.
num_scalar_params counts the number of unique scalar values across all parameter objects.

Also note that pdf and cdf plots are only implemented for 2-dimensional distributions.

FittedContinuousMultivariate
----------------------------
This class is the fitted version of PreFitContinuousMultivariate's subclasses.
It implements the same methods as PreFitContinuousMultivariate, but does not require params as an argument.
It also implements the following additional methods and attributes:

- params (the fitted parameters)
- num_variables (the number of variables the distribution is fitted too)
- fitted_num_data_points (the number of observations used to fit the distribution)
- converged (whether the fitting algorithm converged)
- summary (a summary of the fitted distribution)
- save (save the fitted distribution object)

Multivariate Example
---------------------
Here we use the multivariate normal and multivariate symmetric hyperbolic distributions, though all methods and attributes are generalized.::

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
    from sklarpy.multivariate import mvt_normal

    rvs: np.ndarray = mvt_normal.rvs(1000, my_mvn_params)
    rvs_df: pd.DataFrame = pd.DataFrame(rvs, columns=['Wife Age', 'Husband Age'],
                                        dtype=float)

    # fitting a symmetric hyperbolic dist to our generated data using
    # Maximum Likelihood Estimation
    from sklarpy.multivariate import mvt_shyperbolic

    fitted_msh = mvt_shyperbolic.fit(rvs_df, method='mle', show_progress=True)

    # printing our fitted parameters
    print(fitted_msh.params.to_dict)
    print(fitted_msh.params.cov)


.. code-block:: text

    {'chi': 6.817911964473556, 'psi': 10.0, 'loc': array([[32.99012429],
       [43.91822886]]), 'shape': array([[1.72408489, 2.27711492],
       [2.27711492, 6.27443288]])}

    [[1.78702958 2.36025021]
    [2.36025021 6.50350643]]

Printing a summary of our fit::

        print(fitted_msh.summary())

.. code-block:: text

                                 summary
    Distribution         mvt_shyperbolic
    #Variables                         2
    #Params                            4
    #Scalar Params                     7
    Converged                       True
    Likelihood                       0.0
    Log-Likelihood           -3664.49604
    AIC                       7342.99208
    BIC                      7377.346367
    #Fitted Data Points             1000

Plotting our fitted distribution::

    fitted_msh.pdf_plot(show=False)
    fitted_msh.mc_cdf_plot(show=False)
    fitted_msh.marginal_pairplot(show=False)
    plt.show()

.. image:: https://github.com/tfm000/sklarpy/blob/docs/readthedocs/media/PDF_Mvt_Shyperbolic_PDF_Plot_Plot.png?raw=true
   :alt: Symmetric Hyperbolic PDF
   :scale: 60%
   :align: center

.. image:: https://github.com/tfm000/sklarpy/blob/docs/readthedocs/media/MC_CDF_Mvt_Shyperbolic_MC_CDF_Plot_Plot.png?raw=true
   :alt: Symmetric Hyperbolic PDF
   :scale: 60%
   :align: center

.. image:: https://github.com/tfm000/sklarpy/blob/docs/readthedocs/media/mvt_shyperbolic_marginal_pair_plot.png?raw=true
   :alt: Symmetric Hyperbolic PDF
   :scale: 60%
   :align: center

Saving our fitted parameters::

    fitted_msh.params.save()

Reloading and fitting to another distribution of the same type::

    from sklarpy import load

    loaded_msh_params = load('mvt_shyperbolic.pickle')
    param_fitted_msh = mvt_shyperbolic.fit(params=loaded_msh_params)
