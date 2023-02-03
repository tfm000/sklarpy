# Contains tests for sklarpy's UnivariateFitter class
import pandas as pd
import pytest

from sklarpy.univariate import UnivariateFitter, distributions_map
from sklarpy.univariate._fitted_dists import FittedContinuousUnivariate, FittedDiscreteUnivariate
from sklarpy._utils import FitError, SignificanceError


def test_init(poisson_data, uniform_data, normal_data):
    """Testing UnivariateFitter initialises without errors"""
    for data in (poisson_data, uniform_data, normal_data):
        fitter = UnivariateFitter(data)


def test_fit(poisson_data, uniform_data):
    """Testing the fit method of UnivariateFitter."""

    for data in (uniform_data, poisson_data):
        # fitting with default args
        default_fitter = UnivariateFitter(data).fit()

        # fitting with non-default distribution category
        non_default_fitter1 = UnivariateFitter(data).fit('all')

        # fitting with invalid distribution iterable
        with pytest.raises(TypeError):
            UnivariateFitter(data).fit(data)

        # fitting with invalid distribution type
        with pytest.raises(TypeError):
            UnivariateFitter(data).fit(56)

        # fitting with invalid distribution category
        with pytest.raises(FitError):
            UnivariateFitter(data).fit('invalid category test', raise_error=True)

        # fitting with non-default distribution list
        non_default_fitter2 = UnivariateFitter(data).fit(['normal', 't'])

        # fitting with empty distributions list
        with pytest.raises(ValueError):
            UnivariateFitter(data).fit([])

        # fitting with empty distributions category string
        with pytest.raises(FitError):
            UnivariateFitter(data).fit('', raise_error=True)

        # fitting with an invalid distribution in fit
        with pytest.raises(FitError):
            UnivariateFitter(data).fit(['invalid distribution test'], raise_error=True)

        # fitting with use_processpoolexecutor = False
        default_fitter3 = UnivariateFitter(data).fit(use_processpoolexecutor=False)

        # fitting with numerical = True
        non_default_fitter4 = UnivariateFitter(data).fit('all numerical', numerical=True)

        # fitting same object twice
        UnivariateFitter(data).fit().fit()


def test_get_summary(poisson_data, uniform_data):
    """Testing the get_summary method of UnivariateFitter"""
    continuous_fitter = UnivariateFitter(uniform_data)
    discrete_fitter = UnivariateFitter(poisson_data)

    # testing not implemented when not fit
    with pytest.raises(FitError):
        continuous_fitter.plot()
    with pytest.raises(FitError):
        discrete_fitter.plot()

    continuous_fitter.fit()
    discrete_fitter.fit()

    # testing get summary on both continuous and discrete data
    summary1: pd.DataFrame = continuous_fitter.get_summary()
    summary2: pd.DataFrame = discrete_fitter.get_summary()
    assert isinstance(summary1, pd.DataFrame) and (len(summary1) > 0), "summary not successfully generated when " \
                                                                       "using continuous data"
    assert isinstance(summary2, pd.DataFrame) and (len(summary1) > 0), "summary not successfully generated when " \
                                                                       "using discrete data"
    # testing get_summary returns an empty dataframe when no distributions can be fit
    summary3: pd.DataFrame = UnivariateFitter(uniform_data).fit('all discrete').get_summary()
    assert isinstance(summary3, pd.DataFrame) and (summary3.to_numpy().size == 0), "get_summary does not return an " \
                                                                                   "empty dataframe when no " \
                                                                                   "distributions can be fit"

    # testing different arguments
    summary4: pd.DataFrame = continuous_fitter.get_summary(significant=True, pvalue=0.1)
    assert isinstance(summary4, pd.DataFrame), "get_summary does not return a dataframe when significant is true " \
                                               "and pvalue = 0.1."


def test_get_best(uniform_data, poisson_data, normal_data):
    """Testing the get_best method of UnivariateFitter"""
    continuous_fitter = UnivariateFitter(uniform_data)
    discrete_fitter = UnivariateFitter(poisson_data)

    # testing not implemented when not fit
    with pytest.raises(FitError):
        continuous_fitter.plot()
    with pytest.raises(FitError):
        discrete_fitter.plot()

    continuous_fitter.fit()
    discrete_fitter.fit()

    # checking basic functionality works
    best_uniform = continuous_fitter.get_best()
    best_poisson_data = discrete_fitter.get_best()
    assert isinstance(best_uniform, FittedContinuousUnivariate), "best fit for uniform data should be a continuous" \
                                                                 " distribution"
    assert isinstance(best_poisson_data, FittedDiscreteUnivariate), "best fit for poisson data should be a discrete " \
                                                                    "distribution"

    # checking an error is raised when no best fit is found
    with pytest.raises(SignificanceError):
        UnivariateFitter(normal_data).fit(['uniform']).get_best(significant=True, pvalue=10**-4, raise_error=True)


def test_plot(uniform_data, poisson_data):
    """Testing the plot method of UnivariateFitter"""
    continuous_fitter = UnivariateFitter(uniform_data)
    discrete_fitter = UnivariateFitter(poisson_data)

    # testing not implemented when not fit
    with pytest.raises(FitError):
        continuous_fitter.plot()
    with pytest.raises(FitError):
        discrete_fitter.plot()

    continuous_fitter.fit()
    discrete_fitter.fit()
    continuous_fitter.plot(show=False)
    discrete_fitter.plot(show=False)


def test_fitted_distributions(uniform_data, poisson_data):
    """Testing the fitted_distributions property of UnivariateFitter"""
    continuous_fitter: UnivariateFitter = UnivariateFitter(uniform_data)
    discrete_fitter: UnivariateFitter = UnivariateFitter(poisson_data)

    # testing not implemented when not fit
    with pytest.raises(FitError):
        continuous_fitter.fitted_distributions
    with pytest.raises(FitError):
        discrete_fitter.fitted_distributions

    continuous_fitter.fit()
    discrete_fitter.fit()
    assert isinstance(continuous_fitter.fitted_distributions, dict) and \
           (len(continuous_fitter.fitted_distributions) > 0), "fitted_distributions should be a dictionary containing" \
                                                              " multiple fitted distributions, but this is not the " \
                                                              "case for continuous data"
    assert isinstance(discrete_fitter.fitted_distributions, dict) and \
           (len(discrete_fitter.fitted_distributions) > 0), "fitted_distributions should be a dictionary containing" \
                                                              " multiple fitted distributions, but this is not the" \
                                                            " case for discrete data."

