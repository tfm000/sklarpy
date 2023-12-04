# Contains tests for sklarpy's UnivariateFitter class
import pandas as pd
import pytest
import matplotlib.pyplot as plt

from sklarpy.univariate import UnivariateFitter, distributions_map
from sklarpy.univariate._fitted_dists import FittedContinuousUnivariate, \
    FittedDiscreteUnivariate
from sklarpy.utils._errors import FitError, SignificanceError


def test_init(discrete_data, continuous_data):
    """Testing whether UnivariateFitter initialises without errors"""
    for data in (discrete_data, continuous_data):
        fitter = UnivariateFitter(data)


def test_fit(discrete_data, continuous_data):
    """Testing the fit method of UnivariateFitter."""
    for data in (discrete_data, continuous_data):
        # fitting with default args
        default_fitter = UnivariateFitter(data).fit()

        # fitting with non-default distribution category
        non_default_fitter1 = UnivariateFitter(data).fit('all multimodal',
                                                         multimodal=True)

        # fitting with invalid distribution iterable
        with pytest.raises(TypeError):
            UnivariateFitter(data).fit(data)

        # fitting with invalid distribution type
        with pytest.raises(TypeError):
            UnivariateFitter(data).fit(56)

        # fitting with invalid distribution category
        with pytest.raises(FitError):
            UnivariateFitter(data).fit('invalid category test',
                                       raise_error=True)

        # fitting with non-default distribution list
        non_default_fitter2 = UnivariateFitter(data).fit(['normal',
                                                          'student_t'])

        # fitting with empty distributions list
        with pytest.raises(ValueError):
            UnivariateFitter(data).fit([])

        # fitting with empty distributions category string
        with pytest.raises(FitError):
            UnivariateFitter(data).fit('', raise_error=True)

        # fitting with an invalid distribution in fit
        with pytest.raises(FitError):
            UnivariateFitter(data).fit(['invalid distribution test'],
                                       raise_error=True)

        # fitting with use_processpoolexecutor = True
        default_fitter3 = UnivariateFitter(data).fit(
            use_processpoolexecutor=True)

        # fitting with numerical = True
        non_default_fitter4 = UnivariateFitter(data).fit('all numerical',
                                                         numerical=True)

        # fitting same object twice
        UnivariateFitter(data).fit().fit()


def test_get_summary(discrete_data, continuous_data):
    """Testing the get_summary method of UnivariateFitter"""
    for ufitter in (UnivariateFitter(discrete_data),
                    UnivariateFitter(continuous_data)):
        # testing not implemented when not fit
        with pytest.raises(FitError, match="UnivariateFitter has not been "
                                           "fitted to data. Call .fit "
                                           "method."):
            ufitter.get_summary()

        ufitter.fit()

        # testing get summary on both continuous and discrete data
        summary: pd.DataFrame = ufitter.get_summary()
        assert isinstance(summary, pd.DataFrame) and (len(summary) > 0), \
            "summary not successfully generated."

    # testing get_summary returns an empty dataframe
    # when no distributions can be fit
    summary2: pd.DataFrame = UnivariateFitter(continuous_data).fit(
        'all discrete').get_summary()
    assert (isinstance(summary2, pd.DataFrame)
            and (summary2.to_numpy().size == 0)), \
        "get_summary does not return an empty dataframe when no " \
        "distributions can be fit"

    # testing different arguments
    summary3: pd.DataFrame = ufitter.get_summary(significant=True, pvalue=0.1)
    assert isinstance(summary3, pd.DataFrame), \
        "get_summary does not return a dataframe when significant is true " \
        "and pvalue = 0.1."


def test_get_best(discrete_data, continuous_data):
    """Testing the get_best method of UnivariateFitter"""
    for dtype, ufitter in {'discrete': UnivariateFitter(discrete_data),
                           'continuous': UnivariateFitter(continuous_data)
                           }.items():
        # testing not implemented when not fit
        with pytest.raises(FitError, match="UnivariateFitter has not been "
                                           "fitted to data. Call .fit "
                                           "method."):
            ufitter.get_best()

        ufitter.fit()

        # checking basic functionality works
        best = ufitter.get_best()
        target_dtype = eval(f'Fitted{dtype.title()}Univariate')

        assert isinstance(best, target_dtype), \
            f"best fit for {dtype} data should be a {target_dtype} " \
            f"distribution"

    # checking an error is raised when no best fit is found
    with pytest.raises(SignificanceError):
        UnivariateFitter(continuous_data).fit(['uniform']).get_best(
            significant=True, pvalue=1, raise_error=True)


def test_plot(discrete_data, continuous_data):
    """Testing the plot method of UnivariateFitter"""
    for dtype, ufitter in {'discrete': UnivariateFitter(discrete_data),
                           'continuous': UnivariateFitter(continuous_data)
                           }.items():

        # testing not implemented when not fit
        with pytest.raises(FitError, match="UnivariateFitter has not been "
                                           "fitted to data. Call .fit "
                                           "method."):
            ufitter.plot()

        ufitter.fit()
        ufitter.plot(show=False)
        plt.close()


def test_fitted_distributions(discrete_data, continuous_data):
    """Testing the fitted_distributions property of UnivariateFitter"""
    for dtype, ufitter in {'discrete': UnivariateFitter(discrete_data),
                           'continuous': UnivariateFitter(continuous_data)
                           }.items():

        # testing not implemented when not fit
        with pytest.raises(FitError, match="UnivariateFitter has not been "
                                           "fitted to data. Call .fit "
                                           "method."):
            ufitter.fitted_distributions

        ufitter.fit()
        ufitter.fitted_distributions

        assert isinstance(ufitter.fitted_distributions, dict), \
            f"fitted_distributions is not a dictionary for {dtype} data."

        assert len(ufitter.fitted_distributions) > 0, \
            f"fitted_distributions is empty for {dtype} data."
