# Contains tests for Fitted SklarPy multivariate distributions
import numpy as np
import pandas as pd
import pytest
from typing import Callable
import matplotlib.pyplot as plt
import scipy.stats

from sklarpy.tests.multivariate.helpers import get_dist
from sklarpy.utils._params import Params


@pytest.mark.test_local_only
def test_fitted_logpdf_pdf_cdf_mc_cdf(
        mvt_continuous_data, mvt_discrete_data, pd_mvt_continuous_data,
        pd_mvt_discrete_data, mv_dists_to_test, params_2d):
    """Testing the logpdf, pdf, cdf and mc-cdf functions of fitted multivariate
     distributions"""
    eps: float = 10 ** -5
    num_generate: int = 10

    for name in mv_dists_to_test:
        _, fitted, _ = get_dist(name, params_2d, mvt_continuous_data)
        for func_str in ('logpdf', 'pdf', 'mc_cdf'): #, 'cdf'):
            func: Callable = eval(f'fitted.{func_str}')
            cdf_num: int = 10
            datasets = (mvt_continuous_data[:cdf_num, :],
                        mvt_discrete_data[:cdf_num, :],
                        pd_mvt_continuous_data.iloc[:cdf_num, :],
                        pd_mvt_discrete_data.iloc[:cdf_num, :]) \
                if func_str == 'cdf' else (mvt_continuous_data,
                                           mvt_discrete_data,
                                           pd_mvt_continuous_data,
                                           pd_mvt_discrete_data)

            for data in datasets:
                output = func(x=data, match_datatype=True,
                              num_generate=num_generate)

                np_output = np.asarray(output)
                n, d = np.asarray(data).shape

                # checking same datatype
                assert isinstance(output, type(data)), \
                    f"{func_str} values for {name} do not match the " \
                    f"datatype: {type(data)}."

                # checking the correct size
                assert np_output.size == n, \
                    f"{func_str} values for {name} are not the correct size."

                # checking for nan-values
                assert np.isnan(np_output).sum() == 0, \
                    f'nans present in {name} {func_str} values.'

                # function specific tests
                if func_str == 'pdf':
                    assert np.all(np_output >= -eps), \
                        f"pdf values in {name} are negative."
                elif func_str in ('cdf', 'mc_cdf'):
                    assert np.all((-eps <= np_output) & (output <= 1 + eps)), \
                        f"{func_str} values in {name} outside [0, 1]."

            # checking error if wrong dimension
            new_dataset: np.ndarray = np.zeros((n, d + 1))
            with pytest.raises(
                    ValueError, match="Dimensions implied by parameters do "
                                      "not match those of the dataset."):
                func(x=new_dataset, num_generate=num_generate)


def test_fitted_rvs(mv_dists_to_test, params_2d, mvt_continuous_data):
    """Testing the rvs functions of fitted multivariate distributions."""
    for name in mv_dists_to_test:
        _, fitted, _ = get_dist(name, params_2d, mvt_continuous_data)
        for size in (1, 2, 5, 101):
            rvs = fitted.rvs(size=size)
            np_rvs: np.ndarray = np.asarray(rvs)

            # checking correct shape
            assert np_rvs.shape[0] == size, \
                f"fitted rvs for {name} did not generate the correct number " \
                f"of pseudo-samples."

            # checking for nan values
            assert np.isnan(np_rvs).sum() == 0, \
                f"nan values present in {name} fitted rvs."


def test_fitted_scalars(mvt_continuous_data, mvt_discrete_data,
                        pd_mvt_continuous_data, pd_mvt_discrete_data,
                        mv_dists_to_test, params_2d):
    """Testing the likelihood, loglikelihood, AIC and BIC functions of
    multivariate fitted distributions."""
    for name in mv_dists_to_test:
        _, fitted, _ = get_dist(name, params_2d, mvt_continuous_data)
        for func_str in ('likelihood', 'loglikelihood', 'aic', 'bic'):
            func: Callable = eval(f'fitted.{func_str}')
            for data in (mvt_continuous_data, mvt_discrete_data,
                         pd_mvt_continuous_data, pd_mvt_discrete_data):
                value = func(data=data)

                # checking correct type
                assert isinstance(value, float), \
                    f"{func_str} for {name} is not a float when datatype is " \
                    f"{type(data)}"

                # checking valid number
                assert not np.isnan(value), \
                    f"{func_str} for {name} is is nan when datatype is " \
                    f"{type(data)}"

                if func_str == "likelihood":
                    # checking positive
                    assert value >= 0, \
                        f"{func_str} for {name} is negative when datatype " \
                        f"is {type(data)}."

            # checking error if wrong dimension
            n, d = data.shape
            new_dataset: np.ndarray = np.zeros((n, d + 1))
            with pytest.raises(
                    ValueError,
                    match="Dimensions implied by parameters do "
                          "not match those of the dataset."):
                func(data=new_dataset)


def test_fitted_plots(params_2d, params_3d, mvt_continuous_data):
    """Testing the marginal_pairplot, pdf_plot, cdf_plot and mc_cdf_plot
    methods of fitted multivariate distributions."""
    mvt_continuous_data_3d: np.ndarray = scipy.stats.multivariate_normal.rvs(
        size=(mvt_continuous_data.shape[0], 3))
    kwargs: dict = {'num_points': 2, 'num_generate': 10, 'mc_num_generate': 10,
                    'show': False, 'show_progress': False}
    for name in ('mvt_normal', ):
        _, fitted_2d, _ = get_dist(name, params_2d, mvt_continuous_data)
        _, fitted_3d, _ = get_dist(name, params_3d, mvt_continuous_data_3d)
        for func_str in ('marginal_pairplot', 'pdf_plot',
                         'cdf_plot', 'mc_cdf_plot'):
            func_2d: Callable = eval(f'fitted_2d.{func_str}')
            func_3d: Callable = eval(f'fitted_3d.{func_str}')

            # testing 3d plots
            if func_str == 'marginal_pairplot':
                func_3d(**kwargs)
                plt.close()
            else:
                with pytest.raises(NotImplementedError,
                                   match=f"{func_str} is not "
                                         f"implemented when the number of "
                                         f"variables is not 2."):
                    func_3d(**kwargs)

            # testing 2d plots
            func_2d(**kwargs)
            plt.close()


def test_fitted_params(mv_dists_to_test, params_2d, mvt_continuous_data):
    """Testing the params attribute of fitted multivariate distributions."""
    for name in mv_dists_to_test:
        _, fitted, _ = get_dist(name, params_2d, mvt_continuous_data)

        # checking exists
        assert 'params' in dir(fitted), f"params is not an attribute of " \
                                        f"fitted {name}."

        # checking params object
        assert issubclass(type(fitted.params), Params), \
            f"params of fitted {name} is not a Params object."


def test_fitted_integers(mv_dists_to_test, params_2d, mvt_continuous_data):
    """Testing the num_params, num_scalar_params, num_variables and
    fitted_num_data_points of fitted multivariate distributions."""
    for name in mv_dists_to_test:
        _, fitted, params = get_dist(name, params_2d, mvt_continuous_data)

        for func_str in ('num_params', 'num_scalar_params', 'num_variables',
                         'fitted_num_data_points'):
            # testing exists
            assert func_str in dir(fitted), \
                f'{func_str} is not a method or attribute of fitted {name}.'

            value = eval(f"fitted.{func_str}")
            assert isinstance(value, int), \
                f"{func_str} of {name} is not an integer."
            assert value >= 0, f"{func_str} of {name} is negative."
        assert fitted.num_params == len(params), \
            f"num_params of {name} does not match the length of its params " \
            f"object."


def test_fitted_converged(mv_dists_to_test, params_2d, mvt_continuous_data):
    """Testing converged attributes of fitted multivariate distributions."""
    for name in mv_dists_to_test:
        _, fitted, _ = get_dist(name, params_2d, mvt_continuous_data)

        # testing exists
        assert 'converged' in dir(fitted), \
            f"converged is not an attribute of fitted {name}."

        # testing correct type
        assert isinstance(fitted.converged, bool), \
            f"converged is not a boolean value for fitted {name}."


def test_fitted_summaries(mv_dists_to_test, params_2d, mvt_continuous_data):
    """Testing the summaries of fitted multivariate distributions."""
    for name in mv_dists_to_test:
        _, fitted, _ = get_dist(name, params_2d, mvt_continuous_data)

        # testing exists
        assert 'summary' in dir(fitted), \
            f"summary is not an attribute of fitted {name}."

        # testing correct type
        assert isinstance(fitted.summary, pd.DataFrame), \
            f"summary is not a pandas dataframe for fitted {name}."

        # testing non-empty
        assert len(fitted.summary) > 0, \
            f"summary of fitted {name} is an empty dataframe."
