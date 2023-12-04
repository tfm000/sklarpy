# Contains tests for Pre-Fit SklarPy multivariate distributions
import numpy as np
import pytest
from typing import Callable
import matplotlib.pyplot as plt
import scipy.stats

from sklarpy.multivariate import *
from sklarpy.multivariate._prefit_dists import PreFitContinuousMultivariate
from sklarpy.multivariate._fitted_dists import FittedContinuousMultivariate
from sklarpy.utils._params import Params
from sklarpy.tests.multivariate.helpers import get_dist


def test_correct_type():
    """Testing multivariate distributions are all SklarPy objects."""
    for name in distributions_map['all']:
        dist = eval(name)
        assert issubclass(type(dist), PreFitContinuousMultivariate), \
            f"{name} is not a child class of PreFitContinuousMultivariate"


@pytest.mark.test_local_only
def test_fit_to_data(mvt_continuous_data, mvt_discrete_data,
                     pd_mvt_continuous_data, pd_mvt_discrete_data,
                     mv_dists_to_test):
    """Testing we can fit multivariate distributions to data."""
    for name in mv_dists_to_test:
        dist = eval(name)

        # fitting to both continuous and discrete data,
        # in both numpy and pandas format
        for method in dist._DATA_FIT_METHODS:
            for data in (mvt_continuous_data, pd_mvt_continuous_data,
                         mvt_discrete_data, pd_mvt_discrete_data):
                if isinstance(data, np.ndarray):
                    d: int = data.shape[1]
                else:
                    d: int = len(data.columns)

                try:
                    # fitting to data
                    fitted = dist.fit(data=data, method=method)
                except RuntimeError:
                    continue

                # testing fitted to correct type
                assert issubclass(type(fitted), FittedContinuousMultivariate
                                  ), f"{name} is not fitted to a child " \
                                     f"class of FittedContinuousMultivariate."

                # testing parameters object
                params = fitted.params
                assert issubclass(type(params), Params), \
                    f"{name} fitted parameters are not a child class of " \
                    f"Params."
                assert params.name == name, \
                    f"{name} fitted parameters is not the correct type."
                assert len(params) > 0, \
                    f"{name} fitted parameter object is empty."

                vector_attributes: tuple = ('loc', 'mean', 'gamma')
                matrix_attributes: tuple = ('cov', 'corr')
                scale_attributes: tuple = ('dof', 'chi', 'psi',
                                           'lamb', 'theta')
                to_obj_attributes: tuple = ('dict', 'tuple', 'list')

                for vect_str in vector_attributes:
                    if vect_str in dir(params):
                        vect = eval(f'params.{vect_str}')
                        assert isinstance(vect, np.ndarray), \
                            f"{vect_str} fitted parameter is not an array " \
                            f"for {name}."
                        assert vect.size == d, \
                            f"{vect_str} fitted parameter does not contain " \
                            f"the correct number of elements for {name}."
                        assert vect.shape == (d, 1), \
                            f"{vect_str} fitted parameter is not of {(d, 1)}" \
                            f" shape for {name}."
                        assert np.isnan(vect).sum() == 0, \
                            f"{vect_str} fitted parameter contains nan " \
                            f"values for {name}."

                for mat_str in matrix_attributes:
                    if mat_str in dir(params):
                        mat = eval(f'params.{mat_str}')
                        assert isinstance(mat, np.ndarray), \
                            f"{mat_str} fitted parameter is not an array " \
                            f"for {name}."
                        assert mat.shape == (d, d), \
                            f"{mat_str} fitted parameter is not of {(d, d)} " \
                            f"shape for {name}."
                        assert np.isnan(mat).sum() == 0, \
                            f"{mat_str} fitted parameter contains nan " \
                            f"values for {name}."

                for scale_str in scale_attributes:
                    if scale_str in dir(params):
                        scale = eval(f'params.{scale_str}')
                        assert (isinstance(scale, float)
                                or isinstance(scale, int)), \
                            f"{scale_str} fitted parameter is not a scalar " \
                            f"value for {name}."
                        assert not np.isnan(scale),\
                            f"{scale_str} fitted parameter is nan for {name}."

                for obj_str in to_obj_attributes:
                    assert f'to_{obj_str}' in dir(params), \
                        f"to_{obj_str} attribute does not exist for {name}"
                    obj_target_type = eval(obj_str)
                    obj = eval(f'params.to_{obj_str}')
                    assert isinstance(obj, obj_target_type), \
                        f"to_{obj_str} attribute does not return a " \
                        f"{obj_target_type} for {name}."
                    assert len(obj) == len(params), \
                        f"to_{obj_str} attribute does not contain the " \
                        f"correct number of parameters for {name}."

                # testing we can fit distribution using parameters object.
                params_fitted = dist.fit(params=params)
                assert issubclass(type(params_fitted),
                                  FittedContinuousMultivariate), \
                    f"{name} is not fitted to a child class of " \
                    f"FittedContinuousMultivariate."

                # testing we can fit distribution using tuple object.
                tuple_fitted = dist.fit(params=params.to_tuple)
                assert issubclass(type(tuple_fitted),
                                  FittedContinuousMultivariate), \
                    f"{name} is not fitted to a child class of " \
                    f"FittedContinuousMultivariate."

        # testing for errors if incorrect params object provided
        with pytest.raises(
                TypeError,
                match=f"if params provided, must be a "
                      f"{dist._params_obj} type or tuple of length "
                      f"{dist.num_params}"):
            dist.fit(params=range(1000))


@pytest.mark.test_local_only
def test_prefit_logpdf_pdf_cdf_mc_cdfs(
        mvt_continuous_data, mvt_discrete_data, pd_mvt_continuous_data,
        pd_mvt_discrete_data, mv_dists_to_test, params_2d):
    """Testing the logpdf, pdf, cdf and mc-cdf functions of pre-fit
    multivariate distributions."""
    eps: float = 10 ** -5
    num_generate: int = 10
    cdf_num: int = 10

    for name in mv_dists_to_test:
        dist, _, params = get_dist(name, params_2d, mvt_continuous_data)
        for func_str in ('logpdf', 'pdf', 'mc_cdf'): #, 'cdf'):
            func: Callable = eval(f'dist.{func_str}')
            datasets = (mvt_continuous_data[:cdf_num, :],
                        mvt_discrete_data[:cdf_num, :],
                        pd_mvt_continuous_data.iloc[:cdf_num, :],
                        pd_mvt_discrete_data.iloc[:cdf_num, :]) \
                if func_str == 'cdf' else (mvt_continuous_data,
                                           mvt_discrete_data,
                                           pd_mvt_continuous_data,
                                           pd_mvt_discrete_data)

            for data in datasets:
                output = func(x=data, params=params, match_datatype=True,
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
            new_dataset: np.ndarray = np.zeros((n, d+1))
            with pytest.raises(
                    ValueError, match="Dimensions implied by parameters do "
                                      "not match those of the dataset."):
                func(x=new_dataset, params=params, num_generate=num_generate)


def test_prefit_rvs(mv_dists_to_test, params_2d, mvt_continuous_data):
    """Testing the rvs functions of pre-fit multivariate distributions."""
    for name in mv_dists_to_test:
        dist, _, params = get_dist(name, params_2d, mvt_continuous_data)
        for size in (1, 2, 5, 101):
            rvs = dist.rvs(size=size, params=params)

            # checking correct type
            assert isinstance(rvs, np.ndarray), \
                f"pre-fit rvs values for {name} are not contained in an array."

            # checking correct shape
            assert rvs.shape[0] == size, \
                f"pre-fit rvs for {name} did not generate the correct " \
                f"number of pseudo-samples."

            # checking for nan values
            assert np.isnan(rvs).sum() == 0, \
                f"nan values present in {name} pre-fit rvs."


def test_prefit_scalars(mvt_continuous_data, mvt_discrete_data,
                        pd_mvt_continuous_data, pd_mvt_discrete_data,
                        mv_dists_to_test, params_2d):
    """Testing the likelihood, loglikelihood, AIC and BIC functions of
    multivariate pre-fit distributions."""
    for name in mv_dists_to_test:
        dist, _, params = get_dist(name, params_2d, mvt_continuous_data)
        for func_str in ('likelihood', 'loglikelihood', 'aic', 'bic'):
            func: Callable = eval(f'dist.{func_str}')
            for data in (mvt_continuous_data, mvt_discrete_data,
                         pd_mvt_continuous_data, pd_mvt_discrete_data):
                value = func(data=data, params=params)

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
                func(data=new_dataset, params=params)


def test_prefit_plots(params_2d, params_3d, mvt_continuous_data):
    """Testing the marginal_pairplot, pdf_plot, cdf_plot and mc_cdf_plot
    methods of pre-fit multivariate distributions."""
    mvt_continuous_data_3d: np.ndarray = scipy.stats.multivariate_normal.rvs(
        size=(mvt_continuous_data.shape[0], 3))
    kwargs: dict = {'num_points': 2, 'num_generate': 10, 'mc_num_generate': 10,
                    'show': False, 'show_progress': False}
    for name in ('mvt_normal', ):
        dist, _, dist_params_2d = get_dist(name, params_2d,
                                           mvt_continuous_data)
        _, _, dist_params_3d = get_dist(name, params_3d,
                                        mvt_continuous_data_3d)
        for func_str in ('marginal_pairplot', 'pdf_plot',
                         'cdf_plot', 'mc_cdf_plot'):
            func: Callable = eval(f'dist.{func_str}')

            # testing 3d plots
            if func_str == 'marginal_pairplot':
                func(params=dist_params_3d, **kwargs)
                plt.close()
            else:
                with pytest.raises(NotImplementedError,
                                   match=f"{func_str} is not "
                                         f"implemented when the number of "
                                         f"variables is not 2."):
                    func(params=dist_params_3d, **kwargs)

            # testing 2d plots
            func(params=dist_params_2d, **kwargs)
            plt.close()


def test_prefit_names(mv_dists_to_test):
    """Testing that name of pre-fit multivariate distributions is a string."""
    for name in mv_dists_to_test:
        dist = eval(name)
        assert isinstance(dist.name, str), f"name of {name} is not a string."


def test_prefit_integers(mv_dists_to_test, params_2d, mvt_continuous_data):
    """Testing the num_params and num_scalar_params of pre-fit multivariate
    distributions."""
    for name in mv_dists_to_test:
        dist, _, params = get_dist(name, params_2d, mvt_continuous_data)

        for func_str in ('num_params', 'num_scalar_params(d=3)'):
            value = eval(f"dist.{func_str}")
            assert isinstance(value, int), \
                f"{func_str} of {name} is not an integer."
            assert value >= 0, f"{func_str} of {name} is negative."
        assert dist.num_params == len(params), \
            f"num_params of {name} does not match the length of its params " \
            f"object."
