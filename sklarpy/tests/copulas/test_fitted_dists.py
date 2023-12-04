# Contains tests for Fitted SklarPy copula models
import numpy as np
from typing import Callable
import pytest
import matplotlib.pyplot as plt
import scipy.stats

from sklarpy.copulas import distributions_map
from sklarpy.tests.copulas.helpers import get_dist


def test_fitted_logpdf_pdf_cdf_mc_cdfs(all_mvt_data, copula_params_2d,
                                       all_mdists_2d):
    """Testing the logpdf, pdf, cdf and mc-cdf functions of fitted copula
    models."""
    eps: float = 10 ** -5
    num_generate: int = 10
    cdf_num: int = 10

    for dataset_name, data in all_mvt_data.items():
        mdists = all_mdists_2d[dataset_name]
        for name in distributions_map['all']:
            _, fcopula, _ = get_dist(name, copula_params_2d,
                                                mdists, data)

            for func_str in ('logpdf', 'pdf', 'mc_cdf'): #, 'cdf'):
                func: Callable = eval(f"fcopula.{func_str}")
                if func_str == 'cdf':
                    func_data = data[:cdf_num, :].copy() \
                        if 'pd' not in dataset_name else data.iloc[:cdf_num, :]
                else:
                    func_data = data

                # getting values to test
                output = func(x=data, match_datatype=True,
                              num_generate=num_generate, show_progress=False)
                np_output: np.ndarray = np.asarray(output)
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
                with pytest.raises(ValueError,
                                   match='Dimensions implied by parameters do'
                                         ' not match those of the dataset.'):
                    func(x=new_dataset, match_datatype=True,
                         num_generate=num_generate, show_progress=False)


def test_fitted_copula_logpdf_pdf_cdf_mc_cdfs(all_mvt_uniform_data,
                                              copula_params_2d, all_mdists_2d):
    """Testing the copula-logpdf, copula-pdf, copula-cdf and copula-mc-cdf
    functions of fitted copula models."""
    eps: float = 10 ** -5
    num_generate: int = 10
    cdf_num: int = 10

    mdists = all_mdists_2d['mvt_mixed']
    for dataset_name, data in all_mvt_uniform_data.items():
        for name in distributions_map['all']:
            _, fcopula, _ = get_dist(name, copula_params_2d,
                                                mdists, data)

            for func_str in ('copula_logpdf', 'copula_pdf',
                             'copula_mc_cdf'):  # , 'copula_cdf'):
                func: Callable = eval(f"fcopula.{func_str}")
                if func_str == 'copula_cdf':
                    func_data = data[:cdf_num, :].copy() \
                        if 'pd' not in dataset_name else data.iloc[:cdf_num, :]
                else:
                    func_data = data

                # getting values to test
                output = func(u=data, match_datatype=True,
                              num_generate=num_generate, show_progress=False)
                np_output: np.ndarray = np.asarray(output)
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
                with pytest.raises(ValueError,
                                   match='Dimensions implied by parameters do'
                                         ' not match those of the dataset.'):
                    func(u=new_dataset, match_datatype=True,
                         num_generate=num_generate, show_progress=False)


def test_fitted_rvs(all_mvt_data, copula_params_2d, all_mdists_2d):
    """Testing the rvs and copula-rvs functions of fitted copula models."""
    eps: float = 10 ** -5
    dataset_name: str = 'mvt_mixed'
    data: np.ndarray = all_mvt_data[dataset_name]
    mdists: dict = all_mdists_2d[dataset_name]

    for name in distributions_map['all']:
        _, fcopula, _ = get_dist(name, copula_params_2d, mdists, data)
        for func_str in ('rvs', 'copula_rvs'):
            func: Callable = eval(f"fcopula.{func_str}")
            for size in (1, 2, 5, 101):
                rvs = func(size=size)
                np_rvs: np.ndarray = np.asarray(rvs)

                # checking correct type
                assert isinstance(rvs, type(data)), \
                    f"fitted {func_str} values for {name} are not " \
                    f"the correct type."

                # checking correct shape
                assert rvs.shape[0] == size, \
                    f"fitted {func_str} for {name} did not generate the " \
                    f"correct number of pseudo-samples."

                # checking for nan values
                assert np.isnan(np_rvs).sum() == 0, \
                    f"nan values present in {name} fitted {func_str}."

                # function specific checks
                if func_str == 'copula_rvs' and size > 1:
                    assert np.all((1 - np_rvs > -eps) & (np_rvs > -eps)), \
                        f"fitted {name} copula-rvs are not in the [0, 1]" \
                        f" cdf space."


def test_fitted_scalars(all_mvt_data, copula_params_2d, all_mdists_2d):
    """Testing the likelihood, loglikelihood, AIC and BIC functions of
    fitted copula models."""
    for dataset_name, data in all_mvt_data.items():
        mdists = all_mdists_2d[dataset_name]
        for name in distributions_map['all']:
            _, fcopula, _ = get_dist(name, copula_params_2d,
                                                mdists, data)
            for func_str in ('likelihood', 'loglikelihood', 'aic', 'bic'):
                func: Callable = eval(f"fcopula.{func_str}")

                # calling with data
                value = func(data=data)

                # checking we can call without any data
                value2 = func()

                for val in (value, value2):
                    # checking correct type

                    assert isinstance(val, float), \
                        f"{func_str} for {name} is not a float when datatype" \
                        f" is {type(data)}"

                    # checking valid number
                    assert not np.isnan(val), \
                        f"{func_str} for {name} is is nan when datatype is " \
                        f"{type(data)}"

                    if func_str == "likelihood":
                        # checking positive
                        assert val >= 0, \
                            f"{func_str} for {name} is negative when " \
                            f"datatype is {type(data)}."

                # checking error if wrong dimension
                n, d = data.shape
                new_dataset: np.ndarray = np.zeros((n, d + 1))
                with pytest.raises(
                        ValueError,
                        match='Dimensions implied by parameters do not match '
                              'those of the dataset.'):
                    func(data=new_dataset)


def test_fitted_integers(all_mvt_data, copula_params_2d, all_mdists_2d):
    """Testing the num_marginal_params, num_copula_params,
    num_scalar_params and num_params functions of fitted copula models."""
    dataset_name: str = 'mvt_mixed'
    data: np.ndarray = all_mvt_data[dataset_name]
    mdists = all_mdists_2d[dataset_name]

    for name in distributions_map['all']:
        _, fcopula, _s = get_dist(name, copula_params_2d, mdists,
                                            data)
        for func_str in ("num_scalar_params", "num_copula_params",
                         "num_marginal_params", "num_params"):
            func: Callable = eval(f"fcopula.{func_str}")
            value = func()

            assert isinstance(value, int), \
                f"{func_str} of {name} is not an integer."
            assert value >= 0, f"{func_str} of {name} is negative."


def test_fitted_plots(all_mvt_data, copula_params_2d, copula_params_3d,
                      all_mdists_2d, all_mdists_3d):
    """Testing the marginal_pairplot, pdf_plot, cdf_plot, mc_cdf_plot,
    copula_pdf_plot, copula_cdf_plot and copula_mc_cdf_plot methods of
    fitted copula models."""
    num_generate: int = 10
    mc_num_generate: int = num_generate
    num_points = 2

    mvt_data_2d: np.ndarray = all_mvt_data['mvt_mixed']
    mdists_2d: dict = all_mdists_2d['mvt_mixed']

    mvt_data_3d: np.ndarray = scipy.stats.multivariate_normal.rvs(
        size=(mvt_data_2d.shape[0], 3))
    mdists_3d: dict = all_mdists_3d['mvt_continuous']

    for name in ('gaussian_copula', ):
        _, fcopula_2d, _ = get_dist(name, copula_params_2d, mdists_2d,
                                    mvt_data_2d)

        if name != 'frank_copula':
            _, fcopula_3d, _ = get_dist(name, copula_params_3d, mdists_3d,
                                        mvt_data_3d)

        for func_str in ('marginal_pairplot', 'pdf_plot', 'cdf_plot',
                         'mc_cdf_plot', 'copula_pdf_plot', 'copula_cdf_plot',
                         'copula_mc_cdf_plot'):
            func_2d: Callable = eval(f"fcopula_2d.{func_str}")
            func_3d: Callable = eval(f"fcopula_3d.{func_str}")

            # testing 3d plots
            if name == 'frank_copula':
                pass
            elif func_str == 'marginal_pairplot':
                func_3d(show=False, num_generate=num_generate)
                plt.close()
            else:
                with pytest.raises(NotImplementedError,
                                   match=f"{func_str} is not "
                                         f"implemented when the number of "
                                         f"variables is not 2."):
                    func_3d(show=False, show_progress=False,
                            num_generate=num_generate, num_points=num_points)

            # testing 2d plots
            func_2d(show=False, show_progress=False, num_generate=num_generate,
                    mc_num_generate=mc_num_generate, num_points=num_points)
            plt.close()


def test_fitted_names(all_mvt_data, copula_params_2d, all_mdists_2d):
    dataset_name: str = 'mvt_mixed'
    data: np.ndarray = all_mvt_data[dataset_name]
    mdists = all_mdists_2d[dataset_name]

    for name in distributions_map['all']:
        _, fcopula, _s = get_dist(name, copula_params_2d, mdists, data)
        assert isinstance(fcopula.name, str), \
            f"name of {name} is not a string."
