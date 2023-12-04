# Contains tests for Pre-Fit SklarPy copula models
import numpy as np
from typing import Callable
import pytest
import scipy.stats
import matplotlib.pyplot as plt

from sklarpy.copulas import *
from sklarpy.copulas._prefit_dists import PreFitCopula
from sklarpy.copulas._fitted_dists import FittedCopula
from sklarpy.utils._errors import FitError
from sklarpy.utils._params import Params
from sklarpy.tests.copulas.helpers import get_dist


def test_correct_type():
    """Testing copula distributions are all SklarPy objects."""
    for name in distributions_map['all']:
        copula = eval(name)
        assert issubclass(type(copula), PreFitCopula), \
            f"{name} is not a child class of PreFitCopula."


def test_fit(all_mvt_data):
    """Testing we can fit copula distributions to data."""
    for data in all_mvt_data.values():
        mfitter: MarginalFitter = MarginalFitter(data)
        mfitter.fit()

        for name in distributions_map['all']:
            copula = eval(name)

            for method in copula._mv_object._DATA_FIT_METHODS:
                # testing all fit methods
                if isinstance(data, np.ndarray):
                    d: int = data.shape[1]
                else:
                    d: int = len(data.columns)

                try:
                    # fitting to data
                    fcopula = copula.fit(data=data, method=method)

                    # testing fitted to correct type
                    assert issubclass(type(fcopula), FittedCopula), \
                        f"{name} is not fitted to a child class of " \
                        f"FittedCopula."

                    # testing parameters object (testing already done in
                    # multivariate tests, so not repeating all).
                    copula_params = fcopula.copula_params
                    assert issubclass(type(copula_params), Params), \
                        f"{name} fitted copula parameters are not a child " \
                        f"class of Params."
                    assert len(copula_params) > 0, \
                        f"{name} fitted copula parameter object is empty."

                    # testing we can fit distribution using parameters object.
                    params_fcopula = copula.fit(data=data,
                                                copula_params=copula_params)
                    assert issubclass(type(params_fcopula), FittedCopula), \
                        f"{name} is not a child class of FittedCopula when " \
                        f"fitted to data and copula params."

                    # testing we can fit distribution using MarginalFitter.

                    mfitter_fcopula = copula.fit(data=data, mdists=mfitter)
                    assert issubclass(type(mfitter_fcopula), FittedCopula), \
                        f"{name} is not a child class of FittedCopula when " \
                        f"fitted to data and a MarginalFitter object."

                    # testing we can fit only using copula params and
                    # MarginalFitter.
                    params_mfitter_fcopula = copula.fit(
                        copula_params=copula_params, mdists=mfitter)
                    assert issubclass(type(params_mfitter_fcopula),
                                      FittedCopula
                                      ), f"{name} is not a child class " \
                                         f"of FittedCopula when fitted to" \
                                         f" data and a MarginalFitter object."

                except FitError:
                    if method != 'inverse_kendall_tau':
                        raise
                except RuntimeError:
                    pass


def test_prefit_logpdf_pdf_cdf_mc_cdfs(all_mvt_data, copula_params_2d,
                                       all_mdists_2d):
    """Testing the logpdf, pdf, cdf and mc-cdf functions of pre-fit copula
    models."""
    eps: float = 10 ** -5
    num_generate: int = 10
    cdf_num: int = 10

    for dataset_name, data in all_mvt_data.items():
        mdists = all_mdists_2d[dataset_name]
        for name in distributions_map['all']:
            copula, _, copula_params = get_dist(name, copula_params_2d,
                                                mdists, data)

            for func_str in ('logpdf', 'pdf', 'mc_cdf'): #, 'cdf'):
                func: Callable = eval(f"copula.{func_str}")
                if func_str == 'cdf':
                    func_data = data[:cdf_num, :].copy() \
                        if 'pd' not in dataset_name else data.iloc[:cdf_num, :]
                else:
                    func_data = data

                # getting values to test
                output = func(x=data, copula_params=copula_params,
                              mdists=mdists, match_datatype=True,
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
                                   match='Dimensions implied by parameters do '
                                         'not match those of the dataset.'):
                    func(x=new_dataset, copula_params=copula_params,
                         mdists=mdists, match_datatype=True,
                         num_generate=num_generate, show_progress=False)


def test_prefit_copula_logpdf_pdf_cdf_mc_cdfs(all_mvt_uniform_data,
                                              copula_params_2d, all_mdists_2d):
    """Testing the copula-logpdf, copula-pdf, copula-cdf and copula-mc-cdf
    functions of pre-fit copula models."""
    eps: float = 10 ** -5
    num_generate: int = 10
    cdf_num: int = 10

    mdists = all_mdists_2d['mvt_mixed']
    for dataset_name, data in all_mvt_uniform_data.items():
        for name in distributions_map['all']:
            copula, _, copula_params = get_dist(name, copula_params_2d,
                                                mdists, data)

            for func_str in ('copula_logpdf', 'copula_pdf',
                             'copula_mc_cdf'):  # , 'copula_cdf'):
                func: Callable = eval(f"copula.{func_str}")
                if func_str == 'copula_cdf':
                    func_data = data[:cdf_num, :].copy() \
                        if 'pd' not in dataset_name else data.iloc[:cdf_num, :]
                else:
                    func_data = data

                # getting values to test
                output = func(u=data, copula_params=copula_params,
                              mdists=mdists, match_datatype=True,
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
                    func(u=new_dataset, copula_params=copula_params,
                         mdists=mdists, match_datatype=True,
                         num_generate=num_generate, show_progress=False)


def test_prefit_rvs(all_mvt_data, copula_params_2d, all_mdists_2d):
    """Testing the rvs and copula-rvs functions of pre-fit copula models."""
    eps: float = 10 ** -5
    dataset_name: str = 'mvt_mixed'
    data: np.ndarray = all_mvt_data[dataset_name]
    mdists: dict = all_mdists_2d[dataset_name]

    for name in distributions_map['all']:
        copula, _, copula_params = get_dist(name, copula_params_2d,
                                            mdists, data)
        for func_str in ('rvs', 'copula_rvs'):
            func: Callable = eval(f"copula.{func_str}")
            for size in (1, 2, 5, 101):
                rvs = func(size=size, copula_params=copula_params,
                           mdists=mdists)

                # checking correct type
                assert isinstance(rvs, np.ndarray), \
                    f"pre-fit {func_str} values for {name} are not contained" \
                    f" in an array."

                # checking correct shape
                assert rvs.shape[0] == size, \
                    f"pre-fit {func_str} for {name} did not generate the " \
                    f"correct number of pseudo-samples."

                # checking for nan values
                assert np.isnan(rvs).sum() == 0, \
                    f"nan values present in {name} pre-fit {func_str}."

                # function specific checks
                if func_str == 'copula_rvs' and size > 1:
                    assert np.all((1 - rvs > -eps) & (rvs > -eps)), \
                        f"pre-fit copula-rvs are not in the [0, 1] cdf space."


def test_prefit_scalars(all_mvt_data, copula_params_2d, all_mdists_2d):
    """Testing the likelihood, loglikelihood, AIC and BIC functions of
    pre-fit copula models."""
    for dataset_name, data in all_mvt_data.items():
        mdists = all_mdists_2d[dataset_name]
        for name in distributions_map['all']:
            copula, _, copula_params = get_dist(name, copula_params_2d,
                                                mdists, data)
            for func_str in ('likelihood', 'loglikelihood', 'aic', 'bic'):
                func: Callable = eval(f"copula.{func_str}")
                value = func(data=data, copula_params=copula_params,
                             mdists=mdists)

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
                        match='Dimensions implied by parameters do not match '
                              'those of the dataset.'):
                    func(data=new_dataset, copula_params=copula_params,
                         mdists=mdists)


def test_prefit_integers(all_mvt_data, copula_params_2d, all_mdists_2d):
    """Testing the num_marginal_params, num_copula_params,
    num_scalar_params and num_params functions of pre-fit copula models."""
    dataset_name: str = 'mvt_mixed'
    data: np.ndarray = all_mvt_data[dataset_name]
    mdists = all_mdists_2d[dataset_name]

    for name in distributions_map['all']:
        copula, _, copula_params = get_dist(name, copula_params_2d, mdists,
                                            data)
        for func_str in ("num_scalar_params", "num_copula_params",
                         "num_marginal_params", "num_params"):
            func: Callable = eval(f"copula.{func_str}")
            value = func(copula_params=copula_params, mdists=mdists)

            assert isinstance(value, int), \
                f"{func_str} of {name} is not an integer."
            assert value >= 0, f"{func_str} of {name} is negative."


def test_prefit_plots(all_mvt_data, copula_params_2d, copula_params_3d,
                      all_mdists_2d, all_mdists_3d):
    """Testing the marginal_pairplot, pdf_plot, cdf_plot, mc_cdf_plot,
    copula_pdf_plot, copula_cdf_plot and copula_mc_cdf_plot methods of
    pre-fit copula models."""
    num_generate: int = 10
    mc_num_generate: int = num_generate
    num_points = 2

    mvt_data_2d: np.ndarray = all_mvt_data['mvt_mixed']
    mdists_2d: dict = all_mdists_2d['mvt_mixed']

    mvt_data_3d: np.ndarray = scipy.stats.multivariate_normal.rvs(
        size=(mvt_data_2d.shape[0], 3))
    mdists_3d: dict = all_mdists_3d['mvt_continuous']

    for name in ('gaussian_copula',):
        copula, _, cparams_2d = get_dist(name, copula_params_2d, mdists_2d,
                                         mvt_data_2d)

        if name != 'frank_copula':
            _, _, cparams_3d = get_dist(name, copula_params_3d, mdists_3d,
                                             mvt_data_3d)

        for func_str in ('marginal_pairplot', 'pdf_plot', 'cdf_plot',
                         'mc_cdf_plot', 'copula_pdf_plot', 'copula_cdf_plot',
                         'copula_mc_cdf_plot'):
            func: Callable = eval(f"copula.{func_str}")

            # testing 3d plots
            if name == 'frank_copula':
                pass
            elif func_str == 'marginal_pairplot':
                func(copula_params=cparams_3d, mdists=mdists_3d, show=False,
                     num_generate=num_generate)
                plt.close()
            else:
                with pytest.raises(NotImplementedError,
                                   match=f"{func_str} is not "
                                         f"implemented when the number of "
                                         f"variables is not 2."):
                    func(copula_params=cparams_3d, mdists=mdists_3d,
                         show=False, show_progress=False,
                         num_generate=num_generate, num_points=num_points)

            # testing 2d plots
            func(copula_params=cparams_2d, mdists=mdists_2d, show=False,
                 show_progress=False, num_generate=num_generate,
                 mc_num_generate=mc_num_generate, num_points=num_points)
            plt.close()


def test_prefit_names():
    for name in distributions_map['all']:
        copula = eval(name)
        assert isinstance(copula.name, str), f"name of {name} is not a string."
