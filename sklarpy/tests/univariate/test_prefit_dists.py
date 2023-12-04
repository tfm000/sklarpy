# Contains tests for Pre-Fit SklarPy univariate distributions
import numpy as np
import pandas as pd
from typing import Callable
import pytest
import matplotlib.pyplot as plt

from sklarpy.univariate import *
from sklarpy.univariate._prefit_dists import PreFitUnivariateBase
from sklarpy.tests.univariate.helpers import get_data, get_target_fit, get_dist


def test_correct_type():
    """Testing distributions are all SklarPy objects."""
    for name in distributions_map['all']:
        dist = eval(name)
        assert issubclass(type(dist), PreFitUnivariateBase), \
            f"{name} is not a child class of PreFitUnivariateBase."


def test_fit_to_data(discrete_data, continuous_data, dists_to_test):
    """Testing we can fit distributions to data."""
    for name in dists_to_test:
        data: np.ndarray = get_data(name, continuous_data, discrete_data)
        target_fit = get_target_fit(name, continuous_data, discrete_data)

        try:
            dist = eval(name)
            fitted = dist.fit(data)
            params: tuple = fitted.params
            assert isinstance(fitted, target_fit), f"{name} fitted to " \
                                                   f"wrong distribution type."
            assert isinstance(params, tuple), f"{name} fitted to wrong " \
                                              f"params type."

            if name in distributions_map['all numerical']:
                assert len(params) == 0, \
                    f"{name} is numerical, but params not empty."
            else:
                assert len(params) > 0, \
                    f"{name} is parametric, but params empty."
        except RuntimeError:
            continue


def test_prefit_name():
    """Testing the name of pre-fit distributions is a string."""
    for name in distributions_map['all']:
        dist = eval(name)
        assert isinstance(dist.name, str), f"name of {name} is not a string."


def test_prefit_continuous_or_parametric():
    """Testing dists are continuous or discrete."""
    for name in distributions_map['all']:
        dist = eval(name)
        s: str = dist.continuous_or_discrete
        assert isinstance(s, str), f"continuous_or_discrete is not a string " \
                                   f"for {name}."
        assert s in ('continuous', 'discrete'), f"{name} is not continuous " \
                                                f"or discrete."


def test_fit_to_params(discrete_data, continuous_data, dists_to_test):
    """Testing we can fit distributions to user specified parameters
    """
    for name in dists_to_test:
        data: np.ndarray = get_data(name, continuous_data, discrete_data)
        target_fit = get_target_fit(name, continuous_data, discrete_data)
        try:
            dist, fitted, params = get_dist(name, data)
        except RuntimeError:
            continue

        # testing numerical distributions
        if name in distributions_map['all numerical']:
            with pytest.raises(TypeError):
                dist.fit(params=params)

        # testing parametric distributions
        else:
            dist = eval(name)
            fitted = dist.fit(data)
            params: tuple = fitted.params
            param_fitted = dist.fit(params=params)
            assert isinstance(param_fitted, target_fit), \
                f"{name} fitted to wrong distribution type using params."


def test_prefit_pdfs(discrete_data, continuous_data, dists_to_test):
    """Testing the pdf functions of pre-fit distributions."""
    for name in dists_to_test:
        data: np.ndarray = get_data(name, continuous_data, discrete_data)
        try:
            dist, fitted, params = get_dist(name, data)
        except RuntimeError:
            continue

        # testing numerical distributions
        if name in distributions_map['all numerical']:
            with pytest.raises(NotImplementedError,
                               match='pdf not implemented for non-fitted '
                                     'numerical distributions.'):
                dist.pdf(data, params)

        # testing parametric distributions
        else:
            # checking pdf values are the correct data-type
            pdf_values: np.ndarray = dist.pdf(data, params)
            assert isinstance(pdf_values, np.ndarray), \
                f"pdf values for {name} are not contained in a numpy array"

            # checking same number of pdf values as input
            assert pdf_values.size == data.size, \
                f"number pdf values for {name} do not match the number of " \
                f"inputs"

            # checking for nan values
            assert np.isnan(pdf_values).sum() == 0, \
                f"nan values present in {name} pre-fit pdf"

            # checking all pdf values are non-negative
            assert np.all(pdf_values >= 0), \
                f"pdf values present in {name} are negative"


def test_prefit_cdfs(discrete_data, continuous_data, dists_to_test):
    """Testing the cdf functions of pre-fit distributions."""
    eps: float = 10 ** -5

    for name in dists_to_test:
        data: np.ndarray = get_data(name, continuous_data, discrete_data)
        try:
            dist, fitted, params = get_dist(name, data)
        except RuntimeError:
            continue

        for func_str in ('cdf', 'cdf_approx'):
            func: Callable = eval(f'dist.{func_str}')

            # testing numerical distributions
            if name in distributions_map['all numerical']:
                with pytest.raises(NotImplementedError,
                                   match=f'{func_str} not implemented '
                                         f'for non-fitted numerical '
                                         f'distributions.'):
                    func(data, params)

            # testing parametric distributions
            else:
                # checking cdf values are the correct data-type
                cdf_values: np.ndarray = func(data, params)
                assert isinstance(cdf_values, np.ndarray), \
                    f"{func_str} values for {name} are not contained in a " \
                    f"numpy array"

                # checking same number of cdf values as input
                assert cdf_values.size == data.size, \
                    f"number {func_str} values for {name} do not " \
                    f"match the number of inputs"

                # checking for nan values
                if np.isnan(cdf_values).sum() != 0:
                    breakpoint()
                assert np.isnan(cdf_values).sum() == 0, \
                    f"nan values present in {name} pre-fit {func_str}"

                # checking cdf values are non-decreasing
                sorted_data: np.ndarray = data.copy()
                sorted_data.sort()
                sorted_cdf_values: np.ndarray = dist.cdf(sorted_data, params)
                neighbour_difference: np.ndarray = sorted_cdf_values[1:] \
                                                   - sorted_cdf_values[:-1]
                negative_values: np.ndarray = neighbour_difference[
                    np.where(neighbour_difference < 0)]
                if negative_values.size > 0:
                    # we may have negative_values which are very small and
                    # likely a float rounding error.
                    assert np.all(negative_values > - eps), \
                        f"{func_str }values of {name} are not monotonically " \
                        f"increasing."

                # checking extremes
                assert abs(float(dist.cdf(np.inf, params)) - 1.0) <= eps, \
                    f"{func_str} of {name} is not 1.0 at infinity"
                assert float(dist.cdf(-np.inf, params)) <= eps, \
                    f"{func_str} of {name} is not 0.0 at -infinity"


def test_prefit_ppfs(uniform_data, discrete_data, continuous_data,
                     dists_to_test):
    """Testing the ppf functions of pre-fit distributions."""
    for name in dists_to_test:
        data: np.ndarray = get_data(name, continuous_data, discrete_data)
        try:
            dist, fitted, params = get_dist(name, data)
        except RuntimeError:
            continue

        for func_str in ('ppf', 'ppf_approx'):
            func: Callable = eval(f'dist.{func_str}')

            # testing numerical distributions
            if name in distributions_map['all numerical']:
                with pytest.raises(NotImplementedError,
                                   match=f'{func_str} not implemented '
                                         f'for non-fitted numerical '
                                         f'distributions.'):
                    func(data, params)

            # testing parametric distributions
            else:
                ppf_values: np.ndarray = func(uniform_data, params)

                # checking correct type
                assert isinstance(ppf_values, np.ndarray), \
                    f"{func_str} values for {name} are not contained in a " \
                    f"numpy array"

                # checking same number of ppf values as input
                assert ppf_values.size == uniform_data.size, \
                    f"number {func_str} values for {name} do not match the " \
                    f"number of inputs"

                # checking for nan values
                assert np.isnan(ppf_values).sum() == 0, \
                    f"nan values present in {name} pre-fit {func_str}"


def test_prefit_supports(discrete_data, continuous_data, dists_to_test):
    """Testing the support functions of pre-fit distributions."""
    for name in dists_to_test:
        data: np.ndarray = get_data(name, continuous_data, discrete_data)
        try:
            dist, fitted, params = get_dist(name, data)
        except RuntimeError:
            continue

        # testing numerical distributions
        if name in distributions_map['all numerical']:
            with pytest.raises(NotImplementedError,
                               match="support not implemented for non-fitted "
                                     "numerical distributions."):
                dist.support(params)

        # testing parametric distributions
        else:
            support: tuple = dist.support(params)

            # checking support is a tuple
            assert isinstance(support, tuple), \
                f"support values for {name} are not contained in a tuple"

            # checking for nan values
            assert np.isnan(support).sum() == 0, \
                f"nan values present in {name} pre-fit support"

            # checking only two values in support
            assert len(support) == 2, \
                f"incorrect number of values in prefit-support for {name}"

            # checking lb < ub
            assert support[0] < support[1], \
                f"lb < ub is not satisfied in prefit-support for {name}"


def test_prefit_rvs(discrete_data, continuous_data, dists_to_test):
    """Testing the rvs functions of pre-fit distributions."""
    for name in dists_to_test:
        data: np.ndarray = get_data(name, continuous_data, discrete_data)
        try:
            dist, fitted, params = get_dist(name, data)
        except RuntimeError:
            continue

        # testing numerical distributions
        if name in distributions_map['all numerical']:
            with pytest.raises(NotImplementedError,
                               match="rvs not implemented for non-fitted "
                                      "numerical distributions."):
                dist.rvs((5, 3), params)

        # testing parametric distributions
        else:
            num: int = 10
            for ppf_approx in (True, False):
                for shape in ((num, ), (num, 2), (num, 5), (num, 13)):
                    rvs_values: np.ndarray = dist.rvs(shape, params,
                                                      ppf_approx)

                    # checking correct type
                    assert isinstance(rvs_values, np.ndarray), \
                        f"pre-fit rvs values for {name} are not contained " \
                        f"in an np.ndarray when ppf_approx = {ppf_approx}"

                    # checking for nan values
                    assert np.isnan(rvs_values).sum() == 0, \
                        f"nan values present in {name} pre-fit rvs when " \
                        f"ppf_approx = {ppf_approx}"

                    # checking correct shape
                    assert rvs_values.shape == shape, \
                        f"incorrect shape generated for pre-fit rvs for " \
                        f"{name} when ppf_approx = {ppf_approx}. Target " \
                        f"shape is {shape}, generated shape is " \
                        f"{rvs_values.shape}"


def test_prefit_logpdfs(discrete_data, continuous_data,
                                   dists_to_test):
    """Testing the log-pdf functions of pre-fit distributions."""
    for name in dists_to_test:
        data: np.ndarray = get_data(name, continuous_data, discrete_data)
        try:
            dist, fitted, params = get_dist(name, data)
        except RuntimeError:
            continue

        # testing numerical distributions
        if name in distributions_map['all numerical']:
            with pytest.raises(NotImplementedError,
                               match="logpdf not implemented for non-fitted "
                                      "numerical distributions."):
                dist.logpdf(data, params)

        # testing parametric distributions
        else:
            # checking log-pdf values are the correct data-type
            logpdf_values: np.ndarray = dist.logpdf(data, params)
            assert isinstance(logpdf_values, np.ndarray), \
                f"log-pdf values for {name} are not contained in a numpy array"

            # checking same number of logpdf values as input
            assert logpdf_values.size == data.size, \
                f"number of log-pdf values for {name} do not match the " \
                f"number of inputs"

            # checking for nan values
            assert np.isnan(logpdf_values).sum() == 0, \
                f"nan values present in {name} pre-fit log-pdf"


def test_prefit_scalars(discrete_data, continuous_data, dists_to_test):
    """Testing the likelihood, loglikelihood, AIC, BIC and SSE functions of
    pre-fit distributions."""
    for name in dists_to_test:
        data: np.ndarray = get_data(name, continuous_data, discrete_data)
        try:
            dist, fitted, params = get_dist(name, data)
        except RuntimeError:
            continue

        for func_str in ('likelihood', 'loglikelihood', 'aic', 'bic', 'sse'):
            func: Callable = eval(f"dist.{func_str}")

            # testing numerical distributions
            if name in distributions_map['all numerical']:
                with pytest.raises(NotImplementedError,
                                   match=f"{func_str} not implemented for no"
                                         f"n-fitted numerical distributions."):
                    func(data, params)

            # testing parametric distributions
            else:
                # checking correct type
                value: float = func(data, params)
                assert isinstance(value, float), f"{func_str} for {name} is " \
                                                 f"not a float"

                # checking valid number
                assert not np.isnan(value), f"{func_str} for {name} is is nan"

                if func_str in ("likelihood", "sse"):
                    assert value >= 0, f"{func_str} for {name} is negative."


def test_prefit_gofs(discrete_data, continuous_data, dists_to_test):
    """Testing the gof functions of pre-fit distributions"""
    for name in dists_to_test:
        data: np.ndarray = get_data(name, continuous_data, discrete_data)
        try:
            dist, fitted, params = get_dist(name, data)
        except RuntimeError:
            continue

        # testing numerical distributions
        if name in distributions_map['all numerical']:
            with pytest.raises(NotImplementedError,
                               match=f"gof not implemented for non-fitted "
                                     f"numerical distributions."):
                dist.gof(data, params)

        # testing parametric distributions
        else:
            # checking gof object is a dataframe
            gof: pd.DataFrame = dist.gof(data, params)
            assert isinstance(gof, pd.DataFrame), \
                f"gof for {name} is not a pandas dataframe."

            # checking gof is non-empty
            assert len(gof) > 0, f"gof for {name} is empty."


def test_prefit_plots(discrete_data, continuous_data, dists_to_test):
    """Testing the plot functions of pre-fit distributions"""
    for name in dists_to_test:
        data: np.ndarray = get_data(name, continuous_data, discrete_data)
        try:
            dist, fitted, params = get_dist(name, data)
        except RuntimeError:
            continue

        # testing numerical distributions
        if name in distributions_map['all numerical']:
            with pytest.raises(NotImplementedError,
                               match=f"plot not implemented for non-fitted "
                                     f"numerical distributions."):
                dist.plot(data, params)

        # testing parametric distributions
        else:
            # checking we can plot without errors
            dist.plot(params, show=False)
            plt.close()
