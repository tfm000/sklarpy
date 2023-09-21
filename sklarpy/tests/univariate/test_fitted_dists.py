# Contains tests for fitted SklarPy univariate distributions
import numpy as np
from typing import Callable

from sklarpy.tests.univariate.helpers import get_data, get_fitted_dict


def test_fitted_pdfs(discrete_data, continuous_data, dists_to_test):
    """Testing the pdf functions of fitted univariate distributions."""
    for name in dists_to_test:
        data: np.ndarray = get_data(name, continuous_data, discrete_data)
        try:
            fitted_dict: dict = get_fitted_dict(name, data)
        except RuntimeError:
            continue

        # testing when fitting to both data and parameters
        for fitted_type, fitted in fitted_dict.items():
            # checking pdf values are the correct data-type
            pdf_values: np.ndarray = fitted.pdf(data)
            assert isinstance(pdf_values, np.ndarray), \
                f"pdf values for {name} are not contained in a numpy array " \
                f"when {fitted_type}"

            # checking same number of pdf values as input
            assert pdf_values.size == data.size, \
                f"number pdf values for {name} do not match the number of " \
                f"inputs when {fitted_type}"

            # checking for nan values
            assert np.isnan(pdf_values).sum() == 0, \
                f"nan values present in {name} pdf when {fitted_type}"

            # checking all pdf values are non-negative
            assert np.all(pdf_values >= 0), \
                f"pdf values present in {name} are negative when {fitted_type}"


def test_fitted_cdfs(discrete_data, continuous_data, dists_to_test):
    """Testing the cdf functions of fitted univariate distributions."""
    eps: float = 10 ** -5
    for name in dists_to_test:
        data: np.ndarray = get_data(name, continuous_data, discrete_data)
        try:
            fitted_dict: dict = get_fitted_dict(name, data)
        except RuntimeError:
            continue

        # testing when fitting to both data and parameters
        for fitted_type, fitted in fitted_dict.items():
            for func_str in ('cdf', 'cdf_approx'):
                func: Callable = eval(f'fitted.{func_str}')

                # checking cdf values are the correct data-type
                cdf_values: np.ndarray = func(data)
                assert isinstance(cdf_values, np.ndarray), \
                    f"{func_str} values for {name} are not contained in a " \
                    f"numpy array when {fitted_type}"

                # checking same number of cdf values as input
                assert cdf_values.size == data.size, \
                    f"number {func_str} values for {name} do not match the " \
                    f"number of inputs when {fitted_type}"

                # checking for nan and inf values
                assert ((np.isnan(cdf_values).sum() == 0)
                        and (not np.all(np.isinf(cdf_values)))), \
                    f"nan or inf values present in {name} {func_str} when " \
                    f"{fitted_type}"

                # checking cdf values are non-decreasing
                sorted_data: np.ndarray = data.copy()
                sorted_data.sort()
                sorted_cdf_values: np.ndarray = fitted.cdf(sorted_data)
                neighbour_difference: np.ndarray = sorted_cdf_values[1:] \
                                                   - sorted_cdf_values[:-1]
                negative_values: np.ndarray = neighbour_difference[
                    np.where(neighbour_difference < 0)]
                if negative_values.size > 0:
                    # we may have negative_values which are very small and
                    # likely a float rounding error.
                    assert np.all(negative_values > - eps), \
                        f"{func_str} values of {name} are not monotonically " \
                        f"increasing."

                # checking extremes
                assert abs(float(fitted.cdf(np.inf) - 1.0)) <= eps, \
                    f"{func_str} of {name} is not 1.0 at infinity when " \
                    f"{fitted_type}"
                assert float(fitted.cdf(-np.inf)) <= eps, \
                    f"{func_str} of {name} is not 0.0 at -infinity when " \
                    f"{fitted_type}"


def test_fitted_ppfs(uniform_data, discrete_data, continuous_data,
                    dists_to_test):
    """Testing the ppf functions of fitted univariate distributions."""
    for name in dists_to_test:
        data: np.ndarray = get_data(name, continuous_data, discrete_data)
        try:
            fitted_dict: dict = get_fitted_dict(name, data)
        except RuntimeError:
            continue

        # testing when fitting to both data and parameters
        for fitted_type, fitted in fitted_dict.items():
            for func_str in ('ppf', 'ppf_approx'):
                func: Callable = eval(f'fitted.{func_str}')

                ppf_values: np.ndarray = func(uniform_data)
                # checking correct type
                assert isinstance(ppf_values, np.ndarray), \
                    f"{func_str} values for {name} are not contained in a " \
                    f"numpy array when {fitted_type}"

                # checking same number of ppf values as input
                assert ppf_values.size == uniform_data.size, \
                    f"number {func_str} values for {name} do not match the " \
                    f"number of inputs when {fitted_type}"

                # checking for nan values
                assert np.isnan(ppf_values).sum() == 0, \
                    f"nan values present in {name} {func_str} when " \
                    f"{fitted_type}"


def test_fitted_supports(discrete_data, continuous_data, dists_to_test):
    """Testing the support functions of fitted univariate distributions."""
    for name in dists_to_test:
        data: np.ndarray = get_data(name, continuous_data, discrete_data)
        try:
            fitted_dict: dict = get_fitted_dict(name, data)
        except RuntimeError:
            continue

        # testing when fitting to both data and parameters
        for fitted_type, fitted in fitted_dict.items():
            support: tuple = fitted.support

            # checking support is a tuple
            assert isinstance(support, tuple), \
                f"support values for {name} are not contained in a tuple " \
                f"when {fitted_type}"

            # checking for nan values
            assert np.isnan(support).sum() == 0, \
                f"nan values present in {name} support when {fitted_type}"

            # checking only two values in support
            assert len(support) == 2, \
                f"incorrect number of values in support for {name} when " \
                f"{fitted_type}"

            # checking lb < ub
            assert support[0] < support[1], \
                f"lb < ub is not satisfied in support for {name} when " \
                f"{fitted_type}"


def test_fitted_rvs(discrete_data, continuous_data, dists_to_test):
    """Testing the rvs functions of fitted univariate distributions."""
    for name in dists_to_test:
        data: np.ndarray = get_data(name, continuous_data, discrete_data)
        try:
            fitted_dict: dict = get_fitted_dict(name, data)
        except RuntimeError:
            continue

        # testing when fitting to both data and parameters
        for fitted_type, fitted in fitted_dict.items():
            num: int = 10
            for shape in ((num,), (num, 2), (num, 5), (num, 13, 6)):
                rvs_values: np.ndarray = fitted.rvs(shape)

                # checking correct type
                assert isinstance(rvs_values, np.ndarray), \
                    f"rvs values for {name} are not contained in an " \
                    f"np.ndarray when {fitted_type}"

                # checking for nan values
                assert np.isnan(rvs_values).sum() == 0, \
                    f"nan values present in {name} rvs when {fitted_type}"

                # checking correct shape
                assert rvs_values.shape == shape, \
                    f"incorrect shape generated for rvs for {name} when " \
                    f"{fitted_type}. target shape is {shape}, generated " \
                    f"shape is {rvs_values.shape}"


def test_fitted_logpdfs(discrete_data, continuous_data, dists_to_test):
    """Testing the log-pdf functions of all fitted univariate distributions."""
    for name in dists_to_test:
        data: np.ndarray = get_data(name, continuous_data, discrete_data)
        try:
            fitted_dict: dict = get_fitted_dict(name, data)
        except RuntimeError:
            continue

        # testing when fitting to both data and parameters
        for fitted_type, fitted in fitted_dict.items():
            # checking log-pdf values are the correct data-type
            logpdf_values: np.ndarray = fitted.logpdf(data)
            assert isinstance(logpdf_values, np.ndarray), \
                f"log-pdf values for {name} are not contained in a numpy " \
                f"array when {fitted_type}"

            # checking same number of pdf values as input
            assert logpdf_values.size == data.size, \
                f"number of log-pdf values for {name} do not match the " \
                f"number of inputs when {fitted_type}"

            # checking for nan values
            assert np.isnan(logpdf_values).sum() == 0, \
                f"nan values present in {name} log-pdf when {fitted_type}"


def test_fitted_scalars(discrete_data, continuous_data, dists_to_test):
    """Testing the likelihood, loglikelihood, AIC, BIC and SSE functions of
    fitted distributions."""
    for name in dists_to_test:
        data: np.ndarray = get_data(name, continuous_data, discrete_data)
        try:
            fitted_dict: dict = get_fitted_dict(name, data)
        except RuntimeError:
            continue

        # testing when fitting to both data and parameters
        for fitted_type, fitted in fitted_dict.items():
            for func_str in ('likelihood', 'loglikelihood', 'aic', 'bic',
                             'sse'):
                func: Callable = eval(f"fitted.{func_str}")

                # checking correct type
                value: float = func(data)
                assert isinstance(value, float), f"{func_str} for {name} is " \
                                                 f"not a float"

                # checking valid number
                assert not np.isnan(value), f"{func_str} for {name} is is nan"

                if func_str in ("likelihood", "sse"):
                    assert value >= 0, f"{func_str} for {name} is negative."


# TODO: add other tests