# Contains tests for sklarpy pre-fit univariate distributions
import numpy as np
import pandas as pd
import pytest

from sklarpy.univariate import *
from sklarpy.univariate._fitted_dists import FittedContinuousUnivariate, FittedDiscreteUnivariate
from sklarpy._utils import near_zero
from sklarpy.tests.helpers import get_data, get_dist


def test_fit_to_data(poisson_data, uniform_data):
    """Testing we can fit all distributions to data"""
    for name in distributions_map['all']:
        try:
            if name in distributions_map['all continuous']:
                data: np.ndarray = uniform_data
                target_fit = FittedContinuousUnivariate
            else:
                data: np.ndarray = poisson_data
                target_fit = FittedDiscreteUnivariate

            dist = eval(name)
            fitted = dist.fit(data)
            params: tuple = fitted.params
            assert isinstance(fitted, target_fit), f"{name} fitted to wrong distribution type."
            assert isinstance(params, tuple), f"{name} fitted to wrong params type."
        except RuntimeError:
            continue


def test_prefit_name():
    """Testing the name of all pre-fit distributions is a string."""
    for name in distributions_map['all']:
        dist = eval(name)
        assert isinstance(dist.name, str), f"name of {name} is not a string"


def test_prefit_continuous_or_parametric():
    for name in distributions_map['all']:
        dist = eval(name)
        s: str = dist.continuous_or_discrete
        assert isinstance(s, str), f"continuous_or_discrete is not a string for {name}"
        assert s in ('continuous', 'discrete'), f"{name} is not continuous or discrete"


def test_parametric_fit_to_params(poisson_data, uniform_data):
    """Testing we can fit all parametric distributions to user specified parameters"""
    for name in distributions_map['all parametric']:
        if name in distributions_map['all continuous parametric']:
            data: np.ndarray = uniform_data
            target_fit = FittedContinuousUnivariate
        else:
            data: np.ndarray = poisson_data
            target_fit = FittedDiscreteUnivariate

        try:
            dist = eval(name)
            fitted = dist.fit(data)
            params: tuple = fitted.params
            param_fitted = dist.fit(params=params)
            assert isinstance(param_fitted, target_fit), f"{name} fitted to wrong distribution type using params."
        except RuntimeError:
            continue


def test_prefit_parametric_pdfs(poisson_data, uniform_data):
    """Testing the pdf functions of all pre-fit parametric distributions"""
    for name in distributions_map['all parametric']:
        try:
            data: np.ndarray = get_data(name, uniform_data, poisson_data, ' parametric')
            dist, fitted, params = get_dist(name, data)

            # checking pdf values are the correct data-type
            pdf_values: np.ndarray = dist.pdf(data, params)
            assert isinstance(pdf_values, np.ndarray), f"pdf values for {name} are not contained in a numpy array"

            # checking same number of pdf values as input
            assert pdf_values.size == data.size, f"number pdf values for {name} do not match the number of inputs"

            # checking for nan values
            assert np.isnan(pdf_values).sum() == 0, f"nan values present in {name} pre-fit pdf"

            # checking all pdf values are non-negative
            assert np.all(pdf_values >= 0), f"pdf values present in {name} are negative"
        except RuntimeError:
            continue


def test_prefit_parametric_cdfs(poisson_data, uniform_data):
    """Testing the cdf functions of all pre-fit parametric distributions"""
    for name in distributions_map['all parametric']:
        try:
            data: np.ndarray = get_data(name, uniform_data, poisson_data, ' parametric')
            dist, fitted, params = get_dist(name, data)

            # checking cdf values are the correct data-type
            cdf_values: np.ndarray = dist.cdf(data, params)
            assert isinstance(cdf_values, np.ndarray), f"cdf values for {name} are not contained in a numpy array"

            # checking same number of cdf values as input
            assert cdf_values.size == data.size, f"number cdf values for {name} do not match the number of inputs"

            # checking for nan values
            assert np.isnan(cdf_values).sum() == 0, f"nan values present in {name} pre-fit cdf"

            # checking cdf values are non-decreasing
            sorted_data: np.ndarray = data.copy()
            sorted_data.sort()
            sorted_cdf_values: np.ndarray = dist.cdf(sorted_data, params)
            neighbour_difference: np.ndarray = sorted_cdf_values[1:] - sorted_cdf_values[:-1]
            negative_values: np.ndarray = neighbour_difference[np.where(neighbour_difference < 0)]
            if negative_values.size > 0:
                # we may have negative_values which are very small and likely a float rounding error.
                assert np.all(negative_values > -near_zero), f"cdf values of {name} are not monotonically increasing."

            # checking extremes
            assert dist.cdf(np.inf, params) == 1.0, f"cdf of {name} is not 1.0 at infinity"
            assert dist.cdf(-np.inf, params) == 0.0, f"cdf of {name} is not 0.0 at -infinity"
        except RuntimeError:
            continue


def test_prefit_parametric_ppf(uniform_data, poisson_data):
    """Testing the ppf functions of all pre-fit parametric distributions"""
    for name in distributions_map['all parametric']:
        try:
            data: np.ndarray = get_data(name, uniform_data, poisson_data, ' parametric')
            dist, fitted, params = get_dist(name, data)

            ppf_values: np.ndarray = dist.ppf(uniform_data, params)

            # checking correct type
            assert isinstance(ppf_values, np.ndarray), f"ppf values for {name} are not contained in a numpy array"

            # checking same number of ppf values as input
            assert ppf_values.size == uniform_data.size, f"number ppf values for {name} do not match the number of inputs"

            # checking for nan values
            assert np.isnan(ppf_values).sum() == 0, f"nan values present in {name} pre-fit ppf"
        except RuntimeError:
            continue


def test_prefit_parametric_support(poisson_data, uniform_data):
    """Testing the support functions of all pre-fit parametric distributions"""
    for name in distributions_map['all parametric']:
        try:
            data: np.ndarray = get_data(name, uniform_data, poisson_data, ' parametric')
            dist, fitted, params = get_dist(name, data)

            support: tuple = dist.support(params)

            # checking support is a tuple
            assert isinstance(support, tuple), f"support values for {name} are not contained in a tuple"

            # checking for nan values
            assert np.isnan(support).sum() == 0, f"nan values present in {name} pre-fit support"

            # checking only two values in support
            assert len(support) == 2, f"incorrect number of values in prefit-support for {name}"

            # checking lb < ub
            assert support[0] < support[1], f"lb < ub is not satisfied in prefit-support for {name}"
        except RuntimeError:
            continue


def test_prefit_parametric_rvs(poisson_data, uniform_data):
    """Testing the rvs functions of all pre-fit parametric distributions"""
    for name in distributions_map['all parametric']:
        try:
            data: np.ndarray = get_data(name, uniform_data, poisson_data, ' parametric')
            dist, fitted, params = get_dist(name, data)

            num: int = 10
            for shape in ((num, ), (num, 2), (num, 5), (num, 13)):
                rvs_values: np.ndarray = dist.rvs(shape, params)

                # checking correct type
                assert isinstance(rvs_values, np.ndarray), f"pre-fit rvs values for {name} are not contained in an np.ndarray"

                # checking for nan values
                assert np.isnan(rvs_values).sum() == 0, f"nan values present in {name} pre-fit rvs"

                # checking correct shape
                assert rvs_values.shape == shape, f"incorrect shape generated for pre-fit rvs for {name}. " \
                                                  f"target shape is {shape}, generated shape is {rvs_values.shape}"
        except RuntimeError:
            continue


def test_prefit_parametric_logpdfs(poisson_data, uniform_data):
    """Testing the log-pdf functions of all pre-fit parametric distributions"""
    for name in distributions_map['all parametric']:
        try:
            data: np.ndarray = get_data(name, uniform_data, poisson_data, ' parametric')
            dist, fitted, params = get_dist(name, data)

            # checking log-pdf values are the correct data-type
            logpdf_values: np.ndarray = dist.logpdf(data, params)
            assert isinstance(logpdf_values, np.ndarray), f"log-pdf values for {name} are not contained in a numpy array"

            # checking same number of pdf values as input
            assert logpdf_values.size == data.size, f"number of log-pdf values for {name} do not match the number of inputs"

            # checking for nan values
            assert np.isnan(logpdf_values).sum() == 0, f"nan values present in {name} pre-fit log-pdf"
        except RuntimeError:
            continue


def test_prefit_parametric_likelihood(poisson_data, uniform_data):
    """Testing the likelihood functions of all pre-fit parametric distributions"""
    for name in distributions_map['all parametric']:
        try:
            data: np.ndarray = get_data(name, uniform_data, poisson_data, ' parametric')
            dist, fitted, params = get_dist(name, data)

            # checking likelihood values are the correct type
            likelihood: float = dist.likelihood(data, params)
            assert isinstance(likelihood, float), f"likelihood for {name} is not a float"

            # checking likelihood is a valid number
            valid: bool = not (np.isnan(likelihood) or (likelihood < 0))
            assert valid, f"likelihood for {name} is is nan or negative"
        except RuntimeError:
            continue


def test_prefit_parametric_loglikelihood(poisson_data, uniform_data):
    """Testing the log-likelihood functions of all pre-fit parametric distributions"""
    for name in distributions_map['all parametric']:
        try:
            data: np.ndarray = get_data(name, uniform_data, poisson_data, ' parametric')
            dist, fitted, params = get_dist(name, data)

            # checking log-likelihood values are the correct type
            loglikelihood: float = dist.loglikelihood(data, params)
            assert isinstance(loglikelihood, float), f"log-likelihood for {name} is not a float"

            # checking log-likelihood is a valid number
            valid: bool = not np.isnan(loglikelihood)
            assert valid, f"log-likelihood for {name} is is nan"
        except RuntimeError:
            continue


def test_prefit_parametric_aic(poisson_data, uniform_data):
    """Testing the aic functions of all pre-fit parametric distributions"""
    for name in distributions_map['all parametric']:
        try:
            data: np.ndarray = get_data(name, uniform_data, poisson_data, ' parametric')
            dist, fitted, params = get_dist(name, data)

            # checking aic values are the correct type
            aic: float = dist.aic(data, params)
            assert isinstance(aic, float), f"aic for {name} is not a float"

            # checking aic is a valid number
            valid: bool = not np.isnan(aic)
            assert valid, f"aic for {name} is is nan"
        except RuntimeError:
            continue


def test_prefit_parametric_bic(poisson_data, uniform_data):
    """Testing the bic functions of all pre-fit parametric distributions"""
    for name in distributions_map['all parametric']:
        try:
            data: np.ndarray = get_data(name, uniform_data, poisson_data, ' parametric')
            dist, fitted, params = get_dist(name, data)

            # checking bic values are the correct type
            bic: float = dist.bic(data, params)
            assert isinstance(bic, float), f"bic for {name} is not a float"

            # checking bic is a valid number
            valid: bool = not np.isnan(bic)
            assert valid, f"bic for {name} is is nan"
        except RuntimeError:
            continue


def test_prefit_parametric_sse(poisson_data, uniform_data):
    """Testing the sse functions of all pre-fit parametric distributions"""
    for name in distributions_map['all parametric']:
        try:
            data: np.ndarray = get_data(name, uniform_data, poisson_data, ' parametric')
            dist, fitted, params = get_dist(name, data)

            # checking sse values are the correct type
            sse: float = dist.sse(data, params)
            assert isinstance(sse, float), f"sse for {name} is not a float"

            # checking sse is a valid number
            valid: bool = not (np.isnan(sse) or (sse < 0))
            assert valid, f"sse for {name} is is nan or negative"
        except RuntimeError:
            continue


def test_prefit_parametric_gof(poisson_data, uniform_data):
    """Testing the gof functions of all pre-fit parametric distributions"""
    for name in distributions_map['all parametric']:
        try:
            data: np.ndarray = get_data(name, uniform_data, poisson_data, ' parametric')
            dist, fitted, params = get_dist(name, data)

            # checking gof object is a dataframe
            gof: pd.DataFrame = dist.gof(data, params)
            assert isinstance(gof, pd.DataFrame), f"gof for {name} is not a pandas dataframe"

            # checking gof is non-empty
            assert len(gof) > 0, f"gof for {name} is empty"
        except RuntimeError:
            continue


def test_prefit_parametric_plot(poisson_data, uniform_data):
    """Testing the plot functions of all pre-fit distributions"""
    for name in distributions_map['all parametric']:
        try:
            data: np.ndarray = get_data(name, uniform_data, poisson_data, ' parametric')
            dist, fitted, params = get_dist(name, data)

            # checking we can plot without errors
            dist.plot(params, show=False)
        except RuntimeError:
            continue


################################################################


def test_prefit_numerical_pdfs(poisson_data, uniform_data):
    """Testing the pdf functions of all pre-fit numerical distributions"""
    for name in distributions_map['all numerical']:
        data: np.ndarray = get_data(name, uniform_data, poisson_data, ' numerical')
        dist = eval(name)

        # checking pre-fit before fitting
        with pytest.raises(NotImplementedError):
            pdf_values: np.ndarray = dist.pdf(data, ())

        # checking pre-fit after fitting
        fitted = dist.fit(data)
        params: tuple = fitted.params
        with pytest.raises(NotImplementedError):
            pdf_values: np.ndarray = dist.pdf(data, params)


def test_prefit_numerical_cdfs(poisson_data, uniform_data):
    """Testing the cdf functions of all pre-fit numerical distributions"""
    for name in distributions_map['all numerical']:
        data: np.ndarray = get_data(name, uniform_data, poisson_data, ' numerical')
        dist = eval(name)

        # checking pre-fit before fitting
        with pytest.raises(NotImplementedError):
            cdf_values: np.ndarray = dist.cdf(data, ())

        # checking pre-fit after fitting
        fitted = dist.fit(data)
        params: tuple = fitted.params
        with pytest.raises(NotImplementedError):
            cdf_values: np.ndarray = dist.cdf(data, params)


def test_prefit_numerical_ppfs(poisson_data, uniform_data):
    """Testing the ppf functions of all pre-fit numerical distributions"""
    for name in distributions_map['all numerical']:
        data: np.ndarray = get_data(name, uniform_data, poisson_data, ' numerical')
        dist = eval(name)

        # checking pre-fit before fitting
        with pytest.raises(NotImplementedError):
            ppf_values: np.ndarray = dist.ppf(uniform_data, ())

        # checking pre-fit after fitting
        fitted = dist.fit(data)
        params: tuple = fitted.params
        with pytest.raises(NotImplementedError):
            ppf_values: np.ndarray = dist.ppf(data, params)


def test_prefit_numerical_support(poisson_data, uniform_data):
    """Testing the support functions of all pre-fit numerical distributions"""
    for name in distributions_map['all numerical']:
        data: np.ndarray = get_data(name, uniform_data, poisson_data, ' numerical')
        dist = eval(name)

        # checking pre-fit before fitting
        with pytest.raises(NotImplementedError):
            support: tuple = dist.support(())

        # checking pre-fit after fitting
        fitted = dist.fit(data)
        params: tuple = fitted.params
        with pytest.raises(NotImplementedError):
            support: tuple = dist.support(params)


def test_prefit_numerical_rvs(poisson_data, uniform_data):
    """Testing the rvs functions of all pre-fit numerical distributions"""
    for name in distributions_map['all numerical']:
        data: np.ndarray = get_data(name, uniform_data, poisson_data, ' numerical')
        dist = eval(name)

        # checking pre-fit before fitting
        num: int = 10
        shapes: tuple = ((num,), (num, 2), (num, 5), (num, 13))
        for shape in shapes:
            with pytest.raises(NotImplementedError):
                rvs_values: np.ndarray = dist.rvs(shape, ())

        # checking pre-fit after fitting
        fitted = dist.fit(data)
        params: tuple = fitted.params
        for shape in shapes:
            with pytest.raises(NotImplementedError):
                rvs_values: np.ndarray = dist.rvs(shape, params)


def test_prefit_numerical_logpdfs(poisson_data, uniform_data):
    """Testing the log-pdf functions of all pre-fit numerical distributions"""
    for name in distributions_map['all numerical']:
        data: np.ndarray = get_data(name, uniform_data, poisson_data, ' numerical')
        dist = eval(name)

        # checking pre-fit before fitting
        with pytest.raises(NotImplementedError):
            logpdf_values: np.ndarray = dist.logpdf(data, ())

        # checking pre-fit after fitting
        fitted = dist.fit(data)
        params: tuple = fitted.params
        with pytest.raises(NotImplementedError):
            logpdf_values: np.ndarray = dist.logpdf(data, params)


def test_prefit_numerical_likelihood(poisson_data, uniform_data):
    """Testing the likelihood functions of all pre-fit numerical distributions"""
    for name in distributions_map['all numerical']:
        data: np.ndarray = get_data(name, uniform_data, poisson_data, ' numerical')
        dist = eval(name)

        # checking pre-fit before fitting
        with pytest.raises(NotImplementedError):
            likelihood: float = dist.likelihood(data, ())

        # checking pre-fit after fitting
        fitted = dist.fit(data)
        params: tuple = fitted.params
        with pytest.raises(NotImplementedError):
            likelihood: float = dist.likelihood(data, params)


def test_prefit_numerical_loglikelihood(poisson_data, uniform_data):
    """Testing the log-likelihood functions of all pre-fit numerical distributions"""
    for name in distributions_map['all numerical']:
        data: np.ndarray = get_data(name, uniform_data, poisson_data, ' numerical')
        dist = eval(name)

        # checking pre-fit before fitting
        with pytest.raises(NotImplementedError):
            loglikelihood: float = dist.loglikelihood(data, ())

        # checking pre-fit after fitting
        fitted = dist.fit(data)
        params: tuple = fitted.params
        with pytest.raises(NotImplementedError):
            loglikelihood: float = dist.loglikelihood(data, params)


def test_prefit_numerical_aic(poisson_data, uniform_data):
    """Testing the aic functions of all pre-fit numerical distributions"""
    for name in distributions_map['all numerical']:
        data: np.ndarray = get_data(name, uniform_data, poisson_data, ' numerical')
        dist = eval(name)

        # checking pre-fit before fitting
        with pytest.raises(NotImplementedError):
            aic: float = dist.aic(data, ())

        # checking pre-fit after fitting
        fitted = dist.fit(data)
        params: tuple = fitted.params
        with pytest.raises(NotImplementedError):
            aic: float = dist.aic(data, params)


def test_prefit_numerical_bic(poisson_data, uniform_data):
    """Testing the bic functions of all pre-fit numerical distributions"""
    for name in distributions_map['all numerical']:
        data: np.ndarray = get_data(name, uniform_data, poisson_data, ' numerical')
        dist = eval(name)

        # checking pre-fit before fitting
        with pytest.raises(NotImplementedError):
            bic: float = dist.bic(data, ())

        # checking pre-fit after fitting
        fitted = dist.fit(data)
        params: tuple = fitted.params
        with pytest.raises(NotImplementedError):
            bic: float = dist.bic(data, params)


def test_prefit_numerical_sse(poisson_data, uniform_data):
    """Testing the sse functions of all pre-fit numerical distributions"""
    for name in distributions_map['all numerical']:
        data: np.ndarray = get_data(name, uniform_data, poisson_data, ' numerical')
        dist = eval(name)

        # checking pre-fit before fitting
        with pytest.raises(NotImplementedError):
            sse: float = dist.sse(data, ())

        # checking pre-fit after fitting
        fitted = dist.fit(data)
        params: tuple = fitted.params
        with pytest.raises(NotImplementedError):
            sse: float = dist.sse(data, params)


def test_prefit_numerical_gof(poisson_data, uniform_data):
    """Testing the gof functions of all pre-fit numerical distributions"""
    for name in distributions_map['all numerical']:
        data: np.ndarray = get_data(name, uniform_data, poisson_data, ' numerical')
        dist = eval(name)

        # checking pre-fit before fitting
        with pytest.raises(NotImplementedError):
            gof: pd.DataFrame = dist.gof(data, ())

        # checking pre-fit after fitting
        fitted = dist.fit(data)
        params: tuple = fitted.params
        with pytest.raises(NotImplementedError):
            gof: pd.DataFrame = dist.gof(data, params)


def test_prefit_numerical_plot(poisson_data, uniform_data):
    """Testing the plot functions of all pre-fit numerical distributions"""
    for name in distributions_map['all numerical']:
        data: np.ndarray = get_data(name, uniform_data, poisson_data, ' numerical')
        dist = eval(name)

        # checking pre-fit before fitting
        with pytest.raises(NotImplementedError):
            dist.plot(show=False)

        # checking pre-fit after fitting
        fitted = dist.fit(data)
        params: tuple = fitted.params
        with pytest.raises(NotImplementedError):
            dist.plot(show=False)

