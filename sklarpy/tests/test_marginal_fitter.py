import pandas as pd
import pytest
import numpy as np
from pathlib import Path
import os

from sklarpy.copulas import MarginalFitter
from sklarpy._utils import FitError, SaveError


def test_init(continuous_multivariate_data, discrete_multivariate_data, mixed_multivariate_data):
    """Testing whether MarginalFitter initialises without errors"""
    for data in (continuous_multivariate_data, discrete_multivariate_data, mixed_multivariate_data):
        mfitter = MarginalFitter(data)


def test_fit(continuous_multivariate_data, discrete_multivariate_data, mixed_multivariate_data):
    """Testing the fit method for MarginalFitter"""
    for data in (continuous_multivariate_data, discrete_multivariate_data, mixed_multivariate_data):
        # testing fit with default arguments
        mfitter1: MarginalFitter  = MarginalFitter(data)
        mfitter1.fit()

        # testing fit with non-default arguments affecting all marginals
        univar_opts2: dict = {'distributions': 'all numerical', 'numerical': True, 'multimodal': True, 'pvalue': 0.1}
        mfitter2: MarginalFitter  = MarginalFitter(data)
        mfitter2.fit(univar_opts2)

        # testing fit with non-default arguments affecting individual marginals
        mfitter3: MarginalFitter  = MarginalFitter(data)
        univar_opts3: dict = {i: univar_opts2 for i in range(mfitter3.num_variables)}
        mfitter3.fit(univar_opts3)

        mfitter4: MarginalFitter  = MarginalFitter(data)
        with pytest.raises(ValueError):
            univar_opts4 = univar_opts3.copy()
            univar_opts4.pop(0)
            mfitter4.fit(univar_opts4)


def test_pdfs_cdfs_ppfs_logpdf(mixed_multivariate_data):
    """Testing the cdfs method for MarginalFitter"""
    for func in ('marginal_pdfs', 'marginal_cdfs', 'marginal_ppfs', 'marginal_logpdfs'):
        # df = pd.DataFrame(mixed_multivariate_data, columns=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])
        # mfitter = MarginalFitter(df).fit()
        mfitter: MarginalFitter = MarginalFitter(mixed_multivariate_data)
        if func == 'marginal_ppfs':
            data = np.random.uniform(0, 1, mixed_multivariate_data.shape)
            alternative_data: np.ndarray = np.random.uniform(0, 1, (567, mixed_multivariate_data.shape[1]))
        else:
            data = mixed_multivariate_data
            alternative_data: np.ndarray = data[:int(len(data) / 2)] * 2

        # testing func when not fit
        if func != 'marginal_ppfs':
            with pytest.raises(FitError):
                eval(f"mfitter.{func}()")

        # testing func when fit
        mfitter.fit()
        if func != 'marginal_ppfs':
            u1: np.ndarray = eval(f"mfitter.{func}()")
        else:
            u1: np.ndarray = mfitter.marginal_ppfs(data)
        assert isinstance(u1, np.ndarray) and (u1.shape == data.shape) and np.isnan(u1).sum() == 0, \
            f"{func} should be a np.ndarray with the same shape as the data and containing no nans"

        # testing func when alternative data provided
        u2 = eval(f"mfitter.{func}(alternative_data)")
        assert isinstance(u2, np.ndarray) and (u2.shape == alternative_data.shape) and np.isnan(u2).sum() == 0, \
            f"the {func} values should be numpy arrays with size consistent with the input and containing no nans"

        # testing func raises an exception when the input dimensions are not consistent with the numer of variables
        with pytest.raises(ValueError):
            eval(f"mfitter.{func}(alternative_data[:, int(mfitter.num_variables / 2)])")


def test_marginals(continuous_multivariate_data, discrete_multivariate_data, mixed_multivariate_data):
    """Testing the marginals property of MarginalFitter"""
    for data in (continuous_multivariate_data, discrete_multivariate_data, mixed_multivariate_data):
        mfitter: MarginalFitter = MarginalFitter(data)

        # testing get_marginals when not fit
        with pytest.raises(FitError):
            mfitter.marginals

        # testing when fit
        mfitter.fit()
        marginals: dict = mfitter.marginals
        assert isinstance(marginals, dict) and (len(marginals) == data.shape[1]), \
            "object returned by get_marginals should be a dictionary with a marginal dist for each variable"


def test_summary(continuous_multivariate_data, discrete_multivariate_data, mixed_multivariate_data):
    """Testing the summary property of MarginalFitter"""
    for data in (continuous_multivariate_data, discrete_multivariate_data, mixed_multivariate_data):
        mfitter: MarginalFitter = MarginalFitter(data)

        # testing summary when not fit
        with pytest.raises(FitError):
            mfitter.summary

        # testing summary when fit
        mfitter.fit()
        summary = mfitter.summary
        assert isinstance(summary, pd.DataFrame) and (len(summary.columns) == data.shape[1]), \
            "summary should be a dataframe with a column for each variable"


def test_num_variables(continuous_multivariate_data, discrete_multivariate_data, mixed_multivariate_data):
    """Testing the num_variables property of MarginalFitter"""
    for data in (continuous_multivariate_data, discrete_multivariate_data, mixed_multivariate_data):
        mfitter: MarginalFitter = MarginalFitter(data)

        assert isinstance(mfitter.num_variables, int) and (mfitter.num_variables == data.shape[1])


def test_save(mixed_multivariate_data):
    """Testing the save method of MarginalFitter"""
    mfitter: MarginalFitter = MarginalFitter(mixed_multivariate_data).fit()
    save_location: str = f'{os.getcwd()}/{mfitter.name}.pickle'
    mfitter.save(save_location)
    my_fitted_object = Path(save_location)
    if my_fitted_object.exists():
        my_fitted_object.unlink()
    else:
        raise SaveError(f"unable to save {mfitter.name}")


def test_rvs(mixed_multivariate_data):
    """Testing the marginal_rvs method of MarginalFitter"""
    size: int = 33
    mfitter: MarginalFitter = MarginalFitter(mixed_multivariate_data)

    # testing rvs when not fit
    with pytest.raises(FitError):
        mfitter.marginal_rvs(size)

    # testing summary when fit
    mfitter.fit()
    my_rvs: np.ndarray = mfitter.marginal_rvs(size)
    assert isinstance(my_rvs, np.ndarray) and my_rvs.shape == (size, mixed_multivariate_data.shape[1]), \
        "marginal_rvs should be a numpy array with shape (size, mixed_multivariate_data.shape[1)"


def test_name(mixed_multivariate_data):
    """Testing the name property of MarginalFitter"""
    mfitter: MarginalFitter = MarginalFitter(mixed_multivariate_data)
    assert isinstance(mfitter.name, str), "name should be a string"
