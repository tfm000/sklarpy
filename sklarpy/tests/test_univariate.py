# Contains tests for the sklarpy.univariate package
import pytest
import pandas as pd

import sklarpy
from sklarpy.univariate import *
from sklarpy.univariate.distributions import all_common_names
from sklarpy.univariate._dist_wrappers import FittedContinuousUnivariate, FittedDiscreteUnivariate
from sklarpy._utils._errors import *


def test_individual_fit(poisson_data):
    """Checking we can fit all common distributions."""
    for name in all_common_names:
        dist = eval(name)
        fitted = dist.fit(poisson_data)
        assert (isinstance(fitted, FittedDiscreteUnivariate) or isinstance(fitted, FittedContinuousUnivariate)), \
            "Fitting failed"


def test_univariate_fitter1(normal_data):
    """Checking UnivariateFitter identifies the data is normal."""
    fitter = UnivariateFitter(normal_data)
    fitter.fit()
    summary = fitter.get_summary(significant=True)
    assert "normal" in summary.index, "Failed to fit to correct distribution."


def test_univariate_fitter2(poisson_data):
    """Checking UnivariateFitter identifies the data is poisson."""
    fitter = UnivariateFitter(poisson_data)
    fitter.fit()
    summary = fitter.get_summary(significant=True)
    assert "poisson" in summary.index, "Failed to fit to correct distribution."


def test_multivariate_error(normal_data2):
    """Test to see if multivariate data raises an error in UnivariateFitter."""
    data = pd.DataFrame(normal_data2)
    with pytest.raises(TypeError):
        fitter = UnivariateFitter(data)


def test_univariate_load(gamma_pickle):
    """Test for loading a univariate distribution."""
    dist = sklarpy.load(gamma_pickle)
    assert dist.params == (3.171492772884112, 3.082806644670404, 2.7650184139887717), \
        "Failed to load Univariate distribution."


def test_univariate_load2():
    """Tests if loading a non-existent file raises an error."""
    with pytest.raises(LoadError):
        dist = sklarpy.load(r'dont\put\a\file\here\test')


def test_univariate_save(gamma_pickle):
    """Test for saving a univariate distribution"""
    dist = sklarpy.load(gamma_pickle)
    dist.save(gamma_pickle)
