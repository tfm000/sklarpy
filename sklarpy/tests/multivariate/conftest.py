# Contains pytest fixtures for testing SklarPy multivariate code
import pandas as pd
import pytest
import scipy.stats
import numpy as np

from sklarpy.multivariate import distributions_map

num: int = 100
d: int = 3


@pytest.fixture(scope="session", autouse=True)
def mvt_continuous_data():
    return scipy.stats.multivariate_normal.rvs(size=(num, d))


@pytest.fixture(scope="session", autouse=True)
def mvt_discrete_data():
    poisson_data: np.ndarray = scipy.stats.poisson.rvs(4, size=(num, d + 1))
    for i in range(1, d):
        poisson_data[:, i] = poisson_data[:, i] + poisson_data[:, i + 1]
    return poisson_data[:, :-1]


@pytest.fixture(scope="session", autouse=True)
def pd_mvt_continuous_data():
    data: np.ndarray = scipy.stats.multivariate_normal.rvs(size=(num, d))
    return pd.DataFrame(data, columns=[f'var {i}' for i in range(d)])


@pytest.fixture(scope="session", autouse=True)
def pd_mvt_discrete_data():
    poisson_data: np.ndarray = scipy.stats.poisson.rvs(4, size=(num, d + 1))
    for i in range(1, d):
        poisson_data[:, i] = poisson_data[:, i] + poisson_data[:, i + 1]
    return pd.DataFrame(poisson_data[:, :-1], columns=[f'var {i}' for i in
                                                       range(d)])


@pytest.fixture(scope="session", autouse=True)
def mv_dists_to_test():
    return distributions_map['all']
