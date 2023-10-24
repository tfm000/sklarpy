import pandas as pd
import numpy as np
import pytest
import scipy.stats

num: int = 100
d: int = 2


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
def mvt_mixed_data():
    mixed_data = np.full((num, d), np.nan)
    mixed_data[:, 0] = scipy.stats.multivariate_normal.rvs(size=(num,))
    mixed_data[:, 1] = scipy.stats.poisson.rvs(4, size=(num,))
    return mixed_data


@pytest.fixture(scope="session", autouse=True)
def pd_mvt_mixed_data():
    mixed_data = np.full((num, d), np.nan)
    mixed_data[:, 0] = scipy.stats.multivariate_normal.rvs(size=(num,))
    mixed_data[:, 1] = scipy.stats.poisson.rvs(4, size=(num,))
    return pd.DataFrame(mixed_data, columns=[f'var {i}' for i in range(d)])


@pytest.fixture(scope="session", autouse=True)
def mvt_uniform_data():
    return np.random.uniform(size=(num, d))


@pytest.fixture(scope="session", autouse=True)
def pd_mvt_uniform_data():
    data: np.ndarray = np.random.uniform(size=(num, d))
    return pd.DataFrame(data, columns=[f'var {i}' for i in range(d)])
