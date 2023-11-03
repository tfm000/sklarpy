# Contains pytest fixtures for testing SklarPy misc code
import pytest
import scipy.stats
import numpy as np

num: int = 100  # the number of random numbers to generate
d: int = 3  # number of random variables / dimensions of multivariate data


@pytest.fixture(scope="session", autouse=True)
def continuous_data():
    return np.random.normal(size=num)


@pytest.fixture(scope="session", autouse=True)
def discrete_data():
    return np.random.poisson(10, size=(num,))


@pytest.fixture(scope="session", autouse=True)
def mvt_continuous_data():
    return scipy.stats.multivariate_normal.rvs(size=(num, d))


@pytest.fixture(scope="session", autouse=True)
def mvt_discrete_data():
    poisson_data: np.ndarray = scipy.stats.poisson.rvs(4, size=(num, d + 1))
    for i in range(1, d):
        poisson_data[:, i] = poisson_data[:, i] + poisson_data[:, i + 1]
    return poisson_data[:, :-1]
