# Contains fixtures for tests
import pytest
import numpy as np


num = 1000  # the number of random numbers to generate


@pytest.fixture(scope="session", autouse=True)
def normal_data():
    return np.random.normal(size=num)


@pytest.fixture(scope="session", autouse=True)
def normal_data_2D():
    return np.random.normal(size=(num, 2))


@pytest.fixture(scope="session", autouse=True)
def uniform_data():
    return np.random.uniform(size=num)


@pytest.fixture(scope="session", autouse=True)
def poisson_data():
    return np.random.poisson(10, size=(num,))


@pytest.fixture(scope="session", autouse=True)
def continuous_multivariate_data():
    data: np.ndarray = np.full((num, 10), np.NaN)
    data[:, 0] = np.random.normal(0, 1, (num, ))
    data[:, 1] = np.random.normal(4, 2, (num, ))
    data[:, 2] = np.random.gamma(2, 1, (num, ))
    data[:, 3:5] = np.random.standard_t(3, (num, 2))
    data[:, 5:7] = np.random.uniform(0, 1, (num, 2))
    data[:, 7] = data[:, 0] ** 2
    data[:, 8] = data[:, 4] * 2
    data[:, 9] = data[:, 1:3].sum(axis=1)
    return data


@pytest.fixture(scope="session", autouse=True)
def discrete_multivariate_data():
    data: np.ndarray = np.full((num, 5), np.NaN)
    data[:, :2] = np.random.poisson(4, (num, 2))
    data[:, 2] = np.random.randint(-5, 5, (num, ))
    data[:, 3] = data[:, :2].sum(axis=1)
    data[:, 4] = data[:, 0] + data[:, 3]
    return data


@pytest.fixture(scope="session", autouse=True)
def mixed_multivariate_data():
    data: np.ndarray = np.full((num, 10), np.NaN)
    data[:, :2] = np.random.poisson(4, (num, 2))
    data[:, 2] = np.random.randint(-5, 5, (num,))
    data[:, 3] = data[:, :2].sum(axis=1)
    data[:, 4] = data[:, 0] + data[:, 3]
    data[:, 5] = np.random.normal(4, 2, (num,))
    data[:, 6] = np.random.gamma(2, 1, (num,))
    data[:, 7:9] = np.random.standard_t(3, (num, 2))
    data[:, 9] = np.random.uniform(0, 1, (num, ))
    return data


