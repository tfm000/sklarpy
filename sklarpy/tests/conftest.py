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
