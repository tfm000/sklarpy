# Contains pytest fixtures for testing  SklarPy other code
import pytest
import numpy as np

num: int = 100  # the number of random numbers to generate


@pytest.fixture(scope="session", autouse=True)
def continuous_data():
    return np.random.normal(size=num)


@pytest.fixture(scope="session", autouse=True)
def discrete_data():
    return np.random.poisson(10, size=(num,))
