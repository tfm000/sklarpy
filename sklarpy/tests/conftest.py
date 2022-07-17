# Contains fixtures for tests
import pytest
import os

from sklarpy.univariate import normal, poisson


@pytest.fixture(scope="session", autouse=True)
def normal_data():
    return normal.rvs(1000, 0, 1)


@pytest.fixture(scope="session", autouse=True)
def normal_data2():
    return normal.rvs((1000, 2), 0, 1)


@pytest.fixture(scope="session", autouse=True)
def poisson_data():
    return poisson.rvs(1000, 10)


@pytest.fixture(scope="session", autouse=True)
def gamma_pickle():
    dir_path: str = os.path.dirname(os.path.abspath(__file__))
    return f'{dir_path}\\gamma_test'
