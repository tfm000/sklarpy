# Contains fixtures for tests
import pytest

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
