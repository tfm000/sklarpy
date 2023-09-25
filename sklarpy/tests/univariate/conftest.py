# Contains pytest fixtures for testing SklarPy univariate code
import pytest
import numpy as np

from sklarpy.univariate import distributions_map

num: int = 100  # the number of random numbers to generate


@pytest.fixture(scope="session", autouse=True)
def dists_to_test():
    return {'normal', 'student_t', 'gamma', 'gig', 'ig', 'gh',
            *distributions_map['all common'],
            *distributions_map['all discrete parametric'],
            *distributions_map['all numerical']}


@pytest.fixture(scope="session", autouse=True)
def continuous_data():
    return np.random.normal(size=num)


@pytest.fixture(scope="session", autouse=True)
def discrete_data():
    return np.random.poisson(10, size=(num,))


@pytest.fixture(scope="session", autouse=True)
def uniform_data():
    return np.random.uniform(size=num)

