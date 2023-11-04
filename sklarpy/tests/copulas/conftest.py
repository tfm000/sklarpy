import pandas as pd
import numpy as np
import pytest
import scipy.stats

from sklarpy.univariate import lognorm, poisson, cauchy, gamma, normal

num: int = 100
d: int = 2


def mvt_continuous_data():
    return scipy.stats.multivariate_normal.rvs(size=(num, d))


def mvt_discrete_data():
    poisson_data: np.ndarray = scipy.stats.poisson.rvs(4, size=(num, d + 1))
    for i in range(1, d):
        poisson_data[:, i] = poisson_data[:, i] + poisson_data[:, i + 1]
    return poisson_data[:, :-1]


def pd_mvt_continuous_data():
    data: np.ndarray = scipy.stats.multivariate_normal.rvs(size=(num, d))
    return pd.DataFrame(data, columns=[f'var {i}' for i in range(d)])


def pd_mvt_discrete_data():
    poisson_data: np.ndarray = scipy.stats.poisson.rvs(4, size=(num, d + 1))
    for i in range(1, d):
        poisson_data[:, i] = poisson_data[:, i] + poisson_data[:, i + 1]
    return pd.DataFrame(poisson_data[:, :-1], columns=[f'var {i}' for i in
                                                       range(d)])


def mvt_mixed_data():
    mixed_data = np.full((num, d), np.nan)
    mixed_data[:, 0] = scipy.stats.multivariate_normal.rvs(size=(num,))
    mixed_data[:, 1] = scipy.stats.poisson.rvs(4, size=(num,))
    return mixed_data


def pd_mvt_mixed_data():
    mixed_data = np.full((num, d), np.nan)
    mixed_data[:, 0] = scipy.stats.multivariate_normal.rvs(size=(num,))
    mixed_data[:, 1] = scipy.stats.poisson.rvs(4, size=(num,))
    return pd.DataFrame(mixed_data, columns=[f'var {i}' for i in range(d)])


def mvt_uniform_data():
    return np.random.uniform(size=(num, d))


def pd_mvt_uniform_data():
    data: np.ndarray = np.random.uniform(size=(num, d))
    return pd.DataFrame(data, columns=[f'var {i}' for i in range(d)])


@pytest.fixture(scope="session", autouse=True)
def all_mvt_data():
    return {
        'mvt_continuous': mvt_continuous_data(),
        'mvt_discrete': mvt_discrete_data(),
        'pd_mvt_continuous': pd_mvt_continuous_data(),
        'pd_mvt_discrete': pd_mvt_discrete_data(),
        'mvt_mixed': mvt_mixed_data(),
        'pd_mvt_mixed': pd_mvt_mixed_data(),
    }


@pytest.fixture(scope="session", autouse=True)
def all_mvt_uniform_data():
    return {
        'np_uniform': mvt_uniform_data(),
        'pd_uniform': pd_mvt_uniform_data(),
    }


@pytest.fixture(scope="session", autouse=True)
def copula_params_2d():
    return {
        'clayton_copula': (1.5114589793112714, 2),

        'gumbel_copula': (1.0946807390069395, 2),

        'frank_copula': (0.655891802339665, 2),

        'gaussian_copula': (
            np.array([[0.], [0.]]),
            np.array([[1.00000000e+00, 1.01465364e-17],
                      [1.01465364e-17, 1.00000000e+00]])),

        'gh_copula': (
            -10.0, 5.494053523793585, 10.0,
            np.array([[0.], [0.]]),
            np.array([[1., 0.12029435],
                      [0.12029435, 1.]]),
            np.array([[0.99431474], [0.98828561]])),

        'mh_copula': (
            0.3301436843587767, 10.0,
            np.array([[0.], [0.]]),
            np.array([[1., 0.12029435],
                      [0.12029435, 1.]]),
            np.array([[0.99431474], [0.98828561]])),

        'hyperbolic_copula': (
            0.16085370994086476, 10.0,
            np.array([[0.], [0.]]),
            np.array([[1., 0.12029435],
                      [0.12029435, 1.]]),
            np.array([[0.99431474], [0.98828561]])),

        'nig_copula': (
            0.9333526343018996, 10.0,
            np.array([[0.], [0.]]),
            np.array([[1., 0.12029435],
                      [0.12029435, 1.]]),
            np.array([[0.99431474], [0.98828561]])),

        'skewed_t_copula': (
            10.0,
            np.array([[0.], [0.]]),
            np.array([[1., 0.12029435],
                      [0.12029435, 1.]]),
            np.array([[0.49770954], [0.49516852]])),

        'student_t_copula': (
            100.0,
            np.array([[0.], [0.]]),
            np.array([[1., 0.12029435],
                      [0.12029435, 1.]])),

        'sgh_copula': (
            -10.0, 6.994876984856893, 10.0,
            np.array([[0.], [0.]]),
            np.array([[1., 0.12029435],
                      [0.12029435, 1.]])),

        'smh_copula': (
            0.6164857476885669, 10.0,
            np.array([[0.], [0.]]),
            np.array([[1., 0.12029435],
                      [0.12029435, 1.]])),

        'shyperbolic_copula': (
            0.38355188192345885, 10.0,
            np.array([[0.], [0.]]),
            np.array([[1., 0.12029435],
                      [0.12029435, 1.]])),

        'snig_copula': (
            1.391754560716251, 10.0,
            np.array([[0.], [0.]]),
            np.array([[1., 0.12029435],
                      [0.12029435, 1.]])),
    }


@pytest.fixture(scope="session", autouse=True)
def copula_params_3d():
    return {
        'clayton_copula': (0.8320253555099054, 3),

        'gumbel_copula': (1.0010028572860392, 3),

        'gaussian_copula': (
            np.array([[0.], [0.], [0.]]),
            np.array([[1.00000000e+00, 5.09299062e-18, 3.04959979e-17],
                      [2.04732829e-16, 1.00000000e+00, -2.90580716e-16],
                      [-3.62175368e-17, -3.69147970e-17, 1.00000000e+00]])),

        'gh_copula': (
            -10.0, 5.779700036883725, 10.0,
            np.array([[0.], [0.], [0.]]),
            np.array([[1., 0.00317332, 0.02411494],
                      [0.00317332, 1., 0.03426521],
                      [0.02411494, 0.03426521, 1.]]),
            np.array([[0.99072914], [0.91655511], [0.99732581]])),

        'mh_copula': (
            0.40831049405571085, 10.0,
            np.array([[0.], [0.], [0.]]),
            np.array([[1., 0.00317332, 0.02411494],
                      [0.00317332, 1., 0.03426521],
                      [0.02411494, 0.03426521, 1.]]),
            np.array([[0.99072914], [0.91655511], [0.99732581]])),

        'hyperbolic_copula': (
            0.06892252528659842, 10.0,
            np.array([[0.], [0.], [0.]]),
            np.array([[1., 0.00317332, 0.02411494],
                      [0.00317332, 1., 0.03426521],
                      [0.02411494, 0.03426521, 1.]]),
            np.array([[0.99072914], [0.91655511], [0.99732581]])),

        'nig_copula': (
            1.04414194542558, 10.0,
            np.array([[0.], [0.], [0.]]),
            np.array([[1., 0.00317332, 0.02411494],
                     [0.00317332, 1., 0.03426521],
                     [0.02411494, 0.03426521, 1.]]),
            np.array([[0.99072914], [0.91655511], [0.99732581]])),

        'skewed_t_copula': (
            10.0,
            np.array([[0.], [0.], [0.]]),
            np.array([[1., 0.00317332, 0.02411494],
                      [0.00317332, 1., 0.03426521],
                      [0.02411494, 0.03426521, 1.]]),
            np.array([[0.55576633], [0.54642032], [0.54515928]])),

        'student_t_copula': (
            100.0,
            np.array([[0.], [0.], [0.]]),
            np.array([[1., 0.00317332, 0.02411494],
                   [0.00317332, 1., 0.03426521],
                   [0.02411494, 0.03426521, 1.]])),

        'sgh_copula': (
            -10.0, 7.46609734158415, 10.0,
            np.array([[0.], [0.], [0.]]),
            np.array([[1., 0.00317332, 0.02411494],
                      [0.00317332, 1., 0.03426521],
                      [0.02411494, 0.03426521, 1.]])),

        'smh_copula': (
            0.9105081789228913, 10.0,
            np.array([[0.], [0.], [0.]]),
            np.array([[1., -0.0329966, 0.00253866],
                      [-0.0329966, 1., -0.04758192],
                      [0.00253866, -0.04758192, 1.]])),

        'shyperbolic_copula': (
            0.3697932213423236, 10.0,
            np.array([[0.], [0.], [0.]]),
            np.array([[1., -0.0329966, 0.00253866],
                      [-0.0329966, 1., -0.04758192],
                      [0.00253866, -0.04758192, 1.]])),

        'snig_copula': (
            1.5752917143604088, 10.0,
            np.array([[0.], [0.], [0.]]),
            np.array([[1., 0.00317332, 0.02411494],
                      [0.00317332, 1., 0.03426521],
                      [0.02411494, 0.03426521, 1.]])),
    }

@pytest.fixture(scope="session", autouse=True)
def all_mdists_2d():
    return {
        'mvt_continuous': {
            0: normal.fit(params=(0, 1)),
            1: normal.fit(params=(0, 1))},

        'mvt_discrete': {
            0: poisson.fit(params=(3.95,)),
            1: poisson.fit(params=(7.83,))},

        'pd_mvt_continuous': {
            0: normal.fit(params=(0, 1)),
            1: normal.fit(params=(0, 1))},

        'pd_mvt_discrete': {
            0: poisson.fit(params=(3.84,)),
            1: poisson.fit(params=(7.91,))},

        'mvt_mixed': {
            0: normal.fit(params=(0, 1)),
            1: poisson.fit(params=(3.85,))},

        'pd_mvt_mixed': {
            0: normal.fit(params=(0, 1)),
            1: poisson.fit(params=(4.15,))}}


@pytest.fixture(scope="session", autouse=True)
def all_mdists_3d():
    return {
        'mvt_continuous': {
            0: normal.fit(params=(0, 1)),
            1: normal.fit(params=(0, 1)),
            2: normal.fit(params=(0, 1))},

        'mvt_discrete': {
            0: poisson.fit(params=(4.15,)),
            1: poisson.fit(params=(7.89,)),
            2: poisson.fit(params=(7.92,))},

        'pd_mvt_continuous': {
            0: normal.fit(params=(0, 1)),
            1: normal.fit(params=(0, 1)),
            2: normal.fit(params=(0, 1))},

        'pd_mvt_discrete': {
            0: poisson.fit(params=(3.89,)),
            1: poisson.fit(params=(7.87,)),
            2: poisson.fit(params=(7.87,))},

        'mvt_mixed': {
            0: normal.fit(params=(0, 1)),
            1: poisson.fit(params=(3.86,)),
            2: normal.fit(params=(0, 1))},

        'pd_mvt_mixed': {
            0: normal.fit(params=(0, 1)),
            1: poisson.fit(params=(3.79,)),
            2: normal.fit(params=(0, 1))}}
