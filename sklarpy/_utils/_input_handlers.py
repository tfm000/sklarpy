from numpy import ndarray, asarray

from sklarpy._utils._variable_types import num_or_array


def univariate_num_to_array(x: num_or_array) -> ndarray:
    if isinstance(x, ndarray):
        return x.flatten()
    elif isinstance(x, float) or isinstance(x, int):
        return asarray(x).flatten()
    raise TypeError("input must be a numpy array, float or integer")


def check_params(params: tuple) -> tuple:
    if isinstance(params, tuple):
        return params
    raise TypeError("params must be a tuple")


def check_univariate_data(data: ndarray) -> ndarray:
    if isinstance(data, ndarray):
        return data.flatten()
    else:
        raise TypeError("data must be a np.ndarray.")