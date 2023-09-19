# Contains code for fitting a discrete uniform distribution to data
import numpy as np

__all__ = ['discrete_uniform_fit']


def discrete_uniform_fit(data: np.ndarray) -> tuple:
    """Fitting function for the discrete uniform distribution.
    Returns the minimum and maximum values of the sample.

    Math
    ----
    a = min(x)
    b = max(x)

    See Also:
    ---------
    https://en.wikipedia.org/wiki/Discrete_uniform_distribution

    Parameters
    ----------
    data: np.ndarray
        The data to fit to the discrete uniform distribution in a flattened
        numpy array.

    Returns
    -------
    estimator: tuple
       (a, b)
    """
    return int(data.min()), int(data.max() + 1)
