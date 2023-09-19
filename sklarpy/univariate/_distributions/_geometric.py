# Contains code for fitting a geometric distribution to data
import numpy as np

__all__ = ['geometric_fit']


def geometric_fit(data: np.ndarray) -> tuple:
    """Fitting function for the geometric distribution.
    Returns the MLE estimator of the parameter p.

    Math
    -----
    p_mle = n / sum(x)

    See Also:
    ---------
    https://en.wikipedia.org/wiki/Geometric_distribution

    Parameters
    -----------
    data : np.ndarray
        The data to fit to the geometric distribution in a flattened numpy
        array.

    Returns
    --------
    mle_estimator: tuple
       (p, )
    """
    return len(data) / sum(data),
