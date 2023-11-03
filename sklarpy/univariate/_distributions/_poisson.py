# contains code for fitting a poisson distribution to data
import numpy as np

__all__ = ['poisson_fit']


def poisson_fit(data: np.ndarray) -> tuple:
    """Fitting function for the poisson distribution.
    Returns the MLE estimator of the parameter lambda.

    Math
    ----
    lambda_mle = sum(x) / n

    See Also:
    ---------
    https://en.wikipedia.org/wiki/Poisson_distribution

    Parameters
    ----------
    data : np.ndarray
        The data to fit to the poisson distribution in a flattened numpy array.

    Returns
    -------
    mle_estimator: tuple
       (lambda, )
    """
    return data.mean(),
