# Contains code for fitting a planck exponential distribution to data
import numpy as np

__all__ = ['planck_fit']


def planck_fit(data: np.ndarray) -> tuple:
    """Fitting function for the planck exponential distribution.
    Returns the MLE estimator of the parameter lambda.

    Math
    ----
    lambda_mle = ln(1 + n/sum(x))

    See Also:
    ---------
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.planck.html#scipy.stats.planck

    Parameters
    -----------
    data : np.ndarray
        The data to fit to the planck exponential distribution in a flattened
        numpy array.

    Returns
    -------
    mle_estimator: tuple
       (lambda, )
    """
    return np.log(1 + (len(data) / data.sum())),
