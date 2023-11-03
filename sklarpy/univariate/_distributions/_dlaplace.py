# Contains code for fitting a discrete laplace distribution to data
import numpy as np
import math

__all__ = ['discrete_laplace_fit']


def discrete_laplace_fit(data: np.ndarray) -> tuple:
    """Fitting function for the discrete laplacian distribution.
    Returns the MLE estimator of the parameter alpha.

    Math
    ----
    alpha_mle = arsinh( n/sum(|x_i|) )

    See Also:
    ---------
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.dlaplace.html#scipy.stats.dlaplace

    Parameters
    -----------
    data : np.ndarray
        The data to fit to the discrete laplace distribution in a flattened
        numpy array.

    Returns
    --------
    mle_estimator: tuple
       (alpha, )
    """
    return math.asinh(len(data) / abs(data).sum()),
