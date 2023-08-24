import scipy.integrate
import numpy as np
from typing import Union

__all__ = ['debye']


class DebyeIntegrator:
    @staticmethod
    def _argscheck(n: Union[int, float], x: Union[int, float]):
        for val in (n, x):
            if not (isinstance(val, float) or isinstance(val, int)):
                raise ValueError("n and x must be scalar values")

    @staticmethod
    def integrand(t: Union[int, float], n: Union[int, float]) -> float:
        return (t ** n) / (np.exp(t) - 1)

    @staticmethod
    def debye(n: Union[int, float], x: Union[int, float]) -> float:
        # checking args
        DebyeIntegrator._argscheck(n, x)

        if x == 0:
            # the limit
            return 1.0

        # evaluating integral component
        res: tuple = scipy.integrate.quad(func=DebyeIntegrator.integrand, a=0.0, b=x, args=(n, ))

        # returning result
        return n * (x ** - n) * res[0]


def debye(n: Union[int, float], x: Union[int, float]) -> float:
    """Used to evaluate the family of Debye functions, D_n(x).

    Parameters
    ----------
    n : Union[int, float]
        The n parameter, which specifies the member of the Debye function family to evaluate.
    x : Union[int, float]
        The scalar value to evaluate the Debye function at.

    See Also
    --------
    https://en.wikipedia.org/wiki/Debye_function

    Returns
    -------
    debye: float
        The value of D_n(x)
    """
    return DebyeIntegrator.debye(n=n, x=x)
