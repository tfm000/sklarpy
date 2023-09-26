# Contains code for evaluating the Debye function family
import scipy.integrate
import numpy as np
from typing import Union

__all__ = ['debye']


class DebyeIntegrator:
    """Class used for evaluating the Debye function family."""
    @staticmethod
    def _argscheck(n: Union[int, float], x: Union[int, float]) -> None:
        """Checks whether arguments passed by the user are valid.

        Parameters
        ----------
        n : Union[int, float]
            The n parameter, which specifies the member of the Debye function
            family to evaluate.
        x : Union[int, float]
            The scalar value to evaluate the Debye function at.
        """
        for val in (n, x):
            if not (isinstance(val, float) or isinstance(val, int)):
                raise ValueError("n and x must be scalar values")

    @staticmethod
    def integrand(t: Union[int, float], n: Union[int, float]) -> float:
        """The integrand of the Debye function integral.

        Parameters
        ----------
        n : Union[int, float]
            The n parameter, which specifies the member of the Debye function
            family to evaluate.
        t : Union[int, float]
            The scalar value to evaluate the integrand of the Debye function
            at.

        Returns
        -------
        integrand_value: float
            The value of the integrand of the Debye function.
        """
        return (t ** n) / (np.exp(t) - 1)

    @staticmethod
    def debye(n: Union[int, float], x: Union[int, float]) -> float:
        """Used to evaluate the family of Debye functions, D_n(x).

        Parameters
        ----------
        n : Union[int, float]
            The n parameter, which specifies the member of the Debye function
            family to evaluate.
        x : Union[int, float]
            The scalar value to evaluate the Debye function at.

        Returns
        -------
        debye: float
            The value of D_n(x)
        """
        # checking args
        DebyeIntegrator._argscheck(n, x)

        if x == 0:
            # the limit
            return 1.0

        # evaluating integral component
        res: tuple = scipy.integrate.\
            quad(func=DebyeIntegrator.integrand, a=0.0, b=x, args=(n, ))

        # returning result
        return float(n * (x ** - n) * res[0])


def debye(n: Union[int, float], x: Union[int, float]) -> float:
    """Used to evaluate the family of Debye functions, D_n(x).

    Parameters
    ----------
    n : Union[int, float]
        The n parameter, which specifies the member of the Debye function
        family to evaluate.
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
