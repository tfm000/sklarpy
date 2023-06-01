# contains modified bessel functions
import scipy.integrate
import numpy as np

from sklarpy._utils import num_or_array, numeric

__all__ = ['kv_3']


class modified_bessel_3:
    """Modified bessel function of the 3rd kind."""
    def __init__(self, x: num_or_array, lambda_: numeric):
        """Modified bessel function of the 3rd kind.

        Parameters
        ----------
        x: Union[float, int, np.ndarray]
            The value / values to evaluate the modified bessel function at.
        lambda_: Union[float, int]
            The lambda parameter of the modified bessel function.
        """
        # arg checks
        if isinstance(x, int) or isinstance(x, float):
            self.x: np.ndarray = np.array([x])
            self.scalar_x: bool = True
        elif isinstance(x, np.ndarray):
            self.x: np.ndarray = x.flatten()
            self.scalar_x: bool = False
        else:
            raise TypeError('x must be a float, int or a np.ndarray.')

        if isinstance(lambda_, float) or isinstance(lambda_, int):
            self.lambda_: numeric = abs(lambda_)
        else:
            raise TypeError('lambda_ must be a float or int')

    def _singular_f(self, y: float, x: numeric) -> float:
        """integrand of the modified bessel function of the 3rd kind"""
        return 0.5 * y ** (self.lambda_ - 1) * np.exp(-0.5 * x * (y + (1 / y)))

    def calculate(self) -> num_or_array:
        """Calculates the values of the modified bessel function of the 3rd kind."""
        vals: np.ndarray = np.array([scipy.integrate.quad(self._singular_f, 0.0, np.inf, (x,))[0] for x in self.x], dtype=float)
        return vals[0] if self.scalar_x else vals


def kv_3(x: num_or_array, lambda_: numeric) -> num_or_array:
    """Modified bessel function of the 3rd kind.

    Parameters
    ----------
    x: Union[float, int, np.ndarray]
        The value / values to evaluate the modified bessel function at.
    lambda_: Union[float, int]
        The lambda parameter of the modified bessel function.

    Returns
    -------
    vals: Union[float, np.ndarray]
        The evaluated modified bessel function values
    """
    mb3: modified_bessel_3 = modified_bessel_3(x, lambda_)
    return mb3.calculate()
